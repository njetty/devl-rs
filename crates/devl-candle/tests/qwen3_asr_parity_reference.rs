//! Opt-in exact-token parity of `Qwen3AsrBackend` against the pinned upstream model.
//!
//! Skipped unless these environment variables are set:
//!
//! - `DEVL_QWEN3_ASR_CKPT`: local snapshot of `Qwen/Qwen3-ASR-1.7B` at the pinned revision.
//! - `DEVL_QWEN3_ASR_PARITY_DIR`: directory with `<name>.wav` (mono 16 kHz PCM16) and the upstream
//!   reference `<name>.windowed.json` written by `tests/tools/qwen3_asr_parity_reference.py`.
//!   An optional `<name>.global.json` (default `sdpa` audio attention) is reported for contrast
//!   and is never a pass criterion.
//!
//! Optional: `DEVL_QWEN3_ASR_PARITY_NAMES` (comma separated fixture names, default every
//! `*.windowed.json` in the directory) and `DEVL_QWEN3_ASR_PARITY_CHUNK_SECONDS` (draft cadence,
//! default 2).
//!
//! For every fixture the test compares raw generated token ids, with the same stop set and token
//! budget, between:
//!
//! 1. upstream whole-prompt offline greedy with block-windowed audio attention (the reference),
//! 2. Rust whole-buffer offline greedy (`encode_audio(pcm, None)` then greedy decode), and
//! 3. a `DevlSession` that drafts on a growing buffer and commits, with and without checkpoint
//!    promotion. Its encodes reuse cached windows, and rebuild them when the clamp floor moves.
//!
//! Cache use is read from the backend call log and printed, so a passing run shows which path
//! was exercised. On a mismatch the first differing index, both tokens, the Rust logit gap
//! between them, and the upstream top-2 margin are printed before the test fails. Nothing here
//! asserts a latency, and weights and clips are never committed (`docs/ARCHITECTURE.md` 6.16).

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use common::{Log, Probe};
use devl_candle::qwen3_asr::{Qwen3AsrBackend, Qwen3AsrBackendConfig};
use devl_core::{DevlBackend, DevlSession, SessionConfig, TokenId};

const MAX_NEW_TOKENS: usize = 512;

/// Fixtures built to move the clamp floor mid-stream (see `qwen3_asr_make_fixtures.py`).
const INVALIDATING: [&str; 2] = ["quiet_head", "rising36s"];
/// Fixtures long enough to finalize at least one window before the end of the stream.
const REUSING: [&str; 4] = ["full", "quiet_head", "long36s", "rising36s"];

fn read_wav_i16(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    let mut at = 12;
    let mut data = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            let tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
            let ch = u16::from_le_bytes(body[2..4].try_into().unwrap());
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            assert_eq!(
                (tag, ch, rate, bits),
                (1, 1, 16_000, 16),
                "need mono 16 kHz s16"
            );
        } else if id == b"data" {
            data = Some(body);
        }
        at += 8 + len + (len & 1);
    }
    let (pairs, _) = data.expect("data chunk").as_chunks::<2>();
    pairs
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect()
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

/// Upstream reference record for one fixture and attention mode.
struct Reference {
    ids: Vec<TokenId>,
    margins: Vec<f64>,
    num_samples: usize,
    audio_pad_count: usize,
    prompt_len: usize,
}

fn read_reference(path: &Path, attention: &str) -> Reference {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u64> {
        v[k].as_array()
            .unwrap_or_else(|| panic!("{k} missing in {}", path.display()))
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    };
    assert_eq!(v["attention"], attention, "{}", path.display());
    assert_eq!(v["dtype"], "float32", "{}", path.display());
    assert_eq!(v["max_new_tokens"], MAX_NEW_TOKENS, "{}", path.display());
    Reference {
        ids: ids("generated_ids").into_iter().map(|x| x as u32).collect(),
        margins: v["argmax_margins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
        num_samples: v["num_samples"].as_u64().unwrap() as usize,
        audio_pad_count: v["audio_pad_count"].as_u64().unwrap() as usize,
        prompt_len: v["prompt_len"].as_u64().unwrap() as usize,
    }
}

fn first_difference(a: &[TokenId], b: &[TokenId]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n)
        .find(|&i| a[i] != b[i])
        .or((a.len() != b.len()).then_some(n))
}

/// Top-2 margin of a logit row.
fn margin(row: &[f32]) -> f32 {
    let mut top = [f32::NEG_INFINITY; 2];
    for &x in row {
        if x > top[0] {
            top = [x, top[0]];
        } else if x > top[1] {
            top[1] = x;
        }
    }
    top[0] - top[1]
}

/// Rust offline greedy over the whole buffer with the logits of every step.
fn offline_with_logits(b: &Qwen3AsrBackend, pcm: &[f32]) -> (Vec<TokenId>, Vec<Vec<f32>>) {
    let enc = b.encode_audio(pcm, None).unwrap();
    let mut kv = b.new_kv_cache().unwrap();
    let first = b.verify_draft(&enc, &mut kv, &[]).unwrap();
    let mut tokens = vec![first.argmax_tokens[0]];
    let mut logits = vec![first.logits.unwrap().remove(0)];
    while !b.is_eos_token(*tokens.last().unwrap()) && tokens.len() < MAX_NEW_TOKENS {
        let step = b
            .decode_step(&enc, &mut kv, *tokens.last().unwrap())
            .unwrap();
        tokens.push(step.token);
        logits.push(step.logits.unwrap());
    }
    (tokens, logits)
}

/// Prints how a sequence differs from the reference and returns whether it is equal.
fn report_diff(
    label: &str,
    got: &[TokenId],
    reference: &Reference,
    rust_logits: Option<&[Vec<f32>]>,
) -> bool {
    let Some(i) = first_difference(got, &reference.ids) else {
        eprintln!("  {label}: equal to upstream ({} ids)", got.len());
        return true;
    };
    eprintln!(
        "  {label}: FIRST DIFFERENCE at index {i} (rust {} ids, upstream {} ids)",
        got.len(),
        reference.ids.len()
    );
    eprintln!(
        "    rust token {:?}, upstream token {:?}",
        got.get(i),
        reference.ids.get(i)
    );
    if let Some(m) = reference.margins.get(i) {
        eprintln!("    upstream top-2 logit margin at {i}: {m:.6}");
    }
    if let (Some(rows), Some(&r), Some(&u)) = (rust_logits, got.get(i), reference.ids.get(i)) {
        let row = &rows[i];
        eprintln!(
            "    rust logits: top-2 margin {:.6}, logit[rust] - logit[upstream] = {:.6}",
            margin(row),
            row[r as usize] - row[u as usize]
        );
    }
    eprintln!(
        "    shared prefix: {:?}",
        &reference.ids[..i.min(reference.ids.len())]
    );
    false
}

fn describe_log(log: &Log) {
    let last = log.encodes.last().unwrap();
    eprintln!(
        "    encodes {}, reused windows {}, rebuilt windows {}, floor changes {}, encodes with partial tail {}, \
         prefix tokens prefilled {}",
        log.encodes.len(),
        log.reused_windows(),
        log.invalidated_windows(),
        log.floor_changes(),
        log.tail_encodes(),
        log.encodes
            .iter()
            .map(|e| e.stats.prefilled_prefix_tokens)
            .sum::<usize>(),
    );
    eprintln!(
        "    final encode: {} samples, {} final windows, {} audio tokens, floor {:.4}; verifies {} (rejected drafts {}), \
         rollbacks {}, decode steps {}",
        last.samples,
        last.final_windows,
        last.audio_tokens,
        last.floor,
        log.verifies.len(),
        log.mismatches(),
        log.rollbacks.len(),
        log.decode_steps,
    );
}

#[test]
fn pinned_checkpoint_matches_upstream_windowed_greedy_on_offline_and_streaming_paths() {
    let (Some(ckpt), Some(dir)) = (
        env_path("DEVL_QWEN3_ASR_CKPT"),
        env_path("DEVL_QWEN3_ASR_PARITY_DIR"),
    ) else {
        eprintln!("skipped: DEVL_QWEN3_ASR_CKPT and DEVL_QWEN3_ASR_PARITY_DIR are not set");
        return;
    };
    let names: Vec<String> = match std::env::var("DEVL_QWEN3_ASR_PARITY_NAMES") {
        Ok(list) => list.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => {
            let mut v: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| {
                    let n = e.ok()?.file_name().into_string().ok()?;
                    n.strip_suffix(".windowed.json").map(str::to_string)
                })
                .collect();
            v.sort();
            v
        }
    };
    assert!(!names.is_empty(), "no fixtures in {}", dir.display());
    let chunk_seconds: usize = std::env::var("DEVL_QWEN3_ASR_PARITY_CHUNK_SECONDS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(2);
    let step = chunk_seconds
        .checked_mul(16_000)
        .filter(|&samples| samples > 0)
        .expect("DEVL_QWEN3_ASR_PARITY_CHUNK_SECONDS must be a positive supported duration");

    let t = Instant::now();
    let backend = Qwen3AsrBackend::from_checkpoint_dir(
        &ckpt,
        DType::F32,
        &Device::Cpu,
        Qwen3AsrBackendConfig {
            return_logits: true,
            ..Qwen3AsrBackendConfig::default()
        },
    )
    .unwrap();
    eprintln!(
        "loaded in {:.1?}; dtype f32, CPU, draft cadence {chunk_seconds} s, max tokens {MAX_NEW_TOKENS}",
        t.elapsed()
    );

    let mut failures = Vec::new();
    for name in &names {
        eprintln!("== fixture {name}");
        let pcm = read_wav_i16(&dir.join(format!("{name}.wav")));
        let reference = read_reference(&dir.join(format!("{name}.windowed.json")), "windowed");
        assert_eq!(pcm.len(), reference.num_samples, "{name}: fixture mismatch");

        let t = Instant::now();
        let (offline, logits) = offline_with_logits(&backend, &pcm);
        let enc = backend.encode_audio(&pcm, None).unwrap();
        eprintln!(
            "  {} samples, {} audio tokens, {} final windows; offline greedy took {:.1?}",
            pcm.len(),
            enc.audio_tokens(),
            enc.final_windows(),
            t.elapsed()
        );
        assert_eq!(enc.audio_tokens(), reference.audio_pad_count, "{name}");
        assert_eq!(enc.audio_tokens() + 15, reference.prompt_len, "{name}");
        eprintln!("  upstream ids: {:?}", reference.ids);
        assert!(
            reference
                .ids
                .last()
                .is_some_and(|&t| backend.is_eos_token(t)),
            "{name}: upstream did not stop within the budget"
        );

        if !report_diff("rust offline greedy", &offline, &reference, Some(&logits)) {
            failures.push(format!("{name}: offline"));
        }

        for promotion in [false, true] {
            let probe = Probe::new(&backend);
            let t = Instant::now();
            let mut s = DevlSession::new(
                &probe,
                SessionConfig {
                    max_tokens: MAX_NEW_TOKENS,
                    enable_checkpoint_promotion: promotion,
                },
            )
            .unwrap();
            for chunk in pcm.chunks(step) {
                s.push_audio(chunk).unwrap();
                s.run_draft().unwrap();
            }
            let r = s.commit().unwrap().clone();
            drop(s);
            let log = probe.log();
            eprintln!(
                "  streaming commit, promotion {promotion}: {:.1?}, accepted {}/{} draft ids, \
                 resumed decode steps {}, promoted {}",
                t.elapsed(),
                r.accepted_prefix_len,
                r.draft_len,
                r.decode_steps,
                r.was_promoted
            );
            describe_log(&log);
            if !report_diff("streaming commit", &r.tokens, &reference, None) {
                failures.push(format!("{name}: streaming promotion {promotion}"));
            }
            // Cache paths that this fixture is built to exercise.
            if REUSING.contains(&name.as_str()) {
                assert!(log.reused_windows() > 0, "{name}: no window was reused");
            }
            if INVALIDATING.contains(&name.as_str()) {
                assert!(
                    log.invalidated_windows() > 0 && log.floor_changes() > 0,
                    "{name}: expected the clamp floor to move and windows to be rebuilt"
                );
            }
        }

        // Contrast only: default sdpa attention attends over the whole clip.
        let global = dir.join(format!("{name}.global.json"));
        if global.exists() {
            let g = read_reference(&global, "global");
            let same = g.ids == reference.ids;
            eprintln!(
                "  contrast: upstream whole-clip (sdpa) attention ids {} the windowed ids ({} ids)",
                if same { "equal" } else { "DIFFER from" },
                g.ids.len()
            );
            if !same {
                report_diff(
                    "rust offline greedy vs whole-clip attention",
                    &offline,
                    &g,
                    Some(&logits),
                );
            }
        }
    }
    assert!(failures.is_empty(), "token mismatches: {failures:?}");
}
