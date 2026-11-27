//! Opt-in real-checkpoint smoke test of `Qwen3AsrBackend`.
//!
//! Skipped unless both environment variables are set:
//!
//! - `DEVL_QWEN3_ASR_CKPT`: local snapshot of `Qwen/Qwen3-ASR-1.7B` at the pinned revision.
//! - `DEVL_QWEN3_ASR_WAV`: mono 16 kHz 16-bit PCM clip.
//!
//! The test runs the `DevlSession` protocol on the clip in chunks (several drafts that reuse
//! windows and prefix KV, then a commit) and checks that the committed tokens equal the backend's
//! own offline greedy decode of the whole clip, with and without checkpoint promotion. It also
//! checks that cached windows are reused for a clip longer than one window. It does not compare
//! against upstream token ids: upstream parity is tested by
//! `qwen3_asr_parity_reference.rs`. The audio attention mode is block-windowed (`docs/ARCHITECTURE.md` Section 6.13).
//!
//! Run with `cargo test -p devl-candle --release --test qwen3_asr_backend_reference -- --nocapture`.
//! Weights and clips are never committed.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use devl_candle::qwen3_asr::{Qwen3AsrBackend, Qwen3AsrBackendConfig};
use devl_core::{DevlBackend, DevlSession, SessionConfig};

const MAX_NEW_TOKENS: usize = 128;

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

#[test]
fn session_on_pinned_checkpoint_matches_offline_greedy_f32() {
    let (Some(ckpt), Some(wav)) = (
        env_path("DEVL_QWEN3_ASR_CKPT"),
        env_path("DEVL_QWEN3_ASR_WAV"),
    ) else {
        eprintln!("skipped: DEVL_QWEN3_ASR_CKPT and DEVL_QWEN3_ASR_WAV are not set");
        return;
    };
    let pcm = read_wav_i16(&wav);
    let t = Instant::now();
    let backend = Qwen3AsrBackend::from_checkpoint_dir(
        &ckpt,
        DType::F32,
        &Device::Cpu,
        Qwen3AsrBackendConfig::default(),
    )
    .unwrap();
    eprintln!("load: {:.1?}, {} samples", t.elapsed(), pcm.len());

    let t = Instant::now();
    let enc = backend.encode_audio(&pcm, None).unwrap();
    let mut kv = backend.new_kv_cache().unwrap();
    let mut offline = vec![
        backend
            .verify_draft(&enc, &mut kv, &[])
            .unwrap()
            .argmax_tokens[0],
    ];
    while !backend.is_eos_token(*offline.last().unwrap()) && offline.len() < MAX_NEW_TOKENS {
        let seed = *offline.last().unwrap();
        offline.push(backend.decode_step(&enc, &mut kv, seed).unwrap().token);
    }
    eprintln!(
        "offline: {:.1?}, {} audio tokens, {} final windows, ids {:?}",
        t.elapsed(),
        enc.audio_tokens(),
        enc.final_windows(),
        offline
    );
    assert!(
        backend.is_eos_token(*offline.last().unwrap()),
        "no stop id within the budget"
    );

    // A growing buffer in 2 s steps: later passes must reuse the windows finalized before.
    let step = 2 * 16_000;
    let mut prior = None;
    let mut reused_any = false;
    for end in (step..=pcm.len()).step_by(step).chain([pcm.len()]) {
        let e = backend.encode_audio(&pcm[..end], prior.as_ref()).unwrap();
        reused_any |= e.stats().reused_windows > 0;
        prior = Some(e);
    }
    if pcm.len() > 140_000 {
        assert!(
            reused_any,
            "a clip with several final windows must reuse some"
        );
    }

    for promotion in [false, true] {
        let config = SessionConfig {
            max_tokens: MAX_NEW_TOKENS,
            enable_checkpoint_promotion: promotion,
        };
        let t = Instant::now();
        let mut s = DevlSession::new(&backend, config).unwrap();
        for chunk in pcm.chunks(step) {
            s.push_audio(chunk).unwrap();
            s.run_draft().unwrap();
        }
        let r = s.commit().unwrap();
        eprintln!(
            "session promotion={promotion}: {:.1?}, accepted {}/{} decode_steps {} promoted {}",
            t.elapsed(),
            r.accepted_prefix_len,
            r.draft_len,
            r.decode_steps,
            r.was_promoted
        );
        assert_eq!(r.tokens, offline, "promotion {promotion}");
    }
}
