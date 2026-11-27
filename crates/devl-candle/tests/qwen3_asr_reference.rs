//! Opt-in comparison of the Qwen3-ASR frontend and encoder against the pinned upstream reference.
//!
//! Skipped unless both environment variables are set:
//!
//! - `DEVL_QWEN3_ASR_CKPT`: local snapshot of `Qwen/Qwen3-ASR-1.7B` at the revision pinned in
//!   `docs/ARCHITECTURE.md` Section 6.1.
//! - `DEVL_QWEN3_ASR_FIXTURES`: directory holding, for each fixture name, `<name>.wav` (mono
//!   16 kHz 16-bit PCM), `<name>.mel.npy`, `<name>.audio_embeds.npy` (block-windowed attention) and
//!   `<name>.audio_embeds_global.npy` (default whole-clip attention) produced by the float32
//!   reference procedure in `docs/ARCHITECTURE.md` Section 6.13.
//!
//! Run with `cargo test -p devl-candle --release --test qwen3_asr_reference -- --nocapture`.
//! Weights and fixtures are never committed.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use devl_candle::qwen3_asr::Qwen3AsrAudioEncoder;

/// Fixture names, matching the clips described in `docs/ARCHITECTURE.md` Section 6.13.
const FIXTURES: [&str; 4] = ["full", "crop5s", "quiet_head", "long36s"];

/// Maximum absolute difference allowed on normalized mel features (`f32` frontend).
const MEL_MAX_ABS_TOL: f32 = 2e-4;
/// Maximum absolute embedding difference allowed relative to the reference peak magnitude (`f32`).
const EMBED_MAX_ABS_REL_TOL: f32 = 2e-4;
/// Minimum per-token cosine similarity allowed between embeddings (`f32`).
const EMBED_MIN_COSINE: f32 = 0.99999;

fn read_wav_i16(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    let mut at = 12;
    let (mut fmt_ok, mut data) = (false, None);
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
                "need mono 16 kHz s16 PCM"
            );
            fmt_ok = true;
        } else if id == b"data" {
            data = Some(body);
        }
        at += 8 + len + (len & 1);
    }
    assert!(fmt_ok);
    let (pairs, _) = data.unwrap().as_chunks::<2>();
    pairs
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect()
}

/// Minimal `.npy` reader for C-ordered little-endian `f32` arrays.
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).unwrap();
    assert!(
        header.contains("'<f4'") && header.contains("'fortran_order': False"),
        "{header}"
    );
    let shape_src = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_src
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let (quads, _) = bytes[start + hlen..].as_chunks::<4>();
    let data = quads.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (shape, data)
}

/// Returns (max abs difference, max abs difference over reference peak, min per-token cosine).
fn compare(ours: &[f32], reference: &[f32], tokens: usize) -> (f32, f32, f32) {
    assert_eq!(ours.len(), reference.len());
    let peak = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max = ours
        .iter()
        .zip(reference)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let mut min_cos = 1f32;
    for t in 0..tokens {
        let (a, b) = (
            &ours[t * 2048..(t + 1) * 2048],
            &reference[t * 2048..(t + 1) * 2048],
        );
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        min_cos = min_cos.min((dot / (na * nb)) as f32);
    }
    (max, max / peak, min_cos)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

#[test]
fn frontend_and_encoder_match_pinned_reference_f32() {
    let (Some(ckpt), Some(fixtures)) = (
        env_path("DEVL_QWEN3_ASR_CKPT"),
        env_path("DEVL_QWEN3_ASR_FIXTURES"),
    ) else {
        eprintln!("skipped: DEVL_QWEN3_ASR_CKPT and DEVL_QWEN3_ASR_FIXTURES are not set");
        return;
    };
    let encoder =
        Qwen3AsrAudioEncoder::from_checkpoint_dir(&ckpt, DType::F32, &Device::Cpu).unwrap();

    for name in FIXTURES {
        let pcm = read_wav_i16(&fixtures.join(format!("{name}.wav")));
        let (mel_shape, ref_mel) = read_npy_f32(&fixtures.join(format!("{name}.mel.npy")));

        let mel = encoder.frontend().compute(&pcm).unwrap();
        assert_eq!(mel_shape, vec![128, mel.frames()], "{name}: mel shape");
        let ours = mel.normalized();
        let mel_max = ours
            .iter()
            .zip(&ref_mel)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let mel_mean = ours
            .iter()
            .zip(&ref_mel)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / ours.len() as f64;

        let emb = encoder.encode_mel(&mel).unwrap();
        let tokens = encoder.config().output_tokens(mel.frames());
        assert_eq!(emb.dims(), &[tokens, 2048]);
        let ours_emb: Vec<f32> = emb.flatten_all().unwrap().to_vec1().unwrap();
        let (emb_shape, ref_emb) = read_npy_f32(&fixtures.join(format!("{name}.audio_embeds.npy")));
        assert_eq!(emb_shape, vec![tokens, 2048], "{name}: embedding shape");
        let (emb_max, rel, min_cos) = compare(&ours_emb, &ref_emb, tokens);
        let (_, global_emb) =
            read_npy_f32(&fixtures.join(format!("{name}.audio_embeds_global.npy")));
        let (g_max, g_rel, g_cos) = compare(&ours_emb, &global_emb, tokens);
        eprintln!(
            "{name}: samples={} frames={} tokens={tokens} windows={} mel_max_abs={mel_max:.3e} \
             mel_mean_abs={mel_mean:.3e} windowed: emb_max_abs={emb_max:.3e} rel={rel:.3e} \
             min_cos={min_cos:.8} | global-attention reference: emb_max_abs={g_max:.3e} \
             rel={g_rel:.3e} min_cos={g_cos:.8}",
            pcm.len(),
            mel.frames(),
            encoder.window_count(mel.frames()),
        );
        assert!(mel_max <= MEL_MAX_ABS_TOL, "{name}: mel max abs {mel_max}");
        assert!(
            rel <= EMBED_MAX_ABS_REL_TOL,
            "{name}: embedding rel error {rel}"
        );
        assert!(min_cos >= EMBED_MIN_COSINE, "{name}: min cosine {min_cos}");
    }
}
