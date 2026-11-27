//! Opt-in real-checkpoint check of the Qwen3-ASR decoder.
//!
//! Skipped unless both environment variables are set:
//!
//! - `DEVL_QWEN3_ASR_CKPT`: local snapshot of `Qwen/Qwen3-ASR-1.7B` at the revision pinned in
//!   `docs/ARCHITECTURE.md` Section 6.1.
//! - `DEVL_QWEN3_ASR_WAV`: mono 16 kHz 16-bit PCM clip no longer than the encoder limit.
//!
//! Optional: `DEVL_QWEN3_ASR_EXPECTED_IDS`, a comma separated list of upstream greedy ids
//! (including the final stop id) from the reference procedure in Section 6.7. When set, the
//! generated ids must match exactly.
//!
//! Attention mode: the encoder in this crate uses block-windowed attention. For a clip of at
//! most one 8 s block this equals the upstream default whole-clip attention, so decoder output
//! is comparable with the Section 6.7 reference. For longer clips it is not, and the expected ids
//! must then come from the windowed reference of Section 6.13.
//!
//! The check is decoder-focused: prompt prefill, greedy decode to either stop id, and decode
//! after rollback against a fresh prefill of the accepted prefix.
//!
//! Run with `cargo test -p devl-candle --release --test qwen3_asr_decoder_reference -- --nocapture`.
//! Weights and clips are never committed.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use devl_candle::qwen3_asr::{
    Qwen3AsrAudioEncoder, Qwen3AsrDecoder, argmax_rows, config::Qwen3AsrAudioConfig, is_stop_token,
};

const AUDIO_PAD: u32 = 151_676;
const PREFIX: [u32; 9] = [151_644, 8948, 198, 151_645, 198, 151_644, 872, 198, 151_669];
const SUFFIX: [u32; 6] = [151_670, 151_645, 198, 151_644, 77_091, 198];
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
fn decoder_generates_and_rolls_back_on_pinned_checkpoint_f32() {
    let (Some(ckpt), Some(wav)) = (
        env_path("DEVL_QWEN3_ASR_CKPT"),
        env_path("DEVL_QWEN3_ASR_WAV"),
    ) else {
        eprintln!("skipped: DEVL_QWEN3_ASR_CKPT and DEVL_QWEN3_ASR_WAV are not set");
        return;
    };
    let device = Device::Cpu;
    let pcm = read_wav_i16(&wav);
    let encoder = Qwen3AsrAudioEncoder::from_checkpoint_dir(&ckpt, DType::F32, &device).unwrap();
    let audio = encoder.encode_pcm(&pcm).unwrap();
    let n = audio.dim(0).unwrap();
    let frames = pcm.len().max(8000) / 160;
    let windows = frames.div_ceil(Qwen3AsrAudioConfig::qwen3_asr_1_7b().window_frames());
    println!(
        "samples={} frames={frames} audio_tokens={n} windows={windows}",
        pcm.len()
    );
    drop(encoder);

    let decoder = Qwen3AsrDecoder::from_checkpoint_dir(&ckpt, DType::F32, &device).unwrap();
    let mut prompt = PREFIX.to_vec();
    prompt.extend(std::iter::repeat_n(AUDIO_PAD, n));
    prompt.extend(SUFFIX);
    let p = prompt.len();
    assert_eq!(p, 15 + n);
    let embeds = decoder.embed_prompt(&prompt, AUDIO_PAD, &audio).unwrap();

    let mut cache = decoder.new_cache();
    let logits = decoder.prefill(&embeds, &mut cache, 1).unwrap();
    assert_eq!(cache.len(), p);
    let mut generated = vec![argmax_rows(&logits).unwrap()[0]];
    while !is_stop_token(*generated.last().unwrap()) && generated.len() < MAX_NEW_TOKENS {
        let next = decoder
            .step(*generated.last().unwrap(), &mut cache)
            .unwrap();
        generated.push(next);
    }
    println!("generated_ids={generated:?}");
    assert!(
        is_stop_token(*generated.last().unwrap()),
        "no stop id within {MAX_NEW_TOKENS} tokens"
    );
    // The cache holds the prompt plus every fed token (the last generated id is not fed).
    assert_eq!(cache.len(), p + generated.len() - 1);

    if let Ok(expected) = std::env::var("DEVL_QWEN3_ASR_EXPECTED_IDS") {
        let expected: Vec<u32> = expected
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(
            generated, expected,
            "greedy ids differ from the upstream reference"
        );
    }

    // Decode after rollback versus a fresh prefill of prompt + accepted tokens. Feeding
    // fed[keep] after keeping `keep` tokens must give upstream's next id generated[keep + 1].
    let fed = &generated[..generated.len() - 1];
    for keep in [0, 1, fed.len() / 2, fed.len() - 1] {
        let mut rolled = cache.fork();
        rolled.truncate(p + keep).unwrap();
        let mut ids = prompt.clone();
        ids.extend(&fed[..keep]);
        let mut fresh = decoder.new_cache();
        let accepted = decoder.embed_prompt(&ids, AUDIO_PAD, &audio).unwrap();
        decoder.extend(&accepted, &mut fresh).unwrap();
        assert_eq!(rolled.len(), fresh.len());
        let a = decoder.step_logits(fed[keep], &mut rolled).unwrap();
        let b = decoder.step_logits(fed[keep], &mut fresh).unwrap();
        let diff = (&a - &b)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("keep={keep}: rolled-back vs fresh logits max abs diff {diff:e}");
        assert!(diff < 1e-3);
        let next = argmax_rows(&a.unsqueeze(0).unwrap()).unwrap()[0];
        assert_eq!(next, argmax_rows(&b.unsqueeze(0).unwrap()).unwrap()[0]);
        assert_eq!(next, generated[keep + 1], "keep={keep}: next token");
    }
}
