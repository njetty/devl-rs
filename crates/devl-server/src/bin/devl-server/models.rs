//! Backend construction for the local `devl-server` binary.
//!
//! This file is also compiled into the end-to-end tests with `#[path]`, so the
//! tests build backends, detokenizers, and session settings exactly as the
//! binary does. It holds no protocol logic; every backend is handed to the
//! shared `Coordinator` unchanged.

#![allow(dead_code)]

use devl_core::{SchedulerConfig, SessionConfig, TokenId};
use devl_server::{CoordinatorConfig, StreamLimits};

/// Result type for model loading.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Sample rate every backend here expects.
pub const SAMPLE_RATE_HZ: u64 = 16_000;

/// Whisper accepts at most 30 s of audio per utterance.
pub const WHISPER_MAX_STREAM_SAMPLES: u64 = 30 * SAMPLE_RATE_HZ;

/// Whisper session settings. Matches the `devl` facade examples.
pub const WHISPER_SESSION: SessionConfig = SessionConfig {
    max_tokens: 128,
    enable_checkpoint_promotion: false,
};

/// Qwen3-ASR session settings. Matches the token budget of the parity test
/// in `devl-candle` (`docs/ARCHITECTURE.md` Section 6.16).
pub const QWEN3_ASR_SESSION: SessionConfig = SessionConfig {
    max_tokens: 512,
    enable_checkpoint_promotion: false,
};

/// Session settings for the mock backend.
pub const MOCK_SESSION: SessionConfig = SessionConfig {
    max_tokens: 448,
    enable_checkpoint_promotion: false,
};

/// Coordinator settings used by the binary and the smoke tests alike.
///
/// `max_stream_samples` replaces the default per-stream sample cap when the
/// backend has a shorter input limit.
pub fn coordinator_config(
    session: SessionConfig,
    max_stream_samples: Option<u64>,
    workers: usize,
    max_streams: usize,
    draft_cadence_ms: u64,
) -> CoordinatorConfig {
    let mut limits = StreamLimits::default();
    if let Some(max) = max_stream_samples {
        limits.max_stream_samples = max;
    }
    CoordinatorConfig {
        workers,
        scheduler: SchedulerConfig::default(),
        session,
        limits,
        draft_cadence_samples: draft_cadence_ms * (SAMPLE_RATE_HZ / 1_000),
        max_streams,
    }
}

/// Render token ids as space separated decimals. The mock transcript form.
pub fn mock_detokenize(tokens: &[TokenId]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Deterministic mock backend with checkpoint transcripts at 1 s and 2 s of
/// 16 kHz audio. Its tokens are synthetic, not speech.
#[cfg(feature = "mock")]
pub fn mock_backend() -> devl_core::mock::MockBackend {
    use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
    MockBackend::with_transcripts([
        (16_000, vec![11, 12, MOCK_EOS_TOKEN]),
        (32_000, vec![11, 12, 13, MOCK_EOS_TOKEN]),
    ])
}

#[cfg(feature = "candle")]
#[allow(unused_imports)]
pub use candle_models::*;

#[cfg(feature = "candle")]
mod candle_models {
    use std::path::Path;
    use std::sync::Arc;

    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_transformers::models::whisper::Config as WhisperModelConfig;
    use devl_candle::qwen3_asr::{Qwen3AsrBackend, Qwen3AsrBackendConfig};
    use devl_candle::{CandleWhisperBackend, CandleWhisperConfig};
    use devl_core::TokenId;
    use serde_json::Value;

    use super::Result;

    /// `<asr_text>` marker the pinned Qwen3-ASR checkpoint emits before the
    /// transcript (`docs/ARCHITECTURE.md` Section 6.5).
    const QWEN3_ASR_TEXT_MARKER: TokenId = 151_704;
    /// First special token id. Ids at or above it are not in `vocab.json`.
    const QWEN3_ASR_FIRST_SPECIAL: TokenId = 151_643;

    /// Id to text conversion shared with the coordinator.
    pub type Detokenize = Arc<dyn Fn(&[TokenId]) -> String + Send + Sync>;

    /// Load a local Hugging Face Whisper checkpoint (`config.json` and one
    /// `model.safetensors`) on the CPU in `f32`, with the Whisper
    /// `tokenizer.json` for transcript text.
    pub fn load_whisper(
        dir: &Path,
        tokenizer: &Path,
    ) -> Result<(CandleWhisperBackend, Detokenize)> {
        let config: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let model_config: WhisperModelConfig = serde_json::from_value(config)?;
        let (prompt_tokens, eos_token) = match model_config.vocab_size {
            51_864 => (vec![50_257, 50_362], 50_256),
            51_865 => (vec![50_258, 50_259, 50_359, 50_363], 50_257),
            51_866 => (vec![50_258, 50_259, 50_360, 50_364], 50_257),
            size => return Err(format!("unsupported Whisper vocabulary size {size}").into()),
        };
        let device = Device::Cpu;
        let weights = std::fs::read(dir.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let backend = CandleWhisperBackend::from_var_builder(
            vb,
            CandleWhisperConfig {
                model_config: Some(model_config),
                device,
                prompt_tokens,
                eos_token,
                ..CandleWhisperConfig::default()
            },
        )?;
        let tk = tokenizers::Tokenizer::from_file(tokenizer)
            .map_err(|e| format!("{}: {e}", tokenizer.display()))?;
        let detok: Detokenize = Arc::new(move |tokens| {
            tk.decode(tokens, true)
                .map(|t| t.trim().to_owned())
                .unwrap_or_default()
        });
        Ok((backend, detok))
    }

    /// Load the pinned `Qwen/Qwen3-ASR-1.7B` snapshot on the CPU in `f32`.
    ///
    /// Transcript text comes from the snapshot's `vocab.json` and
    /// `merges.txt` with a byte level decoder. Only ids after the last
    /// `<asr_text>` marker are decoded, and special ids are dropped, so the
    /// `language ...` header the checkpoint emits is not part of the text.
    pub fn load_qwen3_asr(dir: &Path) -> Result<(Arc<Qwen3AsrBackend>, Detokenize)> {
        let backend = Qwen3AsrBackend::from_checkpoint_dir(
            dir,
            DType::F32,
            &Device::Cpu,
            Qwen3AsrBackendConfig::default(),
        )?;
        let bpe = tokenizers::models::bpe::BPE::from_file(
            &dir.join("vocab.json").to_string_lossy(),
            &dir.join("merges.txt").to_string_lossy(),
        )
        .build()
        .map_err(|e| format!("qwen3-asr tokenizer: {e}"))?;
        let mut tk = tokenizers::Tokenizer::new(bpe);
        tk.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        let detok: Detokenize = Arc::new(move |tokens| {
            let start = tokens
                .iter()
                .rposition(|&t| t == QWEN3_ASR_TEXT_MARKER)
                .map_or(0, |i| i + 1);
            let text_ids: Vec<TokenId> = tokens[start..]
                .iter()
                .copied()
                .filter(|&t| t < QWEN3_ASR_FIRST_SPECIAL)
                .collect();
            tk.decode(&text_ids, false)
                .map(|t| t.trim().to_owned())
                .unwrap_or_default()
        });
        Ok((Arc::new(backend), detok))
    }
}
