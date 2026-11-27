//! Qwen3-ASR audio frontend, windowed encoder, and causal text decoder.
//!
//! Converts 16 kHz mono PCM into 2048-wide audio embeddings and decodes them with a
//! position-aware KV cache for the pinned `Qwen/Qwen3-ASR-1.7B` checkpoint (see
//! `docs/ARCHITECTURE.md` Section 6), and exposes them through [`Qwen3AsrBackend`], the
//! [`devl_core::DevlBackend`] implementation with reusable audio windows and prefix KV.
//!
//! - [`frontend`]: variable-length log-mel features with the global clamp floor.
//! - [`config`]: audio tower hyperparameters and token-count arithmetic.
//! - [`encoder`]: the chunked convolution stack and the block-attention transformer.
//! - [`decoder`]: causal prefill and incremental greedy step over absolute RoPE positions.
//! - [`kv_cache`]: forkable and truncatable decoder KV cache.
//! - [`backend`]: the `DevlBackend` implementation and its reuse and invalidation rules.
//! - [`weights`]: selective, memory-safe loading of `thinker.*` tensors.

pub mod backend;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod frontend;
pub mod kv_cache;
pub mod weights;

pub use backend::{
    AUDIO_PAD_ID, DEFAULT_MAX_AUDIO_SAMPLES, EOS_TOKEN_ID, PROMPT_PREFIX_IDS, PROMPT_SUFFIX_IDS,
    Qwen3AsrAudioEncoding, Qwen3AsrBackend, Qwen3AsrBackendConfig, Qwen3AsrDecodeState,
    Qwen3AsrEncodeStats,
};
pub use config::{
    Qwen3AsrAudioConfig, Qwen3AsrTextConfig, STOP_TOKEN_IDS, conv_output_len, is_stop_token,
};
pub use decoder::{Qwen3AsrDecoder, argmax_rows};
pub use encoder::Qwen3AsrAudioEncoder;
pub use frontend::{
    HOP_LENGTH, MAX_SAMPLES, MIN_SAMPLES, N_FFT, QWEN3_ASR_SAMPLE_RATE_HZ, Qwen3AsrMel,
    Qwen3AsrMelFrontend, frame_count, prepare_pcm, samples_to_finalize_frame,
};
pub use kv_cache::Qwen3AsrKvCache;

use thiserror::Error;

/// Errors from the Qwen3-ASR audio frontend, encoder, and loader.
#[derive(Debug, Error)]
pub enum Qwen3AsrError {
    /// Underlying Candle tensor or device error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Filesystem error while reading checkpoint files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// No PCM samples were supplied.
    #[error("input audio is empty")]
    EmptyAudio,

    /// A sample is not finite or lies outside `[-1.0, 1.0]`.
    #[error("invalid PCM sample {value} at index {index} (expected finite value in [-1.0, 1.0])")]
    InvalidSample {
        /// Index of the first offending sample.
        index: usize,
        /// Offending value.
        value: f32,
    },

    /// Input exceeds the supported duration.
    #[error("input audio length {samples} exceeds the maximum of {max_samples} samples")]
    AudioTooLong {
        /// Number of samples supplied.
        samples: usize,
        /// Maximum supported samples.
        max_samples: usize,
    },

    /// Feature tensor does not match the configured mel bin count or is empty.
    #[error("invalid feature shape {shape:?} (expected [{num_mel_bins}, frames > 0])")]
    InvalidFeatures {
        /// Shape received.
        shape: Vec<usize>,
        /// Expected mel bin count.
        num_mel_bins: usize,
    },

    /// Attention window index is past the last window of the clip.
    #[error("window index {index} out of range ({windows} windows)")]
    WindowOutOfRange {
        /// Requested window index.
        index: usize,
        /// Number of windows in the clip.
        windows: usize,
    },

    /// KV cache truncation target is longer than the cached sequence.
    #[error("cannot truncate KV cache of length {len} to {requested}")]
    InvalidCacheLength {
        /// Requested length.
        requested: usize,
        /// Current cached length.
        len: usize,
    },

    /// The sequence would exceed the model's position limit.
    #[error("sequence length {requested} exceeds the maximum of {max} positions")]
    ContextTooLong {
        /// Length the sequence would reach.
        requested: usize,
        /// Maximum supported length.
        max: usize,
    },

    /// KV rollback target is outside the retained generated range.
    #[error(
        "invalid KV rollback of {requested} generated tokens (prompt_len={prompt_len}, cache length {len})"
    )]
    InvalidRollback {
        /// Requested number of generated tokens to retain.
        requested: usize,
        /// Prompt length `P` of the decoder state, `0` if it was never prefilled.
        prompt_len: usize,
        /// Cached length at the time of the call.
        len: usize,
    },

    /// An encoding and a decoder state, or an encoding and a backend, do not belong together.
    #[error("mismatched backend state: {0}")]
    StateMismatch(String),

    /// Decoder input does not match the model or the prompt layout.
    #[error("invalid decoder input: {0}")]
    InvalidInput(String),

    /// Invalid or unsupported model configuration.
    #[error("invalid Qwen3-ASR configuration: {0}")]
    InvalidConfig(String),

    /// Checkpoint files are missing, malformed, or inconsistent.
    #[error("invalid Qwen3-ASR checkpoint: {0}")]
    InvalidCheckpoint(String),
}
