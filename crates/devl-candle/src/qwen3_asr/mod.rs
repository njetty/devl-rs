//! Qwen3-ASR audio frontend and windowed encoder.
//!
//! Converts 16 kHz mono PCM into 2048-wide audio embeddings for the pinned
//! `Qwen/Qwen3-ASR-1.7B` checkpoint (see `docs/ARCHITECTURE.md` Section 6).
//! The decoder, prefix KV reuse, and the [`devl_core::DevlBackend`]
//! implementation are not part of this module.
//!
//! - [`frontend`]: variable-length log-mel features with the global clamp floor.
//! - [`config`]: audio tower hyperparameters and token-count arithmetic.
//! - [`encoder`]: the chunked convolution stack and the block-attention transformer.
//! - [`weights`]: selective, memory-safe loading of `thinker.audio_tower.*` tensors.

pub mod config;
pub mod encoder;
pub mod frontend;
pub mod weights;

pub use config::{Qwen3AsrAudioConfig, conv_output_len};
pub use encoder::Qwen3AsrAudioEncoder;
pub use frontend::{
    HOP_LENGTH, MAX_SAMPLES, MIN_SAMPLES, N_FFT, QWEN3_ASR_SAMPLE_RATE_HZ, Qwen3AsrMel,
    Qwen3AsrMelFrontend, frame_count, prepare_pcm, samples_to_finalize_frame,
};

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

    /// Invalid or unsupported model configuration.
    #[error("invalid Qwen3-ASR configuration: {0}")]
    InvalidConfig(String),

    /// Checkpoint files are missing, malformed, or inconsistent.
    #[error("invalid Qwen3-ASR checkpoint: {0}")]
    InvalidCheckpoint(String),
}
