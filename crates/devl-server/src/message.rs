//! Protocol version, audio format, and the client/server message vocabulary.
//!
//! Control messages are plain Rust types. Their serialization (JSON, protobuf,
//! and so on) is chosen by endpoint adapters; only audio has a defined byte
//! layout (see [`crate::frame`]).

use crate::error::StreamError;
use crate::limits::StreamLimits;

/// Contract version as `major.minor`.
///
/// A different major is incompatible. Within a major, the server accepts a
/// client whose minor is not greater than its own; minor bumps only add
/// optional behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolVersion {
    /// Incompatible-change counter.
    pub major: u16,
    /// Backward-compatible-change counter.
    pub minor: u16,
}

/// Version implemented by this crate.
pub const PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };

/// Wire encoding of one PCM sample. Both encodings are little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleEncoding {
    /// Signed 16-bit integer, decoded as `value / 32768.0`.
    S16Le,
    /// IEEE 754 binary32, which must be finite. Used as-is.
    F32Le,
}

impl SampleEncoding {
    /// Bytes per sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16Le => 2,
            Self::F32Le => 4,
        }
    }
}

/// PCM format declared in `Start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    /// Sample encoding.
    pub encoding: SampleEncoding,
    /// Sample rate in hertz. Version 1 requires [`AudioFormat::REQUIRED_SAMPLE_RATE_HZ`].
    pub sample_rate_hz: u32,
    /// Channel count. Version 1 requires mono (`1`).
    pub channels: u16,
}

impl AudioFormat {
    /// Only sample rate accepted by version 1. Resampling is the client's job.
    pub const REQUIRED_SAMPLE_RATE_HZ: u32 = 16_000;

    /// 16 kHz mono signed 16-bit little-endian PCM.
    pub fn mono_16k_s16() -> Self {
        Self {
            encoding: SampleEncoding::S16Le,
            sample_rate_hz: Self::REQUIRED_SAMPLE_RATE_HZ,
            channels: 1,
        }
    }

    /// 16 kHz mono 32-bit float little-endian PCM.
    pub fn mono_16k_f32() -> Self {
        Self {
            encoding: SampleEncoding::F32Le,
            ..Self::mono_16k_s16()
        }
    }
}

/// One sequenced chunk of PCM bytes in the format declared at `Start`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    /// Frame sequence number. The first frame of a stream is `0`; each
    /// following frame increments by exactly one.
    pub seq: u64,
    /// Raw little-endian sample bytes. A whole number of samples, nonempty.
    pub payload: Vec<u8>,
}

impl AudioFrame {
    /// Build a frame from `i16` samples (test and client helper).
    pub fn from_s16(seq: u64, samples: &[i16]) -> Self {
        Self {
            seq,
            payload: samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
        }
    }

    /// Build a frame from `f32` samples (test and client helper).
    pub fn from_f32(seq: u64, samples: &[f32]) -> Self {
        Self {
            seq,
            payload: samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
        }
    }
}

/// Messages sent by the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientMessage {
    /// Opens the stream. Must be the first message.
    Start {
        /// Client protocol version.
        version: ProtocolVersion,
        /// PCM format of all following audio frames.
        format: AudioFormat,
    },
    /// One audio frame. Allowed only after the server's `Started`.
    Audio(AudioFrame),
    /// End of audio: request the final transcript. No audio may follow.
    Commit {
        /// Total number of audio frames the client sent. Lets the server
        /// detect lost trailing frames.
        frames: u64,
    },
    /// Abandon the stream without a final transcript.
    Cancel,
    /// Graceful goodbye. Legal in any phase; before a terminal event it
    /// abandons the stream like `Cancel`.
    Close,
}

/// Events sent by the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerEvent {
    /// Accepts `Start`; carries the server version, limits, and initial credit.
    Started {
        /// Server protocol version.
        version: ProtocolVersion,
        /// Limits enforced for this stream.
        limits: StreamLimits,
    },
    /// Grants additional audio credit (backpressure release).
    Credit {
        /// Samples added to the client's credit. Nonzero.
        samples: u64,
    },
    /// Speculative transcript for audio received so far. Advisory: partials
    /// may be dropped or coalesced and may be revised by later partials.
    Partial {
        /// Strictly increasing per stream; gaps are allowed.
        revision: u64,
        /// Samples of audio the partial covers.
        audio_samples: u64,
        /// Transcript text.
        text: String,
    },
    /// Final transcript for the committed audio. Terminal.
    Final {
        /// Total samples the transcript covers; equals the samples received.
        audio_samples: u64,
        /// Transcript text.
        text: String,
    },
    /// Acknowledges `Cancel` (or `Close` before a terminal event). Terminal.
    Cancelled,
    /// Stream failed. Terminal.
    Error(StreamError),
}
