//! Stable error codes and the stream error type.

use thiserror::Error;

/// Machine-readable error code carried by [`ServerEvent::Error`](crate::ServerEvent::Error).
///
/// Every error is terminal for the stream: after one is emitted the stream
/// accepts only `Close`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Client protocol major version is not supported by the server.
    UnsupportedVersion,
    /// Requested audio format is not supported.
    UnsupportedFormat,
    /// Message is not legal in the current stream phase.
    InvalidMessageOrder,
    /// Frame bytes violate the framing rules.
    MalformedFrame,
    /// Audio frame sequence number or commit frame count does not match.
    SequenceViolation,
    /// Audio frame exceeds the per-frame sample limit.
    FrameTooLarge,
    /// Total stream audio exceeds the per-stream sample limit.
    StreamTooLong,
    /// Client sent more audio than the credit it was granted.
    BackpressureViolation,
    /// Server-side failure unrelated to client behavior.
    Internal,
}

impl ErrorCode {
    /// Stable lowercase identifier for wire encodings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedVersion => "unsupported_version",
            Self::UnsupportedFormat => "unsupported_format",
            Self::InvalidMessageOrder => "invalid_message_order",
            Self::MalformedFrame => "malformed_frame",
            Self::SequenceViolation => "sequence_violation",
            Self::FrameTooLarge => "frame_too_large",
            Self::StreamTooLong => "stream_too_long",
            Self::BackpressureViolation => "backpressure_violation",
            Self::Internal => "internal",
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A contract violation or server failure for one stream.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code}: {message}")]
pub struct StreamError {
    /// Machine-readable code.
    pub code: ErrorCode,
    /// Human-readable detail. Not part of the stable contract.
    pub message: String,
}

impl StreamError {
    /// Build an error from a code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}
