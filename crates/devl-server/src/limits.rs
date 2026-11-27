//! Per-stream resource limits.

use crate::error::{ErrorCode, StreamError};

/// Resource limits the server advertises in `Started` and enforces per stream.
///
/// All values count mono samples, not bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamLimits {
    /// Maximum samples in a single audio frame.
    pub max_frame_samples: u64,
    /// Maximum total samples accepted over the life of one stream.
    pub max_stream_samples: u64,
    /// Audio credit granted to the client when the stream starts.
    pub initial_credit_samples: u64,
    /// Upper bound on credit the client may hold at any time.
    pub max_outstanding_credit_samples: u64,
}

impl Default for StreamLimits {
    /// Conservative defaults at 16 kHz: 2 s frames, 5 min streams, 10 s of
    /// initial and maximum outstanding credit. These are server defaults, not
    /// model limits; operators may change them.
    fn default() -> Self {
        Self {
            max_frame_samples: 32_000,
            max_stream_samples: 4_800_000,
            initial_credit_samples: 160_000,
            max_outstanding_credit_samples: 160_000,
        }
    }
}

impl StreamLimits {
    /// Check internal consistency: every value is nonzero and
    /// `max_frame_samples <= initial_credit_samples <= max_outstanding_credit_samples`.
    pub fn validate(&self) -> Result<(), StreamError> {
        let ok = self.max_frame_samples > 0
            && self.max_stream_samples > 0
            && self.initial_credit_samples >= self.max_frame_samples
            && self.max_outstanding_credit_samples >= self.initial_credit_samples;
        if ok {
            Ok(())
        } else {
            Err(StreamError::new(
                ErrorCode::Internal,
                "inconsistent stream limits",
            ))
        }
    }
}
