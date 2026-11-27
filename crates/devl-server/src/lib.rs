//! Transport-neutral streaming contract for DEVL servers.
//!
//! This crate defines the versioned client/server message vocabulary, the
//! binary audio framing, the per-stream limits, and a validating per-stream
//! state machine ([`StreamContract`]). It performs no I/O, owns no async
//! runtime, and contains no inference or verification logic. WebSocket/HTTP
//! adapters, the scheduler coordinator, and backend integration are separate
//! work that consume these types. The [`coordinator`] module is the first such
//! consumer: it owns one `DevlSession` per stream and drives it through
//! `PriorityScheduler`.
//!
//! Protocol logic (verification, rollback, commit-over-draft scheduling)
//! stays in `devl-core`. The contract only decides which messages are legal,
//! how audio bytes become `f32` samples for `DevlSession::push_audio`, and
//! when a stream must fail.
//!
//! See `docs/SERVER_CONTRACT.md` for the prose specification.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/njetty/devl-rs/main/assets/icon-ink.svg",
    html_favicon_url = "https://raw.githubusercontent.com/njetty/devl-rs/main/assets/icon-ink.svg"
)]

pub mod coordinator;
pub mod error;
mod events;
pub mod frame;
pub mod limits;
pub mod message;
pub mod state;

pub use coordinator::{
    Coordinator, CoordinatorConfig, CoordinatorError, CoordinatorStats, Detokenizer, StreamSession,
};
pub use devl_core::StreamId;
pub use error::{ErrorCode, StreamError};
pub use frame::{FRAME_HEADER_LEN, decode_audio_frame, decode_samples, encode_audio_frame};
pub use limits::StreamLimits;
pub use message::{
    AudioFormat, AudioFrame, ClientMessage, PROTOCOL_VERSION, ProtocolVersion, SampleEncoding,
    ServerEvent,
};
pub use state::{Accepted, StreamContract, StreamPhase};
