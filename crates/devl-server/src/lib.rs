//! Transport-neutral streaming contract for DEVL servers.
//!
//! This crate defines the versioned client/server message vocabulary, the
//! binary audio framing, the per-stream limits, and a validating per-stream
//! state machine ([`StreamContract`]). The contract itself performs no I/O and
//! contains no inference or verification logic. The
//! [`coordinator`] module owns one `DevlSession` per stream and drives it
//! through `PriorityScheduler`. The [`ws`] module is a synchronous WebSocket
//! adapter over the coordinator, with JSON control messages in [`wire`].
//! The [`http`] module is an HTTP/1.1 adapter over the same coordinator.
//! The [`endpoint`] module defines optional per-stream VAD and endpointing hooks.
//! Backend integration is separate work.
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
pub mod endpoint;
pub mod error;
mod events;
pub mod frame;
pub mod http;
pub mod limits;
pub mod message;
pub mod state;
pub mod wire;
pub mod ws;

pub use coordinator::{
    Coordinator, CoordinatorConfig, CoordinatorError, CoordinatorStats, Detokenizer, StreamSession,
};
pub use devl_core::StreamId;
pub use endpoint::{
    EndpointConfigError, EndpointDecision, EndpointPolicy, EndpointPolicyFactory,
    SilenceEndpointConfig, SilenceEndpointPolicy,
};
pub use error::{ErrorCode, StreamError};
pub use frame::{FRAME_HEADER_LEN, decode_audio_frame, decode_samples, encode_audio_frame};
pub use http::{HttpConfig, HttpServer, RecordDecoder};
pub use limits::StreamLimits;
pub use message::{
    AudioFormat, AudioFrame, ClientMessage, PROTOCOL_VERSION, ProtocolVersion, SampleEncoding,
    ServerEvent,
};
pub use state::{Accepted, StreamContract, StreamPhase};
pub use ws::{WsConfig, WsServer};
