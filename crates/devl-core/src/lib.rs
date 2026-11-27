//! Core protocol, session state machine, and priority scheduler for
//! **DEVL: Decode Early, Verify Late** (arXiv:2611.20064).
//!
//! `devl-core` is runtime- and transport-agnostic: it contains zero ML
//! framework or networking dependencies. Concrete inference runtimes implement
//! [`DevlBackend`] in downstream crates (such as `devl-candle`).

pub mod backend;
pub mod scheduler;
pub mod session;

#[cfg(any(test, feature = "test-utils"))]
pub mod mock;

pub use backend::{DevlBackend, StepOutput, TokenId, VerifyOutput};
pub use scheduler::{
    JobPermit, JobPriority, PriorityScheduler, ScheduledJob, SchedulerConfig, StreamId,
};
pub use session::{
    CommitResult, DevlError, DevlSession, DraftOutcome, DraftResult, SessionConfig, SessionState,
};
