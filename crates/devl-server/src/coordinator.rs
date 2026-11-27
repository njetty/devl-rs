//! Session coordinator: per-stream `DevlSession` ownership and bounded
//! inference dispatch through `PriorityScheduler`.
//!
//! The coordinator is glue. Verification, rollback, draft orchestration, and
//! commit-over-draft ordering stay in `devl-core`: drafts run through
//! `DevlSession::run_draft_cancellable` with the scheduler's
//! `should_preempt_draft` predicate, finals run through `DevlSession::commit`,
//! and job selection is `PriorityScheduler::next_job`.
//!
//! # Model
//!
//! * [`Coordinator`] owns a fixed pool of worker threads and one scheduler.
//! * [`Coordinator::open_stream`] returns a [`StreamSession`]. The handle owns
//!   the connection side of one stream: it feeds [`ClientMessage`]s in with
//!   [`StreamSession::send`] and reads [`ServerEvent`]s with
//!   [`StreamSession::try_recv_event`]. Dropping the handle is a disconnect.
//! * Each stream owns its own `DevlSession` and validating
//!   [`StreamContract`]; no audio, draft, or result is shared across streams.
//!
//! # Draft cadence
//!
//! A checkpoint draft is submitted each time [`CoordinatorConfig::draft_cadence_samples`]
//! new samples have been accepted (default 32 000 samples, 2 s at 16 kHz).
//! Encoding and verification run before the first preemption check, so a
//! coarser cadence bounds how much non-preemptible work a draft adds; see
//! `docs/ARCHITECTURE.md`, Section 6.10 D6.
//!
//! # Bounded buffering
//!
//! * Inbound audio waits in a per-stream pending buffer only while a worker
//!   holds the session. Credit is released (`Credit`) when the buffer is
//!   drained into the session, so pending audio never exceeds the credit the
//!   server granted, and a client that overruns credit gets
//!   `backpressure_violation`.
//! * Outbound events use a bounded queue that coalesces partials and credit.
//! * The scheduler coalesces drafts per stream, so queued jobs are at most one
//!   draft and one commit per stream, and streams are capped by
//!   [`CoordinatorConfig::max_streams`].
//!
//! # Teardown
//!
//! Cancel, close, disconnect, final, error, and shutdown all end a stream the
//! same way: the pending buffer is cleared, the stream's queued scheduler jobs
//! are removed with `PriorityScheduler::cancel_stream`, the `DevlSession` is
//! dropped (by the worker if one is mid-job), and only then does the stream
//! leave the registry and its slot become reusable. A stream whose job is
//! still executing keeps its slot until the worker retires it, so queued plus
//! running jobs never exceed `max_streams` streams' worth (one draft and one
//! commit each) regardless of open/cancel churn. Scheduler permits are RAII
//! and are released when each job finishes, including on panic. Scheduler
//! per-stream metadata is dropped when the stream's last draft stops, so it
//! does not grow with the number of completed streams. Pending `Partial`
//! events are discarded when a terminal event is queued.
//!
//! # Limits
//!
//! `DevlSession::commit` has no preemption hook. A commit already running when
//! a stream is cancelled or the coordinator shuts down runs to completion and
//! its result is discarded, and its stream keeps an admission slot until it
//! returns.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use devl_core::{
    DevlBackend, DevlSession, JobPriority, PriorityScheduler, SchedulerConfig, SessionConfig,
    StreamId, TokenId,
};
use thiserror::Error;

use crate::error::{ErrorCode, StreamError};
use crate::events::EventQueue;
use crate::limits::StreamLimits;
use crate::message::{ClientMessage, PROTOCOL_VERSION, ServerEvent};
use crate::state::{Accepted, StreamContract, StreamPhase};

/// Converts verified token ids to transcript text. Supplied by the backend side.
pub type Detokenizer = Arc<dyn Fn(&[TokenId]) -> String + Send + Sync>;

/// Configuration for [`Coordinator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorConfig {
    /// Inference worker threads. Must be at least 1.
    pub workers: usize,
    /// Scheduler configuration (bounded concurrent drafts).
    pub scheduler: SchedulerConfig,
    /// Per-stream `DevlSession` configuration.
    pub session: SessionConfig,
    /// Per-stream limits advertised in `Started` and enforced by the contract.
    pub limits: StreamLimits,
    /// Accepted samples between checkpoint drafts. Must be at least 1.
    pub draft_cadence_samples: u64,
    /// Maximum simultaneously open streams. Must be at least 1.
    pub max_streams: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            workers: 2,
            scheduler: SchedulerConfig::default(),
            session: SessionConfig::default(),
            limits: StreamLimits::default(),
            draft_cadence_samples: 32_000,
            max_streams: 64,
        }
    }
}

/// Errors from [`Coordinator`] construction and admission.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoordinatorError {
    /// Configuration is inconsistent.
    #[error("invalid coordinator configuration: {0}")]
    InvalidConfig(String),
    /// The coordinator is shutting down or has shut down.
    #[error("coordinator is shut down")]
    ShutDown,
    /// The open-stream cap was reached.
    #[error("stream limit of {max} reached")]
    TooManyStreams {
        /// Configured cap.
        max: usize,
    },
}

/// Point-in-time counters, mainly for tests and operator metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorStats {
    /// Streams currently registered (open and not yet torn down).
    pub open_streams: usize,
    /// Queued commit jobs.
    pub pending_commits: usize,
    /// Queued draft jobs.
    pub pending_drafts: usize,
    /// Executing commit jobs (held scheduler permits).
    pub active_commits: usize,
    /// Executing draft jobs (held scheduler permits).
    pub active_drafts: usize,
    /// Streams with retained scheduler cancellation metadata. Bounded by the
    /// concurrent draft limit, independent of completed streams.
    pub tracked_streams: usize,
}

fn plock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn internal(msg: impl Into<String>) -> StreamError {
    StreamError::new(ErrorCode::Internal, msg)
}

trait StreamControl: Send + Sync {
    /// Fail the stream (unless already terminal) and release its resources.
    fn abort(&self, err: StreamError);
}

struct Registry {
    streams: HashMap<StreamId, Weak<dyn StreamControl>>,
    shutting_down: bool,
}

struct Shared<B: DevlBackend + Clone + Send + Sync + 'static>
where
    B::AudioEncoding: Send,
{
    config: CoordinatorConfig,
    backend: B,
    detokenizer: Detokenizer,
    scheduler: PriorityScheduler<Arc<StreamShared<B>>>,
    registry: Mutex<Registry>,
    next_stream_id: AtomicU64,
}

/// Owns worker threads, the scheduler, and the registry of open streams.
pub struct Coordinator<B: DevlBackend + Clone + Send + Sync + 'static>
where
    B::AudioEncoding: Send,
{
    shared: Arc<Shared<B>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl<B: DevlBackend + Clone + Send + Sync + 'static> Coordinator<B>
where
    B::AudioEncoding: Send,
{
    /// Start the worker pool. Each stream receives a clone of `backend`.
    pub fn new(
        backend: B,
        detokenizer: impl Fn(&[TokenId]) -> String + Send + Sync + 'static,
        config: CoordinatorConfig,
    ) -> Result<Self, CoordinatorError> {
        if config.workers == 0 {
            return Err(CoordinatorError::InvalidConfig(
                "workers must be at least 1".into(),
            ));
        }
        if config.max_streams == 0 {
            return Err(CoordinatorError::InvalidConfig(
                "max_streams must be at least 1".into(),
            ));
        }
        if config.draft_cadence_samples == 0 {
            return Err(CoordinatorError::InvalidConfig(
                "draft_cadence_samples must be at least 1".into(),
            ));
        }
        config
            .limits
            .validate()
            .map_err(|e| CoordinatorError::InvalidConfig(e.message))?;
        config
            .session
            .validate::<B::Error>()
            .map_err(|e| CoordinatorError::InvalidConfig(e.to_string()))?;

        let shared = Arc::new(Shared {
            config,
            backend,
            detokenizer: Arc::new(detokenizer),
            scheduler: PriorityScheduler::new(config.scheduler),
            registry: Mutex::new(Registry {
                streams: HashMap::new(),
                shutting_down: false,
            }),
            next_stream_id: AtomicU64::new(1),
        });

        let workers = (0..config.workers)
            .map(|_| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || worker_loop(&shared))
            })
            .collect();

        Ok(Self {
            shared,
            workers: Mutex::new(workers),
        })
    }

    /// Open a stream and return its connection-side handle.
    pub fn open_stream(&self) -> Result<StreamSession<B>, CoordinatorError> {
        let shared = &self.shared;
        let mut registry = plock(&shared.registry);
        if registry.shutting_down {
            return Err(CoordinatorError::ShutDown);
        }
        if registry.streams.len() >= shared.config.max_streams {
            return Err(CoordinatorError::TooManyStreams {
                max: shared.config.max_streams,
            });
        }
        let id = StreamId(shared.next_stream_id.fetch_add(1, Ordering::Relaxed));
        let contract = StreamContract::new(id, shared.config.limits)
            .map_err(|e| CoordinatorError::InvalidConfig(e.message))?;
        let session = DevlSession::new(shared.backend.clone(), shared.config.session)
            .map_err(|e| CoordinatorError::InvalidConfig(e.to_string()))?;
        let stream = Arc::new(StreamShared {
            id,
            ctx: Arc::clone(shared),
            closed: AtomicBool::new(false),
            state: Mutex::new(StreamState {
                contract,
                pending: Vec::new(),
                samples_since_draft: 0,
                committing: false,
                done: false,
                next_revision: 0,
                last_partial_samples: 0,
                last_partial_text: None,
            }),
            session: Mutex::new(Some(session)),
            events: EventQueue::new(),
        });
        let weak = Arc::downgrade(&stream) as Weak<dyn StreamControl>;
        registry.streams.insert(id, weak);
        Ok(StreamSession { stream })
    }

    /// Current counters.
    pub fn stats(&self) -> CoordinatorStats {
        let s = &self.shared;
        CoordinatorStats {
            open_streams: plock(&s.registry).streams.len(),
            pending_commits: s.scheduler.pending_commits(),
            pending_drafts: s.scheduler.pending_drafts(),
            active_commits: s.scheduler.active_commits(),
            active_drafts: s.scheduler.active_drafts(),
            tracked_streams: s.scheduler.tracked_streams(),
        }
    }

    /// Stop admitting streams, fail every open stream with `internal`
    /// ("server shutting down"), release all sessions, and join the workers.
    ///
    /// Running drafts yield at their next step boundary. A running commit
    /// finishes and its result is discarded. Idempotent; also run on drop.
    pub fn shutdown(&self) {
        let open: Vec<Arc<dyn StreamControl>> = {
            let mut registry = plock(&self.shared.registry);
            registry.shutting_down = true;
            registry
                .streams
                .values()
                .filter_map(Weak::upgrade)
                .collect()
        };
        for stream in open {
            stream.abort(internal("server shutting down"));
        }
        self.shared.scheduler.shutdown();
        let handles = std::mem::take(&mut *plock(&self.workers));
        for handle in handles {
            let _ = handle.join();
        }
    }
}

impl<B: DevlBackend + Clone + Send + Sync + 'static> Drop for Coordinator<B>
where
    B::AudioEncoding: Send,
{
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop<B: DevlBackend + Clone + Send + Sync + 'static>(shared: &Shared<B>)
where
    B::AudioEncoding: Send,
{
    while let Some(job) = shared.scheduler.next_job() {
        let priority = job.priority();
        let seq = job.seq();
        // `_permit` is released at the end of this iteration on every path.
        let (stream, _permit) = job.into_parts();
        let ran = catch_unwind(AssertUnwindSafe(|| match priority {
            JobPriority::Commit => stream.run_commit(),
            JobPriority::Draft => stream.run_draft(seq),
        }));
        if ran.is_err() {
            stream.fail(internal("inference worker panicked"));
        }
        stream.release_session(true);
    }
}

struct StreamState {
    contract: StreamContract,
    /// Audio accepted but not yet in the session. Bounded by granted credit.
    pending: Vec<f32>,
    samples_since_draft: u64,
    committing: bool,
    /// Stream ended; nothing further is emitted or scheduled.
    done: bool,
    next_revision: u64,
    last_partial_samples: u64,
    last_partial_text: Option<String>,
}

/// Lock order: `state` and `session` are never held together by the
/// connection side. Workers take `session` then `state` briefly.
struct StreamShared<B: DevlBackend + Clone + Send + Sync + 'static>
where
    B::AudioEncoding: Send,
{
    id: StreamId,
    ctx: Arc<Shared<B>>,
    /// Lock-free mirror of `state.done`, polled by the preemption predicate.
    closed: AtomicBool,
    state: Mutex<StreamState>,
    session: Mutex<Option<DevlSession<B>>>,
    events: EventQueue,
}

impl<B: DevlBackend + Clone + Send + Sync + 'static> StreamControl for StreamShared<B>
where
    B::AudioEncoding: Send,
{
    fn abort(&self, err: StreamError) {
        self.fail(err);
        self.release_session(false);
    }
}

impl<B: DevlBackend + Clone + Send + Sync + 'static> StreamShared<B>
where
    B::AudioEncoding: Send,
{
    fn handle(self: &Arc<Self>, msg: &ClientMessage) -> Result<(), StreamError> {
        let mut st = plock(&self.state);
        let accepted = match st.contract.receive(msg) {
            Ok(a) => a,
            Err(e) => {
                self.fail_locked(&mut st, e.clone());
                drop(st);
                self.release_session(false);
                return Err(e);
            }
        };
        match accepted {
            Accepted::Started { .. } => {
                let ev = ServerEvent::Started {
                    version: PROTOCOL_VERSION,
                    limits: st.contract.limits(),
                };
                self.emit_locked(&mut st, ev);
            }
            Accepted::Audio { samples, .. } => {
                let n = samples.len() as u64;
                st.pending.extend_from_slice(&samples);
                st.samples_since_draft += n;
                if st.samples_since_draft >= self.ctx.config.draft_cadence_samples
                    && !st.committing
                    && !st.done
                {
                    st.samples_since_draft = 0;
                    self.ctx.scheduler.submit_draft(self.id, Arc::clone(self));
                }
                drop(st);
                self.pump(false);
            }
            Accepted::Commit => {
                if !st.done {
                    st.committing = true;
                    self.ctx.scheduler.submit_commit(self.id, Arc::clone(self));
                }
            }
            Accepted::Cancel => {
                self.emit_locked(&mut st, ServerEvent::Cancelled);
                self.finish_locked(&mut st);
                drop(st);
                self.release_session(false);
            }
            Accepted::Close => {
                self.finish_locked(&mut st);
                drop(st);
                self.release_session(false);
            }
            Accepted::Ignored => {}
        }
        Ok(())
    }

    /// Validate `event` against the contract, then queue it. A rejected event
    /// is a coordinator bug and fails the stream.
    fn emit_locked(&self, st: &mut StreamState, event: ServerEvent) -> bool {
        match st.contract.emit(&event) {
            Ok(()) => {
                self.events.push(event);
                true
            }
            Err(e) => {
                self.fail_locked(st, internal(format!("invalid server event: {}", e.message)));
                false
            }
        }
    }

    fn fail_locked(&self, st: &mut StreamState, err: StreamError) {
        if !st.done && st.contract.emit(&ServerEvent::Error(err.clone())).is_ok() {
            self.events.push(ServerEvent::Error(err));
        }
        self.finish_locked(st);
    }

    fn finish_locked(&self, st: &mut StreamState) {
        st.done = true;
        st.pending = Vec::new();
        self.closed.store(true, Ordering::Release);
    }

    fn fail(&self, err: StreamError) {
        let mut st = plock(&self.state);
        self.fail_locked(&mut st, err);
    }

    /// Disconnect: end the stream without emitting events.
    fn disconnect(&self) {
        {
            let mut st = plock(&self.state);
            self.finish_locked(&mut st);
        }
        self.release_session(false);
    }

    /// Retire an ended stream: remove its queued jobs, drop its session, and
    /// free its registry slot.
    ///
    /// Queued jobs are removed from the scheduler immediately
    /// ([`PriorityScheduler::cancel_stream`]), so ended streams never leave
    /// payloads behind in the queues. The registry slot is freed only once the
    /// session is actually released. `blocking = false` skips the session if a
    /// worker holds it; that worker calls this again with `blocking = true`
    /// after its job. A stream that is still executing therefore keeps its
    /// admission slot until the worker retires, which bounds live sessions
    /// (queued, running, or retiring) by `max_streams`.
    ///
    /// Lock order: no `state` lock is held here. The scheduler lock is taken
    /// only inside `cancel_stream` and never while holding `session`, and
    /// removed payloads are dropped after the scheduler lock is released.
    fn release_session(&self, blocking: bool) {
        if !self.closed.load(Ordering::Acquire) {
            return;
        }
        drop(self.ctx.scheduler.cancel_stream(self.id));
        let session = if blocking {
            Some(plock(&self.session))
        } else {
            match self.session.try_lock() {
                Ok(g) => Some(g),
                Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            }
        };
        if let Some(mut guard) = session {
            let released = guard.take();
            drop(guard);
            drop(released);
            plock(&self.ctx.registry).streams.remove(&self.id);
        }
    }

    /// Move pending audio into the session and release credit.
    ///
    /// Non-blocking callers (the connection side) yield if a worker holds the
    /// session; that worker re-checks `pending` after its job.
    fn pump(&self, blocking: bool) {
        loop {
            {
                let st = plock(&self.state);
                if st.done {
                    drop(st);
                    self.release_session(blocking);
                    return;
                }
                if st.pending.is_empty() {
                    return;
                }
            }
            let mut guard = if blocking {
                plock(&self.session)
            } else {
                match self.session.try_lock() {
                    Ok(g) => g,
                    Err(TryLockError::Poisoned(p)) => p.into_inner(),
                    Err(TryLockError::WouldBlock) => return,
                }
            };
            match guard.as_mut() {
                Some(session) => self.drain_into(session),
                None => return,
            }
        }
    }

    fn drain_into(&self, session: &mut DevlSession<B>) {
        let chunk = {
            let mut st = plock(&self.state);
            if st.done {
                return;
            }
            std::mem::take(&mut st.pending)
        };
        if chunk.is_empty() {
            return;
        }
        if let Err(e) = session.push_audio(&chunk) {
            self.fail(internal(e.to_string()));
            return;
        }
        let mut st = plock(&self.state);
        if !st.done && st.contract.phase() == StreamPhase::Streaming {
            let ev = ServerEvent::Credit {
                samples: chunk.len() as u64,
            };
            self.emit_locked(&mut st, ev);
        }
    }

    fn run_draft(&self, seq: u64) {
        {
            let st = plock(&self.state);
            if st.done || st.committing {
                return;
            }
        }
        let outcome = {
            let mut guard = plock(&self.session);
            let Some(session) = guard.as_mut() else {
                return;
            };
            self.drain_into(session);
            if self.closed.load(Ordering::Acquire) || plock(&self.state).committing {
                return;
            }
            let scheduler = &self.ctx.scheduler;
            // Backend errors in a draft are non-fatal and leave the session
            // intact; the commit pass re-runs the work and surfaces any
            // persistent failure.
            session
                .run_draft_cancellable(|| {
                    self.closed.load(Ordering::Acquire)
                        || scheduler.should_preempt_draft(self.id, seq)
                })
                .ok()
        };
        if let Some(outcome) = outcome {
            let result = outcome.result();
            let text = (self.ctx.detokenizer)(&result.tokens);
            self.publish_partial(result.audio_samples as u64, text);
        }
        self.pump(true);
    }

    /// Emit a partial unless it is stale: the stream committed, was cancelled
    /// or failed, covers less audio than a partial already sent, or repeats
    /// the previous text.
    fn publish_partial(&self, audio_samples: u64, text: String) {
        let mut st = plock(&self.state);
        if st.done
            || st.committing
            || st.contract.phase() != StreamPhase::Streaming
            || audio_samples < st.last_partial_samples
            || st.last_partial_text.as_deref() == Some(text.as_str())
        {
            return;
        }
        let revision = st.next_revision;
        let ev = ServerEvent::Partial {
            revision,
            audio_samples,
            text: text.clone(),
        };
        if self.emit_locked(&mut st, ev) {
            st.next_revision += 1;
            st.last_partial_samples = audio_samples;
            st.last_partial_text = Some(text);
        }
    }

    fn run_commit(&self) {
        let result = {
            let mut guard = plock(&self.session);
            let Some(session) = guard.as_mut() else {
                return;
            };
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            self.drain_into(session);
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            let samples = session.audio_buffer().len() as u64;
            session
                .commit()
                .map(|r| (r.tokens.clone(), samples))
                .map_err(|e| e.to_string())
        };
        let mut st = plock(&self.state);
        if st.done {
            // Cancelled or shut down while committing: discard the result.
            return;
        }
        match result {
            Ok((tokens, samples)) => {
                let ev = ServerEvent::Final {
                    audio_samples: samples,
                    text: (self.ctx.detokenizer)(&tokens),
                };
                if self.emit_locked(&mut st, ev) {
                    self.finish_locked(&mut st);
                }
            }
            Err(msg) => self.fail_locked(&mut st, internal(msg)),
        }
    }
}

/// Connection-side handle for one stream.
///
/// Dropping the handle is a disconnect: the stream's session, pending audio,
/// and registry slot are released without emitting events.
pub struct StreamSession<B: DevlBackend + Clone + Send + Sync + 'static>
where
    B::AudioEncoding: Send,
{
    stream: Arc<StreamShared<B>>,
}

impl<B: DevlBackend + Clone + Send + Sync + 'static> StreamSession<B>
where
    B::AudioEncoding: Send,
{
    /// Identifier used for this stream's scheduler jobs.
    pub fn stream_id(&self) -> StreamId {
        self.stream.id
    }

    /// Current contract phase.
    pub fn phase(&self) -> StreamPhase {
        plock(&self.stream.state).contract.phase()
    }

    /// Audio samples accepted but not yet handed to the session.
    pub fn pending_samples(&self) -> usize {
        plock(&self.stream.state).pending.len()
    }

    /// Process one client message.
    ///
    /// `Err` means the message violated the contract; the matching `Error`
    /// event has been queued and the stream is terminal. Late cancels and
    /// in-flight messages after a cancel return `Ok`.
    pub fn send(&self, msg: &ClientMessage) -> Result<(), StreamError> {
        self.stream.handle(msg)
    }

    /// Next queued event, if any.
    pub fn try_recv_event(&self) -> Option<ServerEvent> {
        self.stream.events.try_pop()
    }

    /// Wait up to `timeout` for the next event.
    pub fn recv_event_timeout(&self, timeout: Duration) -> Option<ServerEvent> {
        self.stream.events.pop_timeout(timeout)
    }
}

impl<B: DevlBackend + Clone + Send + Sync + 'static> Drop for StreamSession<B>
where
    B::AudioEncoding: Send,
{
    fn drop(&mut self) {
        self.stream.disconnect();
    }
}
