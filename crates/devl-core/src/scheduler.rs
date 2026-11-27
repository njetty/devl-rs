//! Final-over-draft priority scheduler with per-stream coalescing and
//! cooperative step-level preemption.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Opaque identifier for a logical streaming utterance / client connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u64);

/// Scheduling priority tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobPriority {
    /// Final utterance verification and residual decode (strictly higher priority).
    Commit,
    /// Background speculative checkpoint draft.
    Draft,
}

/// Configuration for [`PriorityScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Maximum number of checkpoint drafts permitted to execute concurrently.
    /// Defaults to `4` (Section 2 of arXiv:2611.20064v1).
    pub max_concurrent_drafts: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_drafts: 4,
        }
    }
}

struct QueuedJob<T> {
    stream_id: StreamId,
    seq: u64,
    payload: T,
}

struct QueueState<T> {
    next_seq: u64,
    commit_queue: VecDeque<QueuedJob<T>>,
    draft_queue: VecDeque<QueuedJob<T>>,
    active_commits: usize,
    active_drafts: usize,
    /// Most recent commit sequence number per stream, used to cancel older
    /// in-flight drafts for the same stream without ABA races.
    last_commit_seq: HashMap<StreamId, u64>,
    shutdown: bool,
}

struct SchedulerInner<T> {
    config: SchedulerConfig,
    state: Mutex<QueueState<T>>,
    condvar: Condvar,
    pending_commits: AtomicUsize,
    has_cancelled_streams: AtomicBool,
}

/// RAII guard representing an active scheduler execution slot.
///
/// Dropping `JobPermit` decrements the active commit/draft counter and wakes
/// workers waiting in [`PriorityScheduler::next_job`].
pub struct JobPermit {
    on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl Drop for JobPermit {
    fn drop(&mut self) {
        if let Some(release) = self.on_drop.take() {
            release();
        }
    }
}

impl std::fmt::Debug for JobPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobPermit").finish_non_exhaustive()
    }
}

/// A dequeued job paired with its RAII slot permit.
pub struct ScheduledJob<T> {
    stream_id: StreamId,
    priority: JobPriority,
    seq: u64,
    payload: T,
    permit: JobPermit,
}

impl<T> ScheduledJob<T> {
    /// Stream that submitted this job.
    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    /// Priority tier of this job.
    pub fn priority(&self) -> JobPriority {
        self.priority
    }

    /// Monotonic submission sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Borrow the job payload.
    pub fn payload(&self) -> &T {
        &self.payload
    }

    /// Consume the job into `(payload, permit)`. Keep `permit` alive until the
    /// job finishes executing so concurrency accounting remains accurate.
    pub fn into_parts(self) -> (T, JobPermit) {
        (self.payload, self.permit)
    }
}

/// Thread-safe, runtime-agnostic priority scheduler that always prioritizes
/// [`JobPriority::Commit`] jobs ahead of [`JobPriority::Draft`] jobs.
pub struct PriorityScheduler<T> {
    inner: Arc<SchedulerInner<T>>,
}

impl<T> Clone for PriorityScheduler<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Send + 'static> PriorityScheduler<T> {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            inner: Arc::new(SchedulerInner {
                config,
                state: Mutex::new(QueueState {
                    next_seq: 1,
                    commit_queue: VecDeque::new(),
                    draft_queue: VecDeque::new(),
                    active_commits: 0,
                    active_drafts: 0,
                    last_commit_seq: HashMap::new(),
                    shutdown: false,
                }),
                condvar: Condvar::new(),
                pending_commits: AtomicUsize::new(0),
                has_cancelled_streams: AtomicBool::new(false),
            }),
        }
    }

    /// Return the scheduler configuration.
    pub fn config(&self) -> SchedulerConfig {
        self.inner.config
    }

    /// Enqueue a high-priority commit job for `stream_id`.
    ///
    /// Immediately evicts any queued draft for the same `stream_id` and marks
    /// any currently executing draft for that stream for preemption.
    pub fn submit_commit(&self, stream_id: StreamId, payload: T) -> u64 {
        let mut state = self.inner.state.lock().expect("scheduler mutex poisoned");
        let seq = state.next_seq;
        state.next_seq += 1;

        // Evict stale queued drafts for this stream.
        state.draft_queue.retain(|job| job.stream_id != stream_id);
        state.last_commit_seq.insert(stream_id, seq);
        self.inner
            .has_cancelled_streams
            .store(true, Ordering::Release);

        state.commit_queue.push_back(QueuedJob {
            stream_id,
            seq,
            payload,
        });
        self.inner.pending_commits.fetch_add(1, Ordering::Release);
        self.inner.condvar.notify_all();
        seq
    }

    /// Enqueue a low-priority checkpoint draft job for `stream_id`.
    ///
    /// Coalesces per stream: if an older draft for `stream_id` is already
    /// queued, it is replaced by the newer checkpoint.
    pub fn submit_draft(&self, stream_id: StreamId, payload: T) -> u64 {
        let mut state = self.inner.state.lock().expect("scheduler mutex poisoned");
        let seq = state.next_seq;
        state.next_seq += 1;

        state.draft_queue.retain(|job| job.stream_id != stream_id);
        state.draft_queue.push_back(QueuedJob {
            stream_id,
            seq,
            payload,
        });
        self.inner.condvar.notify_one();
        seq
    }

    /// Cooperative step-level preemption check for an executing draft job.
    ///
    /// Returns `true` if:
    /// - Any commit job is currently queued waiting for execution, or
    /// - `stream_id` received a commit newer than `draft_seq`.
    pub fn should_preempt_draft(&self, stream_id: StreamId, draft_seq: u64) -> bool {
        if self.inner.pending_commits.load(Ordering::Acquire) > 0 {
            return true;
        }
        if !self.inner.has_cancelled_streams.load(Ordering::Acquire) {
            return false;
        }
        let state = self.inner.state.lock().expect("scheduler mutex poisoned");
        state
            .last_commit_seq
            .get(&stream_id)
            .is_some_and(|&commit_seq| commit_seq > draft_seq)
    }

    /// Non-blocking dequeue of the highest-priority runnable job.
    ///
    /// Always prefers [`JobPriority::Commit`] over [`JobPriority::Draft`].
    /// Drafts are only returned when the commit queue is empty and
    /// `active_drafts < config.max_concurrent_drafts`.
    pub fn try_next_job(&self) -> Option<ScheduledJob<T>> {
        let mut state = self.inner.state.lock().expect("scheduler mutex poisoned");
        self.pop_runnable_locked(&mut state)
    }

    /// Blocking dequeue of the highest-priority runnable job.
    ///
    /// Returns `None` once [`Self::shutdown`] has been called and no runnable
    /// jobs remain.
    pub fn next_job(&self) -> Option<ScheduledJob<T>> {
        let mut state = self.inner.state.lock().expect("scheduler mutex poisoned");
        loop {
            if let Some(job) = self.pop_runnable_locked(&mut state) {
                return Some(job);
            }
            let draft_blocked_forever =
                state.draft_queue.is_empty() || self.inner.config.max_concurrent_drafts == 0;
            if state.shutdown && state.commit_queue.is_empty() && draft_blocked_forever {
                return None;
            }
            state = self
                .inner
                .condvar
                .wait(state)
                .expect("scheduler mutex poisoned");
        }
    }

    /// Signal workers to exit once remaining runnable jobs drain.
    pub fn shutdown(&self) {
        let mut state = self.inner.state.lock().expect("scheduler mutex poisoned");
        state.shutdown = true;
        self.inner.condvar.notify_all();
    }

    /// Number of queued commit jobs waiting to start.
    pub fn pending_commits(&self) -> usize {
        self.inner.pending_commits.load(Ordering::Acquire)
    }

    /// Number of queued draft jobs waiting to start.
    pub fn pending_drafts(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("scheduler mutex poisoned")
            .draft_queue
            .len()
    }

    /// Number of currently executing commit jobs.
    pub fn active_commits(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("scheduler mutex poisoned")
            .active_commits
    }

    /// Number of currently executing draft jobs.
    pub fn active_drafts(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("scheduler mutex poisoned")
            .active_drafts
    }

    fn pop_runnable_locked(&self, state: &mut QueueState<T>) -> Option<ScheduledJob<T>> {
        if let Some(job) = state.commit_queue.pop_front() {
            self.inner.pending_commits.fetch_sub(1, Ordering::AcqRel);
            state.active_commits += 1;
            let permit = self.make_permit(JobPriority::Commit);
            return Some(ScheduledJob {
                stream_id: job.stream_id,
                priority: JobPriority::Commit,
                seq: job.seq,
                payload: job.payload,
                permit,
            });
        }

        if state.active_drafts < self.inner.config.max_concurrent_drafts
            && let Some(job) = state.draft_queue.pop_front()
        {
            state.active_drafts += 1;
            let permit = self.make_permit(JobPriority::Draft);
            return Some(ScheduledJob {
                stream_id: job.stream_id,
                priority: JobPriority::Draft,
                seq: job.seq,
                payload: job.payload,
                permit,
            });
        }

        None
    }

    fn make_permit(&self, priority: JobPriority) -> JobPermit {
        let inner = Arc::clone(&self.inner);
        JobPermit {
            on_drop: Some(Box::new(move || {
                let mut state = inner.state.lock().expect("scheduler mutex poisoned");
                match priority {
                    JobPriority::Commit => {
                        state.active_commits = state.active_commits.saturating_sub(1);
                    }
                    JobPriority::Draft => {
                        state.active_drafts = state.active_drafts.saturating_sub(1);
                    }
                }
                inner.condvar.notify_all();
            })),
        }
    }
}
