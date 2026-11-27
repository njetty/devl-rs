//! Deterministic mock-backed tests for the session coordinator.
//!
//! Concurrency is ordered with explicit gates: the mock backend's step hook
//! parks a worker inside `decode_step` until the test releases it, so every
//! queueing decision is made while the worker is at a known point.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{
    DevlBackend, SchedulerConfig, SessionConfig, StepOutput, StreamId, TokenId, VerifyOutput,
};
use devl_server::*;

const WAIT: Duration = Duration::from_secs(20);
const SAMPLE_VALUE: i16 = 100;

/// Mock wrapper that counts live clones so tests can prove sessions are dropped.
struct Tracked {
    inner: MockBackend,
    live: Arc<AtomicUsize>,
}

impl Tracked {
    fn new(inner: MockBackend) -> (Self, Arc<AtomicUsize>) {
        let live = Arc::new(AtomicUsize::new(1));
        (
            Self {
                inner,
                live: Arc::clone(&live),
            },
            live,
        )
    }
}

impl Clone for Tracked {
    fn clone(&self) -> Self {
        self.live.fetch_add(1, Ordering::SeqCst);
        Self {
            inner: self.inner.clone(),
            live: Arc::clone(&self.live),
        }
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

impl DevlBackend for Tracked {
    type Error = <MockBackend as DevlBackend>::Error;
    type AudioEncoding = <MockBackend as DevlBackend>::AudioEncoding;
    type KvCache = <MockBackend as DevlBackend>::KvCache;

    fn eos_token(&self) -> TokenId {
        self.inner.eos_token()
    }
    fn new_kv_cache(&self) -> Result<Self::KvCache, Self::Error> {
        self.inner.new_kv_cache()
    }
    fn encode_audio(
        &self,
        pcm: &[f32],
        prior: Option<&Self::AudioEncoding>,
    ) -> Result<Self::AudioEncoding, Self::Error> {
        self.inner.encode_audio(pcm, prior)
    }
    fn verify_draft(
        &self,
        enc: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        draft: &[TokenId],
    ) -> Result<VerifyOutput, Self::Error> {
        self.inner.verify_draft(enc, kv, draft)
    }
    fn rollback_kv(&self, kv: &mut Self::KvCache, n: usize) -> Result<(), Self::Error> {
        self.inner.rollback_kv(kv, n)
    }
    fn decode_step(
        &self,
        enc: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        tok: TokenId,
    ) -> Result<StepOutput, Self::Error> {
        self.inner.decode_step(enc, kv, tok)
    }
}

/// Test side of a gate that parks the first `blocks` decode steps.
struct Gate {
    entered: Receiver<()>,
    release: Sender<()>,
}

impl Gate {
    fn wait_entered(&self) {
        self.entered
            .recv_timeout(WAIT)
            .expect("worker never entered the gate");
    }
    fn release(&self) {
        self.release.send(()).unwrap();
    }
}

/// Decrement a counter if positive. `true` means a token was taken.
fn take_one(counter: &Mutex<usize>) -> bool {
    let mut n = counter.lock().unwrap();
    if *n == 0 {
        return false;
    }
    *n -= 1;
    true
}

fn gated(transcripts: Vec<(usize, Vec<TokenId>)>, blocks: usize) -> (MockBackend, Gate) {
    let (entered_tx, entered) = channel();
    let (release, release_rx) = channel();
    let entered_tx = Mutex::new(entered_tx);
    let release_rx = Mutex::new(release_rx);
    let remaining = Mutex::new(blocks);
    let backend = MockBackend::with_transcripts(transcripts).with_step_hook(move |_| {
        if take_one(&remaining) {
            let _ = entered_tx.lock().unwrap().send(());
            // Bounded so a failing test cannot hang worker teardown.
            let _ = release_rx.lock().unwrap().recv_timeout(WAIT);
        }
    });
    (backend, Gate { entered, release })
}

fn text(tokens: &[TokenId]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

type Coord = Coordinator<Tracked>;

fn coordinator(backend: MockBackend, config: CoordinatorConfig) -> (Coord, Arc<AtomicUsize>) {
    let (tracked, live) = Tracked::new(backend);
    (Coordinator::new(tracked, text, config).unwrap(), live)
}

fn config() -> CoordinatorConfig {
    CoordinatorConfig {
        workers: 1,
        scheduler: SchedulerConfig {
            max_concurrent_drafts: 1,
        },
        ..CoordinatorConfig::default()
    }
}

fn audio(seq: u64, n: usize) -> ClientMessage {
    ClientMessage::Audio(AudioFrame::from_s16(seq, &vec![SAMPLE_VALUE; n]))
}

fn commit(frames: u64) -> ClientMessage {
    ClientMessage::Commit { frames }
}

fn start(c: &Coord) -> StreamSession<Tracked> {
    let s = c.open_stream().unwrap();
    s.send(&ClientMessage::Start {
        version: PROTOCOL_VERSION,
        format: AudioFormat::mono_16k_s16(),
    })
    .unwrap();
    assert!(matches!(next(&s), ServerEvent::Started { .. }));
    s
}

fn next(s: &StreamSession<Tracked>) -> ServerEvent {
    s.recv_event_timeout(WAIT)
        .expect("timed out waiting for event")
}

fn is_terminal(e: &ServerEvent) -> bool {
    matches!(
        e,
        ServerEvent::Final { .. } | ServerEvent::Cancelled | ServerEvent::Error(_)
    )
}

/// Collect events up to and including the terminal one.
fn until_terminal(s: &StreamSession<Tracked>) -> Vec<ServerEvent> {
    let mut out = Vec::new();
    loop {
        let e = next(s);
        let done = is_terminal(&e);
        out.push(e);
        if done {
            return out;
        }
    }
}

fn wait_partial(s: &StreamSession<Tracked>) -> (u64, String) {
    loop {
        if let ServerEvent::Partial {
            audio_samples,
            text,
            ..
        } = next(s)
        {
            return (audio_samples, text);
        }
    }
}

fn final_of(events: &[ServerEvent]) -> (u64, String) {
    match events.last() {
        Some(ServerEvent::Final {
            audio_samples,
            text,
        }) => (*audio_samples, text.clone()),
        other => panic!("expected Final, got {other:?}"),
    }
}

fn reference(mock: &MockBackend, n: usize) -> String {
    let pcm = vec![f32::from(SAMPLE_VALUE) / 32768.0; n];
    text(&mock.reference_offline_decode(&pcm, 448))
}

fn assert_clean(c: &Coord, live: &AtomicUsize) {
    c.shutdown();
    let s = c.stats();
    assert_eq!(
        (s.open_streams, s.pending_commits, s.pending_drafts),
        (0, 0, 0)
    );
    assert_eq!((s.active_commits, s.active_drafts), (0, 0));
    assert_eq!(live.load(Ordering::SeqCst), 1, "a session was not dropped");
}

#[test]
fn independent_streams_do_not_leak_audio_or_results() {
    let mock = MockBackend::with_transcripts([
        (160, vec![11, 12, 13, MOCK_EOS_TOKEN]),
        (320, vec![21, 22, MOCK_EOS_TOKEN]),
    ]);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            workers: 2,
            ..config()
        },
    );
    let a = start(&c);
    let b = start(&c);
    assert_ne!(a.stream_id(), b.stream_id());

    a.send(&audio(0, 80)).unwrap();
    b.send(&audio(0, 160)).unwrap();
    a.send(&audio(1, 80)).unwrap();
    b.send(&audio(1, 160)).unwrap();
    a.send(&commit(2)).unwrap();
    b.send(&commit(2)).unwrap();

    assert_eq!(final_of(&until_terminal(&a)), (160, "11 12 13 0".into()));
    assert_eq!(final_of(&until_terminal(&b)), (320, "21 22 0".into()));
    drop((a, b));
    assert_clean(&c, &live);
}

#[test]
fn partials_then_final_match_offline_decode() {
    let mock = MockBackend::with_transcripts([
        (160, vec![11, 12, MOCK_EOS_TOKEN]),
        (320, vec![11, 12, 13, MOCK_EOS_TOKEN]),
    ]);
    let expected = reference(&mock, 320);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 160,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 160)).unwrap();
    assert_eq!(wait_partial(&s), (160, "11 12 0".into()));
    s.send(&audio(1, 160)).unwrap();
    s.send(&commit(2)).unwrap();
    let events = until_terminal(&s);
    assert_eq!(final_of(&events), (320, expected));
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn queued_final_beats_queued_draft_and_running_draft_yields() {
    let log: Arc<Mutex<Vec<(TokenId, usize)>>> = Arc::default();
    let (mock, gate) = gated(
        vec![
            (160, vec![101, 102, 103, 104, 105, 106, MOCK_EOS_TOKEN]),
            (320, vec![201, 202, 203, MOCK_EOS_TOKEN]),
            (480, vec![301, 302, MOCK_EOS_TOKEN]),
        ],
        1,
    );
    let (tracked, live) = Tracked::new(mock);
    let sink = Arc::clone(&log);
    let c = Coordinator::new(
        tracked,
        move |t: &[TokenId]| {
            sink.lock().unwrap().push((t[0], t.len()));
            text(t)
        },
        CoordinatorConfig {
            draft_cadence_samples: 160,
            ..config()
        },
    )
    .unwrap();

    let a = start(&c);
    let b = start(&c);
    let d = start(&c);

    // The only worker parks inside A's first decode step.
    a.send(&audio(0, 160)).unwrap();
    gate.wait_entered();

    // B's draft queues. D's draft queues, then D's commit evicts it.
    b.send(&audio(0, 320)).unwrap();
    d.send(&audio(0, 480)).unwrap();
    d.send(&commit(1)).unwrap();
    assert_eq!(c.stats().pending_commits, 1);
    assert_eq!(c.stats().pending_drafts, 1);

    gate.release();
    assert_eq!(final_of(&until_terminal(&d)), (480, "301 302 0".into()));
    assert_eq!(wait_partial(&b), (320, "201 202 203 0".into()));
    assert_eq!(wait_partial(&a), (160, "101 102".into()));

    // A yielded after one step, the final ran next, the queued draft last.
    assert_eq!(*log.lock().unwrap(), vec![(101, 2), (301, 3), (201, 4)]);
    drop((a, b, d));
    assert_clean(&c, &live);
}

#[test]
fn partial_is_suppressed_after_commit() {
    let (mock, gate) = gated(vec![(160, vec![101, 102, 103, 104, MOCK_EOS_TOKEN])], 1);
    // Reference decode uses an ungated twin so it cannot consume the gate.
    let expected = reference(
        &MockBackend::with_transcripts([(160, vec![101, 102, 103, 104, MOCK_EOS_TOKEN])]),
        160,
    );
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 160,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 160)).unwrap();
    gate.wait_entered();
    s.send(&commit(1)).unwrap();
    gate.release();

    let events = until_terminal(&s);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ServerEvent::Partial { .. })),
        "stale partial leaked: {events:?}"
    );
    assert_eq!(final_of(&events), (160, expected));
    drop(s);
    assert_clean(&c, &live);
}

fn tight_limits() -> StreamLimits {
    StreamLimits {
        max_frame_samples: 8,
        max_stream_samples: 64,
        initial_credit_samples: 16,
        max_outstanding_credit_samples: 16,
    }
}

/// Stream whose worker is parked mid-draft with 8 samples already drained.
fn parked_with_credit_returned() -> (Coord, Arc<AtomicUsize>, Gate, StreamSession<Tracked>) {
    let (mock, gate) = gated(vec![(8, vec![1, 2, 3, 4, 5, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            limits: tight_limits(),
            draft_cadence_samples: 8,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 8)).unwrap();
    gate.wait_entered();
    assert_eq!(next(&s), ServerEvent::Credit { samples: 8 });
    (c, live, gate, s)
}

#[test]
fn credit_is_withheld_while_session_is_busy_then_released() {
    let (c, live, gate, s) = parked_with_credit_returned();
    s.send(&audio(1, 8)).unwrap();
    s.send(&audio(2, 8)).unwrap();
    assert_eq!(s.pending_samples(), 16);
    assert!(s.try_recv_event().is_none(), "credit released too early");

    gate.release();
    let mut granted = 0;
    while granted < 16 {
        if let ServerEvent::Credit { samples } = next(&s) {
            granted += samples;
        }
    }
    assert_eq!(granted, 16);
    assert_eq!(s.pending_samples(), 0);
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn overrunning_credit_fails_instead_of_buffering() {
    let (c, live, gate, s) = parked_with_credit_returned();
    s.send(&audio(1, 8)).unwrap();
    s.send(&audio(2, 8)).unwrap();
    let err = s.send(&audio(3, 8)).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackpressureViolation);
    assert_eq!(s.pending_samples(), 0, "pending audio must be cleared");
    assert!(
        matches!(next(&s), ServerEvent::Error(e) if e.code == ErrorCode::BackpressureViolation)
    );
    gate.release();
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn cancel_during_draft_stops_partials_and_releases_everything() {
    let (mock, gate) = gated(vec![(160, vec![1, 2, 3, 4, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 160,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 160)).unwrap();
    gate.wait_entered();
    s.send(&ClientMessage::Cancel).unwrap();
    // A duplicate cancel is ignored. Audio after the acknowledged cancel is
    // an ordering error that must not emit a second terminal event.
    s.send(&ClientMessage::Cancel).unwrap();
    assert_eq!(
        s.send(&audio(1, 8)).unwrap_err().code,
        ErrorCode::InvalidMessageOrder
    );
    // The running draft holds the slot until the worker retires.
    assert_eq!(c.stats().open_streams, 1);
    gate.release();
    wait_open_streams(&c, 0);

    c.shutdown();
    let mut events = Vec::new();
    while let Some(e) = s.try_recv_event() {
        events.push(e);
    }
    assert_eq!(events.iter().filter(|e| is_terminal(e)).count(), 1);
    assert_eq!(events.last(), Some(&ServerEvent::Cancelled));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ServerEvent::Partial { .. }))
    );
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn cancel_during_commit_discards_the_final() {
    let (mock, gate) = gated(vec![(8, vec![1, 2, 3, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(mock, config());
    let s = start(&c);
    s.send(&audio(0, 8)).unwrap();
    s.send(&commit(1)).unwrap();
    gate.wait_entered();
    assert_eq!(c.stats().active_commits, 1);
    s.send(&ClientMessage::Cancel).unwrap();
    gate.release();

    c.shutdown();
    let mut events = Vec::new();
    while let Some(e) = s.try_recv_event() {
        events.push(e);
    }
    assert_eq!(events.last(), Some(&ServerEvent::Cancelled));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ServerEvent::Final { .. }))
    );
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn disconnect_releases_session_and_frees_the_slot() {
    let (mock, gate) = gated(vec![(160, vec![1, 2, 3, 4, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 160,
            max_streams: 1,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 160)).unwrap();
    gate.wait_entered();
    assert_eq!(
        c.open_stream().err(),
        Some(CoordinatorError::TooManyStreams { max: 1 })
    );

    drop(s);
    // The running draft still holds the slot until the worker retires.
    assert_eq!(c.stats().open_streams, 1);
    gate.release();
    wait_open_streams(&c, 0);
    let replacement = start(&c);
    drop(replacement);
    assert_clean(&c, &live);
}

#[test]
fn shutdown_fails_open_streams_and_releases_permits() {
    let (mock, gate) = gated(vec![(160, vec![1, 2, 3, 4, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 160,
            ..config()
        },
    );
    let running = start(&c);
    let queued = start(&c);
    let idle = start(&c);
    running.send(&audio(0, 160)).unwrap();
    gate.wait_entered();
    queued.send(&audio(0, 160)).unwrap();
    idle.send(&audio(0, 8)).unwrap();

    std::thread::scope(|scope| {
        let stopper = scope.spawn(|| c.shutdown());
        for s in [&running, &queued, &idle] {
            let events = until_terminal(s);
            assert!(
                matches!(events.last(), Some(ServerEvent::Error(e)) if e.code == ErrorCode::Internal),
                "{events:?}"
            );
        }
        gate.release();
        stopper.join().unwrap();
    });

    assert_eq!(c.open_stream().err(), Some(CoordinatorError::ShutDown));
    for s in [&running, &queued, &idle] {
        assert!(s.try_recv_event().is_none());
    }
    drop((running, queued, idle));
    assert_clean(&c, &live);
}

#[test]
fn finalized_stream_releases_session_and_registry_slot() {
    let (mock, _gate) = gated(vec![(8, vec![1, MOCK_EOS_TOKEN])], 0);
    let (c, live) = coordinator(mock, config());
    let s = start(&c);
    s.send(&audio(0, 8)).unwrap();
    s.send(&commit(1)).unwrap();
    assert_eq!(final_of(&until_terminal(&s)), (8, "1 0".into()));
    // Close after the terminal event just ends the stream.
    s.send(&ClientMessage::Close).unwrap();
    assert_clean(&c, &live);
    assert_eq!(s.phase(), StreamPhase::Finalized);
}

#[test]
fn admission_is_capped_and_slots_are_reused() {
    let (mock, _gate) = gated(vec![], 0);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            max_streams: 2,
            ..config()
        },
    );
    let a = c.open_stream().unwrap();
    let _b = c.open_stream().unwrap();
    assert_eq!(
        c.open_stream().err(),
        Some(CoordinatorError::TooManyStreams { max: 2 })
    );
    drop(a);
    let _c = c.open_stream().unwrap();
    drop((_b, _c));
    assert_clean(&c, &live);
}

#[test]
fn contract_violation_emits_error_and_releases() {
    let (mock, _gate) = gated(vec![], 0);
    let (c, live) = coordinator(mock, config());
    let s = start(&c);
    let err = s.send(&audio(5, 8)).unwrap_err();
    assert_eq!(err.code, ErrorCode::SequenceViolation);
    assert!(matches!(next(&s), ServerEvent::Error(e) if e.code == ErrorCode::SequenceViolation));
    assert_eq!(c.stats().open_streams, 0);
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn backend_failure_on_commit_is_an_internal_error() {
    let mock = MockBackend::with_transcripts([(8, vec![1, MOCK_EOS_TOKEN])]);
    let (c, live) = coordinator(mock.clone(), config());
    let s = start(&c);
    s.send(&audio(0, 8)).unwrap();
    mock.fail_next_encode();
    s.send(&commit(1)).unwrap();
    let events = until_terminal(&s);
    assert!(matches!(
        events.last(),
        Some(ServerEvent::Error(e)) if e.code == ErrorCode::Internal
    ));
    s.send(&ClientMessage::Close).unwrap();
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn backend_panic_fails_only_that_stream() {
    let panics = Mutex::new(1usize);
    let mock = MockBackend::with_transcripts([
        (160, vec![1, 2, 3, MOCK_EOS_TOKEN]),
        (8, vec![7, MOCK_EOS_TOKEN]),
    ])
    .with_step_hook(move |_| {
        if take_one(&panics) {
            panic!("injected backend panic");
        }
    });
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 160,
            ..config()
        },
    );
    let bad = start(&c);
    bad.send(&audio(0, 160)).unwrap();
    let events = until_terminal(&bad);
    assert!(matches!(
        events.last(),
        Some(ServerEvent::Error(e)) if e.code == ErrorCode::Internal
    ));

    // The single worker survived and still serves other streams.
    let good = start(&c);
    good.send(&audio(0, 8)).unwrap();
    good.send(&commit(1)).unwrap();
    assert_eq!(final_of(&until_terminal(&good)), (8, "7 0".into()));
    drop((bad, good));
    assert_clean(&c, &live);
}

#[test]
fn invalid_configuration_is_rejected() {
    let bad = |f: fn(&mut CoordinatorConfig)| {
        let mut cfg = config();
        f(&mut cfg);
        Coordinator::new(MockBackend::new(), text, cfg).err()
    };
    assert!(bad(|c| c.workers = 0).is_some());
    assert!(bad(|c| c.max_streams = 0).is_some());
    assert!(bad(|c| c.draft_cadence_samples = 0).is_some());
    assert!(
        bad(|c| c.session = SessionConfig {
            max_tokens: 0,
            ..SessionConfig::default()
        })
        .is_some()
    );
    assert!(bad(|c| c.limits.max_frame_samples = 0).is_some());
}

#[test]
fn stream_ids_are_unique_per_open() {
    let (mock, _gate) = gated(vec![], 0);
    let (c, _live) = coordinator(mock, config());
    let ids: Vec<StreamId> = (0..4)
        .map(|_| c.open_stream().unwrap())
        .map(|s| s.stream_id())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_by_key(|id| id.0);
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len());
}

/// Poll until the registry drains or fail after `WAIT`.
fn wait_open_streams(c: &Coord, n: usize) {
    let deadline = std::time::Instant::now() + WAIT;
    while c.stats().open_streams != n {
        assert!(
            std::time::Instant::now() < deadline,
            "registry never drained"
        );
        std::thread::yield_now();
    }
}

#[test]
fn queued_cancellations_cannot_accumulate_behind_a_busy_worker() {
    let (mock, gate) = gated(vec![(8, vec![1, 2, 3, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            max_streams: 2,
            draft_cadence_samples: 8,
            ..config()
        },
    );
    // Stream A parks the only worker inside a draft.
    let a = start(&c);
    a.send(&audio(0, 8)).unwrap();
    gate.wait_entered();
    let base = c.stats();
    assert_eq!(base.open_streams, 1);
    assert_eq!((base.pending_commits, base.pending_drafts), (0, 0));

    for round in 0..50 {
        let b = start(&c);
        b.send(&audio(0, 8)).unwrap();
        // Queue a commit, then cancel it while the worker is busy elsewhere.
        b.send(&commit(1)).unwrap();
        assert_eq!(c.stats().pending_commits, 1, "round {round}");
        b.send(&ClientMessage::Cancel).unwrap();
        let s = c.stats();
        assert_eq!(
            (s.pending_commits, s.pending_drafts, s.open_streams),
            (0, 0, 1),
            "round {round}"
        );
        // A queued draft that is cancelled directly is removed too.
        let d = start(&c);
        d.send(&audio(0, 8)).unwrap();
        assert_eq!(c.stats().pending_drafts, 1, "round {round}");
        drop(d);
        let s = c.stats();
        assert_eq!(
            (s.pending_commits, s.pending_drafts, s.open_streams),
            (0, 0, 1),
            "round {round}"
        );
    }
    // Admission was never exceeded and counters are back at baseline.
    assert!(c.open_stream().is_ok());
    gate.release();
    drop(a);
    wait_open_streams(&c, 1);
    assert_eq!(c.stats().active_commits, 0);
    c.shutdown();
    assert_eq!(live.load(Ordering::SeqCst), 1);
}

#[test]
fn running_cancellation_holds_its_slot_until_the_worker_retires() {
    let (mock, gate) = gated(vec![(8, vec![1, 2, 3, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            max_streams: 1,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 8)).unwrap();
    s.send(&commit(1)).unwrap();
    gate.wait_entered();
    s.send(&ClientMessage::Cancel).unwrap();
    // The commit cannot be interrupted, so the slot stays occupied.
    assert_eq!(c.stats().open_streams, 1);
    assert_eq!(c.stats().active_commits, 1);
    assert_eq!(
        c.open_stream().err(),
        Some(CoordinatorError::TooManyStreams { max: 1 })
    );
    gate.release();
    wait_open_streams(&c, 0);
    let t = start(&c);
    drop(t);
    drop(s);
    assert_clean(&c, &live);
}

#[test]
fn scheduler_metadata_does_not_grow_per_completed_stream() {
    let (c, live) = coordinator(
        MockBackend::with_transcripts([(8, vec![1, 2, MOCK_EOS_TOKEN])]),
        CoordinatorConfig {
            draft_cadence_samples: 8,
            ..config()
        },
    );
    for _ in 0..40 {
        let s = start(&c);
        s.send(&audio(0, 8)).unwrap();
        s.send(&commit(1)).unwrap();
        let events = until_terminal(&s);
        assert!(matches!(events.last(), Some(ServerEvent::Final { .. })));
    }
    for _ in 0..40 {
        let s = start(&c);
        s.send(&audio(0, 8)).unwrap();
        s.send(&ClientMessage::Cancel).unwrap();
    }
    wait_open_streams(&c, 0);
    // The registry retires the stream before the worker drops its scheduler permit.
    // Join the workers before asserting final scheduler counters.
    c.shutdown();
    let s = c.stats();
    assert_eq!(s.tracked_streams, 0);
    assert_eq!(
        (
            s.pending_commits,
            s.pending_drafts,
            s.active_commits,
            s.active_drafts
        ),
        (0, 0, 0, 0)
    );
    assert_clean(&c, &live);
}

#[test]
fn cancel_discards_unread_partial_and_late_sends_are_inert() {
    let (mock, gate) = gated(vec![(8, vec![1, 2, 3, MOCK_EOS_TOKEN])], 1);
    let (c, live) = coordinator(
        mock,
        CoordinatorConfig {
            draft_cadence_samples: 8,
            ..config()
        },
    );
    let s = start(&c);
    s.send(&audio(0, 8)).unwrap();
    gate.wait_entered();
    gate.release();
    // Wait until the draft has published a partial, without reading it.
    let deadline = std::time::Instant::now() + WAIT;
    while c.stats().active_drafts != 0 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    s.send(&ClientMessage::Cancel).unwrap();
    let mut events = Vec::new();
    while let Some(e) = s.try_recv_event() {
        events.push(e);
    }
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ServerEvent::Partial { .. })),
        "stale partial survived cancel: {events:?}"
    );
    assert_eq!(events.last(), Some(&ServerEvent::Cancelled));
    // Commit and cancel after the terminal event do not schedule or emit.
    assert_eq!(
        s.send(&commit(1)).unwrap_err().code,
        ErrorCode::InvalidMessageOrder
    );
    s.send(&ClientMessage::Cancel).unwrap();
    assert_eq!(c.stats().pending_commits, 0);
    assert!(s.try_recv_event().is_none());
    drop(s);
    assert_clean(&c, &live);
}
