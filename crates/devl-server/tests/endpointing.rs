//! Deterministic tests for server-side endpointing hooks.
//!
//! Policies count samples, never time. Where a draft must be observed before
//! the next frame is sent, the test waits for the partial instead of sleeping.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{SchedulerConfig, TokenId};
use devl_server::*;

const WAIT: Duration = Duration::from_secs(20);
const FRAME: usize = 160;

type Coord = Coordinator<MockBackend>;
type Session = StreamSession<MockBackend>;

fn text(tokens: &[TokenId]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn config() -> CoordinatorConfig {
    CoordinatorConfig {
        workers: 1,
        scheduler: SchedulerConfig {
            max_concurrent_drafts: 1,
        },
        // Cadence drafts stay out of the way unless a test lowers this.
        draft_cadence_samples: 1_000_000,
        ..CoordinatorConfig::default()
    }
}

fn mock() -> MockBackend {
    MockBackend::with_transcripts([
        (FRAME * 3, vec![21, 22, MOCK_EOS_TOKEN]),
        (FRAME * 4, vec![21, 22, 23, MOCK_EOS_TOKEN]),
    ])
}

fn with_policy(backend: MockBackend, cfg: CoordinatorConfig, f: EndpointPolicyFactory) -> Coord {
    Coordinator::with_endpointing(backend, text, cfg, f).unwrap()
}

fn silence_policy(end: u64, checkpoint: Option<u64>, max: Option<u64>) -> EndpointPolicyFactory {
    SilenceEndpointPolicy::factory(SilenceEndpointConfig {
        speech_threshold: 0.1,
        checkpoint_silence_samples: checkpoint,
        end_silence_samples: end,
        max_turn_samples: max,
    })
    .unwrap()
}

fn frame(seq: u64, value: i16) -> ClientMessage {
    ClientMessage::Audio(AudioFrame::from_s16(seq, &vec![value; FRAME]))
}
fn speech(seq: u64) -> ClientMessage {
    frame(seq, 8000)
}
fn quiet(seq: u64) -> ClientMessage {
    frame(seq, 0)
}

fn start(c: &Coord) -> Session {
    let s = c.open_stream().unwrap();
    s.send(&ClientMessage::Start {
        version: PROTOCOL_VERSION,
        format: AudioFormat::mono_16k_s16(),
    })
    .unwrap();
    assert!(matches!(next(&s), ServerEvent::Started { .. }));
    s
}

fn next(s: &Session) -> ServerEvent {
    s.recv_event_timeout(WAIT)
        .expect("timed out waiting for event")
}

fn is_terminal(e: &ServerEvent) -> bool {
    matches!(
        e,
        ServerEvent::Final { .. } | ServerEvent::Cancelled | ServerEvent::Error(_)
    )
}

fn until_terminal(s: &Session) -> Vec<ServerEvent> {
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

fn wait_partial(s: &Session) -> (u64, String) {
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

/// Offline reference over frames given as s16 values, one per frame.
fn reference(mock: &MockBackend, frames: &[i16]) -> String {
    let pcm: Vec<f32> = frames
        .iter()
        .flat_map(|v| vec![f32::from(*v) / 32768.0; FRAME])
        .collect();
    text(&mock.reference_offline_decode(&pcm, 448))
}

fn assert_no_more(s: &Session) {
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(s.try_recv_event(), None, "unexpected event after terminal");
}

fn assert_clean(c: &Coord) {
    c.shutdown();
    let st = c.stats();
    assert_eq!(
        (st.open_streams, st.pending_commits, st.pending_drafts),
        (0, 0, 0)
    );
    assert_eq!((st.active_commits, st.active_drafts), (0, 0));
}

/// Policy driven by a fixed per-frame script; `Continue` once exhausted.
struct Script {
    decisions: Vec<EndpointDecision>,
    seen: usize,
    dropped: Arc<AtomicUsize>,
}

impl EndpointPolicy for Script {
    fn on_audio(&mut self, _frame: &[f32], _total: u64) -> EndpointDecision {
        let d = self
            .decisions
            .get(self.seen)
            .copied()
            .unwrap_or(EndpointDecision::Continue);
        self.seen += 1;
        d
    }
}

impl Drop for Script {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn scripted(decisions: Vec<EndpointDecision>) -> (EndpointPolicyFactory, Arc<AtomicUsize>) {
    let dropped = Arc::new(AtomicUsize::new(0));
    let d = Arc::clone(&dropped);
    let f: EndpointPolicyFactory = Arc::new(move |_| {
        Box::new(Script {
            decisions: decisions.clone(),
            seen: 0,
            dropped: Arc::clone(&d),
        })
    });
    (f, dropped)
}

#[test]
fn default_keeps_manual_commit_and_draft_cadence() {
    let backend = mock();
    let c = Coordinator::new(
        backend.clone(),
        text,
        CoordinatorConfig {
            draft_cadence_samples: FRAME as u64 * 3,
            ..config()
        },
    )
    .unwrap();
    let s = start(&c);
    for i in 0..3 {
        s.send(&speech(i)).unwrap();
    }
    assert_eq!(wait_partial(&s), (FRAME as u64 * 3, "21 22 0".into()));
    // Long silence never finalizes without a policy.
    for i in 3..8 {
        s.send(&quiet(i)).unwrap();
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(s.phase(), StreamPhase::Streaming);
    s.send(&ClientMessage::Commit { frames: 8 }).unwrap();
    let events = until_terminal(&s);
    assert_eq!(final_of(&events).0, FRAME as u64 * 8);
    drop(s);
    assert_clean(&c);
}

#[test]
fn scripted_speech_and_silence_checkpoints_then_finalizes_in_order() {
    let backend = mock();
    // 3 speech frames, then 160 silence checkpoints, 320 ends the turn.
    let c = with_policy(
        backend.clone(),
        config(),
        silence_policy(FRAME as u64 * 2, Some(FRAME as u64), None),
    );
    let s = start(&c);
    s.send(&speech(0)).unwrap();
    s.send(&speech(1)).unwrap();
    s.send(&speech(2)).unwrap();
    s.send(&quiet(3)).unwrap(); // checkpoint requested at 4 frames
    assert_eq!(wait_partial(&s), (FRAME as u64 * 4, "21 22 23 0".into()));
    s.send(&quiet(4)).unwrap(); // end of turn at 5 frames

    let mut events = Vec::new();
    events.extend(until_terminal(&s));
    let expected = reference(&backend, &[8000, 8000, 8000, 0, 0]);
    assert_eq!(final_of(&events), (FRAME as u64 * 5, expected));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ServerEvent::Partial { .. })),
        "no partial after the end of turn: {events:?}"
    );
    assert_eq!(s.phase(), StreamPhase::Finalized);

    // Late input from a client that has not seen the final yet is inert.
    s.send(&quiet(5)).unwrap();
    s.send(&ClientMessage::Commit { frames: 5 }).unwrap();
    s.send(&ClientMessage::Cancel).unwrap();
    assert_no_more(&s);
    drop(s);
    assert_clean(&c);
}

#[test]
fn silence_only_does_not_finalize_and_manual_commit_still_works() {
    let c = with_policy(
        mock(),
        config(),
        silence_policy(FRAME as u64, Some(FRAME as u64), None),
    );
    let s = start(&c);
    for i in 0..10 {
        s.send(&quiet(i)).unwrap();
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(s.phase(), StreamPhase::Streaming);
    assert!(
        !std::iter::from_fn(|| s.try_recv_event())
            .any(|e| is_terminal(&e) || matches!(e, ServerEvent::Partial { .. })),
        "silence must not draft or finalize"
    );
    s.send(&ClientMessage::Commit { frames: 10 }).unwrap();
    let events = until_terminal(&s);
    assert_eq!(final_of(&events).0, FRAME as u64 * 10);
    drop(s);
    assert_clean(&c);
}

#[test]
fn long_turn_is_ended_by_max_turn_samples() {
    let backend = mock();
    let c = with_policy(
        backend.clone(),
        config(),
        silence_policy(FRAME as u64, None, Some(FRAME as u64 * 4)),
    );
    let s = start(&c);
    // Continuous speech never reaches the silence threshold.
    for i in 0..4 {
        s.send(&speech(i)).unwrap();
    }
    let events = until_terminal(&s);
    assert_eq!(
        final_of(&events),
        (FRAME as u64 * 4, reference(&backend, &[8000; 4]))
    );
    drop(s);
    assert_clean(&c);
}

#[test]
fn disconnect_and_cancel_drop_the_policy_without_events() {
    let (f, dropped) = scripted(vec![]);
    let c = with_policy(mock(), config(), f);
    let s = start(&c);
    s.send(&speech(0)).unwrap();
    drop(s);
    // Disconnect releases the stream, so the slot frees and nobody is told.
    let t = std::time::Instant::now();
    while c.stats().open_streams != 0 {
        assert!(t.elapsed() < WAIT);
        std::thread::yield_now();
    }

    let s = start(&c);
    s.send(&speech(0)).unwrap();
    s.send(&ClientMessage::Cancel).unwrap();
    assert_eq!(until_terminal(&s).last(), Some(&ServerEvent::Cancelled));
    drop(s);
    assert_clean(&c);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        2,
        "each stream owns and drops its policy"
    );
}

#[test]
fn policy_state_is_per_stream() {
    let c = with_policy(
        mock(),
        config(),
        silence_policy(FRAME as u64 * 2, None, None),
    );
    let a = start(&c);
    let b = start(&c);
    a.send(&speech(0)).unwrap();
    a.send(&quiet(1)).unwrap();
    // b has heard no speech: its silence cannot end the turn.
    for i in 0..4 {
        b.send(&quiet(i)).unwrap();
    }
    a.send(&quiet(2)).unwrap();
    assert_eq!(final_of(&until_terminal(&a)).0, FRAME as u64 * 3);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(b.phase(), StreamPhase::Streaming);
    drop((a, b));
    assert_clean(&c);
}

#[test]
fn manual_commit_after_auto_commit_yields_exactly_one_final() {
    use EndpointDecision::*;
    let (f, _) = scripted(vec![Continue, EndTurn]);
    let c = with_policy(mock(), config(), f);
    let s = start(&c);
    s.send(&speech(0)).unwrap();
    s.send(&speech(1)).unwrap(); // auto end of turn
    // The client's manual commit crosses the server's final.
    s.send(&ClientMessage::Commit { frames: 2 }).unwrap();
    let events = until_terminal(&s);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, ServerEvent::Final { .. }))
            .count(),
        1
    );
    assert_eq!(final_of(&events).0, FRAME as u64 * 2);
    s.send(&ClientMessage::Commit { frames: 2 }).unwrap();
    assert_no_more(&s);
    drop(s);
    assert_clean(&c);
}

#[test]
fn manual_commit_before_policy_decision_wins_and_policy_is_not_consulted() {
    use EndpointDecision::*;
    let (f, _) = scripted(vec![Continue, EndTurn]);
    let c = with_policy(mock(), config(), f);
    let s = start(&c);
    s.send(&speech(0)).unwrap();
    s.send(&ClientMessage::Commit { frames: 1 }).unwrap();
    // Audio after a manual commit is still a contract error; no auto commit.
    assert_eq!(
        s.send(&speech(1)).unwrap_err().code,
        ErrorCode::InvalidMessageOrder
    );
    let events = until_terminal(&s);
    // The manual final may land before or after the rejected audio; either
    // way there is one terminal event and the policy never committed.
    assert_eq!(events.iter().filter(|e| is_terminal(e)).count(), 1);
    assert_no_more(&s);
    drop(s);
    assert_clean(&c);
}

#[test]
fn concurrent_manual_and_auto_triggers_finalize_at_most_once() {
    use EndpointDecision::*;
    for _ in 0..100 {
        let (f, _) = scripted(vec![Continue, EndTurn]);
        let c = with_policy(mock(), config(), f);
        let s = Arc::new(start(&c));
        s.send(&speech(0)).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let auto = {
            let (s, b) = (Arc::clone(&s), Arc::clone(&barrier));
            std::thread::spawn(move || {
                b.wait();
                let _ = s.send(&speech(1));
            })
        };
        let manual = {
            let (s, b) = (Arc::clone(&s), Arc::clone(&barrier));
            std::thread::spawn(move || {
                b.wait();
                let _ = s.send(&ClientMessage::Commit { frames: 1 });
            })
        };
        auto.join().unwrap();
        manual.join().unwrap();
        // Whichever trigger loses is an error or a no-op, never a second final.
        let events = until_terminal(&s);
        let finals = events
            .iter()
            .filter(|e| matches!(e, ServerEvent::Final { .. }))
            .count();
        assert!(finals <= 1, "{events:?}");
        assert_eq!(events.iter().filter(|e| is_terminal(e)).count(), 1);
        assert_eq!(s.try_recv_event(), None);
        drop(s);
        assert_clean(&c);
    }
}

#[test]
fn checkpoint_draft_is_suppressed_when_the_turn_ends_while_it_runs() {
    use EndpointDecision::*;
    let (entered_tx, entered) = channel::<()>();
    let (release, release_rx): (Sender<()>, Receiver<()>) = channel();
    let (entered_tx, release_rx) = (Mutex::new(entered_tx), Mutex::new(release_rx));
    let armed = Mutex::new(true);
    let backend = MockBackend::with_transcripts([(FRAME, vec![41, 42, 43, MOCK_EOS_TOKEN])])
        .with_step_hook(move |_| {
            let mut a = armed.lock().unwrap();
            if *a {
                *a = false;
                let _ = entered_tx.lock().unwrap().send(());
                let _ = release_rx.lock().unwrap().recv_timeout(WAIT);
            }
        });
    let (f, _) = scripted(vec![Checkpoint, EndTurn]);
    let c = with_policy(backend, config(), f);
    let s = start(&c);
    s.send(&speech(0)).unwrap();
    entered.recv_timeout(WAIT).expect("draft never started");
    s.send(&speech(1)).unwrap(); // end of turn while the draft runs
    release.send(()).unwrap();
    let events = until_terminal(&s);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ServerEvent::Partial { .. })),
        "stale partial leaked: {events:?}"
    );
    assert_eq!(final_of(&events).0, FRAME as u64 * 2);
    drop(s);
    assert_clean(&c);
}

#[test]
fn end_turn_wins_over_checkpoint_on_the_same_frame() {
    // A policy that asks for both can only express one decision; EndTurn is
    // applied and no draft is queued behind the final.
    use EndpointDecision::*;
    let (f, _) = scripted(vec![EndTurn]);
    let c = with_policy(mock(), config(), f);
    let s = start(&c);
    s.send(&speech(0)).unwrap();
    let events = until_terminal(&s);
    assert_eq!(final_of(&events).0, FRAME as u64);
    assert_eq!(c.stats().pending_drafts, 0);
    drop(s);
    assert_clean(&c);
}

#[test]
fn panicking_policy_fails_only_its_stream() {
    struct Boom;
    impl EndpointPolicy for Boom {
        fn on_audio(&mut self, _: &[f32], _: u64) -> EndpointDecision {
            panic!("policy bug");
        }
    }
    let f: EndpointPolicyFactory = Arc::new(|_| Box::new(Boom));
    let c = with_policy(mock(), config(), f);
    let s = start(&c);
    assert_eq!(s.send(&speech(0)).unwrap_err().code, ErrorCode::Internal);
    let events = until_terminal(&s);
    assert!(matches!(events.last(), Some(ServerEvent::Error(e)) if e.code == ErrorCode::Internal));
    drop(s);
    assert_clean(&c);
}
