//! Integration tests verifying exact offline greedy equivalence across all
//! verification paths, session lifecycle & error recovery, and scheduler
//! commit-over-draft preemption.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{
    DevlError, DevlSession, JobPriority, PriorityScheduler, SchedulerConfig, SessionConfig,
    SessionState, StreamId, TokenId,
};

fn samples(len: usize) -> Vec<f32> {
    vec![0.25_f32; len]
}

#[test]
fn exact_equivalence_full_draft_match_with_eos() {
    // Audio stays consistent between draft and commit -> 100% draft match ending in EOS.
    let backend = MockBackend::with_transcripts([
        (1600, vec![10, 20, 30, 40, MOCK_EOS_TOKEN]),
        (2400, vec![10, 20, 30, 40, MOCK_EOS_TOKEN]),
    ]);
    let config = SessionConfig {
        max_tokens: 64,
        enable_checkpoint_promotion: false,
    };
    let mut session = DevlSession::new(&backend, config).unwrap();

    session.push_audio(&samples(1600)).unwrap();
    let draft = session.run_draft().unwrap();
    assert_eq!(draft.tokens, vec![10, 20, 30, 40, MOCK_EOS_TOKEN]);
    assert_eq!(session.state(), SessionState::Drafting);

    // Append trailing silence/audio that keeps the exact same offline transcript.
    session.push_audio(&samples(800)).unwrap();
    let final_res = session.commit().unwrap();

    let expected = backend.reference_offline_decode(&samples(2400), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.accepted_prefix_len, 5);
    assert_eq!(final_res.draft_len, 5);
    assert_eq!(final_res.decode_steps, 0);
    assert!(!final_res.was_promoted);
    assert_eq!(session.state(), SessionState::Finalized);
    // Second encode_audio call should have received `prior_encoding`.
    assert!(
        backend
            .metrics()
            .prior_encoding_reuses
            .load(Ordering::Relaxed)
            >= 1
    );
}

#[test]
fn exact_equivalence_full_draft_match_without_eos_resumes_decode() {
    // Draft matches 100% of its tokens (non-EOS prefix) and utterance continues.
    let backend = MockBackend::with_transcripts([(2400, vec![10, 20, 30, 40, 50, MOCK_EOS_TOKEN])]);
    let config = SessionConfig {
        max_tokens: 64,
        enable_checkpoint_promotion: false,
    };
    let mut session = DevlSession::new(&backend, config).unwrap();
    session.push_audio(&samples(2400)).unwrap();
    session.seed_draft_for_test(vec![10, 20, 30], 1600);

    let final_res = session.commit().unwrap();
    let expected = backend.reference_offline_decode(&samples(2400), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.accepted_prefix_len, 3);
    assert_eq!(final_res.draft_len, 3);
    // Needs decode_step(30)->40, decode_step(40)->50, decode_step(50)->EOS.
    assert_eq!(final_res.decode_steps, 3);
}

#[test]
fn exact_equivalence_partial_prefix_mismatch_with_kv_rollback() {
    // Partial transcript matches first 2 tokens, then diverges when full audio arrives.
    let backend = MockBackend::with_transcripts([
        (1600, vec![10, 20, 99, 100, MOCK_EOS_TOKEN]),
        (3200, vec![10, 20, 31, 41, 51, MOCK_EOS_TOKEN]),
    ]);
    let config = SessionConfig::default();
    let mut session = DevlSession::new(&backend, config).unwrap();

    session.push_audio(&samples(1600)).unwrap();
    let draft = session.run_draft().unwrap();
    assert_eq!(draft.tokens, vec![10, 20, 99, 100, MOCK_EOS_TOKEN]);

    session.push_audio(&samples(1600)).unwrap();
    let final_res = session.commit().unwrap();

    let expected = backend.reference_offline_decode(&samples(3200), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.accepted_prefix_len, 2);
    assert_eq!(final_res.draft_len, 5);
    // Verified prefix is [10, 20, 31]; residual steps decode 41, 51, EOS.
    assert_eq!(final_res.decode_steps, 3);
}

#[test]
fn exact_equivalence_partial_mismatch_terminating_immediately_at_eos() {
    // Draft predicted extra words, but true offline output ends earlier with EOS.
    let backend = MockBackend::with_transcripts([
        (1600, vec![10, 20, 30, 40, MOCK_EOS_TOKEN]),
        (2400, vec![10, 20, MOCK_EOS_TOKEN]),
    ]);
    let config = SessionConfig::default();
    let mut session = DevlSession::new(&backend, config).unwrap();

    session.push_audio(&samples(1600)).unwrap();
    session.run_draft().unwrap();
    session.push_audio(&samples(800)).unwrap();

    let final_res = session.commit().unwrap();
    let expected = backend.reference_offline_decode(&samples(2400), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.accepted_prefix_len, 2);
    // Correction token at index 2 is EOS -> 0 residual decode steps.
    assert_eq!(final_res.decode_steps, 0);
}

#[test]
fn exact_equivalence_zero_match_draft() {
    // Draft diverges immediately at token 0.
    let backend = MockBackend::with_transcripts([
        (1600, vec![77, 88, 99, MOCK_EOS_TOKEN]),
        (3200, vec![11, 22, 33, MOCK_EOS_TOKEN]),
    ]);
    let config = SessionConfig::default();
    let mut session = DevlSession::new(&backend, config).unwrap();

    session.push_audio(&samples(1600)).unwrap();
    session.run_draft().unwrap();
    session.push_audio(&samples(1600)).unwrap();

    let final_res = session.commit().unwrap();
    let expected = backend.reference_offline_decode(&samples(3200), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.accepted_prefix_len, 0);
    assert_eq!(final_res.draft_len, 4);
    // Verified prefix is [11]; residual steps decode 22, 33, EOS.
    assert_eq!(final_res.decode_steps, 3);
}

#[test]
fn exact_equivalence_empty_draft_and_empty_utterance() {
    // Commit directly from Idle without ever running a checkpoint draft.
    let backend = MockBackend::with_transcripts([
        (2000, vec![5, 6, 7, MOCK_EOS_TOKEN]),
        (0, vec![MOCK_EOS_TOKEN]),
    ]);
    let config = SessionConfig::default();
    let mut session = DevlSession::new(&backend, config).unwrap();

    session.push_audio(&samples(2000)).unwrap();
    assert_eq!(session.state(), SessionState::Idle);
    let final_res = session.commit().unwrap();

    let expected = backend.reference_offline_decode(&samples(2000), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.accepted_prefix_len, 0);
    assert_eq!(final_res.draft_len, 0);
    assert_eq!(final_res.decode_steps, 3);

    // Reset and commit an empty (EOS-only) utterance with no draft.
    session.reset();
    assert_eq!(session.state(), SessionState::Idle);
    let empty_res = session.commit().unwrap();
    assert_eq!(empty_res.tokens, vec![MOCK_EOS_TOKEN]);
    assert_eq!(empty_res.decode_steps, 0);
}

#[test]
fn checkpoint_promotion_skips_reverification_when_audio_unchanged() {
    let backend = MockBackend::with_transcripts([(1600, vec![1, 2, 3, MOCK_EOS_TOKEN])]);
    let config = SessionConfig {
        max_tokens: 32,
        enable_checkpoint_promotion: true,
    };
    let mut session = DevlSession::new(&backend, config).unwrap();

    session.push_audio(&samples(1600)).unwrap();
    session.run_draft().unwrap();
    let verifies_before_commit = backend.metrics().verify_calls.load(Ordering::Relaxed);

    let final_res = session.commit().unwrap();
    assert!(final_res.was_promoted);
    assert_eq!(
        backend.metrics().verify_calls.load(Ordering::Relaxed),
        verifies_before_commit
    );
    let expected = backend.reference_offline_decode(&samples(1600), config.max_tokens);
    assert_eq!(final_res.tokens, expected);
}

#[test]
fn config_validation_and_draft_max_tokens_capping() {
    let backend = MockBackend::with_transcripts([(1600, vec![1, 2, 3, 4, 5, MOCK_EOS_TOKEN])]);

    let bad_cfg = SessionConfig {
        max_tokens: 0,
        enable_checkpoint_promotion: true,
    };
    assert!(matches!(
        DevlSession::new(&backend, bad_cfg),
        Err(DevlError::InvalidConfig(_))
    ));

    let small_cfg = SessionConfig {
        max_tokens: 3,
        enable_checkpoint_promotion: false,
    };
    let mut session = DevlSession::new(&backend, small_cfg).unwrap();
    session.push_audio(&samples(1600)).unwrap();
    // Seed a draft longer than max_tokens; commit must cap draft to 3 tokens.
    session.seed_draft_for_test(vec![1, 2, 3, 4, 5, MOCK_EOS_TOKEN], 800);

    let final_res = session.commit().unwrap();
    let expected = backend.reference_offline_decode(&samples(1600), small_cfg.max_tokens);
    assert_eq!(final_res.tokens, expected);
    assert_eq!(final_res.tokens.len(), 3);
    assert_eq!(final_res.draft_len, 3);
}

#[test]
fn error_recovery_draft_non_fatal_and_commit_enters_failed_until_reset() {
    let backend = MockBackend::with_transcripts([(1600, vec![42, 43, MOCK_EOS_TOKEN])]);
    let mut session = DevlSession::new(&backend, SessionConfig::default()).unwrap();

    session.push_audio(&samples(1600)).unwrap();

    // Speculative draft error is non-fatal: session remains Idle and commit succeeds.
    backend.fail_next_encode();
    assert!(matches!(session.run_draft(), Err(DevlError::Backend(_))));
    assert_eq!(session.state(), SessionState::Idle);

    // Commit failure transitions session to Failed.
    backend.fail_next_verify();
    assert!(matches!(session.commit(), Err(DevlError::Backend(_))));
    assert_eq!(session.state(), SessionState::Failed);

    // Operations are blocked in Failed state.
    assert!(matches!(
        session.push_audio(&samples(160)),
        Err(DevlError::InvalidState {
            state: SessionState::Failed,
            ..
        })
    ));
    assert!(matches!(
        session.commit(),
        Err(DevlError::InvalidState {
            state: SessionState::Failed,
            ..
        })
    ));

    // Reset from Failed recovers cleanly to Idle.
    session.reset();
    assert_eq!(session.state(), SessionState::Idle);
    session.push_audio(&samples(1600)).unwrap();
    let res = session.commit().unwrap();
    assert_eq!(res.tokens, vec![42, 43, MOCK_EOS_TOKEN]);
    assert_eq!(session.state(), SessionState::Finalized);
}

#[test]
fn scheduler_commits_strictly_preempt_queued_drafts_and_coalesce() {
    let scheduler = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: 2,
    });

    // Queue multiple drafts first, including two drafts for StreamId(1) (coalesced).
    scheduler.submit_draft(StreamId(1), "draft-1-old");
    scheduler.submit_draft(StreamId(1), "draft-1-new");
    scheduler.submit_draft(StreamId(2), "draft-2");
    scheduler.submit_draft(StreamId(3), "draft-3");
    assert_eq!(scheduler.pending_drafts(), 3);

    // Submit commits for StreamId(2) (evicts draft-2) and StreamId(4).
    scheduler.submit_commit(StreamId(2), "commit-2");
    scheduler.submit_commit(StreamId(4), "commit-4");
    assert_eq!(scheduler.pending_commits(), 2);
    assert_eq!(scheduler.pending_drafts(), 2);

    // Commits must dequeue first, in FIFO order.
    let job1 = scheduler.try_next_job().unwrap();
    assert_eq!(job1.priority(), JobPriority::Commit);
    assert_eq!(*job1.payload(), "commit-2");

    let job2 = scheduler.try_next_job().unwrap();
    assert_eq!(job2.priority(), JobPriority::Commit);
    assert_eq!(*job2.payload(), "commit-4");

    // Now drafts dequeue, respecting max_concurrent_drafts = 2 and coalesced payload.
    let job3 = scheduler.try_next_job().unwrap();
    assert_eq!(job3.priority(), JobPriority::Draft);
    assert_eq!(*job3.payload(), "draft-1-new");

    let job4 = scheduler.try_next_job().unwrap();
    assert_eq!(job4.priority(), JobPriority::Draft);
    assert_eq!(*job4.payload(), "draft-3");

    // Enqueue another draft while 2 drafts are active -> blocked until permit drops.
    scheduler.submit_draft(StreamId(5), "draft-5");
    assert!(scheduler.try_next_job().is_none());

    // Even while draft slots are full, a new commit still runs immediately.
    scheduler.submit_commit(StreamId(6), "commit-6");
    let job5 = scheduler.try_next_job().unwrap();
    assert_eq!(job5.priority(), JobPriority::Commit);
    assert_eq!(*job5.payload(), "commit-6");

    // Dropping one active draft permit frees the draft slot.
    drop(job3);
    let job6 = scheduler.try_next_job().unwrap();
    assert_eq!(job6.priority(), JobPriority::Draft);
    assert_eq!(*job6.payload(), "draft-5");
    drop((job1, job2, job4, job5, job6));
    assert_eq!(scheduler.active_commits(), 0);
    assert_eq!(scheduler.active_drafts(), 0);
}

#[test]
fn scheduler_step_level_preemption_yields_in_flight_draft_for_commit() {
    // End-to-end test: a single worker starts a long draft; after step 0, a
    // commit arrives on another stream. The draft must yield immediately after
    // that step, letting the commit execute before queued background drafts.
    let scheduler = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: 4,
    });
    let commit_submitted = Arc::new(AtomicBool::new(false));
    let draft_started_barrier = Arc::new(Barrier::new(2));
    let commit_ready_barrier = Arc::new(Barrier::new(2));

    let long_seq: Vec<TokenId> = (1..=20).chain(std::iter::once(MOCK_EOS_TOKEN)).collect();
    let commit_seq = vec![100, 101, MOCK_EOS_TOKEN];

    let b_started = Arc::clone(&draft_started_barrier);
    let b_ready = Arc::clone(&commit_ready_barrier);
    let hook_fired = Arc::new(AtomicBool::new(false));
    let hook_fired_clone = Arc::clone(&hook_fired);

    let backend = Arc::new(
        MockBackend::with_transcripts([(1600, long_seq.clone()), (800, commit_seq.clone())])
            .with_step_hook(move |_step_idx| {
                // Only synchronize on the very first decode_step of the draft.
                if !hook_fired_clone.swap(true, Ordering::SeqCst) {
                    b_started.wait();
                    b_ready.wait();
                }
            }),
    );

    enum Task {
        Draft(DevlSession<Arc<MockBackend>>),
        Commit(DevlSession<Arc<MockBackend>>),
    }

    let mut draft_session_1 =
        DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
    draft_session_1.push_audio(&samples(1600)).unwrap();

    let mut draft_session_2 =
        DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
    draft_session_2.push_audio(&samples(1600)).unwrap();

    scheduler.submit_draft(StreamId(1), Task::Draft(draft_session_1));
    scheduler.submit_draft(StreamId(2), Task::Draft(draft_session_2));

    let worker_scheduler = scheduler.clone();
    let execution_order = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let worker_order = Arc::clone(&execution_order);
    let draft_preempted = Arc::new(AtomicBool::new(false));
    let worker_preempted = Arc::clone(&draft_preempted);
    let draft_steps_taken = Arc::new(AtomicUsize::new(0));
    let worker_draft_steps = Arc::clone(&draft_steps_taken);

    let worker = thread::spawn(move || {
        while let Some(job) = worker_scheduler.next_job() {
            let stream_id = job.stream_id();
            let seq = job.seq();
            let (task, _permit) = job.into_parts();
            match task {
                Task::Draft(mut session) => {
                    let outcome = session
                        .run_draft_cancellable(|| {
                            worker_scheduler.should_preempt_draft(stream_id, seq)
                        })
                        .unwrap();
                    if stream_id == StreamId(1) {
                        worker_preempted.store(outcome.is_preempted(), Ordering::SeqCst);
                        worker_draft_steps.store(outcome.result().decode_steps, Ordering::SeqCst);
                        worker_order.lock().unwrap().push("draft-1");
                    } else {
                        worker_order.lock().unwrap().push("draft-2");
                    }
                }
                Task::Commit(mut session) => {
                    let res = session.commit().unwrap();
                    assert_eq!(res.tokens, vec![100, 101, MOCK_EOS_TOKEN]);
                    worker_order.lock().unwrap().push("commit-3");
                }
            }
        }
    });

    // Wait until draft-1 executes its first decode_step, then enqueue commit-3.
    draft_started_barrier.wait();
    let mut commit_session =
        DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
    commit_session.push_audio(&samples(800)).unwrap();
    scheduler.submit_commit(StreamId(3), Task::Commit(commit_session));
    commit_submitted.store(true, Ordering::SeqCst);
    commit_ready_barrier.wait();

    scheduler.shutdown();
    worker.join().unwrap();

    assert!(commit_submitted.load(Ordering::SeqCst));
    assert!(
        draft_preempted.load(Ordering::SeqCst),
        "in-flight draft should have been cooperatively preempted"
    );
    assert_eq!(
        draft_steps_taken.load(Ordering::SeqCst),
        1,
        "draft must yield immediately after the single running decode step"
    );
    assert_eq!(
        *execution_order.lock().unwrap(),
        vec!["draft-1", "commit-3", "draft-2"],
        "commit must run immediately after preempted draft-1 and ahead of queued draft-2"
    );
}
