//! Multi-threaded stress tests for [`PriorityScheduler`].
//!
//! Validates high-concurrency multi-stream execution (>= 32 streams), bounded
//! `max_concurrent_drafts`, strict FIFO commit ordering ahead of drafts, rapid
//! per-stream draft coalescing and commit eviction under contention, step-level
//! cooperative preemption under multi-stream load, and panic-safe RAII
//! `JobPermit` accounting with zero deadlocks and zero sleeps.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{
    DevlSession, JobPriority, PriorityScheduler, SchedulerConfig, SessionConfig, SessionState,
    StreamId, TokenId,
};

fn samples(len: usize, value: f32) -> Vec<f32> {
    vec![value; len]
}

#[test]
fn high_concurrency_64_streams_bounded_drafts_and_exact_commits() {
    const NUM_STREAMS: usize = 64;
    const NUM_PRODUCERS: usize = 4;
    const NUM_WORKERS: usize = 8;
    const MAX_CONCURRENT_DRAFTS: usize = 4;
    const MAX_TOKENS: usize = 32;

    // Assign every stream distinct partial and final audio lengths so each
    // utterance maps to a unique reference transcript.
    let mut transcripts = Vec::with_capacity(NUM_STREAMS * 2);
    let mut partial_pcms = Vec::with_capacity(NUM_STREAMS);
    let mut final_pcms = Vec::with_capacity(NUM_STREAMS);

    for i in 0..NUM_STREAMS {
        let partial_len = 800 + i * 16;
        let final_len = 2_400 + i * 16;
        let base = 100 + (i as TokenId) * 10;

        // Partial draft matches first 2 tokens, then diverges at token 2 to
        // exercise KV rollback at commit across all 64 concurrent streams.
        let partial_seq = vec![base, base + 1, base + 90, base + 91, MOCK_EOS_TOKEN];
        let final_seq = vec![base, base + 1, base + 2, base + 3, base + 4, MOCK_EOS_TOKEN];

        transcripts.push((partial_len, partial_seq));
        transcripts.push((final_len, final_seq));
        partial_pcms.push(samples(partial_len, 0.25));
        final_pcms.push(samples(final_len, 0.25));
    }

    let backend = Arc::new(MockBackend::with_transcripts(transcripts));
    let expected_offline: Vec<Vec<TokenId>> = final_pcms
        .iter()
        .map(|pcm| backend.reference_offline_decode(pcm, MAX_TOKENS))
        .collect();

    struct StreamEntry {
        session: DevlSession<Arc<MockBackend>>,
        final_pcm: Vec<f32>,
        committed_tokens: Option<Vec<TokenId>>,
    }

    let streams: Vec<Arc<Mutex<StreamEntry>>> = (0..NUM_STREAMS)
        .map(|i| {
            let mut session = DevlSession::new(
                Arc::clone(&backend),
                SessionConfig {
                    max_tokens: MAX_TOKENS,
                    enable_checkpoint_promotion: false,
                },
            )
            .unwrap();
            session.push_audio(&partial_pcms[i]).unwrap();
            Arc::new(Mutex::new(StreamEntry {
                session,
                final_pcm: final_pcms[i].clone(),
                committed_tokens: None,
            }))
        })
        .collect();

    enum WorkItem {
        Draft(Arc<Mutex<StreamEntry>>),
        Commit(Arc<Mutex<StreamEntry>>),
    }

    let scheduler = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: MAX_CONCURRENT_DRAFTS,
    });

    // Occupy every draft slot before releasing the producers, so the peak
    // assertion still exercises the limit when commit submissions are fast.
    let initial_drafts: [u64; MAX_CONCURRENT_DRAFTS] = std::array::from_fn(|idx| {
        scheduler.submit_draft(
            StreamId(idx as u64),
            WorkItem::Draft(Arc::clone(&streams[idx])),
        )
    });
    let draft_slots_filled = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));
    let start_barrier = Arc::new(Barrier::new(NUM_PRODUCERS + 1));
    let in_flight_drafts = Arc::new(AtomicUsize::new(0));
    let peak_active_drafts = Arc::new(AtomicUsize::new(0));
    let commits_completed = Arc::new(AtomicUsize::new(0));

    let mut workers = Vec::with_capacity(NUM_WORKERS);
    for _ in 0..NUM_WORKERS {
        let sched = scheduler.clone();
        let filled = Arc::clone(&draft_slots_filled);
        let in_flight = Arc::clone(&in_flight_drafts);
        let peak = Arc::clone(&peak_active_drafts);
        let commits_done = Arc::clone(&commits_completed);

        workers.push(thread::spawn(move || {
            while let Some(job) = sched.next_job() {
                let stream_id = job.stream_id();
                let seq = job.seq();
                let priority = job.priority();
                let (item, _permit) = job.into_parts();

                match item {
                    WorkItem::Draft(entry) => {
                        assert_eq!(priority, JobPriority::Draft);
                        let active_now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(active_now, Ordering::SeqCst);
                        peak.fetch_max(sched.active_drafts(), Ordering::SeqCst);
                        if initial_drafts.contains(&seq) {
                            filled.wait();
                        }

                        {
                            let mut guard = entry.lock().unwrap();
                            if matches!(
                                guard.session.state(),
                                SessionState::Idle | SessionState::Drafting
                            ) {
                                let _ = guard
                                    .session
                                    .run_draft_cancellable(|| {
                                        sched.should_preempt_draft(stream_id, seq)
                                    })
                                    .unwrap();
                            }
                        }

                        in_flight.fetch_sub(1, Ordering::SeqCst);
                    }
                    WorkItem::Commit(entry) => {
                        assert_eq!(priority, JobPriority::Commit);
                        let mut guard = entry.lock().unwrap();
                        let buffered = guard.session.audio_buffer().len();
                        let tail = guard.final_pcm[buffered..].to_vec();
                        if !tail.is_empty() {
                            guard.session.push_audio(&tail).unwrap();
                        }
                        let res = guard.session.commit().unwrap();
                        guard.committed_tokens = Some(res.tokens.clone());
                        commits_done.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        }));
    }

    let streams_per_producer = NUM_STREAMS / NUM_PRODUCERS;
    let mut producers = Vec::with_capacity(NUM_PRODUCERS);
    for p in 0..NUM_PRODUCERS {
        let sched = scheduler.clone();
        let start = Arc::clone(&start_barrier);
        let chunk_start = p * streams_per_producer;
        let chunk_end = chunk_start + streams_per_producer;
        let producer_streams: Vec<(usize, Arc<Mutex<StreamEntry>>)> = (chunk_start..chunk_end)
            .map(|idx| (idx, Arc::clone(&streams[idx])))
            .collect();

        producers.push(thread::spawn(move || {
            start.wait();
            // Interleave multiple draft rounds and commits across this producer's streams.
            for round in 0..3 {
                for &(idx, ref entry) in &producer_streams {
                    sched.submit_draft(StreamId(idx as u64), WorkItem::Draft(Arc::clone(entry)));
                }
                if round == 1 {
                    for &(idx, ref entry) in producer_streams.iter().step_by(2) {
                        sched.submit_commit(
                            StreamId(idx as u64),
                            WorkItem::Commit(Arc::clone(entry)),
                        );
                    }
                }
            }
            for &(idx, ref entry) in producer_streams.iter().skip(1).step_by(2) {
                sched.submit_commit(StreamId(idx as u64), WorkItem::Commit(Arc::clone(entry)));
            }
        }));
    }

    draft_slots_filled.wait();
    start_barrier.wait();
    for producer in producers {
        producer.join().unwrap();
    }
    scheduler.shutdown();
    for worker in workers {
        worker.join().unwrap();
    }

    assert_eq!(commits_completed.load(Ordering::SeqCst), NUM_STREAMS);
    assert_eq!(
        peak_active_drafts.load(Ordering::SeqCst),
        MAX_CONCURRENT_DRAFTS,
        "the observed peak must reach the configured limit without exceeding it"
    );
    assert_eq!(scheduler.active_commits(), 0);
    assert_eq!(scheduler.active_drafts(), 0);
    assert_eq!(scheduler.pending_commits(), 0);
    assert_eq!(scheduler.pending_drafts(), 0);

    for (i, (entry, expected)) in streams.iter().zip(&expected_offline).enumerate() {
        let guard = entry.lock().unwrap();
        assert_eq!(guard.session.state(), SessionState::Finalized);
        assert_eq!(
            guard.committed_tokens.as_ref().expect("stream committed"),
            expected,
            "stream {i} final tokens diverged from offline decode"
        );
    }
}

#[test]
fn fifo_commit_ordering_under_heavy_draft_load() {
    const NUM_DRAFTS: usize = 64;
    const NUM_COMMITS: usize = 48;
    const MAX_CONCURRENT_DRAFTS: usize = 4;

    // Part 1: Under a backlog of 64 queued drafts, all 48 commits must dequeue
    // strictly before any draft and in exact FIFO submission order.
    {
        let scheduler = PriorityScheduler::new(SchedulerConfig {
            max_concurrent_drafts: MAX_CONCURRENT_DRAFTS,
        });

        for i in 0..NUM_DRAFTS {
            scheduler.submit_draft(StreamId(1_000 + i as u64), i);
        }
        for i in 0..NUM_COMMITS {
            scheduler.submit_commit(StreamId(i as u64), i);
        }

        assert_eq!(scheduler.pending_drafts(), NUM_DRAFTS);
        assert_eq!(scheduler.pending_commits(), NUM_COMMITS);

        scheduler.shutdown();

        let worker_sched = scheduler.clone();
        let dequeued = thread::spawn(move || {
            let mut order = Vec::with_capacity(NUM_COMMITS + NUM_DRAFTS);
            while let Some(job) = worker_sched.next_job() {
                let priority = job.priority();
                let (payload, _permit) = job.into_parts();
                order.push((priority, payload));
            }
            order
        })
        .join()
        .unwrap();

        assert_eq!(dequeued.len(), NUM_COMMITS + NUM_DRAFTS);
        for (idx, &(priority, payload)) in dequeued[..NUM_COMMITS].iter().enumerate() {
            assert_eq!(
                priority,
                JobPriority::Commit,
                "position {idx} dequeued a draft before all commits drained"
            );
            assert_eq!(
                payload, idx,
                "commit order violated FIFO at position {idx}: expected {idx}, got {payload}"
            );
        }
        for &(priority, _) in &dequeued[NUM_COMMITS..] {
            assert_eq!(priority, JobPriority::Draft);
        }
        assert_eq!(scheduler.active_commits(), 0);
        assert_eq!(scheduler.active_drafts(), 0);
    }

    // Part 2: Even while all `max_concurrent_drafts` slots are held by active
    // workers, an ordered batch of commits still dequeues immediately in FIFO order.
    {
        let scheduler = PriorityScheduler::new(SchedulerConfig {
            max_concurrent_drafts: MAX_CONCURRENT_DRAFTS,
        });
        for i in 0..MAX_CONCURRENT_DRAFTS + 8 {
            scheduler.submit_draft(StreamId(2_000 + i as u64), i);
        }

        let drafts_saturated = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));
        let release_drafts = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));

        let mut draft_holders = Vec::with_capacity(MAX_CONCURRENT_DRAFTS);
        for _ in 0..MAX_CONCURRENT_DRAFTS {
            let sched = scheduler.clone();
            let sat = Arc::clone(&drafts_saturated);
            let rel = Arc::clone(&release_drafts);
            draft_holders.push(thread::spawn(move || {
                let job = sched.next_job().expect("expected draft job");
                assert_eq!(job.priority(), JobPriority::Draft);
                sat.wait();
                rel.wait();
                drop(job);
            }));
        }

        drafts_saturated.wait();
        assert_eq!(scheduler.active_drafts(), MAX_CONCURRENT_DRAFTS);
        // Draft slots are full, so no additional draft can dequeue.
        assert!(scheduler.try_next_job().is_none());

        // Submit 32 commits while draft slots stay saturated.
        for i in 0..32_usize {
            scheduler.submit_commit(StreamId(3_000 + i as u64), i);
        }

        let mut observed_commits = Vec::with_capacity(32);
        for _ in 0..32 {
            let job = scheduler
                .try_next_job()
                .expect("commit must dequeue even when draft slots are saturated");
            assert_eq!(job.priority(), JobPriority::Commit);
            observed_commits.push(*job.payload());
            drop(job);
        }
        assert_eq!(observed_commits, (0..32).collect::<Vec<_>>());

        release_drafts.wait();
        for h in draft_holders {
            h.join().unwrap();
        }
        scheduler.shutdown();
        while scheduler.next_job().is_some() {}
        assert_eq!(scheduler.active_commits(), 0);
        assert_eq!(scheduler.active_drafts(), 0);
    }
}

#[test]
fn rapid_per_stream_coalescing_and_commit_eviction_under_contention() {
    const NUM_STREAMS: usize = 32;
    const ROUNDS_PER_STREAM: usize = 200;
    const NUM_PRODUCERS: usize = 4;

    // Phase 1: Contended multi-producer draft coalescing + commit eviction while
    // verifying pending_drafts never exceeds active stream count.
    {
        let scheduler = PriorityScheduler::new(SchedulerConfig {
            max_concurrent_drafts: 4,
        });
        let start = Arc::new(Barrier::new(NUM_PRODUCERS));
        let max_observed_pending = Arc::new(AtomicUsize::new(0));
        let per_producer = NUM_STREAMS / NUM_PRODUCERS;

        let mut producers = Vec::with_capacity(NUM_PRODUCERS);
        for p in 0..NUM_PRODUCERS {
            let sched = scheduler.clone();
            let barrier = Arc::clone(&start);
            let max_pending = Arc::clone(&max_observed_pending);
            producers.push(thread::spawn(move || {
                barrier.wait();
                let lo = p * per_producer;
                let hi = lo + per_producer;
                for round in 0..ROUNDS_PER_STREAM {
                    for s in lo..hi {
                        sched.submit_draft(StreamId(s as u64), round);
                    }
                    max_pending.fetch_max(sched.pending_drafts(), Ordering::SeqCst);
                }
            }));
        }

        for p in producers {
            p.join().unwrap();
        }

        assert!(
            max_observed_pending.load(Ordering::SeqCst) <= NUM_STREAMS,
            "pending drafts {} exceeded active stream count {NUM_STREAMS}",
            max_observed_pending.load(Ordering::SeqCst)
        );
        assert_eq!(scheduler.pending_drafts(), NUM_STREAMS);

        // Interleave commits for all even streams -> must evict all even drafts.
        for s in (0..NUM_STREAMS).step_by(2) {
            scheduler.submit_commit(StreamId(s as u64), ROUNDS_PER_STREAM);
        }
        assert_eq!(scheduler.pending_commits(), NUM_STREAMS / 2);
        assert_eq!(scheduler.pending_drafts(), NUM_STREAMS / 2);

        let mut evicted_check = HashSet::new();
        let mut odd_draft_generations = HashMap::new();
        while let Some(job) = scheduler.try_next_job() {
            match job.priority() {
                JobPriority::Commit => {
                    assert_eq!(job.stream_id().0 % 2, 0);
                    evicted_check.insert(job.stream_id());
                }
                JobPriority::Draft => {
                    assert_eq!(
                        job.stream_id().0 % 2,
                        1,
                        "even stream {:?} draft should have been evicted by commit",
                        job.stream_id()
                    );
                    let prev = odd_draft_generations.insert(job.stream_id(), *job.payload());
                    assert!(prev.is_none(), "duplicate draft dequeued for same stream");
                }
            }
        }

        assert_eq!(evicted_check.len(), NUM_STREAMS / 2);
        assert_eq!(odd_draft_generations.len(), NUM_STREAMS / 2);
        for (&stream_id, &gen_idx) in &odd_draft_generations {
            assert_eq!(
                gen_idx,
                ROUNDS_PER_STREAM - 1,
                "stream {stream_id:?} did not retain its newest coalesced draft"
            );
        }
        assert_eq!(scheduler.active_commits(), 0);
        assert_eq!(scheduler.active_drafts(), 0);
    }

    // Phase 2: Live contended producers and workers simultaneously coalescing
    // drafts and interleaving commits across 32 streams.
    {
        let scheduler = PriorityScheduler::new(SchedulerConfig {
            max_concurrent_drafts: 4,
        });
        let start = Arc::new(Barrier::new(NUM_PRODUCERS + 4));
        let max_pending_drafts = Arc::new(AtomicUsize::new(0));
        let commits_seen = Arc::new(Mutex::new(HashSet::<StreamId>::new()));

        let mut workers = Vec::new();
        for _ in 0..4 {
            let sched = scheduler.clone();
            let barrier = Arc::clone(&start);
            let max_pending = Arc::clone(&max_pending_drafts);
            let seen = Arc::clone(&commits_seen);
            workers.push(thread::spawn(move || {
                barrier.wait();
                while let Some(job) = sched.next_job() {
                    max_pending.fetch_max(sched.pending_drafts(), Ordering::SeqCst);
                    if job.priority() == JobPriority::Commit {
                        seen.lock().unwrap().insert(job.stream_id());
                    }
                    drop(job);
                }
            }));
        }

        let per_producer = NUM_STREAMS / NUM_PRODUCERS;
        let mut producers = Vec::new();
        for p in 0..NUM_PRODUCERS {
            let sched = scheduler.clone();
            let barrier = Arc::clone(&start);
            let max_pending = Arc::clone(&max_pending_drafts);
            producers.push(thread::spawn(move || {
                barrier.wait();
                let lo = p * per_producer;
                let hi = lo + per_producer;
                for round in 0..100 {
                    for s in lo..hi {
                        sched.submit_draft(StreamId(s as u64), round);
                    }
                    max_pending.fetch_max(sched.pending_drafts(), Ordering::SeqCst);
                }
                for s in lo..hi {
                    sched.submit_commit(StreamId(s as u64), 100);
                }
            }));
        }

        for p in producers {
            p.join().unwrap();
        }
        scheduler.shutdown();
        for w in workers {
            w.join().unwrap();
        }

        assert!(max_pending_drafts.load(Ordering::SeqCst) <= NUM_STREAMS);
        assert_eq!(commits_seen.lock().unwrap().len(), NUM_STREAMS);
        assert_eq!(scheduler.active_commits(), 0);
        assert_eq!(scheduler.active_drafts(), 0);
    }
}

#[test]
fn step_level_cooperative_preemption_under_multi_stream_load() {
    const MAX_CONCURRENT_DRAFTS: usize = 4;
    const NUM_WORKERS: usize = 4;
    const NUM_QUEUED_BG_DRAFTS: usize = 8;
    const NUM_COMMITS: usize = 12;

    let long_draft_seq: Vec<TokenId> = (1..=24).chain(std::iter::once(MOCK_EOS_TOKEN)).collect();
    let commit_seq: Vec<TokenId> = vec![500, 501, 502, 503, MOCK_EOS_TOKEN];

    let drafts_entered_step = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));
    let commits_enqueued = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));
    let hook_counter = Arc::new(AtomicUsize::new(0));

    let b_entered = Arc::clone(&drafts_entered_step);
    let b_enqueued = Arc::clone(&commits_enqueued);
    let hook_count = Arc::clone(&hook_counter);

    let backend = MockBackend::with_transcripts([(1_600, long_draft_seq), (960, commit_seq)]);
    let expected_commit = backend.reference_offline_decode(&samples(960, 0.25), 64);
    let backend = Arc::new(backend.with_step_hook(move |_| {
        // The first four calls are the first decode steps of four distinct drafts.
        if hook_count.fetch_add(1, Ordering::SeqCst) < MAX_CONCURRENT_DRAFTS {
            b_entered.wait();
            b_enqueued.wait();
        }
    }));

    enum Task {
        ActiveDraft(DevlSession<Arc<MockBackend>>),
        BackgroundDraft(DevlSession<Arc<MockBackend>>),
        Commit(DevlSession<Arc<MockBackend>>),
    }

    let scheduler = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: MAX_CONCURRENT_DRAFTS,
    });

    // Enqueue 4 active drafts + 8 additional queued background drafts.
    for i in 0..MAX_CONCURRENT_DRAFTS {
        let mut s = DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
        s.push_audio(&samples(1_600, 0.25)).unwrap();
        scheduler.submit_draft(StreamId(i as u64), Task::ActiveDraft(s));
    }
    for i in 0..NUM_QUEUED_BG_DRAFTS {
        let mut s = DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
        s.push_audio(&samples(1_600, 0.25)).unwrap();
        scheduler.submit_draft(StreamId(100 + i as u64), Task::BackgroundDraft(s));
    }

    // Hold commit completion until all 4 in-flight drafts have finished yielding so
    // `pending_commits` cannot drain to 0 while the 4th worker is leaving `step_hook`.
    let yield_gate = Arc::new((Mutex::new(0usize), Condvar::new()));
    let active_draft_outcomes = Arc::new(Mutex::new(Vec::<(StreamId, bool, usize)>::new()));
    let event_log = Arc::new(Mutex::new(Vec::<(&'static str, StreamId)>::new()));
    let commit_results = Arc::new(Mutex::new(Vec::<(StreamId, Vec<TokenId>)>::new()));

    let mut workers = Vec::with_capacity(NUM_WORKERS);
    for _ in 0..NUM_WORKERS {
        let sched = scheduler.clone();
        let gate = Arc::clone(&yield_gate);
        let outcomes = Arc::clone(&active_draft_outcomes);
        let log = Arc::clone(&event_log);
        let results = Arc::clone(&commit_results);

        workers.push(thread::spawn(move || {
            while let Some(job) = sched.next_job() {
                let stream_id = job.stream_id();
                let seq = job.seq();
                let (task, _permit) = job.into_parts();

                match task {
                    Task::ActiveDraft(mut session) => {
                        let outcome = session
                            .run_draft_cancellable(|| sched.should_preempt_draft(stream_id, seq))
                            .unwrap();
                        outcomes.lock().unwrap().push((
                            stream_id,
                            outcome.is_preempted(),
                            outcome.result().decode_steps,
                        ));
                        log.lock().unwrap().push(("active-draft", stream_id));

                        let (lock, cvar) = &*gate;
                        let mut yielded = lock.lock().unwrap();
                        *yielded += 1;
                        if *yielded == MAX_CONCURRENT_DRAFTS {
                            cvar.notify_all();
                        }
                    }
                    Task::BackgroundDraft(mut session) => {
                        let _ = session
                            .run_draft_cancellable(|| sched.should_preempt_draft(stream_id, seq))
                            .unwrap();
                        log.lock().unwrap().push(("bg-draft", stream_id));
                    }
                    Task::Commit(mut session) => {
                        // Wait until all 4 active drafts have yielded so at most
                        // NUM_WORKERS - 1 commits can be dequeued early, keeping
                        // pending_commits > 0 for every active draft check.
                        let (lock, cvar) = &*gate;
                        let mut yielded = lock.lock().unwrap();
                        while *yielded < MAX_CONCURRENT_DRAFTS {
                            yielded = cvar.wait(yielded).unwrap();
                        }
                        drop(yielded);

                        let res = session.commit().unwrap();
                        results
                            .lock()
                            .unwrap()
                            .push((stream_id, res.tokens.clone()));
                        log.lock().unwrap().push(("commit", stream_id));
                    }
                }
            }
        }));
    }

    // Wait until all 4 active drafts are inside decode_step #0, then enqueue commits.
    drafts_entered_step.wait();
    for i in 0..NUM_COMMITS {
        let mut s = DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
        s.push_audio(&samples(960, 0.25)).unwrap();
        scheduler.submit_commit(StreamId(200 + i as u64), Task::Commit(s));
    }
    commits_enqueued.wait();

    scheduler.shutdown();
    for w in workers {
        w.join().unwrap();
    }

    // All 4 in-flight drafts must have yielded after at most 1 decode_step.
    let outcomes = active_draft_outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), MAX_CONCURRENT_DRAFTS);
    for &(stream_id, is_preempted, steps) in outcomes.iter() {
        assert!(
            is_preempted,
            "active draft for {stream_id:?} was not cooperatively preempted"
        );
        assert_eq!(
            steps, 1,
            "active draft for {stream_id:?} ran {steps} steps instead of yielding after 1 step"
        );
    }

    // Every commit must match offline greedy output exactly.
    let results = commit_results.lock().unwrap();
    assert_eq!(results.len(), NUM_COMMITS);
    for (stream_id, tokens) in results.iter() {
        assert_eq!(
            tokens, &expected_commit,
            "commit for {stream_id:?} diverged from offline greedy decode"
        );
    }

    // No background draft may finish before all commits have been dequeued and
    // started; verify at least all commits finished cleanly and zero slots leaked.
    let events = event_log.lock().unwrap();
    let first_bg_pos = events
        .iter()
        .position(|&(kind, _)| kind == "bg-draft")
        .expect("background drafts should drain after commits");
    let commits_before_bg = events[..first_bg_pos]
        .iter()
        .filter(|&&(kind, _)| kind == "commit")
        .count();
    assert!(
        commits_before_bg >= NUM_COMMITS - (NUM_WORKERS - 1),
        "commits should run ahead of queued background drafts (commits before first bg-draft: {commits_before_bg})"
    );
    assert_eq!(scheduler.active_commits(), 0);
    assert_eq!(scheduler.active_drafts(), 0);

    // Also verify same-stream cancellation (even when pending_commits == 0) followed
    // by commit exactness on the preempted session.
    let cancel_sched = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: 2,
    });
    let draft_seq = cancel_sched.submit_draft(StreamId(42), ());
    let draft_job = cancel_sched.try_next_job().unwrap();
    assert!(!cancel_sched.should_preempt_draft(StreamId(42), draft_seq));
    cancel_sched.submit_commit(StreamId(42), ());
    let commit_job = cancel_sched.try_next_job().unwrap();
    assert_eq!(cancel_sched.pending_commits(), 0);
    assert!(
        cancel_sched.should_preempt_draft(StreamId(42), draft_seq),
        "in-flight draft must still preempt when its own stream received a newer commit"
    );
    drop((draft_job, commit_job));

    let mut preempted_session =
        DevlSession::new(Arc::clone(&backend), SessionConfig::default()).unwrap();
    preempted_session.push_audio(&samples(1_600, 0.25)).unwrap();
    let mut step_checks = 0usize;
    let draft_out = preempted_session
        .run_draft_cancellable(|| {
            step_checks += 1;
            step_checks > 1
        })
        .unwrap();
    assert!(draft_out.is_preempted());
    let final_res = preempted_session.commit().unwrap();
    assert_eq!(
        final_res.tokens,
        backend.reference_offline_decode(&samples(1_600, 0.25), 448)
    );
}

#[test]
fn panic_safe_raii_job_permit_accounting_and_worker_wakeup() {
    // Part 1: A panic while holding the sole Draft permit releases the slot and
    // wakes a blocked worker waiting in `next_job()`.
    {
        let scheduler = PriorityScheduler::new(SchedulerConfig {
            max_concurrent_drafts: 1,
        });
        scheduler.submit_draft(StreamId(1), "panic-draft");
        scheduler.submit_draft(StreamId(2), "wakeup-draft");

        let held_draft = scheduler.try_next_job().expect("first draft dequeued");
        assert_eq!(held_draft.priority(), JobPriority::Draft);
        assert_eq!(scheduler.active_drafts(), 1);
        assert!(scheduler.try_next_job().is_none());

        let worker_ready = Arc::new(Barrier::new(2));
        let w_ready = Arc::clone(&worker_ready);
        let w_sched = scheduler.clone();
        let (woke, resumed) = mpsc::channel();

        let blocked_worker = thread::spawn(move || {
            w_ready.wait();
            let job = w_sched
                .next_job()
                .expect("worker must wake up after panicked draft drops permit");
            assert_eq!(job.priority(), JobPriority::Draft);
            assert_eq!(*job.payload(), "wakeup-draft");
            let stream_id = job.stream_id();
            drop(job);
            woke.send(stream_id).unwrap();
            assert!(w_sched.next_job().is_none());
        });

        worker_ready.wait();

        // Trigger a controlled panic while holding `held_draft` (and its JobPermit).
        let panic_res = catch_unwind(AssertUnwindSafe(move || {
            let (_payload, _permit) = held_draft.into_parts();
            panic!("controlled draft panic");
        }));
        assert!(panic_res.is_err());

        // Check wakeup before shutdown can send its own notification.
        let resumed_stream = resumed.recv_timeout(Duration::from_secs(10));
        scheduler.shutdown();
        blocked_worker.join().unwrap();

        assert_eq!(resumed_stream.unwrap(), StreamId(2));
        assert_eq!(scheduler.active_drafts(), 0);
        assert_eq!(scheduler.active_commits(), 0);
    }

    // Part 2: Controlled panics across both Commit and Draft permits under
    // multi-worker contention never leak permits or deadlock.
    {
        const NUM_WORKERS: usize = 4;
        const NUM_COMMITS: usize = 16;
        const NUM_DRAFTS: usize = 16;

        let scheduler = PriorityScheduler::new(SchedulerConfig {
            max_concurrent_drafts: 2,
        });

        for i in 0..NUM_DRAFTS {
            scheduler.submit_draft(StreamId(100 + i as u64), i % 2 == 0);
        }
        for i in 0..NUM_COMMITS {
            scheduler.submit_commit(StreamId(i as u64), i % 2 == 0);
        }
        scheduler.shutdown();

        let caught_commit_panics = Arc::new(AtomicUsize::new(0));
        let caught_draft_panics = Arc::new(AtomicUsize::new(0));
        let ok_commits = Arc::new(AtomicUsize::new(0));
        let ok_drafts = Arc::new(AtomicUsize::new(0));

        let mut workers = Vec::with_capacity(NUM_WORKERS);
        for _ in 0..NUM_WORKERS {
            let sched = scheduler.clone();
            let c_panics = Arc::clone(&caught_commit_panics);
            let d_panics = Arc::clone(&caught_draft_panics);
            let c_ok = Arc::clone(&ok_commits);
            let d_ok = Arc::clone(&ok_drafts);

            workers.push(thread::spawn(move || {
                while let Some(job) = sched.next_job() {
                    let priority = job.priority();
                    let res = catch_unwind(AssertUnwindSafe(move || {
                        let (should_panic, _permit) = job.into_parts();
                        if should_panic {
                            panic!("controlled worker panic for {priority:?}");
                        }
                    }));
                    match (priority, res) {
                        (JobPriority::Commit, Err(_)) => {
                            c_panics.fetch_add(1, Ordering::SeqCst);
                        }
                        (JobPriority::Draft, Err(_)) => {
                            d_panics.fetch_add(1, Ordering::SeqCst);
                        }
                        (JobPriority::Commit, Ok(())) => {
                            c_ok.fetch_add(1, Ordering::SeqCst);
                        }
                        (JobPriority::Draft, Ok(())) => {
                            d_ok.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                    // Verify scheduler methods remain usable without mutex poisoning.
                    let _ = sched.active_commits();
                    let _ = sched.active_drafts();
                }
            }));
        }

        for w in workers {
            w.join().unwrap();
        }

        assert_eq!(caught_commit_panics.load(Ordering::SeqCst), NUM_COMMITS / 2);
        assert_eq!(ok_commits.load(Ordering::SeqCst), NUM_COMMITS / 2);
        assert_eq!(caught_draft_panics.load(Ordering::SeqCst), NUM_DRAFTS / 2);
        assert_eq!(ok_drafts.load(Ordering::SeqCst), NUM_DRAFTS / 2);
        assert_eq!(scheduler.active_commits(), 0);
        assert_eq!(scheduler.active_drafts(), 0);
        assert_eq!(scheduler.pending_commits(), 0);
        assert_eq!(scheduler.pending_drafts(), 0);
    }
}
