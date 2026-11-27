//! `PriorityScheduler::cancel_stream` and bounded per-stream metadata.

use devl_core::{JobPriority, PriorityScheduler, SchedulerConfig, StreamId};

fn sched() -> PriorityScheduler<u32> {
    PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: 2,
    })
}

#[test]
fn cancel_stream_removes_only_that_streams_queued_jobs() {
    let s = sched();
    s.submit_commit(StreamId(1), 10);
    s.submit_draft(StreamId(2), 20);
    s.submit_draft(StreamId(1), 11);
    s.submit_commit(StreamId(3), 30);
    let mut removed = s.cancel_stream(StreamId(1));
    removed.sort_unstable();
    assert_eq!(removed, vec![10, 11]);
    assert_eq!((s.pending_commits(), s.pending_drafts()), (1, 1));
    // Order and priority of the remaining jobs are preserved.
    let j = s.try_next_job().unwrap();
    assert_eq!(
        (j.stream_id(), j.priority()),
        (StreamId(3), JobPriority::Commit)
    );
    drop(j);
    let j = s.try_next_job().unwrap();
    assert_eq!(
        (j.stream_id(), j.priority()),
        (StreamId(2), JobPriority::Draft)
    );
    drop(j);
    assert!(s.try_next_job().is_none());
    assert_eq!(s.tracked_streams(), 0);
}

#[test]
fn cancel_stream_preempts_running_draft_and_releases_metadata() {
    let s = sched();
    let seq = s.submit_draft(StreamId(1), 1);
    let job = s.try_next_job().unwrap();
    assert!(!s.should_preempt_draft(StreamId(1), seq));
    assert!(s.cancel_stream(StreamId(1)).is_empty());
    assert!(s.should_preempt_draft(StreamId(1), seq));
    assert!(!s.should_preempt_draft(StreamId(2), seq));
    assert!(s.tracked_streams() > 0);
    drop(job);
    assert_eq!(s.tracked_streams(), 0);
    assert_eq!(s.active_drafts(), 0);
    assert!(!s.should_preempt_draft(StreamId(1), seq));
}

#[test]
fn commit_still_preempts_older_running_draft_of_same_stream() {
    let s = sched();
    let seq = s.submit_draft(StreamId(7), 1);
    let job = s.try_next_job().unwrap();
    let commit_seq = s.submit_commit(StreamId(7), 2);
    assert!(commit_seq > seq);
    assert!(s.should_preempt_draft(StreamId(7), seq));
    drop(s.try_next_job().unwrap());
    // Queue drained, but this stream's older draft is still running.
    assert!(s.should_preempt_draft(StreamId(7), seq));
    drop(job);
    assert_eq!(s.tracked_streams(), 0);
}

#[test]
fn metadata_is_bounded_across_many_completed_streams() {
    let s = sched();
    for i in 0..1000 {
        let id = StreamId(i);
        s.submit_draft(id, 0);
        s.submit_commit(id, 1);
        while let Some(j) = s.try_next_job() {
            drop(j);
        }
        s.cancel_stream(id);
    }
    assert_eq!(s.tracked_streams(), 0);
    assert_eq!((s.active_commits(), s.active_drafts()), (0, 0));
    assert_eq!((s.pending_commits(), s.pending_drafts()), (0, 0));
}

#[test]
fn cancel_unblocks_shutdown_with_only_cancelled_jobs_queued() {
    let s = sched();
    s.submit_commit(StreamId(1), 1);
    s.cancel_stream(StreamId(1));
    s.shutdown();
    assert!(s.next_job().is_none());
}
