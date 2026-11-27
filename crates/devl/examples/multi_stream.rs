//! Replay independent Whisper streams with a scheduler and one inference worker.
mod support;

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use devl::{DevlSession, JobPriority, PriorityScheduler, SchedulerConfig, StreamId};
use serde_json::json;
use support::{
    CHUNK_SAMPLES, Inputs, Result, SESSION_CONFIG, commit_record, emit, is_checkpoint, load_wav,
};

enum Work {
    Draft,
    Commit { tail: Vec<f32>, endpoint: Instant },
}

fn main() -> Result<()> {
    let inputs = Inputs::parse("multi_stream", 2, usize::MAX)?;
    let (backend, model) = inputs.load_backend()?;
    let scheduler = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: 1,
    });
    let mut sessions = Vec::new();
    let mut tails = Vec::new();
    for (index, path) in inputs.fixtures.iter().enumerate() {
        let mut pcm = load_wav(path)?;
        if pcm.len() < 2 {
            return Err(format!("{}: expected at least two samples", path.display()).into());
        }
        let chunks = pcm.len().div_ceil(CHUNK_SAMPLES);
        let split = (1..chunks)
            .find(|&chunk| is_checkpoint(chunk, chunks))
            .map_or(pcm.len() / 2, |chunk| chunk * CHUNK_SAMPLES);
        let tail = pcm.split_off(split);
        let mut session = DevlSession::new(&backend, SESSION_CONFIG)?;
        for chunk in pcm.chunks(CHUNK_SAMPLES) {
            session.push_audio(chunk)?;
        }
        sessions.push(session);
        tails.push(tail);
        scheduler.submit_draft(StreamId(index as u64), Work::Draft);
    }

    thread::scope(|scope| {
        let (boundary_tx, boundary_rx) = mpsc::channel();
        let (submitted_tx, submitted_rx) = mpsc::channel();
        let producer_scheduler = scheduler.clone();
        scope.spawn(move || {
            if boundary_rx.recv().is_ok() {
                // End another stream first, then the stream with the running draft.
                let endpoint = Instant::now();
                for (index, tail) in tails.into_iter().enumerate().rev() {
                    producer_scheduler
                        .submit_commit(StreamId(index as u64), Work::Commit { tail, endpoint });
                }
                producer_scheduler.shutdown();
                let _ = submitted_tx.send(());
            }
        });

        let mut boundary_tx = Some(boundary_tx);
        let mut release_tails = || {
            if let Some(sender) = boundary_tx.take() {
                sender.send(()).expect("producer should be waiting");
                submitted_rx.recv().expect("producer should submit commits");
            }
        };
        while let Some(job) = scheduler.next_job() {
            let stream = job.stream_id();
            let seq = job.seq();
            let priority = job.priority();
            let (work, _permit) = job.into_parts();
            let session = &mut sessions[stream.0 as usize];
            match (priority, work) {
                (JobPriority::Draft, Work::Draft) => {
                    let outcome = session.run_draft_cancellable(|| {
                        // Synchronize at a decode-step boundary so the demonstration
                        // is repeatable regardless of model or host speed.
                        release_tails();
                        scheduler.should_preempt_draft(stream, seq)
                    })?;
                    // A draft that immediately reaches EOS has no decode-step callback.
                    release_tails();
                    emit(&json!({
                        "event": "draft",
                        "stream": stream.0,
                        "preempted": outcome.is_preempted(),
                        "tokens": outcome.result().tokens,
                    }))?;
                }
                (JobPriority::Commit, Work::Commit { tail, endpoint }) => {
                    for chunk in tail.chunks(CHUNK_SAMPLES) {
                        session.push_audio(chunk)?;
                    }
                    let commit = session.commit()?;
                    let elapsed = endpoint.elapsed();
                    emit(&json!({
                        "event": "final",
                        "stream": stream.0,
                        "fixture": inputs.fixtures[stream.0 as usize],
                        "model": model,
                        "commit": commit_record(commit, elapsed),
                    }))?;
                }
                _ => unreachable!("job priority must match its payload"),
            }
        }
        Ok(())
    })
}
