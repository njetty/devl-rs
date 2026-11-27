//! Measure finalization latency and token parity on local Whisper fixtures.
mod support;

use std::env;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use devl::DevlSession;
use serde_json::{Value, json};
use support::{
    CHECKPOINT_CHUNKS, CHUNK_SAMPLES, DTYPE, Inputs, Result, SESSION_CONFIG, WerAggregate,
    commit_record, emit, is_checkpoint, load_wav, milliseconds, normalize_words, ratio,
    reference_for,
};
use sysinfo::System;

fn git(args: &[&str]) -> Result<String> {
    let output = Command::new("git").args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn blob_id(path: &Path) -> Result<String> {
    git(&[
        "hash-object",
        "--",
        path.to_str().ok_or("path must be valid UTF-8")?,
    ])
}

fn provenance(inputs: &Inputs, model: &str) -> Result<Value> {
    let mut hardware = System::new();
    hardware.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
    let mut cpu_models: Vec<_> = hardware
        .cpus()
        .iter()
        .map(sysinfo::Cpu::brand)
        .filter(|brand| !brand.is_empty())
        .collect();
    cpu_models.sort_unstable();
    cpu_models.dedup();
    let environment: serde_json::Map<String, Value> = [
        "CANDLE_BARRIER_POOL_SPIN_LIMIT",
        "CANDLE_CPU_AFFINITY",
        "CANDLE_CPU_MASK",
        "CANDLE_NUM_THREADS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_TARGET_DIR",
        "RAYON_NUM_THREADS",
        "RUSTFLAGS",
    ]
    .into_iter()
    .filter_map(|key| env::var(key).ok().map(|value| (key.into(), json!(value))))
    .collect();
    Ok(json!({
        "argv": env::args().collect::<Vec<_>>(),
        "cwd": env::current_dir()?,
        "environment": environment,
        "commit_sha": git(&["rev-parse", "HEAD"])?,
        "git_status": git(&["status", "--porcelain"])?,
        "hardware": {
            "cpu_models": cpu_models,
            "logical_cpus": hardware.cpus().len(),
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
            "kernel_version": System::kernel_version(),
            "inference_device": inputs.device,
        },
        "model_id": model,
        "model_dir": inputs.model_dir,
        "model_config_git_blob": blob_id(&inputs.model_dir.join("config.json"))?,
        "model_weights_git_blob": blob_id(&inputs.model_dir.join("model.safetensors"))?,
        "dtype": format!("{DTYPE:?}"),
        "debug_assertions": cfg!(debug_assertions),
        "metal_enabled": cfg!(feature = "metal"),
        "chunk_samples": CHUNK_SAMPLES,
        "checkpoint_interval_chunks": CHECKPOINT_CHUNKS,
        "max_tokens": SESSION_CONFIG.max_tokens,
        "checkpoint_promotion": SESSION_CONFIG.enable_checkpoint_promotion,
    }))
}

fn latency_summary(times: &mut [Duration]) -> Value {
    times.sort_unstable();
    let middle = times.len() / 2;
    let median = if times.len().is_multiple_of(2) {
        (milliseconds(times[middle - 1]) + milliseconds(times[middle])) / 2.0
    } else {
        milliseconds(times[middle])
    };
    let p99_index = (times.len() * 99).div_ceil(100) - 1;
    json!({"median_ms": median, "p99_ms": milliseconds(times[p99_index])})
}

fn main() -> Result<()> {
    let inputs = Inputs::parse("bench_ttfs", 1, usize::MAX)?;
    let tokenizer = inputs.load_tokenizer()?;
    let references = inputs.load_references()?;
    let (backend, model) = inputs.load_backend()?;
    let provenance = provenance(&inputs, &model)?;
    let device = &backend.config().device;
    let mut samples = Vec::new();
    let mut offline_times = Vec::new();
    let mut devl_times = Vec::new();
    let mut exact_matches = 0;
    let mut accepted_tokens = 0;
    let mut draft_tokens = 0;
    let mut offline_wer = WerAggregate::new();
    let mut devl_wer = WerAggregate::new();

    for path in &inputs.fixtures {
        let pcm = load_wav(path)?;
        let fixture_git_blob = blob_id(path)?;

        let endpoint = Instant::now();
        let offline = backend.offline_greedy_decode(&pcm, SESSION_CONFIG.max_tokens)?;
        device.synchronize()?;
        let offline_ttfs = endpoint.elapsed();

        let mut session = DevlSession::new(&backend, SESSION_CONFIG)?;
        let chunks = pcm.chunks(CHUNK_SAMPLES);
        let mut draft_work = Duration::ZERO;
        let mut checkpoints = 0;
        for (index, chunk) in chunks.clone().enumerate() {
            session.push_audio(chunk)?;
            if is_checkpoint(index + 1, chunks.len()) {
                let start = Instant::now();
                session.run_draft()?;
                device.synchronize()?;
                draft_work += start.elapsed();
                checkpoints += 1;
            }
        }
        let endpoint = Instant::now();
        let commit = session.commit()?;
        device.synchronize()?;
        let devl_ttfs = endpoint.elapsed();

        let exact_match = commit.tokens == offline;
        exact_matches += usize::from(exact_match);
        accepted_tokens += commit.accepted_prefix_len;
        draft_tokens += commit.draft_len;
        offline_times.push(offline_ttfs);
        devl_times.push(devl_ttfs);

        let mut sample = json!({
            "fixture": path,
            "fixture_git_blob": fixture_git_blob,
            "audio_samples": pcm.len(),
            "offline_tokens": offline,
            "offline_ttfs_ms": milliseconds(offline_ttfs),
            "devl": commit_record(commit, devl_ttfs),
            "draft_work_ms": milliseconds(draft_work),
            "checkpoints": checkpoints,
            "exact_match": exact_match,
        });

        if let Some(tokenizer) = &tokenizer {
            sample["offline_transcript"] = json!(tokenizer.decode(&offline)?);
            sample["devl_transcript"] = json!(tokenizer.decode(&commit.tokens)?);
        }

        if let Some(references) = &references {
            let reference_text = reference_for(references, path)?;
            let reference_words = normalize_words(reference_text);
            let mut offline_hypothesis = Vec::new();
            let mut devl_hypothesis = Vec::new();
            if let Some(tokenizer) = &tokenizer {
                offline_hypothesis = normalize_words(&tokenizer.decode(&offline)?);
                devl_hypothesis = normalize_words(&tokenizer.decode(&commit.tokens)?);
            }
            let offline_wer_score = offline_wer.add(&reference_words, &offline_hypothesis);
            let devl_wer_score = devl_wer.add(&reference_words, &devl_hypothesis);
            sample["reference_words"] = json!(reference_words.len());
            sample["offline_wer"] = json!(offline_wer_score);
            sample["devl_wer"] = json!(devl_wer_score);
        }

        samples.push(sample);
    }
    let mut summary = json!({
        "sample_count": samples.len(),
        "offline_ttfs": latency_summary(&mut offline_times),
        "devl_ttfs": latency_summary(&mut devl_times),
        "accepted_tokens": accepted_tokens,
        "draft_tokens": draft_tokens,
        "accepted_prefix_ratio": ratio(accepted_tokens, draft_tokens),
        "exact_matches": exact_matches,
        "exact_match_rate": ratio(exact_matches, samples.len()),
    });
    if references.is_some() {
        summary["reference_words"] = json!(offline_wer.reference_words);
        summary["offline_wer_errors"] = json!(offline_wer.errors);
        summary["devl_wer_errors"] = json!(devl_wer.errors);
        summary["offline_wer"] = json!(offline_wer.wer());
        summary["devl_wer"] = json!(devl_wer.wer());
    }
    emit(&json!({
        "provenance": provenance,
        "samples": samples,
        "summary": summary,
    }))?;
    if exact_matches != samples.len() {
        return Err("DEVL tokens differed from offline greedy tokens".into());
    }
    Ok(())
}
