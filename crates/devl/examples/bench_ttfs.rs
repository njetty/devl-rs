//! Measure finalization latency and token parity on local Whisper fixtures.
mod support;
#[path = "support/tokenizer.rs"]
mod tokenizer;

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use devl::DevlSession;
use serde_json::{Value, json};
use support::{
    ArgsSpec, CHECKPOINT_CHUNKS, CHUNK_SAMPLES, DTYPE, Inputs, Result, SESSION_CONFIG,
    commit_record, emit, is_checkpoint, load_wav, milliseconds, ratio,
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
    let mut value = json!({
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
    });
    if let Some(tokenizer) = &inputs.tokenizer {
        value["tokenizer_git_blob"] = json!(blob_id(tokenizer)?);
    }
    if let Some(references) = &inputs.references {
        value["references_git_blob"] = json!(blob_id(references)?);
    }
    Ok(value)
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

/// Load a reference transcript map from JSON: WAV file basename to transcript.
fn load_references(path: &Path) -> Result<HashMap<String, String>> {
    let raw = fs::read_to_string(path)?;
    let references: HashMap<String, String> = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "{}: expected a JSON object mapping WAV basenames to transcripts: {err}",
            path.display()
        )
    })?;
    Ok(references)
}

/// Reference lookup keyed on the fixture's file name.
fn reference_for<'a>(references: &'a HashMap<String, String>, fixture: &Path) -> Result<&'a str> {
    let key = fixture
        .file_name()
        .ok_or_else(|| format!("{}: path has no file name", fixture.display()))?
        .to_str()
        .ok_or_else(|| format!("{}: file name is not valid UTF-8", fixture.display()))?;
    references.get(key).map(String::as_str).ok_or_else(|| {
        format!(
            "{}: no reference transcript for WAV basename '{key}'",
            fixture.display()
        )
        .into()
    })
}

/// Distinct fixture paths that share one WAV basename pair ambiguously with
/// the reference map; repeated references to the same path are deliberate.
fn validate_fixture_basenames(fixtures: &[PathBuf]) -> Result<()> {
    let mut by_name: HashMap<&std::ffi::OsStr, Vec<&Path>> = HashMap::new();
    for fixture in fixtures {
        let name = fixture
            .file_name()
            .ok_or_else(|| format!("{}: path has no file name", fixture.display()))?;
        by_name.entry(name).or_default().push(fixture);
    }
    for (name, paths) in &by_name {
        let mut distinct: Vec<&Path> = Vec::new();
        for path in paths {
            if !distinct.contains(path) {
                distinct.push(path);
            }
        }
        if distinct.len() > 1 {
            return Err(format!(
                "ambiguous WAV basename '{}' for distinct fixture paths: {}",
                name.to_string_lossy(),
                distinct
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .into());
        }
    }
    Ok(())
}

/// Fetch and normalize a fixture reference transcript; a reference that
/// normalizes to no words is a hard error, never a silent zero-word score.
fn scored_reference_words(
    references: &HashMap<String, String>,
    fixture: &Path,
) -> Result<Vec<String>> {
    let reference_text = reference_for(references, fixture)?;
    let words = normalize_words(reference_text);
    if words.is_empty() {
        return Err(format!(
            "{}: reference transcript normalizes to no words",
            fixture.display()
        )
        .into());
    }
    Ok(words)
}

/// Normalize text to lowercase word tokens: split on any non-alphanumeric
/// character and drop empty fragments. Punctuation and case differences are
/// therefore ignored by WER.
fn normalize_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.to_lowercase().chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Word-level Levenshtein distance over normalized word tokens: each
/// substitution, deletion, or insertion costs one.
fn word_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let m = hypothesis.len();
    let mut previous: Vec<usize> = (0..=m).collect();
    let mut current = vec![0usize; m + 1];
    for (i, reference_word) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, hypothesis_word) in hypothesis.iter().enumerate() {
            let cost = usize::from(reference_word != hypothesis_word);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[m]
}

/// Corpus-level WER accumulator over reference words.
#[derive(Default)]
struct WerTotals {
    reference_words: usize,
    errors: usize,
}

impl WerTotals {
    fn add(&mut self, reference_words: usize, errors: usize) {
        self.reference_words += reference_words;
        self.errors += errors;
    }

    fn wer(&self) -> Option<f64> {
        (self.reference_words != 0).then(|| self.errors as f64 / self.reference_words as f64)
    }
}

fn main() -> Result<()> {
    let inputs = Inputs::parse(
        "bench_ttfs",
        ArgsSpec {
            min_files: 1,
            max_files: usize::MAX,
            allow_tokenizer: true,
            allow_references: true,
        },
    )?;
    let tokenizer = tokenizer::load(inputs.tokenizer.as_deref())?;
    let references = match &inputs.references {
        Some(path) => Some(load_references(path)?),
        None => None,
    };
    if references.is_some() {
        // Reference scoring needs transcripts; the parse step already requires
        // --tokenizer alongside --references, and tokenizer::load only fails
        // when the file cannot be loaded, so it cannot be absent here.
        if tokenizer.is_none() {
            return Err("--references requires --tokenizer for transcript normalization".into());
        }
        validate_fixture_basenames(&inputs.fixtures)?;
    }
    let (backend, model) = inputs.load_backend()?;
    let provenance = provenance(&inputs, &model)?;
    let device = &backend.config().device;
    let mut samples = Vec::new();
    let mut offline_times = Vec::new();
    let mut devl_times = Vec::new();
    let mut exact_matches = 0;
    let mut accepted_tokens = 0;
    let mut draft_tokens = 0;
    let mut offline_wer = WerTotals::default();
    let mut devl_wer = WerTotals::default();

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

        // Decode each hypothesis exactly once; the same text feeds emission
        // and WER scoring.
        let offline_text = tokenizer
            .as_ref()
            .map(|tokenizer| tokenizer.decode(&offline))
            .transpose()?;
        let devl_text = tokenizer
            .as_ref()
            .map(|tokenizer| tokenizer.decode(&commit.tokens))
            .transpose()?;

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

        if let Some(text) = &offline_text {
            sample["offline_transcript"] = json!(text);
        }
        if let Some(text) = &devl_text {
            sample["devl_transcript"] = json!(text);
        }

        if let Some(references) = &references {
            // References require --tokenizer (validated by the parser and by
            // the check above), so the hypotheses are present here.
            let reference_words = scored_reference_words(references, path)?;
            let offline_hypothesis = normalize_words(
                offline_text
                    .as_deref()
                    .ok_or("--references requires --tokenizer for transcript normalization")?,
            );
            let devl_hypothesis = normalize_words(
                devl_text
                    .as_deref()
                    .ok_or("--references requires --tokenizer for transcript normalization")?,
            );
            let offline_errors = word_distance(&reference_words, &offline_hypothesis);
            let devl_errors = word_distance(&reference_words, &devl_hypothesis);
            let offline_wer_score = offline_errors as f64 / reference_words.len() as f64;
            let devl_wer_score = devl_errors as f64 / reference_words.len() as f64;
            offline_wer.add(reference_words.len(), offline_errors);
            devl_wer.add(reference_words.len(), devl_errors);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn normalize_words_splits_on_non_alphanumeric_characters() {
        assert_eq!(normalize_words("Hello, World!"), words("hello world"));
        assert_eq!(
            normalize_words("it's a test... of  normalization"),
            words("it s a test of normalization")
        );
        assert_eq!(
            normalize_words("  spaces\n\t everywhere  "),
            words("spaces everywhere")
        );
        assert_eq!(normalize_words("!!!"), Vec::<String>::new());
        assert_eq!(normalize_words(""), Vec::<String>::new());
    }

    #[test]
    fn word_distance_counts_substitutions_deletions_and_insertions() {
        // 1 substitution
        assert_eq!(word_distance(&words("a b c"), &words("a x c")), 1);
        // 1 deletion
        assert_eq!(word_distance(&words("a b c"), &words("a c")), 1);
        // 1 insertion
        assert_eq!(word_distance(&words("a c"), &words("a b c")), 1);
        // mixed: substitution + deletion + insertion
        assert_eq!(word_distance(&words("a b c"), &words("a x d e")), 3);
        // identical and empty cases
        assert_eq!(word_distance(&words("a b"), &words("a b")), 0);
        assert_eq!(word_distance(&words("a b"), &Vec::<String>::new()), 2);
    }

    #[test]
    fn corpus_totals_are_micro_averaged_over_reference_words() {
        let mut totals = WerTotals::default();
        totals.add(10, 2);
        totals.add(30, 8);
        assert_eq!(totals.reference_words, 40);
        assert_eq!(totals.errors, 10);
        assert_eq!(totals.wer(), Some(0.25));

        let empty = WerTotals::default();
        assert_eq!(empty.wer(), None);
    }

    #[test]
    fn reference_lookup_is_keyed_on_basename() {
        let mut references = HashMap::new();
        references.insert("a.wav".to_string(), "hello world".to_string());
        let fixture = PathBuf::from("some/dir/a.wav");
        assert_eq!(reference_for(&references, &fixture).unwrap(), "hello world");

        let err = reference_for(&references, &PathBuf::from("no/such/b.wav")).unwrap_err();
        assert!(err.to_string().contains("b.wav"));
    }

    #[test]
    fn empty_normalized_reference_is_rejected() {
        let mut references = HashMap::new();
        references.insert("a.wav".to_string(), "!!!".to_string());
        let err = scored_reference_words(&references, &PathBuf::from("dir/a.wav"))
            .expect_err("empty normalized reference must be rejected");
        assert!(err.to_string().contains("no words"));

        let mut references = HashMap::new();
        references.insert("a.wav".to_string(), "hello".to_string());
        assert_eq!(
            scored_reference_words(&references, &PathBuf::from("dir/a.wav")).unwrap(),
            words("hello")
        );
    }

    #[test]
    fn duplicate_basename_is_rejected_unless_it_is_the_same_path() {
        let same = PathBuf::from("some/a.wav");
        let different = PathBuf::from("other/a.wav");

        validate_fixture_basenames(&[same.clone(), different.clone()])
            .expect_err("distinct fixture paths with one basename must be rejected");

        validate_fixture_basenames(&[same.clone(), same.clone()])
            .expect("deliberate repeats of one path must be allowed");
    }
}
