// Shared helpers are compiled once per example; individual examples use
// different subsets of these items.
#![allow(dead_code)]

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::Config as WhisperModelConfig;
use devl::candle::{
    CandleWhisperBackend, CandleWhisperConfig, WHISPER_MAX_SAMPLES, WHISPER_SAMPLE_RATE_HZ,
};
use devl::{CommitResult, SessionConfig};
use hound::{SampleFormat, WavReader};
use serde_json::{Value, json};

pub(super) type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
pub(super) const CHUNK_SAMPLES: usize = WHISPER_SAMPLE_RATE_HZ / 10;
pub(super) const CHECKPOINT_CHUNKS: usize = 20;
pub(super) const SESSION_CONFIG: SessionConfig = SessionConfig {
    max_tokens: 128,
    enable_checkpoint_promotion: false,
};
pub(super) const DTYPE: DType = DType::F32;

pub(super) struct Inputs {
    pub(super) device: String,
    pub(super) model_dir: PathBuf,
    pub(super) fixtures: Vec<PathBuf>,
    pub(super) tokenizer: Option<PathBuf>,
    pub(super) references: Option<PathBuf>,
}

impl Inputs {
    pub(super) fn parse(example: &str, min_files: usize, max_files: usize) -> Result<Self> {
        let argv: Vec<String> = env::args_os()
            .map(|arg| {
                arg.into_string()
                    .map_err(|_| "arguments must be valid UTF-8")
            })
            .collect::<std::result::Result<_, _>>()?;
        let usage = format!(
            "usage: {example} <cpu|metal> <model-dir> {} [--tokenizer <path>] [--references <path>]",
            if max_files == 1 {
                "<audio.wav>"
            } else {
                "<audio.wav>..."
            }
        );
        let [_, device, model_dir, rest @ ..] = argv.as_slice() else {
            return Err(usage.into());
        };
        if !matches!(device.as_str(), "cpu" | "metal") {
            return Err(usage.into());
        }
        let mut fixtures = Vec::new();
        let mut tokenizer = None;
        let mut references = None;
        let mut index = 0;
        while index < rest.len() {
            let arg = rest[index].as_str();
            match arg {
                "--tokenizer" => {
                    index += 1;
                    let Some(path) = rest.get(index) else {
                        return Err(usage.into());
                    };
                    tokenizer = Some(PathBuf::from(path));
                }
                "--references" => {
                    index += 1;
                    let Some(path) = rest.get(index) else {
                        return Err(usage.into());
                    };
                    references = Some(PathBuf::from(path));
                }
                _ if arg.starts_with("--") => return Err(usage.into()),
                _ => fixtures.push(PathBuf::from(arg)),
            }
            index += 1;
        }
        if !(min_files..=max_files).contains(&fixtures.len()) {
            return Err(usage.into());
        }
        if references.is_some() && tokenizer.is_none() {
            return Err("--references requires --tokenizer for transcript normalization".into());
        }
        Ok(Self {
            device: device.clone(),
            model_dir: model_dir.into(),
            fixtures,
            tokenizer,
            references,
        })
    }

    pub(super) fn load_backend(&self) -> Result<(CandleWhisperBackend, String)> {
        let config: Value = serde_json::from_slice(&fs::read(self.model_dir.join("config.json"))?)?;
        let model_id = config["_name_or_path"]
            .as_str()
            .filter(|name| !name.is_empty())
            .map_or_else(|| self.model_dir.display().to_string(), str::to_owned);
        let model_config: WhisperModelConfig = serde_json::from_value(config)?;
        let (prompt_tokens, eos_token) = match model_config.vocab_size {
            51_864 => (vec![50_257, 50_362], 50_256),
            51_865 => (vec![50_258, 50_259, 50_359, 50_363], 50_257),
            51_866 => (vec![50_258, 50_259, 50_360, 50_364], 50_257),
            size => return Err(format!("unsupported Whisper vocabulary size {size}").into()),
        };
        let device = if self.device == "metal" {
            if !cfg!(feature = "metal") {
                return Err("metal requires --features metal".into());
            }
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };
        let weights = fs::read(self.model_dir.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DTYPE, &device)?;
        let backend = CandleWhisperBackend::from_var_builder(
            vb,
            CandleWhisperConfig {
                model_config: Some(model_config),
                device: device.clone(),
                prompt_tokens,
                eos_token,
                ..CandleWhisperConfig::default()
            },
        )?;
        Ok((backend, model_id))
    }

    pub(super) fn load_tokenizer(&self) -> Result<Option<Tokenizer>> {
        match &self.tokenizer {
            Some(path) => Ok(Some(Tokenizer::load(path)?)),
            None => Ok(None),
        }
    }

    pub(super) fn load_references(&self) -> Result<Option<HashMap<String, String>>> {
        match &self.references {
            Some(path) => load_references(path).map(Some),
            None => Ok(None),
        }
    }
}

pub(super) fn load_wav(path: &Path) -> Result<Vec<f32>> {
    read_wav(path).map_err(|err| format!("{}: {err}", path.display()).into())
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate as usize != WHISPER_SAMPLE_RATE_HZ {
        return Err("expected a 16 kHz mono WAV".into());
    }
    if !(1..=WHISPER_MAX_SAMPLES).contains(&(reader.duration() as usize)) {
        return Err("expected nonempty audio of at most 30 seconds".into());
    }
    let pcm = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|s| f32::from(s) / 32768.0))
            .collect::<hound::Result<Vec<_>>>()?,
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<hound::Result<Vec<_>>>()?,
        _ => return Err("expected PCM16 or IEEE Float32 WAV samples".into()),
    };
    if pcm.iter().any(|sample| !(-1.0..=1.0).contains(sample)) {
        return Err("WAV samples must be finite and normalized to [-1, 1]".into());
    }
    Ok(pcm)
}

pub(super) fn is_checkpoint(chunk: usize, total_chunks: usize) -> bool {
    let first = CHECKPOINT_CHUNKS.min(total_chunks.div_ceil(2));
    chunk < total_chunks && (chunk == first || chunk.is_multiple_of(CHECKPOINT_CHUNKS))
}

pub(super) fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub(super) fn commit_record(commit: &CommitResult, elapsed: Duration) -> Value {
    json!({
        "tokens": commit.tokens,
        "ttfs_ms": milliseconds(elapsed),
        "accepted_prefix_len": commit.accepted_prefix_len,
        "draft_len": commit.draft_len,
        "accepted_prefix_ratio": ratio(commit.accepted_prefix_len, commit.draft_len),
        "decode_steps": commit.decode_steps,
    })
}

pub(super) fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

pub(super) fn emit(value: &Value) -> Result<()> {
    use io::Write as _;
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?;
    Ok(())
}

/// A Whisper tokenizer loaded from a `tokenizer.json` file.
pub(super) struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| format!("{}: {err}", path.display()))?;
        Ok(Self { inner })
    }

    /// Decode token ids into human-readable text, dropping special tokens.
    pub(super) fn decode(&self, tokens: &[u32]) -> Result<String> {
        let text = self
            .inner
            .decode(tokens, true)
            .map_err(|err| format!("tokenizer decode failed: {err}"))?;
        Ok(text.trim().to_owned())
    }
}

/// Load a reference transcript map from JSON: WAV file basename to transcript.
pub(super) fn load_references(path: &Path) -> Result<HashMap<String, String>> {
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
pub(super) fn reference_for<'a>(
    references: &'a HashMap<String, String>,
    fixture: &Path,
) -> Result<&'a str> {
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

/// Normalize text to lowercase word tokens: split on any non-alphanumeric
/// character and drop empty fragments. Punctuation and case differences are
/// therefore ignored by WER.
pub(super) fn normalize_words(text: &str) -> Vec<String> {
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

/// Aggregate reference word counts and word-level errors across samples.
pub(super) struct WerAggregate {
    pub(super) reference_words: usize,
    pub(super) errors: usize,
}

impl WerAggregate {
    pub(super) fn new() -> Self {
        Self {
            reference_words: 0,
            errors: 0,
        }
    }

    pub(super) fn add(&mut self, reference: &[String], hypothesis: &[String]) -> Option<f64> {
        if reference.is_empty() {
            return None;
        }
        let n = reference.len();
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
        self.reference_words += n;
        self.errors += previous[m];
        Some(previous[m] as f64 / n as f64)
    }

    pub(super) fn wer(&self) -> Option<f64> {
        (self.reference_words != 0).then(|| self.errors as f64 / self.reference_words as f64)
    }
}
