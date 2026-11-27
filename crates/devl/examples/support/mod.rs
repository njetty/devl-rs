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
            "usage: {example} <cpu|metal> <model-dir> <audio.wav>{}",
            if max_files == 1 { "" } else { " <more.wav>..." }
        );
        let [_, device, model_dir, files @ ..] = argv.as_slice() else {
            return Err(usage.into());
        };
        if !(min_files..=max_files).contains(&files.len())
            || !matches!(device.as_str(), "cpu" | "metal")
        {
            return Err(usage.into());
        }
        Ok(Self {
            device: device.clone(),
            model_dir: model_dir.into(),
            fixtures: files.iter().map(PathBuf::from).collect(),
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
