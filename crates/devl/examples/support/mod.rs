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

pub(super) struct ArgsSpec {
    pub(super) min_files: usize,
    pub(super) max_files: usize,
    /// Whether this command accepts `--tokenizer <path>`.
    pub(super) allow_tokenizer: bool,
    /// Whether this command accepts `--references <path>`.
    pub(super) allow_references: bool,
}

#[derive(Debug)]
pub(super) struct Inputs {
    pub(super) device: String,
    pub(super) model_dir: PathBuf,
    pub(super) fixtures: Vec<PathBuf>,
    // Read by the examples that accept --tokenizer / --references. multi_stream
    // never reads them, so the field-level allow is narrow by construction.
    #[allow(dead_code)]
    pub(super) tokenizer: Option<PathBuf>,
    #[allow(dead_code)]
    pub(super) references: Option<PathBuf>,
}

impl Inputs {
    pub(super) fn parse(example: &str, spec: ArgsSpec) -> Result<Self> {
        let argv: Vec<String> = env::args_os()
            .map(|arg| {
                arg.into_string()
                    .map_err(|_| "arguments must be valid UTF-8")
            })
            .collect::<std::result::Result<_, _>>()?;
        Self::parse_argv(example, &spec, &argv)
    }

    /// Parse the argument vector. Pure and testable; `parse` supplies the
    /// actual process arguments.
    fn parse_argv(example: &str, spec: &ArgsSpec, argv: &[String]) -> Result<Self> {
        let usage = format!(
            "usage: {example} <cpu|metal> <model-dir> {}{}{}",
            if spec.max_files == 1 {
                "<audio.wav>"
            } else {
                "<audio.wav>..."
            },
            if spec.allow_tokenizer {
                " [--tokenizer <path>]"
            } else {
                ""
            },
            if spec.allow_references {
                " [--references <path>]"
            } else {
                ""
            }
        );
        let [_, device, model_dir, rest @ ..] = argv else {
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
                    if !spec.allow_tokenizer {
                        return Err(format!("{example} does not accept --tokenizer").into());
                    }
                    if tokenizer.is_some() {
                        return Err("--tokenizer may only be given once".into());
                    }
                    let Some(value) = rest.get(index + 1) else {
                        return Err("--tokenizer requires a <path> value".into());
                    };
                    if value.starts_with("--") {
                        return Err("--tokenizer requires a <path> value".into());
                    }
                    tokenizer = Some(PathBuf::from(value));
                    index += 1;
                }
                "--references" => {
                    if !spec.allow_references {
                        return Err(format!("{example} does not accept --references").into());
                    }
                    if references.is_some() {
                        return Err("--references may only be given once".into());
                    }
                    let Some(value) = rest.get(index + 1) else {
                        return Err("--references requires a <path> value".into());
                    };
                    if value.starts_with("--") {
                        return Err("--references requires a <path> value".into());
                    }
                    references = Some(PathBuf::from(value));
                    index += 1;
                }
                _ if arg.starts_with("--") => {
                    return Err(format!("unknown option '{arg}'").into());
                }
                _ => fixtures.push(PathBuf::from(arg)),
            }
            index += 1;
        }
        if !(spec.min_files..=spec.max_files).contains(&fixtures.len()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(example: &str, rest: &[&str]) -> Vec<String> {
        let mut argv = vec![example.to_string(), "cpu".to_string(), "model".to_string()];
        argv.extend(rest.iter().map(|arg| arg.to_string()));
        argv
    }

    fn file_stream_spec() -> ArgsSpec {
        ArgsSpec {
            min_files: 1,
            max_files: 1,
            allow_tokenizer: true,
            allow_references: false,
        }
    }

    fn multi_stream_spec() -> ArgsSpec {
        ArgsSpec {
            min_files: 2,
            max_files: usize::MAX,
            allow_tokenizer: false,
            allow_references: false,
        }
    }

    fn bench_spec() -> ArgsSpec {
        ArgsSpec {
            min_files: 1,
            max_files: usize::MAX,
            allow_tokenizer: true,
            allow_references: true,
        }
    }

    #[test]
    fn legacy_positional_invocation_is_preserved() {
        let inputs = Inputs::parse_argv(
            "file_stream",
            &file_stream_spec(),
            &argv("file_stream", &["a.wav"]),
        )
        .expect("parse");
        assert_eq!(inputs.fixtures, vec![PathBuf::from("a.wav")]);
        assert!(inputs.tokenizer.is_none());
        assert!(inputs.references.is_none());

        let inputs = Inputs::parse_argv(
            "multi_stream",
            &multi_stream_spec(),
            &argv("multi_stream", &["a.wav", "b.wav"]),
        )
        .expect("parse");
        assert_eq!(inputs.fixtures.len(), 2);

        let inputs = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv("bench_ttfs", &["a.wav", "b.wav"]),
        )
        .expect("parse");
        assert_eq!(inputs.fixtures.len(), 2);
    }

    #[test]
    fn file_stream_accepts_tokenizer_and_rejects_references() {
        let inputs = Inputs::parse_argv(
            "file_stream",
            &file_stream_spec(),
            &argv("file_stream", &["--tokenizer", "tok.json", "a.wav"]),
        )
        .expect("parse");
        assert_eq!(inputs.tokenizer, Some(PathBuf::from("tok.json")));
        assert_eq!(inputs.fixtures, vec![PathBuf::from("a.wav")]);

        let err = Inputs::parse_argv(
            "file_stream",
            &file_stream_spec(),
            &argv("file_stream", &["--references", "refs.json", "a.wav"]),
        )
        .expect_err("file_stream must reject --references");
        assert!(err.to_string().contains("--references"));
    }

    #[test]
    fn multi_stream_rejects_both_scoped_flags() {
        let err = Inputs::parse_argv(
            "multi_stream",
            &multi_stream_spec(),
            &argv(
                "multi_stream",
                &["--tokenizer", "tok.json", "a.wav", "b.wav"],
            ),
        )
        .expect_err("multi_stream must reject --tokenizer");
        assert!(err.to_string().contains("--tokenizer"));

        let err = Inputs::parse_argv(
            "multi_stream",
            &multi_stream_spec(),
            &argv(
                "multi_stream",
                &["--references", "refs.json", "a.wav", "b.wav"],
            ),
        )
        .expect_err("multi_stream must reject --references");
        assert!(err.to_string().contains("--references"));
    }

    #[test]
    fn repeated_flags_are_rejected() {
        let err = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv(
                "bench_ttfs",
                &["--tokenizer", "a.json", "--tokenizer", "b.json", "a.wav"],
            ),
        )
        .expect_err("repeated --tokenizer must be rejected");
        assert!(err.to_string().contains("--tokenizer"));

        let err = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv(
                "bench_ttfs",
                &[
                    "--tokenizer",
                    "a.json",
                    "--references",
                    "r.json",
                    "--references",
                    "s.json",
                    "a.wav",
                ],
            ),
        )
        .expect_err("repeated --references must be rejected");
        assert!(err.to_string().contains("--references"));
    }

    #[test]
    fn missing_flag_values_are_rejected() {
        let err = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv("bench_ttfs", &["--tokenizer"]),
        )
        .expect_err("missing --tokenizer value must be rejected");
        assert!(err.to_string().contains("--tokenizer"));

        let err = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv(
                "bench_ttfs",
                &["--references", "--tokenizer", "tok.json", "a.wav"],
            ),
        )
        .expect_err("a flag where a value belongs must be rejected");
        assert!(err.to_string().contains("--references"));
    }

    #[test]
    fn references_requires_tokenizer() {
        let err = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv("bench_ttfs", &["--references", "refs.json", "a.wav"]),
        )
        .expect_err("--references requires --tokenizer");
        assert!(err.to_string().contains("--references"));
    }

    #[test]
    fn unknown_flags_and_bad_devices_are_rejected() {
        let err = Inputs::parse_argv(
            "bench_ttfs",
            &bench_spec(),
            &argv("bench_ttfs", &["--fast", "a.wav"]),
        )
        .expect_err("unknown option must be rejected");
        assert!(err.to_string().contains("--fast"));

        let mut args = argv("bench_ttfs", &["a.wav"]);
        args[1] = "cuda".to_string();
        let err = Inputs::parse_argv("bench_ttfs", &bench_spec(), &args)
            .expect_err("bad device must be rejected");
        assert!(err.to_string().contains("usage:"));

        let err = Inputs::parse_argv(
            "file_stream",
            &file_stream_spec(),
            &argv("file_stream", &["a.wav", "b.wav"]),
        )
        .expect_err("file_stream allows exactly one fixture");
        assert!(err.to_string().contains("usage:"));
    }
}
