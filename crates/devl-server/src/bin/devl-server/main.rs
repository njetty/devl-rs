//! Local `devl-server`: serves the streaming contract over HTTP and WebSocket.
//!
//! One `Coordinator` (one worker pool and one `PriorityScheduler`) backs both
//! listeners. The backend is chosen at start: `mock` (default, deterministic
//! and synthetic), `whisper`, or `qwen3-asr` (the last two need
//! `--features candle`). The process prints one `ready` line, then one
//! `http` and one `ws` line with the bound addresses, and runs until its
//! standard input closes.
//!
//! This is a local development and smoke-test server. It has no TLS,
//! authentication, or rate limiting. See `docs/SERVER_CONTRACT.md`.

mod models;

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use devl_core::{DevlBackend, SessionConfig, TokenId};
use devl_server::{
    Coordinator, HttpConfig, HttpServer, SilenceEndpointConfig, SilenceEndpointPolicy, WsConfig,
    WsServer,
};
use models::{Result, SAMPLE_RATE_HZ};

const USAGE: &str = "\
usage: devl-server [options]

  --backend <mock|whisper|qwen3-asr>   default mock (whisper and qwen3-asr need --features candle)
  --model <dir>                        local checkpoint directory (whisper, qwen3-asr)
  --tokenizer <path>                   Whisper tokenizer.json (default <model>/tokenizer.json)
  --http <addr|off>                    default 127.0.0.1:8080 (port 0 picks a free port)
  --ws <addr|off>                      default 127.0.0.1:8081
  --workers <n>                        inference workers, default 2
  --max-streams <n>                    simultaneous streams, default 8
  --draft-cadence-ms <ms>              checkpoint draft cadence, default 2000
  --end-silence-ms <ms>                enable silence endpointing: end the turn after this much silence
  --checkpoint-silence-ms <ms>         with endpointing: request a draft after this much silence
  --speech-threshold <peak>            with endpointing: speech peak in [0,1], default 0.01
  --max-turn-ms <ms>                   with endpointing: end the turn after this much audio
  -h, --help
";

struct Args {
    backend: String,
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    http: Option<String>,
    ws: Option<String>,
    workers: usize,
    max_streams: usize,
    draft_cadence_ms: u64,
    end_silence_ms: Option<u64>,
    checkpoint_silence_ms: Option<u64>,
    speech_threshold: f32,
    max_turn_ms: Option<u64>,
}

fn parse_args() -> Result<Option<Args>> {
    let mut args = Args {
        backend: "mock".into(),
        model: None,
        tokenizer: None,
        http: Some("127.0.0.1:8080".into()),
        ws: Some("127.0.0.1:8081".into()),
        workers: 2,
        max_streams: 8,
        draft_cadence_ms: 2_000,
        end_silence_ms: None,
        checkpoint_silence_ms: None,
        speech_threshold: 0.01,
        max_turn_ms: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            print!("{USAGE}");
            return Ok(None);
        }
        let mut value = || {
            it.next()
                .ok_or_else(|| format!("{flag} needs a value\n\n{USAGE}"))
        };
        let addr = |v: String| (v != "off").then_some(v);
        match flag.as_str() {
            "--backend" => args.backend = value()?,
            "--model" => args.model = Some(value()?.into()),
            "--tokenizer" => args.tokenizer = Some(value()?.into()),
            "--http" => args.http = addr(value()?),
            "--ws" => args.ws = addr(value()?),
            "--workers" => args.workers = value()?.parse()?,
            "--max-streams" => args.max_streams = value()?.parse()?,
            "--draft-cadence-ms" => args.draft_cadence_ms = duration_ms(&flag, &value()?)?,
            "--end-silence-ms" => args.end_silence_ms = Some(duration_ms(&flag, &value()?)?),
            "--checkpoint-silence-ms" => {
                args.checkpoint_silence_ms = Some(duration_ms(&flag, &value()?)?)
            }
            "--speech-threshold" => args.speech_threshold = value()?.parse()?,
            "--max-turn-ms" => args.max_turn_ms = Some(duration_ms(&flag, &value()?)?),
            other => return Err(format!("unknown option {other}\n\n{USAGE}").into()),
        }
    }
    Ok(Some(args))
}

fn duration_ms(flag: &str, value: &str) -> Result<u64> {
    let ms: u64 = value.parse()?;
    if ms > u64::MAX / (SAMPLE_RATE_HZ / 1_000) {
        return Err(format!("{flag} is too large").into());
    }
    Ok(ms)
}

fn ms_to_samples(ms: u64) -> u64 {
    ms * (SAMPLE_RATE_HZ / 1_000)
}

fn serve<B>(
    backend: B,
    detokenize: impl Fn(&[TokenId]) -> String + Send + Sync + 'static,
    session: SessionConfig,
    max_stream_samples: Option<u64>,
    args: &Args,
) -> Result<()>
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let config = models::coordinator_config(
        session,
        max_stream_samples,
        args.workers,
        args.max_streams,
        args.draft_cadence_ms,
    );
    let coordinator = Arc::new(match args.end_silence_ms {
        None => Coordinator::new(backend, detokenize, config)?,
        Some(end) => {
            let factory = SilenceEndpointPolicy::factory(SilenceEndpointConfig {
                speech_threshold: args.speech_threshold,
                checkpoint_silence_samples: args.checkpoint_silence_ms.map(ms_to_samples),
                end_silence_samples: ms_to_samples(end),
                max_turn_samples: args.max_turn_ms.map(ms_to_samples),
            })?;
            Coordinator::with_endpointing(backend, detokenize, config, factory)?
        }
    });

    let http = args
        .http
        .as_deref()
        .map(|a| HttpServer::bind(a, Arc::clone(&coordinator), HttpConfig::default()))
        .transpose()?;
    let ws = args
        .ws
        .as_deref()
        .map(|a| WsServer::bind(a, Arc::clone(&coordinator), WsConfig::default()))
        .transpose()?;

    let mut out = std::io::stdout().lock();
    writeln!(out, "ready backend={}", args.backend)?;
    if let Some(s) = &http {
        writeln!(out, "http http://{}/v1/stream", s.local_addr())?;
    }
    if let Some(s) = &ws {
        writeln!(out, "ws ws://{}", s.local_addr())?;
    }
    writeln!(out, "close stdin to stop")?;
    out.flush()?;
    drop(out);

    let stdin_result = io::copy(&mut io::stdin().lock(), &mut io::sink());
    if let Some(s) = &http {
        s.shutdown();
    }
    if let Some(s) = &ws {
        s.shutdown();
    }
    coordinator.shutdown();
    stdin_result?;
    Ok(())
}

#[cfg(feature = "candle")]
fn require_model(args: &Args) -> Result<&std::path::Path> {
    args.model
        .as_deref()
        .ok_or_else(|| format!("--model is required for backend {}", args.backend).into())
}

fn run(args: &Args) -> Result<()> {
    match args.backend.as_str() {
        #[cfg(feature = "mock")]
        "mock" => serve(
            models::mock_backend(),
            models::mock_detokenize,
            models::MOCK_SESSION,
            None,
            args,
        ),
        #[cfg(feature = "candle")]
        "whisper" => {
            let dir = require_model(args)?;
            let tokenizer = args
                .tokenizer
                .clone()
                .unwrap_or_else(|| dir.join("tokenizer.json"));
            let (backend, detok) = models::load_whisper(dir, &tokenizer)?;
            serve(
                backend,
                move |t| detok(t),
                models::WHISPER_SESSION,
                Some(models::WHISPER_MAX_STREAM_SAMPLES),
                args,
            )
        }
        #[cfg(feature = "candle")]
        "qwen3-asr" => {
            let dir = require_model(args)?;
            let (backend, detok) = models::load_qwen3_asr(dir)?;
            let max = backend.config().max_audio_samples as u64;
            serve(
                backend,
                move |t| detok(t),
                models::QWEN3_ASR_SESSION,
                Some(max),
                args,
            )
        }
        other => Err(format!(
            "backend {other} is unavailable in this build (features: mock={}, candle={})",
            cfg!(feature = "mock"),
            cfg!(feature = "candle"),
        )
        .into()),
    }
}

fn main() {
    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => return,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("devl-server: {e}");
        std::process::exit(1);
    }
}
