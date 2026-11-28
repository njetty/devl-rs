//! Reference client for the `devl-server` HTTP and WebSocket transports.
//!
//! Streams a mono 16 kHz signed 16-bit WAV (or raw little-endian s16 samples)
//! and prints every server event as one JSON line. Exits nonzero unless the
//! stream ends with a `final` event.
//!
//! ```text
//! cargo run -p devl-server --example stream_client -- \
//!     --transport http --addr 127.0.0.1:8080 clip.wav
//! cargo run -p devl-server --example stream_client -- \
//!     --transport ws --addr 127.0.0.1:8081 clip.wav
//! ```
//!
//! Options: `--realtime` paces frames at real time, `--no-commit` leaves the
//! end of the turn to a server-side endpoint policy, and `--frame-ms <ms>`
//! sets the audio frame size (default 500).

#[path = "support/client.rs"]
mod client;

use std::path::PathBuf;
use std::process::ExitCode;

use client::{Replay, Transport};

const USAGE: &str = "usage: stream_client --transport <http|ws> --addr <host:port> \
[--realtime] [--no-commit] [--frame-ms <ms>] <audio.wav|audio.s16le>";

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("stream_client: {e}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let mut transport = None;
    let mut addr = None;
    let mut path: Option<PathBuf> = None;
    let mut replay = Replay::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--transport" => {
                transport = Some(match it.next().as_deref() {
                    Some("http") => Transport::Http,
                    Some("ws") => Transport::Ws,
                    other => return Err(format!("unknown transport {other:?}").into()),
                });
            }
            "--addr" => addr = it.next(),
            "--realtime" => replay.realtime = true,
            "--no-commit" => replay.commit = false,
            "--frame-ms" => {
                let ms: usize = it.next().ok_or("--frame-ms needs a value")?.parse()?;
                replay.frame_samples = ms * 16;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}").into());
            }
            other => path = Some(other.into()),
        }
    }
    let transport = transport.ok_or("--transport is required")?;
    let addr = addr.ok_or("--addr is required")?;
    let path = path.ok_or("an audio file is required")?;

    let samples = client::read_pcm_s16(&path)?;
    let outcome = client::run(transport, &addr, &samples, &replay)?;
    for event in &outcome.events {
        println!("{event}");
    }
    Ok(outcome.terminal().is_some_and(|t| t["type"] == "final"))
}
