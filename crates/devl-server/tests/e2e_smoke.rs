//! Process-level end-to-end smoke tests for the local `devl-server` binary.
//!
//! Each test starts the real binary on loopback with the deterministic mock
//! backend and drives it over TCP with the reference client that the
//! `stream_client` example uses. This covers what the in-process adapter tests
//! (`http.rs`, `websocket.rs`) do not: the binary's argument handling and
//! startup, both listeners over one shared coordinator, streaming through a
//! client that respects credit, finalization against the backend's offline
//! decode, endpointing, disconnect recovery, protocol errors, and clean
//! shutdown when stdin closes. Per-adapter edge cases (malformed framing,
//! limits, deadlines, shutdown with open streams) stay in those files.
//!
//! The mock backend's tokens are synthetic. These tests check transport and
//! protocol behavior, not speech recognition.

#![cfg(feature = "mock")]

#[path = "../examples/support/client.rs"]
mod client;
#[path = "../src/bin/devl-server/models.rs"]
mod models;

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use client::{Outcome, Replay, Transport};
use devl_core::mock::MOCK_EOS_TOKEN;

const TRANSPORTS: [Transport; 2] = [Transport::Http, Transport::Ws];
const WAIT: Duration = Duration::from_secs(30);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    http: String,
    ws: String,
}

impl Server {
    fn start(extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_devl-server"))
            .args(["--http", "127.0.0.1:0", "--ws", "127.0.0.1:0"])
            .args(extra)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn devl-server");
        let stdin = child.stdin.take();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let (mut http, mut ws) = (None, None);
        while http.is_none() || ws.is_none() {
            let line = lines
                .next()
                .expect("server exited before listening")
                .unwrap();
            if let Some(url) = line.strip_prefix("http http://") {
                http = Some(url.trim_end_matches("/v1/stream").to_owned());
            } else if let Some(url) = line.strip_prefix("ws ws://") {
                ws = Some(url.to_owned());
            }
        }
        // Keep draining stdout so the child never blocks on a full pipe.
        thread::spawn(move || lines.for_each(drop));
        Self {
            child,
            stdin,
            http: http.unwrap(),
            ws: ws.unwrap(),
        }
    }

    fn addr(&self, t: Transport) -> &str {
        match t {
            Transport::Http => &self.http,
            Transport::Ws => &self.ws,
        }
    }

    fn run(&self, t: Transport, samples: &[i16], replay: &Replay) -> Outcome {
        client::run(t, self.addr(t), samples, replay).expect("client run")
    }

    /// Close stdin and require a clean, prompt exit.
    fn stop(mut self) {
        drop(self.stdin.take());
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "server exited with {status}");
                return;
            }
            assert!(Instant::now() < deadline, "server did not stop");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn tone(n: usize) -> Vec<i16> {
    vec![100; n]
}

/// The text the mock backend's offline decode yields for `samples`.
fn offline_text(samples: &[i16]) -> String {
    let pcm: Vec<f32> = samples.iter().map(|s| f32::from(*s) / 32768.0).collect();
    let tokens =
        models::mock_backend().reference_offline_decode(&pcm, models::MOCK_SESSION.max_tokens);
    assert_eq!(tokens.last(), Some(&MOCK_EOS_TOKEN));
    models::mock_detokenize(&tokens)
}

fn assert_final(outcome: &Outcome, samples: &[i16]) {
    assert_eq!(outcome.terminal_count(), 1, "{:?}", outcome.events);
    let fin = outcome.terminal().unwrap();
    assert_eq!(fin["type"], "final", "{fin}");
    assert_eq!(fin["audio_samples"], samples.len(), "{fin}");
    assert_eq!(fin["text"], offline_text(samples), "{fin}");
    assert_eq!(outcome.events[0]["type"], "started");
    // Partials are advisory but, when present, ordered and bounded by the audio.
    let mut last_revision = None;
    for p in outcome.partials() {
        let rev = p["revision"].as_u64().unwrap();
        assert!(last_revision.is_none_or(|l| rev > l), "{p}");
        last_revision = Some(rev);
        assert!(p["audio_samples"].as_u64().unwrap() <= samples.len() as u64);
    }
}

#[test]
fn both_transports_finalize_to_offline_decode_on_one_server() {
    let server = Server::start(&["--draft-cadence-ms", "500"]);
    // 2 s lands on the mock's second checkpoint transcript, and 11 000 samples
    // does not, so both the scripted and the derived transcript are covered.
    for samples in [tone(32_000), tone(11_000)] {
        for t in TRANSPORTS {
            let outcome = server.run(t, &samples, &Replay::default());
            assert_final(&outcome, &samples);
            if t == Transport::Http {
                assert_eq!(outcome.http_status, Some(200));
            }
        }
    }
    server.stop();
}

#[test]
fn split_frames_do_not_change_the_final() {
    let server = Server::start(&["--draft-cadence-ms", "250"]);
    let samples = tone(32_000);
    for t in TRANSPORTS {
        for frame_samples in [160, 4_000, 32_000] {
            let replay = Replay {
                frame_samples,
                ..Replay::default()
            };
            assert_final(&server.run(t, &samples, &replay), &samples);
        }
    }
    server.stop();
}

#[test]
fn disconnect_releases_the_only_stream_slot() {
    let server = Server::start(&["--max-streams", "1", "--draft-cadence-ms", "500"]);
    let samples = tone(32_000);
    for t in TRANSPORTS {
        let aborted = server.run(
            t,
            &samples,
            &Replay {
                frame_samples: 4_000,
                abort_after_frames: Some(2),
                ..Replay::default()
            },
        );
        assert!(aborted.aborted);
        assert!(aborted.terminal().is_none());

        // The abandoned stream must be retired before the next one is admitted.
        let deadline = Instant::now() + WAIT;
        let outcome = loop {
            let outcome = server.run(t, &samples, &Replay::default());
            let full = outcome.terminal().is_some_and(|e| e["type"] == "final");
            if full || Instant::now() > deadline {
                break outcome;
            }
            thread::sleep(Duration::from_millis(20));
        };
        assert_final(&outcome, &samples);
    }
    server.stop();
}

#[test]
fn protocol_errors_end_one_stream_and_the_server_keeps_serving() {
    let server = Server::start(&["--max-streams", "1"]);
    let samples = tone(16_000);
    for t in TRANSPORTS {
        // Declaring the wrong frame count in `commit` is a sequence violation.
        let bad = server.run(
            t,
            &samples,
            &Replay {
                commit_frames: Some(99),
                ..Replay::default()
            },
        );
        assert_eq!(bad.terminal_count(), 1, "{:?}", bad.events);
        let err = bad.terminal().unwrap();
        assert_eq!(err["type"], "error", "{err}");
        assert_eq!(err["code"], "sequence_violation", "{err}");

        let deadline = Instant::now() + WAIT;
        let good = loop {
            let outcome = server.run(t, &samples, &Replay::default());
            if outcome.terminal().is_some_and(|e| e["type"] == "final") || Instant::now() > deadline
            {
                break outcome;
            }
            thread::sleep(Duration::from_millis(20));
        };
        assert_final(&good, &samples);
    }
    server.stop();
}

#[test]
fn server_endpointing_finalizes_without_a_client_commit() {
    // Speech then 500 ms of silence ends the turn; the client never commits.
    let server = Server::start(&[
        "--end-silence-ms",
        "500",
        "--speech-threshold",
        "0.001",
        "--draft-cadence-ms",
        "500",
    ]);
    let mut samples = tone(32_000);
    samples.extend(vec![0; 8_000]);
    for t in TRANSPORTS {
        let outcome = server.run(
            t,
            &samples,
            &Replay {
                commit: false,
                ..Replay::default()
            },
        );
        assert_final(&outcome, &samples);
    }
    server.stop();
}

#[test]
fn unavailable_backend_exits_with_an_error() {
    let status = Command::new(env!("CARGO_BIN_EXE_devl-server"))
        .args(["--backend", "does-not-exist", "--http", "127.0.0.1:0"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(1));

    let status = Command::new(env!("CARGO_BIN_EXE_devl-server"))
        .arg("--no-such-flag")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
}
