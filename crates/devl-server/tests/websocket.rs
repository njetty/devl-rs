//! Loopback tests for the WebSocket adapter.
//!
//! Every test runs a real `WsServer` on 127.0.0.1 port 0 over a mock-backed
//! shared `Coordinator`. Ordering comes from the protocol itself (the client
//! waits for specific events); timeouts only bound failures.

use std::net::TcpStream;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{SchedulerConfig, TokenId};
use devl_server::*;
use serde_json::{Value, json};
use tungstenite::protocol::CloseFrame;
use tungstenite::protocol::frame::coding::CloseCode;
use tungstenite::{Message, WebSocket};

const WAIT: Duration = Duration::from_secs(20);
const SAMPLE_VALUE: i16 = 100;

type Client = WebSocket<TcpStream>;
type Coord = Coordinator<MockBackend>;
type Case = (&'static str, Box<dyn Fn(&mut Client)>, &'static str);

fn text(tokens: &[TokenId]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn coordinator_config() -> CoordinatorConfig {
    CoordinatorConfig {
        workers: 2,
        scheduler: SchedulerConfig {
            max_concurrent_drafts: 2,
        },
        draft_cadence_samples: 160,
        ..CoordinatorConfig::default()
    }
}

struct Fixture {
    coord: Arc<Coord>,
    server: WsServer,
}

fn fixture_with(mock: MockBackend, cc: CoordinatorConfig, wc: WsConfig) -> Fixture {
    let coord = Arc::new(Coordinator::new(mock, text, cc).unwrap());
    let server = WsServer::bind("127.0.0.1:0", Arc::clone(&coord), wc).unwrap();
    Fixture { coord, server }
}

fn fixture(mock: MockBackend) -> Fixture {
    fixture_with(mock, coordinator_config(), WsConfig::default())
}

impl Fixture {
    fn connect(&self) -> Client {
        let stream = TcpStream::connect(self.server.local_addr()).unwrap();
        stream.set_read_timeout(Some(WAIT)).unwrap();
        let url = format!("ws://{}", self.server.local_addr());
        let (ws, _) = tungstenite::client(url, stream).unwrap();
        ws
    }

    /// Wait until the coordinator and adapter hold no streams or connections.
    fn assert_retired(&self) {
        let deadline = Instant::now() + WAIT;
        loop {
            let s = self.coord.stats();
            if s.open_streams == 0
                && s.pending_commits == 0
                && s.pending_drafts == 0
                && self.server.active_connections() == 0
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "not retired: {s:?}, connections {}",
                self.server.active_connections()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}

fn send_json(ws: &mut Client, v: Value) {
    ws.send(Message::text(v.to_string())).unwrap();
}

fn start_msg() -> Value {
    json!({"type":"start","version":{"major":1,"minor":0},
        "format":{"encoding":"s16le","sample_rate_hz":16000,"channels":1}})
}

fn send_audio(ws: &mut Client, seq: u64, n: usize) {
    let frame = AudioFrame::from_s16(seq, &vec![SAMPLE_VALUE; n]);
    ws.send(Message::binary(encode_audio_frame(&frame)))
        .unwrap();
}

/// Next JSON event, or `None` once the server closed the connection.
fn next_event(ws: &mut Client) -> Option<Value> {
    loop {
        match ws.read() {
            Ok(Message::Text(t)) => return Some(serde_json::from_str(t.as_str()).unwrap()),
            Ok(Message::Close(_)) => {
                // Flush the automatic close reply so the server can finish.
                let _ = ws.flush();
                return None;
            }
            Err(_) => return None,
            Ok(_) => {}
        }
    }
}

fn expect_type(ws: &mut Client, ty: &str) -> Value {
    loop {
        let v = next_event(ws).unwrap_or_else(|| panic!("closed while waiting for {ty}"));
        if v["type"] == ty {
            return v;
        }
        // Only advisory events may be skipped.
        assert!(
            v["type"] == "partial" || v["type"] == "credit",
            "unexpected {v} while waiting for {ty}"
        );
    }
}

fn open(ws: &mut Client) -> Value {
    send_json(ws, start_msg());
    expect_type(ws, "started")
}

/// Drain to the close handshake, returning the terminal event.
fn finish(ws: &mut Client) -> Value {
    let mut terminal = None;
    while let Some(v) = next_event(ws) {
        if matches!(v["type"].as_str(), Some("final" | "cancelled" | "error")) {
            assert!(terminal.is_none(), "second terminal event: {v}");
            terminal = Some(v);
        }
    }
    terminal.expect("connection closed without a terminal event")
}

fn expect_error(ws: &mut Client, code: &str) {
    let v = finish(ws);
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["code"], code, "{v}");
}

#[test]
fn start_stream_partial_commit_final() {
    let mock = MockBackend::with_transcripts([
        (160, vec![11, 12, MOCK_EOS_TOKEN]),
        (320, vec![11, 12, 13, MOCK_EOS_TOKEN]),
    ]);
    let expected =
        text(&mock.reference_offline_decode(&vec![f32::from(SAMPLE_VALUE) / 32768.0; 320], 448));
    let f = fixture(mock);
    let mut ws = f.connect();

    let started = open(&mut ws);
    assert_eq!(started["version"]["major"], 1);
    assert_eq!(
        started["limits"]["max_frame_samples"],
        StreamLimits::default().max_frame_samples
    );

    send_audio(&mut ws, 0, 160);
    let partial = expect_type(&mut ws, "partial");
    assert_eq!(partial["audio_samples"], 160);
    assert_eq!(partial["text"], "11 12 0");

    send_audio(&mut ws, 1, 160);
    send_json(&mut ws, json!({"type":"commit","frames":2}));
    let fin = finish(&mut ws);
    assert_eq!(fin["type"], "final");
    assert_eq!(fin["audio_samples"], 320);
    assert_eq!(fin["text"], expected);
    f.assert_retired();
}

#[test]
fn independent_connections_do_not_share_results() {
    let mock = MockBackend::with_transcripts([
        (160, vec![11, 12, 13, MOCK_EOS_TOKEN]),
        (320, vec![21, 22, MOCK_EOS_TOKEN]),
    ]);
    let f = fixture(mock);
    let mut a = f.connect();
    let mut b = f.connect();
    open(&mut a);
    open(&mut b);

    let ta = thread::spawn(move || {
        send_audio(&mut a, 0, 80);
        send_audio(&mut a, 1, 80);
        send_json(&mut a, json!({"type":"commit","frames":2}));
        finish(&mut a)
    });
    let tb = thread::spawn(move || {
        send_audio(&mut b, 0, 160);
        send_audio(&mut b, 1, 160);
        send_json(&mut b, json!({"type":"commit","frames":2}));
        finish(&mut b)
    });
    let (fa, fb) = (ta.join().unwrap(), tb.join().unwrap());
    assert_eq!(
        (fa["audio_samples"].as_u64(), fa["text"].as_str()),
        (Some(160), Some("11 12 13 0"))
    );
    assert_eq!(
        (fb["audio_samples"].as_u64(), fb["text"].as_str()),
        (Some(320), Some("21 22 0"))
    );
    f.assert_retired();
}

#[test]
fn malformed_input_gets_protocol_errors_and_cleanup() {
    let f = fixture(MockBackend::new());
    let cases: Vec<Case> = vec![
        (
            "invalid json",
            Box::new(|ws| ws.send(Message::text("{nope")).unwrap()),
            "malformed_frame",
        ),
        (
            "unknown type",
            Box::new(|ws| send_json(ws, json!({"type":"dance"}))),
            "malformed_frame",
        ),
        (
            "short binary frame",
            Box::new(|ws| {
                open(ws);
                ws.send(Message::binary(vec![1, 1, 0])).unwrap();
            }),
            "malformed_frame",
        ),
        (
            "wrong layout version",
            Box::new(|ws| {
                open(ws);
                let mut b = encode_audio_frame(&AudioFrame::from_s16(0, &[1, 2]));
                b[0] = 9;
                ws.send(Message::binary(b)).unwrap();
            }),
            "unsupported_version",
        ),
        (
            "odd payload length",
            Box::new(|ws| {
                open(ws);
                let frame = AudioFrame {
                    seq: 0,
                    payload: vec![1, 2, 3],
                };
                ws.send(Message::binary(encode_audio_frame(&frame)))
                    .unwrap();
            }),
            "malformed_frame",
        ),
        (
            "audio before start",
            Box::new(|ws| send_audio(ws, 0, 16)),
            "invalid_message_order",
        ),
        (
            "sequence gap",
            Box::new(|ws| {
                open(ws);
                send_audio(ws, 5, 16);
            }),
            "sequence_violation",
        ),
        (
            "commit frame count mismatch",
            Box::new(|ws| {
                open(ws);
                send_audio(ws, 0, 16);
                send_json(ws, json!({"type":"commit","frames":3}));
            }),
            "sequence_violation",
        ),
        (
            "unsupported sample rate",
            Box::new(|ws| {
                send_json(
                    ws,
                    json!({"type":"start","version":{"major":1,"minor":0},
                    "format":{"encoding":"s16le","sample_rate_hz":8000,"channels":1}}),
                );
            }),
            "unsupported_format",
        ),
    ];
    for (name, act, code) in cases {
        let mut ws = f.connect();
        act(&mut ws);
        let v = finish(&mut ws);
        assert_eq!(v["type"], "error", "{name}: {v}");
        assert_eq!(v["code"], code, "{name}: {v}");
        f.assert_retired();
    }
}

#[test]
fn oversized_input_is_rejected() {
    let cc = CoordinatorConfig {
        limits: StreamLimits {
            max_frame_samples: 160,
            max_stream_samples: 1_000,
            initial_credit_samples: 1_600,
            max_outstanding_credit_samples: 1_600,
        },
        ..coordinator_config()
    };
    let f = fixture_with(MockBackend::new(), cc, WsConfig::default());

    // Within the byte cap (f32 sized) but over the sample limit for s16.
    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 161);
    expect_error(&mut ws, "frame_too_large");
    f.assert_retired();

    // Beyond the message byte cap: refused by the WebSocket layer.
    let mut ws = f.connect();
    open(&mut ws);
    let cap = WsConfig::default()
        .max_text_bytes
        .max(160 * 4 + FRAME_HEADER_LEN);
    ws.send(Message::binary(vec![0u8; cap + 1])).unwrap();
    expect_error(&mut ws, "frame_too_large");
    f.assert_retired();

    // Oversized control message.
    let mut ws = f.connect();
    ws.send(Message::text(
        "x".repeat(WsConfig::default().max_text_bytes + 1),
    ))
    .unwrap();
    expect_error(&mut ws, "frame_too_large");
    f.assert_retired();

    // Exceeding the total stream length.
    let mut ws = f.connect();
    open(&mut ws);
    for seq in 0..7 {
        send_audio(&mut ws, seq, 160);
    }
    expect_error(&mut ws, "stream_too_long");
    f.assert_retired();
}

#[test]
fn credit_overrun_is_a_backpressure_violation() {
    let cc = CoordinatorConfig {
        limits: StreamLimits {
            max_frame_samples: 100,
            max_stream_samples: 100_000,
            initial_credit_samples: 100,
            max_outstanding_credit_samples: 100,
        },
        // No drafts: audio drains only via the worker, so a burst can overrun.
        draft_cadence_samples: u64::MAX,
        ..coordinator_config()
    };
    let f = fixture_with(MockBackend::new(), cc, WsConfig::default());
    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 100);
    send_audio(&mut ws, 1, 100);
    send_audio(&mut ws, 2, 100);
    // Credit is released as audio drains, so the terminal outcome is either a
    // violation or, if the server kept up, a clean commit. Both retire cleanly.
    send_json(&mut ws, json!({"type":"commit","frames":3}));
    let v = finish(&mut ws);
    assert!(matches!(v["type"].as_str(), Some("final" | "error")), "{v}");
    if v["type"] == "error" {
        assert_eq!(v["code"], "backpressure_violation");
    }
    f.assert_retired();
}

#[test]
fn cancel_and_close_retire_the_stream() {
    let f = fixture(MockBackend::with_transcripts([(
        160,
        vec![1, 2, MOCK_EOS_TOKEN],
    )]));

    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 160);
    send_json(&mut ws, json!({"type":"cancel"}));
    assert_eq!(finish(&mut ws)["type"], "cancelled");
    f.assert_retired();

    // Close while streaming acts as a cancel.
    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 160);
    send_json(&mut ws, json!({"type":"close"}));
    assert_eq!(finish(&mut ws)["type"], "cancelled");
    f.assert_retired();

    // Close before start ends the connection without events.
    let mut ws = f.connect();
    send_json(&mut ws, json!({"type":"close"}));
    assert!(next_event(&mut ws).is_none());
    f.assert_retired();
}

#[test]
fn disconnects_release_streams_in_every_phase() {
    let f = fixture(MockBackend::with_transcripts([(
        160,
        vec![1, 2, MOCK_EOS_TOKEN],
    )]));

    // Abrupt TCP drop mid-stream.
    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 160);
    drop(ws);
    f.assert_retired();

    // Drop straight after commit, before the final is read.
    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 160);
    send_json(&mut ws, json!({"type":"commit","frames":1}));
    drop(ws);
    f.assert_retired();

    // Peer close frame mid-stream.
    let mut ws = f.connect();
    open(&mut ws);
    send_audio(&mut ws, 0, 160);
    ws.close(Some(CloseFrame {
        code: CloseCode::Normal,
        reason: "".into(),
    }))
    .unwrap();
    while ws.read().is_ok() {}
    f.assert_retired();

    // Connect and vanish without a handshake.
    drop(TcpStream::connect(f.server.local_addr()).unwrap());
    f.assert_retired();
}

#[test]
fn full_coordinator_rejects_with_an_error_event() {
    let cc = CoordinatorConfig {
        max_streams: 1,
        ..coordinator_config()
    };
    let f = fixture_with(MockBackend::new(), cc, WsConfig::default());
    let mut first = f.connect();
    open(&mut first);

    let mut second = f.connect();
    expect_error(&mut second, "internal");

    // The first stream is unaffected and a freed slot is reusable.
    send_json(&mut first, json!({"type":"commit","frames":0}));
    assert_eq!(finish(&mut first)["type"], "final");
    f.assert_retired();
    let mut third = f.connect();
    open(&mut third);
    send_json(&mut third, json!({"type":"cancel"}));
    assert_eq!(finish(&mut third)["type"], "cancelled");
    f.assert_retired();
}

#[test]
fn connection_cap_refuses_extra_connections() {
    let f = fixture_with(
        MockBackend::new(),
        coordinator_config(),
        WsConfig {
            max_connections: 1,
            ..WsConfig::default()
        },
    );
    let mut first = f.connect();
    open(&mut first);

    let stream = TcpStream::connect(f.server.local_addr()).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    let url = format!("ws://{}", f.server.local_addr());
    assert!(tungstenite::client(url, stream).is_err());

    send_json(&mut first, json!({"type":"cancel"}));
    assert_eq!(finish(&mut first)["type"], "cancelled");
    f.assert_retired();
}

#[test]
fn idle_connections_are_closed() {
    let f = fixture_with(
        MockBackend::new(),
        coordinator_config(),
        WsConfig {
            idle_timeout: Duration::from_millis(100),
            ..WsConfig::default()
        },
    );
    let mut ws = f.connect();
    open(&mut ws);
    // No further input: the server closes and frees the stream.
    while ws.read().is_ok() {}
    f.assert_retired();
}

#[test]
fn shutdown_closes_connections_and_joins() {
    let f = fixture(MockBackend::new());
    let mut ws = f.connect();
    open(&mut ws);
    f.server.shutdown();
    while ws.read().is_ok() {}
    assert_eq!(f.server.active_connections(), 0);
    f.coord.shutdown();
    assert_eq!(f.coord.stats().open_streams, 0);
    // New connections are refused once stopped.
    assert!(
        TcpStream::connect(f.server.local_addr()).is_err() || {
            let s = TcpStream::connect(f.server.local_addr()).unwrap();
            s.set_read_timeout(Some(Duration::from_millis(500)))
                .unwrap();
            tungstenite::client(format!("ws://{}", f.server.local_addr()), s).is_err()
        }
    );
}
