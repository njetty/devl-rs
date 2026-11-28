//! Loopback tests for the HTTP adapter.
//!
//! Every test runs a real `HttpServer` on 127.0.0.1 port 0 over a mock-backed
//! shared `Coordinator`. Clients are `hyper` (incremental request body and
//! incremental response), `curl` when installed, and a raw TCP client for
//! malformed requests. Ordering comes from the protocol itself (the client
//! waits for specific events); timeouts only bound failures.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant};

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{SchedulerConfig, TokenId};
use devl_server::http::{
    REQUEST_CONTENT_TYPE, STREAM_PATH, encode_audio_record, encode_message_record, encode_record,
};
use devl_server::*;
use hyper::body::{Body, Bytes, Frame};
use hyper::client::conn::http1 as client_http1;
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use tokio::sync::mpsc;

const WAIT: Duration = Duration::from_secs(20);
const SAMPLE_VALUE: i16 = 100;

type Coord = Coordinator<MockBackend>;

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

fn mock_two_checkpoints() -> MockBackend {
    MockBackend::with_transcripts([
        (160, vec![11, 12, MOCK_EOS_TOKEN]),
        (320, vec![11, 12, 13, MOCK_EOS_TOKEN]),
    ])
}

struct Fixture {
    coord: Arc<Coord>,
    server: HttpServer,
}

fn fixture_with(mock: MockBackend, cc: CoordinatorConfig, hc: HttpConfig) -> Fixture {
    let coord = Arc::new(Coordinator::new(mock, text, cc).unwrap());
    let server = HttpServer::bind("127.0.0.1:0", Arc::clone(&coord), hc).unwrap();
    Fixture { coord, server }
}

fn fixture(mock: MockBackend) -> Fixture {
    fixture_with(mock, coordinator_config(), HttpConfig::default())
}

impl Fixture {
    fn addr(&self) -> SocketAddr {
        self.server.local_addr()
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

fn start_msg() -> ClientMessage {
    ClientMessage::Start {
        version: PROTOCOL_VERSION,
        format: AudioFormat::mono_16k_s16(),
    }
}

fn audio(seq: u64, n: usize) -> ClientMessage {
    ClientMessage::Audio(AudioFrame::from_s16(seq, &vec![SAMPLE_VALUE; n]))
}

fn record(msg: &ClientMessage) -> Vec<u8> {
    encode_message_record(msg)
}

// ---------------------------------------------------------------------------
// hyper client: incremental request body, incremental response.
// ---------------------------------------------------------------------------

/// Request body fed from a channel, so the test decides when each chunk
/// leaves. No size hint, so hyper uses chunked transfer encoding.
struct ChanBody(mpsc::Receiver<Bytes>);

impl Body for ChanBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        self.0.poll_recv(cx).map(|c| c.map(|b| Ok(Frame::data(b))))
    }
}

struct Client {
    tx: Option<mpsc::Sender<Bytes>>,
    events: std_mpsc::Receiver<Value>,
    status: StatusCode,
}

impl Client {
    fn connect(addr: SocketAddr) -> Self {
        Self::connect_with(addr, REQUEST_CONTENT_TYPE)
    }

    fn connect_with(addr: SocketAddr, content_type: &'static str) -> Self {
        let (body_tx, body_rx) = mpsc::channel::<Bytes>(64);
        let (ev_tx, events) = std_mpsc::channel();
        let (status_tx, status_rx) = std_mpsc::channel();
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
                let (mut sender, conn) = client_http1::handshake(TokioIo::new(tcp)).await.unwrap();
                tokio::spawn(conn);
                let req = Request::post(STREAM_PATH)
                    .header("host", "localhost")
                    .header("content-type", content_type)
                    .body(ChanBody(body_rx))
                    .unwrap();
                let Ok(resp) = sender.send_request(req).await else {
                    let _ = status_tx.send(StatusCode::BAD_GATEWAY);
                    return;
                };
                let _ = status_tx.send(resp.status());
                let mut body = resp.into_body();
                let mut pending = Vec::new();
                while let Some(Ok(frame)) =
                    std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await
                {
                    let Ok(data) = frame.into_data() else {
                        continue;
                    };
                    pending.extend_from_slice(&data);
                    while let Some(nl) = pending.iter().position(|b| *b == b'\n') {
                        let line: Vec<u8> = pending.drain(..=nl).collect();
                        let v = serde_json::from_slice(&line).unwrap();
                        if ev_tx.send(v).is_err() {
                            return;
                        }
                    }
                }
            });
        });
        let status = status_rx.recv_timeout(WAIT).expect("no response head");
        Self {
            tx: Some(body_tx),
            events,
            status,
        }
    }

    fn send_bytes(&self, bytes: Vec<u8>) {
        if let Some(tx) = &self.tx {
            // A closed channel means the connection ended; tests that care
            // check the events.
            let _ = tx.blocking_send(Bytes::from(bytes));
        }
    }

    fn send(&self, msg: &ClientMessage) {
        self.send_bytes(record(msg));
    }

    /// End the request body cleanly.
    fn end(&mut self) {
        self.tx = None;
    }

    /// Next event, or `None` once the response ended.
    fn next(&self) -> Option<Value> {
        match self.events.recv_timeout(WAIT) {
            Ok(v) => Some(v),
            Err(std_mpsc::RecvTimeoutError::Disconnected) => None,
            Err(std_mpsc::RecvTimeoutError::Timeout) => panic!("timed out waiting for an event"),
        }
    }

    fn expect_type(&self, ty: &str) -> Value {
        loop {
            let v = self
                .next()
                .unwrap_or_else(|| panic!("response ended while waiting for {ty}"));
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

    fn open(&self) -> Value {
        self.send(&start_msg());
        self.expect_type("started")
    }

    /// Read to the end of the response, returning the single terminal event.
    fn finish(&self) -> Value {
        let mut terminal = None;
        while let Some(v) = self.next() {
            if matches!(v["type"].as_str(), Some("final" | "cancelled" | "error")) {
                assert!(terminal.is_none(), "second terminal event: {v}");
                terminal = Some(v);
            }
        }
        terminal.expect("response ended without a terminal event")
    }

    fn expect_error(&self, code: &str) {
        let v = self.finish();
        assert_eq!(v["type"], "error", "{v}");
        assert_eq!(v["code"], code, "{v}");
    }
}

// ---------------------------------------------------------------------------
// Acceptance: chunks in, events out, final delivery.
// ---------------------------------------------------------------------------

#[test]
fn partials_arrive_before_the_upload_ends_and_final_matches_offline() {
    let mock = mock_two_checkpoints();
    let expected =
        text(&mock.reference_offline_decode(&vec![f32::from(SAMPLE_VALUE) / 32768.0; 320], 448));
    let f = fixture(mock);
    let mut c = Client::connect(f.addr());
    assert_eq!(c.status, StatusCode::OK);

    let started = c.open();
    assert_eq!(started["version"]["major"], 1);
    assert_eq!(
        started["limits"]["max_frame_samples"],
        StreamLimits::default().max_frame_samples
    );

    // The request body is still open while the partial is delivered.
    c.send(&audio(0, 160));
    let partial = c.expect_type("partial");
    assert_eq!(partial["audio_samples"], 160);
    assert_eq!(partial["text"], "11 12 0");

    c.send(&audio(1, 160));
    c.send(&ClientMessage::Commit { frames: 2 });
    c.end();
    let fin = c.finish();
    assert_eq!(fin["type"], "final");
    assert_eq!(fin["audio_samples"], 320);
    assert_eq!(fin["text"], expected);
    f.assert_retired();
}

#[test]
fn input_split_at_every_byte_is_decoded() {
    let mock = mock_two_checkpoints();
    let expected =
        text(&mock.reference_offline_decode(&vec![f32::from(SAMPLE_VALUE) / 32768.0; 320], 448));
    let f = fixture(mock);
    let mut c = Client::connect(f.addr());
    let mut bytes = Vec::new();
    for m in [
        start_msg(),
        audio(0, 160),
        audio(1, 160),
        ClientMessage::Commit { frames: 2 },
    ] {
        bytes.extend(record(&m));
    }
    // One byte per HTTP chunk, so every record straddles many chunks.
    for b in bytes {
        c.send_bytes(vec![b]);
    }
    c.end();
    let fin = c.finish();
    assert_eq!(fin["type"], "final", "{fin}");
    assert_eq!(fin["text"], expected);
    f.assert_retired();
}

#[test]
fn partial_record_waits_for_the_rest() {
    let f = fixture(mock_two_checkpoints());
    let c = Client::connect(f.addr());
    c.open();
    let rec = record(&audio(0, 160));
    let (head, tail) = rec.split_at(rec.len() / 2);
    c.send_bytes(head.to_vec());
    // Nothing can be delivered for half a record: the next event only comes
    // after the rest arrives.
    thread::sleep(Duration::from_millis(100));
    assert!(c.events.try_recv().is_err());
    c.send_bytes(tail.to_vec());
    assert_eq!(c.expect_type("partial")["audio_samples"], 160);
    c.send(&ClientMessage::Cancel);
    assert_eq!(c.finish()["type"], "cancelled");
    f.assert_retired();
}

#[test]
fn known_length_upload_in_one_request() {
    let f = fixture(mock_two_checkpoints());
    let mut body = Vec::new();
    for m in [
        start_msg(),
        audio(0, 160),
        audio(1, 160),
        ClientMessage::Commit { frames: 2 },
    ] {
        body.extend(record(&m));
    }
    let mut s = raw_connect(f.addr());
    write!(
        s,
        "POST {STREAM_PATH} HTTP/1.1\r\nHost: x\r\nContent-Type: {REQUEST_CONTENT_TYPE}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .unwrap();
    s.write_all(&body).unwrap();
    let resp = raw_read_all(&mut s);
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert!(resp.contains("application/x-ndjson"), "{resp}");
    assert!(resp.contains(r#""type":"final""#), "{resp}");
    f.assert_retired();
}

#[test]
fn cancel_and_close_terminate_the_stream() {
    let f = fixture(mock_two_checkpoints());
    let mut c = Client::connect(f.addr());
    c.open();
    c.send(&audio(0, 160));
    c.send(&ClientMessage::Cancel);
    assert_eq!(c.finish()["type"], "cancelled");
    c.end();

    let mut c = Client::connect(f.addr());
    c.open();
    c.send(&ClientMessage::Close);
    assert_eq!(c.finish()["type"], "cancelled");
    c.end();

    let c = Client::connect(f.addr());
    c.send(&ClientMessage::Close);
    assert!(c.finish_or_end());
    f.assert_retired();
}

impl Client {
    /// `true` if the response ended without a terminal event.
    fn finish_or_end(&self) -> bool {
        while let Some(v) = self.next() {
            if matches!(v["type"].as_str(), Some("cancelled" | "final" | "error")) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Same coordinator and scheduler as other transports.
// ---------------------------------------------------------------------------

#[test]
fn http_and_websocket_share_one_coordinator() {
    let mock = mock_two_checkpoints();
    let coord = Arc::new(Coordinator::new(mock, text, coordinator_config()).unwrap());
    let http = HttpServer::bind("127.0.0.1:0", Arc::clone(&coord), HttpConfig::default()).unwrap();
    let ws = WsServer::bind("127.0.0.1:0", Arc::clone(&coord), WsConfig::default()).unwrap();

    // An HTTP stream is open and holds a coordinator slot while a WebSocket
    // stream runs; both finals come from the one worker pool.
    let mut hc = Client::connect(http.local_addr());
    hc.open();
    assert_eq!(coord.stats().open_streams, 1);

    let stream = TcpStream::connect(ws.local_addr()).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    let (mut wc, _) = tungstenite::client(format!("ws://{}", ws.local_addr()), stream).unwrap();
    wc.send(tungstenite::Message::text(
        devl_server::wire::encode_client_message(&start_msg()).unwrap(),
    ))
    .unwrap();
    loop {
        if let tungstenite::Message::Text(t) = wc.read().unwrap()
            && t.as_str().contains(r#""type":"started""#)
        {
            break;
        }
    }
    assert_eq!(coord.stats().open_streams, 2);

    hc.send(&audio(0, 160));
    hc.send(&audio(1, 160));
    hc.send(&ClientMessage::Commit { frames: 2 });
    hc.end();
    let fin = hc.finish();
    assert_eq!(fin["type"], "final");
    assert_eq!(fin["audio_samples"], 320);

    wc.send(tungstenite::Message::binary(encode_audio_frame(
        &AudioFrame::from_s16(0, &vec![SAMPLE_VALUE; 160]),
    )))
    .unwrap();
    wc.send(tungstenite::Message::text(
        devl_server::wire::encode_client_message(&ClientMessage::Commit { frames: 1 }).unwrap(),
    ))
    .unwrap();
    let mut got_final = false;
    while let Ok(m) = wc.read() {
        if let tungstenite::Message::Text(t) = m
            && t.as_str().contains(r#""type":"final""#)
        {
            got_final = true;
            break;
        }
    }
    assert!(got_final);
    drop(wc);
    drop(http);
    drop(ws);
    let deadline = Instant::now() + WAIT;
    while coord.stats().open_streams != 0 {
        assert!(Instant::now() < deadline, "streams not released");
        thread::sleep(Duration::from_millis(5));
    }
}

// ---------------------------------------------------------------------------
// Premature disconnect, EOF, and malformed input.
// ---------------------------------------------------------------------------

#[test]
fn disconnect_mid_stream_releases_the_stream() {
    let f = fixture(mock_two_checkpoints());
    // Reset after audio, never reading the response.
    let mut s = raw_stream(f.addr());
    raw_chunk(&mut s, &record(&start_msg()));
    raw_chunk(&mut s, &record(&audio(0, 160)));
    wait_open(&f, 1);
    drop(s);
    f.assert_retired();

    // Drop mid-record: half an audio record, then the socket closes.
    let mut s = raw_stream(f.addr());
    raw_chunk(&mut s, &record(&start_msg()));
    let rec = record(&audio(0, 160));
    raw_chunk(&mut s, &rec[..rec.len() / 2]);
    drop(s);
    f.assert_retired();

    // Drop mid HTTP chunk: the chunk header promises more than arrives.
    let mut s = raw_stream(f.addr());
    raw_chunk(&mut s, &record(&start_msg()));
    s.write_all(b"1000\r\nabc").unwrap();
    drop(s);
    f.assert_retired();

    // Client abandons the response of a committed stream.
    let mut c = Client::connect(f.addr());
    c.open();
    c.send(&audio(0, 160));
    c.send(&ClientMessage::Commit { frames: 1 });
    c.end();
    drop(c);
    f.assert_retired();
}

#[test]
fn clean_eof_before_commit_is_an_error() {
    let f = fixture(mock_two_checkpoints());
    let mut c = Client::connect(f.addr());
    c.open();
    c.send(&audio(0, 160));
    c.end();
    c.expect_error("invalid_message_order");
    f.assert_retired();

    let mut c = Client::connect(f.addr());
    c.end();
    c.expect_error("invalid_message_order");
    f.assert_retired();
}

#[test]
fn clean_eof_inside_a_record_is_malformed() {
    let f = fixture(mock_two_checkpoints());
    let mut c = Client::connect(f.addr());
    c.open();
    let rec = record(&audio(0, 160));
    c.send_bytes(rec[..rec.len() - 1].to_vec());
    c.end();
    c.expect_error("malformed_frame");
    f.assert_retired();
}

#[test]
fn invalid_input_gets_one_terminal_error_and_cleanup() {
    type Case = (&'static str, Vec<Vec<u8>>, &'static str);
    let mut bad_version = record(&audio(0, 2));
    bad_version[4] = 9;
    let odd_payload = encode_audio_record(&AudioFrame {
        seq: 0,
        payload: vec![1, 2, 3],
    });
    let start_format = |rate: u32| {
        encode_record(
            json!({"type":"start","version":{"major":1,"minor":0},
                "format":{"encoding":"s16le","sample_rate_hz":rate,"channels":1}})
            .to_string()
            .as_bytes(),
        )
    };
    let cases: Vec<Case> = vec![
        (
            "invalid json",
            vec![encode_record(b"{nope")],
            "malformed_frame",
        ),
        (
            "unknown type",
            vec![encode_record(br#"{"type":"dance"}"#)],
            "malformed_frame",
        ),
        (
            "zero-length record",
            vec![encode_record(b"")],
            "malformed_frame",
        ),
        (
            "short audio record",
            vec![record(&start_msg()), encode_record(&[1, 1, 0])],
            "malformed_frame",
        ),
        (
            "wrong layout version",
            vec![record(&start_msg()), bad_version],
            "unsupported_version",
        ),
        (
            "odd payload length",
            vec![record(&start_msg()), odd_payload],
            "malformed_frame",
        ),
        (
            "audio before start",
            vec![record(&audio(0, 16))],
            "invalid_message_order",
        ),
        (
            "second start",
            vec![record(&start_msg()), record(&start_msg())],
            "invalid_message_order",
        ),
        (
            "sequence gap",
            vec![record(&start_msg()), record(&audio(5, 16))],
            "sequence_violation",
        ),
        (
            "duplicate sequence",
            vec![
                record(&start_msg()),
                record(&audio(0, 16)),
                record(&audio(0, 16)),
            ],
            "sequence_violation",
        ),
        (
            "commit count mismatch",
            vec![
                record(&start_msg()),
                record(&audio(0, 16)),
                record(&ClientMessage::Commit { frames: 3 }),
            ],
            "sequence_violation",
        ),
        (
            "unsupported sample rate",
            vec![start_format(8000)],
            "unsupported_format",
        ),
    ];
    let f = fixture(MockBackend::new());
    for (name, records, code) in cases {
        let mut c = Client::connect(f.addr());
        for r in records {
            c.send_bytes(r);
        }
        c.end();
        let v = c.finish();
        assert_eq!(v["type"], "error", "{name}: {v}");
        assert_eq!(v["code"], code, "{name}: {v}");
        f.assert_retired();
    }
}

#[test]
fn bytes_after_a_terminal_error_are_discarded() {
    let f = fixture(MockBackend::new());
    let mut c = Client::connect(f.addr());
    c.send(&audio(0, 16)); // before start: terminal error
    for seq in 0..50 {
        c.send(&audio(seq, 16));
    }
    c.end();
    c.expect_error("invalid_message_order");
    f.assert_retired();
}

// ---------------------------------------------------------------------------
// Limits.
// ---------------------------------------------------------------------------

fn small_limits() -> StreamLimits {
    StreamLimits {
        max_frame_samples: 160,
        max_stream_samples: 400,
        initial_credit_samples: 1_600,
        max_outstanding_credit_samples: 1_600,
    }
}

#[test]
fn oversized_input_is_rejected() {
    let cc = CoordinatorConfig {
        limits: small_limits(),
        ..coordinator_config()
    };
    let f = fixture_with(MockBackend::new(), cc, HttpConfig::default());

    // The length prefix alone is refused; the payload is never sent.
    let c = Client::connect(f.addr());
    c.open();
    c.send_bytes(u32::MAX.to_le_bytes().to_vec());
    c.expect_error("frame_too_large");
    f.assert_retired();

    // Control record over the text cap.
    let c = Client::connect(f.addr());
    let mut big = br#"{"type":"close","pad":""#.to_vec();
    big.extend(vec![b'a'; 20 * 1024]);
    big.extend(br#""}"#);
    c.send_bytes(encode_record(&big));
    c.expect_error("frame_too_large");
    f.assert_retired();

    // Within the byte cap (f32 sized) but over the sample limit for s16.
    let c = Client::connect(f.addr());
    c.open();
    c.send(&audio(0, 161));
    c.expect_error("frame_too_large");
    f.assert_retired();

    // Total stream samples.
    let c = Client::connect(f.addr());
    c.open();
    c.send(&audio(0, 160));
    c.send(&audio(1, 160));
    c.send(&audio(2, 160));
    c.expect_error("stream_too_long");
    f.assert_retired();
}

#[test]
fn total_upload_size_is_capped() {
    let cc = CoordinatorConfig {
        limits: small_limits(),
        ..coordinator_config()
    };
    let hc = HttpConfig {
        max_body_bytes: 256,
        ..HttpConfig::default()
    };
    let f = fixture_with(MockBackend::new(), cc, hc);
    let c = Client::connect(f.addr());
    c.open();
    c.send_bytes(vec![0u8; 512]);
    c.expect_error("stream_too_long");
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
        draft_cadence_samples: 100,
        ..coordinator_config()
    };
    let (entered_tx, entered_rx) = std_mpsc::sync_channel(1);
    let (release_tx, release_rx) = std_mpsc::channel::<()>();
    let release_rx = std::sync::Mutex::new(release_rx);
    let mock = MockBackend::with_transcripts([(100, vec![11, 12, MOCK_EOS_TOKEN])]).with_step_hook(
        move |_| {
            let _ = entered_tx.try_send(());
            let _ = release_rx.lock().unwrap().recv_timeout(WAIT);
        },
    );
    let f = fixture_with(mock, cc, HttpConfig::default());
    let c = Client::connect(f.addr());
    c.open();
    c.send(&audio(0, 100));
    entered_rx.recv_timeout(WAIT).expect("draft did not start");
    assert_eq!(c.expect_type("credit")["samples"], 100);
    // The draft holds the session, so the pending frame cannot restore credit.
    c.send(&audio(1, 100));
    c.send(&audio(2, 1));
    c.expect_error("backpressure_violation");
    drop(release_tx);
    f.assert_retired();
}

#[test]
fn slow_request_head_is_cut_off_at_an_absolute_deadline() {
    let hc = HttpConfig {
        header_timeout: Duration::from_millis(300),
        ..HttpConfig::default()
    };
    let f = fixture_with(MockBackend::new(), coordinator_config(), hc);
    let mut s = raw_connect(f.addr());
    let started = Instant::now();
    // Dribble a head that never completes; each byte alone would be "alive".
    for b in b"POST /v1/stream HTTP/1.1\r\nHost: x\r\n" {
        if s.write_all(&[*b]).is_err() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
        if started.elapsed() > WAIT {
            panic!("head deadline not enforced");
        }
    }
    let mut sink = Vec::new();
    let _ = s.read_to_end(&mut sink);
    assert!(started.elapsed() < Duration::from_secs(5));
    f.assert_retired();
}

#[test]
fn idle_client_is_failed() {
    let hc = HttpConfig {
        idle_timeout: Duration::from_millis(150),
        ..HttpConfig::default()
    };
    let f = fixture_with(mock_two_checkpoints(), coordinator_config(), hc);
    let c = Client::connect(f.addr());
    c.open();
    c.expect_error("internal");
    f.assert_retired();
}

#[test]
fn stream_duration_is_an_absolute_deadline() {
    let hc = HttpConfig {
        max_stream_duration: Duration::from_millis(400),
        idle_timeout: Duration::from_secs(60),
        ..HttpConfig::default()
    };
    let f = fixture_with(mock_two_checkpoints(), coordinator_config(), hc);
    let c = Client::connect(f.addr());
    c.open();
    // Keep the connection busy so idle never fires; the deadline must.
    let tx = c.tx.clone().unwrap();
    let feeder = thread::spawn(move || {
        for seq in 0..100 {
            if tx
                .blocking_send(Bytes::from(record(&audio(seq, 1))))
                .is_err()
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
    });
    let started = Instant::now();
    c.expect_error("internal");
    assert!(started.elapsed() < Duration::from_secs(10));
    drop(c);
    feeder.join().unwrap();
    f.assert_retired();
}

#[test]
fn connections_beyond_the_cap_are_closed() {
    let hc = HttpConfig {
        max_connections: 1,
        ..HttpConfig::default()
    };
    let f = fixture_with(mock_two_checkpoints(), coordinator_config(), hc);
    let held = Client::connect(f.addr());
    held.open();
    let mut s = raw_connect(f.addr());
    let _ = write!(
        s,
        "POST {STREAM_PATH} HTTP/1.1\r\nHost: x\r\nContent-Type: {REQUEST_CONTENT_TYPE}\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n"
    );
    let mut sink = Vec::new();
    let _ = s.read_to_end(&mut sink);
    assert!(sink.is_empty(), "over-cap connection was served");
    drop(held);
    f.assert_retired();
}

#[test]
fn stream_cap_is_reported_as_unavailable() {
    let cc = CoordinatorConfig {
        max_streams: 1,
        ..coordinator_config()
    };
    let f = fixture_with(mock_two_checkpoints(), cc, HttpConfig::default());
    let held = f.coord.open_stream().unwrap();
    let c = Client::connect(f.addr());
    assert_eq!(c.status, StatusCode::SERVICE_UNAVAILABLE);
    c.expect_error("internal");
    drop(held);
    f.assert_retired();
}

#[test]
fn shutdown_ends_open_streams() {
    let hc = HttpConfig {
        shutdown_grace: Duration::from_millis(500),
        ..HttpConfig::default()
    };
    let f = fixture_with(mock_two_checkpoints(), coordinator_config(), hc);
    let c = Client::connect(f.addr());
    c.open();
    f.server.shutdown();
    let mut terminal = None;
    while let Ok(v) = c.events.recv_timeout(WAIT) {
        if v["type"] == "error" {
            terminal = Some(v);
        }
    }
    if let Some(v) = terminal {
        assert_eq!(v["code"], "internal");
    }
    f.assert_retired();
}

// ---------------------------------------------------------------------------
// HTTP-level request validation (raw client).
// ---------------------------------------------------------------------------

fn raw_connect(addr: SocketAddr) -> TcpStream {
    let s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(WAIT)).unwrap();
    s
}

fn raw_read_all(s: &mut TcpStream) -> String {
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    String::from_utf8_lossy(&out).into_owned()
}

/// A raw chunked stream request with the head already sent.
fn raw_stream(addr: SocketAddr) -> TcpStream {
    let mut s = raw_connect(addr);
    write!(
        s,
        "POST {STREAM_PATH} HTTP/1.1\r\nHost: x\r\nContent-Type: {REQUEST_CONTENT_TYPE}\r\n\
         Transfer-Encoding: chunked\r\n\r\n"
    )
    .unwrap();
    s
}

fn raw_chunk(s: &mut TcpStream, bytes: &[u8]) {
    write!(s, "{:x}\r\n", bytes.len()).unwrap();
    s.write_all(bytes).unwrap();
    s.write_all(b"\r\n").unwrap();
    s.flush().unwrap();
}

fn wait_open(f: &Fixture, n: usize) {
    let deadline = Instant::now() + WAIT;
    while f.coord.stats().open_streams != n {
        assert!(Instant::now() < deadline, "stream never opened");
        thread::sleep(Duration::from_millis(5));
    }
}

fn raw_request(f: &Fixture, head: &str) -> String {
    let mut s = raw_connect(f.addr());
    s.write_all(head.as_bytes()).unwrap();
    raw_read_all(&mut s)
}

#[test]
fn request_validation() {
    let f = fixture(MockBackend::new());

    let r = raw_request(
        &f,
        "POST /nope HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );
    assert!(r.starts_with("HTTP/1.1 404"), "{r}");

    let r = raw_request(
        &f,
        "GET /v1/stream HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
    );
    assert!(r.starts_with("HTTP/1.1 405"), "{r}");
    assert!(r.to_ascii_lowercase().contains("allow: post"), "{r}");

    let r = raw_request(
        &f,
        "POST /v1/stream HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n",
    );
    assert!(r.starts_with("HTTP/1.1 415"), "{r}");

    let r = raw_request(&f, "POST /v1/stream HTTP/1.0\r\nContent-Length: 0\r\n\r\n");
    assert!(
        r.starts_with("HTTP/1.0 505") || r.starts_with("HTTP/1.1 505"),
        "{r}"
    );

    let r = raw_request(&f, "this is not http\r\n\r\n");
    assert!(r.is_empty() || r.starts_with("HTTP/1.1 400"), "{r}");
    f.assert_retired();
}

// ---------------------------------------------------------------------------
// curl, a standard HTTP client, as the real incremental client.
// ---------------------------------------------------------------------------

fn curl_available() -> bool {
    Command::new("curl")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// curl uploads `-T -` with blocking reads of its standard input, so it
/// surfaces response lines whenever it next wakes for input, not while idle.
/// The test therefore keeps feeding tiny valid frames until the partial shows
/// up, and sends the commit only after that: the partial is observed while
/// the upload is still open.
#[test]
fn curl_streams_chunks_and_sees_partial_before_final() {
    if !curl_available() {
        eprintln!("curl not installed; skipping");
        return;
    }
    let mock = mock_two_checkpoints();
    let f = fixture(mock.clone());
    let url = format!("http://{}{STREAM_PATH}", f.addr());
    let mut child = Command::new("curl")
        .args(["-sS", "-N", "--no-buffer", "-X", "POST", "-T", "-"])
        .args(["-H", "Transfer-Encoding: chunked", "-H", "Expect:"])
        .args(["-H", &format!("Content-Type: {REQUEST_CONTENT_TYPE}")])
        .arg(&url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, lines) = std_mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { return };
            if tx
                .send(serde_json::from_str::<Value>(&line).unwrap())
                .is_err()
            {
                return;
            }
        }
    });

    let mut write = |msg: &ClientMessage| {
        stdin.write_all(&record(msg)).unwrap();
        stdin.flush().unwrap();
    };
    write(&start_msg());
    write(&audio(0, 160));
    let mut seq = 1;
    let mut seen_started = false;
    let deadline = Instant::now() + WAIT;
    loop {
        assert!(Instant::now() < deadline, "no partial from curl");
        match lines.recv_timeout(Duration::from_millis(50)) {
            Ok(v) => match v["type"].as_str() {
                Some("started") => seen_started = true,
                Some("partial") => {
                    assert!(seen_started);
                    break;
                }
                Some("credit") => {}
                _ => panic!("unexpected {v}"),
            },
            Err(_) => {
                write(&audio(seq, 1));
                seq += 1;
            }
        }
    }
    // The upload is still open: the final cannot have been produced yet.
    assert!(child.try_wait().unwrap().is_none());
    assert_eq!(f.coord.stats().open_streams, 1);

    write(&ClientMessage::Commit { frames: seq });
    // End the upload: curl then wakes from its stdin read and surfaces the
    // final it already received.
    drop(stdin);
    let fin = loop {
        let v = lines.recv_timeout(WAIT).expect("curl produced no final");
        if v["type"] == "final" {
            break v;
        }
        assert!(v["type"] == "partial" || v["type"] == "credit", "{v}");
    };
    let total = 160 + usize::try_from(seq - 1).unwrap();
    let expected =
        text(&mock.reference_offline_decode(&vec![f32::from(SAMPLE_VALUE) / 32768.0; total], 448));
    assert_eq!(fin["audio_samples"], total);
    assert_eq!(fin["text"], expected);
    assert!(child.wait().unwrap().success());
    f.assert_retired();
}

#[test]
fn curl_disconnect_mid_upload_releases_the_stream() {
    if !curl_available() {
        eprintln!("curl not installed; skipping");
        return;
    }
    let f = fixture(mock_two_checkpoints());
    let url = format!("http://{}{STREAM_PATH}", f.addr());
    let mut child = Command::new("curl")
        .args(["-sS", "-N", "-X", "POST", "-T", "-"])
        .args(["-H", "Transfer-Encoding: chunked", "-H", "Expect:"])
        .args(["-H", &format!("Content-Type: {REQUEST_CONTENT_TYPE}")])
        .arg(&url)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(&record(&start_msg())).unwrap();
    stdin.write_all(&record(&audio(0, 160))).unwrap();
    stdin.flush().unwrap();
    wait_open(&f, 1);
    child.kill().unwrap();
    let _ = child.wait();
    f.assert_retired();
}
