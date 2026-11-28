//! Blocking reference client for both `devl-server` transports.
//!
//! Shared by the `stream_client` example and the end-to-end tests (included
//! with `#[path]`). It speaks the wire protocol directly: WebSocket through
//! `tungstenite`, HTTP through a hand-written chunked request and response over
//! one `TcpStream`. It respects audio credit, so it can replay a clip as fast
//! as the server accepts it.

#![allow(dead_code)]

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use devl_server::http::{REQUEST_CONTENT_TYPE, STREAM_PATH, encode_message_record};
use devl_server::wire::encode_client_message;
use devl_server::{AudioFormat, AudioFrame, ClientMessage, PROTOCOL_VERSION, encode_audio_frame};
use serde_json::Value;
use tungstenite::{Message, WebSocket};

/// How long a read may block before the client gives up.
pub const READ_TIMEOUT: Duration = Duration::from_secs(600);

/// Transport to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// `POST /v1/stream` with a chunked upload and an NDJSON response.
    Http,
    /// A WebSocket connection with JSON control frames and binary audio frames.
    Ws,
}

/// What to send and when.
#[derive(Debug, Clone)]
pub struct Replay {
    /// Samples per audio frame (capped by the server's `max_frame_samples`).
    pub frame_samples: usize,
    /// Sleep one frame duration after each frame.
    pub realtime: bool,
    /// Send `Commit` after the last frame. When false the client only waits,
    /// which is how a server-side endpoint policy ends the turn.
    pub commit: bool,
    /// Frame count to claim in `Commit` instead of the true count.
    pub commit_frames: Option<u64>,
    /// Drop the connection after this many frames without committing.
    pub abort_after_frames: Option<usize>,
}

impl Default for Replay {
    fn default() -> Self {
        Self {
            frame_samples: 8_000,
            realtime: false,
            commit: true,
            commit_frames: None,
            abort_after_frames: None,
        }
    }
}

/// Everything the server said, in order.
#[derive(Debug, Default)]
pub struct Outcome {
    /// HTTP status of the response (HTTP transport only).
    pub http_status: Option<u16>,
    /// Every JSON event received, including `started`, `credit`, and `partial`.
    pub events: Vec<Value>,
    /// True when the client dropped the connection on purpose.
    pub aborted: bool,
}

impl Outcome {
    /// The terminal event (`final`, `cancelled`, or `error`), if one arrived.
    pub fn terminal(&self) -> Option<&Value> {
        self.events
            .iter()
            .find(|e| matches!(e["type"].as_str(), Some("final" | "cancelled" | "error")))
    }

    /// Number of terminal events received. A correct server sends one.
    pub fn terminal_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e["type"].as_str(), Some("final" | "cancelled" | "error")))
            .count()
    }

    /// Partial transcripts in arrival order.
    pub fn partials(&self) -> Vec<&Value> {
        self.events
            .iter()
            .filter(|e| e["type"] == "partial")
            .collect()
    }
}

fn is_terminal(v: &Value) -> bool {
    matches!(v["type"].as_str(), Some("final" | "cancelled" | "error"))
}

enum Msg {
    Status(u16),
    Event(Value),
}

enum Link {
    Http {
        out: TcpStream,
        rx: mpsc::Receiver<Msg>,
    },
    Ws(Box<WebSocket<TcpStream>>),
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_line(r: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\r', '\n']).to_owned()))
}

/// Parse a chunked `application/x-ndjson` response and forward each line.
fn read_http_response(stream: TcpStream, tx: &mpsc::Sender<Msg>) -> io::Result<()> {
    let mut r = BufReader::new(stream);
    let status_line = read_line(&mut r)?.ok_or_else(|| bad("no response"))?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| bad(format!("bad status line {status_line:?}")))?;
    let _ = tx.send(Msg::Status(status));
    let mut chunked = false;
    let mut length = None;
    while let Some(h) = read_line(&mut r)? {
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        chunked |= lower.starts_with("transfer-encoding:") && lower.contains("chunked");
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse::<usize>().ok();
        }
    }
    let mut pending: Vec<u8> = Vec::new();
    let emit = |pending: &mut Vec<u8>| {
        while let Some(nl) = pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = pending.drain(..=nl).collect();
            if let Ok(v) = serde_json::from_slice::<Value>(&line) {
                let _ = tx.send(Msg::Event(v));
            }
        }
    };
    if chunked {
        loop {
            let size_line = read_line(&mut r)?.ok_or_else(|| bad("truncated chunked body"))?;
            let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
                .map_err(|_| bad(format!("bad chunk size {size_line:?}")))?;
            if size == 0 {
                return Ok(());
            }
            let mut data = vec![0; size];
            r.read_exact(&mut data)?;
            let mut crlf = [0; 2];
            r.read_exact(&mut crlf)?;
            pending.extend_from_slice(&data);
            emit(&mut pending);
        }
    }
    let mut data = Vec::new();
    match length {
        Some(n) => {
            data.resize(n, 0);
            r.read_exact(&mut data)?;
        }
        None => {
            r.read_to_end(&mut data)?;
        }
    }
    pending.extend_from_slice(&data);
    emit(&mut pending);
    Ok(())
}

impl Drop for Link {
    fn drop(&mut self) {
        // The response reader thread holds a second handle to the socket, so
        // dropping `out` alone would leave the connection open.
        if let Link::Http { out, .. } = self {
            let _ = out.shutdown(std::net::Shutdown::Both);
        }
    }
}

impl Link {
    fn connect(transport: Transport, addr: &str) -> io::Result<Self> {
        let tcp = TcpStream::connect(addr)?;
        tcp.set_nodelay(true)?;
        match transport {
            Transport::Http => {
                let mut out = tcp.try_clone()?;
                write!(
                    out,
                    "POST {STREAM_PATH} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: \
                     {REQUEST_CONTENT_TYPE}\r\nTransfer-Encoding: chunked\r\n\r\n"
                )?;
                let (tx, rx) = mpsc::channel();
                thread::spawn(move || {
                    let _ = read_http_response(tcp, &tx);
                });
                Ok(Link::Http { out, rx })
            }
            Transport::Ws => {
                tcp.set_read_timeout(Some(READ_TIMEOUT))?;
                let (ws, _) = tungstenite::client(format!("ws://{addr}"), tcp)
                    .map_err(|e| io::Error::other(e.to_string()))?;
                Ok(Link::Ws(Box::new(ws)))
            }
        }
    }

    fn send(&mut self, msg: &ClientMessage) -> io::Result<()> {
        match self {
            Link::Http { out, .. } => {
                let record = encode_message_record(msg);
                write!(out, "{:x}\r\n", record.len())?;
                out.write_all(&record)?;
                out.write_all(b"\r\n")?;
                out.flush()
            }
            Link::Ws(ws) => {
                let m = match msg {
                    ClientMessage::Audio(frame) => Message::binary(encode_audio_frame(frame)),
                    other => Message::text(
                        encode_client_message(other).expect("non-audio message has a JSON form"),
                    ),
                };
                ws.send(m).map_err(|e| io::Error::other(e.to_string()))
            }
        }
    }

    /// End the HTTP request body. No-op for WebSocket.
    fn end_upload(&mut self) -> io::Result<()> {
        match self {
            Link::Http { out, .. } => {
                out.write_all(b"0\r\n\r\n")?;
                out.flush()
            }
            Link::Ws(_) => Ok(()),
        }
    }

    /// Next event, or `None` when the server closed the stream.
    fn recv(&mut self, out: &mut Outcome) -> io::Result<Option<Value>> {
        match self {
            Link::Http { rx, .. } => loop {
                match rx.recv_timeout(READ_TIMEOUT) {
                    Ok(Msg::Status(s)) => out.http_status = Some(s),
                    Ok(Msg::Event(v)) => return Ok(Some(v)),
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        return Err(io::Error::new(io::ErrorKind::TimedOut, "no event"));
                    }
                }
            },
            Link::Ws(ws) => loop {
                match ws.read() {
                    Ok(Message::Text(t)) => {
                        return serde_json::from_str(t.as_str())
                            .map(Some)
                            .map_err(|e| bad(e.to_string()));
                    }
                    Ok(Message::Close(_)) => {
                        let _ = ws.flush();
                        return Ok(None);
                    }
                    Ok(_) => {}
                    Err(
                        tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed,
                    ) => {
                        return Ok(None);
                    }
                    Err(e) => return Err(io::Error::other(e.to_string())),
                }
            },
        }
    }
}

/// Stream `samples` (16 kHz mono s16) to the server and collect its events.
///
/// Returns once a terminal event arrives, the server closes the stream, or
/// the replay aborts the connection by request.
pub fn run(
    transport: Transport,
    addr: &str,
    samples: &[i16],
    replay: &Replay,
) -> io::Result<Outcome> {
    let mut link = Link::connect(transport, addr)?;
    let mut outcome = Outcome::default();
    // A server at its stream cap answers with a terminal error and may close
    // before `start` is written. Ignore the send error and read the answer.
    let _ = link.send(&ClientMessage::Start {
        version: PROTOCOL_VERSION,
        format: AudioFormat::mono_16k_s16(),
    });

    // Wait for `started`, which carries the limits and the initial credit.
    let (mut credit, max_frame) = loop {
        let Some(ev) = link.recv(&mut outcome)? else {
            return Ok(outcome);
        };
        let started = ev["type"] == "started";
        let terminal = is_terminal(&ev);
        let limits = ev["limits"].clone();
        outcome.events.push(ev);
        if started {
            break (
                limits["initial_credit_samples"].as_u64().unwrap_or(0),
                limits["max_frame_samples"].as_u64().unwrap_or(1),
            );
        }
        if terminal {
            return Ok(outcome);
        }
    };

    let frame_samples = replay
        .frame_samples
        .min(usize::try_from(max_frame).unwrap_or(usize::MAX))
        .max(1);
    let mut frames = 0u64;
    let mut finished = false;
    'upload: for chunk in samples.chunks(frame_samples) {
        let need = chunk.len() as u64;
        while credit < need {
            let Some(ev) = link.recv(&mut outcome)? else {
                finished = true;
                break 'upload;
            };
            if ev["type"] == "credit" {
                credit += ev["samples"].as_u64().unwrap_or(0);
            }
            let terminal = is_terminal(&ev);
            outcome.events.push(ev);
            if terminal {
                finished = true;
                break 'upload;
            }
        }
        if link
            .send(&ClientMessage::Audio(AudioFrame::from_s16(frames, chunk)))
            .is_err()
        {
            // The server ended the stream (for example by endpointing) and
            // closed its side. Its terminal event is still readable.
            break;
        }
        credit -= need;
        frames += 1;
        if replay.abort_after_frames == Some(frames as usize) {
            outcome.aborted = true;
            return Ok(outcome);
        }
        if replay.realtime {
            thread::sleep(Duration::from_millis(need * 1000 / 16_000));
        }
    }

    if !finished && replay.commit {
        let claimed = replay.commit_frames.unwrap_or(frames);
        // A send failure means the server already ended the stream.
        let _ = link.send(&ClientMessage::Commit { frames: claimed });
        let _ = link.end_upload();
    }
    while !finished {
        let Some(ev) = link.recv(&mut outcome)? else {
            break;
        };
        finished = is_terminal(&ev);
        outcome.events.push(ev);
    }
    if replay.abort_after_frames.is_some() {
        outcome.aborted = true;
    }
    Ok(outcome)
}

/// Read a mono 16 kHz signed 16-bit PCM WAV file, or raw little-endian s16
/// samples when the file does not start with `RIFF`.
pub fn read_pcm_s16(path: &std::path::Path) -> io::Result<Vec<i16>> {
    let bytes = std::fs::read(path)?;
    let body: &[u8] = if bytes.starts_with(b"RIFF") {
        let mut at = 12;
        let mut data = None;
        while at + 8 <= bytes.len() {
            let id = &bytes[at..at + 4];
            let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            let end = (at + 8 + len).min(bytes.len());
            let chunk = &bytes[at + 8..end];
            if id == b"fmt " && chunk.len() >= 16 {
                let tag = u16::from_le_bytes(chunk[0..2].try_into().unwrap());
                let channels = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
                let rate = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
                if (tag, channels, rate, bits) != (1, 1, 16_000, 16) {
                    return Err(bad("WAV must be mono 16 kHz signed 16-bit PCM"));
                }
            } else if id == b"data" {
                data = Some(chunk);
            }
            at += 8 + len + (len & 1);
        }
        data.ok_or_else(|| bad("WAV has no data chunk"))?
    } else {
        &bytes
    };
    Ok(body
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b))
        .collect())
}
