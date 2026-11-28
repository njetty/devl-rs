//! HTTP/1.1 streaming adapter over the shared [`Coordinator`].
//!
//! One `POST /v1/stream` request carries one stream (one utterance). The
//! request body is the upload: a sequence of length-prefixed records (see
//! [`RecordDecoder`]), sent with `Transfer-Encoding: chunked` or a known
//! `Content-Length`. The response is the event stream: status 200 with a
//! chunked `application/x-ndjson` body, one JSON event per line in the same
//! encoding as the WebSocket adapter ([`crate::wire`]). The server starts
//! the response before the request body ends, so a client that reads and
//! writes concurrently sees `started`, `credit`, and `partial` events while
//! it is still uploading, and `final` after its last record.
//!
//! The adapter owns the transport only. HTTP parsing, chunked decoding, and
//! connection handling are done by `hyper`; every contract, scheduler, and
//! credit decision is made by the [`Coordinator`] through a
//! [`StreamSession`], exactly as for the WebSocket adapter.
//!
//! # Threading
//!
//! The adapter owns a small private tokio runtime. The coordinator API is
//! synchronous and non-blocking for the calls used here (`send` and
//! `try_recv_event`), so connection tasks call it directly and poll for
//! events every [`HttpConfig::poll_interval`].
//!
//! # Bounds
//!
//! * [`HttpConfig::max_connections`] caps simultaneous connections; extra TCP
//!   connections are closed immediately.
//! * [`HttpConfig::header_timeout`] is an absolute deadline for the request
//!   head, so a peer sending headers slowly cannot hold a slot.
//! * A record is at most the larger of one maximal audio frame and
//!   [`HttpConfig::max_text_bytes`]; the length prefix is checked before the
//!   payload is buffered. [`HttpConfig::max_body_bytes`] caps the whole
//!   upload.
//! * [`HttpConfig::max_stream_duration`] is an absolute deadline from the
//!   start of the stream, [`HttpConfig::idle_timeout`] bounds inbound silence
//!   (not applied while committing), and [`HttpConfig::write_timeout`] bounds
//!   how long an event may wait for a client that is not reading.
//! * After a terminal event the adapter keeps discarding upload bytes for at
//!   most [`HttpConfig::drain_timeout`] and [`HttpConfig::drain_bytes`], so a
//!   client still sending receives the response rather than a reset.
//!
//! # Teardown
//!
//! Every exit path (terminal event, `close`, input error, disconnect,
//! deadline, shutdown) ends by dropping the `StreamSession`, which releases
//! pending audio, scheduler jobs, and the session. A transport failure
//! (reset, truncated chunked body) releases the stream without events.

use std::convert::Infallible;
use std::future::poll_fn;
use std::io;
use std::net::{SocketAddr, TcpListener as StdTcpListener, ToSocketAddrs};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use devl_core::DevlBackend;
use hyper::body::{Body, Bytes, Frame, Incoming};
use hyper::header::{ALLOW, CONTENT_TYPE};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, Version};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, MissedTickBehavior};

use crate::coordinator::{Coordinator, CoordinatorError, StreamSession};
use crate::error::{ErrorCode, StreamError};
use crate::frame::{FRAME_HEADER_LEN, decode_audio_frame, encode_audio_frame};
use crate::message::{AudioFrame, ClientMessage, ServerEvent};
use crate::state::StreamPhase;
use crate::wire::{encode_client_message, encode_server_event, parse_client_message};

/// Path of the streaming endpoint.
pub const STREAM_PATH: &str = "/v1/stream";
/// Required request `Content-Type`.
pub const REQUEST_CONTENT_TYPE: &str = "application/vnd.devl.stream";
/// Response `Content-Type` of the event stream.
pub const RESPONSE_CONTENT_TYPE: &str = "application/x-ndjson";
/// Length of the little-endian `u32` prefix of every upload record.
pub const RECORD_PREFIX_LEN: usize = 4;

/// Frame one record: a little-endian `u32` length followed by `payload`.
///
/// # Panics
///
/// Panics if `payload` is longer than `u32::MAX` bytes.
pub fn encode_record(payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).expect("record longer than u32::MAX");
    let mut out = Vec::with_capacity(RECORD_PREFIX_LEN + payload.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Encode a client message as one upload record.
///
/// Control messages are JSON records and audio frames are binary envelope
/// records (see [`crate::frame`]). The two are told apart by the first
/// payload byte: `{` for JSON, the protocol major for audio.
pub fn encode_message_record(msg: &ClientMessage) -> Vec<u8> {
    match msg {
        ClientMessage::Audio(frame) => encode_audio_record(frame),
        other => encode_record(
            encode_client_message(other)
                .expect("non-audio message has a JSON form")
                .as_bytes(),
        ),
    }
}

/// Encode one audio frame as an upload record.
pub fn encode_audio_record(frame: &AudioFrame) -> Vec<u8> {
    encode_record(&encode_audio_frame(frame))
}

/// Decode one record payload into a client message.
///
/// A payload starting with `{` is a JSON control message of at most
/// `max_text_bytes`; anything else is an audio envelope. An empty payload is
/// malformed.
pub fn decode_record(payload: &[u8], max_text_bytes: usize) -> Result<ClientMessage, StreamError> {
    match payload.first() {
        None => Err(StreamError::new(ErrorCode::MalformedFrame, "empty record")),
        Some(b'{') => {
            if payload.len() > max_text_bytes {
                return Err(StreamError::new(
                    ErrorCode::FrameTooLarge,
                    "control record exceeds the size limit",
                ));
            }
            let text = std::str::from_utf8(payload).map_err(|_| {
                StreamError::new(ErrorCode::MalformedFrame, "control record is not UTF-8")
            })?;
            parse_client_message(text)
        }
        Some(_) => decode_audio_frame(payload).map(ClientMessage::Audio),
    }
}

/// Incremental decoder for the upload record framing.
///
/// Bytes are pushed as they arrive, in any split, and complete records are
/// pulled out. The length prefix is validated as soon as its four bytes are
/// present, before any payload is buffered, so memory is bounded by the
/// record limit plus one pushed chunk.
#[derive(Debug)]
pub struct RecordDecoder {
    buf: Vec<u8>,
    pos: usize,
    max_record_bytes: usize,
}

impl RecordDecoder {
    /// A decoder that refuses records longer than `max_record_bytes`.
    pub fn new(max_record_bytes: usize) -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            max_record_bytes,
        }
    }

    /// Append received bytes.
    pub fn push(&mut self, bytes: &[u8]) {
        if self.pos > 0 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        self.buf.extend_from_slice(bytes);
    }

    /// The next complete record payload, or `None` if more bytes are needed.
    ///
    /// A zero length or a length over the limit is an error; the decoder is
    /// not usable after an error.
    pub fn next_record(&mut self) -> Result<Option<Vec<u8>>, StreamError> {
        let avail = &self.buf[self.pos..];
        let Some(prefix) = avail.first_chunk::<RECORD_PREFIX_LEN>() else {
            return Ok(None);
        };
        let len = u32::from_le_bytes(*prefix) as usize;
        if len == 0 {
            return Err(StreamError::new(
                ErrorCode::MalformedFrame,
                "zero-length record",
            ));
        }
        if len > self.max_record_bytes {
            return Err(StreamError::new(
                ErrorCode::FrameTooLarge,
                format!(
                    "record of {len} bytes exceeds the limit of {}",
                    self.max_record_bytes
                ),
            ));
        }
        let end = RECORD_PREFIX_LEN + len;
        if avail.len() < end {
            return Ok(None);
        }
        let record = avail[RECORD_PREFIX_LEN..end].to_vec();
        self.pos += end;
        Ok(Some(record))
    }

    /// `true` if buffered bytes do not form a whole record, so an end of
    /// input now would truncate one.
    pub fn has_partial(&self) -> bool {
        self.pos < self.buf.len()
    }
}

/// Adapter settings.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Maximum simultaneous connections, including those sending a request head.
    pub max_connections: usize,
    /// Maximum size of one control (JSON) record in bytes.
    pub max_text_bytes: usize,
    /// Maximum total upload size in bytes.
    pub max_body_bytes: usize,
    /// Cadence at which a stream's queued events are delivered.
    pub poll_interval: Duration,
    /// Absolute time allowed to receive the request head, from the first
    /// read on the connection.
    pub header_timeout: Duration,
    /// Absolute time from the start of a stream to its forced end.
    pub max_stream_duration: Duration,
    /// Inbound silence after which a stream is failed. Not applied while the
    /// stream is committing.
    pub idle_timeout: Duration,
    /// Time one event may wait for a client that is not reading before the
    /// client is dropped. A per-event bound, not a deadline for the stream.
    pub write_timeout: Duration,
    /// Time to keep discarding upload bytes after a terminal event.
    pub drain_timeout: Duration,
    /// Upload bytes to discard after a terminal event at most.
    pub drain_bytes: usize,
    /// Time allowed for connections to finish on shutdown before they are
    /// aborted.
    pub shutdown_grace: Duration,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            max_connections: 64,
            max_text_bytes: 16 * 1024,
            max_body_bytes: 128 * 1024 * 1024,
            poll_interval: Duration::from_millis(10),
            header_timeout: Duration::from_secs(5),
            max_stream_duration: Duration::from_secs(15 * 60),
            idle_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(10),
            drain_timeout: Duration::from_secs(1),
            drain_bytes: 1024 * 1024,
            shutdown_grace: Duration::from_secs(2),
        }
    }
}

struct Shared {
    config: HttpConfig,
    max_record_bytes: usize,
    active: AtomicUsize,
}

/// Decrements the active-connection count on every exit path.
struct ActiveGuard(Arc<Shared>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Ctx<B: DevlBackend + Clone + Send + Sync + 'static>
where
    B::AudioEncoding: Send,
{
    shared: Arc<Shared>,
    coordinator: Arc<Coordinator<B>>,
    stop: watch::Receiver<bool>,
}

/// A running HTTP listener bound to a [`Coordinator`].
///
/// Dropping the server shuts it down.
pub struct HttpServer {
    addr: SocketAddr,
    shared: Arc<Shared>,
    stop: watch::Sender<bool>,
    runtime: Mutex<Option<tokio::runtime::Runtime>>,
}

impl HttpServer {
    /// Bind `addr` and start accepting connections.
    ///
    /// The coordinator is shared: every request opens its stream on it.
    pub fn bind<B>(
        addr: impl ToSocketAddrs,
        coordinator: Arc<Coordinator<B>>,
        config: HttpConfig,
    ) -> io::Result<Self>
    where
        B: DevlBackend + Clone + Send + Sync + 'static,
        B::AudioEncoding: Send,
    {
        let listener = StdTcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let audio_bytes = usize::try_from(coordinator.limits().max_frame_samples)
            .unwrap_or(usize::MAX)
            .saturating_mul(4)
            .saturating_add(FRAME_HEADER_LEN);
        let shared = Arc::new(Shared {
            max_record_bytes: audio_bytes.max(config.max_text_bytes),
            config,
            active: AtomicUsize::new(0),
        });
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("devl-http")
            .enable_all()
            .build()?;
        let (stop, stop_rx) = watch::channel(false);
        let ctx = Arc::new(Ctx {
            shared: Arc::clone(&shared),
            coordinator,
            stop: stop_rx,
        });
        runtime.spawn(accept_loop(listener, ctx));
        Ok(Self {
            addr,
            shared,
            stop,
            runtime: Mutex::new(Some(runtime)),
        })
    }

    /// Address the listener is bound to (useful with port 0).
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Connections currently open, including those sending a request head.
    pub fn active_connections(&self) -> usize {
        self.shared.active.load(Ordering::SeqCst)
    }

    /// Stop accepting, end every stream, and stop the adapter's threads.
    ///
    /// Open streams fail with `internal` and are released. Connections that
    /// have not finished within [`HttpConfig::shutdown_grace`] are aborted.
    /// The coordinator is not shut down; its owner does that.
    pub fn shutdown(&self) {
        let _ = self.stop.send(true);
        let rt = self
            .runtime
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(rt) = rt {
            rt.shutdown_timeout(self.shared.config.shutdown_grace);
        }
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn accept_loop<B>(std_listener: StdTcpListener, ctx: Arc<Ctx<B>>)
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let Ok(listener) = TcpListener::from_std(std_listener) else {
        return;
    };
    let mut stop = ctx.stop.clone();
    loop {
        tokio::select! {
            _ = stop.changed() => return,
            accepted = listener.accept() => {
                let Ok((tcp, _)) = accepted else {
                    tokio::time::sleep(ctx.shared.config.poll_interval).await;
                    continue;
                };
                if ctx.shared.active.fetch_add(1, Ordering::SeqCst)
                    >= ctx.shared.config.max_connections
                {
                    ctx.shared.active.fetch_sub(1, Ordering::SeqCst);
                    drop(tcp);
                    continue;
                }
                let guard = ActiveGuard(Arc::clone(&ctx.shared));
                let ctx = Arc::clone(&ctx);
                tokio::spawn(async move {
                    let _guard = guard;
                    serve_connection(tcp, ctx).await;
                });
            }
        }
    }
}

async fn serve_connection<B>(tcp: tokio::net::TcpStream, ctx: Arc<Ctx<B>>)
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let _ = tcp.set_nodelay(true);
    let cfg = &ctx.shared.config;
    // The stream task enforces `max_stream_duration` itself and reports it;
    // this is the backstop for everything else on the connection.
    let limit =
        cfg.header_timeout + cfg.max_stream_duration + cfg.drain_timeout + cfg.write_timeout;
    let service = {
        let ctx = Arc::clone(&ctx);
        service_fn(move |req| handle(req, Arc::clone(&ctx)))
    };
    let conn = http1::Builder::new()
        .timer(TokioTimer::new())
        .header_read_timeout(cfg.header_timeout)
        .keep_alive(false)
        .serve_connection(TokioIo::new(tcp), service);
    tokio::pin!(conn);
    let mut stop = ctx.stop.clone();
    tokio::select! {
        _ = &mut conn => {}
        _ = tokio::time::sleep(limit) => {}
        () = async {
            let _ = stop.changed().await;
            // Let stream tasks report the shutdown before the socket closes.
            tokio::time::sleep(cfg.write_timeout.min(Duration::from_secs(1))).await;
        } => {}
    }
}

/// Response body: an optional first chunk, then chunks from a channel until
/// its senders are dropped.
struct EventBody {
    first: Option<Bytes>,
    rx: Option<mpsc::Receiver<Bytes>>,
}

impl Body for EventBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        if let Some(first) = self.first.take() {
            return Poll::Ready(Some(Ok(Frame::data(first))));
        }
        match self.rx.as_mut() {
            None => Poll::Ready(None),
            Some(rx) => rx
                .poll_recv(cx)
                .map(|chunk| chunk.map(|b| Ok(Frame::data(b)))),
        }
    }
}

fn reject_response(status: StatusCode, code: ErrorCode, message: &str) -> Response<EventBody> {
    let event = ServerEvent::Error(StreamError::new(code, message));
    let mut line = encode_server_event(&event);
    line.push('\n');
    let mut resp = Response::new(EventBody {
        first: Some(Bytes::from(line)),
        rx: None,
    });
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        CONTENT_TYPE,
        hyper::header::HeaderValue::from_static(RESPONSE_CONTENT_TYPE),
    );
    resp
}

async fn handle<B>(
    req: Request<Incoming>,
    ctx: Arc<Ctx<B>>,
) -> Result<Response<EventBody>, Infallible>
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    if req.version() != Version::HTTP_11 {
        return Ok(reject_response(
            StatusCode::HTTP_VERSION_NOT_SUPPORTED,
            ErrorCode::MalformedFrame,
            "HTTP/1.1 is required",
        ));
    }
    if req.uri().path() != STREAM_PATH {
        return Ok(reject_response(
            StatusCode::NOT_FOUND,
            ErrorCode::MalformedFrame,
            "unknown path",
        ));
    }
    if req.method() != Method::POST {
        let mut resp = reject_response(
            StatusCode::METHOD_NOT_ALLOWED,
            ErrorCode::MalformedFrame,
            "use POST",
        );
        resp.headers_mut()
            .insert(ALLOW, hyper::header::HeaderValue::from_static("POST"));
        return Ok(resp);
    }
    let content_type = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        });
    if content_type.as_deref() != Some(REQUEST_CONTENT_TYPE) {
        return Ok(reject_response(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ErrorCode::MalformedFrame,
            "Content-Type must be application/vnd.devl.stream",
        ));
    }
    let session = match ctx.coordinator.open_stream() {
        Ok(s) => s,
        Err(e @ (CoordinatorError::TooManyStreams { .. } | CoordinatorError::ShutDown)) => {
            return Ok(reject_response(
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorCode::Internal,
                &e.to_string(),
            ));
        }
        Err(e) => {
            return Ok(reject_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                ErrorCode::Internal,
                &e.to_string(),
            ));
        }
    };
    let (tx, rx) = mpsc::channel(8);
    let body = req.into_body();
    tokio::spawn(run_stream(body, session, tx, ctx));
    let mut resp = Response::new(EventBody {
        first: None,
        rx: Some(rx),
    });
    resp.headers_mut().insert(
        CONTENT_TYPE,
        hyper::header::HeaderValue::from_static(RESPONSE_CONTENT_TYPE),
    );
    Ok(resp)
}

/// The client is gone or stopped reading; nothing more can be sent.
struct Gone;

struct Run<B: DevlBackend + Clone + Send + Sync + 'static>
where
    B::AudioEncoding: Send,
{
    session: StreamSession<B>,
    tx: mpsc::Sender<Bytes>,
    write_timeout: Duration,
    terminal_sent: bool,
}

impl<B> Run<B>
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    async fn emit(&mut self, event: &ServerEvent) -> Result<(), Gone> {
        let mut line = encode_server_event(event);
        line.push('\n');
        match tokio::time::timeout(self.write_timeout, self.tx.send(Bytes::from(line))).await {
            Ok(Ok(())) => Ok(()),
            _ => Err(Gone),
        }
    }

    /// Deliver every queued event. Stops after a terminal event.
    async fn pump(&mut self) -> Result<(), Gone> {
        while !self.terminal_sent {
            let Some(event) = self.session.try_recv_event() else {
                break;
            };
            self.emit(&event).await?;
            self.terminal_sent = matches!(
                event,
                ServerEvent::Final { .. } | ServerEvent::Cancelled | ServerEvent::Error(_)
            );
        }
        Ok(())
    }

    /// Fail the stream for an error the contract did not see: flush what the
    /// stream already produced, then send one `error` unless a terminal event
    /// went out.
    async fn reject(&mut self, code: ErrorCode, message: impl Into<String>) -> Result<(), Gone> {
        self.pump().await?;
        if !self.terminal_sent {
            self.emit(&ServerEvent::Error(StreamError::new(code, message)))
                .await?;
            self.terminal_sent = true;
        }
        Ok(())
    }
}

async fn next_data(body: &mut Incoming) -> Option<Result<Bytes, hyper::Error>> {
    loop {
        match poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)).await? {
            Ok(frame) => {
                if let Ok(data) = frame.into_data() {
                    return Some(Ok(data));
                }
                // Trailers are not part of the contract.
            }
            Err(e) => return Some(Err(e)),
        }
    }
}

async fn run_stream<B>(
    mut body: Incoming,
    session: StreamSession<B>,
    tx: mpsc::Sender<Bytes>,
    ctx: Arc<Ctx<B>>,
) where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let cfg = &ctx.shared.config;
    let mut run = Run {
        session,
        tx,
        write_timeout: cfg.write_timeout,
        terminal_sent: false,
    };
    let drain = matches!(stream_loop(&mut run, &mut body, &ctx).await, Ok(false));
    // Free the stream slot before any drain wait.
    let Run { session, tx, .. } = run;
    drop(session);
    if drain {
        discard_upload(&mut body, cfg.drain_timeout, cfg.drain_bytes).await;
    }
    drop(tx);
}

/// Returns `Ok(true)` if the upload ended cleanly, `Ok(false)` if the stream
/// ended with upload bytes possibly outstanding, and `Err` if the client is gone.
async fn stream_loop<B>(run: &mut Run<B>, body: &mut Incoming, ctx: &Ctx<B>) -> Result<bool, Gone>
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let cfg = &ctx.shared.config;
    let deadline = Instant::now() + cfg.max_stream_duration;
    let mut tick = tokio::time::interval(cfg.poll_interval.max(Duration::from_millis(1)));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut stop = ctx.stop.clone();
    let mut decoder = RecordDecoder::new(ctx.shared.max_record_bytes);
    let mut input_done = false;
    let mut closed = false;
    let mut last_inbound = Instant::now();
    let mut body_bytes = 0usize;

    loop {
        run.pump().await?;
        if run.terminal_sent || closed {
            return Ok(input_done);
        }
        tokio::select! {
            () = run.tx.closed() => return Err(Gone),
            () = tokio::time::sleep_until(deadline) => {
                run.reject(ErrorCode::Internal, "stream deadline exceeded").await?;
            }
            _ = stop.changed() => {
                run.reject(ErrorCode::Internal, "server shutting down").await?;
            }
            _ = tick.tick() => {
                let committing = matches!(run.session.phase(), StreamPhase::Committing);
                if !input_done && !committing && last_inbound.elapsed() >= cfg.idle_timeout {
                    run.reject(ErrorCode::Internal, "idle timeout").await?;
                }
            }
            chunk = next_data(body), if !input_done => match chunk {
                // Reset or truncated chunked body: release silently.
                Some(Err(_)) => return Err(Gone),
                Some(Ok(data)) => {
                    last_inbound = Instant::now();
                    body_bytes = body_bytes.saturating_add(data.len());
                    if body_bytes > cfg.max_body_bytes {
                        run.reject(ErrorCode::StreamTooLong, "request body exceeds the size limit")
                            .await?;
                        continue;
                    }
                    decoder.push(&data);
                    loop {
                        let record = match decoder.next_record() {
                            Ok(Some(r)) => r,
                            Ok(None) => break,
                            Err(e) => {
                                run.reject(e.code, e.message).await?;
                                break;
                            }
                        };
                        match decode_record(&record, cfg.max_text_bytes) {
                            Ok(msg) => {
                                closed = msg == ClientMessage::Close;
                                // A contract error is queued as an `Error`
                                // event by the coordinator and delivered by
                                // `pump`.
                                if run.session.send(&msg).is_err() || closed {
                                    break;
                                }
                            }
                            Err(e) => {
                                run.reject(e.code, e.message).await?;
                                break;
                            }
                        }
                    }
                }
                None => {
                    input_done = true;
                    let waiting = matches!(
                        run.session.phase(),
                        StreamPhase::Committing | StreamPhase::CancelPending
                    ) || run.session.phase().is_terminal();
                    if decoder.has_partial() {
                        run.reject(ErrorCode::MalformedFrame, "request body ended inside a record")
                            .await?;
                    } else if !waiting && !closed {
                        run.reject(
                            ErrorCode::InvalidMessageOrder,
                            "request body ended before commit",
                        )
                        .await?;
                    }
                }
            },
        }
    }
}

/// Discard upload bytes until the body ends, the byte budget is spent, or the
/// absolute deadline passes.
async fn discard_upload(body: &mut Incoming, timeout: Duration, budget: usize) {
    let deadline = Instant::now() + timeout;
    let mut left = budget;
    while let Ok(Some(Ok(data))) = tokio::time::timeout_at(deadline, next_data(body)).await {
        match left.checked_sub(data.len()) {
            Some(rest) => left = rest,
            None => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{AudioFormat, PROTOCOL_VERSION};

    fn sample_messages() -> Vec<ClientMessage> {
        vec![
            ClientMessage::Start {
                version: PROTOCOL_VERSION,
                format: AudioFormat::mono_16k_s16(),
            },
            ClientMessage::Audio(AudioFrame::from_s16(0, &[1, -2, 3])),
            ClientMessage::Commit { frames: 1 },
            ClientMessage::Cancel,
            ClientMessage::Close,
        ]
    }

    #[test]
    fn records_round_trip_at_every_split() {
        let msgs = sample_messages();
        let bytes: Vec<u8> = msgs.iter().flat_map(encode_message_record).collect();
        for split in 0..=bytes.len() {
            let mut dec = RecordDecoder::new(1024);
            let mut out = Vec::new();
            for part in [&bytes[..split], &bytes[split..]] {
                dec.push(part);
                while let Some(r) = dec.next_record().unwrap() {
                    out.push(decode_record(&r, 1024).unwrap());
                }
            }
            assert_eq!(out, msgs, "split at {split}");
            assert!(!dec.has_partial());
        }
    }

    #[test]
    fn byte_at_a_time_decodes() {
        let msgs = sample_messages();
        let bytes: Vec<u8> = msgs.iter().flat_map(encode_message_record).collect();
        let mut dec = RecordDecoder::new(1024);
        let mut out = Vec::new();
        for b in bytes {
            dec.push(&[b]);
            while let Some(r) = dec.next_record().unwrap() {
                out.push(decode_record(&r, 1024).unwrap());
            }
        }
        assert_eq!(out, msgs);
    }

    #[test]
    fn oversize_is_refused_from_the_prefix_alone() {
        let mut dec = RecordDecoder::new(16);
        dec.push(&17u32.to_le_bytes());
        let e = dec.next_record().unwrap_err();
        assert_eq!(e.code, ErrorCode::FrameTooLarge);
    }

    #[test]
    fn zero_length_and_truncation() {
        let mut dec = RecordDecoder::new(16);
        dec.push(&0u32.to_le_bytes());
        assert_eq!(
            dec.next_record().unwrap_err().code,
            ErrorCode::MalformedFrame
        );

        let mut dec = RecordDecoder::new(16);
        dec.push(&[4, 0, 0]);
        assert!(dec.next_record().unwrap().is_none());
        assert!(dec.has_partial());
    }

    #[test]
    fn decode_record_rules() {
        let e = decode_record(b"", 64).unwrap_err();
        assert_eq!(e.code, ErrorCode::MalformedFrame);
        let e = decode_record(&[b'{', 0xff, b'}'], 64).unwrap_err();
        assert_eq!(e.code, ErrorCode::MalformedFrame);
        let mut big = b"{\"type\":\"close\",\"pad\":\"".to_vec();
        big.extend(vec![b'a'; 100]);
        big.extend(b"\"}");
        let e = decode_record(&big, 64).unwrap_err();
        assert_eq!(e.code, ErrorCode::FrameTooLarge);
    }
}
