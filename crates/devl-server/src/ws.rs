//! WebSocket adapter over the shared [`Coordinator`].
//!
//! One connection carries one stream (one utterance). Control messages are
//! JSON text frames (see [`crate::wire`]); audio is binary frames in the
//! envelope of [`crate::frame`]. The adapter owns the transport only: every
//! contract decision, scheduler decision, and credit decision is made by the
//! [`Coordinator`] through a [`StreamSession`].
//!
//! # Threading
//!
//! The coordinator API is synchronous, so the adapter is too: one accept
//! thread plus one thread per connection, with no async runtime. A connection
//! thread alternates between reading with a short socket timeout and draining
//! its stream's events, so outbound latency is bounded by
//! [`WsConfig::poll_interval`].
//!
//! # Bounds
//!
//! * Connections are capped by [`WsConfig::max_connections`]; further TCP
//!   connections are closed before the handshake. Open streams are separately
//!   capped by the coordinator, and a rejected stream gets an `internal` error.
//! * Inbound messages are capped at the larger of one maximal audio frame
//!   (`FRAME_HEADER_LEN + 4 * max_frame_samples` bytes) and
//!   [`WsConfig::max_text_bytes`]; the WebSocket layer refuses anything larger
//!   before buffering it.
//! * The opening handshake has an absolute deadline of
//!   [`WsConfig::handshake_timeout`], measured from accept on a monotonic
//!   clock. It covers every read and write of the handshake, so a peer that
//!   dribbles bytes cannot hold a connection slot past it, and server
//!   shutdown interrupts a handshake in progress within one
//!   [`WsConfig::poll_interval`].
//! * Outbound traffic is the coordinator's bounded event queue; a client that
//!   stops reading hits [`WsConfig::write_timeout`] and is disconnected.
//!   `write_timeout` bounds each blocking write call, not a whole message
//!   or connection.
//! * A connection with no inbound message for [`WsConfig::idle_timeout`]
//!   (except while waiting for a final) is closed.
//!
//! # Teardown
//!
//! Every exit path (terminal event, peer close, protocol error, I/O error,
//! idle timeout, server shutdown) ends by dropping the `StreamSession`, which
//! releases the stream's pending audio, scheduler jobs, and session. Errors
//! the contract did not see (undecodable frames, oversized messages) are
//! reported with one `error` event if no terminal event was already sent.
//! A peer `Close` is answered with a flushed close reply, and every closing
//! handshake (ours or the peer's) is bounded by [`WsConfig::close_timeout`] in
//! total, with writes during it capped at the same value.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use devl_core::DevlBackend;
use tungstenite::protocol::frame::coding::CloseCode;
use tungstenite::protocol::{CloseFrame, WebSocketConfig};
use tungstenite::{Message, WebSocket};

use crate::coordinator::{Coordinator, StreamSession};
use crate::error::{ErrorCode, StreamError};
use crate::frame::{FRAME_HEADER_LEN, decode_audio_frame};
use crate::message::{ClientMessage, ServerEvent};
use crate::state::StreamPhase;
use crate::wire::{encode_server_event, parse_client_message};

/// Adapter settings.
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// Maximum simultaneous connections, including those mid-handshake.
    pub max_connections: usize,
    /// Maximum size of one control (text) message in bytes.
    pub max_text_bytes: usize,
    /// Socket read timeout; also the event-drain cadence of a connection.
    pub poll_interval: Duration,
    /// Absolute time allowed for the whole opening handshake, from accept.
    pub handshake_timeout: Duration,
    /// Time allowed for one blocking outbound write call before the client is
    /// dropped. This is a per-operation bound, not a deadline for a message.
    pub write_timeout: Duration,
    /// Inbound silence after which a connection is closed. Not applied while
    /// the stream is committing, since the client then waits on the server.
    pub idle_timeout: Duration,
    /// Total time a closing handshake may take, including flushing our close
    /// frame and waiting for the peer's reply.
    pub close_timeout: Duration,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            max_connections: 64,
            max_text_bytes: 16 * 1024,
            poll_interval: Duration::from_millis(10),
            handshake_timeout: Duration::from_secs(5),
            write_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(60),
            close_timeout: Duration::from_secs(1),
        }
    }
}

struct Shared {
    config: WsConfig,
    max_message_bytes: usize,
    stopping: AtomicBool,
    active: AtomicUsize,
    connections: Mutex<Vec<JoinHandle<()>>>,
}

fn plock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Decrements the active-connection count on every exit path.
struct ActiveGuard(Arc<Shared>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A running WebSocket listener bound to a [`Coordinator`].
///
/// Dropping the server shuts it down.
pub struct WsServer {
    addr: SocketAddr,
    shared: Arc<Shared>,
    accept: Mutex<Option<JoinHandle<()>>>,
}

impl WsServer {
    /// Bind `addr` and start accepting connections.
    ///
    /// The coordinator is shared: every connection opens its stream on it.
    pub fn bind<B>(
        addr: impl ToSocketAddrs,
        coordinator: Arc<Coordinator<B>>,
        config: WsConfig,
    ) -> io::Result<Self>
    where
        B: DevlBackend + Clone + Send + Sync + 'static,
        B::AudioEncoding: Send,
    {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let audio_bytes = usize::try_from(coordinator.limits().max_frame_samples)
            .unwrap_or(usize::MAX)
            .saturating_mul(4)
            .saturating_add(FRAME_HEADER_LEN);
        let shared = Arc::new(Shared {
            max_message_bytes: audio_bytes.max(config.max_text_bytes),
            config,
            stopping: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            connections: Mutex::new(Vec::new()),
        });
        let accept = {
            let shared = Arc::clone(&shared);
            thread::Builder::new()
                .name("devl-ws-accept".into())
                .spawn(move || accept_loop(&listener, &shared, &coordinator))?
        };
        Ok(Self {
            addr,
            shared,
            accept: Mutex::new(Some(accept)),
        })
    }

    /// Address the listener is bound to (useful with port 0).
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Connections currently open, including those mid-handshake.
    pub fn active_connections(&self) -> usize {
        self.shared.active.load(Ordering::SeqCst)
    }

    /// Stop accepting, close every connection, and join all adapter threads.
    ///
    /// Streams are released as their connections end. The coordinator is not
    /// shut down; its owner does that.
    pub fn shutdown(&self) {
        self.shared.stopping.store(true, Ordering::SeqCst);
        if let Some(h) = plock(&self.accept).take() {
            let _ = h.join();
        }
        let handles = std::mem::take(&mut *plock(&self.shared.connections));
        for h in handles {
            let _ = h.join();
        }
    }
}

impl Drop for WsServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn accept_loop<B>(listener: &TcpListener, shared: &Arc<Shared>, coordinator: &Arc<Coordinator<B>>)
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    while !shared.stopping.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                if shared.active.fetch_add(1, Ordering::SeqCst) >= shared.config.max_connections {
                    shared.active.fetch_sub(1, Ordering::SeqCst);
                    drop(stream);
                    continue;
                }
                let guard = ActiveGuard(Arc::clone(shared));
                let sh = Arc::clone(shared);
                let co = Arc::clone(coordinator);
                let spawned = thread::Builder::new()
                    .name("devl-ws-conn".into())
                    .spawn(move || {
                        let _guard = guard;
                        serve_connection(stream, &sh, &co);
                    });
                // On spawn failure the closure (and its guard) is dropped,
                // which releases the count and the socket.
                if let Ok(h) = spawned {
                    let mut conns = plock(&shared.connections);
                    conns.retain(|c| !c.is_finished());
                    conns.push(h);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(shared.config.poll_interval);
            }
            Err(_) => thread::sleep(shared.config.poll_interval),
        }
    }
}

/// A socket whose reads and writes honour an optional absolute deadline and
/// the server's stop flag. With no deadline it is a plain pass-through.
///
/// Used so the opening handshake can be bounded in total (the OS timeout is
/// per call) and interrupted by shutdown.
struct Conn {
    tcp: TcpStream,
    /// Absolute handshake deadline; `None` once the handshake is done.
    deadline: Option<Instant>,
    shared: Arc<Shared>,
    poll: Duration,
    write_timeout: Duration,
}

impl Conn {
    fn remaining(&self) -> io::Result<Option<Duration>> {
        match self.deadline {
            None => Ok(None),
            Some(d) => d
                .checked_duration_since(Instant::now())
                .filter(|r| !r.is_zero())
                .map(Some)
                .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "handshake deadline")),
        }
    }

    fn interrupted(&self) -> io::Result<()> {
        if self.deadline.is_some() && self.shared.stopping.load(Ordering::SeqCst) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "server stopping",
            ));
        }
        Ok(())
    }
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.deadline.is_none() {
            return self.tcp.read(buf);
        }
        loop {
            self.interrupted()?;
            let slice = self.remaining()?.map_or(self.poll, |r| r.min(self.poll));
            self.tcp.set_read_timeout(Some(slice))?;
            match self.tcp.read(buf) {
                Err(e) if is_timeout(&e) => {}
                other => return other,
            }
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.deadline.is_some() {
            self.interrupted()?;
            let slice = self
                .remaining()?
                .map_or(self.write_timeout, |r| r.min(self.write_timeout));
            self.tcp.set_write_timeout(Some(slice))?;
        }
        self.tcp.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.tcp.flush()
    }
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

/// What ended the connection loop.
enum Exit {
    /// A terminal event was delivered or the client sent `Close`; do the
    /// closing handshake.
    Finished,
    /// The peer sent a WebSocket close frame; flush our reply.
    PeerClosed,
    /// The peer is gone or the transport failed; nothing more can be sent.
    Gone,
    /// The server is stopping.
    Stopping,
    /// Close with a protocol-level code (after any error event was sent).
    Code(CloseCode),
}

fn serve_connection<B>(stream: TcpStream, shared: &Arc<Shared>, coordinator: &Coordinator<B>)
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let cfg = &shared.config;
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_nodelay(true);
    let Some(deadline) = Instant::now().checked_add(cfg.handshake_timeout) else {
        return;
    };
    let conn = Conn {
        tcp: stream,
        deadline: Some(deadline),
        shared: Arc::clone(shared),
        poll: cfg.poll_interval.max(Duration::from_millis(1)),
        write_timeout: cfg.write_timeout,
    };
    let ws_cfg = WebSocketConfig::default()
        .max_message_size(Some(shared.max_message_bytes))
        .max_frame_size(Some(shared.max_message_bytes))
        .write_buffer_size(0)
        .max_write_buffer_size(64 * 1024);
    let Ok(mut ws) = tungstenite::accept_with_config(conn, Some(ws_cfg)) else {
        return;
    };
    // Handshake done: drop the deadline and switch to per-call timeouts.
    ws.get_mut().deadline = None;
    let tcp = &ws.get_ref().tcp;
    if tcp.set_read_timeout(Some(cfg.poll_interval)).is_err()
        || tcp.set_write_timeout(Some(cfg.write_timeout)).is_err()
    {
        return;
    }

    let session = match coordinator.open_stream() {
        Ok(s) => s,
        Err(e) => {
            let err = StreamError::new(ErrorCode::Internal, e.to_string());
            let _ = send_event(&mut ws, &ServerEvent::Error(err));
            close(&mut ws, CloseCode::Again, cfg.close_timeout);
            return;
        }
    };

    let mut conn = Connection {
        ws,
        session,
        shared,
        terminal_sent: false,
        last_inbound: Instant::now(),
    };
    match conn.run() {
        Exit::Finished | Exit::PeerClosed => {
            close(&mut conn.ws, CloseCode::Normal, cfg.close_timeout);
        }
        Exit::Stopping => close(&mut conn.ws, CloseCode::Away, cfg.close_timeout),
        Exit::Code(code) => close(&mut conn.ws, code, cfg.close_timeout),
        Exit::Gone => {}
    }
    // `conn.session` drops here: the stream is torn down on every path.
}

struct Connection<'a, B>
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    ws: WebSocket<Conn>,
    session: StreamSession<B>,
    shared: &'a Shared,
    terminal_sent: bool,
    last_inbound: Instant,
}

impl<B> Connection<'_, B>
where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    fn run(&mut self) -> Exit {
        loop {
            if self.shared.stopping.load(Ordering::SeqCst) {
                return Exit::Stopping;
            }
            let mut client_closed = false;
            match self.ws.read() {
                Ok(Message::Text(text)) => {
                    self.last_inbound = Instant::now();
                    if text.len() > self.shared.config.max_text_bytes {
                        return self.reject(
                            ErrorCode::FrameTooLarge,
                            "control message exceeds the size limit",
                            CloseCode::Size,
                        );
                    }
                    match parse_client_message(text.as_str()) {
                        Ok(msg) => {
                            client_closed = msg == ClientMessage::Close;
                            // A contract error is queued as an `Error` event
                            // by the coordinator and delivered by `pump`.
                            let _ = self.session.send(&msg);
                        }
                        Err(e) => {
                            return self.reject(e.code, &e.message, CloseCode::Invalid);
                        }
                    }
                }
                Ok(Message::Binary(bytes)) => {
                    self.last_inbound = Instant::now();
                    match decode_audio_frame(&bytes) {
                        Ok(frame) => {
                            let _ = self.session.send(&ClientMessage::Audio(frame));
                        }
                        Err(e) => return self.reject(e.code, &e.message, CloseCode::Invalid),
                    }
                }
                Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => {}
                // tungstenite queued the close reply; `close` flushes it.
                Ok(Message::Close(_)) => return Exit::PeerClosed,
                Err(tungstenite::Error::Io(e)) if is_timeout(&e) => {}
                Err(tungstenite::Error::Capacity(_)) => {
                    return self.reject(
                        ErrorCode::FrameTooLarge,
                        "message exceeds the size limit",
                        CloseCode::Size,
                    );
                }
                Err(tungstenite::Error::Utf8(_) | tungstenite::Error::Protocol(_)) => {
                    return self.reject(
                        ErrorCode::MalformedFrame,
                        "websocket protocol violation",
                        CloseCode::Protocol,
                    );
                }
                Err(_) => return Exit::Gone,
            }

            if self.pump().is_err() {
                return Exit::Gone;
            }
            if self.terminal_sent || client_closed {
                return Exit::Finished;
            }
            let committing = self.session.phase() == StreamPhase::Committing;
            if !committing && self.last_inbound.elapsed() >= self.shared.config.idle_timeout {
                return Exit::Code(CloseCode::Policy);
            }
        }
    }

    /// Deliver every queued event. Stops after a terminal event.
    fn pump(&mut self) -> Result<(), tungstenite::Error> {
        while !self.terminal_sent {
            let Some(event) = self.session.try_recv_event() else {
                break;
            };
            send_event(&mut self.ws, &event)?;
            self.terminal_sent = matches!(
                event,
                ServerEvent::Final { .. } | ServerEvent::Cancelled | ServerEvent::Error(_)
            );
        }
        Ok(())
    }

    /// Fail the connection for an error the contract did not see: flush what
    /// the stream already produced, then send one `error` unless a terminal
    /// event went out.
    fn reject(&mut self, code: ErrorCode, message: &str, close: CloseCode) -> Exit {
        if self.pump().is_err() {
            return Exit::Gone;
        }
        if !self.terminal_sent {
            let err = StreamError::new(code, message);
            if send_event(&mut self.ws, &ServerEvent::Error(err)).is_err() {
                return Exit::Gone;
            }
            self.terminal_sent = true;
        }
        Exit::Code(close)
    }
}

fn send_event(ws: &mut WebSocket<Conn>, event: &ServerEvent) -> Result<(), tungstenite::Error> {
    ws.send(Message::text(encode_server_event(event)))
}

/// Send a close frame and give the peer a bounded time to answer, so a
/// well-behaved client reads every queued event before the socket closes.
///
/// The whole exchange is bounded by `wait`: writes are capped at `wait` and
/// the reply loop ends at an absolute deadline, whatever the peer sends. When
/// the peer closed first this flushes tungstenite's queued reply instead.
fn close(ws: &mut WebSocket<Conn>, code: CloseCode, wait: Duration) {
    let Some(deadline) = Instant::now().checked_add(wait) else {
        return;
    };
    let cap = wait
        .max(Duration::from_millis(1))
        .min(ws.get_ref().write_timeout);
    let _ = ws.get_ref().tcp.set_write_timeout(Some(cap));
    let frame = CloseFrame {
        code,
        reason: "".into(),
    };
    if ws.close(Some(frame)).is_err() {
        return;
    }
    while Instant::now() < deadline {
        match ws.read() {
            Ok(_) => {}
            Err(tungstenite::Error::Io(e)) if is_timeout(&e) => {}
            Err(_) => return,
        }
    }
}
