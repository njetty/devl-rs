//! Per-stream contract state machine.
//!
//! [`StreamContract`] validates both directions of one stream: client
//! messages via [`StreamContract::receive`] and server events via
//! [`StreamContract::emit`]. It holds counters only; it never touches a
//! backend, a scheduler, or a socket.

use crate::error::{ErrorCode, StreamError};
use crate::frame::decode_samples;
use crate::limits::StreamLimits;
use crate::message::{AudioFormat, ClientMessage, PROTOCOL_VERSION, ProtocolVersion, ServerEvent};
use devl_core::StreamId;

/// Lifecycle phase of one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPhase {
    /// No `Start` received yet.
    AwaitingStart,
    /// `Start` accepted; audio and partials flow once `Started` is emitted.
    Streaming,
    /// `Commit` accepted; waiting for the final transcript.
    Committing,
    /// `Final` emitted. Terminal.
    Finalized,
    /// Cancel requested; `Cancelled` not yet emitted. In-flight client
    /// messages are ignored and server partials are suppressed.
    CancelPending,
    /// `Cancelled` emitted. Terminal.
    Cancelled,
    /// A violation or server failure ended the stream. Terminal once the
    /// `Error` event has been emitted.
    Failed,
}

impl StreamPhase {
    /// `true` for phases in which only `Close` is meaningful.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finalized | Self::Cancelled | Self::Failed)
    }
}

/// Outcome of a successfully validated client message.
#[derive(Debug, Clone, PartialEq)]
pub enum Accepted {
    /// `Start` accepted with the negotiated format. Emit `Started` next.
    Started {
        /// Format of subsequent audio frames.
        format: AudioFormat,
    },
    /// Audio accepted and decoded.
    Audio {
        /// Frame sequence number.
        seq: u64,
        /// Mono samples, ready for `DevlSession::push_audio`.
        samples: Vec<f32>,
    },
    /// `Commit` accepted; the caller should schedule a commit job.
    Commit,
    /// `Cancel` or `Close` abandoned the stream; emit `Cancelled`.
    Cancel,
    /// `Close` after a terminal event; tear down the transport.
    Close,
    /// Message was a harmless late arrival (in-flight after cancel, or a
    /// duplicate cancel/close) and must be dropped.
    Ignored,
}

/// Validating state for one stream.
#[derive(Debug)]
pub struct StreamContract {
    stream_id: StreamId,
    limits: StreamLimits,
    phase: StreamPhase,
    format: Option<AudioFormat>,
    started_emitted: bool,
    error_emitted: bool,
    next_seq: u64,
    samples_received: u64,
    credit: u64,
    last_revision: Option<u64>,
}

fn order(msg: &str) -> StreamError {
    StreamError::new(ErrorCode::InvalidMessageOrder, msg)
}

impl StreamContract {
    /// Create state for a stream. Fails if `limits` are inconsistent.
    pub fn new(stream_id: StreamId, limits: StreamLimits) -> Result<Self, StreamError> {
        limits.validate()?;
        Ok(Self {
            stream_id,
            limits,
            phase: StreamPhase::AwaitingStart,
            format: None,
            started_emitted: false,
            error_emitted: false,
            next_seq: 0,
            samples_received: 0,
            credit: 0,
            last_revision: None,
        })
    }

    /// Identifier used to submit this stream's jobs to `PriorityScheduler`.
    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    /// Current phase.
    pub fn phase(&self) -> StreamPhase {
        self.phase
    }

    /// Limits in force.
    pub fn limits(&self) -> StreamLimits {
        self.limits
    }

    /// Total samples accepted so far.
    pub fn samples_received(&self) -> u64 {
        self.samples_received
    }

    /// Audio credit the client currently holds, in samples.
    pub fn credit(&self) -> u64 {
        self.credit
    }

    /// Number of audio frames accepted so far.
    pub fn frames_received(&self) -> u64 {
        self.next_seq
    }

    fn fail(&mut self, err: StreamError) -> StreamError {
        if !self.phase.is_terminal() {
            self.phase = StreamPhase::Failed;
        }
        err
    }

    /// Validate a client message and advance the state.
    ///
    /// On `Err` the stream moves to [`StreamPhase::Failed`] (unless already
    /// terminal); the caller emits `ServerEvent::Error` with the returned
    /// error. Late `Cancel`/`Close` and in-flight messages after a cancel are
    /// reported as [`Accepted::Ignored`] rather than errors.
    pub fn receive(&mut self, msg: &ClientMessage) -> Result<Accepted, StreamError> {
        match self.receive_inner(msg) {
            Ok(a) => Ok(a),
            Err(e) => Err(self.fail(e)),
        }
    }

    fn receive_inner(&mut self, msg: &ClientMessage) -> Result<Accepted, StreamError> {
        use StreamPhase as P;
        match (msg, self.phase) {
            (ClientMessage::Close, p) if p.is_terminal() => Ok(Accepted::Close),
            (ClientMessage::Close, P::CancelPending) => Ok(Accepted::Ignored),
            (ClientMessage::Close, P::AwaitingStart) => Ok(Accepted::Close),
            (ClientMessage::Close, _) => {
                self.phase = P::CancelPending;
                Ok(Accepted::Cancel)
            }
            (ClientMessage::Cancel, p) if p.is_terminal() || p == P::CancelPending => {
                Ok(Accepted::Ignored)
            }
            (ClientMessage::Cancel, P::AwaitingStart) => Err(order("cancel before start")),
            (ClientMessage::Cancel, _) => {
                self.phase = P::CancelPending;
                Ok(Accepted::Cancel)
            }
            (_, P::CancelPending) => Ok(Accepted::Ignored),
            (ClientMessage::Start { version, format }, P::AwaitingStart) => {
                self.start(*version, *format)
            }
            (ClientMessage::Start { .. }, _) => Err(order("start already received")),
            (_, P::AwaitingStart) => Err(order("first message must be start")),
            (ClientMessage::Audio(frame), P::Streaming) => {
                if !self.started_emitted {
                    return Err(order("audio before started"));
                }
                if frame.seq != self.next_seq {
                    return Err(StreamError::new(
                        ErrorCode::SequenceViolation,
                        format!("expected seq {}, got {}", self.next_seq, frame.seq),
                    ));
                }
                let format = self.format.expect("format set after start");
                let samples =
                    decode_samples(format, &frame.payload, self.limits.max_frame_samples)?;
                let n = samples.len() as u64;
                if n > self.credit {
                    return Err(StreamError::new(
                        ErrorCode::BackpressureViolation,
                        format!("frame of {n} samples exceeds credit {}", self.credit),
                    ));
                }
                if n > self.limits.max_stream_samples - self.samples_received {
                    return Err(StreamError::new(
                        ErrorCode::StreamTooLong,
                        format!("stream exceeds {} samples", self.limits.max_stream_samples),
                    ));
                }
                self.credit -= n;
                self.samples_received += n;
                self.next_seq += 1;
                Ok(Accepted::Audio {
                    seq: frame.seq,
                    samples,
                })
            }
            (ClientMessage::Commit { frames }, P::Streaming) => {
                if !self.started_emitted {
                    return Err(order("commit before started"));
                }
                if *frames != self.next_seq {
                    return Err(StreamError::new(
                        ErrorCode::SequenceViolation,
                        format!(
                            "commit declares {frames} frames, received {}",
                            self.next_seq
                        ),
                    ));
                }
                self.phase = P::Committing;
                Ok(Accepted::Commit)
            }
            (ClientMessage::Audio(_), _) => Err(order("audio after commit or terminal event")),
            (ClientMessage::Commit { .. }, _) => {
                Err(order("commit after commit or terminal event"))
            }
        }
    }

    fn start(
        &mut self,
        version: ProtocolVersion,
        format: AudioFormat,
    ) -> Result<Accepted, StreamError> {
        if version.major != PROTOCOL_VERSION.major || version.minor > PROTOCOL_VERSION.minor {
            return Err(StreamError::new(
                ErrorCode::UnsupportedVersion,
                format!(
                    "client {}.{}, server {}.{}",
                    version.major, version.minor, PROTOCOL_VERSION.major, PROTOCOL_VERSION.minor
                ),
            ));
        }
        if format.sample_rate_hz != AudioFormat::REQUIRED_SAMPLE_RATE_HZ || format.channels != 1 {
            return Err(StreamError::new(
                ErrorCode::UnsupportedFormat,
                format!(
                    "{} Hz x {} channels; version 1 requires 16000 Hz mono",
                    format.sample_rate_hz, format.channels
                ),
            ));
        }
        self.format = Some(format);
        self.phase = StreamPhase::Streaming;
        Ok(Accepted::Started { format })
    }

    /// Validate a server event and advance the state.
    ///
    /// Illegal events (for example a partial after `Commit`) return
    /// `InvalidMessageOrder` without changing state so the caller can drop
    /// them; a second terminal event is always illegal.
    pub fn emit(&mut self, event: &ServerEvent) -> Result<(), StreamError> {
        use StreamPhase as P;
        match (event, self.phase) {
            (ServerEvent::Started { version, limits }, P::Streaming) if !self.started_emitted => {
                if *version != PROTOCOL_VERSION {
                    return Err(StreamError::new(
                        ErrorCode::Internal,
                        "started version differs from server version",
                    ));
                }
                if *limits != self.limits {
                    return Err(StreamError::new(
                        ErrorCode::Internal,
                        "started limits differ from stream limits",
                    ));
                }
                self.started_emitted = true;
                self.credit = self.limits.initial_credit_samples;
                Ok(())
            }
            (ServerEvent::Credit { samples }, P::Streaming) if self.started_emitted => {
                if *samples == 0
                    || *samples > self.limits.max_outstanding_credit_samples - self.credit
                {
                    return Err(StreamError::new(
                        ErrorCode::Internal,
                        "credit grant is zero or exceeds the outstanding maximum",
                    ));
                }
                self.credit += samples;
                Ok(())
            }
            (
                ServerEvent::Partial {
                    revision,
                    audio_samples,
                    ..
                },
                P::Streaming,
            ) if self.started_emitted => {
                if self.last_revision.is_some_and(|r| *revision <= r) {
                    return Err(order("partial revision must increase"));
                }
                if *audio_samples > self.samples_received {
                    return Err(order("partial covers audio not yet received"));
                }
                self.last_revision = Some(*revision);
                Ok(())
            }
            (ServerEvent::Final { audio_samples, .. }, P::Committing) => {
                if *audio_samples != self.samples_received {
                    return Err(order("final must cover all received audio"));
                }
                self.phase = P::Finalized;
                Ok(())
            }
            (ServerEvent::Cancelled, P::CancelPending) => {
                self.phase = P::Cancelled;
                Ok(())
            }
            (ServerEvent::Error(_), P::Failed) if !self.error_emitted => {
                self.error_emitted = true;
                Ok(())
            }
            (ServerEvent::Error(_), p) if !p.is_terminal() => {
                self.phase = P::Failed;
                self.error_emitted = true;
                Ok(())
            }
            _ => Err(order("server event is not legal in the current phase")),
        }
    }
}
