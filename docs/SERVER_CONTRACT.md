# `devl-server` Streaming Contract

Status: contract version `1.0`, implemented in `crates/devl-server` as types, binary audio framing, limits, and a validating per-stream state machine. The session coordinator (`Coordinator`, `StreamSession`) drives per-stream sessions through `PriorityScheduler`. A synchronous WebSocket adapter (`WsServer`) maps the contract onto connections. HTTP endpoints, backend selection, endpointing, authentication, and deployment are not implemented; see [Boundaries](#boundaries).

The contract is transport-neutral. It defines what a client and server may say to each other for one stream, not how bytes travel. Control message serialization is chosen by endpoint adapters; the WebSocket adapter uses JSON (see [WebSocket adapter](#websocket-adapter)). Audio has a defined byte layout.

## Stream model

One stream is one utterance: audio is sent in order, then committed once, and the server returns one final transcript. This mirrors `DevlSession` (`push_audio`, periodic drafts, `commit`). A client wanting multiple utterances opens multiple streams. Multiplexing many streams over one connection is an adapter concern; the contract only requires a `StreamId` per stream.

```text
client                                   server
  | Start{version, format} ------------>  |
  | <------------ Started{version, limits}|   initial credit granted
  | Audio{seq=0} ---------------------->  |
  | <------------------ Partial{rev, ..}  |   zero or more, advisory
  | Audio{seq=1} ---------------------->  |
  | <------------------------ Credit{n}   |   backpressure release
  | Commit{frames=2} ------------------>  |
  | <---------------------- Final{text}   |   terminal
  | Close ----------------------------->  |
```

## Versioning

- Version is `major.minor`. This document defines `1.0`.
- A different major is rejected with `unsupported_version`. A client minor greater than the server minor is also rejected. Minor bumps only add optional behavior.
- The binary audio envelope carries the major in its first byte so a misrouted or stale frame is detected without parsing control messages.

## Audio format

Declared once in `Start`; every audio frame uses it.

| Property | Version 1 |
| :--- | :--- |
| Sample rate | 16000 Hz. Other rates are rejected with `unsupported_format`; clients resample. |
| Channels | 1 (mono). |
| Encodings | `S16Le` (signed 16-bit, little-endian, decoded as `value / 32768.0`) or `F32Le` (finite IEEE 754 binary32, little-endian, used as-is). |

Decoded samples are mono `f32`, the input type of `DevlSession::push_audio`. The server performs no normalization or resampling.

### Audio frame envelope

All integers little-endian. The envelope assumes the transport delivers whole messages; stream transports add their own length prefix.

| Offset | Size | Field |
| :--- | :--- | :--- |
| 0 | 1 | Layout version (the protocol major) |
| 1 | 1 | Kind: `1` = audio |
| 2 | 8 | `seq` (`u64`) |
| 10 | n | PCM payload |

Payload rules: nonempty, a whole number of samples, at most `max_frame_samples` samples, and no NaN or infinite `F32Le` values.

## Messages

Client to server:

| Message | Rule |
| :--- | :--- |
| `Start{version, format}` | First message, exactly once. |
| `Audio{seq, payload}` | Only after `Started` is received. |
| `Commit{frames}` | End of audio after `Started`. `frames` is the number of audio frames sent. Zero frames is legal. |
| `Cancel` | Abandon the stream. |
| `Close` | Graceful goodbye, legal in any phase. |

Server to client:

| Event | Rule |
| :--- | :--- |
| `Started{version, limits}` | Once, after `Start` is accepted. Reports the server protocol version and grants `initial_credit_samples`. |
| `Credit{samples}` | Grants more audio credit. Nonzero; held credit never exceeds `max_outstanding_credit_samples`. |
| `Partial{revision, audio_samples, text}` | Only before `Commit`. `revision` strictly increases (gaps allowed); `audio_samples` never exceeds the audio received. |
| `Final{audio_samples, text}` | Only after `Commit`; `audio_samples` equals the audio received. Terminal. |
| `Cancelled` | Acknowledges cancellation. Terminal. |
| `Error{code, message}` | Stream failed. Terminal. |

Exactly one terminal event is emitted per stream.

## Partial semantics

Partials are advisory speculative transcripts. The server may drop or coalesce them under load, consistent with `PriorityScheduler` draft coalescing, so clients must not assume a partial per audio frame. A later partial may revise earlier text. Only `Final` is authoritative.

## Sequencing

- Audio `seq` starts at 0 and increases by exactly 1. A gap, duplicate, or reorder is a `sequence_violation`.
- `Commit.frames` must equal the number of frames accepted, so lost trailing frames are detected.
- Transports must deliver one stream's messages in order.

## Limits

Advertised in `Started`, counted in mono samples:

| Limit | Effect when exceeded |
| :--- | :--- |
| `max_frame_samples` | `frame_too_large` |
| `max_stream_samples` | `stream_too_long` |
| credit (see below) | `backpressure_violation` |

`StreamLimits::default()` is 2 s frames, 5 min per stream, and 10 s of credit at 16 kHz. These are server defaults, not model limits; backend duration bounds are enforced by the backend and surfaced by the integration issue.

## Backpressure

Backpressure is credit-based and counted in samples. The server grants initial credit in `Started`; each accepted frame consumes credit equal to its sample count; the server replenishes with `Credit` as it consumes buffered audio. A client must not send a frame larger than its remaining credit and should wait for `Credit` when out. Overrunning credit is a violation, not an implicit queue: the server never buffers unbounded audio on behalf of a fast client. The coordinator decides when to release credit (for example as `push_audio` drains the stream buffer); that policy is outside this contract.

## Cancellation

- `Cancel` is valid while streaming or committing. The server stops work for the stream (a commit-phase cancel discards the pending final), suppresses further partials, and emits `Cancelled`.
- Messages that arrive after a cancel request are ignored, not errors, because they were in flight.
- A `Cancel` or duplicate `Cancel` after a terminal event is ignored, so a cancel that crosses a `Final` on the wire is harmless.
- `Close` before a terminal event behaves like `Cancel`; after one, it ends the stream. `Close` before `Start` is accepted.

## Errors

Every error is terminal for its stream: the server emits one `Error` event and then accepts only `Close`. Codes: `unsupported_version`, `unsupported_format`, `invalid_message_order`, `malformed_frame`, `sequence_violation`, `frame_too_large`, `stream_too_long`, `backpressure_violation`, `internal`. Codes are stable; messages are not.

## Boundaries

In this crate (`devl-server`):

- Message, format, limit, and error types; audio envelope encode/decode; PCM to `f32` conversion.
- `StreamContract`, a state machine with no I/O that validates client messages (`receive`) and server events (`emit`).
- `Coordinator` and `StreamSession`: a worker pool that owns one `DevlSession` per stream, submits drafts and commits to `PriorityScheduler`, and releases credit. See [Coordinator](#coordinator).

- `WsServer` and the JSON control encoding in `wire`: a WebSocket adapter over the coordinator. See [WebSocket adapter](#websocket-adapter).

Not in this crate yet, each owned by later issues:

- HTTP endpoint adapter.
- Backend integration and selection.
- VAD and endpointing. Today the client decides when to `Commit`.
- Authentication, TLS, rate limiting, and deployment.

Dependencies: `devl-server` depends on `devl-core` (for `StreamId`), `thiserror`, `serde`, `serde_json`, and `tungstenite` (handshake feature only, blocking I/O) for the WebSocket adapter. It has no async runtime, and `devl-core` is unchanged: protocol ownership, verification, rollback, and commit-over-draft scheduling remain there.

## Intended integration

An adapter owns the transport and one `StreamContract` per stream. For each inbound message it calls `receive`:

- `Started{format}`: reply `Started` through `emit`.
- `Audio{samples}`: hand samples to the coordinator, which calls `DevlSession::push_audio` and, on its checkpoint cadence, submits a draft job; the resulting tokens become a `Partial` after detokenization, which belongs to the backend side.
- `Commit`: submit a commit job (`submit_commit` evicts queued drafts for the stream); the result becomes `Final`.
- `Cancel`: cancel the stream's jobs and reply `Cancelled`.
- `Err(e)`: emit `Error(e)` and stop accepting input.

Checkpoint cadence matters because encoding and verification run before the first preemption check (see `docs/ARCHITECTURE.md`, Section 6.10 D6). The coordinator submits a draft every `draft_cadence_samples` accepted samples (default 32000, 2 s at 16 kHz).

## Coordinator

`Coordinator::new(backend, detokenizer, config)` starts a fixed worker pool over one `PriorityScheduler`. `open_stream` returns a `StreamSession`, the connection-side handle for one stream: `send(&ClientMessage)` runs the contract and the coordinator, and `try_recv_event` / `recv_event_timeout` read `ServerEvent`s. The API is synchronous and std-only; adapters bridge it to their runtime. Scheduling and verification are not reimplemented: drafts run `DevlSession::run_draft_cancellable` with `PriorityScheduler::should_preempt_draft`, finals run `DevlSession::commit`, and job order is `PriorityScheduler::next_job`.

- **Isolation.** Each stream owns its `DevlSession`, contract, pending audio, and event queue. Stream ids are assigned by the coordinator.
- **Final over draft.** `Commit` calls `submit_commit`, which evicts the stream's queued draft and flags its running draft for preemption. A running draft yields at the next step boundary; encoding and verification before the first check are not interruptible.
- **Stale partials.** A partial is dropped if the stream has committed, was cancelled or failed, covers less audio than an earlier partial, or repeats the previous text. A draft preempted by another stream's commit still publishes its verified prefix.
- **Backpressure.** Audio waits in a pending buffer only while a worker holds the session. `Credit` is granted when the buffer is drained into the session, so pending audio never exceeds granted credit and overrunning credit is `backpressure_violation`. Outbound events use a bounded queue that keeps one pending partial and one pending credit grant. Queued jobs are at most one draft and one commit per stream, and open streams are capped by `max_streams`.
- **Teardown.** Cancel, close, disconnect (dropping the handle), final, error, and shutdown clear pending audio, remove the stream's queued jobs with `PriorityScheduler::cancel_stream`, drop the session (the worker does so if mid-job), and only then free the stream slot. A stream whose job is executing keeps its slot until the worker retires, so open-stream admission bounds queued and running work: with a worker gated on one stream, repeated open, commit, cancel, and reopen cannot grow the queues beyond the cap. An uncancellable running commit can therefore delay slot reuse until it returns. Pending `Partial` events are discarded when a terminal event is queued, and messages sent after the terminal event schedule nothing. Job permits are RAII and release on every path including panics, which fail only the affected stream with `internal`. Scheduler metadata for a stream is dropped when its last draft stops.
- **Shutdown.** `Coordinator::shutdown` stops admission, fails open streams with `internal`, and joins the workers. `DevlSession::commit` has no preemption hook, so a commit already running finishes and its result is discarded. The same applies to a commit-phase `Cancel`.
- **Draft errors.** A backend error inside a draft is non-fatal and emits nothing; the commit pass re-runs the work and reports any persistent failure.

## WebSocket adapter

`WsServer::bind(addr, Arc<Coordinator>, WsConfig)` listens on a TCP address and serves one stream per WebSocket connection. All stream decisions stay in the coordinator; the adapter parses frames, calls `StreamSession::send`, and forwards `try_recv_event` output. It is synchronous: one accept thread and one thread per connection, with no async runtime.

### Framing

- Control messages are text frames, one JSON object each, with a `type` field. Client types: `start` (`version{major,minor}`, `format{encoding: "s16le"|"f32le", sample_rate_hz, channels}`), `commit` (`frames`), `cancel`, `close`. Server types: `started` (`version`, `limits`), `credit` (`samples`), `partial` (`revision`, `audio_samples`, `text`), `final` (`audio_samples`, `text`), `cancelled`, `error` (`code`, `message`). Unknown extra fields on client messages are ignored.
- Audio is a binary frame in the envelope above. One WebSocket message is one audio frame.
- A message that is not valid JSON of a known type, or an undecodable binary frame, is `malformed_frame` (`unsupported_version` for a wrong envelope version).

### Lifecycle

A terminal event is followed by a WebSocket close frame (normal closure) and the server waits briefly for the client's reply. `close` from the client ends the connection after any `cancelled` event. A peer close frame is answered with a flushed close reply, and a dropped connection is a disconnect; both release the stream without events. Every closing handshake is bounded in total by `close_timeout`. Protocol violations the adapter detects itself are answered with one `error` event (unless a terminal event already went out) and a close frame with code 1002 (protocol), 1007 (invalid payload), or 1009 (too big).

### Bounds

- `max_connections` caps simultaneous connections; extra TCP connections are closed before the handshake. The coordinator's `max_streams` is separate; a connection that cannot open a stream receives `error` code `internal` and is closed.
- Inbound messages are capped at the larger of one maximal audio frame (`10 + 4 * max_frame_samples` bytes) and `max_text_bytes` (default 16 KiB). Larger messages are refused by the WebSocket layer, answered with `frame_too_large`, and closed with 1009. A frame within the byte cap but over `max_frame_samples` is `frame_too_large` from the coordinator.
- `handshake_timeout` is an absolute deadline for the whole opening handshake, measured from accept on a monotonic clock, so a peer sending bytes slowly cannot hold a connection slot past it. Shutdown interrupts a handshake in progress within one `poll_interval`.
- Outbound buffering is the coordinator's bounded event queue plus a 64 KiB socket write buffer. A client that stops reading hits `write_timeout` and is disconnected. `write_timeout` bounds each blocking write call, not a whole message or connection.
- `idle_timeout` closes a connection with no inbound message, except while the stream is committing.
- `WsServer::shutdown` (also on drop) stops accepting, closes connections with 1001, and joins all adapter threads. The coordinator is shut down separately by its owner.
