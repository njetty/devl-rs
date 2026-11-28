# Roadmap

This document defines the high-level, multi-release plan for `devl-rs`. It is intentionally kept at the release and phase level; active implementation tasks, acceptance criteria, and day-to-day progress are tracked exclusively through GitHub Issues and Milestones.

Do not add task checkboxes or per-issue tracking here. Update this document only when a release ships or when phase scope intentionally changes.

For trait contracts, session invariants, and backend design details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## v0.1.0 — Core Protocol & Whisper Local Backend

Released on crates.io as [`devl-core` 0.1.0](https://crates.io/crates/devl-core/0.1.0), [`devl-candle` 0.1.0](https://crates.io/crates/devl-candle/0.1.0), and [`devl` 0.1.0](https://crates.io/crates/devl/0.1.0). This release establishes the runtime-agnostic DEVL protocol, priority scheduler, and local Candle Whisper backend with reproducible local benchmarking.

### Phase 1: Core Protocol, Scheduler, and Facade

- Runtime- and transport-agnostic `devl-core` crate with zero ML, async, or networking dependencies.
- `DevlBackend` trait, `DevlSession` state machine (`Idle`, `Drafting`, `Committing`, `Finalized`, `Failed`), single-pass draft verification, longest-prefix acceptance, KV-cache rollback, and checkpoint promotion.
- `PriorityScheduler` with strict commit-over-draft ordering, per-stream draft coalescing/cancellation, bounded concurrent drafts, and step-level cooperative preemption.
- Deterministic `MockBackend` (`test-utils` feature) and unit/integration test suite covering exact greedy parity across all verification paths.
- `devl` facade crate re-exporting `devl-core` and feature-gated optional backends.

### Phase 2: `devl-candle` Whisper Backend

- Deterministic Whisper log-mel audio frontend (80-bin and 128-bin configurations) matching offline preprocessing from stream start.
- Stateless (`&self`, `Send + Sync`) Candle Whisper encoder and text decoder loaded from standard Hugging Face weight layouts.
- Explicit per-pass `WhisperKvCache` with cached cross-attention KV and truncatable self-attention KV (`rollback_kv`).
- Single-pass `verify_draft` and autoregressive `decode_step` implementations with configurable token suppression.
- Deterministic synthetic-weight tests in CI proving exact greedy parity between DEVL verification/rollback/resume and full offline decoding.

### Phase 3: Real-Audio Parity, Benchmarks, Examples, and Release

- Real-audio parity test suite validating exact greedy equivalence between streaming DEVL sessions and offline Whisper decoding on local fixtures.
- Runnable workspace examples demonstrating single-stream sessions and multi-stream `PriorityScheduler` execution.
- Reproducible local benchmark harness measuring TTFS, draft acceptance ratio, and exact token-match rate with explicit recording of hardware, model, dtype, command, and commit SHA.
- Public API audit, rustdoc completeness check, packaging verification, and `v0.1.0` publication to crates.io.

---

## v0.2.0 — Qwen3-ASR Backend & Network Server

Released on crates.io as [`devl-core` 0.2.0](https://crates.io/crates/devl-core/0.2.0), [`devl-candle` 0.2.0](https://crates.io/crates/devl-candle/0.2.0), and [`devl` 0.2.0](https://crates.io/crates/devl/0.2.0). This release adds Qwen3-ASR window and KV reuse and a local HTTP and WebSocket server. The server is distributed from the [`v0.2.0` source tag](https://github.com/njetty/devl-rs/tree/v0.2.0).

### Qwen3-ASR Backend in `devl-candle`

- Fixed-window audio encoder and cached prefix KV state reuse via `DevlBackend::encode_audio(..., prior_encoding)`.
- Qwen3 decoder KV-cache truncation with RoPE position offset alignment under `rollback_kv`.
- Parity test coverage verifying exact greedy equivalence under unchanged and invalidated normalization paths.

### `devl-server` Transport Layer

- Separate `crates/devl-server` workspace crate depending on `devl-core` and configured backends (keeping `devl-core` free of async and networking dependencies).
- WebSocket and HTTP streaming endpoints with per-connection session lifecycle management and stream multiplexing onto `PriorityScheduler`.
- Pluggable VAD and endpointing hooks driving periodic checkpoint drafts and final turn commits.

---

## v0.3.0 — Batched Execution, Model Expansion & Ecosystem

Scale concurrent inference throughput and broaden backend architecture coverage.

### Dynamic Batching

- Batched verification and decode execution across concurrent streams while preserving strict commit-over-draft scheduling semantics and documenting floating-point tie behavior.

### Additional Model Families & State Architectures

- Support for additional offline ASR model families under `DevlBackend`.
- `ReplaySSM` state-history rollback support for state-space and hybrid-attention decoders alongside transformer KV caches.

### Ecosystem Polish

- Structured telemetry hooks, extended client/server examples, and packaging refinements driven by downstream usage.
