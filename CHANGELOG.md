# Changelog

All notable changes to this workspace are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

### Added
- `devl-core`: defaulted `DevlBackend::is_eos_token` predicate so models with more than one stop id stop correctly; all stop comparisons in `devl-core` use it.
- `devl-candle`: Qwen3-ASR backend (`qwen3_asr` module) with the audio frontend, fixed-window encoder, `Qwen3AsrBackend` implementing `DevlBackend` with reusable audio windows and prefix KV, a configurable input cap (120 s by default), and offline greedy parity validation against the pinned `Qwen/Qwen3-ASR-1.7B` revision (F32, CPU; scope and limits in `docs/ARCHITECTURE.md` Section 6).
- `devl-candle`: Qwen3-ASR causal text decoder with prefill, incremental greedy step, absolute RoPE positions, a forkable and truncatable KV cache, and selective pinned local loading of the decoder weights.
- `devl-server`: pluggable per-stream VAD and endpointing hooks (`EndpointPolicy`, `Coordinator::with_endpointing`) that can request checkpoint drafts and end the turn, a deterministic `SilenceEndpointPolicy`, and a shared contract transition so manual and automatic commit cannot finalize the same turn twice.
- `devl-server`: a local `devl-server` binary (mock, Whisper, and Qwen3-ASR backends; HTTP and WebSocket on one coordinator; optional silence endpointing), a `stream_client` reference example, a CI process-level smoke test with the mock backend, and opt-in real-model smoke tests that compare final token ids with a direct session over both transports.
- `devl-server`: versioned streaming contract (`docs/SERVER_CONTRACT.md`), a validating per-stream state machine, a session coordinator that drives one `DevlSession` per stream through `PriorityScheduler` on a bounded worker pool, and WebSocket and HTTP adapters over that coordinator.

### Changed
- `devl-candle` crate description now names Qwen3-ASR.
- `devl-server` is source-distributed for this release (`publish = false`): run it from the `v0.2.0` tag as described in the README. The release workflow publishes `devl-core`, `devl-candle`, and `devl` only.
- `Cargo.lock` is not committed, so builds resolve dependencies to the newest versions the manifests allow.

## [0.1.0]

### Added
- Deterministic Whisper log-mel extraction with 80- and 128-bin filter banks and a 30-second input limit.
- Local Candle Whisper inference from Hugging Face weights, with causal draft verification, configurable token suppression, and per-layer KV cache rollback.
- Synthetic-weight and opt-in real-audio greedy parity tests, plus concurrent scheduler stress tests.
- Single-stream and multi-stream WAV replay examples, and a TTFS and token-parity benchmark that records run provenance.
- DEVL marks in the repository README and crate documentation, with license and attribution files in all three crate archives.
- Multi-platform GitHub Actions CI covering `linux-x86_64`, `linux-arm64`, `macos-arm64` (including `--features metal`), and `windows-x86_64`.
- Tag-triggered crates.io Trusted Publishing release workflow (`.github/workflows/release.yml`) and Dependabot configuration.
- `docs.rs` feature metadata, stricter workspace lints (`unsafe_code = "forbid"`, `missing_docs = "warn"`), `rustfmt.toml`, and `CONTRIBUTING.md`.

## [0.1.0-alpha.1] - 2026-11-26

### Added
- `devl-core`: `DevlBackend` trait, `DevlSession` state machine (`Idle`, `Drafting`, `Committing`, `Finalized`, `Failed`), single-pass draft verification with longest-prefix acceptance and KV-cache rollback, final-over-draft `PriorityScheduler` with step-level cooperative preemption, and deterministic `MockBackend` (`test-utils` feature).
- `devl-candle`: Crate scaffolding, `CandleWhisperBackend` trait skeleton, and optional `metal`/`cuda` feature flags.
- `devl`: Top-level facade crate re-exporting `devl-core` and feature-gated `devl-candle`.
- `docs/ARCHITECTURE.md`: Protocol specification, exactness invariants, scheduler rules, and backend design notes.
