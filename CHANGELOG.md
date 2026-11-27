# Changelog

All notable changes to this workspace are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
