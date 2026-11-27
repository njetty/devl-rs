# Changelog

All notable changes to this workspace are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Multi-platform GitHub Actions CI covering `linux-x86_64`, `linux-arm64`, `macos-arm64` (including `--features metal`), and `windows-x86_64`.
- Tag-triggered crates.io Trusted Publishing release workflow (`.github/workflows/release.yml`) and Dependabot configuration.
- `docs.rs` feature metadata, stricter workspace lints (`unsafe_code = "forbid"`, `missing_docs = "warn"`), `rustfmt.toml`, and `CONTRIBUTING.md`.

## [0.1.0-alpha.1] - 2026-11-26

### Added
- `devl-core`: `DevlBackend` trait, `DevlSession` state machine (`Idle`, `Drafting`, `Committing`, `Finalized`, `Failed`), single-pass draft verification with longest-prefix acceptance and KV-cache rollback, final-over-draft `PriorityScheduler` with step-level cooperative preemption, and deterministic `MockBackend` (`test-utils` feature).
- `devl-candle`: Crate scaffolding, `CandleWhisperBackend` trait skeleton, and optional `metal`/`cuda` feature flags.
- `devl`: Top-level facade crate re-exporting `devl-core` and feature-gated `devl-candle`.
- `docs/ARCHITECTURE.md`: Protocol specification, exactness invariants, scheduler rules, and backend design notes.
