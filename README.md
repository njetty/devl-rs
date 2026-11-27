# devl-rs

An independent, unofficial Rust implementation of **DEVL: Decode Early, Verify Late** by Doyeob Kim ([arXiv:2611.20064](https://arxiv.org/abs/2611.20064)). This project has no affiliation with, and is not endorsed by, the paper author or Nari Labs.

Offline speech recognition models typically achieve lower error rates than streaming-trained models, but running them only after turn end places the entire decode time on the critical path before a response can begin. `devl-rs` bridges this gap by running periodic speculative drafts over incoming audio and verifying the latest draft against the complete utterance in a single parallel forward pass at commit time, preserving exact greedy offline output while reducing time-to-final-segment (TTFS).

## Workspace Architecture

```text
                       +-------------------+
                       |       devl        |
                       |   (facade crate)  |
                       +-------------------+
                          /             \
                         v               v (optional: candle / metal / cuda)
          +-------------------+     +-------------------+
          |     devl-core     | <-- |    devl-candle    |
          | (trait, session,  |     | (Candle Whisper / |
          |  scheduler, mock) |     |  Qwen3-ASR impl)  |
          +-------------------+     +-------------------+
```

| Crate | Responsibility | Features |
| :--- | :--- | :--- |
| `devl` | Top-level facade re-exporting `devl-core` (`devl::*`, `devl::core`) and optional backends (`devl::candle`). | `candle`, `metal`, `cuda`, `test-utils` |
| `devl-core` | Runtime- and transport-agnostic core crate (zero ML or networking dependencies). Owns `DevlBackend`, `DevlSession`, single-pass verification and KV rollback, `PriorityScheduler`, and `MockBackend`. | `test-utils` |
| `devl-candle` | Local inference backend over `candle-core`, `candle-nn`, and `candle-transformers` for Whisper and Qwen3-ASR. | `metal`, `cuda` |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for trait signatures, exactness invariants, session states, and scheduler preemption rules, and [docs/ROADMAP.md](docs/ROADMAP.md) for the multi-release plan.

## Usage

Depend on the top-level `devl` facade crate (optionally enabling `candle`, `metal`, `cuda`, or `test-utils`), or depend directly on `devl-core` when implementing a custom inference backend:

```toml
# Facade crate (core only by default)
[dependencies]
devl = "0.1.0-alpha.1"

# Facade crate with Candle backend (or features = ["metal"] / ["cuda"])
# devl = { version = "0.1.0-alpha.1", features = ["candle"] }

# Core protocol only (zero ML dependencies)
# devl-core = "0.1.0-alpha.1"
```

## Build and Test

```bash
# Check formatting
cargo fmt --check

# Lint all workspace targets
cargo clippy --workspace --all-targets -- -D warnings

# Run workspace unit, integration, and doc tests
cargo test --workspace

# Test facade crate with deterministic mock utilities enabled
cargo test -p devl --features test-utils

# Verify facade crate compiles with the Candle backend feature enabled
cargo check -p devl --features candle

# Inspect crates.io package contents
cargo package --list -p devl
cargo package --list -p devl-core
```

## Status

Status: `v0.1-alpha` in active development.

- **Implemented:** `devl-core` (`DevlBackend` trait, `DevlSession` state machine, single-pass verify/rollback/resume loop, `PriorityScheduler` with step-level cooperative preemption, and deterministic `MockBackend` test suite) and the `devl` facade crate.
- **In progress (Phase 2):** `devl-candle` Whisper and Qwen3-ASR inference implementations, deterministic mel extraction, and per-layer KV-cache truncation (currently compiling trait stubs).
- **Planned (`v0.2`):** `devl-server` transport layer (WebSocket/HTTP, stream multiplexing, and VAD/endpointing hooks).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, local verification, pull request rules, and signed-commit requirements. Contributors and coding agents must also follow [AGENTS.md](AGENTS.md) for workspace boundaries, issue execution workflow, and clean-room documentation rules.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). See [NOTICE](NOTICE) for attribution.
