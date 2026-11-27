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

## Local Whisper examples

Use a local Hugging Face Whisper checkpoint directory with `config.json` and
a single `model.safetensors` file. For example, use `openai/whisper-tiny.en` for
English audio. The examples use F32 inference, a built-in Slaney mel filter
bank, and an English transcription prompt. Each input must be a nonempty
16 kHz mono PCM16 or IEEE Float32 WAV, at most 30 seconds long.

Run from the repository root with paths to your local files:

```bash
model=/path/to/whisper-tiny.en
wav=/path/to/sample_16k_mono.wav
other_wav=/path/to/another_16k_mono.wav

cargo run --release -p devl --features candle --example file_stream -- cpu "$model" "$wav"
cargo run --release -p devl --features candle --example multi_stream -- cpu "$model" "$wav" "$other_wav"
cargo run --release -p devl --features candle --example bench_ttfs -- cpu "$model" "$wav" "$other_wav" > ../ttfs.json
```

On Apple Silicon, use `--features metal` and pass `metal` in place of `cpu`
to run on the first Metal device.

- `file_stream` replays 100 ms chunks as fast as the session can process them.
  It drafts every two seconds of audio before the final chunk, with an earlier
  checkpoint for shorter inputs spanning multiple chunks. It emits JSON lines
  containing draft tokens and final tokens with verified prefix statistics.
- `multi_stream` shares the model across two or more sessions and uses one
  inference worker. The replay producer submits commits while the first draft
  is at a decode-step boundary, allowing the running draft to yield and pending
  drafts for ending streams to be evicted. A draft that reaches EOS in prefill
  can complete before the producer submits the commits. Each stream's TTFS
  includes its queue wait and delivery of its remaining audio to its session.
- `bench_ttfs` measures one trial per WAV argument in order. Repeat paths to
  collect repeated trials. It runs the uncached offline greedy reference first
  on each fixture, then replays the same PCM through DEVL with checkpoint
  promotion disabled. Offline timing starts with full PCM available. DEVL
  timing starts after the final chunk has been pushed. Both stop when the final
  tokens are available and the device has synchronized. Model and WAV loading
  are outside the measurements. The harness reports draft work separately.

The benchmark emits one JSON object with the executable argument vector,
working directory, selected environment settings, checkout SHA and status,
hardware, model, device, dtype, content identifiers (Git blob hashes) for
model files and fixtures, and per-trial results. Run the Cargo command
above from a clean checkout to tie the binary to the recorded SHA. The summary
reports the median (averaging the two middle observations for an even sample
count), nearest-rank p99, accepted tokens divided by total verified draft
tokens (`null` when no draft tokens were verified), and exact token-match
rate. A parity mismatch produces the JSON report and a nonzero exit status.

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
- **Implemented:** Local Whisper inference over Candle, deterministic mel extraction, per-layer KV-cache truncation, and local streaming examples.
- **Planned (`v0.2`):** Qwen3-ASR inference over Candle.
- **Planned (`v0.2`):** `devl-server` transport layer (WebSocket/HTTP, stream multiplexing, and VAD/endpointing hooks).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, local verification, pull request rules, and signed-commit requirements. Contributors and coding agents must also follow [AGENTS.md](AGENTS.md) for workspace boundaries, issue execution workflow, and clean-room documentation rules.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). See [NOTICE](NOTICE) for attribution.
