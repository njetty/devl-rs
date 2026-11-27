<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/njetty/devl-rs/main/assets/icon-ink.svg">
  <img src="https://raw.githubusercontent.com/njetty/devl-rs/main/assets/icon-paper.svg" width="48" height="48" alt="">
</picture>

# devl-rs

An independent, unofficial Rust implementation of **DEVL: Decode Early, Verify Late** by Doyeob Kim ([arXiv:2611.20064](https://arxiv.org/abs/2611.20064)). This project has no affiliation with, and is not endorsed by, the paper author or Nari Labs.

Offline speech recognition models typically achieve lower error rates than streaming-trained models, but running them only after turn end places the entire decode time on the critical path before a response can begin. `devl-rs` bridges this gap by running periodic speculative drafts over incoming audio and verifying the latest draft against the complete utterance in a single parallel forward pass at commit time to reduce time-to-final-segment (TTFS). With equivalent deterministic inference, the final tokens match offline greedy decoding.

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
          | (trait, session,  |     | (Candle Whisper   |
          |  scheduler, mock) |     |  backend)         |
          +-------------------+     +-------------------+
```

| Crate | Responsibility | Features |
| :--- | :--- | :--- |
| `devl` | Top-level facade re-exporting `devl-core` (`devl::*`, `devl::core`) and optional backends (`devl::candle`). | `candle`, `metal`, `cuda`, `test-utils` |
| `devl-core` | Runtime- and transport-agnostic core crate (zero ML or networking dependencies). Owns `DevlBackend`, `DevlSession`, single-pass verification and KV rollback, `PriorityScheduler`, and `MockBackend`. | `test-utils` |
| `devl-candle` | Local inference backend over `candle-core`, `candle-nn`, and `candle-transformers` for Whisper. | `metal`, `cuda` |

See [docs/ARCHITECTURE.md](https://github.com/njetty/devl-rs/blob/main/docs/ARCHITECTURE.md) for trait signatures, exactness invariants, session states, and scheduler preemption rules, and [docs/ROADMAP.md](https://github.com/njetty/devl-rs/blob/main/docs/ROADMAP.md) for the multi-release plan.

## Usage

Depend on the top-level `devl` facade crate (optionally enabling `candle`, `metal`, `cuda`, or `test-utils`), or depend directly on `devl-core` when implementing a custom inference backend:

```toml
# Facade crate (core only by default)
[dependencies]
devl = "0.1.0"

# Facade crate with Candle backend (or features = ["metal"] / ["cuda"])
# devl = { version = "0.1.0", features = ["candle"] }

# Core protocol only (zero ML dependencies)
# devl-core = "0.1.0"
```

On ARM64 Linux, add the following to your application's workspace root
`Cargo.toml` if you enable Candle. It avoids an upstream FP16 assembly build
error in unoptimized dev builds.

```toml
[profile.dev.package.gemm-f16]
opt-level = 3

[profile.dev.package.gemm-common]
opt-level = 3
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

Both `file_stream` and `bench_ttfs` accept an optional
`--tokenizer <path>` flag pointing at a Whisper `tokenizer.json`. With it,
the emitted JSON adds a `transcript` field alongside the tokens: `file_stream`
adds it to each draft event and to the final event, and `bench_ttfs` adds
`offline_transcript` and `devl_transcript` to each per-fixture sample.
Token-only invocations are unchanged.

`bench_ttfs` also accepts an optional `--references <path>` flag pointing at a
JSON object mapping WAV file basenames to reference transcripts (for example
`{"sample.wav": "the reference transcript"}`). It requires `--tokenizer`.
When both flags are given, each per-fixture sample additionally reports
`reference_words`, `offline_wer`, and `devl_wer`, and the summary reports
`reference_words`, `offline_wer_errors`, `devl_wer_errors`, `offline_wer`,
and `devl_wer` across the corpus.

WER is normalized and aggregated as follows. Reference and hypothesis text are
lowercased and split on any non-alphanumeric character into word tokens (so
punctuation and case differences do not affect the score). Per fixture, the
word error rate is the word-level Levenshtein distance (substitutions +
deletions + insertions) divided by the reference word count; a fixture with
zero reference words contributes no WER score and is not added to the corpus
terms. The corpus score is the total word error count over the total reference
word count across all fixtures (micro averaging), not a macro average of
per-fixture scores. A reference missing for a fixture is a hard error rather
than a skipped sample.

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
cargo package --list -p devl-candle
```

## Measured Whisper TTFS

Measured with `bench_ttfs` on 12 OpenSLR LibriSpeech test-clean clips (the
first sorted utterance for each of the first 12 sorted speakers) prepared
under `/tmp/devl20-fixtures.RnKUTI` (12 WAVs, `references.json` mapping WAV
basenames to reference transcripts, `model/` containing
`openai/whisper-tiny.en` config, weights, and tokenizer), in the current
checkout of this repository, on a fixed ARM64 Linux host (20 logical CPUs
reported as `Cortex-A725`/`Cortex-X925`, kernel `7.0.0-1025-nvidia`, F32
inference, no Metal, no CUDA). Model identifier `openai/whisper-tiny.en`
(`config.json` git blob `31c9f36`, `model.safetensors` git blob `29e1288`).
The fixture WAVs are 16 kHz mono PCM16, at most 30 seconds long. The command
was (run from the repository root, against a clean checkout):

```bash
cargo run --release -p devl --features candle --example bench_ttfs -- \
  cpu /tmp/devl20-fixtures.RnKUTI/model \
  /tmp/devl20-fixtures.RnKUTI/wav/*.wav \
  --tokenizer /tmp/devl20-fixtures.RnKUTI/model/tokenizer.json \
  --references /tmp/devl20-fixtures.RnKUTI/references.json
```

Measured at commit `3e10c1f8ba515a38ec3a753d9e330e4b9fa81027` (HEAD at measure
time; `git status` clean). Median and p99 are over the 12 clips (median
averaging the two middle observations for an even sample count;
nearest-rank p99).

| Metric | Offline greedy | DEVL |
| :--- | :--- | :--- |
| TTFS median (ms) | 1475.0 | 331.5 |
| TTFS p99 (ms) | 2816.1 | 880.0 |

Speedup is **4.45x** on median TTFS (`1475.0 / 331.5`) and **3.20x** on p99
(`2816.1 / 880.0`).

| Summary metric | Value |
| :--- | :--- |
| Sample count | 12 |
| Accepted prefix ratio (accepted / draft tokens) | 311 / 404 = 0.770 (77.0%) |
| Exact token-match rate | 12 / 12 = 100.0% |
| Offline WER | 0.0506 (17 errors / 336 reference words) |
| DEVL WER | 0.0506 (17 errors / 336 reference words) |

Offline and DEVL WER are identical by construction: the exact token-match rate
is 1.0, so both decoders produce the same token stream, and both are scored
through the same tokenizer and normalization. The value is a WER on the
reference transcripts, not a claim that the model is good at this task.

`file_stream` transcripts from the same model are also available with the
optional `--tokenizer` flag; for the `2300-131720-0000.wav` clip it emitted
the final transcript `"The Paris plant, like that at the Crystal Palace, was a
temporary exhibit."`, matching its reference transcript.

`openai/whisper-large-v3-turbo` was not run: no compatible weights were
prepared in the fixture directory at measurement time.

## Status

- **Implemented:** `devl-core` (`DevlBackend` trait, `DevlSession` state machine, single-pass verify/rollback/resume loop, `PriorityScheduler` with step-level cooperative preemption, and deterministic `MockBackend` test suite) and the `devl` facade crate.
- **Implemented:** Local Whisper inference over Candle, deterministic mel extraction, per-layer KV-cache truncation, and local streaming examples.
- **Planned (`v0.2`):** Qwen3-ASR inference over Candle.
- **Planned (`v0.2`):** `devl-server` transport layer (WebSocket/HTTP, stream multiplexing, and VAD/endpointing hooks).

## Contributing

See [CONTRIBUTING.md](https://github.com/njetty/devl-rs/blob/main/CONTRIBUTING.md) for setup, local verification, pull request rules, and signed-commit requirements. Contributors and coding agents must also follow [AGENTS.md](https://github.com/njetty/devl-rs/blob/main/AGENTS.md) for workspace boundaries, issue execution workflow, and clean-room documentation rules.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). See [NOTICE](NOTICE) for attribution.
