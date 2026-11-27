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
`offline_transcript` and `devl_transcript` to each per-fixture sample. Each
hypothesis is decoded once and the same text is reused for emission and WER
scoring. Token-only invocations are unchanged.

The scoped flags are command-specific: `file_stream` accepts `--tokenizer` only,
`multi_stream` accepts neither flag, and `bench_ttfs` accepts both. A command
that does not accept a flag rejects it; unknown options, repeated flags, and
flags with a missing value are rejected as well. Legacy positional
invocations (`<cpu|metal> <model-dir> <audio.wav>...`) are unchanged.

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
deletions + insertions) divided by the reference word count. The corpus score
is the total word error count over the total reference word count across all
fixtures (micro averaging), not a macro average of per-fixture scores. A
reference missing for a fixture, a reference that normalizes to no words, or
two distinct fixture paths sharing one WAV basename are all hard errors rather
than skipped samples; deliberate repeats of the same path are allowed.

The benchmark emits one JSON object with the executable argument vector,
working directory, selected environment settings, checkout SHA and status,
hardware, model, device, dtype, content identifiers (Git blob hashes) for
model files and fixtures — plus the `tokenizer.json` and the references file
when those flags are used — and per-trial results. Run the Cargo command
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
first sorted utterance for each of the first 12 sorted speakers). Source: the
official OpenSLR LibriSpeech corpus (<https://www.openslr.org/12/>,
`test-clean.tar.gz`). Conversion: the selected utterances were decoded from the official test-clean
audio into 16 kHz mono PCM16 16-bit WAVs (LibriSpeech audio is already 16 kHz;
the clips were kept at most 30 seconds long), with one `references.json`
mapping WAV basenames to the LibriSpeech reference transcripts. The 12 fixtures (WAV basenames) are
`1089-134686-0000.wav`, `1188-133604-0000.wav`, `121-121726-0000.wav`,
`1221-135766-0000.wav`, `1284-1180-0000.wav`, `1320-122612-0000.wav`,
`1580-141083-0000.wav`, `1995-1826-0000.wav`, `2094-142345-0000.wav`,
`2300-131720-0000.wav`, `237-126133-0000.wav`, and `260-123286-0000.wav`.
The prepared corpus lives in a temporary fixture directory
`/tmp/devl20-fixtures.RnKUTI` (12 WAVs, `references.json`, and `model/`
containing the `openai/whisper-tiny.en` config, weights, and tokenizer) so
the exact command below is reproducible; the clips, selection, and conversion
are documented above rather than taken from the path alone. Model identifier
`openai/whisper-tiny.en` (`config.json` git blob `31c9f36`,
`model.safetensors` git blob `29e1288`, `tokenizer.json` git blob `b5bd266`).
The clips are 16 kHz mono PCM16, at most 30 seconds long. The command was run
from the repository root, against a clean checkout, on a fixed ARM64 Linux
host (20 logical CPUs reported as `Cortex-A725`/`Cortex-X925`, kernel
`7.0.0-1025-nvidia`, F32 inference, no Metal, no CUDA). Measured at commit
`d1b64e3451d476aa380eafac61311bb03d2c24fa` (HEAD at measure time; `git status`
clean). Median and p99 are over the 12 clips (median averaging the two
middle observations for an even sample count; nearest-rank p99, which for 12
samples is the maximum observation).

```bash
cargo run --release -p devl --features candle --example bench_ttfs -- \
  cpu /tmp/devl20-fixtures.RnKUTI/model \
  /tmp/devl20-fixtures.RnKUTI/wav/*.wav \
  --tokenizer /tmp/devl20-fixtures.RnKUTI/model/tokenizer.json \
  --references /tmp/devl20-fixtures.RnKUTI/references.json
```

`file_stream` transcripts from the same model are also available with the
optional `--tokenizer` flag:

```bash
cargo run --release -p devl --features candle --example file_stream -- \
  cpu /tmp/devl20-fixtures.RnKUTI/model \
  /tmp/devl20-fixtures.RnKUTI/wav/2300-131720-0000.wav \
  --tokenizer /tmp/devl20-fixtures.RnKUTI/model/tokenizer.json
```

For the `2300-131720-0000.wav` clip, `file_stream` emits the final transcript
`"The Paris plant, like that at the Crystal Palace, was a temporary
exhibit."`, matching its reference transcript.

| Metric | Offline greedy | DEVL |
| :--- | :--- | :--- |
| TTFS median (ms) | 1479.0 | 334.9 |
| TTFS p99 (ms) | 2787.1 | 887.9 |

Speedup is **4.42x** on median TTFS (`1479.0 / 334.9`) and **3.14x** on p99
(`2787.1 / 887.9`).

| Summary metric | Value |
| :--- | :--- |
| Sample count | 12 |
| Accepted prefix ratio (accepted / draft tokens) | 311 / 404 = 0.770 (77.0%) |
| Exact token-match rate | 12 / 12 = 100.0% |
| Offline WER | 0.0506 (17 errors / 336 reference words) |
| DEVL WER | 0.0506 (17 errors / 336 reference words) |

The benchmark runs drafts synchronously during audio replay and records their
cost separately. TTFS measures only the wait after the final chunk is
available. With 12 clips, the nearest-rank p99 is the slowest observation.
Both paths produce identical tokens on this sample, so their WER is also
identical.

`openai/whisper-large-v3-turbo` was not run: no compatible weights were
prepared in the fixture directory at measurement time.

## Status

- **Implemented:** `devl-core` (`DevlBackend` trait, `DevlSession` state machine, single-pass verify/rollback/resume loop, `PriorityScheduler` with step-level cooperative preemption, and deterministic `MockBackend` test suite) and the `devl` facade crate.
- **Implemented:** Local Whisper inference over Candle, deterministic mel extraction, per-layer KV-cache truncation, and local streaming examples.
- **In progress (`v0.2`):** Qwen3-ASR over Candle. The windowed audio encoder, decoder, and `DevlBackend` integration with reusable audio windows and prefix KV are implemented. Exact greedy token parity with the pinned upstream checkpoint is validated locally on five clips (`f32`, CPU), see `docs/ARCHITECTURE.md` Section 6.16 for scope and limits.
- **In progress (`v0.2`):** `devl-server`. The transport-neutral streaming contract and per-stream state machine are implemented (see `docs/SERVER_CONTRACT.md`). Planned: WebSocket/HTTP adapters, stream multiplexing onto `PriorityScheduler`, and VAD/endpointing hooks.

## Contributing

See [CONTRIBUTING.md](https://github.com/njetty/devl-rs/blob/main/CONTRIBUTING.md) for setup, local verification, pull request rules, and signed-commit requirements. Contributors and coding agents must also follow [AGENTS.md](https://github.com/njetty/devl-rs/blob/main/AGENTS.md) for workspace boundaries, issue execution workflow, and clean-room documentation rules.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). See [NOTICE](NOTICE) for attribution.
