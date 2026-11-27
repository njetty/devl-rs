# Contributing to `devl-rs`

## Crate Boundaries

Maintain the strict one-way workspace dependency graph documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):

- `devl-core` must remain free of ML runtime, async runtime, and networking dependencies.
- `devl-candle` implements `DevlBackend` only; verification, rollback selection, draft loop orchestration, and commit-over-draft scheduling belong in `devl-core`.
- `devl` is a thin facade re-exporting `devl-core` and feature-gated backends.

## Local Checks

Run the full check suite before opening a pull request:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p devl --all-targets --features candle,test-utils -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features devl/candle
cargo test --workspace
cargo test -p devl --features test-utils
```

On macOS Apple Silicon, also verify the Metal feature flag:

```bash
cargo check -p devl-candle --all-targets --features metal
cargo check -p devl --all-targets --features metal
```

## Engineering & Reporting Standards

- Keep public items documented (`missing_docs` is enabled workspace-wide) and avoid `unsafe` (`unsafe_code = "forbid"`).
- Do not include fabricated, extrapolated, or unmeasured latency/WER numbers in docs, PR descriptions, or comments. Only record metrics actually measured locally with reproducible commands.
- Keep documentation and comments concise and technical.

## License

All contributions are licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)).
