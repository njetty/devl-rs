## Summary

- What changed and why (link any relevant issue or design section in `docs/ARCHITECTURE.md`).

## Checklist

- [ ] Commits are cryptographically signed (SSH, GPG, or S/MIME verified on GitHub).
- [ ] Crate boundaries preserved (`devl-core` stays free of ML/async/network dependencies; protocol logic stays in `devl-core`).
- [ ] `cargo fmt --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [ ] `cargo test --workspace` and `cargo test -p devl --features test-utils` pass.
- [ ] Public items are documented (`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features devl/candle`).
- [ ] Any reported metrics (latency, TTFS, WER) were measured locally with reproducible commands.
