## Summary

- What changes for users or callers, and why.
- Link any relevant issue or section in `docs/ARCHITECTURE.md`.

## Verification

List the exact commands run locally (and platform/features if relevant):

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p devl --features test-utils
```

## Readiness checklist

Required CI checks are enforced live on `HEAD` by GitHub's `Protect main` ruleset; check off the pre-push items below before opening a draft pull request:

- [ ] Every commit has a GitHub-verified cryptographic signature (`required_signatures`).
- [ ] Pull request title is a plain imperative sentence with no bracketed prefix (a single `type:` label is set by a maintainer).
- [ ] Crate boundaries preserved (`devl-core` stays free of ML/async/network dependencies; protocol logic lives in `devl-core`).
- [ ] Local verification gate passed (`cargo fmt --check`, `cargo clippy`, `cargo test`, and `cargo doc`).
- [ ] Public items are documented (`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features devl/candle`) and `docs/ARCHITECTURE.md` is updated if contracts changed.
- [ ] Any reported latency, TTFS, or WER numbers were measured locally with reproducible commands (no fabricated metrics).
