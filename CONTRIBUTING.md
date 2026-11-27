# Contributing to `devl-rs`

`devl-rs` is an independent, unofficial open-source Rust implementation of **DEVL: Decode Early, Verify Late** ([arXiv:2611.20064](https://arxiv.org/abs/2611.20064)), developed in public at [njetty/devl-rs](https://github.com/njetty/devl-rs). Pull requests from outside contributors are welcome and go through the same CI checks as maintainer changes. Only maintainers can push branches directly to the repository, manage labels, or trigger manual workflows. Outside contributors work from a fork and open pull requests against `main`.

Architectural boundaries, trait contracts, exactness invariants, and scheduler preemption rules live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The high-level multi-release plan lives in [docs/ROADMAP.md](docs/ROADMAP.md), and the contributor/agent execution contract lives in [AGENTS.md](AGENTS.md). Read those documents before changing code, whether you write the change yourself or with a coding agent.

## Report a problem

[Choose an issue template](https://github.com/njetty/devl-rs/issues/new/choose) (defined in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/)):

- **Feature / Task** (`feature.yml`) for atomic implementation work with explicit goal, scope, non-goals, dependencies, and acceptance criteria.
- **Bug Report** (`bug.yml`) for incorrect behavior, exactness regressions, or build failures.
- **Milestone / Release Tracking** (`milestone_release.yml`) for release exit criteria and pre-release checklists.

When filing a bug report, describe what you ran, what you expected, and what happened instead, including:

- Output of `rustc --version` and `cargo --version`.
- Target platform and feature flags (for example, `linux-x86_64`, `linux-arm64`, `macos-arm64 --features metal`, or `windows-x86_64`).
- Minimal reproduction code or test case (prefer a deterministic `MockBackend` test when reporting a `devl-core` protocol or scheduler bug).

Issues are public; remove credentials, private paths, and proprietary audio transcripts before posting. For security vulnerabilities, use GitHub's private vulnerability reporting on the repository rather than opening a public issue.

## Set up a checkout

You need:

- A stable Rust toolchain supporting Rust edition 2024 (`rustup update stable`) with `rustfmt` and `clippy` components installed (`rustup component add rustfmt clippy`).
- Git configured for commit signature verification (see [Open a pull request](#open-a-pull-request)).
- Platform-specific toolchains only when testing optional GPU features: Xcode command-line tools on macOS Apple Silicon for `--features metal`, or the NVIDIA CUDA Toolkit (`nvcc`) for `--features cuda`.

Clone the repository (or your fork) and verify the workspace builds and tests cleanly:

```bash
git clone https://github.com/njetty/devl-rs.git
cd devl-rs
cargo test --workspace
cargo test -p devl --features test-utils
```

### Platform notes

- **`linux-arm64` (`aarch64-unknown-linux-gnu`):** Upstream `gemm-f16` (pulled transitively by `candle-core`) fails to compile in `opt-level = 0` debug builds on generic ARM64 Linux targets. The workspace root [Cargo.toml](Cargo.toml) sets `opt-level = 3` for `gemm-f16` and `gemm-common` under `[profile.dev.package.*]`; do not remove those overrides unless upstream `gemm-f16` resolves the issue.
- **`macos-arm64`:** Metal compilation can be verified locally with `cargo check -p devl-candle --all-targets --features metal` and `cargo check -p devl --all-targets --features metal`.
- **`cuda`:** Standard GitHub-hosted CI runners do not provide GPUs; verify `--features cuda` locally on a CUDA-capable host before changing CUDA code paths.

## Make a change

Start every change from the latest `main`, and keep each pull request focused on one change. Put unrelated formatting, renames, or dependency upgrades in separate pull requests so each change can be reviewed and reverted independently.

Before implementing a non-trivial feature, confirm that the design answers:

1. Which crate owns the behavior (`devl-core`, `devl-candle`, or `devl`)?
2. Does it preserve the one-way dependency graph (`devl` → `devl-core`, `devl-candle` → `devl-core`)?
3. Does it preserve exact greedy offline equivalence under deterministic inputs and kernels?
4. Which tests, rustdoc updates, and `docs/ARCHITECTURE.md` changes land with it?

If any of these are unsettled, open an issue or draft design note before writing code.

These rules apply to every change:

- **Crate boundaries:** `crates/devl-core` must have zero ML runtime, async runtime, or networking dependencies. Verification, prefix matching, KV-rollback selection, draft loop orchestration, and commit-over-draft scheduling belong in `devl-core`, never duplicated in backend crates.
- **Facade discipline:** `crates/devl` only re-exports `devl-core` and feature-gated backend crates; it holds no protocol or inference implementation logic.
- **Workspace lints:** `unsafe_code = "forbid"`, `missing_docs = "warn"`, `rust_2018_idioms = "warn"`, and `clippy::all = "warn"` apply workspace-wide and run with `-D warnings` in CI. Document every public type, field, variant, method, and module.
- **Dependencies:** Avoid adding new crates to `devl-core`. Justify any new dependency in `devl-candle` in the pull request description.
- **No fabricated metrics:** Never add invented, extrapolated, or unmeasured latency, TTFS, throughput, or WER numbers in docs, comments, or pull request descriptions. Only report metrics actually measured locally alongside the exact commands and hardware used.
- **No untracked artifacts:** Do not commit `Cargo.lock`, `/target`, downloaded model weights, or machine-specific home-directory paths.
- **No speculative docs:** Do not document behavior that has not been implemented yet unless clearly marked as planned status.

If a coding agent assists with your change, point it at [AGENTS.md](AGENTS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md), and this file. Keep all prose (commit messages, pull request titles and descriptions, rustdoc, and markdown docs) dry, technical, and concise.

## Test your change

Run the full local verification suite before pushing:

```bash
# Formatting (must match rustfmt.toml)
cargo fmt --check

# Clippy across default and optional facade features
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p devl --all-targets --features candle,test-utils -- -D warnings

# Rustdoc with warnings denied
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features devl/candle

# Unit, integration, and doc tests
cargo test --workspace
cargo test -p devl --features test-utils

# Feature compilation checks
cargo check -p devl --all-targets --features candle

# Verify crates.io package manifests stay clean
cargo package --list -p devl-core --allow-dirty
cargo package --list -p devl-candle --allow-dirty
cargo package --list -p devl --allow-dirty
```

On macOS Apple Silicon, also run:

```bash
cargo check -p devl-candle --all-targets --features metal
cargo check -p devl --all-targets --features metal
```

### Exactness testing requirements

- Changes to `DevlSession` or verification/rollback logic in `devl-core` must maintain 100% token-exact equivalence against `MockBackend::reference_offline_decode` across all four paths: (a) 100% draft match (with and without EOS), (b) partial prefix mismatch with KV rollback, (c) zero-match draft, and (d) empty draft.
- Changes to `PriorityScheduler` must preserve strict commit-over-draft ordering, bounded concurrent drafts, per-stream coalescing/eviction, and step-level cooperative preemption.
- Changes to `devl-candle` decoders must include deterministic unit tests (using synthetic weights so CI runs without network downloads) verifying exact greedy parity between `verify_draft` + `rollback_kv` + `decode_step` and full offline decoding.

## Open a pull request

1. **Sign every commit before pushing.** The `Protect main` ruleset enforces verified cryptographic signatures (`required_signatures`) on `main`; unsigned commits block merging. Set up [GitHub commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification) with an SSH, GPG, or S/MIME key. For SSH signing:

   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global commit.gpgsign true
   ```

   Add the corresponding public key under **GitHub Settings → SSH and GPG keys** as a **Signing Key** (separate from an Authentication Key).
2. **Push your branch.** Maintainers push feature branches to `njetty/devl-rs`. Outside contributors push to their fork and open a pull request against `main`.
3. **Open as a draft first.** Once the local verification gate passes, open a draft pull request (`gh pr create --draft`) so the multi-platform CI matrix starts early and the diff can be reviewed before marking ready.
4. **Write a plain imperative title.** For example: `Implement Whisper self-attention KV cache and rollback`. Omit bracketed prefixes such as `[feat]` or `[bug]`—use a `type:` label instead.
5. **Complete the pull request template.** Explain what changes for callers and why, list the exact local verification commands run, and check off the pre-push items in the readiness checklist. Live CI status is enforced on GitHub rather than via a manual PR-body checkbox.

### Choose a label

Every pull request and issue needs exactly one `type:` label before merge:

| Label | Use when the pull request or issue |
| :--- | :--- |
| `type: bug` | Fixes incorrect behavior |
| `type: feature` | Adds a new user-facing capability |
| `type: improvement` | Improves existing behavior or performance |
| `type: docs` | Changes documentation only |
| `type: maintenance` | Changes internal tooling, dependencies, CI, or code structure without changing public behavior |
| `type: release` | Prepares or repairs a crates.io release |
| `type: security` | Fixes or hardens a security boundary |

Apply `crate:` scope labels alongside the `type:` label whenever an issue or pull request targets specific workspace crates:

| Label | Use when changes affect |
| :--- | :--- |
| `crate: devl-core` | `crates/devl-core` |
| `crate: devl-candle` | `crates/devl-candle` |
| `crate: devl` | `crates/devl` facade |

Maintainers assign labels when opening a pull request or triaging an issue and update them if the scope changes. Outside contributors leave labels unset on pull requests; a maintainer adds them during review.

## CI checks

CI (`.github/workflows/ci.yml`) runs on every push to `main` and every pull request (including drafts). A passing result counts only for the current `HEAD` commit; pushing a new commit requires a fresh passing run.

All six checks below are required by the `Protect main` ruleset before a pull request can squash-merge:

| Check | Runner | What it runs |
| :--- | :--- | :--- |
| `Format, Clippy, Docs` | `ubuntu-latest` | `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo clippy -p devl --all-targets --features candle,test-utils -- -D warnings`, and `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features devl/candle` |
| `Test (linux-x86_64)` | `ubuntu-latest` | `cargo test --workspace`, `cargo test -p devl --features test-utils`, `cargo check -p devl --all-targets --features candle` |
| `Test (linux-arm64)` | `ubuntu-24.04-arm` | Same test and `candle` feature checks on native Linux ARM64 |
| `Test (macos-arm64)` | `macos-latest` | Same test and `candle` checks plus `cargo check --all-targets --features metal` on `devl-candle` and `devl` |
| `Test (windows-x86_64)` | `windows-latest` | Same test and `candle` feature checks on Windows MSVC |
| `Package Check` | `ubuntu-latest` | `cargo package --list` for `devl-core`, `devl-candle`, and `devl` |

If CI fails on any platform, push the smallest fix (signed) to the same pull request and wait for all six checks to pass on the new commit.

## Mark it ready for review

Readiness and merge are gated by live GitHub status checks enforced on `HEAD` by the `Protect main` ruleset rather than a manual checkbox in the pull request description. Mark the pull request ready for review (`gh pr ready`) when the diff has been reviewed and all of the following hold for its current `HEAD` commit:

1. All six required live GitHub CI checks pass on the current `HEAD` commit (queued, cancelled, skipped, or earlier-commit runs do not count).
2. Every commit in the pull request has a GitHub-verified cryptographic signature.
3. All pre-push items in `.github/PULL_REQUEST_TEMPLATE.md` are satisfied, and the diff against `main` contains only files needed for this change, with no unrelated edits.
4. All affected documentation (`docs/ARCHITECTURE.md`, rustdoc, and `README.md` if applicable) is updated in the same pull request.

For delegated agent workflows, the implementation subthread opens the draft pull request after local verification and hands back immediately; the parent orchestrator or maintainer monitors live CI checks, marks the pull request ready, and squash-merges (see [AGENTS.md](AGENTS.md)).

Pull requests are squash-merged into `main` to preserve linear history, and merged branches are deleted automatically.

## Update the docs

Update documentation in the same pull request as the behavior it changes:

- Update [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) whenever trait signatures, session states, scheduler rules, or backend designs change.
- Update [docs/ROADMAP.md](docs/ROADMAP.md) only when a release ships or phase scope intentionally changes (never add per-issue checklists to it).
- Update [AGENTS.md](AGENTS.md) and `CONTRIBUTING.md` when workflow contracts, local setup steps, CI checks, or PR rules change.
- Update rustdoc comments alongside any public API addition or change (`missing_docs` and `RUSTDOCFLAGS="-D warnings"` enforce this in CI).
- Update `README.md` only when one of its statements becomes outdated or a major capability changes (for example, completing Phase 2 Whisper support or adding `devl-server`).
- Do not edit `CHANGELOG.md` or bump workspace crate versions in feature pull requests; maintainers handle version bumps and changelog entries in dedicated `type: release` pull requests.

## Releases

Crates on crates.io (`devl-core`, `devl-candle`, and `devl`) are immutable per version and are never updated automatically on pushes to `main`.

Maintainers cut a release in two steps:

1. Open a `type: release` pull request bumping `version` in `[workspace.package]` and `[workspace.dependencies]` in [Cargo.toml](Cargo.toml), and moving `[Unreleased]` notes into the new version section in [CHANGELOG.md](CHANGELOG.md). Wait for all six CI checks to pass and squash-merge to `main`.
2. Tag the merged `main` commit with `v<version>` (matching `Cargo.toml` exactly) and push the tag:

   ```bash
   git tag -s v0.1.0-alpha.2 -m "Release v0.1.0-alpha.2"
   git push origin v0.1.0-alpha.2
   ```

Pushing a `v*` tag triggers [`.github/workflows/release.yml`](.github/workflows/release.yml), which verifies the tag matches `Cargo.toml`, runs `fmt`/`clippy`/`test`, authenticates with crates.io via OIDC Trusted Publishing (`rust-lang/crates-io-auth-action@v1`), and runs `cargo publish --workspace` in topological dependency order (`devl-core` → `devl-candle` → `devl`).

## License

`devl-rs` is licensed under the [Apache License, Version 2.0](LICENSE). Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repository is licensed under the same terms, as described in Section 5 of the license.
