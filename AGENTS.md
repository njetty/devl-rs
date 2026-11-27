# Agent and Contributor Workflow (`AGENTS.md`)

Operational rules for coding agents and contributors working in `njetty/devl-rs`.

## Project Context & Boundaries

- `devl-rs` is an independent, unofficial Apache 2.0 Rust implementation of **DEVL: Decode Early, Verify Late** (`arXiv:2611.20064`). This repository has no affiliation with, and is not endorsed by, the paper author or Nari Labs.
- Maintain the strict one-way Cargo workspace dependency graph:
  - `devl-core` <- `devl-candle` <- `devl` (facade)
  - `devl` depends on `devl-core` (always) and `devl-candle` (optional feature gates: `candle`, `metal`, `cuda`). It re-exports items and holds no protocol or inference logic.
  - `devl-core` must have zero ML runtime, async runtime, or networking dependencies (`std` plus `thiserror` only). Never introduce ML or networking crates into `devl-core`.
  - Protocol logic (verification, prefix matching, KV-rollback selection, draft loop orchestration, commit-over-draft scheduling) lives exclusively in `devl-core`; backend crates must not duplicate it.
- Consult [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before modifying any public trait, core type, exactness invariant, session state transition, or scheduler rule.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, CI requirements, label selection, pull request conventions, and cryptographic commit signing (`required_signatures`).

## Roadmap vs. GitHub Issues Contract

- [docs/ROADMAP.md](docs/ROADMAP.md) is the high-level multi-release plan (`v0.1.0` -> `v0.2.0` -> `v0.3.0`).
  - Do not add task checkboxes (`- [ ]`) or per-issue tracking to `docs/ROADMAP.md`.
  - Update `docs/ROADMAP.md` only when a release ships or phase scope intentionally changes.
- GitHub Issues and Milestones are the single source of truth for active execution in the current release.
- Every actionable issue must be atomically implementable and explicitly define:
  - **Goal**
  - **Scope / Files**
  - **Non-Goals**
  - **Dependencies**
  - **Acceptance Criteria**
- Do not create speculative issues for future milestones (`v0.2.0+`) unless explicitly asked.

## Issue Execution Workflow (Required for Every Task)

1. **Inspect before coding:** Read the assigned issue with `gh issue view <number> --comments` and verify alignment with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the active phase in [docs/ROADMAP.md](docs/ROADMAP.md).
2. **Use a dedicated branch:** Work on a branch named `<type>/<issue-number>-<short-slug>` (for example, `feat/4-whisper-mel-frontend`) unless explicitly instructed to commit directly to `main`.
3. **Sign and scope commits:** Keep every commit cryptographically signed (`required_signatures` is enforced on `main`), scoped strictly to one issue, and linked in commit messages and PR descriptions using `Closes #<number>` (or `Refs #<number>` for partial work).
4. **Enforce acceptance criteria:** Never close an issue unless every acceptance criterion is satisfied and all verification commands pass.
5. **Prevent silent scope creep:** If an out-of-scope bug or follow-up task is discovered during implementation, do not expand the current diff. Propose a follow-up issue and create it via `gh issue create` only after confirmation.

## Verification Gate (Run Before Every Commit / PR)

Run the workspace verification commands locally before committing or opening a pull request:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
```

Also run feature-gated and CI-matching checks whenever public APIs, facade exports, or backend code are touched:

```bash
cargo test -p devl --features test-utils
cargo check -p devl --all-targets --features candle
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features devl/candle
```

On macOS Apple Silicon, also verify Metal feature compilation:

```bash
cargo check -p devl-candle --all-targets --features metal
cargo check -p devl --all-targets --features metal
```

## Documentation & Clean-Room Rules

- **No fabricated metrics:** Never invent, estimate, or extrapolate benchmark numbers, latency tables, TTFS values, or WER metrics. Record only measurements produced locally and tag them with hardware, model identifier, dtype, exact command, and commit SHA.
- **Clean-room writing:** Never copy verbatim prose, tables, or diagrams from `arXiv:2611.20064`. Write all documentation, rustdoc, and diagrams independently from first principles.
- **Tone:** Keep all documentation, commit messages, pull request descriptions, and issue text dry, concise, and technical. Avoid hype and emojis.
