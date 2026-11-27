//! DEVL session state machine, incremental audio buffering, draft verification,
//! KV-cache rollback, and resumed greedy decoding.

use thiserror::Error;

use crate::backend::{DevlBackend, TokenId};

/// Lifecycle state of a [`DevlSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Initial or post-reset state with no active draft yet.
    Idle,
    /// At least one checkpoint draft has been produced while audio is streaming.
    Drafting,
    /// Transient state while executing final verification and residual decoding.
    Committing,
    /// Utterance has been committed and its final transcript is immutable.
    Finalized,
    /// Terminal error state entered if [`DevlSession::commit`] fails; recover via [`DevlSession::reset`].
    Failed,
}

/// Errors produced by [`DevlSession`] operations.
#[derive(Debug, Error)]
pub enum DevlError<E> {
    /// Invalid session configuration.
    #[error("invalid session configuration: {0}")]
    InvalidConfig(&'static str),

    /// Operation is not permitted in the current [`SessionState`].
    #[error("cannot perform `{action}` while session is in state `{state:?}`")]
    InvalidState {
        /// Name of the rejected operation.
        action: &'static str,
        /// Current session state when the operation was attempted.
        state: SessionState,
    },

    /// Backend violated the [`DevlBackend::verify_draft`] output length contract.
    #[error("verify_draft returned {actual} argmax tokens, expected {expected}")]
    InvalidVerifyOutput {
        /// Expected number of argmax predictions (`draft.len().max(1)`).
        expected: usize,
        /// Actual number of argmax predictions returned by the backend.
        actual: usize,
    },

    /// Underlying inference backend error.
    #[error("backend error: {0}")]
    Backend(#[from] E),
}

/// Configuration for [`DevlSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionConfig {
    /// Maximum number of tokens (including EOS) generated per draft or commit.
    /// Must be `>= 1`.
    pub max_tokens: usize,

    /// When `true`, if the latest checkpoint ran to EOS on the exact complete
    /// audio buffer (no new samples arrived since), `commit()` promotes that
    /// draft directly without re-running `verify_draft` (Algorithm 1, line 19).
    pub enable_checkpoint_promotion: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 448,
            enable_checkpoint_promotion: true,
        }
    }
}

impl SessionConfig {
    /// Validate configuration invariants (`max_tokens >= 1`).
    pub fn validate<E>(&self) -> Result<(), DevlError<E>> {
        if self.max_tokens == 0 {
            return Err(DevlError::InvalidConfig("max_tokens must be at least 1"));
        }
        Ok(())
    }
}

/// Result of a checkpoint draft pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftResult {
    /// Speculative draft tokens produced by this checkpoint.
    pub tokens: Vec<TokenId>,
    /// Number of leading tokens accepted from the previous draft.
    pub accepted_prefix_len: usize,
    /// Length of the previous draft (after capping to `max_tokens`).
    pub prior_draft_len: usize,
    /// Number of autoregressive `decode_step` calls executed beyond verification.
    pub decode_steps: usize,
    /// Number of accumulated PCM samples seen by this checkpoint.
    pub audio_samples: usize,
    /// Whether decoding terminated with `eos_token()` (as opposed to hitting
    /// `max_tokens` or being preempted).
    pub reached_eos: bool,
}

/// Outcome of a cooperatively cancellable checkpoint draft pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftOutcome {
    /// Checkpoint finished (reached EOS or `max_tokens`).
    Completed(DraftResult),
    /// Checkpoint yielded early because the preemption predicate returned `true`.
    Preempted(DraftResult),
}

impl DraftOutcome {
    /// Access the underlying [`DraftResult`] regardless of completion or preemption.
    pub fn result(&self) -> &DraftResult {
        match self {
            Self::Completed(r) | Self::Preempted(r) => r,
        }
    }

    /// Returns `true` if the draft was preempted before finishing.
    pub fn is_preempted(&self) -> bool {
        matches!(self, Self::Preempted(_))
    }
}

/// Result of committing an utterance via DEVL verification and residual decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitResult {
    /// Final greedy token sequence matching offline decoding.
    pub tokens: Vec<TokenId>,
    /// Length of the longest matching prefix accepted from the latest draft.
    pub accepted_prefix_len: usize,
    /// Length of the draft verified or promoted at commit time, capped at `max_tokens`.
    pub draft_len: usize,
    /// Number of autoregressive `decode_step` calls executed after rollback.
    pub decode_steps: usize,
    /// Whether the final was promoted directly from a matching EOS-terminated checkpoint.
    pub was_promoted: bool,
}

/// Internal result of the shared verify-rollback-continue loop.
struct DecodeLoopOutput {
    tokens: Vec<TokenId>,
    accepted_prefix_len: usize,
    draft_len: usize,
    decode_steps: usize,
    reached_eos: bool,
    preempted: bool,
}

/// Single-utterance DEVL streaming session.
pub struct DevlSession<B: DevlBackend> {
    backend: B,
    config: SessionConfig,
    state: SessionState,
    audio_buffer: Vec<f32>,
    last_encoding: Option<B::AudioEncoding>,
    latest_draft: Option<Vec<TokenId>>,
    draft_audio_len: usize,
    draft_reached_eos: bool,
    final_result: Option<CommitResult>,
}

impl<B: DevlBackend> DevlSession<B> {
    /// Create a new session in [`SessionState::Idle`].
    pub fn new(backend: B, config: SessionConfig) -> Result<Self, DevlError<B::Error>> {
        config.validate()?;
        Ok(Self {
            backend,
            config,
            state: SessionState::Idle,
            audio_buffer: Vec::new(),
            last_encoding: None,
            latest_draft: None,
            draft_audio_len: 0,
            draft_reached_eos: false,
            final_result: None,
        })
    }

    /// Current lifecycle state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Active session configuration.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Reference to the underlying backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Accumulated mono `f32` PCM samples for the current utterance.
    pub fn audio_buffer(&self) -> &[f32] {
        &self.audio_buffer
    }

    /// Latest speculative draft tokens, if any checkpoint has run.
    pub fn latest_draft(&self) -> Option<&[TokenId]> {
        self.latest_draft.as_deref()
    }

    /// Final commit result, available once in [`SessionState::Finalized`].
    pub fn final_result(&self) -> Option<&CommitResult> {
        self.final_result.as_ref()
    }

    /// Append a chunk of mono `f32` PCM samples to the utterance buffer.
    ///
    /// Valid only in [`SessionState::Idle`] and [`SessionState::Drafting`].
    pub fn push_audio(&mut self, pcm: &[f32]) -> Result<(), DevlError<B::Error>> {
        match self.state {
            SessionState::Idle | SessionState::Drafting => {
                self.audio_buffer.extend_from_slice(pcm);
                Ok(())
            }
            state => Err(DevlError::InvalidState {
                action: "push_audio",
                state,
            }),
        }
    }

    /// Run a speculative checkpoint draft over the audio buffered so far.
    ///
    /// This is equivalent to [`Self::run_draft_cancellable`] with a preemption
    /// predicate that always returns `false`.
    pub fn run_draft(&mut self) -> Result<DraftResult, DevlError<B::Error>> {
        match self.run_draft_cancellable(|| false)? {
            DraftOutcome::Completed(result) | DraftOutcome::Preempted(result) => Ok(result),
        }
    }

    /// Run a speculative checkpoint draft with cooperative step-level preemption.
    ///
    /// Before every autoregressive [`DevlBackend::decode_step`], `should_preempt`
    /// is invoked. If it returns `true`, decoding yields immediately and retains
    /// the verified prefix so far (with `reached_eos = false`, ensuring it cannot
    /// be directly promoted at commit).
    ///
    /// Backend errors during a speculative draft are non-fatal: the audio buffer,
    /// previous draft, and session state (`Idle` or `Drafting`) are left intact.
    pub fn run_draft_cancellable<F>(
        &mut self,
        should_preempt: F,
    ) -> Result<DraftOutcome, DevlError<B::Error>>
    where
        F: FnMut() -> bool,
    {
        match self.state {
            SessionState::Idle | SessionState::Drafting => {}
            state => {
                return Err(DevlError::InvalidState {
                    action: "run_draft",
                    state,
                });
            }
        }

        let encoding = self
            .backend
            .encode_audio(&self.audio_buffer, self.last_encoding.as_ref())?;
        let prior_draft = self.latest_draft.as_deref().unwrap_or(&[]);
        let loop_out = self.verify_and_continue(&encoding, prior_draft, should_preempt)?;

        let audio_samples = self.audio_buffer.len();
        self.last_encoding = Some(encoding);
        self.latest_draft = Some(loop_out.tokens.clone());
        self.draft_audio_len = audio_samples;
        self.draft_reached_eos = loop_out.reached_eos && !loop_out.preempted;
        self.state = SessionState::Drafting;

        let result = DraftResult {
            tokens: loop_out.tokens,
            accepted_prefix_len: loop_out.accepted_prefix_len,
            prior_draft_len: loop_out.draft_len,
            decode_steps: loop_out.decode_steps,
            audio_samples,
            reached_eos: self.draft_reached_eos,
        };

        if loop_out.preempted {
            Ok(DraftOutcome::Preempted(result))
        } else {
            Ok(DraftOutcome::Completed(result))
        }
    }

    /// Commit the accumulated utterance and produce the exact greedy final transcript.
    ///
    /// Transitions `Idle | Drafting -> Committing -> Finalized` on success, or
    /// `Committing -> Failed` on any backend or protocol error.
    pub fn commit(&mut self) -> Result<&CommitResult, DevlError<B::Error>> {
        match self.state {
            SessionState::Idle | SessionState::Drafting => {}
            state => {
                return Err(DevlError::InvalidState {
                    action: "commit",
                    state,
                });
            }
        }

        self.state = SessionState::Committing;

        // Direct checkpoint promotion (Algorithm 1, line 19): safe because
        // `devl-core` itself produced the EOS-terminated greedy draft on the
        // exact same complete audio buffer.
        if self.config.enable_checkpoint_promotion
            && self.draft_reached_eos
            && self.draft_audio_len == self.audio_buffer.len()
            && let Some(draft) = self.latest_draft.as_ref()
        {
            let promoted_len = draft.len().min(self.config.max_tokens);
            let tokens = draft[..promoted_len].to_vec();
            self.final_result = Some(CommitResult {
                accepted_prefix_len: tokens.len(),
                draft_len: tokens.len(),
                decode_steps: 0,
                was_promoted: true,
                tokens,
            });
            self.state = SessionState::Finalized;
            return Ok(self.final_result.as_ref().expect("just set"));
        }

        let commit_outcome = (|| -> Result<CommitResult, DevlError<B::Error>> {
            let encoding = self
                .backend
                .encode_audio(&self.audio_buffer, self.last_encoding.as_ref())?;
            let draft = self.latest_draft.as_deref().unwrap_or(&[]);
            let loop_out = self.verify_and_continue(&encoding, draft, || false)?;
            self.last_encoding = Some(encoding);
            Ok(CommitResult {
                tokens: loop_out.tokens,
                accepted_prefix_len: loop_out.accepted_prefix_len,
                draft_len: loop_out.draft_len,
                decode_steps: loop_out.decode_steps,
                was_promoted: false,
            })
        })();

        match commit_outcome {
            Ok(result) => {
                self.final_result = Some(result);
                self.state = SessionState::Finalized;
                Ok(self.final_result.as_ref().expect("just set"))
            }
            Err(err) => {
                self.state = SessionState::Failed;
                Err(err)
            }
        }
    }

    /// Reset the session to [`SessionState::Idle`], clearing all buffered audio,
    /// cached encodings, drafts, and final results.
    ///
    /// Valid from any state (`Idle`, `Drafting`, `Committing`, `Finalized`, `Failed`).
    pub fn reset(&mut self) {
        self.state = SessionState::Idle;
        self.audio_buffer.clear();
        self.last_encoding = None;
        self.latest_draft = None;
        self.draft_audio_len = 0;
        self.draft_reached_eos = false;
        self.final_result = None;
    }

    /// Seed an explicit draft token sequence for deterministic testing of edge
    /// cases (e.g., drafts exceeding `max_tokens`, malformed drafts, or
    /// non-EOS drafts).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn seed_draft_for_test(&mut self, draft: Vec<TokenId>, audio_samples: usize) {
        self.draft_reached_eos = draft.last().is_some_and(|&t| self.backend.is_eos_token(t));
        self.draft_audio_len = audio_samples;
        self.latest_draft = Some(draft);
        if self.state == SessionState::Idle {
            self.state = SessionState::Drafting;
        }
    }

    /// Core Algorithm 1 verify -> longest matching prefix -> KV rollback -> greedy resume.
    fn verify_and_continue<F>(
        &self,
        encoding: &B::AudioEncoding,
        draft: &[TokenId],
        mut should_preempt: F,
    ) -> Result<DecodeLoopOutput, DevlError<B::Error>>
    where
        F: FnMut() -> bool,
    {
        let is_eos = |token: TokenId| self.backend.is_eos_token(token);
        let capped_draft = &draft[..draft.len().min(self.config.max_tokens)];
        let k = capped_draft.len();

        let mut kv = self.backend.new_kv_cache()?;
        let verify = self.backend.verify_draft(encoding, &mut kv, capped_draft)?;

        let expected = k.max(1);
        if verify.argmax_tokens.len() != expected {
            return Err(DevlError::InvalidVerifyOutput {
                expected,
                actual: verify.argmax_tokens.len(),
            });
        }

        let mut accepted_prefix_len = 0usize;
        let mut accepted_eos = false;
        for (&draft_tok, &argmax_tok) in capped_draft.iter().zip(&verify.argmax_tokens) {
            if draft_tok != argmax_tok {
                break;
            }
            accepted_prefix_len += 1;
            if is_eos(draft_tok) {
                accepted_eos = true;
                break;
            }
        }

        let mut tokens = if accepted_eos || (k > 0 && accepted_prefix_len == k) {
            capped_draft[..accepted_prefix_len].to_vec()
        } else {
            let mut prefix = Vec::with_capacity(accepted_prefix_len + 1);
            prefix.extend_from_slice(&capped_draft[..accepted_prefix_len]);
            prefix.push(verify.argmax_tokens[accepted_prefix_len]);
            prefix
        };

        // Truncate KV after `tokens[..tokens.len() - 1]` so `tokens.last()` serves
        // uniformly as the decode seed (Algorithm 1, line 7).
        self.backend.rollback_kv(&mut kv, tokens.len() - 1)?;

        let mut decode_steps = 0usize;
        let mut preempted = false;

        while !tokens.last().is_some_and(|&t| is_eos(t)) && tokens.len() < self.config.max_tokens {
            if should_preempt() {
                preempted = true;
                break;
            }
            let seed = *tokens.last().expect("verified prefix is non-empty");
            let step = self.backend.decode_step(encoding, &mut kv, seed)?;
            tokens.push(step.token);
            decode_steps += 1;
        }

        let reached_eos = tokens.last().is_some_and(|&t| is_eos(t));

        Ok(DecodeLoopOutput {
            tokens,
            accepted_prefix_len,
            draft_len: k,
            decode_steps,
            reached_eos,
            preempted,
        })
    }
}
