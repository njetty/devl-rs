//! Deterministic mock backend for testing exact DEVL equivalence, KV-cache
//! rollback invariants, error recovery, and cooperative step-level preemption.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use thiserror::Error;

use crate::backend::{DevlBackend, StepOutput, TokenId, VerifyOutput};

/// Default EOS token ID used by [`MockBackend`].
pub const MOCK_EOS_TOKEN: TokenId = 0;

/// Error returned by [`MockBackend`] when fault injection or invariant checks fail.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MockError {
    /// Injected error for testing failure paths.
    #[error("injected mock failure during `{stage}`")]
    Injected { stage: &'static str },

    /// Attempted to roll back KV cache past its populated generated length.
    #[error(
        "rollback_kv requested accepted_len={requested}, but kv only has {available} generated tokens"
    )]
    InvalidRollback { requested: usize, available: usize },
}

/// Encoded audio handle produced by [`MockBackend`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockAudioEncoding {
    /// Number of PCM samples encoded.
    pub audio_len: usize,
    /// Deterministic reference greedy output (EOS-terminated) for this audio.
    pub reference_tokens: Vec<TokenId>,
    /// Whether this encoding reused a prefix from `prior_encoding`.
    pub reused_prior: bool,
}

/// Stateful decoder KV cache for [`MockBackend`].
///
/// Explicitly stores `prompt_len` and the exact generated token history fed into
/// the decoder. If a caller forgets to roll back rejected draft tokens,
/// subsequent `decode_step` calls detect the mismatched history and emit a
/// deterministic divergent token, ensuring tests catch any KV rollback bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockKvCache {
    /// Fixed prompt length `P` tracked internally by the backend.
    pub prompt_len: usize,
    /// Generated token positions currently cached after `prompt_len`.
    pub generated_tokens: Vec<TokenId>,
}

/// Execution telemetry recorded by [`MockBackend`].
#[derive(Debug, Default)]
pub struct MockMetrics {
    pub encode_calls: AtomicUsize,
    pub verify_calls: AtomicUsize,
    pub rollback_calls: AtomicUsize,
    pub decode_step_calls: AtomicUsize,
    pub prior_encoding_reuses: AtomicUsize,
}

type StepHook = Arc<dyn Fn(usize) + Send + Sync>;

/// Deterministic, stateful mock implementation of [`DevlBackend`].
///
/// Maps accumulated PCM length (via an exact lookup table or deterministic
/// fallback rule) to a reference greedy token sequence. Both `verify_draft` and
/// `decode_step` condition every prediction on the exact prefix currently in
/// [`MockKvCache`], emitting poisoned tokens if the prefix diverged from the
/// offline reference.
#[derive(Clone)]
pub struct MockBackend {
    eos_token: TokenId,
    sample_rate_hz: u32,
    prompt_len: usize,
    /// Maps exact PCM sample count -> EOS-terminated reference token sequence.
    transcripts_by_len: Arc<BTreeMap<usize, Vec<TokenId>>>,
    include_logits: bool,
    fail_next_encode: Arc<AtomicBool>,
    fail_next_verify: Arc<AtomicBool>,
    fail_next_decode_step: Arc<AtomicBool>,
    step_hook: Option<StepHook>,
    metrics: Arc<MockMetrics>,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockBackend {
    /// Create a mock backend with default settings (`eos_token = 0`, `16_000 Hz`).
    pub fn new() -> Self {
        Self {
            eos_token: MOCK_EOS_TOKEN,
            sample_rate_hz: 16_000,
            prompt_len: 4,
            transcripts_by_len: Arc::new(BTreeMap::new()),
            include_logits: false,
            fail_next_encode: Arc::new(AtomicBool::new(false)),
            fail_next_verify: Arc::new(AtomicBool::new(false)),
            fail_next_decode_step: Arc::new(AtomicBool::new(false)),
            step_hook: None,
            metrics: Arc::new(MockMetrics::default()),
        }
    }

    /// Create a mock backend pre-populated with exact `audio_len -> tokens` mappings.
    ///
    /// Each sequence should end with `eos_token` (if not, `eos_token` is appended
    /// automatically when resolving the reference transcript).
    pub fn with_transcripts(entries: impl IntoIterator<Item = (usize, Vec<TokenId>)>) -> Self {
        let map: BTreeMap<usize, Vec<TokenId>> = entries.into_iter().collect();
        Self {
            transcripts_by_len: Arc::new(map),
            ..Self::new()
        }
    }

    /// Enable populating synthetic per-position logits on `verify_draft` and `decode_step`.
    pub fn with_logits(mut self, include_logits: bool) -> Self {
        self.include_logits = include_logits;
        self
    }

    /// Attach a callback invoked on every `decode_step(step_index)` call.
    pub fn with_step_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        self.step_hook = Some(Arc::new(hook));
        self
    }

    /// Inject a one-shot failure on the next `encode_audio` call.
    pub fn fail_next_encode(&self) {
        self.fail_next_encode.store(true, Ordering::SeqCst);
    }

    /// Inject a one-shot failure on the next `verify_draft` call.
    pub fn fail_next_verify(&self) {
        self.fail_next_verify.store(true, Ordering::SeqCst);
    }

    /// Inject a one-shot failure on the next `decode_step` call.
    pub fn fail_next_decode_step(&self) {
        self.fail_next_decode_step.store(true, Ordering::SeqCst);
    }

    /// Inspect recorded backend call metrics.
    pub fn metrics(&self) -> &MockMetrics {
        &self.metrics
    }

    /// Compute the exact reference offline greedy output for `pcm`, capped at `max_tokens`.
    ///
    /// Runs pure autoregressive offline decoding (empty draft prefill followed
    /// by step-by-step `decode_step`) using this backend's exact transition function.
    pub fn reference_offline_decode(&self, pcm: &[f32], max_tokens: usize) -> Vec<TokenId> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let encoding = self
            .encode_audio(pcm, None)
            .expect("reference_offline_decode encode_audio failed");
        let mut kv = self
            .new_kv_cache()
            .expect("reference_offline_decode new_kv_cache failed");
        let first = self
            .verify_draft(&encoding, &mut kv, &[])
            .expect("reference_offline_decode initial prefill failed");
        let mut tokens = vec![first.argmax_tokens[0]];
        while tokens.last() != Some(&self.eos_token) && tokens.len() < max_tokens {
            let seed = *tokens.last().expect("non-empty");
            let step = self
                .decode_step(&encoding, &mut kv, seed)
                .expect("reference_offline_decode decode_step failed");
            tokens.push(step.token);
        }
        tokens
    }

    fn resolve_reference(&self, pcm: &[f32]) -> Vec<TokenId> {
        if let Some(tokens) = self.transcripts_by_len.get(&pcm.len()) {
            let mut seq = tokens.clone();
            if seq.last() != Some(&self.eos_token) {
                seq.push(self.eos_token);
            }
            return seq;
        }

        // Deterministic fallback derived from audio length and quantized content.
        if pcm.is_empty() {
            return vec![self.eos_token];
        }
        let token_count = (pcm.len() / 160).clamp(1, 32);
        let mut seq = Vec::with_capacity(token_count + 1);
        for i in 0..token_count {
            let sample = pcm[i * (pcm.len() / token_count)];
            let bucket = ((sample.abs() * 100.0) as u32) % 97;
            seq.push(10 + (i as u32) * 100 + bucket);
        }
        seq.push(self.eos_token);
        seq
    }

    /// Predict the next greedy token given `reference` and the causal `prefix`
    /// already present in the KV cache.
    fn predict_next(&self, reference: &[TokenId], prefix: &[TokenId]) -> TokenId {
        let pos = prefix.len();
        let prefix_matches = pos <= reference.len() && prefix == &reference[..pos];
        if prefix_matches {
            reference.get(pos).copied().unwrap_or(self.eos_token)
        } else {
            // Deterministic divergent token simulating corrupted causal state.
            let last = prefix.last().copied().unwrap_or(1);
            900_000 + ((last.wrapping_mul(31).wrapping_add(pos as u32)) % 997)
        }
    }

    fn synthetic_logits(&self, winner: TokenId) -> Vec<f32> {
        // Small deterministic diagnostic logit vector with peak at index `winner % 16`.
        let mut v = vec![0.0_f32; 16];
        v[(winner as usize) % 16] = 10.0;
        v
    }
}

impl DevlBackend for MockBackend {
    type Error = MockError;
    type AudioEncoding = MockAudioEncoding;
    type KvCache = MockKvCache;

    fn eos_token(&self) -> TokenId {
        self.eos_token
    }

    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    fn new_kv_cache(&self) -> Result<Self::KvCache, Self::Error> {
        Ok(MockKvCache {
            prompt_len: self.prompt_len,
            generated_tokens: Vec::new(),
        })
    }

    fn encode_audio(
        &self,
        pcm: &[f32],
        prior_encoding: Option<&Self::AudioEncoding>,
    ) -> Result<Self::AudioEncoding, Self::Error> {
        self.metrics.encode_calls.fetch_add(1, Ordering::Relaxed);
        if self.fail_next_encode.swap(false, Ordering::SeqCst) {
            return Err(MockError::Injected {
                stage: "encode_audio",
            });
        }
        let reused_prior = prior_encoding.is_some_and(|p| p.audio_len <= pcm.len());
        if reused_prior {
            self.metrics
                .prior_encoding_reuses
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(MockAudioEncoding {
            audio_len: pcm.len(),
            reference_tokens: self.resolve_reference(pcm),
            reused_prior,
        })
    }

    fn verify_draft(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        draft: &[TokenId],
    ) -> Result<VerifyOutput, Self::Error> {
        self.metrics.verify_calls.fetch_add(1, Ordering::Relaxed);
        if self.fail_next_verify.swap(false, Ordering::SeqCst) {
            return Err(MockError::Injected {
                stage: "verify_draft",
            });
        }

        let k = draft.len();
        kv.prompt_len = self.prompt_len;
        kv.generated_tokens.clear();
        kv.generated_tokens
            .extend_from_slice(&draft[..k.saturating_sub(1)]);

        let out_len = k.max(1);
        let mut argmax_tokens = Vec::with_capacity(out_len);
        let mut logits = self.include_logits.then(|| Vec::with_capacity(out_len));

        for i in 0..out_len {
            let prefix = &draft[..i.min(k)];
            let tok = self.predict_next(&encoding.reference_tokens, prefix);
            argmax_tokens.push(tok);
            if let Some(ref mut l) = logits {
                l.push(self.synthetic_logits(tok));
            }
        }

        Ok(VerifyOutput {
            argmax_tokens,
            logits,
        })
    }

    fn rollback_kv(&self, kv: &mut Self::KvCache, accepted_len: usize) -> Result<(), Self::Error> {
        self.metrics.rollback_calls.fetch_add(1, Ordering::Relaxed);
        if accepted_len > kv.generated_tokens.len() {
            return Err(MockError::InvalidRollback {
                requested: accepted_len,
                available: kv.generated_tokens.len(),
            });
        }
        kv.generated_tokens.truncate(accepted_len);
        Ok(())
    }

    fn decode_step(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        input_token: TokenId,
    ) -> Result<StepOutput, Self::Error> {
        let step_idx = self
            .metrics
            .decode_step_calls
            .fetch_add(1, Ordering::Relaxed);
        if let Some(ref hook) = self.step_hook {
            hook(step_idx);
        }
        if self.fail_next_decode_step.swap(false, Ordering::SeqCst) {
            return Err(MockError::Injected {
                stage: "decode_step",
            });
        }

        kv.generated_tokens.push(input_token);
        let token = self.predict_next(&encoding.reference_tokens, &kv.generated_tokens);
        let logits = self.include_logits.then(|| self.synthetic_logits(token));
        Ok(StepOutput { token, logits })
    }
}
