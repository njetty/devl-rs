//! Shared test probe that records what a `DevlSession` asks of a `Qwen3AsrBackend`.
//!
//! The session owns its encodings and drafts, so reuse, invalidation, and mismatch are not
//! visible from its public API. [`Probe`] forwards every call to the wrapped backend, including
//! the stop predicate, and keeps a log of the calls.

#![allow(dead_code)]

use std::cell::RefCell;

use devl_candle::qwen3_asr::{
    Qwen3AsrAudioEncoding, Qwen3AsrBackend, Qwen3AsrDecodeState, Qwen3AsrEncodeStats, Qwen3AsrError,
};
use devl_core::{DevlBackend, StepOutput, TokenId, VerifyOutput};

/// One `encode_audio` call.
#[derive(Debug, Clone)]
pub struct EncodeRecord {
    /// Samples in the buffer passed to the call.
    pub samples: usize,
    /// Final windows of the prior encoding, `None` when no prior was passed.
    pub prior_final_windows: Option<usize>,
    /// Clamp floor of the prior encoding.
    pub prior_floor: Option<f32>,
    /// Final windows of the new encoding.
    pub final_windows: usize,
    /// Clamp floor of the new encoding.
    pub floor: f32,
    /// Audio tokens of the new encoding.
    pub audio_tokens: usize,
    /// Counters of the new encoding.
    pub stats: Qwen3AsrEncodeStats,
}

impl EncodeRecord {
    /// Prior final windows that the new encoding did not take over.
    pub fn invalidated_windows(&self) -> usize {
        self.prior_final_windows
            .map_or(0, |p| p.saturating_sub(self.stats.reused_windows))
    }
}

/// One `verify_draft` call.
#[derive(Debug, Clone)]
pub struct VerifyRecord {
    /// Draft passed in.
    pub draft: Vec<TokenId>,
    /// Greedy argmax at each verified position.
    pub argmax: Vec<TokenId>,
}

impl VerifyRecord {
    /// Longest prefix of the draft equal to the argmax tokens.
    pub fn accepted(&self) -> usize {
        self.draft
            .iter()
            .zip(&self.argmax)
            .take_while(|(d, a)| d == a)
            .count()
    }

    /// Whether some draft token was rejected.
    pub fn mismatched(&self) -> bool {
        self.accepted() < self.draft.len()
    }
}

/// Log of the calls seen by a [`Probe`].
#[derive(Debug, Default, Clone)]
pub struct Log {
    /// `encode_audio` calls in order.
    pub encodes: Vec<EncodeRecord>,
    /// `verify_draft` calls in order.
    pub verifies: Vec<VerifyRecord>,
    /// Arguments of `rollback_kv` in order.
    pub rollbacks: Vec<usize>,
    /// Number of `decode_step` calls.
    pub decode_steps: usize,
}

impl Log {
    /// Total prior final windows that were rebuilt instead of reused.
    pub fn invalidated_windows(&self) -> usize {
        self.encodes
            .iter()
            .map(EncodeRecord::invalidated_windows)
            .sum()
    }

    /// Total final windows taken over from a prior.
    pub fn reused_windows(&self) -> usize {
        self.encodes.iter().map(|e| e.stats.reused_windows).sum()
    }

    /// Encode calls that left a non-final tail window uncached.
    pub fn tail_encodes(&self) -> usize {
        self.encodes
            .iter()
            .filter(|e| e.stats.tail_windows > 0)
            .count()
    }

    /// Encode calls whose prior had a different clamp floor.
    pub fn floor_changes(&self) -> usize {
        self.encodes
            .iter()
            .filter(|e| e.prior_floor.is_some_and(|f| f != e.floor))
            .count()
    }

    /// Verify calls with a nonempty draft that the model rejected in part.
    pub fn mismatches(&self) -> usize {
        self.verifies.iter().filter(|v| v.mismatched()).count()
    }
}

/// Recording wrapper around a [`Qwen3AsrBackend`].
pub struct Probe<'a> {
    inner: &'a Qwen3AsrBackend,
    log: RefCell<Log>,
}

impl<'a> Probe<'a> {
    /// Wraps `inner` with an empty log.
    pub fn new(inner: &'a Qwen3AsrBackend) -> Self {
        Self {
            inner,
            log: RefCell::new(Log::default()),
        }
    }

    /// Copy of the log so far.
    pub fn log(&self) -> Log {
        self.log.borrow().clone()
    }
}

impl DevlBackend for Probe<'_> {
    type Error = Qwen3AsrError;
    type AudioEncoding = Qwen3AsrAudioEncoding;
    type KvCache = Qwen3AsrDecodeState;

    fn eos_token(&self) -> TokenId {
        self.inner.eos_token()
    }

    fn is_eos_token(&self, token: TokenId) -> bool {
        self.inner.is_eos_token(token)
    }

    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz()
    }

    fn new_kv_cache(&self) -> Result<Self::KvCache, Self::Error> {
        self.inner.new_kv_cache()
    }

    fn encode_audio(
        &self,
        pcm: &[f32],
        prior: Option<&Self::AudioEncoding>,
    ) -> Result<Self::AudioEncoding, Self::Error> {
        let enc = self.inner.encode_audio(pcm, prior)?;
        self.log.borrow_mut().encodes.push(EncodeRecord {
            samples: pcm.len(),
            prior_final_windows: prior.map(|p| p.final_windows()),
            prior_floor: prior.map(|p| p.floor()),
            final_windows: enc.final_windows(),
            floor: enc.floor(),
            audio_tokens: enc.audio_tokens(),
            stats: enc.stats(),
        });
        Ok(enc)
    }

    fn verify_draft(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        draft: &[TokenId],
    ) -> Result<VerifyOutput, Self::Error> {
        let out = self.inner.verify_draft(encoding, kv, draft)?;
        self.log.borrow_mut().verifies.push(VerifyRecord {
            draft: draft.to_vec(),
            argmax: out.argmax_tokens.clone(),
        });
        Ok(out)
    }

    fn rollback_kv(&self, kv: &mut Self::KvCache, accepted_len: usize) -> Result<(), Self::Error> {
        self.log.borrow_mut().rollbacks.push(accepted_len);
        self.inner.rollback_kv(kv, accepted_len)
    }

    fn decode_step(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        input_token: TokenId,
    ) -> Result<StepOutput, Self::Error> {
        self.log.borrow_mut().decode_steps += 1;
        self.inner.decode_step(encoding, kv, input_token)
    }
}
