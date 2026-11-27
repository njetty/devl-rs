//! Stop-id predicate regression tests (`DevlBackend::is_eos_token`).
//!
//! Every stop decision in `devl-core` must go through the predicate, wrappers must forward it,
//! and backends that only implement `eos_token` must keep the single-id behavior.

use std::sync::Arc;

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{DevlBackend, DevlSession, SessionConfig, StepOutput, TokenId, VerifyOutput};

/// Alternate stop id that is not the canonical `eos_token()`.
const ALT_STOP: TokenId = 30;

/// Wraps the mock and stops on `ALT_STOP` as well as the canonical EOS.
struct TwoStop(MockBackend);

impl DevlBackend for TwoStop {
    type Error = <MockBackend as DevlBackend>::Error;
    type AudioEncoding = <MockBackend as DevlBackend>::AudioEncoding;
    type KvCache = <MockBackend as DevlBackend>::KvCache;

    fn eos_token(&self) -> TokenId {
        self.0.eos_token()
    }

    fn is_eos_token(&self, token: TokenId) -> bool {
        token == self.0.eos_token() || token == ALT_STOP
    }

    fn new_kv_cache(&self) -> Result<Self::KvCache, Self::Error> {
        self.0.new_kv_cache()
    }

    fn encode_audio(
        &self,
        pcm: &[f32],
        prior: Option<&Self::AudioEncoding>,
    ) -> Result<Self::AudioEncoding, Self::Error> {
        self.0.encode_audio(pcm, prior)
    }

    fn verify_draft(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        draft: &[TokenId],
    ) -> Result<VerifyOutput, Self::Error> {
        self.0.verify_draft(encoding, kv, draft)
    }

    fn rollback_kv(&self, kv: &mut Self::KvCache, n: usize) -> Result<(), Self::Error> {
        self.0.rollback_kv(kv, n)
    }

    fn decode_step(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        token: TokenId,
    ) -> Result<StepOutput, Self::Error> {
        self.0.decode_step(encoding, kv, token)
    }
}

fn pcm(len: usize) -> Vec<f32> {
    vec![0.25; len]
}

fn two_stop() -> TwoStop {
    // The model keeps going after `ALT_STOP`, as a Qwen3-style model could.
    TwoStop(MockBackend::with_transcripts([(
        1600,
        vec![10, 20, ALT_STOP, 40, 50, MOCK_EOS_TOKEN],
    )]))
}

fn config(promotion: bool) -> SessionConfig {
    SessionConfig {
        max_tokens: 64,
        enable_checkpoint_promotion: promotion,
    }
}

#[test]
fn resumed_decode_stops_on_alternate_stop_and_keeps_its_id() {
    for promotion in [false, true] {
        let backend = two_stop();
        let mut s = DevlSession::new(&backend, config(promotion)).unwrap();
        s.push_audio(&pcm(1600)).unwrap();
        let r = s.commit().unwrap();
        assert_eq!(r.tokens, vec![10, 20, ALT_STOP]);
        assert_ne!(*r.tokens.last().unwrap(), backend.eos_token());
        // One empty-draft verification, then two decode steps (20 and ALT_STOP).
        assert_eq!(r.decode_steps, 2);
    }
}

#[test]
fn draft_reports_and_promotes_alternate_stop() {
    let backend = two_stop();
    let mut s = DevlSession::new(&backend, config(true)).unwrap();
    s.push_audio(&pcm(1600)).unwrap();
    let draft = s.run_draft().unwrap().clone();
    assert_eq!(draft.tokens, vec![10, 20, ALT_STOP]);
    assert!(draft.reached_eos);
    let r = s.commit().unwrap();
    assert!(r.was_promoted);
    assert_eq!(r.tokens, vec![10, 20, ALT_STOP]);
    assert_eq!(r.decode_steps, 0);
}

#[test]
fn alternate_stop_inside_a_longer_draft_ends_acceptance() {
    let backend = two_stop();
    let mut s = DevlSession::new(&backend, config(false)).unwrap();
    s.push_audio(&pcm(1600)).unwrap();
    // Every draft token matches the reference, including the tokens after the alternate stop.
    s.seed_draft_for_test(vec![10, 20, ALT_STOP, 40, 50], 1600);
    let r = s.commit().unwrap();
    assert_eq!(r.tokens, vec![10, 20, ALT_STOP]);
    assert_eq!(r.accepted_prefix_len, 3);
    assert_eq!(r.decode_steps, 0);
}

#[test]
fn wrappers_forward_the_predicate() {
    let backend = two_stop();
    assert!(backend.is_eos_token(ALT_STOP));
    assert!(<&TwoStop as DevlBackend>::is_eos_token(&&backend, ALT_STOP));
    assert!(Arc::new(&backend).is_eos_token(ALT_STOP));
    assert!(!Arc::new(&backend).is_eos_token(10));
    let shared = Arc::new(two_stop());
    let mut s = DevlSession::new(Arc::clone(&shared), config(false)).unwrap();
    s.push_audio(&pcm(1600)).unwrap();
    assert_eq!(s.commit().unwrap().tokens, vec![10, 20, ALT_STOP]);
}

#[test]
fn default_predicate_is_the_single_eos_comparison() {
    let backend =
        MockBackend::with_transcripts([(1600, vec![10, 20, ALT_STOP, 40, MOCK_EOS_TOKEN])]);
    assert!(backend.is_eos_token(MOCK_EOS_TOKEN));
    assert!(!backend.is_eos_token(ALT_STOP));
    assert!(Arc::new(&backend).is_eos_token(backend.eos_token()));
    let mut s = DevlSession::new(&backend, config(false)).unwrap();
    s.push_audio(&pcm(1600)).unwrap();
    let r = s.commit().unwrap();
    assert_eq!(r.tokens, backend.reference_offline_decode(&pcm(1600), 64));
    assert_eq!(r.tokens, vec![10, 20, ALT_STOP, 40, MOCK_EOS_TOKEN]);
}
