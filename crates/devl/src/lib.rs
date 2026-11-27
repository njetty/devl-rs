//! Top-level facade crate for **DEVL: Decode Early, Verify Late** (`devl-rs`).
//!
//! `devl` re-exports [`devl_core`] at the crate root (and under [`core`]) and
//! exposes optional inference backends behind Cargo feature flags:
//!
//! - `candle`: enables `devl::candle` (`devl-candle`) for local CPU inference.
//! - `metal`: enables `candle` with Metal acceleration.
//! - `cuda`: enables `candle` with CUDA acceleration.
//! - `test-utils`: enables `devl_core::mock` (`MockBackend`) for deterministic testing.
//!
//! ```rust
//! use devl::{CommitResult, DevlBackend, DevlError, DevlSession, SessionConfig};
//!
//! fn decode<B: DevlBackend>(
//!     backend: B,
//!     first_chunk: &[f32],
//!     last_chunk: &[f32],
//! ) -> Result<CommitResult, DevlError<B::Error>> {
//!     let mut session = DevlSession::new(backend, SessionConfig::default())?;
//!     session.push_audio(first_chunk)?;
//!     let _draft = session.run_draft()?;
//!     session.push_audio(last_chunk)?;
//!     session.commit().cloned()
//! }
//! ```

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/njetty/devl-rs/main/assets/icon-ink.svg",
    html_favicon_url = "https://raw.githubusercontent.com/njetty/devl-rs/main/assets/icon-ink.svg"
)]

pub use devl_core as core;
pub use devl_core::*;

#[cfg(feature = "candle")]
pub use devl_candle as candle;

#[cfg(test)]
mod tests {
    use super::{
        DevlBackend, DevlSession, SessionConfig, SessionState, StepOutput, TokenId, VerifyOutput,
    };

    #[derive(Debug)]
    struct DummyError;

    impl std::fmt::Display for DummyError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("dummy error")
        }
    }

    impl std::error::Error for DummyError {}

    struct DummyBackend;

    impl DevlBackend for DummyBackend {
        type Error = DummyError;
        type AudioEncoding = ();
        type KvCache = ();

        fn eos_token(&self) -> TokenId {
            0
        }

        fn new_kv_cache(&self) -> Result<Self::KvCache, Self::Error> {
            Ok(())
        }

        fn encode_audio(
            &self,
            _pcm: &[f32],
            _prior_encoding: Option<&Self::AudioEncoding>,
        ) -> Result<Self::AudioEncoding, Self::Error> {
            Ok(())
        }

        fn verify_draft(
            &self,
            _encoding: &Self::AudioEncoding,
            _kv: &mut Self::KvCache,
            draft: &[TokenId],
        ) -> Result<VerifyOutput, Self::Error> {
            Ok(VerifyOutput {
                argmax_tokens: vec![0; draft.len().max(1)],
                logits: None,
            })
        }

        fn rollback_kv(
            &self,
            _kv: &mut Self::KvCache,
            _accepted_len: usize,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn decode_step(
            &self,
            _encoding: &Self::AudioEncoding,
            _kv: &mut Self::KvCache,
            _input_token: TokenId,
        ) -> Result<StepOutput, Self::Error> {
            Ok(StepOutput {
                token: 0,
                logits: None,
            })
        }
    }

    #[test]
    fn reexports_core_types_with_default_features() {
        let mut session = DevlSession::new(DummyBackend, SessionConfig::default()).unwrap();
        assert_eq!(session.state(), SessionState::Idle);
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, vec![0]);
        assert_eq!(session.state(), SessionState::Finalized);
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn reexports_mock_backend_with_test_utils_feature() {
        use crate::mock::{MOCK_EOS_TOKEN, MockBackend};

        let backend = MockBackend::with_transcripts([(1600, vec![7, 8, MOCK_EOS_TOKEN])]);
        let mut session = DevlSession::new(&backend, SessionConfig::default()).unwrap();
        session.push_audio(&vec![0.1_f32; 1600]).unwrap();
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, vec![7, 8, MOCK_EOS_TOKEN]);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn exposes_qwen3_asr_backend_behind_the_candle_feature() {
        fn assert_backend<B: DevlBackend + Send + Sync>() {}
        assert_backend::<crate::candle::qwen3_asr::Qwen3AsrBackend>();
        assert_eq!(
            crate::candle::qwen3_asr::Qwen3AsrBackendConfig::default().eos_token,
            151_645
        );
    }
}
