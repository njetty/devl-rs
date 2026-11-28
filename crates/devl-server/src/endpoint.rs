//! Pluggable server-side VAD and endpointing hooks.
//!
//! An [`EndpointPolicy`] watches the audio of one stream and tells the
//! coordinator when to schedule a checkpoint draft or finalize the turn. The
//! policy decides only *when*; it never touches the session, the scheduler, or
//! the contract. A decision is applied by the coordinator through the same
//! state validation as a client `Commit`, so manual and automatic triggers
//! cannot finalize a turn twice.
//!
//! No policy is installed by default. Without one the client decides when to
//! commit and drafts follow `CoordinatorConfig::draft_cadence_samples`, exactly
//! as before.
//!
//! # Semantics
//!
//! * One policy instance per stream, created by an [`EndpointPolicyFactory`]
//!   when the stream is opened. State is therefore per stream; nothing is
//!   shared across streams unless the factory shares it.
//! * [`EndpointPolicy::on_audio`] runs once per accepted audio frame, in
//!   order, on the thread that sent the frame, while the stream's state lock
//!   is held. It must be fast and must not block. A panic fails the stream
//!   with `internal`.
//! * Resolution is one frame. A policy cannot end a turn in the middle of a
//!   frame, and audio in the frame that triggered the end of turn is part of
//!   the turn.
//! * [`EndpointDecision::Checkpoint`] requests a draft in addition to the
//!   cadence drafts and resets the cadence counter. Drafts stay advisory and
//!   are coalesced by the scheduler, so a checkpoint is a request, not a
//!   guarantee of one partial.
//! * [`EndpointDecision::EndTurn`] moves the stream to the committing phase
//!   and submits the final, as a client `Commit` would. It wins over a
//!   checkpoint requested for the same frame.
//! * One turn per stream. After an automatic end of turn the policy is not
//!   consulted again. Client `Audio` and `Commit` that were in flight when the
//!   server committed are ignored, not errors, so a manual commit that loses
//!   the race produces no second final and no error. The final covers exactly
//!   the audio accepted before the end of turn. A client `Cancel` or `Close`
//!   still cancels the pending final.
//! * Disconnect, cancel, error, and shutdown drop the policy with the stream.
//!   There is no reset: a new turn is a new stream.

use std::sync::Arc;

use devl_core::StreamId;

/// What the coordinator should do after a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointDecision {
    /// Nothing; keep streaming.
    Continue,
    /// Submit a checkpoint draft now.
    Checkpoint,
    /// The turn is over; commit and produce the final.
    EndTurn,
}

/// Per-stream VAD and endpointing hook. See the [module docs](self).
pub trait EndpointPolicy: Send {
    /// Observe one accepted frame of mono 16 kHz samples. `total_samples` is
    /// the stream's accepted sample count including this frame.
    fn on_audio(&mut self, frame: &[f32], total_samples: u64) -> EndpointDecision;
}

/// Creates the policy for a newly opened stream. Called while the
/// coordinator's admission lock is held, so it must be cheap.
pub type EndpointPolicyFactory = Arc<dyn Fn(StreamId) -> Box<dyn EndpointPolicy> + Send + Sync>;

/// Configuration for [`SilenceEndpointPolicy`], in samples at 16 kHz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceEndpointConfig {
    /// A frame whose peak absolute sample is at least this is speech.
    pub speech_threshold: f32,
    /// Silence after speech that requests one checkpoint per pause. `None`
    /// disables policy checkpoints.
    pub checkpoint_silence_samples: Option<u64>,
    /// Silence after speech that ends the turn. At least 1.
    pub end_silence_samples: u64,
    /// End the turn once this many samples were accepted, speech or not.
    /// `None` leaves long turns to the stream limits.
    pub max_turn_samples: Option<u64>,
}

/// Invalid [`SilenceEndpointConfig`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid endpoint configuration: {0}")]
pub struct EndpointConfigError(pub &'static str);

impl SilenceEndpointConfig {
    /// Check internal consistency.
    pub fn validate(&self) -> Result<(), EndpointConfigError> {
        if !(self.speech_threshold.is_finite() && self.speech_threshold > 0.0) {
            return Err(EndpointConfigError(
                "speech_threshold must be finite and positive",
            ));
        }
        if self.end_silence_samples == 0 {
            return Err(EndpointConfigError(
                "end_silence_samples must be at least 1",
            ));
        }
        if self.checkpoint_silence_samples == Some(0) {
            return Err(EndpointConfigError(
                "checkpoint_silence_samples must be at least 1",
            ));
        }
        if self.max_turn_samples == Some(0) {
            return Err(EndpointConfigError("max_turn_samples must be at least 1"));
        }
        Ok(())
    }
}

/// Deterministic amplitude-and-silence policy.
///
/// Intended for tests and as a reference implementation, not as a production
/// VAD: it classifies whole frames by peak amplitude and counts samples, never
/// wall-clock time, so the same audio always yields the same decisions.
///
/// * Silence before the first speech is ignored, so silence-only audio never
///   ends the turn (unless `max_turn_samples` is set).
/// * After speech, accumulated silence requests one checkpoint when it reaches
///   `checkpoint_silence_samples` and ends the turn at `end_silence_samples`.
///   Speech resets the silence count and re-arms the checkpoint.
/// * `max_turn_samples` ends the turn from stream start.
#[derive(Debug, Clone)]
pub struct SilenceEndpointPolicy {
    config: SilenceEndpointConfig,
    heard_speech: bool,
    silence: u64,
    checkpointed: bool,
}

impl SilenceEndpointPolicy {
    /// Create a policy with a validated configuration.
    pub fn new(config: SilenceEndpointConfig) -> Result<Self, EndpointConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            heard_speech: false,
            silence: 0,
            checkpointed: false,
        })
    }

    /// A factory producing independent policies with this configuration.
    pub fn factory(
        config: SilenceEndpointConfig,
    ) -> Result<EndpointPolicyFactory, EndpointConfigError> {
        config.validate()?;
        Ok(Arc::new(move |_| {
            Box::new(Self {
                config,
                heard_speech: false,
                silence: 0,
                checkpointed: false,
            })
        }))
    }
}

impl EndpointPolicy for SilenceEndpointPolicy {
    fn on_audio(&mut self, frame: &[f32], total_samples: u64) -> EndpointDecision {
        if self
            .config
            .max_turn_samples
            .is_some_and(|m| total_samples >= m)
        {
            return EndpointDecision::EndTurn;
        }
        let peak = frame.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
        if peak >= self.config.speech_threshold {
            self.heard_speech = true;
            self.silence = 0;
            self.checkpointed = false;
            return EndpointDecision::Continue;
        }
        if !self.heard_speech {
            return EndpointDecision::Continue;
        }
        self.silence += frame.len() as u64;
        if self.silence >= self.config.end_silence_samples {
            return EndpointDecision::EndTurn;
        }
        match self.config.checkpoint_silence_samples {
            Some(c) if self.silence >= c && !self.checkpointed => {
                self.checkpointed = true;
                EndpointDecision::Checkpoint
            }
            _ => EndpointDecision::Continue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SilenceEndpointConfig {
        SilenceEndpointConfig {
            speech_threshold: 0.1,
            checkpoint_silence_samples: Some(100),
            end_silence_samples: 300,
            max_turn_samples: None,
        }
    }

    #[test]
    fn rejects_invalid_configs() {
        for bad in [
            SilenceEndpointConfig {
                speech_threshold: 0.0,
                ..cfg()
            },
            SilenceEndpointConfig {
                speech_threshold: f32::NAN,
                ..cfg()
            },
            SilenceEndpointConfig {
                end_silence_samples: 0,
                ..cfg()
            },
            SilenceEndpointConfig {
                checkpoint_silence_samples: Some(0),
                ..cfg()
            },
            SilenceEndpointConfig {
                max_turn_samples: Some(0),
                ..cfg()
            },
        ] {
            assert!(SilenceEndpointPolicy::new(bad).is_err());
            assert!(SilenceEndpointPolicy::factory(bad).is_err());
        }
        assert!(SilenceEndpointPolicy::new(cfg()).is_ok());
    }

    #[test]
    fn pause_checkpoints_once_and_speech_rearms() {
        use EndpointDecision::*;
        let mut p = SilenceEndpointPolicy::new(cfg()).unwrap();
        let speech = [0.5_f32; 100];
        let quiet = [0.0_f32; 100];
        assert_eq!(
            p.on_audio(&quiet, 100),
            Continue,
            "leading silence is ignored"
        );
        assert_eq!(p.on_audio(&speech, 200), Continue);
        assert_eq!(p.on_audio(&quiet, 300), Checkpoint);
        assert_eq!(
            p.on_audio(&quiet, 400),
            Continue,
            "one checkpoint per pause"
        );
        assert_eq!(p.on_audio(&speech, 500), Continue);
        assert_eq!(p.on_audio(&quiet, 600), Checkpoint, "speech re-arms");
        assert_eq!(p.on_audio(&quiet, 700), Continue);
        assert_eq!(p.on_audio(&quiet, 800), EndTurn);
    }

    #[test]
    fn max_turn_ends_even_without_pause() {
        let mut p = SilenceEndpointPolicy::new(SilenceEndpointConfig {
            max_turn_samples: Some(200),
            ..cfg()
        })
        .unwrap();
        let speech = [0.5_f32; 100];
        assert_eq!(p.on_audio(&speech, 100), EndpointDecision::Continue);
        assert_eq!(p.on_audio(&speech, 200), EndpointDecision::EndTurn);
    }
}
