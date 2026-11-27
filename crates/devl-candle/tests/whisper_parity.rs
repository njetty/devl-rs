//! End-to-end streaming vs. offline greedy parity tests for [`CandleWhisperBackend`].
//!
//! Includes:
//! 1. Deterministic, network-free CI parity tests using synthetic multi-frequency
//!    audio streams and synthetic Whisper weights covering incremental chunking,
//!    partial draft divergence with KV rollback, checkpoint promotion vs. trailing
//!    audio non-promotion, multi-turn `reset()` reuse, cooperative preemption via
//!    `run_draft_cancellable`, and concurrent multi-stream execution through
//!    `PriorityScheduler`.
//! 2. An `#[ignore]` real-audio Whisper parity test runnable locally against
//!    Hugging Face `safetensors` weights and a 16 kHz mono WAV recording.
//!
//! ## Running the Real-Audio Parity Test Locally
//!
//! Provide a local Whisper model directory containing `config.json` and
//! `model.safetensors`, plus a 16 kHz mono PCM16 or IEEE Float32 WAV file
//! (at most 30 seconds):
//!
//! ```bash
//! DEVL_WHISPER_DIR=/path/to/whisper-tiny \
//! DEVL_WHISPER_AUDIO_WAV=/path/to/sample_16k_mono.wav \
//! cargo test -p devl-candle --test whisper_parity real_audio_whisper_streaming_matches_offline_greedy -- --ignored --nocapture
//! ```
//!
//! Optional environment variables:
//! - `DEVL_WHISPER_MEL_FILTERS`: path to raw little-endian `f32` mel filter bank
//!   bytes (`melfilters.bytes` for 80 bins, `melfilters128.bytes` for 128 bins).
//!   Defaults to [`MelFilterBank::slaney`].
//! - `DEVL_WHISPER_PROMPT_TOKENS`: comma-separated decoder prompt token IDs
//!   (for example `50258,50259,50359,50363` for multilingual Whisper v1/v2 or
//!   `50258,50259,50360,50364` for Whisper v3/v3-turbo). Defaults according to
//!   `config.vocab_size`.
//! - `DEVL_WHISPER_EOS_TOKEN`: decoder EOS (`<|endoftext|>`) token ID. Defaults
//!   to `50256` for English-only models (`vocab_size == 51864`) and `50257`
//!   otherwise.
//! - `DEVL_WHISPER_MAX_TOKENS`: maximum generated token budget (default `128`).
//! - `DEVL_WHISPER_CHUNK_MS`: streaming chunk duration in milliseconds (default `500`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::Config as WhisperModelConfig;
use devl_candle::{
    CandleWhisperBackend, CandleWhisperConfig, MelFilterBank, WHISPER_MAX_SAMPLES,
    WHISPER_MEL_BINS_V1_V2, WHISPER_SAMPLE_RATE_HZ,
};
use devl_core::{
    DevlSession, JobPriority, PriorityScheduler, SchedulerConfig, SessionConfig, SessionState,
    StreamId, TokenId,
};

const EOS_TOKEN: TokenId = 63;
const PROMPT_TOKENS: [TokenId; 3] = [60, 61, 62];
const SUPPRESSED_TOKENS: [TokenId; 2] = [0, 1];
const CHUNK_100MS_SAMPLES: usize = WHISPER_SAMPLE_RATE_HZ / 10; // 1,600 samples at 16 kHz

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / ((1_u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    }

    fn tensor<S: Into<Shape>>(&mut self, shape: S, scale: f32, device: &Device) -> Tensor {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let data: Vec<f32> = (0..elem_count).map(|_| self.next_f32(scale)).collect();
        Tensor::from_vec(data, shape, device).expect("synthetic tensor creation failed")
    }
}

fn tiny_whisper_config() -> WhisperModelConfig {
    WhisperModelConfig {
        num_mel_bins: WHISPER_MEL_BINS_V1_V2,
        max_source_positions: 1500,
        d_model: 16,
        encoder_attention_heads: 2,
        encoder_layers: 2,
        vocab_size: 64,
        max_target_positions: 32,
        decoder_attention_heads: 2,
        decoder_layers: 2,
        suppress_tokens: SUPPRESSED_TOKENS.to_vec(),
    }
}

fn insert_attn_weights(
    tensors: &mut HashMap<String, Tensor>,
    rng: &mut DeterministicRng,
    prefix: &str,
    d_model: usize,
    proj_scale: f32,
    device: &Device,
) {
    for proj in ["q_proj", "v_proj", "out_proj"] {
        tensors.insert(
            format!("{prefix}.{proj}.weight"),
            rng.tensor((d_model, d_model), proj_scale, device),
        );
        tensors.insert(
            format!("{prefix}.{proj}.bias"),
            rng.tensor(d_model, 0.05, device),
        );
    }
    tensors.insert(
        format!("{prefix}.k_proj.weight"),
        rng.tensor((d_model, d_model), proj_scale, device),
    );
}

fn insert_layer_norm(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    d_model: usize,
    device: &Device,
) {
    tensors.insert(
        format!("{prefix}.weight"),
        Tensor::ones(d_model, DType::F32, device).unwrap(),
    );
    tensors.insert(
        format!("{prefix}.bias"),
        Tensor::zeros(d_model, DType::F32, device).unwrap(),
    );
}

/// Build a deterministic synthetic Whisper weight map whose cross-attention
/// projections are strongly coupled to encoder hidden states so distinct audio
/// prefixes naturally yield distinct token trajectories and partial divergences,
/// while still terminating cleanly at `EOS_TOKEN` near `eos_step`.
fn build_synthetic_whisper_tensors(
    cfg: &WhisperModelConfig,
    seed: u64,
    eos_step: usize,
    cross_attn_scale: f32,
    device: &Device,
) -> HashMap<String, Tensor> {
    let mut rng = DeterministicRng::new(seed);
    let mut tensors = HashMap::new();
    let d = cfg.d_model;

    // Encoder
    tensors.insert(
        "model.encoder.conv1.weight".into(),
        rng.tensor((d, cfg.num_mel_bins, 3), 0.18, device),
    );
    tensors.insert(
        "model.encoder.conv1.bias".into(),
        rng.tensor(d, 0.05, device),
    );
    tensors.insert(
        "model.encoder.conv2.weight".into(),
        rng.tensor((d, d, 3), 0.18, device),
    );
    tensors.insert(
        "model.encoder.conv2.bias".into(),
        rng.tensor(d, 0.05, device),
    );
    for i in 0..cfg.encoder_layers {
        let lp = format!("model.encoder.layers.{i}");
        insert_attn_weights(
            &mut tensors,
            &mut rng,
            &format!("{lp}.self_attn"),
            d,
            0.25,
            device,
        );
        insert_layer_norm(
            &mut tensors,
            &format!("{lp}.self_attn_layer_norm"),
            d,
            device,
        );
        tensors.insert(
            format!("{lp}.fc1.weight"),
            rng.tensor((d * 4, d), 0.15, device),
        );
        tensors.insert(format!("{lp}.fc1.bias"), rng.tensor(d * 4, 0.05, device));
        tensors.insert(
            format!("{lp}.fc2.weight"),
            rng.tensor((d, d * 4), 0.15, device),
        );
        tensors.insert(format!("{lp}.fc2.bias"), rng.tensor(d, 0.05, device));
        insert_layer_norm(&mut tensors, &format!("{lp}.final_layer_norm"), d, device);
    }
    insert_layer_norm(&mut tensors, "model.encoder.layer_norm", d, device);

    // Decoder embeddings
    let mut embed_data: Vec<f32> = (0..cfg.vocab_size * d).map(|_| rng.next_f32(0.5)).collect();
    let eos_start = (EOS_TOKEN as usize) * d;
    for (j, slot) in embed_data[eos_start..eos_start + d].iter_mut().enumerate() {
        *slot = if j % 2 == 0 { 0.4 } else { -0.4 };
    }
    let eos_vec = embed_data[eos_start..eos_start + d].to_vec();
    tensors.insert(
        "model.decoder.embed_tokens.weight".into(),
        Tensor::from_vec(embed_data, (cfg.vocab_size, d), device).unwrap(),
    );

    let mut pos_data: Vec<f32> = (0..cfg.max_target_positions * d)
        .map(|_| rng.next_f32(0.3))
        .collect();
    let stop_pos = PROMPT_TOKENS.len() - 1 + eos_step;
    for pos in 0..stop_pos.min(cfg.max_target_positions) {
        let pos_start = pos * d;
        for (slot, &ev) in pos_data[pos_start..pos_start + d].iter_mut().zip(&eos_vec) {
            *slot -= ev * 2.5;
        }
    }
    if stop_pos < cfg.max_target_positions {
        let pos_start = stop_pos * d;
        for (slot, &ev) in pos_data[pos_start..pos_start + d].iter_mut().zip(&eos_vec) {
            *slot = ev * 9.0;
        }
    }
    tensors.insert(
        "model.decoder.embed_positions.weight".into(),
        Tensor::from_vec(pos_data, (cfg.max_target_positions, d), device).unwrap(),
    );

    for i in 0..cfg.decoder_layers {
        let lp = format!("model.decoder.layers.{i}");
        insert_attn_weights(
            &mut tensors,
            &mut rng,
            &format!("{lp}.self_attn"),
            d,
            0.25,
            device,
        );
        insert_layer_norm(
            &mut tensors,
            &format!("{lp}.self_attn_layer_norm"),
            d,
            device,
        );
        insert_attn_weights(
            &mut tensors,
            &mut rng,
            &format!("{lp}.encoder_attn"),
            d,
            cross_attn_scale,
            device,
        );
        insert_layer_norm(
            &mut tensors,
            &format!("{lp}.encoder_attn_layer_norm"),
            d,
            device,
        );
        tensors.insert(
            format!("{lp}.fc1.weight"),
            rng.tensor((d * 4, d), 0.15, device),
        );
        tensors.insert(format!("{lp}.fc1.bias"), rng.tensor(d * 4, 0.05, device));
        tensors.insert(
            format!("{lp}.fc2.weight"),
            rng.tensor((d, d * 4), 0.15, device),
        );
        tensors.insert(format!("{lp}.fc2.bias"), rng.tensor(d, 0.05, device));
        insert_layer_norm(&mut tensors, &format!("{lp}.final_layer_norm"), d, device);
    }
    insert_layer_norm(&mut tensors, "model.decoder.layer_norm", d, device);

    tensors
}

fn build_backend(seed: u64, eos_step: usize, cross_attn_scale: f32) -> CandleWhisperBackend {
    let model_cfg = tiny_whisper_config();
    let device = Device::Cpu;
    let tensors =
        build_synthetic_whisper_tensors(&model_cfg, seed, eos_step, cross_attn_scale, &device);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let config = CandleWhisperConfig {
        device,
        model_config: Some(model_cfg),
        mel_filters: None,
        eos_token: EOS_TOKEN,
        sample_rate_hz: WHISPER_SAMPLE_RATE_HZ as u32,
        prompt_tokens: PROMPT_TOKENS.to_vec(),
        return_logits: false,
    };
    CandleWhisperBackend::from_var_builder(vb, config).expect("backend load should succeed")
}

/// Synthesize a deterministic multi-frequency speech-like waveform with time-varying
/// harmonics and amplitude modulation across segments so incremental prefixes have
/// genuinely evolving spectra.
fn multi_frequency_waveform(
    duration_samples: usize,
    base_hz: f32,
    second_half_hz: f32,
    amp: f32,
) -> Vec<f32> {
    let sr = WHISPER_SAMPLE_RATE_HZ as f32;
    let half = duration_samples / 2;
    (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sr;
            let f0 = if i < half { base_hz } else { second_half_hz };
            let envelope = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 3.5 * t).sin();
            let s = 0.55 * (2.0 * std::f32::consts::PI * f0 * t).sin()
                + 0.30 * (2.0 * std::f32::consts::PI * (f0 * 2.1) * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * (f0 * 3.4) * t).cos();
            (amp * envelope * s).clamp(-0.95, 0.95)
        })
        .collect()
}

#[test]
fn multi_chunk_incremental_streaming_parity_and_partial_rollback() {
    let backend = build_backend(202_609, 5, 0.75);
    let max_tokens = 12;
    let full_pcm = multi_frequency_waveform(CHUNK_100MS_SAMPLES * 12, 220.0, 1_180.0, 0.55);
    let expected = backend
        .offline_greedy_decode(&full_pcm, max_tokens)
        .unwrap();
    assert_eq!(expected.last(), Some(&EOS_TOKEN));

    let mut session = DevlSession::new(
        &backend,
        SessionConfig {
            max_tokens,
            enable_checkpoint_promotion: false,
        },
    )
    .unwrap();

    for (idx, chunk) in full_pcm.chunks(CHUNK_100MS_SAMPLES).enumerate() {
        session.push_audio(chunk).unwrap();
        // Leave the 600 ms checkpoint in place as the final six chunks arrive.
        if matches!(idx + 1, 2 | 4 | 6) {
            let draft = session.run_draft().unwrap();
            assert!(!draft.tokens.is_empty());
            assert_eq!(draft.audio_samples, (idx + 1) * CHUNK_100MS_SAMPLES);
        }
    }

    let draft_len = session.latest_draft().unwrap().len();
    let committed = session.commit().unwrap();
    assert_eq!(committed.tokens, expected);
    assert_eq!(committed.draft_len, draft_len);
    // The final audio changes the third token. The speculative KV cache also
    // contains positions after that token, so commit must roll back and continue.
    assert_eq!(committed.accepted_prefix_len, 2);
    assert!(draft_len > committed.accepted_prefix_len + 2);
    assert!(committed.decode_steps > 0);
    assert!(!committed.was_promoted);
    assert_eq!(session.state(), SessionState::Finalized);
}

#[test]
fn checkpoint_promotion_when_audio_complete_vs_non_promotion_with_trailing_audio() {
    let backend = Arc::new(build_backend(202_602, 5, 0.35));
    let max_tokens = 12;
    let full_pcm = multi_frequency_waveform(CHUNK_100MS_SAMPLES * 10, 310.0, 740.0, 0.45);
    let offline_tokens = backend
        .offline_greedy_decode(&full_pcm, max_tokens)
        .unwrap();
    assert_eq!(offline_tokens.last(), Some(&EOS_TOKEN));

    let promo_cfg = SessionConfig {
        max_tokens,
        enable_checkpoint_promotion: true,
    };

    // Case 1: EOS-terminated checkpoint on complete audio, no trailing samples -> promoted.
    {
        let mut session = DevlSession::new(Arc::clone(&backend), promo_cfg).unwrap();
        for (idx, chunk) in full_pcm.chunks(CHUNK_100MS_SAMPLES).enumerate() {
            session.push_audio(chunk).unwrap();
            if idx == 4 {
                let mid_draft = session.run_draft().unwrap();
                assert!(mid_draft.reached_eos);
            }
        }
        let final_draft = session.run_draft().unwrap();
        assert!(final_draft.reached_eos);
        assert_eq!(final_draft.audio_samples, full_pcm.len());

        let commit_res = session.commit().unwrap();
        assert!(commit_res.was_promoted);
        assert_eq!(commit_res.decode_steps, 0);
        assert_eq!(commit_res.accepted_prefix_len, offline_tokens.len());
        assert_eq!(commit_res.tokens, offline_tokens);
    }

    // Case 2: Trailing audio arrives after an EOS-terminated checkpoint -> must NOT promote.
    {
        let mut session = DevlSession::new(Arc::clone(&backend), promo_cfg).unwrap();
        let split = full_pcm.len() - CHUNK_100MS_SAMPLES;
        for chunk in full_pcm[..split].chunks(CHUNK_100MS_SAMPLES) {
            session.push_audio(chunk).unwrap();
        }
        let pre_tail_draft = session.run_draft().unwrap();
        assert!(pre_tail_draft.reached_eos);

        // Append trailing 100 ms chunk before committing.
        session.push_audio(&full_pcm[split..]).unwrap();
        let commit_res = session.commit().unwrap();
        assert!(!commit_res.was_promoted);
        assert_eq!(commit_res.tokens, offline_tokens);
    }
}

#[test]
fn multi_turn_reset_reuse_has_zero_cross_utterance_leakage() {
    let backend = build_backend(202_609, 5, 0.75);
    let max_tokens = 12;
    let session_cfg = SessionConfig {
        max_tokens,
        enable_checkpoint_promotion: false,
    };

    let full_pcm = multi_frequency_waveform(CHUNK_100MS_SAMPLES * 12, 220.0, 1_180.0, 0.55);
    let utterances = [
        &full_pcm[..CHUNK_100MS_SAMPLES * 6],
        &full_pcm[..],
        &multi_frequency_waveform(CHUNK_100MS_SAMPLES * 8, 920.0, 1_840.0, 0.40),
    ];
    let expected = utterances.map(|pcm| backend.offline_greedy_decode(pcm, max_tokens).unwrap());
    assert_ne!(
        expected[0], expected[1],
        "successive turns must distinguish old audio"
    );

    let mut session = DevlSession::new(&backend, session_cfg).unwrap();

    for (turn_idx, pcm) in utterances.into_iter().enumerate() {
        assert_eq!(session.state(), SessionState::Idle);
        assert!(session.audio_buffer().is_empty());
        assert!(session.latest_draft().is_none());
        assert!(session.final_result().is_none());

        let half_chunks = (pcm.len() / CHUNK_100MS_SAMPLES) / 2;
        for (idx, chunk) in pcm.chunks(CHUNK_100MS_SAMPLES).enumerate() {
            session.push_audio(chunk).unwrap();
            if idx + 1 == half_chunks {
                let draft = session.run_draft().unwrap();
                assert!(!draft.tokens.is_empty());
            }
        }

        let commit_res = session.commit().unwrap();
        assert_eq!(
            commit_res.tokens, expected[turn_idx],
            "turn {turn_idx} diverged from offline greedy decode"
        );
        assert_eq!(session.state(), SessionState::Finalized);

        session.reset();
    }

    // Also verify resetting mid-draft (from SessionState::Drafting) leaks zero state
    // into a subsequent turn.
    session.push_audio(utterances[0]).unwrap();
    session.run_draft().unwrap();
    assert_eq!(session.state(), SessionState::Drafting);
    session.reset();
    assert_eq!(session.state(), SessionState::Idle);
    assert!(session.latest_draft().is_none());

    session.push_audio(utterances[1]).unwrap();
    let commit_after_abort = session.commit().unwrap();
    assert_eq!(commit_after_abort.tokens, expected[1]);
}

#[test]
fn cooperative_draft_preemption_mid_utterance_preserves_exact_commit() {
    let backend = Arc::new(build_backend(202_604, 6, 0.35));
    let max_tokens = 12;
    let full_pcm = multi_frequency_waveform(CHUNK_100MS_SAMPLES * 10, 260.0, 880.0, 0.50);
    let offline_tokens = backend
        .offline_greedy_decode(&full_pcm, max_tokens)
        .unwrap();
    assert_eq!(offline_tokens.last(), Some(&EOS_TOKEN));
    assert!(offline_tokens.len() >= 6);

    // Scenario 1: Preempt mid-utterance, push remaining audio, commit -> exact parity.
    {
        let mut session = DevlSession::new(
            Arc::clone(&backend),
            SessionConfig {
                max_tokens,
                enable_checkpoint_promotion: true,
            },
        )
        .unwrap();

        let mid = CHUNK_100MS_SAMPLES * 5;
        for chunk in full_pcm[..mid].chunks(CHUNK_100MS_SAMPLES) {
            session.push_audio(chunk).unwrap();
        }

        let mut checks = 0usize;
        let outcome = session
            .run_draft_cancellable(|| {
                checks += 1;
                checks > 2
            })
            .unwrap();
        assert!(outcome.is_preempted());
        assert!(!outcome.result().reached_eos);
        assert_eq!(outcome.result().decode_steps, 2);
        assert_eq!(session.state(), SessionState::Drafting);

        for chunk in full_pcm[mid..].chunks(CHUNK_100MS_SAMPLES) {
            session.push_audio(chunk).unwrap();
        }
        let commit_res = session.commit().unwrap();
        assert!(!commit_res.was_promoted);
        assert_eq!(commit_res.tokens, offline_tokens);
    }

    // Scenario 2: Preempt on complete audio with promotion enabled -> must NOT promote
    // because the preempted draft did not reach EOS, yet commit still matches offline.
    {
        let mut session = DevlSession::new(
            Arc::clone(&backend),
            SessionConfig {
                max_tokens,
                enable_checkpoint_promotion: true,
            },
        )
        .unwrap();
        session.push_audio(&full_pcm).unwrap();

        let mut checks = 0usize;
        let outcome = session
            .run_draft_cancellable(|| {
                checks += 1;
                checks > 1
            })
            .unwrap();
        assert!(outcome.is_preempted());
        assert!(!outcome.result().reached_eos);

        let commit_res = session.commit().unwrap();
        assert!(
            !commit_res.was_promoted,
            "preempted draft must never be promoted directly"
        );
        assert_eq!(commit_res.tokens, offline_tokens);
    }
}

#[test]
fn concurrent_shared_backend_scheduler_streams_match_offline_decode() {
    const NUM_STREAMS: usize = 6;
    const MAX_CONCURRENT_DRAFTS: usize = 2;
    const NUM_WORKERS: usize = 4;
    const MAX_TOKENS: usize = 10;

    let backend = Arc::new(build_backend(202_605, 5, 0.40));
    let scheduler = PriorityScheduler::new(SchedulerConfig {
        max_concurrent_drafts: MAX_CONCURRENT_DRAFTS,
    });

    let stream_audio: Vec<Vec<f32>> = (0..NUM_STREAMS)
        .map(|i| {
            let base = 200.0 + (i as f32) * 95.0;
            let second = 520.0 + (i as f32) * 140.0;
            let chunks = 6 + (i % 3) * 2;
            multi_frequency_waveform(CHUNK_100MS_SAMPLES * chunks, base, second, 0.45)
        })
        .collect();

    let expected_offline: Vec<Vec<TokenId>> = stream_audio
        .iter()
        .map(|pcm| backend.offline_greedy_decode(pcm, MAX_TOKENS).unwrap())
        .collect();

    struct StreamState {
        session: DevlSession<Arc<CandleWhisperBackend>>,
        full_pcm: Vec<f32>,
        committed_tokens: Option<Vec<TokenId>>,
    }

    let streams: Vec<Arc<Mutex<StreamState>>> = stream_audio
        .iter()
        .map(|pcm| {
            let mut session = DevlSession::new(
                Arc::clone(&backend),
                SessionConfig {
                    max_tokens: MAX_TOKENS,
                    enable_checkpoint_promotion: false,
                },
            )
            .unwrap();
            // Preload first half of utterance for background checkpoint drafts.
            let mid = pcm.len() / 2;
            session.push_audio(&pcm[..mid]).unwrap();
            Arc::new(Mutex::new(StreamState {
                session,
                full_pcm: pcm.clone(),
                committed_tokens: None,
            }))
        })
        .collect();

    enum JobKind {
        Draft { stream: Arc<Mutex<StreamState>> },
        Commit { stream: Arc<Mutex<StreamState>> },
    }

    // Queue initial drafts for all streams before starting workers.
    for (i, stream) in streams.iter().enumerate() {
        scheduler.submit_draft(
            StreamId(i as u64),
            JobKind::Draft {
                stream: Arc::clone(stream),
            },
        );
    }

    let start_barrier = Arc::new(Barrier::new(NUM_WORKERS + 1));
    let drafts_at_step_boundary = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));
    let commits_submitted = Arc::new(Barrier::new(MAX_CONCURRENT_DRAFTS + 1));
    let peak_active_drafts = Arc::new(AtomicUsize::new(0));
    let in_flight_drafts = Arc::new(AtomicUsize::new(0));
    let commits_executed = Arc::new(AtomicUsize::new(0));

    let mut workers = Vec::with_capacity(NUM_WORKERS);
    for _ in 0..NUM_WORKERS {
        let sched = scheduler.clone();
        let start = Arc::clone(&start_barrier);
        let at_boundary = Arc::clone(&drafts_at_step_boundary);
        let submitted = Arc::clone(&commits_submitted);
        let peak = Arc::clone(&peak_active_drafts);
        let in_flight = Arc::clone(&in_flight_drafts);
        let commits_done = Arc::clone(&commits_executed);

        workers.push(thread::spawn(move || {
            start.wait();
            while let Some(job) = sched.next_job() {
                let stream_id = job.stream_id();
                let seq = job.seq();
                let priority = job.priority();
                let (kind, _permit) = job.into_parts();

                match kind {
                    JobKind::Draft { stream } => {
                        assert_eq!(priority, JobPriority::Draft);
                        let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(current, Ordering::SeqCst);
                        peak.fetch_max(sched.active_drafts(), Ordering::SeqCst);

                        let mut guard = stream.lock().unwrap();
                        let mut checks = 0;
                        let outcome = guard
                            .session
                            .run_draft_cancellable(|| {
                                checks += 1;
                                if checks == 2 {
                                    at_boundary.wait();
                                    submitted.wait();
                                }
                                sched.should_preempt_draft(stream_id, seq)
                            })
                            .unwrap();
                        assert!(outcome.is_preempted());
                        assert_eq!(outcome.result().decode_steps, 1);

                        in_flight.fetch_sub(1, Ordering::SeqCst);
                    }
                    JobKind::Commit { stream } => {
                        assert_eq!(priority, JobPriority::Commit);
                        let mut guard = stream.lock().unwrap();
                        let already_buffered = guard.session.audio_buffer().len();
                        let remaining = guard.full_pcm[already_buffered..].to_vec();
                        if !remaining.is_empty() {
                            guard.session.push_audio(&remaining).unwrap();
                        }
                        let res = guard.session.commit().unwrap();
                        guard.committed_tokens = Some(res.tokens.clone());
                        commits_done.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        }));
    }

    start_barrier.wait();
    // Both draft slots are occupied at a decode-step boundary.
    drafts_at_step_boundary.wait();
    for (i, stream) in streams.iter().enumerate() {
        scheduler.submit_commit(
            StreamId(i as u64),
            JobKind::Commit {
                stream: Arc::clone(stream),
            },
        );
    }

    commits_submitted.wait();
    scheduler.shutdown();
    for worker in workers {
        worker.join().unwrap();
    }

    assert_eq!(commits_executed.load(Ordering::SeqCst), NUM_STREAMS);
    assert!(
        peak_active_drafts.load(Ordering::SeqCst) <= MAX_CONCURRENT_DRAFTS,
        "peak active drafts {} exceeded limit {}",
        peak_active_drafts.load(Ordering::SeqCst),
        MAX_CONCURRENT_DRAFTS
    );
    assert_eq!(scheduler.active_commits(), 0);
    assert_eq!(scheduler.active_drafts(), 0);
    assert_eq!(scheduler.pending_commits(), 0);
    assert_eq!(scheduler.pending_drafts(), 0);

    for (i, (stream, expected)) in streams.iter().zip(&expected_offline).enumerate() {
        let guard = stream.lock().unwrap();
        assert_eq!(guard.session.state(), SessionState::Finalized);
        assert_eq!(
            guard.committed_tokens.as_ref().expect("stream committed"),
            expected,
            "stream {i} final tokens diverged from offline greedy decode"
        );
    }
}

/// Parse a 16 kHz mono PCM16 or IEEE Float32 RIFF/WAVE file into normalized
/// `f32` samples. PCM16 is scaled to `[-1.0, 1.0)`.
fn parse_wav_16k_mono_pcm(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a valid RIFF/WAVE file".into());
    }

    let riff_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let bytes = bytes
        .get(..riff_size.checked_add(8).ok_or("RIFF size overflow")?)
        .filter(|bytes| bytes.len() >= 12)
        .ok_or("invalid RIFF size")?;

    let mut audio_format: Option<u16> = None;
    let mut num_channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut data_slice: Option<&[u8]> = None;

    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .expect("4-byte chunk size"),
        ) as usize;
        offset += 8;

        if chunk_size > bytes.len() - offset {
            return Err(format!(
                "WAV chunk {:?} exceeds file length (offset={offset}, size={chunk_size}, total={})",
                String::from_utf8_lossy(chunk_id),
                bytes.len()
            ));
        }
        let chunk_data = &bytes[offset..offset + chunk_size];

        match chunk_id {
            b"fmt " => {
                if chunk_data.len() < 16 {
                    return Err("WAV fmt chunk shorter than 16 bytes".into());
                }
                let raw_fmt = u16::from_le_bytes([chunk_data[0], chunk_data[1]]);
                let channels = u16::from_le_bytes([chunk_data[2], chunk_data[3]]);
                let rate = u32::from_le_bytes([
                    chunk_data[4],
                    chunk_data[5],
                    chunk_data[6],
                    chunk_data[7],
                ]);
                let bps = u16::from_le_bytes([chunk_data[14], chunk_data[15]]);

                let effective_fmt = if raw_fmt == 0xFFFE {
                    // WAVE_FORMAT_EXTENSIBLE carries the format tag in the first
                    // two bytes of SubFormat (offset 24 inside fmt chunk).
                    if chunk_data.len() < 40 {
                        return Err("WAVE_FORMAT_EXTENSIBLE fmt chunk truncated".into());
                    }
                    u16::from_le_bytes([chunk_data[24], chunk_data[25]])
                } else {
                    raw_fmt
                };

                audio_format = Some(effective_fmt);
                num_channels = Some(channels);
                sample_rate = Some(rate);
                bits_per_sample = Some(bps);
            }
            b"data" => {
                data_slice = Some(chunk_data);
            }
            _ => {}
        }

        // RIFF chunks are word-aligned (padded to even byte boundary).
        offset += chunk_size + (chunk_size & 1);
    }

    if offset != bytes.len() {
        return Err("truncated RIFF chunk or padding".into());
    }

    let fmt = audio_format.ok_or("WAV missing fmt chunk")?;
    let channels = num_channels.ok_or("WAV missing channel count")?;
    let rate = sample_rate.ok_or("WAV missing sample rate")?;
    let bps = bits_per_sample.ok_or("WAV missing bits per sample")?;
    let data = data_slice
        .filter(|data| !data.is_empty())
        .ok_or("WAV missing audio samples")?;

    if channels != 1 {
        return Err(format!("expected mono (1 channel) WAV, got {channels}"));
    }
    if rate != WHISPER_SAMPLE_RATE_HZ as u32 {
        return Err(format!(
            "expected {WHISPER_SAMPLE_RATE_HZ} Hz WAV, got {rate} Hz"
        ));
    }

    match (fmt, bps) {
        (1, 16) => {
            if !data.len().is_multiple_of(2) {
                return Err("PCM16 WAV data byte length is not a multiple of 2".into());
            }
            let (frames, _) = data.as_chunks::<2>();
            Ok(frames
                .iter()
                .map(|&b| f32::from(i16::from_le_bytes(b)) / 32_768.0)
                .collect())
        }
        (3, 32) => {
            if !data.len().is_multiple_of(4) {
                return Err("IEEE float32 WAV data byte length is not a multiple of 4".into());
            }
            let (frames, _) = data.as_chunks::<4>();
            Ok(frames.iter().map(|&b| f32::from_le_bytes(b)).collect())
        }
        _ => Err(format!(
            "unsupported WAV encoding: format={fmt}, bits_per_sample={bps} (expected PCM16 or IEEE F32)"
        )),
    }
}

fn encode_test_wav(
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    extra_chunk: Option<(&[u8; 4], &[u8])>,
    data: &[u8],
) -> Vec<u8> {
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
    let block_align = channels * (bits_per_sample / 8);

    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&16_u32.to_le_bytes());
    body.extend_from_slice(&audio_format.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&sample_rate.to_le_bytes());
    body.extend_from_slice(&byte_rate.to_le_bytes());
    body.extend_from_slice(&block_align.to_le_bytes());
    body.extend_from_slice(&bits_per_sample.to_le_bytes());

    if let Some((tag, payload)) = extra_chunk {
        body.extend_from_slice(tag);
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            body.push(0);
        }
    }

    body.extend_from_slice(b"data");
    body.extend_from_slice(&(data.len() as u32).to_le_bytes());
    body.extend_from_slice(data);

    let mut wav = Vec::with_capacity(8 + body.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(body.len() as u32).to_le_bytes());
    wav.extend_from_slice(&body);
    wav
}

#[test]
fn wav_helper_parses_pcm16_f32_and_rejects_invalid_headers() {
    // PCM16 with an odd-length JUNK chunk to test RIFF word alignment.
    let pcm16_samples: [i16; 4] = [0, 16_384, -16_384, -32_768];
    let pcm16_bytes: Vec<u8> = pcm16_samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let wav16 = encode_test_wav(1, 1, 16_000, 16, Some((b"JUNK", b"odd")), &pcm16_bytes);
    let parsed16 = parse_wav_16k_mono_pcm(&wav16).unwrap();
    assert_eq!(parsed16, vec![0.0, 0.5, -0.5, -1.0]);

    // IEEE Float32.
    let f32_samples: [f32; 3] = [0.25, -0.75, 0.125];
    let f32_bytes: Vec<u8> = f32_samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let wav32 = encode_test_wav(3, 1, 16_000, 32, None, &f32_bytes);
    let parsed32 = parse_wav_16k_mono_pcm(&wav32).unwrap();
    assert_eq!(parsed32, f32_samples);

    // Reject stereo.
    let stereo = encode_test_wav(1, 2, 16_000, 16, None, &pcm16_bytes);
    assert!(parse_wav_16k_mono_pcm(&stereo).is_err());

    // Reject wrong sample rate.
    let bad_rate = encode_test_wav(1, 1, 44_100, 16, None, &pcm16_bytes);
    assert!(parse_wav_16k_mono_pcm(&bad_rate).is_err());

    // Reject non-WAV bytes.
    assert!(parse_wav_16k_mono_pcm(b"not a wav file").is_err());
}

fn default_whisper_prompt_and_eos(model_cfg: &WhisperModelConfig) -> (Vec<TokenId>, TokenId) {
    match model_cfg.vocab_size {
        // English-only Whisper models (e.g., whisper-tiny.en): SOT + <|notimestamps|>, EOT=50256.
        51_864 => (vec![50_257, 50_362], 50_256),
        // Whisper large-v3 / large-v3-turbo: SOT + <|en|> + <|transcribe|> + <|notimestamps|>.
        51_866 => (vec![50_258, 50_259, 50_360, 50_364], 50_257),
        // Multilingual Whisper v1/v2 (vocab_size == 51_865) and other defaults.
        _ => (vec![50_258, 50_259, 50_359, 50_363], 50_257),
    }
}

fn parse_env_token_list(var: &str) -> Option<Vec<TokenId>> {
    let raw = std::env::var(var).ok()?;
    let tokens = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<TokenId>()
                .unwrap_or_else(|e| panic!("invalid token ID '{s}' in {var}: {e}"))
        })
        .collect::<Vec<_>>();
    assert!(!tokens.is_empty(), "{var} must not be empty when set");
    Some(tokens)
}

/// Real-audio Whisper streaming vs. offline greedy parity test.
///
/// Requires `DEVL_WHISPER_DIR` (containing `config.json` and `model.safetensors`)
/// and `DEVL_WHISPER_AUDIO_WAV` (a 16 kHz mono PCM16/F32 WAV file <= 30 s).
/// See the module-level documentation for exact invocation instructions and
/// optional environment variables.
#[test]
#[ignore = "requires local Whisper safetensors weights (DEVL_WHISPER_DIR) and 16 kHz mono WAV (DEVL_WHISPER_AUDIO_WAV)"]
fn real_audio_whisper_streaming_matches_offline_greedy() {
    let model_dir = PathBuf::from(std::env::var("DEVL_WHISPER_DIR").expect(
        "DEVL_WHISPER_DIR must point to a directory with config.json and model.safetensors",
    ));
    let wav_path = PathBuf::from(
        std::env::var("DEVL_WHISPER_AUDIO_WAV")
            .expect("DEVL_WHISPER_AUDIO_WAV must point to a 16 kHz mono PCM16/F32 WAV file"),
    );

    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");

    let config_json = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", config_path.display()));
    let model_config: WhisperModelConfig = serde_json::from_str(&config_json)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", config_path.display()));

    let mel_filters = match std::env::var("DEVL_WHISPER_MEL_FILTERS") {
        Ok(path) => Some(
            MelFilterBank::from_le_bytes_file(model_config.num_mel_bins, &path)
                .unwrap_or_else(|e| panic!("failed to load mel filters from {path}: {e}")),
        ),
        Err(_) => None,
    };

    let (default_prompt, default_eos) = default_whisper_prompt_and_eos(&model_config);
    let prompt_tokens =
        parse_env_token_list("DEVL_WHISPER_PROMPT_TOKENS").unwrap_or(default_prompt);
    let eos_token = std::env::var("DEVL_WHISPER_EOS_TOKEN")
        .ok()
        .map(|v| {
            v.trim()
                .parse::<TokenId>()
                .unwrap_or_else(|e| panic!("invalid DEVL_WHISPER_EOS_TOKEN '{v}': {e}"))
        })
        .unwrap_or(default_eos);
    let max_tokens = std::env::var("DEVL_WHISPER_MAX_TOKENS")
        .ok()
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("invalid DEVL_WHISPER_MAX_TOKENS '{v}': {e}"))
        })
        .unwrap_or(128);
    let chunk_ms = std::env::var("DEVL_WHISPER_CHUNK_MS")
        .ok()
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("invalid DEVL_WHISPER_CHUNK_MS '{v}': {e}"))
        })
        .unwrap_or(500);
    assert!(chunk_ms > 0, "DEVL_WHISPER_CHUNK_MS must be > 0");

    let wav_bytes = std::fs::read(&wav_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", wav_path.display()));
    let pcm = parse_wav_16k_mono_pcm(&wav_bytes)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", wav_path.display()));
    assert!(
        pcm.len() <= WHISPER_MAX_SAMPLES,
        "WAV has {} samples, exceeding Whisper 30 s window ({WHISPER_MAX_SAMPLES})",
        pcm.len()
    );

    let device = Device::Cpu;
    // Load weights via buffered safetensors (zero unsafe code, workspace forbids unsafe).
    let weights_bytes = std::fs::read(&weights_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", weights_path.display()));
    let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, &device)
        .expect("failed to load Whisper safetensors weights");

    let backend = CandleWhisperBackend::from_var_builder(
        vb,
        CandleWhisperConfig {
            device,
            model_config: Some(model_config),
            mel_filters,
            eos_token,
            sample_rate_hz: WHISPER_SAMPLE_RATE_HZ as u32,
            prompt_tokens,
            return_logits: false,
        },
    )
    .expect("failed to construct CandleWhisperBackend");

    let offline_tokens = backend
        .offline_greedy_decode(&pcm, max_tokens)
        .expect("offline_greedy_decode failed");

    let chunk_samples = WHISPER_SAMPLE_RATE_HZ
        .checked_mul(chunk_ms)
        .expect("DEVL_WHISPER_CHUNK_MS is too large")
        / 1000;
    let total_chunks = pcm.chunks(chunk_samples).len();

    let mut session = DevlSession::new(
        &backend,
        SessionConfig {
            max_tokens,
            // Force full verify_draft + rollback_kv + decode_step at commit.
            enable_checkpoint_promotion: false,
        },
    )
    .expect("session creation failed");

    for (idx, chunk) in pcm.chunks(chunk_samples).enumerate() {
        session.push_audio(chunk).expect("push_audio failed");
        let is_last_chunk = idx + 1 == total_chunks;
        // Draft early, then at every other chunk before the final commit.
        if !is_last_chunk && idx.is_multiple_of(2) {
            let draft = session.run_draft().expect("run_draft failed");
            assert!(!draft.tokens.is_empty());
        }
    }

    let commit_result = session.commit().expect("commit failed");
    assert_eq!(
        commit_result.tokens, offline_tokens,
        "streaming DEVL commit tokens diverged from offline greedy decode"
    );
}
