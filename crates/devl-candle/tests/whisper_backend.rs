//! Synthetic-weight integration tests for `CandleWhisperBackend`.
//!
//! Verifies stateless `Send + Sync` compliance, `verify_draft` KV population
//! and `return_logits` / `suppress_tokens` behavior, `rollback_kv` truncation
//! and bounds checks, `decode_step` parity against full-sequence recomputation,
//! and 100% token-exact equivalence between `DevlSession<CandleWhisperBackend>`
//! and offline greedy decoding across all four DEVL verification paths.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::Config as WhisperModelConfig;
use devl_candle::{
    CandleBackendError, CandleWhisperBackend, CandleWhisperConfig, MelFilterBank,
    WHISPER_MEL_BINS_V1_V2, WHISPER_MEL_BINS_V3, WHISPER_SAMPLE_RATE_HZ, WhisperAudioEncoding,
    WhisperKvCache,
};
use devl_core::{DevlBackend, DevlSession, SessionConfig, SessionState, TokenId};

const EOS_TOKEN: TokenId = 63;
const PROMPT_TOKENS: [TokenId; 3] = [60, 61, 62];
const SUPPRESSED_TOKENS: [TokenId; 2] = [0, 1];

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
    device: &Device,
) {
    let scale = 0.25;
    for proj in ["q_proj", "v_proj", "out_proj"] {
        tensors.insert(
            format!("{prefix}.{proj}.weight"),
            rng.tensor((d_model, d_model), scale, device),
        );
        tensors.insert(
            format!("{prefix}.{proj}.bias"),
            rng.tensor(d_model, 0.05, device),
        );
    }
    tensors.insert(
        format!("{prefix}.k_proj.weight"),
        rng.tensor((d_model, d_model), scale, device),
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

/// Build a deterministic synthetic Whisper weight map matching Hugging Face
/// (`model.encoder.*` and `model.decoder.*`) parameter names.
///
/// The positional embedding at `prompt_len + eos_step` is biased toward the EOS
/// token embedding so greedy decoding terminates cleanly with `EOS_TOKEN` within
/// a short sequence while still exercising every encoder and decoder layer.
fn build_synthetic_whisper_tensors(
    cfg: &WhisperModelConfig,
    seed: u64,
    eos_step: usize,
    device: &Device,
) -> HashMap<String, Tensor> {
    let mut rng = DeterministicRng::new(seed);
    let mut tensors = HashMap::new();
    let d = cfg.d_model;

    // Encoder
    tensors.insert(
        "model.encoder.conv1.weight".into(),
        rng.tensor((d, cfg.num_mel_bins, 3), 0.15, device),
    );
    tensors.insert(
        "model.encoder.conv1.bias".into(),
        rng.tensor(d, 0.05, device),
    );
    tensors.insert(
        "model.encoder.conv2.weight".into(),
        rng.tensor((d, d, 3), 0.15, device),
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
    // Give EOS a distinct unit-scale direction (matching other tokens' norm) so
    // it does not dominate early steps, while biasing earlier positional
    // embeddings away from EOS and the target stop position toward EOS.
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
            *slot = ev * 8.0;
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

fn build_backend(return_logits: bool, seed: u64, eos_step: usize) -> CandleWhisperBackend {
    let model_cfg = tiny_whisper_config();
    let device = Device::Cpu;
    let tensors = build_synthetic_whisper_tensors(&model_cfg, seed, eos_step, &device);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let config = CandleWhisperConfig {
        device,
        model_config: Some(model_cfg),
        mel_filters: None,
        eos_token: EOS_TOKEN,
        sample_rate_hz: WHISPER_SAMPLE_RATE_HZ as u32,
        prompt_tokens: PROMPT_TOKENS.to_vec(),
        return_logits,
    };
    CandleWhisperBackend::from_var_builder(vb, config).expect("backend load should succeed")
}

fn synthetic_pcm(len: usize, freq_hz: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / WHISPER_SAMPLE_RATE_HZ as f32;
            0.3 * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn pick_different_token(avoid: TokenId) -> TokenId {
    for candidate in 2..EOS_TOKEN {
        if candidate != avoid {
            return candidate;
        }
    }
    unreachable!()
}

#[test]
fn backend_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CandleWhisperBackend>();
    assert_send_sync::<WhisperAudioEncoding>();
    assert_send_sync::<WhisperKvCache>();
}

#[test]
fn verify_draft_kv_population_return_logits_and_suppress_tokens() {
    let pcm = synthetic_pcm(WHISPER_SAMPLE_RATE_HZ, 440.0);

    let backend_no_logits = build_backend(false, 42, 5);
    let enc = backend_no_logits.encode_audio(&pcm, None).unwrap();
    let mut kv = backend_no_logits.new_kv_cache().unwrap();
    let draft = vec![10, 11, 12, 13];
    let verify = backend_no_logits
        .verify_draft(&enc, &mut kv, &draft)
        .unwrap();

    assert_eq!(verify.argmax_tokens.len(), draft.len());
    assert!(verify.logits.is_none());
    assert_eq!(kv.prompt_len, PROMPT_TOKENS.len());
    // Populates max(k - 1, 0) generated KV positions beyond prompt_len.
    assert_eq!(kv.seq_len(), PROMPT_TOKENS.len() + draft.len() - 1);

    // Empty draft populates 0 generated KV positions and returns 1 argmax token.
    let verify_empty = backend_no_logits.verify_draft(&enc, &mut kv, &[]).unwrap();
    assert_eq!(verify_empty.argmax_tokens.len(), 1);
    assert_eq!(kv.seq_len(), PROMPT_TOKENS.len());

    // With return_logits = true, logits are materialized and suppressed tokens are -inf.
    let backend_logits = build_backend(true, 42, 5);
    let enc_l = backend_logits.encode_audio(&pcm, None).unwrap();
    let mut kv_l = backend_logits.new_kv_cache().unwrap();
    let verify_l = backend_logits
        .verify_draft(&enc_l, &mut kv_l, &draft)
        .unwrap();
    let logits = verify_l.logits.expect("logits should be present");
    assert_eq!(logits.len(), draft.len());
    for (row, &argmax_tok) in logits.iter().zip(&verify_l.argmax_tokens) {
        assert_eq!(row.len(), 64);
        for &suppressed in &SUPPRESSED_TOKENS {
            assert!(
                row[suppressed as usize].is_infinite()
                    && row[suppressed as usize].is_sign_negative()
            );
            assert_ne!(argmax_tok, suppressed);
        }
    }
}

#[test]
fn rollback_kv_truncates_self_attn_preserves_cross_attn_and_checks_bounds() {
    let backend = build_backend(false, 99, 6);
    let pcm = synthetic_pcm(WHISPER_SAMPLE_RATE_HZ, 500.0);
    let enc = backend.encode_audio(&pcm, None).unwrap();
    let mut kv = backend.new_kv_cache().unwrap();

    let draft = vec![5, 6, 7, 8, 9];
    backend.verify_draft(&enc, &mut kv, &draft).unwrap();
    assert_eq!(kv.seq_len(), PROMPT_TOKENS.len() + 4);

    // Snapshot cross-attention tensors before rollback.
    let cross_before: Vec<(Vec<usize>, Vec<f32>)> = kv
        .cross_attn_kv
        .iter()
        .map(|slot| {
            let (k, _) = slot.as_ref().expect("cross-attn KV must be populated");
            (
                k.dims().to_vec(),
                k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            )
        })
        .collect();

    // Roll back to accepted_len = 2 (prompt_len + 2).
    backend.rollback_kv(&mut kv, 2).unwrap();
    assert_eq!(kv.seq_len(), PROMPT_TOKENS.len() + 2);
    for slot in &kv.self_attn_kv {
        let (k, v) = slot.as_ref().unwrap();
        assert_eq!(k.dim(2).unwrap(), PROMPT_TOKENS.len() + 2);
        assert_eq!(v.dim(2).unwrap(), PROMPT_TOKENS.len() + 2);
    }

    // Cross-attention KV stays completely intact.
    for (slot, (dims_before, vals_before)) in kv.cross_attn_kv.iter().zip(&cross_before) {
        let (k, _) = slot.as_ref().unwrap();
        assert_eq!(k.dims(), dims_before.as_slice());
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            *vals_before
        );
    }

    // Out-of-bounds rollback returns structured InvalidRollback.
    let err = backend.rollback_kv(&mut kv, 3).unwrap_err();
    assert!(matches!(
        err,
        CandleBackendError::InvalidRollback {
            requested: 3,
            prompt_len: 3,
            seq_len: 5,
        }
    ));
}

#[test]
fn decode_step_matches_full_sequence_recomputation() {
    let backend = build_backend(true, 1234, 6);
    let pcm = synthetic_pcm(WHISPER_SAMPLE_RATE_HZ, 350.0);
    let enc = backend.encode_audio(&pcm, None).unwrap();

    // Start from empty draft verify, then execute 5 autoregressive decode steps.
    let mut kv = backend.new_kv_cache().unwrap();
    let verify = backend.verify_draft(&enc, &mut kv, &[]).unwrap();
    backend.rollback_kv(&mut kv, 0).unwrap();

    let mut context = PROMPT_TOKENS.to_vec();
    let mut seed = verify.argmax_tokens[0];
    let mut kv_tokens = vec![seed];

    for step_idx in 0..5 {
        let step_out = backend.decode_step(&enc, &mut kv, seed).unwrap();
        context.push(seed);
        assert_eq!(kv.seq_len(), context.len());

        // Compare against full-sequence recomputation with no KV cache.
        let hidden = backend
            .model()
            .decoder
            .forward_full_no_cache(&context, &enc.encoder_hidden_states)
            .unwrap();
        let last_hidden = hidden.narrow(1, context.len() - 1, 1).unwrap();
        let full_logits = backend
            .model()
            .decoder
            .project_logits(&last_hidden)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let step_logits = step_out.logits.expect("return_logits is enabled");

        assert_eq!(step_logits.len(), full_logits.len());
        for (tok_id, (&a, &b)) in step_logits.iter().zip(&full_logits).enumerate() {
            if a.is_infinite() && b.is_infinite() {
                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
            } else {
                let diff = (a - b).abs();
                assert!(
                    diff < 1e-4,
                    "logit mismatch at step {step_idx}, token {tok_id}: kv={a}, full={b}, diff={diff}"
                );
            }
        }

        let full_argmax = full_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as TokenId;
        assert_eq!(step_out.token, full_argmax);

        seed = step_out.token;
        kv_tokens.push(seed);
    }

    assert_eq!(kv_tokens.len(), 6);
}

#[test]
fn devl_session_exact_equivalence_across_all_four_paths() {
    let backend = Arc::new(build_backend(false, 2026, 4));
    let pcm = synthetic_pcm(WHISPER_SAMPLE_RATE_HZ * 2, 440.0);
    let max_tokens = 12;

    let offline_tokens = backend.offline_greedy_decode(&pcm, max_tokens).unwrap();
    assert_eq!(
        offline_tokens.last(),
        Some(&EOS_TOKEN),
        "synthetic model should emit EOS within budget"
    );
    assert!(
        offline_tokens.len() >= 4,
        "expected at least 4 tokens to exercise partial rollback paths, got {:?}",
        offline_tokens
    );

    let session_cfg = SessionConfig {
        max_tokens,
        // Disable shortcut promotion so commit always exercises verify_draft + rollback_kv.
        enable_checkpoint_promotion: false,
    };

    // (a1) 100% draft match ending in EOS -> single verify pass, zero decode steps.
    {
        let mut session = DevlSession::new(Arc::clone(&backend), session_cfg).unwrap();
        session.push_audio(&pcm).unwrap();
        session.seed_draft_for_test(offline_tokens.clone(), pcm.len());
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, offline_tokens);
        assert_eq!(res.accepted_prefix_len, offline_tokens.len());
        assert_eq!(res.decode_steps, 0);
        assert_eq!(session.state(), SessionState::Finalized);
    }

    // (a2) 100% draft match without EOS -> resumes greedy decode from matched prefix.
    {
        let prefix = offline_tokens[..2].to_vec();
        let mut session = DevlSession::new(Arc::clone(&backend), session_cfg).unwrap();
        session.push_audio(&pcm).unwrap();
        session.seed_draft_for_test(prefix.clone(), pcm.len());
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, offline_tokens);
        assert_eq!(res.accepted_prefix_len, prefix.len());
        assert_eq!(res.decode_steps, offline_tokens.len() - prefix.len());
    }

    // (b) Partial prefix mismatch with KV rollback.
    {
        let mut partial_draft = offline_tokens[..2].to_vec();
        partial_draft.push(pick_different_token(offline_tokens[2]));
        partial_draft.push(pick_different_token(offline_tokens[3]));
        let mut session = DevlSession::new(Arc::clone(&backend), session_cfg).unwrap();
        session.push_audio(&pcm).unwrap();
        session.seed_draft_for_test(partial_draft, pcm.len());
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, offline_tokens);
        assert_eq!(res.accepted_prefix_len, 2);
        assert_eq!(res.decode_steps, offline_tokens.len() - 3);
    }

    // (c) Zero-match draft.
    {
        let wrong_draft = vec![
            pick_different_token(offline_tokens[0]),
            pick_different_token(offline_tokens[1]),
            pick_different_token(offline_tokens[2]),
        ];
        let mut session = DevlSession::new(Arc::clone(&backend), session_cfg).unwrap();
        session.push_audio(&pcm).unwrap();
        session.seed_draft_for_test(wrong_draft, pcm.len());
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, offline_tokens);
        assert_eq!(res.accepted_prefix_len, 0);
        assert_eq!(res.decode_steps, offline_tokens.len() - 1);
    }

    // (d) Empty draft (direct commit from Idle).
    {
        let mut session = DevlSession::new(Arc::clone(&backend), session_cfg).unwrap();
        session.push_audio(&pcm).unwrap();
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, offline_tokens);
        assert_eq!(res.draft_len, 0);
        assert_eq!(res.accepted_prefix_len, 0);
        assert_eq!(res.decode_steps, offline_tokens.len() - 1);
    }

    // End-to-end incremental streaming chunks with checkpoint draft + commit.
    {
        let mut session = DevlSession::new(Arc::clone(&backend), session_cfg).unwrap();
        let mid = pcm.len() / 2;
        session.push_audio(&pcm[..mid]).unwrap();
        let draft = session.run_draft().unwrap();
        assert!(!draft.tokens.is_empty());
        session.push_audio(&pcm[mid..]).unwrap();
        let res = session.commit().unwrap();
        assert_eq!(res.tokens, offline_tokens);
    }
}

#[test]
fn from_var_builder_validates_config() {
    let device = Device::Cpu;
    let empty_vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &device);

    // Missing model_config.
    let err = CandleWhisperBackend::from_var_builder(
        empty_vb.clone(),
        CandleWhisperConfig {
            prompt_tokens: PROMPT_TOKENS.to_vec(),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, CandleBackendError::InvalidConfig(_)));

    // Empty prompt_tokens.
    let err = CandleWhisperBackend::from_var_builder(
        empty_vb.clone(),
        CandleWhisperConfig {
            model_config: Some(tiny_whisper_config()),
            prompt_tokens: Vec::new(),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, CandleBackendError::InvalidConfig(_)));

    // Mismatched mel filter bank bin count.
    let err = CandleWhisperBackend::from_var_builder(
        empty_vb,
        CandleWhisperConfig {
            model_config: Some(tiny_whisper_config()),
            mel_filters: Some(MelFilterBank::slaney(WHISPER_MEL_BINS_V3).unwrap()),
            prompt_tokens: PROMPT_TOKENS.to_vec(),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        CandleBackendError::MelBinMismatch {
            model_mel_bins: 80,
            filter_mel_bins: 128,
        }
    ));
}
