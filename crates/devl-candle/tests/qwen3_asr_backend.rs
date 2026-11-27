//! Deterministic `Qwen3AsrBackend` tests with synthetic weights.
//!
//! These run in CI without any checkpoint. They use a small topology (a 40-frame attention
//! window of 8 tokens, a 32-wide decoder, a 64-token vocabulary) with the real frontend, so the
//! window finality, clamp-floor, and prefix-reuse rules are exercised exactly as for the pinned
//! checkpoint, only with a shorter window. The opt-in real-checkpoint smoke test lives in
//! `qwen3_asr_backend_reference.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use devl_candle::qwen3_asr::{
    Qwen3AsrAudioConfig, Qwen3AsrAudioEncoder, Qwen3AsrAudioEncoding, Qwen3AsrBackend,
    Qwen3AsrBackendConfig, Qwen3AsrDecodeState, Qwen3AsrDecoder, Qwen3AsrError, Qwen3AsrTextConfig,
    argmax_rows,
};
use devl_core::{DevlBackend, DevlSession, SessionConfig, TokenId};

const PAD: u32 = 7;
const PREFIX: [u32; 3] = [1, 2, 3];
const SUFFIX: [u32; 2] = [4, 5];
const WINDOW_SAMPLES: usize = 40 * 160;
/// Samples after which the last frame of window `k` is final: `160 * (40 * k + 39) + 200`.
const fn final_at(k: usize) -> usize {
    160 * (40 * k + 39) + 200
}

fn audio_config() -> Qwen3AsrAudioConfig {
    Qwen3AsrAudioConfig {
        num_mel_bins: 128,
        d_model: 16,
        encoder_layers: 2,
        encoder_attention_heads: 2,
        encoder_ffn_dim: 32,
        downsample_hidden_size: 4,
        output_dim: 32,
        n_window: 5,
        n_window_infer: 40,
    }
}

fn text_config() -> Qwen3AsrTextConfig {
    Qwen3AsrTextConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        max_position_embeddings: 512,
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn tensor(&mut self, shape: &[usize], scale: f32, bias: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| bias + self.next_unit() * scale).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

struct Weights {
    map: HashMap<String, Tensor>,
    rng: Lcg,
}

impl Weights {
    fn rand(&mut self, name: &str, shape: &[usize], scale: f32) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.rng.next_unit() * scale).collect();
        self.map.insert(
            name.to_string(),
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap(),
        );
    }

    fn layer_norm(&mut self, prefix: &str, d: usize) {
        self.map.insert(
            format!("{prefix}.weight"),
            Tensor::ones(d, DType::F32, &Device::Cpu).unwrap(),
        );
        self.map.insert(
            format!("{prefix}.bias"),
            Tensor::zeros(d, DType::F32, &Device::Cpu).unwrap(),
        );
    }
}

fn synthetic_vb(cfg: &Qwen3AsrAudioConfig, dtype: DType) -> VarBuilder<'static> {
    let mut w = Weights {
        map: HashMap::new(),
        rng: Lcg(0x5eed),
    };
    let (d, c) = (cfg.d_model, cfg.downsample_hidden_size);
    let f_out = cfg.num_mel_bins.div_ceil(2).div_ceil(2).div_ceil(2);
    w.rand("conv2d1.weight", &[c, 1, 3, 3], 0.3);
    w.rand("conv2d1.bias", &[c], 0.1);
    w.rand("conv2d2.weight", &[c, c, 3, 3], 0.2);
    w.rand("conv2d2.bias", &[c], 0.1);
    w.rand("conv2d3.weight", &[c, c, 3, 3], 0.2);
    w.rand("conv2d3.bias", &[c], 0.1);
    w.rand("conv_out.weight", &[d, c * f_out], 0.1);
    for i in 0..cfg.encoder_layers {
        let p = format!("layers.{i}");
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            w.rand(&format!("{p}.self_attn.{n}.weight"), &[d, d], 0.25);
            w.rand(&format!("{p}.self_attn.{n}.bias"), &[d], 0.05);
        }
        w.rand(&format!("{p}.fc1.weight"), &[cfg.encoder_ffn_dim, d], 0.25);
        w.rand(&format!("{p}.fc1.bias"), &[cfg.encoder_ffn_dim], 0.05);
        w.rand(&format!("{p}.fc2.weight"), &[d, cfg.encoder_ffn_dim], 0.25);
        w.rand(&format!("{p}.fc2.bias"), &[d], 0.05);
        w.layer_norm(&format!("{p}.self_attn_layer_norm"), d);
        w.layer_norm(&format!("{p}.final_layer_norm"), d);
    }
    w.layer_norm("ln_post", d);
    w.rand("proj1.weight", &[d, d], 0.25);
    w.rand("proj1.bias", &[d], 0.05);
    w.rand("proj2.weight", &[cfg.output_dim, d], 0.25);
    w.rand("proj2.bias", &[cfg.output_dim], 0.05);
    VarBuilder::from_tensors(w.map, dtype, &Device::Cpu)
}

/// Decoder tensors under checkpoint names (`thinker.model.*`, optional `thinker.lm_head.weight`).
fn weights(cfg: &Qwen3AsrTextConfig, seed: u64, with_head: bool) -> HashMap<String, Tensor> {
    let mut rng = Lcg(seed);
    let (h, i, hd) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
    let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
    let mut m = HashMap::new();
    let mut put = |name: String, t: Tensor| {
        m.insert(name, t);
    };
    put(
        "thinker.model.embed_tokens.weight".into(),
        rng.tensor(&[cfg.vocab_size, h], 1.0, 0.0),
    );
    for l in 0..cfg.num_hidden_layers {
        let p = format!("thinker.model.layers.{l}");
        put(
            format!("{p}.input_layernorm.weight"),
            rng.tensor(&[h], 0.3, 1.0),
        );
        put(
            format!("{p}.post_attention_layernorm.weight"),
            rng.tensor(&[h], 0.3, 1.0),
        );
        put(
            format!("{p}.self_attn.q_norm.weight"),
            rng.tensor(&[hd], 0.3, 1.0),
        );
        put(
            format!("{p}.self_attn.k_norm.weight"),
            rng.tensor(&[hd], 0.3, 1.0),
        );
        for (name, shape) in [
            ("q_proj", [nh * hd, h]),
            ("k_proj", [nkv * hd, h]),
            ("v_proj", [nkv * hd, h]),
            ("o_proj", [h, nh * hd]),
        ] {
            put(
                format!("{p}.self_attn.{name}.weight"),
                rng.tensor(&shape, 0.25, 0.0),
            );
        }
        for (name, shape) in [
            ("gate_proj", [i, h]),
            ("up_proj", [i, h]),
            ("down_proj", [h, i]),
        ] {
            put(
                format!("{p}.mlp.{name}.weight"),
                rng.tensor(&shape, 0.25, 0.0),
            );
        }
    }
    put(
        "thinker.model.norm.weight".into(),
        rng.tensor(&[h], 0.3, 1.0),
    );
    if with_head {
        put(
            "thinker.lm_head.weight".into(),
            rng.tensor(&[cfg.vocab_size, h], 1.0, 0.0),
        );
    }
    m
}

fn backend_config() -> Qwen3AsrBackendConfig {
    Qwen3AsrBackendConfig {
        prompt_prefix: PREFIX.to_vec(),
        audio_pad_id: PAD,
        prompt_suffix: SUFFIX.to_vec(),
        eos_token: 63,
        stop_tokens: vec![63],
        max_audio_samples: 200_000,
        return_logits: true,
    }
}

fn build(cfg: Qwen3AsrBackendConfig, text: Qwen3AsrTextConfig) -> Qwen3AsrBackend {
    let a = audio_config();
    let encoder = Qwen3AsrAudioEncoder::load(a.clone(), synthetic_vb(&a, DType::F32)).unwrap();
    let vb = VarBuilder::from_tensors(weights(&text, 11, true), DType::F32, &Device::Cpu);
    let decoder = Qwen3AsrDecoder::load(text, vb.pp("thinker")).unwrap();
    Qwen3AsrBackend::new(encoder, decoder, cfg).unwrap()
}

fn backend() -> Qwen3AsrBackend {
    build(backend_config(), text_config())
}

fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.next_unit() * amp).collect()
}

fn tone(range: std::ops::Range<usize>, amp: f32) -> Vec<f32> {
    range
        .map(|i| amp * (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16_000.0).sin())
        .collect()
}

fn cat(parts: &[&[f32]]) -> Vec<f32> {
    parts.concat()
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Verification through the backend and through one whole-prompt prefill must agree.
///
/// The whole-prompt path never splits the prompt into a cached prefix and a recomputed
/// remainder, so it is an independent oracle for the reuse logic.
fn oracle_logits(b: &Qwen3AsrBackend, pcm: &[f32], draft: &[TokenId]) -> Vec<Vec<f32>> {
    let audio = b.encoder().encode_pcm(pcm).unwrap();
    let n = audio.dim(0).unwrap();
    let mut ids = PREFIX.to_vec();
    ids.extend(std::iter::repeat_n(PAD, n));
    ids.extend(SUFFIX);
    let prompt = b.decoder().embed_prompt(&ids, PAD, &audio).unwrap();
    let k = draft.len();
    let embeds = if k > 1 {
        let d = b.decoder().embed_tokens(&draft[..k - 1]).unwrap();
        Tensor::cat(&[&prompt, &d], 0).unwrap()
    } else {
        prompt
    };
    let mut cache = b.decoder().new_cache();
    b.decoder()
        .prefill(&embeds, &mut cache, k.max(1))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
}

/// Offline greedy decode with no prior encoding, stopping on the backend's stop set.
fn offline(b: &Qwen3AsrBackend, pcm: &[f32], max_tokens: usize) -> Vec<TokenId> {
    let enc = b.encode_audio(pcm, None).unwrap();
    let mut kv = b.new_kv_cache().unwrap();
    let mut tokens = vec![b.verify_draft(&enc, &mut kv, &[]).unwrap().argmax_tokens[0]];
    while !b.is_eos_token(*tokens.last().unwrap()) && tokens.len() < max_tokens {
        let seed = *tokens.last().unwrap();
        tokens.push(b.decode_step(&enc, &mut kv, seed).unwrap().token);
    }
    tokens
}

/// Greedy tokens of `pcm` ignoring stop ids (the 63 stop is outside what these clips emit often).
fn greedy(b: &Qwen3AsrBackend, pcm: &[f32], n: usize) -> Vec<TokenId> {
    let enc = b.encode_audio(pcm, None).unwrap();
    let mut kv = b.new_kv_cache().unwrap();
    let mut tokens = vec![b.verify_draft(&enc, &mut kv, &[]).unwrap().argmax_tokens[0]];
    while tokens.len() < n {
        let seed = *tokens.last().unwrap();
        tokens.push(b.decode_step(&enc, &mut kv, seed).unwrap().token);
    }
    tokens
}

fn verify_logits(
    b: &Qwen3AsrBackend,
    enc: &Qwen3AsrAudioEncoding,
    draft: &[TokenId],
) -> (Vec<TokenId>, Vec<Vec<f32>>, Qwen3AsrDecodeState) {
    let mut kv = b.new_kv_cache().unwrap();
    let out = b.verify_draft(enc, &mut kv, draft).unwrap();
    (out.argmax_tokens, out.logits.unwrap(), kv)
}

#[test]
fn backend_is_send_sync_and_defaults_to_the_pinned_prompt() {
    fn check<T: Send + Sync>() {}
    check::<Qwen3AsrBackend>();
    check::<Qwen3AsrAudioEncoding>();
    check::<Qwen3AsrDecodeState>();
    let cfg = Qwen3AsrBackendConfig::default();
    assert_eq!(cfg.prompt_prefix.len(), 9);
    assert_eq!(cfg.prompt_suffix.len(), 6);
    assert_eq!(cfg.audio_pad_id, 151_676);
    assert_eq!(cfg.eos_token, 151_645);
    assert_eq!(cfg.stop_tokens, vec![151_645, 151_643]);
    assert_eq!(cfg.max_audio_samples, 120 * 16_000);
}

#[test]
fn verify_matches_whole_prompt_prefill_for_every_window_layout() {
    let b = backend();
    // Short clip (no final window), exactly at the first and second finality boundary,
    // one past it, and several windows with a partial tail.
    for samples in [
        8_000,
        final_at(1) - 1,
        final_at(1),
        final_at(1) + 1,
        20_000,
        26_000,
    ] {
        let pcm = noise(samples, 0.4, 3);
        let enc = b.encode_audio(&pcm, None).unwrap();
        let cached = 3 + 8 * enc.final_windows();
        assert_eq!(enc.cached_prefix_len(), cached, "samples {samples}");
        for draft in [vec![], vec![9], vec![9, 10, 11, 12]] {
            let (tokens, logits, kv) = verify_logits(&b, &enc, &draft);
            let expected = oracle_logits(&b, &pcm, &draft);
            assert_eq!(tokens.len(), draft.len().max(1));
            let d = max_diff(&logits, &expected);
            assert!(d < 1e-4, "samples {samples} draft {draft:?}: diff {d}");
            let ids = argmax_rows(&Tensor::new(expected.clone(), &Device::Cpu).unwrap()).unwrap();
            assert_eq!(tokens, ids, "samples {samples} draft {draft:?}");
            let p = PREFIX.len() + enc.audio_tokens() + SUFFIX.len();
            assert_eq!(kv.prompt_len(), p);
            assert_eq!(kv.len(), p + draft.len().saturating_sub(1));
        }
    }
}

#[test]
fn finality_boundary_decides_cached_windows() {
    let b = backend();
    let pcm = noise(final_at(1) + 1, 0.4, 3);
    for (samples, finals) in [
        (8_000, 1),
        (final_at(1) - 1, 1),
        (final_at(1), 2),
        (final_at(2) - 1, 2),
        (final_at(2), 3),
    ] {
        let pcm = [pcm.clone(), noise(30_000, 0.4, 4)].concat();
        let enc = b.encode_audio(&pcm[..samples], None).unwrap();
        assert_eq!(enc.final_windows(), finals, "samples {samples}");
        assert_eq!(enc.stats().encoded_windows, finals);
        assert_eq!(enc.stats().reused_windows, 0);
    }
}

#[test]
fn growing_audio_reuses_final_windows_and_prefix_kv() {
    let b = backend();
    let head = noise(20_000, 0.5, 5);
    let full = cat(&[&head, &noise(9_000, 0.05, 6)]);
    let prior = b.encode_audio(&head, None).unwrap();
    assert_eq!(prior.final_windows(), 3);
    assert_eq!(prior.stats().prefilled_prefix_tokens, PREFIX.len());

    let grown = b.encode_audio(&full, Some(&prior)).unwrap();
    let fresh = b.encode_audio(&full, None).unwrap();
    // Window 3 becomes final only when the clip is long enough. The three prior windows are
    // reused with their KV, and the prompt prefix is not prefilled again.
    assert_eq!(grown.final_windows(), fresh.final_windows());
    assert_eq!(grown.stats().reused_windows, 3);
    assert_eq!(grown.stats().encoded_windows, grown.final_windows() - 3);
    assert_eq!(grown.stats().prefilled_prefix_tokens, 0);
    assert_eq!(fresh.stats().reused_windows, 0);
    assert_eq!(grown.cached_prefix_len(), fresh.cached_prefix_len());

    for draft in [vec![], vec![5, 6, 7]] {
        let (t1, l1, _) = verify_logits(&b, &grown, &draft);
        let (t2, l2, _) = verify_logits(&b, &fresh, &draft);
        assert_eq!(t1, t2);
        assert!(max_diff(&l1, &l2) < 1e-6);
    }
}

#[test]
fn unchanged_audio_reuses_everything_and_prior_is_not_mutated() {
    let b = backend();
    let pcm = noise(20_000, 0.5, 5);
    let prior = b.encode_audio(&pcm, None).unwrap();
    let before = verify_logits(&b, &prior, &[8, 9]).1;
    let again = b.encode_audio(&pcm, Some(&prior)).unwrap();
    assert_eq!(again.stats().reused_windows, prior.final_windows());
    assert_eq!(again.stats().encoded_windows, 0);
    assert_eq!(again.floor(), prior.floor());
    // Verifying and extending forks must leave the shared prefix untouched.
    let after = verify_logits(&b, &prior, &[8, 9]).1;
    assert_eq!(before, after);
    assert_eq!(prior.cached_prefix_len(), 3 + 8 * 3);
    assert_eq!(verify_logits(&b, &again, &[8, 9]).1, before);
}

#[test]
fn floor_change_with_inactive_clamp_keeps_windows_and_stays_exact() {
    let b = backend();
    let head = noise(20_000, 0.1, 7);
    let full = cat(&[&head, &noise(9_000, 0.9, 8)]);
    let prior = b.encode_audio(&head, None).unwrap();
    let grown = b.encode_audio(&full, Some(&prior)).unwrap();
    let fresh = b.encode_audio(&full, None).unwrap();
    // The louder tail raises the global maximum and with it the floor.
    assert!(grown.floor() > prior.floor() + 0.5);
    // Scenario guard: every reused window has a raw minimum above both floors, so the clamp is
    // inactive under both and the features are identical.
    let mel = b.encoder().frontend().compute(&full).unwrap();
    for k in 0..3 {
        let min = mel.min_raw_in_frames(40 * k, 40 * (k + 1)).unwrap();
        assert!(min >= grown.floor().max(prior.floor()), "window {k}");
    }
    assert_eq!(grown.stats().reused_windows, 3);
    let (t1, l1, _) = verify_logits(&b, &grown, &[5, 6]);
    let (t2, l2, _) = verify_logits(&b, &fresh, &[5, 6]);
    assert_eq!(t1, t2);
    assert!(max_diff(&l1, &l2) < 1e-6);
}

#[test]
fn floor_change_with_active_clamp_invalidates_and_rebuilds() {
    let b = backend();
    // A quiet tone has a dynamic range wider than 8 decades, so its low bins sit at the clamp.
    let head = tone(0..20_000, 0.01);
    let full = cat(&[&head, &tone(20_000..29_000, 0.9)]);
    let prior = b.encode_audio(&head, None).unwrap();
    let grown = b.encode_audio(&full, Some(&prior)).unwrap();
    let fresh = b.encode_audio(&full, None).unwrap();
    let mel = b.encoder().frontend().compute(&full).unwrap();
    // Scenario guard: the floor moved up and the clamp is active for window 0 under it.
    assert!(grown.floor() > prior.floor());
    assert!(mel.min_raw_in_frames(0, 40).unwrap() < grown.floor());
    assert_eq!(
        grown.stats().reused_windows,
        0,
        "stale windows must not be reused"
    );
    assert_eq!(grown.stats().encoded_windows, grown.final_windows());
    assert_eq!(grown.stats().prefilled_prefix_tokens, PREFIX.len());
    let (t1, l1, _) = verify_logits(&b, &grown, &[5, 6]);
    let (t2, l2, _) = verify_logits(&b, &fresh, &[5, 6]);
    assert_eq!(t1, t2);
    assert!(max_diff(&l1, &l2) < 1e-6);
    // The stale prior itself still verifies against its own audio.
    let oracle = oracle_logits(&b, &head, &[5, 6]);
    assert!(max_diff(&verify_logits(&b, &prior, &[5, 6]).1, &oracle) < 1e-4);
}

#[test]
fn altered_samples_invalidate_from_the_first_affected_window() {
    let b = backend();
    let pcm = noise(26_000, 0.4, 9);
    let prior = b.encode_audio(&pcm, None).unwrap();
    assert_eq!(prior.final_windows(), 4);
    // Window `k` is determined by samples up to `final_at(k) + 1`. Pick an index inside each.
    for (index, expect_reused) in [
        (10, 0),
        (WINDOW_SAMPLES + 100, 1),
        (2 * WINDOW_SAMPLES + 100, 2),
        (3 * WINDOW_SAMPLES + 100, 3),
        (final_at(3) + 50, 4),
    ] {
        let mut altered = pcm.clone();
        altered[index] = if altered[index] > 0.0 { -0.3 } else { 0.3 };
        let reused = b.encode_audio(&altered, Some(&prior)).unwrap();
        let fresh = b.encode_audio(&altered, None).unwrap();
        assert_eq!(
            reused.stats().reused_windows,
            expect_reused,
            "index {index}"
        );
        let (t1, l1, _) = verify_logits(&b, &reused, &[5, 6]);
        let (t2, l2, _) = verify_logits(&b, &fresh, &[5, 6]);
        assert_eq!(t1, t2, "index {index}");
        assert!(max_diff(&l1, &l2) < 1e-6, "index {index}");
    }
}

#[test]
fn shorter_or_unrelated_prior_is_never_trusted() {
    let b = backend();
    let pcm = noise(26_000, 0.4, 9);
    let prior = b.encode_audio(&pcm, None).unwrap();
    // A prefix of the prior audio may reuse windows that are final in both.
    let shorter = b.encode_audio(&pcm[..16_000], Some(&prior)).unwrap();
    assert_eq!(shorter.stats().reused_windows, shorter.final_windows());
    let fresh = b.encode_audio(&pcm[..16_000], None).unwrap();
    assert_eq!(
        verify_logits(&b, &shorter, &[5]).1,
        verify_logits(&b, &fresh, &[5]).1
    );
    // Entirely different audio and a prior from another backend instance both rebuild.
    let other = noise(26_000, 0.4, 10);
    assert_eq!(
        b.encode_audio(&other, Some(&prior))
            .unwrap()
            .stats()
            .reused_windows,
        0
    );
    let b2 = backend();
    let foreign = b2.encode_audio(&pcm, Some(&prior)).unwrap();
    assert_eq!(foreign.stats().reused_windows, 0);
    // A foreign encoding is rejected at verification time.
    let mut kv = b2.new_kv_cache().unwrap();
    assert!(matches!(
        b2.verify_draft(&prior, &mut kv, &[]),
        Err(Qwen3AsrError::StateMismatch(_))
    ));
}

#[test]
fn verification_offsets_are_aligned_for_empty_and_nonempty_drafts() {
    let b = backend();
    let pcm = noise(22_000, 0.4, 12);
    let g = greedy(&b, &pcm, 8);
    let enc = b.encode_audio(&pcm, None).unwrap();
    // Exact greedy prefixes of every length verify position by position (Section 2 semantics).
    for k in 0..=6 {
        let (tokens, _, kv) = verify_logits(&b, &enc, &g[..k]);
        if k == 0 {
            assert_eq!(tokens, vec![g[0]]);
        } else {
            assert_eq!(tokens, g[..k].to_vec(), "k {k}");
        }
        assert_eq!(kv.len(), kv.prompt_len() + k.saturating_sub(1));
    }
    // After a wrong token at position 2 the earlier argmaxes are unaffected.
    let mut wrong = g[..5].to_vec();
    wrong[2] = (wrong[2] + 1) % 62;
    let (tokens, _, _) = verify_logits(&b, &enc, &wrong);
    assert_eq!(tokens[..3], g[..3]);
}

#[test]
fn rollback_then_decode_matches_offline_greedy() {
    let b = backend();
    let pcm = noise(22_000, 0.4, 12);
    let g = greedy(&b, &pcm, 10);
    let enc = b.encode_audio(&pcm, None).unwrap();
    let mut draft = g[..6].to_vec();
    draft[3] = (draft[3] + 7) % 62;
    // Only positions before the altered token (index 3) are valid cache contents.
    for keep in [0usize, 1, 2, 3] {
        let mut kv = b.new_kv_cache().unwrap();
        b.verify_draft(&enc, &mut kv, &draft).unwrap();
        b.rollback_kv(&mut kv, keep).unwrap();
        assert_eq!(kv.len(), kv.prompt_len() + keep);
        let mut out = g[..keep].to_vec();
        // Seed with the verified token at `keep`, as the core loop does.
        let mut seed = g[keep];
        out.push(seed);
        while out.len() < 10 {
            seed = b.decode_step(&enc, &mut kv, seed).unwrap().token;
            out.push(seed);
        }
        assert_eq!(out, g, "keep {keep}");
    }
    // Rollback never reaches into the prompt and never past the cached length.
    let mut kv = b.new_kv_cache().unwrap();
    assert!(matches!(
        b.rollback_kv(&mut kv, 0),
        Err(Qwen3AsrError::InvalidRollback { .. })
    ));
    b.verify_draft(&enc, &mut kv, &draft).unwrap();
    let len = kv.len();
    assert!(matches!(
        b.rollback_kv(&mut kv, draft.len()),
        Err(Qwen3AsrError::InvalidRollback { .. })
    ));
    assert_eq!(kv.len(), len);
}

#[test]
fn forked_passes_do_not_observe_each_other() {
    let b = backend();
    let pcm = noise(22_000, 0.4, 12);
    let enc = b.encode_audio(&pcm, None).unwrap();
    let (_, l_ref, _) = verify_logits(&b, &enc, &[5, 6, 7]);
    let mut kv_a = b.new_kv_cache().unwrap();
    let mut kv_b = b.new_kv_cache().unwrap();
    b.verify_draft(&enc, &mut kv_a, &[5, 6, 7]).unwrap();
    b.verify_draft(&enc, &mut kv_b, &[9, 9, 9]).unwrap();
    b.rollback_kv(&mut kv_a, 0).unwrap();
    for t in [3, 4, 5] {
        b.decode_step(&enc, &mut kv_a, t).unwrap();
    }
    assert_eq!(kv_b.len(), kv_b.prompt_len() + 2);
    assert_eq!(verify_logits(&b, &enc, &[5, 6, 7]).1, l_ref);
}

#[test]
fn error_paths_are_reported_and_leave_state_unchanged() {
    let b = backend();
    let pcm = noise(22_000, 0.4, 12);
    let enc = b.encode_audio(&pcm, None).unwrap();

    assert!(matches!(
        b.encode_audio(&[], None),
        Err(Qwen3AsrError::EmptyAudio)
    ));
    assert!(matches!(
        b.encode_audio(&[0.0, f32::NAN], None),
        Err(Qwen3AsrError::InvalidSample { .. })
    ));
    assert!(matches!(
        b.encode_audio(&[0.5, 1.5], None),
        Err(Qwen3AsrError::InvalidSample { .. })
    ));
    let too_long = vec![0.0; 200_001];
    assert!(matches!(
        b.encode_audio(&too_long, None),
        Err(Qwen3AsrError::AudioTooLong {
            samples: 200_001,
            max_samples: 200_000
        })
    ));

    // A decode step needs a prefilled state that belongs to the encoding.
    let mut fresh_kv = b.new_kv_cache().unwrap();
    assert!(matches!(
        b.decode_step(&enc, &mut fresh_kv, 3),
        Err(Qwen3AsrError::StateMismatch(_))
    ));
    let other = b.encode_audio(&noise(12_000, 0.4, 1), None).unwrap();
    assert_ne!(other.audio_tokens(), enc.audio_tokens());
    let mut kv = b.new_kv_cache().unwrap();
    b.verify_draft(&enc, &mut kv, &[5]).unwrap();
    assert!(matches!(
        b.decode_step(&other, &mut kv, 3),
        Err(Qwen3AsrError::StateMismatch(_))
    ));

    // A bad draft id fails without touching the state.
    let len = kv.len();
    assert!(matches!(
        b.verify_draft(&enc, &mut kv, &[5, 64, 6]),
        Err(Qwen3AsrError::InvalidInput(_))
    ));
    assert_eq!((kv.len(), kv.prompt_len()), (len, enc.audio_tokens() + 5));
    assert!(matches!(
        b.decode_step(&enc, &mut kv, 64),
        Err(Qwen3AsrError::InvalidInput(_))
    ));
    assert_eq!(kv.len(), len);

    // Exceeding the model's position limit is an error, not a truncation. The prefix of
    // 3 + 3 * 8 = 27 positions fits a limit of 28, the 30-token prompt does not.
    let limited = |max| {
        build(
            backend_config(),
            Qwen3AsrTextConfig {
                max_position_embeddings: max,
                ..text_config()
            },
        )
    };
    let clip = noise(20_000, 0.4, 1);
    let at_limit = limited(28);
    let enc_limit = at_limit.encode_audio(&clip, None).unwrap();
    let mut kv = at_limit.new_kv_cache().unwrap();
    assert!(matches!(
        at_limit.verify_draft(&enc_limit, &mut kv, &[]),
        Err(Qwen3AsrError::ContextTooLong { .. })
    ));
    assert!(kv.is_empty() && kv.prompt_len() == 0);
    assert!(matches!(
        limited(20).encode_audio(&clip, None),
        Err(Qwen3AsrError::ContextTooLong { .. })
    ));
}

#[test]
fn configuration_is_validated() {
    let a = audio_config();
    let enc = || Qwen3AsrAudioEncoder::load(a.clone(), synthetic_vb(&a, DType::F32)).unwrap();
    let text = text_config();
    let dec = |t: &Qwen3AsrTextConfig| {
        let vb = VarBuilder::from_tensors(weights(t, 11, true), DType::F32, &Device::Cpu);
        Qwen3AsrDecoder::load(t.clone(), vb.pp("thinker")).unwrap()
    };
    let try_cfg = |f: &dyn Fn(&mut Qwen3AsrBackendConfig)| {
        let mut c = backend_config();
        f(&mut c);
        Qwen3AsrBackend::new(enc(), dec(&text), c)
    };
    assert!(try_cfg(&|_| {}).is_ok());
    assert!(try_cfg(&|c| c.stop_tokens = vec![1]).is_err());
    assert!(try_cfg(&|c| c.prompt_suffix = vec![]).is_err());
    assert!(try_cfg(&|c| c.prompt_prefix = vec![]).is_err());
    assert!(try_cfg(&|c| c.prompt_suffix.push(PAD)).is_err());
    assert!(try_cfg(&|c| c.stop_tokens.push(64)).is_err());
    assert!(try_cfg(&|c| c.max_audio_samples = 100).is_err());
    assert!(try_cfg(&|c| c.max_audio_samples = 1_200 * 16_000 + 1).is_err());
    // Encoder width must equal the decoder hidden size.
    let wide = Qwen3AsrTextConfig {
        hidden_size: 48,
        ..text.clone()
    };
    let r = Qwen3AsrBackend::new(enc(), dec(&wide), backend_config());
    assert!(matches!(r, Err(Qwen3AsrError::InvalidConfig(_))));
}

#[test]
fn both_stop_ids_stop_generation_and_keep_their_identity() {
    // Pick two distinct tokens that the model emits at known positions.
    let probe = backend();
    let pcm = noise(22_000, 0.4, 12);
    let g = greedy(&probe, &pcm, 24);
    let first_new = |from: usize, avoid: &[u32]| {
        (from..g.len())
            .find(|&i| !g[..i].contains(&g[i]) && !avoid.contains(&g[i]))
            .expect("greedy output has distinct tokens")
    };
    let a = first_new(2, &[]);
    let b_idx = first_new(a + 2, &[g[a]]);
    let (alt, eos) = (g[a], g[b_idx]);

    let cfg = Qwen3AsrBackendConfig {
        eos_token: eos,
        stop_tokens: vec![eos, alt],
        ..backend_config()
    };
    let b = build(cfg, text_config());
    assert_eq!(b.eos_token(), eos);
    assert!(b.is_eos_token(eos) && b.is_eos_token(alt));
    assert!(!b.is_eos_token(g[0]) || g[0] == alt || g[0] == eos);
    // Wrappers forward the predicate.
    assert!(Arc::new(&b).is_eos_token(alt));
    assert!(<&Qwen3AsrBackend as DevlBackend>::is_eos_token(&&b, alt));

    // Offline decode stops at the alternate stop id and returns it unchanged.
    let expected = g[..=a].to_vec();
    assert_eq!(offline(&b, &pcm, 24), expected);
    assert_ne!(*expected.last().unwrap(), b.eos_token());

    for promotion in [false, true] {
        let config = SessionConfig {
            max_tokens: 24,
            enable_checkpoint_promotion: promotion,
        };
        // Drafting and committing through the session ends at the same place.
        let mut s = DevlSession::new(&b, config).unwrap();
        s.push_audio(&pcm[..14_000]).unwrap();
        s.run_draft().unwrap();
        s.push_audio(&pcm[14_000..]).unwrap();
        let draft = s.run_draft().unwrap().clone();
        assert!(draft.reached_eos, "draft must stop on either stop id");
        assert_eq!(draft.tokens, expected);
        assert_eq!(s.commit().unwrap().tokens, expected);

        // A draft that ends in the alternate stop id and verifies is accepted whole.
        let mut s = DevlSession::new(&b, config).unwrap();
        s.push_audio(&pcm).unwrap();
        s.seed_draft_for_test(expected.clone(), pcm.len());
        let r = s.commit().unwrap();
        assert_eq!(r.tokens, expected);
        assert_eq!((r.accepted_prefix_len, r.decode_steps), (expected.len(), 0));

        // An alternate stop id inside a longer draft ends acceptance there.
        let mut s = DevlSession::new(&b, config).unwrap();
        s.push_audio(&pcm).unwrap();
        let mut long = expected.clone();
        long.extend_from_slice(&g[a + 1..a + 4]);
        s.seed_draft_for_test(long, pcm.len() + 1);
        let r = s.commit().unwrap();
        assert_eq!(r.tokens, expected);
        assert_eq!(r.decode_steps, 0);
    }

    // The default single-stop predicate is unchanged for the same weights.
    let single = build(backend_config(), text_config());
    assert!(single.is_eos_token(63) && !single.is_eos_token(alt));
}

#[test]
fn session_with_growing_audio_matches_offline_greedy() {
    let cfg = Qwen3AsrBackendConfig {
        eos_token: 63,
        stop_tokens: (50..64).collect(),
        ..backend_config()
    };
    let b = build(cfg, text_config());
    let cases: Vec<Vec<f32>> = vec![
        noise(20_000, 0.5, 21),
        // Louder audio arrives later, so cached windows go stale between drafts.
        cat(&[&noise(15_000, 0.05, 22), &noise(14_000, 0.9, 23)]),
        cat(&[&tone(0..16_000, 0.01), &tone(16_000..30_000, 0.9)]),
        // Short clip padded at the tail.
        noise(5_000, 0.4, 24),
    ];
    for (n, pcm) in cases.iter().enumerate() {
        for promotion in [false, true] {
            let config = SessionConfig {
                max_tokens: 12,
                enable_checkpoint_promotion: promotion,
            };
            let expected = offline(&b, pcm, config.max_tokens);
            let mut s = DevlSession::new(&b, config).unwrap();
            for chunk in pcm.chunks(4_800) {
                s.push_audio(chunk).unwrap();
                s.run_draft().unwrap();
            }
            let result = s.commit().unwrap();
            assert_eq!(result.tokens, expected, "case {n} promotion {promotion}");
        }
        // No draft at all (empty draft path).
        let config = SessionConfig {
            max_tokens: 12,
            enable_checkpoint_promotion: false,
        };
        let mut s = DevlSession::new(&b, config).unwrap();
        s.push_audio(pcm).unwrap();
        assert_eq!(s.commit().unwrap().tokens, offline(&b, pcm, 12), "case {n}");
    }
}
