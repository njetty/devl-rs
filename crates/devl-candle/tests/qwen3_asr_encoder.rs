//! Deterministic Qwen3-ASR audio encoder tests with synthetic weights.
//!
//! These run in CI without any checkpoint. The real-checkpoint comparison lives in
//! `qwen3_asr_reference.rs` and is opt-in.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use devl_candle::qwen3_asr::{
    Qwen3AsrAudioConfig, Qwen3AsrAudioEncoder, Qwen3AsrError, conv_output_len,
};

/// Small topology with a 10-frame chunk (2 tokens) and a 40-frame window (8 tokens).
fn tiny_config() -> Qwen3AsrAudioConfig {
    Qwen3AsrAudioConfig {
        num_mel_bins: 128,
        d_model: 16,
        encoder_layers: 2,
        encoder_attention_heads: 2,
        encoder_ffn_dim: 32,
        downsample_hidden_size: 4,
        output_dim: 12,
        n_window: 5,
        n_window_infer: 40,
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

fn tiny_encoder(dtype: DType) -> Qwen3AsrAudioEncoder {
    let cfg = tiny_config();
    let vb = synthetic_vb(&cfg, dtype);
    Qwen3AsrAudioEncoder::load(cfg, vb).unwrap()
}

/// Deterministic normalized-feature-like tensor `[128, frames]`.
fn features(frames: usize) -> Tensor {
    let data: Vec<f32> = (0..128 * frames)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) * 1.5 - 0.2)
        .collect();
    Tensor::from_vec(data, (128, frames), &Device::Cpu).unwrap()
}

fn rows(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
}

#[test]
fn pinned_config_arithmetic() {
    let cfg = Qwen3AsrAudioConfig::qwen3_asr_1_7b();
    assert_eq!(cfg.chunk_frames(), 100);
    assert_eq!(cfg.tokens_per_chunk(), 13);
    assert_eq!(cfg.window_frames(), 800);
    assert_eq!(cfg.tokens_per_window(), 104);
    // Values observed from the pinned reference (docs/ARCHITECTURE.md 6.7): F = 1505, N = 196.
    assert_eq!(cfg.output_tokens(1505), 196);
    assert_eq!(cfg.output_tokens(50), 7);
    assert_eq!(cfg.output_tokens(100), 13);
    assert_eq!(cfg.output_tokens(800), 104);
    assert_eq!(cfg.output_tokens(801), 105);
    assert_eq!(cfg.output_tokens(0), 0);
    // Cross-check against the conv chain for every remainder.
    for r in 1..100usize {
        let a = (r - 1) / 2 + 1;
        let b = (a - 1) / 2 + 1;
        assert_eq!(conv_output_len(r), (b - 1) / 2 + 1);
    }
}

#[test]
fn config_json_round_trip_and_rejections() {
    let json = r#"{"thinker_config":{"audio_config":{
        "num_mel_bins":128,"d_model":1024,"encoder_layers":24,"encoder_attention_heads":16,
        "encoder_ffn_dim":4096,"downsample_hidden_size":480,"output_dim":2048,
        "n_window":50,"n_window_infer":800,"activation_function":"gelu","extra":1}}}"#;
    let cfg = Qwen3AsrAudioConfig::from_checkpoint_config_json(json).unwrap();
    assert_eq!(cfg, Qwen3AsrAudioConfig::qwen3_asr_1_7b());
    assert!(Qwen3AsrAudioConfig::from_checkpoint_config_json("{}").is_err());
    assert!(Qwen3AsrAudioConfig::from_checkpoint_config_json("not json").is_err());
    assert!(
        Qwen3AsrAudioConfig::from_checkpoint_config_json(&json.replace("\"gelu\"", "\"relu\""))
            .is_err()
    );
    let mut bad = Qwen3AsrAudioConfig::qwen3_asr_1_7b();
    bad.n_window_infer = 750;
    assert!(bad.validate().is_err());
    bad = Qwen3AsrAudioConfig::qwen3_asr_1_7b();
    bad.encoder_attention_heads = 7;
    assert!(bad.validate().is_err());
}

#[test]
fn token_counts_match_formula_for_full_and_partial_windows() {
    let enc = tiny_encoder(DType::F32);
    let cfg = enc.config().clone();
    assert_eq!((cfg.tokens_per_chunk(), cfg.tokens_per_window()), (2, 8));
    // Exact window (40), exact chunk multiples, partial chunk, partial window, multiple windows.
    for frames in [1usize, 3, 9, 10, 11, 25, 39, 40, 41, 50, 79, 80, 83, 97] {
        let out = enc.encode_features(&features(frames)).unwrap();
        assert_eq!(
            out.dims(),
            &[cfg.output_tokens(frames), 12],
            "frames {frames}"
        );
        assert_eq!(enc.window_count(frames), frames.div_ceil(40));
    }
}

#[test]
fn complete_windows_do_not_depend_on_later_frames() {
    let enc = tiny_encoder(DType::F32);
    let base = features(97);
    let short = base.narrow(1, 0, 80).unwrap();
    let long = enc.encode_features(&base).unwrap();
    let ref_short = enc.encode_features(&short).unwrap();
    // Windows 0 and 1 (frames 0..80) are complete in both clips, so they match exactly.
    assert_eq!(rows(&long.narrow(0, 0, 16).unwrap()), rows(&ref_short));
    // Individual window API agrees with slicing the full result.
    for w in 0..2 {
        let one = enc.encode_window(&base, w).unwrap();
        assert_eq!(one.dims(), &[8, 12]);
        assert_eq!(rows(&one), rows(&long.narrow(0, w * 8, 8).unwrap()));
    }
}

#[test]
fn windows_do_not_attend_across_each_other() {
    let enc = tiny_encoder(DType::F32);
    let a = features(80);
    // Perturb only window 1 and confirm window 0 is bit-identical.
    let mut data = a.to_vec2::<f32>().unwrap();
    for row in &mut data {
        for v in &mut row[40..] {
            *v += 0.5;
        }
    }
    let b = Tensor::from_vec(
        data.into_iter().flatten().collect::<Vec<f32>>(),
        (128, 80),
        &Device::Cpu,
    )
    .unwrap();
    let ea = enc.encode_features(&a).unwrap();
    let eb = enc.encode_features(&b).unwrap();
    assert_eq!(
        rows(&ea.narrow(0, 0, 8).unwrap()),
        rows(&eb.narrow(0, 0, 8).unwrap())
    );
    assert_ne!(
        rows(&ea.narrow(0, 8, 8).unwrap()),
        rows(&eb.narrow(0, 8, 8).unwrap())
    );
}

#[test]
fn attention_is_bidirectional_inside_a_window() {
    let enc = tiny_encoder(DType::F32);
    let a = features(40);
    // Change only the last chunk of the window; earlier tokens must change through attention.
    let mut data = a.to_vec2::<f32>().unwrap();
    for row in &mut data {
        for v in &mut row[30..] {
            *v += 0.7;
        }
    }
    let b = Tensor::from_vec(
        data.into_iter().flatten().collect::<Vec<f32>>(),
        (128, 40),
        &Device::Cpu,
    )
    .unwrap();
    let ea = rows(&enc.encode_features(&a).unwrap());
    let eb = rows(&enc.encode_features(&b).unwrap());
    assert_ne!(ea[0], eb[0], "first token must see the last chunk");
}

#[test]
fn trailing_partial_window_changes_when_frames_are_appended() {
    let enc = tiny_encoder(DType::F32);
    let base = features(95);
    let partial = enc
        .encode_features(&base.narrow(1, 0, 85).unwrap())
        .unwrap();
    let extended = enc.encode_features(&base).unwrap();
    // Window 2 is partial in both (frames 80..85 versus 80..95), so its tokens differ.
    let tail_a = rows(&partial.narrow(0, 16, partial.dims()[0] - 16).unwrap());
    let tail_b = rows(&extended.narrow(0, 16, 2).unwrap());
    assert_ne!(tail_a[0], tail_b[0]);
    // Complete windows are unchanged.
    assert_eq!(
        rows(&partial.narrow(0, 0, 16).unwrap()),
        rows(&extended.narrow(0, 0, 16).unwrap())
    );
}

#[test]
fn partial_chunk_is_zero_padded_to_the_longest_chunk() {
    // For frames >= chunk, a partial tail chunk sees zero padding up to 10 frames. Encoding the
    // same tail alone (clip shorter than a chunk, no padding) differs because conv bias and
    // GELU make padded positions nonzero. This pins the upstream batching behavior.
    let enc = tiny_encoder(DType::F32);
    let base = features(25);
    let full = enc.encode_features(&base).unwrap();
    let tail_alone = enc
        .encode_features(&base.narrow(1, 20, 5).unwrap())
        .unwrap();
    assert_eq!(full.dims()[0], 2 + 2 + conv_output_len(5));
    let tail_in_clip = rows(&full.narrow(0, 4, 1).unwrap());
    // Same token count, different values.
    assert_eq!(tail_in_clip.len(), tail_alone.dims()[0]);
    assert_ne!(tail_in_clip, rows(&tail_alone));
}

#[test]
fn encode_pcm_handles_short_and_invalid_audio() {
    let enc = tiny_encoder(DType::F32);
    let cfg = enc.config().clone();
    let pcm = |n: usize| -> Vec<f32> { (0..n).map(|i| 0.4 * (i as f32 * 0.05).sin()).collect() };
    // 8000 samples is the minimum: 50 frames. Shorter clips pad to it.
    let min = enc.encode_pcm(&pcm(8_000)).unwrap();
    assert_eq!(min.dims()[0], cfg.output_tokens(50));
    let short = enc.encode_pcm(&pcm(1_000)).unwrap();
    assert_eq!(short.dims(), min.dims());
    // Full window plus trailing partial: 11_000 frames/... use 45_000 samples = 281 frames.
    let long = enc.encode_pcm(&pcm(45_000)).unwrap();
    assert_eq!(long.dims()[0], cfg.output_tokens(281));
    assert!(matches!(
        enc.encode_pcm(&[]),
        Err(Qwen3AsrError::EmptyAudio)
    ));
    let mut bad = pcm(9_000);
    bad[5] = f32::NAN;
    assert!(matches!(
        enc.encode_pcm(&bad),
        Err(Qwen3AsrError::InvalidSample { index: 5, .. })
    ));
    bad[5] = 2.0;
    assert!(matches!(
        enc.encode_pcm(&bad),
        Err(Qwen3AsrError::InvalidSample { .. })
    ));
}

#[test]
fn invalid_feature_shapes_and_window_indices_are_rejected() {
    let enc = tiny_encoder(DType::F32);
    let wrong_bins = Tensor::zeros((64, 20), DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        enc.encode_features(&wrong_bins),
        Err(Qwen3AsrError::InvalidFeatures { .. })
    ));
    let empty = Tensor::zeros((128, 0), DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        enc.encode_features(&empty),
        Err(Qwen3AsrError::InvalidFeatures { .. })
    ));
    let rank3 = Tensor::zeros((1, 128, 20), DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        enc.encode_features(&rank3),
        Err(Qwen3AsrError::InvalidFeatures { .. })
    ));
    assert!(matches!(
        enc.encode_window(&features(41), 2),
        Err(Qwen3AsrError::WindowOutOfRange {
            index: 2,
            windows: 2
        })
    ));
}

#[test]
fn encoding_is_deterministic() {
    let enc = tiny_encoder(DType::F32);
    let f = features(83);
    assert_eq!(
        rows(&enc.encode_features(&f).unwrap()),
        rows(&enc.encode_features(&f).unwrap())
    );
}
