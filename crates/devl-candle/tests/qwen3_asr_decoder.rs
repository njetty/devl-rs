//! Deterministic Qwen3-ASR decoder tests with synthetic weights.
//!
//! These run in CI without any checkpoint. The opt-in real-checkpoint check lives in
//! `qwen3_asr_decoder_reference.rs`. They exercise the decoder only, so they say nothing about
//! the audio attention mode (block-windowed versus whole-clip, `docs/ARCHITECTURE.md` 6.13).

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use devl_candle::qwen3_asr::weights::{
    AUDIO_TOWER_PREFIX, audio_tower_var_builder, text_decoder_var_builder,
};
use devl_candle::qwen3_asr::{
    Qwen3AsrDecoder, Qwen3AsrError, Qwen3AsrTextConfig, STOP_TOKEN_IDS, argmax_rows, is_stop_token,
};

const AUDIO_ID: u32 = 7;
const TOL: f32 = 1e-4;

/// Small topology. `head_dim` differs from `hidden / heads` and GQA uses two groups.
fn tiny_config() -> Qwen3AsrTextConfig {
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
        max_position_embeddings: 256,
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

fn tiny_decoder() -> Qwen3AsrDecoder {
    let cfg = tiny_config();
    let vb = VarBuilder::from_tensors(weights(&cfg, 11, true), DType::F32, &Device::Cpu);
    Qwen3AsrDecoder::load(cfg, vb.pp("thinker")).unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let (a, b) = (
        a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_close(a: &Tensor, b: &Tensor, what: &str) {
    let d = max_abs_diff(a, b);
    assert!(d < TOL, "{what}: max abs diff {d}");
}

/// Deterministic token sequence in `[8, vocab)`.
fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| 8 + ((rng.next_unit() + 1.0) * 0.5 * 55.0) as u32)
        .collect()
}

/// Logits of every position from a fresh single-pass causal replay.
fn replay_logits(dec: &Qwen3AsrDecoder, ids: &[u32]) -> Tensor {
    let mut cache = dec.new_cache();
    let e = dec.embed_tokens(ids).unwrap();
    dec.prefill(&e, &mut cache, ids.len()).unwrap()
}

/// Logits of each position computed one token at a time.
fn incremental_logits(dec: &Qwen3AsrDecoder, ids: &[u32]) -> Tensor {
    let mut cache = dec.new_cache();
    let rows: Vec<Tensor> = ids
        .iter()
        .map(|&t| dec.step_logits(t, &mut cache).unwrap())
        .collect();
    Tensor::stack(&rows, 0).unwrap()
}

#[test]
fn prefill_and_incremental_agree_with_full_replay() {
    let dec = tiny_decoder();
    let ids = tokens(24, 3);
    let replay = replay_logits(&dec, &ids);
    assert_eq!(replay.dims(), &[24, 64]);
    assert_close(&incremental_logits(&dec, &ids), &replay, "token by token");

    // Chunked prefill with mixed chunk sizes.
    let mut cache = dec.new_cache();
    let mut rows = Vec::new();
    let mut at = 0;
    for chunk in [1, 5, 2, 9, 7] {
        let e = dec.embed_tokens(&ids[at..at + chunk]).unwrap();
        rows.push(dec.prefill(&e, &mut cache, chunk).unwrap());
        at += chunk;
        assert_eq!(cache.len(), at);
    }
    assert_close(&Tensor::cat(&rows, 0).unwrap(), &replay, "chunked prefill");
}

#[test]
fn matches_candle_reference_implementation() {
    // Independent check of the layer structure (QK norm, GQA, RoPE) against Candle's Qwen3.
    let cfg = tiny_config();
    let tensors = weights(&cfg, 21, true);
    let reference_cfg: qwen3::Config = serde_json::from_value(serde_json::json!({
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "head_dim": cfg.head_dim,
        "attention_bias": false,
        "num_key_value_heads": cfg.num_key_value_heads,
        "max_position_embeddings": cfg.max_position_embeddings,
        "sliding_window": null,
        "max_window_layers": cfg.num_hidden_layers,
        "tie_word_embeddings": false,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "use_sliding_window": false,
        "hidden_act": "silu",
    }))
    .unwrap();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).pp("thinker");
    let mut reference = qwen3::ModelForCausalLM::new(&reference_cfg, vb.clone()).unwrap();
    let dec = Qwen3AsrDecoder::load(cfg, vb).unwrap();

    let ids = tokens(12, 5);
    let ours = replay_logits(&dec, &ids);
    // Prefill of the first 7 tokens, then one token at a time with explicit offsets.
    let input = Tensor::from_slice(&ids[..7], (1, 7), &Device::Cpu).unwrap();
    let first = reference.forward(&input, 0).unwrap().squeeze(0).unwrap();
    assert_close(
        &first.get(0).unwrap(),
        &ours.get(6).unwrap(),
        "reference prefill, last row",
    );
    for (p, &t) in ids.iter().enumerate().skip(7) {
        let input = Tensor::from_slice(&[t], (1, 1), &Device::Cpu).unwrap();
        let got = reference.forward(&input, p).unwrap().squeeze(0).unwrap();
        assert_close(
            &got.get(0).unwrap(),
            &ours.get(p).unwrap(),
            &format!("reference step at position {p}"),
        );
    }
}

#[test]
fn positions_are_absolute_and_offset_prefill_matches() {
    let dec = tiny_decoder();
    let ids = tokens(20, 9);
    let replay = replay_logits(&dec, &ids);
    // Splitting the prefill at any boundary, including 1 and len - 1, must not change later rows.
    for split in [1, 2, 5, 9, 15, 19] {
        let mut cache = dec.new_cache();
        let head = dec.embed_tokens(&ids[..split]).unwrap();
        dec.extend(&head, &mut cache).unwrap();
        assert_eq!(cache.len(), split);
        let tail = dec.embed_tokens(&ids[split..]).unwrap();
        let rows = ids.len() - split;
        let got = dec.prefill(&tail, &mut cache, rows).unwrap();
        assert_close(
            &got,
            &replay.narrow(0, split, rows).unwrap(),
            &format!("split at {split}"),
        );
        assert_eq!(cache.len(), ids.len());
    }
    // The same token at a different absolute position yields different logits.
    let mut at_zero = dec.new_cache();
    let a = dec.step_logits(ids[0], &mut at_zero).unwrap();
    let mut later = dec.new_cache();
    dec.extend(&dec.embed_tokens(&ids[1..4]).unwrap(), &mut later)
        .unwrap();
    let b = dec.step_logits(ids[0], &mut later).unwrap();
    assert!(
        max_abs_diff(&a, &b) > 1e-2,
        "position must affect the output"
    );
}

#[test]
fn rollback_boundaries_and_invalid_lengths() {
    let dec = tiny_decoder();
    let ids = tokens(10, 4);
    let mut cache = dec.new_cache();
    dec.extend(&dec.embed_tokens(&ids).unwrap(), &mut cache)
        .unwrap();
    assert_eq!(cache.len(), 10);

    // Beyond the cached length is rejected and leaves the cache intact.
    for bad in [11, 12, usize::MAX] {
        let err = cache.truncate(bad).unwrap_err();
        assert!(matches!(
            err,
            Qwen3AsrError::InvalidCacheLength { requested, len: 10 } if requested == bad
        ));
        assert_eq!(cache.len(), 10);
    }
    // Current length is a no-op.
    cache.truncate(10).unwrap();
    assert_eq!(cache.len(), 10);
    // Boundaries: one less, middle, one, zero. Each leaves the next position at the new length.
    for keep in [9, 5, 1, 0] {
        cache.truncate(keep).unwrap();
        assert_eq!(cache.len(), keep);
        assert_eq!(cache.is_empty(), keep == 0);
    }
    // After emptying, the cache decodes like a new one.
    let a = dec.step_logits(ids[0], &mut cache).unwrap();
    let b = dec.step_logits(ids[0], &mut dec.new_cache()).unwrap();
    assert_close(&a, &b, "decode after truncate to zero");
    assert!(cache.truncate(2).is_err());
}

#[test]
fn fork_isolation() {
    let dec = tiny_decoder();
    let ids = tokens(12, 6);
    let mut base = dec.new_cache();
    dec.extend(&dec.embed_tokens(&ids).unwrap(), &mut base)
        .unwrap();
    let expected_next = dec.step_logits(30, &mut base.fork()).unwrap();

    // Truncating a fork leaves the original untouched.
    let mut truncated = base.fork();
    truncated.truncate(4).unwrap();
    assert_eq!((truncated.len(), base.len()), (4, 12));
    assert_close(
        &dec.step_logits(30, &mut base.fork()).unwrap(),
        &expected_next,
        "original after fork truncation",
    );

    // Appending to a fork leaves the original and sibling forks untouched.
    let mut grown = base.fork();
    let sibling = base.fork();
    for t in [20, 21, 22] {
        dec.step_logits(t, &mut grown).unwrap();
    }
    assert_eq!((grown.len(), base.len(), sibling.len()), (15, 12, 12));
    assert_close(
        &dec.step_logits(30, &mut base.clone()).unwrap(),
        &expected_next,
        "original after fork append",
    );
    assert_close(
        &dec.step_logits(30, &mut sibling.clone()).unwrap(),
        &expected_next,
        "sibling after fork append",
    );

    // Appending to the original after truncating and regrowing a fork does not leak either way.
    let mut regrown = base.fork();
    regrown.truncate(6).unwrap();
    dec.extend(&dec.embed_tokens(&[40, 41, 42]).unwrap(), &mut regrown)
        .unwrap();
    dec.step_logits(50, &mut base).unwrap();
    assert_eq!((regrown.len(), base.len()), (9, 13));
    let mut fresh = dec.new_cache();
    let mut accepted = ids[..6].to_vec();
    accepted.extend([40, 41, 42]);
    dec.extend(&dec.embed_tokens(&accepted).unwrap(), &mut fresh)
        .unwrap();
    assert_close(
        &dec.step_logits(33, &mut regrown).unwrap(),
        &dec.step_logits(33, &mut fresh).unwrap(),
        "regrown fork versus fresh prefix",
    );
}

#[test]
fn decode_after_rollback_matches_fresh_accepted_prefix() {
    let dec = tiny_decoder();
    // Prompt with an audio span, as the real prompt layout: prefix, audio pads, suffix.
    let audio_rows = 5;
    let mut prompt = vec![20, 21, 22];
    prompt.extend(std::iter::repeat_n(AUDIO_ID, audio_rows));
    prompt.extend([23, 24]);
    let p = prompt.len();
    let audio = Lcg(77).tensor(&[audio_rows, 32], 1.0, 0.0);
    let prompt_embeds = dec.embed_prompt(&prompt, AUDIO_ID, &audio).unwrap();
    assert_eq!(prompt_embeds.dims(), &[p, 32]);

    let generated = tokens(14, 8);
    let mut cache = dec.new_cache();
    dec.extend(&prompt_embeds, &mut cache).unwrap();
    assert_eq!(cache.len(), p);
    for &t in &generated {
        dec.step(t, &mut cache).unwrap();
    }
    assert_eq!(cache.len(), p + generated.len());

    // Roll back to every accepted count, covering n = 0 (prompt only), 1, and the full length.
    for n in [0, 1, 2, 7, 13, 14] {
        let mut rolled = cache.fork();
        rolled.truncate(p + n).unwrap();
        assert_eq!(rolled.len(), p + n);

        let mut fresh = dec.new_cache();
        dec.extend(&prompt_embeds, &mut fresh).unwrap();
        if n > 0 {
            dec.extend(&dec.embed_tokens(&generated[..n]).unwrap(), &mut fresh)
                .unwrap();
        }
        // Continue decoding for several greedy steps from the same next token on both.
        let mut next = 25;
        for step in 0..4 {
            let a = dec.step_logits(next, &mut rolled).unwrap();
            let b = dec.step_logits(next, &mut fresh).unwrap();
            assert_close(&a, &b, &format!("n={n} step={step}"));
            next = argmax_rows(&a.unsqueeze(0).unwrap()).unwrap()[0];
            assert_eq!(next, argmax_rows(&b.unsqueeze(0).unwrap()).unwrap()[0]);
        }
    }
}

#[test]
fn embed_prompt_places_audio_in_order_and_validates() {
    let dec = tiny_decoder();
    let ids = [10, AUDIO_ID, AUDIO_ID, 11, AUDIO_ID, 12];
    let audio = Lcg(5).tensor(&[3, 32], 1.0, 0.0);
    let got = dec.embed_prompt(&ids, AUDIO_ID, &audio).unwrap();
    let text = dec.embed_tokens(&ids).unwrap();
    for (row, src) in [(1, 0), (2, 1), (4, 2)] {
        assert_close(
            &got.get(row).unwrap(),
            &audio.get(src).unwrap(),
            "audio row",
        );
    }
    for row in [0, 3, 5] {
        assert_close(&got.get(row).unwrap(), &text.get(row).unwrap(), "text row");
    }
    let too_few = audio.narrow(0, 0, 2).unwrap();
    assert!(matches!(
        dec.embed_prompt(&ids, AUDIO_ID, &too_few),
        Err(Qwen3AsrError::InvalidInput(_))
    ));
    let wrong_width = Lcg(5).tensor(&[3, 16], 1.0, 0.0);
    assert!(dec.embed_prompt(&ids, AUDIO_ID, &wrong_width).is_err());
    assert!(dec.embed_tokens(&[64]).is_err());
    assert!(dec.embed_tokens(&[]).is_err());
}

#[test]
fn rejects_bad_inputs_without_corrupting_the_cache() {
    let dec = tiny_decoder();
    let mut cache = dec.new_cache();
    let e = dec.embed_tokens(&[8, 9, 10]).unwrap();
    assert!(dec.prefill(&e, &mut cache, 0).is_err());
    assert!(dec.prefill(&e, &mut cache, 4).is_err());
    let wrong = Tensor::zeros((2, 16), DType::F32, &Device::Cpu).unwrap();
    assert!(dec.extend(&wrong, &mut cache).is_err());
    assert_eq!(cache.len(), 0);

    // Position limit: a pass that would exceed it fails and leaves the cache unchanged.
    let max = dec.config().max_position_embeddings;
    let long = dec.embed_tokens(&tokens(max, 2)).unwrap();
    dec.extend(&long, &mut cache).unwrap();
    assert_eq!(cache.len(), max);
    assert!(matches!(
        dec.step(8, &mut cache),
        Err(Qwen3AsrError::ContextTooLong { requested, max: m }) if requested == max + 1 && m == max
    ));
    assert_eq!(cache.len(), max);

    // A cache from a decoder with a different depth is rejected.
    let mut cfg = tiny_config();
    cfg.num_hidden_layers = 1;
    let other = Qwen3AsrDecoder::load(
        cfg.clone(),
        VarBuilder::from_tensors(weights(&cfg, 3, true), DType::F32, &Device::Cpu).pp("thinker"),
    )
    .unwrap();
    assert!(dec.extend(&e, &mut other.new_cache()).is_err());
}

#[test]
fn greedy_argmax_ties_and_nan() {
    let t = Tensor::from_vec(vec![1.0_f32, 3.0, 3.0, 0.0, 5.0, 5.0], (2, 3), &Device::Cpu).unwrap();
    assert_eq!(argmax_rows(&t).unwrap(), vec![1, 1]);
    let nan = Tensor::from_vec(vec![1.0_f32, f32::NAN], (1, 2), &Device::Cpu).unwrap();
    assert!(argmax_rows(&nan).is_err());
}

#[test]
fn both_upstream_stop_ids_are_recognized_and_ids_are_not_rewritten() {
    assert_eq!(STOP_TOKEN_IDS, [151_645, 151_643]);
    assert!(is_stop_token(151_645) && is_stop_token(151_643));
    assert!(!is_stop_token(151_644));
    // The decoder treats a stop id as an ordinary token id: it is neither rewritten nor
    // rejected when it lies inside the vocabulary.
    let mut cfg = tiny_config();
    cfg.vocab_size = 151_700;
    let tiny = tiny_config();
    let mut t = weights(&tiny, 2, false);
    t.insert(
        "thinker.model.embed_tokens.weight".into(),
        Lcg(1).tensor(&[cfg.vocab_size, 32], 1.0, 0.0),
    );
    let dec = Qwen3AsrDecoder::load(
        cfg,
        VarBuilder::from_tensors(t, DType::F32, &Device::Cpu).pp("thinker"),
    )
    .unwrap();
    let e = dec.embed_tokens(&STOP_TOKEN_IDS).unwrap();
    assert_eq!(e.dims(), &[2, 32]);
    let logits = dec.step_logits(151_643, &mut dec.new_cache()).unwrap();
    assert_eq!(logits.dims(), &[151_700]);
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("devl-qwen3-dec-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn selective_loader_reads_decoder_only_and_ties_missing_head() {
    let cfg = tiny_config();
    let dir = temp_dir("load");
    let mut shard_a = weights(&cfg, 31, false);
    // An audio tower tensor must never be read by the decoder loader.
    shard_a.insert(
        format!("{AUDIO_TOWER_PREFIX}conv_out.weight"),
        Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
    );
    let head = shard_a.remove("thinker.model.norm.weight").unwrap();
    let shard_b = HashMap::from([("thinker.model.norm.weight".to_string(), head)]);
    candle_core::safetensors::save(&shard_a, dir.join("a.safetensors")).unwrap();
    candle_core::safetensors::save(&shard_b, dir.join("b.safetensors")).unwrap();
    let mut weight_map = serde_json::Map::new();
    for name in shard_a.keys() {
        weight_map.insert(name.clone(), "a.safetensors".into());
    }
    weight_map.insert("thinker.model.norm.weight".into(), "b.safetensors".into());
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": weight_map }).to_string(),
    )
    .unwrap();

    let vb = text_decoder_var_builder(&dir, DType::F32, &Device::Cpu).unwrap();
    assert!(!vb.contains_tensor("audio_tower.conv_out.weight"));
    let dec = Qwen3AsrDecoder::load(cfg.clone(), vb).unwrap();

    // No lm_head tensor: the embedding is used as the tied head.
    let mut full = weights(&cfg, 31, false);
    let embed = full["thinker.model.embed_tokens.weight"].clone();
    let reference = Qwen3AsrDecoder::load(
        cfg.clone(),
        VarBuilder::from_tensors(full.clone(), DType::F32, &Device::Cpu).pp("thinker"),
    )
    .unwrap();
    let ids = tokens(6, 1);
    assert_close(
        &replay_logits(&dec, &ids),
        &replay_logits(&reference, &ids),
        "loaded versus in-memory",
    );
    full.insert("thinker.lm_head.weight".into(), embed);
    let explicit = Qwen3AsrDecoder::load(
        cfg,
        VarBuilder::from_tensors(full, DType::F32, &Device::Cpu).pp("thinker"),
    )
    .unwrap();
    assert_close(
        &replay_logits(&dec, &ids),
        &replay_logits(&explicit, &ids),
        "tied head equals explicit identical head",
    );

    // The audio loader still works from the same index and does not need decoder tensors.
    assert!(audio_tower_var_builder(&dir, DType::F32, &Device::Cpu).is_ok());

    // An indexed tensor missing from its shard is an error.
    let mut bad_map = weight_map.clone();
    bad_map.insert("thinker.model.ghost.weight".into(), "a.safetensors".into());
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": bad_map }).to_string(),
    )
    .unwrap();
    assert!(matches!(
        text_decoder_var_builder(&dir, DType::F32, &Device::Cpu),
        Err(Qwen3AsrError::InvalidCheckpoint(_))
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn pinned_loader_rejects_unpinned_checkpoints_and_configs() {
    let dir = temp_dir("pinned");
    // Missing pinned shards.
    assert!(matches!(
        Qwen3AsrDecoder::from_checkpoint_dir(&dir, DType::F32, &Device::Cpu),
        Err(Qwen3AsrError::InvalidCheckpoint(_))
    ));
    // Wrong shard size.
    std::fs::write(dir.join("model-00001-of-00002.safetensors"), b"x").unwrap();
    std::fs::write(dir.join("model-00002-of-00002.safetensors"), b"x").unwrap();
    assert!(matches!(
        Qwen3AsrDecoder::from_checkpoint_dir(&dir, DType::F32, &Device::Cpu),
        Err(Qwen3AsrError::InvalidCheckpoint(_))
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn text_config_parsing_and_validation() {
    let json = r#"{"thinker_config":{"text_config":{
        "vocab_size":151936,"hidden_size":2048,"intermediate_size":6144,
        "num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,
        "head_dim":128,"rms_norm_eps":1e-06,"rope_theta":1000000.0,
        "max_position_embeddings":65536,"hidden_act":"silu","attention_bias":false,
        "sliding_window":null,"extra":1}}}"#;
    assert_eq!(
        Qwen3AsrTextConfig::from_checkpoint_config_json(json).unwrap(),
        Qwen3AsrTextConfig::qwen3_asr_1_7b()
    );
    for (from, to) in [
        ("\"silu\"", "\"gelu\""),
        ("\"attention_bias\":false", "\"attention_bias\":true"),
        ("\"sliding_window\":null", "\"sliding_window\":128"),
        ("\"head_dim\":128", "\"head_dim\":127"),
        ("\"num_key_value_heads\":8", "\"num_key_value_heads\":5"),
    ] {
        let bad = json.replace(from, to);
        assert!(
            matches!(
                Qwen3AsrTextConfig::from_checkpoint_config_json(&bad),
                Err(Qwen3AsrError::InvalidConfig(_))
            ),
            "{to}"
        );
    }
    assert!(Qwen3AsrTextConfig::from_checkpoint_config_json("{}").is_err());
}
