//! Causal Qwen3-ASR text decoder with absolute RoPE positions and an explicit KV cache.
//!
//! Structure (upstream `Qwen3ASRThinkerTextModel`, `docs/ARCHITECTURE.md` Sections 6.2 and 6.6):
//! token embeddings, 28 pre-norm layers (RMSNorm, grouped-query attention with per-head
//! `q_norm`/`k_norm` before RoPE, gated SiLU MLP), a final RMSNorm, and a bias-free head.
//!
//! Position handling: the three upstream multi-axis RoPE sections receive identical positions
//! for a single unpadded sequence, so the rotation equals standard 1-D rotate-half RoPE. The
//! absolute position of a token is the cache length when it is fed. No offset state exists
//! beyond [`Qwen3AsrKvCache::len`].
//!
//! Attention mode: this module is decoder-only. It takes embeddings as input and is agnostic to
//! how the audio embeddings were produced. The audio encoder in this crate uses block-windowed
//! attention, while the upstream default `sdpa` path attends over the whole clip for clips longer
//! than one 8 s block (Section 6.13). Decoder equivalence claims here never depend on that choice.
//!
//! Stop ids: both `151645` and `151643` stop upstream generation ([`super::STOP_TOKEN_IDS`]). This
//! module only predicts tokens and never stops or rewrites them.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias};
use candle_transformers::utils::repeat_kv;

use super::Qwen3AsrError;
use super::config::Qwen3AsrTextConfig;
use super::kv_cache::{LayerKv, Qwen3AsrKvCache};
use super::weights::{text_decoder_var_builder, verify_pinned_shard_sizes};

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &Qwen3AsrTextConfig, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        Ok(Self {
            q_proj: linear_no_bias(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            q_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
        })
    }

    /// `x` is `[len, hidden]`. Returns the attention output and this layer's updated cache.
    fn forward(
        &self,
        x: &Tensor,
        past: Option<&LayerKv>,
        rope: &Rope,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, LayerKv)> {
        let len = x.dim(0)?;
        let split = |y: Tensor, heads: usize| -> candle_core::Result<Tensor> {
            y.reshape((1, len, heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q_proj.forward(x)?, self.heads)?;
        let k = split(self.k_proj.forward(x)?, self.kv_heads)?;
        let v = split(self.v_proj.forward(x)?, self.kv_heads)?;
        // Normalize per head, then rotate. Cached keys are post-RoPE.
        let q = rope.apply(&self.q_norm.forward(&q)?)?;
        let k = rope.apply(&self.k_norm.forward(&k)?)?;

        let (k_all, v_all) = match past {
            Some(p) => (Tensor::cat(&[&p.k, &k], 2)?, Tensor::cat(&[&p.v, &v], 2)?),
            None => (k, v),
        };
        let groups = self.heads / self.kv_heads;
        let k_rep = repeat_kv(k_all.clone(), groups)?.contiguous()?;
        let v_rep = repeat_kv(v_all.clone(), groups)?.contiguous()?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores =
            (q.matmul(&k_rep.transpose(2, 3)?.contiguous()?)? * scale)?.to_dtype(DType::F32)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v_rep.dtype())?;
        let ctx = probs
            .matmul(&v_rep)?
            .transpose(1, 2)?
            .reshape((len, self.heads * self.head_dim))?;
        Ok((self.o_proj.forward(&ctx)?, LayerKv { k: k_all, v: v_all }))
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &Qwen3AsrTextConfig, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        self.down_proj.forward(&(gate * self.up_proj.forward(x)?)?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Layer {
    fn load(cfg: &Qwen3AsrTextConfig, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let eps = cfg.rms_norm_eps;
        Ok(Self {
            attn: Attention::load(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(cfg.hidden_size, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }
}

/// Rotate-half RoPE tables for a run of consecutive absolute positions.
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    /// Tables for positions `offset..offset + len`, each of shape `[len, head_dim / 2]`.
    fn new(
        inv_freq: &[f32],
        offset: usize,
        len: usize,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let half = inv_freq.len();
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for pos in offset..offset + len {
            for &f in inv_freq {
                let angle = pos as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (len, half), device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (len, half), device)?.to_dtype(dtype)?,
        })
    }

    /// `x` is `[1, heads, len, head_dim]`.
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, &self.cos, &self.sin)
    }
}

/// Qwen3-ASR text decoder: prefill, incremental greedy step, and head.
///
/// All methods take `&self`, so one loaded decoder serves many independent
/// [`Qwen3AsrKvCache`] values.
pub struct Qwen3AsrDecoder {
    cfg: Qwen3AsrTextConfig,
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    inv_freq: Vec<f32>,
    dtype: DType,
    device: Device,
}

impl std::fmt::Debug for Qwen3AsrDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrDecoder")
            .field("cfg", &self.cfg)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Qwen3AsrDecoder {
    /// Build the decoder from a [`VarBuilder`] rooted at `thinker`.
    ///
    /// Reads `model.*` and `lm_head.weight`. When `lm_head.weight` is absent the embedding
    /// matrix is used as a tied head. The builder dtype is the compute dtype.
    pub fn load(cfg: Qwen3AsrTextConfig, vb: VarBuilder<'_>) -> Result<Self, Qwen3AsrError> {
        cfg.validate()?;
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let model = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(&cfg, model.pp(format!("layers.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model.pp("norm"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };
        let inv_freq = (0..cfg.head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / cfg.head_dim as f64) as f32)
            .collect();
        Ok(Self {
            cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            inv_freq,
            dtype,
            device,
        })
    }

    /// Load the pinned `Qwen/Qwen3-ASR-1.7B` text decoder from a local snapshot directory.
    ///
    /// Checks that both shards have the pinned names and sizes, that `config.json` equals the
    /// pinned text configuration, then reads only `thinker.model.*` and `thinker.lm_head.*`.
    /// Nothing is downloaded and no audio tower tensor is read.
    pub fn from_checkpoint_dir(
        dir: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, Qwen3AsrError> {
        verify_pinned_shard_sizes(dir)?;
        let config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let cfg = Qwen3AsrTextConfig::from_checkpoint_config_json(&config_json)?;
        if cfg != Qwen3AsrTextConfig::qwen3_asr_1_7b() {
            return Err(Qwen3AsrError::InvalidConfig(
                "text_config differs from the pinned Qwen3-ASR-1.7B configuration".into(),
            ));
        }
        let vb = text_decoder_var_builder(dir, dtype, device)?;
        Self::load(cfg, vb)
    }

    /// Hyperparameters in use.
    pub fn config(&self) -> &Qwen3AsrTextConfig {
        &self.cfg
    }

    /// Compute dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Execution device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Empty cache sized for this decoder.
    pub fn new_cache(&self) -> Qwen3AsrKvCache {
        Qwen3AsrKvCache::new(self.cfg.num_hidden_layers)
    }

    fn check_ids(&self, ids: &[u32]) -> Result<(), Qwen3AsrError> {
        match ids.iter().find(|&&t| t as usize >= self.cfg.vocab_size) {
            Some(t) => Err(Qwen3AsrError::InvalidInput(format!(
                "token id {t} is outside the vocabulary of {}",
                self.cfg.vocab_size
            ))),
            None => Ok(()),
        }
    }

    /// Token embeddings of shape `[ids.len(), hidden_size]`.
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor, Qwen3AsrError> {
        if ids.is_empty() {
            return Err(Qwen3AsrError::InvalidInput("no token ids".into()));
        }
        self.check_ids(ids)?;
        let ids = Tensor::from_slice(ids, ids.len(), &self.device)?;
        Ok(self.embed_tokens.forward(&ids)?)
    }

    /// Embed a prompt and place audio embeddings at every `audio_token_id` position, in order.
    ///
    /// Mirrors the upstream `masked_scatter` (Section 6.5). `audio` is `[n, hidden_size]` and
    /// `n` must equal the number of `audio_token_id` entries in `ids`.
    pub fn embed_prompt(
        &self,
        ids: &[u32],
        audio_token_id: u32,
        audio: &Tensor,
    ) -> Result<Tensor, Qwen3AsrError> {
        let text = self.embed_tokens(ids)?;
        let pads = ids.iter().filter(|&&t| t == audio_token_id).count();
        let (rows, width) = audio.dims2()?;
        if rows != pads || width != self.cfg.hidden_size {
            return Err(Qwen3AsrError::InvalidInput(format!(
                "audio embeddings are [{rows}, {width}] but the prompt has {pads} audio positions \
                 of width {}",
                self.cfg.hidden_size
            )));
        }
        let audio = audio.to_dtype(self.dtype)?.to_device(&self.device)?;
        let mut parts = Vec::new();
        let (mut at, mut used) = (0, 0);
        while at < ids.len() {
            let is_audio = ids[at] == audio_token_id;
            let run = ids[at..]
                .iter()
                .take_while(|&&t| (t == audio_token_id) == is_audio)
                .count();
            if is_audio {
                parts.push(audio.narrow(0, used, run)?);
                used += run;
            } else {
                parts.push(text.narrow(0, at, run)?);
            }
            at += run;
        }
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Run `embeds` (`[len, hidden_size]`) through all layers at positions `cache.len()..`.
    ///
    /// Returns the final-norm hidden states of the last `tail` rows (`None` when `tail` is 0).
    /// The cache is updated only when the whole pass succeeds.
    fn forward(
        &self,
        embeds: &Tensor,
        cache: &mut Qwen3AsrKvCache,
        tail: usize,
    ) -> Result<Option<Tensor>, Qwen3AsrError> {
        let (len, width) = embeds.dims2()?;
        if len == 0 || width != self.cfg.hidden_size {
            return Err(Qwen3AsrError::InvalidInput(format!(
                "embeddings are [{len}, {width}], expected [n > 0, {}]",
                self.cfg.hidden_size
            )));
        }
        if tail > len {
            return Err(Qwen3AsrError::InvalidInput(format!(
                "requested {tail} logit rows from {len} input rows"
            )));
        }
        if cache.num_layers() != self.layers.len() {
            return Err(Qwen3AsrError::InvalidInput(format!(
                "cache has {} layers, decoder has {}",
                cache.num_layers(),
                self.layers.len()
            )));
        }
        let offset = cache.len();
        let total = offset + len;
        if total > self.cfg.max_position_embeddings {
            return Err(Qwen3AsrError::ContextTooLong {
                requested: total,
                max: self.cfg.max_position_embeddings,
            });
        }
        let rope = Rope::new(&self.inv_freq, offset, len, self.dtype, &self.device)?;
        // Row i sits at absolute position offset + i and sees keys 0..=offset + i.
        let mask = if len > 1 {
            let data: Vec<f32> = (0..len)
                .flat_map(|i| {
                    (0..total).map(move |j| {
                        if j > offset + i {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Some(Tensor::from_vec(data, (len, total), &self.device)?)
        } else {
            None
        };

        let mut x = embeds.to_dtype(self.dtype)?.to_device(&self.device)?;
        let mut updated = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let normed = layer.input_layernorm.forward(&x)?;
            let (attn, kv) =
                layer
                    .attn
                    .forward(&normed, cache.layer(index), &rope, mask.as_ref())?;
            x = (x + attn)?;
            let mlp = layer
                .mlp
                .forward(&layer.post_attention_layernorm.forward(&x)?)?;
            x = (x + mlp)?;
            updated.push(kv);
        }
        let logits = if tail == 0 {
            None
        } else {
            let last = x.narrow(0, len - tail, tail)?;
            let hidden = self.norm.forward(&last)?;
            Some(self.lm_head.forward(&hidden)?.to_dtype(DType::F32)?)
        };
        cache.commit(updated, len);
        Ok(logits)
    }

    /// Append `embeds` to the cache without computing logits.
    ///
    /// Positions continue from `cache.len()`, so this also extends an existing prefix.
    pub fn extend(
        &self,
        embeds: &Tensor,
        cache: &mut Qwen3AsrKvCache,
    ) -> Result<(), Qwen3AsrError> {
        self.forward(embeds, cache, 0).map(|_| ())
    }

    /// Causal prefill of `embeds` (`[len, hidden_size]`) at positions `cache.len()..`.
    ///
    /// Returns `f32` logits of shape `[logit_rows, vocab_size]` for the last `logit_rows`
    /// input rows. Row `r` is the next-token distribution after input row `len - logit_rows + r`.
    /// `logit_rows` must be in `1..=len`. Requesting only the rows needed avoids a
    /// `len x vocab_size` head projection.
    pub fn prefill(
        &self,
        embeds: &Tensor,
        cache: &mut Qwen3AsrKvCache,
        logit_rows: usize,
    ) -> Result<Tensor, Qwen3AsrError> {
        if logit_rows == 0 {
            return Err(Qwen3AsrError::InvalidInput(
                "logit_rows must be at least 1 (use extend to skip the head)".into(),
            ));
        }
        self.forward(embeds, cache, logit_rows)?
            .ok_or_else(|| Qwen3AsrError::InvalidInput("prefill produced no logit rows".into()))
    }

    /// Feed one token at absolute position `cache.len()` and return its `f32` logits `[vocab_size]`.
    pub fn step_logits(
        &self,
        token: u32,
        cache: &mut Qwen3AsrKvCache,
    ) -> Result<Tensor, Qwen3AsrError> {
        let embeds = self.embed_tokens(&[token])?;
        Ok(self.prefill(&embeds, cache, 1)?.squeeze(0)?)
    }

    /// Feed one token and return the greedy next token. No suppression is applied.
    pub fn step(&self, token: u32, cache: &mut Qwen3AsrKvCache) -> Result<u32, Qwen3AsrError> {
        let mut next = cache.fork();
        let logits = self.step_logits(token, &mut next)?;
        let token = argmax_rows(&logits.unsqueeze(0)?)?[0];
        *cache = next;
        Ok(token)
    }
}

/// Greedy argmax of each row of `[rows, vocab]` logits. Ties resolve to the lowest id.
///
/// Fails on NaN, which would make the selection undefined.
pub fn argmax_rows(logits: &Tensor) -> Result<Vec<u32>, Qwen3AsrError> {
    let rows = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    rows.iter()
        .map(|row| {
            let mut best: Option<(usize, f32)> = None;
            for (i, &v) in row.iter().enumerate() {
                if v.is_nan() {
                    return Err(Qwen3AsrError::InvalidInput("logits contain NaN".into()));
                }
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((i, v));
                }
            }
            best.map(|(i, _)| i as u32)
                .ok_or_else(|| Qwen3AsrError::InvalidInput("empty logits row".into()))
        })
        .collect()
}
