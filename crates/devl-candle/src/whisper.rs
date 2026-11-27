//! Stateless Whisper audio encoder and KV-cached text decoder over Candle.
//!
//! Unlike `candle_transformers::models::whisper::model::TextDecoder` (which
//! mutably stores cross-attention KV inside attention blocks and recomputes
//! self-attention over the full prefix on every step), [`AudioEncoder`] and
//! [`TextDecoder`] here are stateless (`&self`, `Send + Sync`) and accept an
//! explicit [`WhisperKvCache`] per pass. This allows `Arc<CandleWhisperBackend>`
//! to be shared across [`devl_core::PriorityScheduler`] worker threads without
//! locking model weights, while supporting $O(1)$ self-attention KV truncation
//! via `Tensor::narrow` on [`devl_core::DevlBackend::rollback_kv`].

use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm,
    linear, linear_no_bias,
};
use candle_transformers::models::whisper::Config as WhisperModelConfig;
use devl_core::TokenId;

use crate::{CandleBackendError, WhisperKvCache};

fn load_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder<'_>,
) -> CandleResult<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn load_layer_norm(size: usize, vb: VarBuilder<'_>) -> CandleResult<LayerNorm> {
    layer_norm(size, 1e-5, vb)
}

fn sinusoids(
    length: usize,
    channels: usize,
    device: &Device,
    dtype: DType,
) -> CandleResult<Tensor> {
    let half_channels = channels / 2;
    let max_timescale = 10_000.0_f32;
    let log_timescale_increment = if half_channels > 1 {
        max_timescale.ln() / (half_channels - 1) as f32
    } else {
        0.0
    };
    let inv_timescales: Vec<f32> = (0..half_channels)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0_u32, length as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let shape = (length, half_channels);
    let scaled_time = (arange.broadcast_as(shape)? * inv_timescales.broadcast_as(shape)?)?;
    Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    scale: f64,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        let head_dim = n_state / n_head;
        let scale = (head_dim as f64).powf(-0.25);
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            scale,
        })
    }

    fn reshape_head(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        x.reshape((n_batch, n_ctx, self.n_head, n_state / self.n_head))?
            .transpose(1, 2)
    }

    fn project_kv(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let k = (self.reshape_head(&self.key.forward(x)?)? * self.scale)?.contiguous()?;
        let v = self.reshape_head(&self.value.forward(x)?)?.contiguous()?;
        Ok((k, v))
    }

    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let q = (self.reshape_head(q)? * self.scale)?.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut qk = q.matmul(&k_t)?;
        if let Some(mask) = mask {
            let n_ctx = q.dim(2)?;
            let mask_slice = mask.narrow(0, 0, n_ctx)?.narrow(1, 0, n_ctx)?;
            qk = qk.broadcast_add(&mask_slice)?;
        }
        let weights = candle_nn::ops::softmax_last_dim(&qk)?;
        let wv = weights
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .flatten_from(2)?;
        self.out.forward(&wv)
    }

    /// Stateless self-attention with no KV cache (used by encoder and no-cache reference passes).
    fn forward_no_cache(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let q = self.query.forward(x)?;
        let (k, v) = self.project_kv(x)?;
        self.attend(&q, &k, &v, mask)
    }

    /// Prefill causal self-attention over `x`, overwriting `cache` with the new `(k, v)`.
    fn forward_self_prefill(
        &self,
        x: &Tensor,
        mask: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> CandleResult<Tensor> {
        let q = self.query.forward(x)?;
        let (k, v) = self.project_kv(x)?;
        *cache = Some((k.clone(), v.clone()));
        self.attend(&q, &k, &v, Some(mask))
    }

    /// Single-step self-attention appending `x` (`[1, 1, d_model]`) onto `cache`.
    fn forward_self_step(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> CandleResult<Tensor> {
        let q = self.query.forward(x)?;
        let (k_new, v_new) = self.project_kv(x)?;
        let (k, v) = match cache.as_ref() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[prev_k, &k_new], 2)?,
                Tensor::cat(&[prev_v, &v_new], 2)?,
            ),
            None => (k_new, v_new),
        };
        *cache = Some((k.clone(), v.clone()));
        self.attend(&q, &k, &v, None)
    }

    /// Cross-attention over encoder hidden states `xa`, caching `(k, v)` per utterance.
    fn forward_cross(
        &self,
        x: &Tensor,
        xa: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        flush_cache: bool,
    ) -> CandleResult<Tensor> {
        let q = self.query.forward(x)?;
        if flush_cache {
            *cache = None;
        }
        let (k, v) = if let Some((k, v)) = cache.as_ref() {
            (k.clone(), v.clone())
        } else {
            let (k, v) = self.project_kv(xa)?;
            *cache = Some((k.clone(), v.clone()));
            (k, v)
        };
        self.attend(&q, &k, &v, None)
    }
}

#[derive(Debug, Clone)]
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(
        n_state: usize,
        n_head: usize,
        has_cross_attn: bool,
        vb: VarBuilder<'_>,
    ) -> CandleResult<Self> {
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = load_layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if has_cross_attn {
            let cross = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
            let cross_ln = load_layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross, cross_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = load_layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward_mlp(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mlp = self
            .mlp_linear2
            .forward(&self.mlp_linear1.forward(&self.mlp_ln.forward(x)?)?.gelu()?)?;
        x + mlp
    }

    fn forward_encoder(&self, x: &Tensor) -> CandleResult<Tensor> {
        let attn = self
            .attn
            .forward_no_cache(&self.attn_ln.forward(x)?, None)?;
        let x = (x + attn)?;
        self.forward_mlp(&x)
    }

    fn forward_decoder_no_cache(
        &self,
        x: &Tensor,
        xa: &Tensor,
        mask: &Tensor,
    ) -> CandleResult<Tensor> {
        let attn = self
            .attn
            .forward_no_cache(&self.attn_ln.forward(x)?, Some(mask))?;
        let mut x = (x + attn)?;
        if let Some((cross, ln)) = &self.cross_attn {
            let mut scratch = None;
            x = (&x + cross.forward_cross(&ln.forward(&x)?, xa, &mut scratch, true)?)?;
        }
        self.forward_mlp(&x)
    }

    fn forward_decoder_prefill(
        &self,
        x: &Tensor,
        xa: &Tensor,
        mask: &Tensor,
        self_cache: &mut Option<(Tensor, Tensor)>,
        cross_cache: &mut Option<(Tensor, Tensor)>,
    ) -> CandleResult<Tensor> {
        let attn = self
            .attn
            .forward_self_prefill(&self.attn_ln.forward(x)?, mask, self_cache)?;
        let mut x = (x + attn)?;
        if let Some((cross, ln)) = &self.cross_attn {
            x = (&x + cross.forward_cross(&ln.forward(&x)?, xa, cross_cache, true)?)?;
        }
        self.forward_mlp(&x)
    }

    fn forward_decoder_step(
        &self,
        x: &Tensor,
        xa: &Tensor,
        self_cache: &mut Option<(Tensor, Tensor)>,
        cross_cache: &mut Option<(Tensor, Tensor)>,
    ) -> CandleResult<Tensor> {
        let attn = self
            .attn
            .forward_self_step(&self.attn_ln.forward(x)?, self_cache)?;
        let mut x = (x + attn)?;
        if let Some((cross, ln)) = &self.cross_attn {
            x = (&x + cross.forward_cross(&ln.forward(&x)?, xa, cross_cache, false)?)?;
        }
        self.forward_mlp(&x)
    }
}

/// Stateless (`&self`, `Send + Sync`) Whisper audio encoder.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    dtype: DType,
}

impl AudioEncoder {
    /// Load Whisper encoder weights from `vb` (typically rooted at `model.encoder`).
    pub fn load(vb: VarBuilder<'_>, cfg: &WhisperModelConfig) -> CandleResult<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let dtype = vb.dtype();

        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };

        let conv1 = load_conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = load_conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(n_ctx, n_state, vb.device(), dtype)?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(format!("layers.{i}")))
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let ln_post = load_layer_norm(n_state, vb.pp("layer_norm"))?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            dtype,
        })
    }

    /// Encode a log-mel spectrogram tensor `[1, num_mel_bins, 3000]` into hidden
    /// states of shape `[1, 1500, d_model]`.
    pub fn forward(&self, mel: &Tensor) -> CandleResult<Tensor> {
        let x = if mel.dtype() == self.dtype {
            mel.clone()
        } else {
            mel.to_dtype(self.dtype)?
        };
        let x = self.conv1.forward(&x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let (_, seq_len, _) = x.dims3()?;
        let pos = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward_encoder(&x)?;
        }
        self.ln_post.forward(&x)
    }
}

/// Stateless (`&self`, `Send + Sync`) Whisper text decoder with external KV caching.
#[derive(Debug, Clone)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
    suppress_mask: Option<Tensor>,
    max_target_positions: usize,
    device: Device,
}

impl TextDecoder {
    /// Load Whisper decoder weights from `vb` (typically rooted at `model.decoder`).
    pub fn load(vb: VarBuilder<'_>, cfg: &WhisperModelConfig) -> CandleResult<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, vb.pp(format!("layers.{i}")))
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let ln = load_layer_norm(n_state, vb.pp("layer_norm"))?;

        let mask_data: Vec<f32> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask_data, (n_ctx, n_ctx), &device)?.to_dtype(dtype)?;

        let suppress_mask = if cfg.suppress_tokens.is_empty() {
            None
        } else {
            let mut mask_vec = vec![0.0_f32; cfg.vocab_size];
            for &tok in &cfg.suppress_tokens {
                if let Some(slot) = mask_vec.get_mut(tok as usize) {
                    *slot = f32::NEG_INFINITY;
                }
            }
            Some(Tensor::from_vec(mask_vec, cfg.vocab_size, &device)?.to_dtype(dtype)?)
        };

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            suppress_mask,
            max_target_positions: n_ctx,
            device,
        })
    }

    /// Number of decoder transformer layers.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Maximum target sequence length supported by positional embeddings.
    pub fn max_target_positions(&self) -> usize {
        self.max_target_positions
    }

    /// Prefill `tokens` (`[seq_len]`) into `kv` in one parallel causal forward pass.
    ///
    /// Resets both self-attention and cross-attention caches in `kv` before running,
    /// populating `seq_len` positions per layer and returning final hidden states
    /// of shape `[1, seq_len, d_model]`.
    pub fn forward_prefill(
        &self,
        tokens: &[TokenId],
        encoder_hidden_states: &Tensor,
        kv: &mut WhisperKvCache,
    ) -> Result<Tensor, CandleBackendError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(CandleBackendError::InvalidConfig(
                "decoder prefill requires at least one token",
            ));
        }
        if seq_len > self.max_target_positions {
            return Err(CandleBackendError::ContextLengthExceeded {
                seq_len,
                max_target_positions: self.max_target_positions,
            });
        }

        let n_layers = self.blocks.len();
        kv.self_attn_kv.clear();
        kv.self_attn_kv.resize(n_layers, None);
        kv.cross_attn_kv.clear();
        kv.cross_attn_kv.resize(n_layers, None);

        let token_tensor = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let tok_emb = self.token_embedding.forward(&token_tensor)?;
        let pos_emb = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = tok_emb.broadcast_add(&pos_emb)?;

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_decoder_prefill(
                &x,
                encoder_hidden_states,
                &self.mask,
                &mut kv.self_attn_kv[i],
                &mut kv.cross_attn_kv[i],
            )?;
        }

        Ok(self.ln.forward(&x)?)
    }

    /// Execute one autoregressive decode step for `input_token` at position `kv.seq_len()`.
    ///
    /// Appends one position to each layer's self-attention KV cache and reuses
    /// cached cross-attention KV (computing it lazily if empty). Returns final
    /// hidden states of shape `[1, 1, d_model]`.
    pub fn forward_step(
        &self,
        input_token: TokenId,
        encoder_hidden_states: &Tensor,
        kv: &mut WhisperKvCache,
    ) -> Result<Tensor, CandleBackendError> {
        let pos = kv.seq_len();
        if pos >= self.max_target_positions {
            return Err(CandleBackendError::ContextLengthExceeded {
                seq_len: pos + 1,
                max_target_positions: self.max_target_positions,
            });
        }

        let n_layers = self.blocks.len();
        if kv.self_attn_kv.len() != n_layers {
            kv.self_attn_kv.resize(n_layers, None);
        }
        if kv.cross_attn_kv.len() != n_layers {
            kv.cross_attn_kv.resize(n_layers, None);
        }

        let token_tensor = Tensor::new(&[input_token], &self.device)?.unsqueeze(0)?;
        let tok_emb = self.token_embedding.forward(&token_tensor)?;
        let pos_emb = self.positional_embedding.narrow(0, pos, 1)?;
        let mut x = tok_emb.broadcast_add(&pos_emb)?;

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_decoder_step(
                &x,
                encoder_hidden_states,
                &mut kv.self_attn_kv[i],
                &mut kv.cross_attn_kv[i],
            )?;
        }

        Ok(self.ln.forward(&x)?)
    }

    /// Stateless full-sequence causal forward pass without any KV caching.
    ///
    /// Useful as an exact offline reference when verifying KV-cached decoding
    /// parity. Returns final hidden states of shape `[1, tokens.len(), d_model]`.
    pub fn forward_full_no_cache(
        &self,
        tokens: &[TokenId],
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor, CandleBackendError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(CandleBackendError::InvalidConfig(
                "decoder forward requires at least one token",
            ));
        }
        if seq_len > self.max_target_positions {
            return Err(CandleBackendError::ContextLengthExceeded {
                seq_len,
                max_target_positions: self.max_target_positions,
            });
        }

        let token_tensor = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let tok_emb = self.token_embedding.forward(&token_tensor)?;
        let pos_emb = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = tok_emb.broadcast_add(&pos_emb)?;

        for block in &self.blocks {
            x = block.forward_decoder_no_cache(&x, encoder_hidden_states, &self.mask)?;
        }

        Ok(self.ln.forward(&x)?)
    }

    /// Project decoder hidden states `[batch, seq_len, d_model]` to vocabulary
    /// logits `[batch, seq_len, vocab_size]` using tied token embeddings, and
    /// apply configured `suppress_tokens` masking.
    pub fn project_logits(&self, hidden: &Tensor) -> CandleResult<Tensor> {
        let batch_size = hidden.dim(0)?;
        let weight = self
            .token_embedding
            .embeddings()
            .broadcast_left(batch_size)?;
        let logits = hidden.matmul(&weight.t()?)?;
        match &self.suppress_mask {
            Some(mask) => logits.broadcast_add(mask),
            None => Ok(logits),
        }
    }

    /// Compute on-device greedy argmax token IDs along the last dimension.
    pub fn argmax_last_dim(logits: &Tensor) -> CandleResult<Tensor> {
        logits.argmax(D::Minus1)
    }

    /// Truncate every decoder layer's self-attention KV cache to
    /// `kv.prompt_len + accepted_len` using `Tensor::narrow`, keeping
    /// cross-attention KV caches intact.
    pub fn rollback_kv(
        &self,
        kv: &mut WhisperKvCache,
        accepted_len: usize,
    ) -> Result<(), CandleBackendError> {
        let seq_len = kv.seq_len();
        let target_len =
            kv.prompt_len
                .checked_add(accepted_len)
                .ok_or(CandleBackendError::InvalidRollback {
                    requested: accepted_len,
                    prompt_len: kv.prompt_len,
                    seq_len,
                })?;

        if kv.self_attn_kv.len() != self.blocks.len()
            || seq_len < kv.prompt_len
            || target_len > seq_len
        {
            return Err(CandleBackendError::InvalidRollback {
                requested: accepted_len,
                prompt_len: kv.prompt_len,
                seq_len,
            });
        }

        for slot in &mut kv.self_attn_kv {
            let Some((k, v)) = slot.as_mut() else {
                return Err(CandleBackendError::InvalidRollback {
                    requested: accepted_len,
                    prompt_len: kv.prompt_len,
                    seq_len,
                });
            };
            if k.dim(2)? != seq_len || v.dim(2)? != seq_len {
                return Err(CandleBackendError::InvalidRollback {
                    requested: accepted_len,
                    prompt_len: kv.prompt_len,
                    seq_len,
                });
            }
            if target_len < seq_len {
                *k = k.narrow(2, 0, target_len)?;
                *v = v.narrow(2, 0, target_len)?;
            }
        }

        Ok(())
    }
}

/// Combined stateless Whisper model loaded from standard Hugging Face weights.
#[derive(Debug, Clone)]
pub struct WhisperModel {
    /// Whisper audio encoder (`model.encoder.*`).
    pub encoder: AudioEncoder,
    /// Whisper text decoder (`model.decoder.*`).
    pub decoder: TextDecoder,
}

impl WhisperModel {
    /// Load `model.encoder.*` and `model.decoder.*` from `vb`.
    pub fn load(vb: VarBuilder<'_>, cfg: &WhisperModelConfig) -> CandleResult<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), cfg)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), cfg)?;
        Ok(Self { encoder, decoder })
    }
}
