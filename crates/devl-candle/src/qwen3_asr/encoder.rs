//! Chunked convolution stack and block-attention transformer of the Qwen3-ASR audio tower.
//!
//! Structure (upstream `Qwen3ASRAudioEncoder`):
//!
//! 1. The mel sequence is cut into chunks of `2 * n_window` frames. Each chunk
//!    runs independently through three stride-2 convolutions (GELU after each),
//!    `conv_out`, and a chunk-local sinusoidal position embedding.
//! 2. Tokens are grouped into attention windows of `n_window_infer` frames.
//!    Attention is bidirectional inside a window and never crosses windows.
//! 3. `ln_post`, `proj1`, GELU, `proj2` map each token to `output_dim`.
//!
//! A trailing partial chunk is zero-padded to the longest chunk length of the
//! clip before the convolutions (as upstream batches chunks), and only its
//! valid output tokens are kept. This matters because convolution bias and
//! GELU make padded positions nonzero, which affects the last valid tokens.

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv2d, layer_norm, linear,
    linear_no_bias,
};

use super::Qwen3AsrError;
use super::config::{Qwen3AsrAudioConfig, conv_output_len};
use super::frontend::{Qwen3AsrMel, Qwen3AsrMelFrontend};
use super::weights::audio_tower_var_builder;

const LAYER_NORM_EPS: f64 = 1e-5;

struct EncoderLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    heads: usize,
}

impl EncoderLayer {
    fn load(cfg: &Qwen3AsrAudioConfig, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let d = cfg.d_model;
        let attn = vb.pp("self_attn");
        Ok(Self {
            q_proj: linear(d, d, attn.pp("q_proj"))?,
            k_proj: linear(d, d, attn.pp("k_proj"))?,
            v_proj: linear(d, d, attn.pp("v_proj"))?,
            out_proj: linear(d, d, attn.pp("out_proj"))?,
            self_attn_layer_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("self_attn_layer_norm"))?,
            fc1: linear(d, cfg.encoder_ffn_dim, vb.pp("fc1"))?,
            fc2: linear(cfg.encoder_ffn_dim, d, vb.pp("fc2"))?,
            final_layer_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("final_layer_norm"))?,
            heads: cfg.encoder_attention_heads,
        })
    }

    /// `x` has shape `[tokens, d_model]`; attention spans all `tokens` (one window).
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (t, d) = x.dims2()?;
        let head_dim = d / self.heads;
        let h = self.self_attn_layer_norm.forward(x)?;
        let split = |y: Tensor| -> candle_core::Result<Tensor> {
            y.reshape((t, self.heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = split(self.q_proj.forward(&h)?)?;
        let k = split(self.k_proj.forward(&h)?)?;
        let v = split(self.v_proj.forward(&h)?)?;
        let scale = (head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
        // Softmax in f32 as the upstream eager path does.
        let probs =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        let attn = probs
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, d))?;
        let x = (x + self.out_proj.forward(&attn)?)?;
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        x + h
    }
}

/// Qwen3-ASR audio tower with its mel frontend.
pub struct Qwen3AsrAudioEncoder {
    cfg: Qwen3AsrAudioConfig,
    frontend: Qwen3AsrMelFrontend,
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv_out: Linear,
    /// `[tokens_per_chunk, d_model]` sinusoidal table in the model dtype.
    positions: Tensor,
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    dtype: DType,
    device: Device,
}

impl std::fmt::Debug for Qwen3AsrAudioEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrAudioEncoder")
            .field("cfg", &self.cfg)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// Sinusoidal table `[length, channels]`, `[sin | cos]` halves, as upstream `SinusoidsPositionEmbedding`.
fn sinusoid_table(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let increment = (10_000.0_f64).ln() / (half as f64 - 1.0).max(1.0);
    let inv: Vec<f32> = (0..half)
        .map(|i| ((-increment) as f32 * i as f32).exp())
        .collect();
    let mut table = vec![0.0_f32; length * channels];
    for pos in 0..length {
        for (i, &w) in inv.iter().enumerate() {
            let angle = pos as f32 * w;
            table[pos * channels + i] = angle.sin();
            table[pos * channels + half + i] = angle.cos();
        }
    }
    table
}

impl Qwen3AsrAudioEncoder {
    /// Build the encoder from a [`VarBuilder`] rooted at `thinker.audio_tower`.
    ///
    /// The builder dtype is the compute dtype.
    pub fn load(cfg: Qwen3AsrAudioConfig, vb: VarBuilder<'_>) -> Result<Self, Qwen3AsrError> {
        cfg.validate()?;
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let c = cfg.downsample_hidden_size;
        let conv1 = conv2d(1, c, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2 = conv2d(c, c, 3, conv_cfg, vb.pp("conv2d2"))?;
        let conv3 = conv2d(c, c, 3, conv_cfg, vb.pp("conv2d3"))?;
        let freq_out = cfg.num_mel_bins.div_ceil(2).div_ceil(2).div_ceil(2);
        let conv_out = linear_no_bias(c * freq_out, cfg.d_model, vb.pp("conv_out"))?;
        let tokens = cfg.tokens_per_chunk();
        let positions = Tensor::from_vec(
            sinusoid_table(tokens, cfg.d_model),
            (tokens, cfg.d_model),
            &device,
        )?
        .to_dtype(dtype)?;
        let layers = (0..cfg.encoder_layers)
            .map(|i| EncoderLayer::load(&cfg, vb.pp(format!("layers.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ln_post = layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("ln_post"))?;
        let proj1 = linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;
        let frontend = Qwen3AsrMelFrontend::new(cfg.num_mel_bins)?;
        Ok(Self {
            cfg,
            frontend,
            conv1,
            conv2,
            conv3,
            conv_out,
            positions,
            layers,
            ln_post,
            proj1,
            proj2,
            dtype,
            device,
        })
    }

    /// Load the pinned `Qwen/Qwen3-ASR-1.7B` audio tower from a local snapshot directory.
    ///
    /// Reads `config.json` and `model.safetensors.index.json`, then only the
    /// `thinker.audio_tower.*` tensors from the listed shards. Nothing is downloaded.
    pub fn from_checkpoint_dir(
        dir: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, Qwen3AsrError> {
        let config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let cfg = Qwen3AsrAudioConfig::from_checkpoint_config_json(&config_json)?;
        let vb = audio_tower_var_builder(dir, dtype, device)?;
        Self::load(cfg, vb)
    }

    /// Hyperparameters in use.
    pub fn config(&self) -> &Qwen3AsrAudioConfig {
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

    /// The mel frontend matching this encoder.
    pub fn frontend(&self) -> &Qwen3AsrMelFrontend {
        &self.frontend
    }

    /// Number of attention windows for `frames` mel frames.
    pub fn window_count(&self, frames: usize) -> usize {
        frames.div_ceil(self.cfg.window_frames())
    }

    /// Encode PCM to audio embeddings of shape `[tokens, output_dim]`.
    ///
    /// Applies the frontend (validation, short-clip pad, mel, global clamp) and then
    /// [`Self::encode_features`].
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<Tensor, Qwen3AsrError> {
        let mel = self.frontend.compute(pcm)?;
        self.encode_mel(&mel)
    }

    /// Encode precomputed [`Qwen3AsrMel`] features.
    pub fn encode_mel(&self, mel: &Qwen3AsrMel) -> Result<Tensor, Qwen3AsrError> {
        self.encode_features(&mel.to_tensor(&self.device)?)
    }

    /// Encode normalized features of shape `[num_mel_bins, frames]` (any float dtype).
    pub fn encode_features(&self, features: &Tensor) -> Result<Tensor, Qwen3AsrError> {
        let frames = self.check_features(features)?;
        let windows = self.window_count(frames);
        let parts = (0..windows)
            .map(|w| self.encode_window_checked(features, frames, w))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Encode one attention window of a clip.
    ///
    /// `features` is the whole clip `[num_mel_bins, frames]`, because the padding rule for a
    /// trailing partial chunk depends on the clip length. The result has
    /// `tokens_per_window` rows for a complete window and fewer for the last window.
    pub fn encode_window(
        &self,
        features: &Tensor,
        window_index: usize,
    ) -> Result<Tensor, Qwen3AsrError> {
        let frames = self.check_features(features)?;
        self.encode_window_checked(features, frames, window_index)
    }

    fn check_features(&self, features: &Tensor) -> Result<usize, Qwen3AsrError> {
        let dims = features.dims();
        if dims.len() != 2 || dims[0] != self.cfg.num_mel_bins || dims[1] == 0 {
            return Err(Qwen3AsrError::InvalidFeatures {
                shape: dims.to_vec(),
                num_mel_bins: self.cfg.num_mel_bins,
            });
        }
        Ok(dims[1])
    }

    fn encode_window_checked(
        &self,
        features: &Tensor,
        frames: usize,
        window_index: usize,
    ) -> Result<Tensor, Qwen3AsrError> {
        let windows = self.window_count(frames);
        if window_index >= windows {
            return Err(Qwen3AsrError::WindowOutOfRange {
                index: window_index,
                windows,
            });
        }
        let chunk = self.cfg.chunk_frames();
        let start = window_index * self.cfg.window_frames();
        let end = (start + self.cfg.window_frames()).min(frames);
        // Upstream pads every chunk to the longest chunk of the clip.
        let pad_to = chunk.min(frames);
        let x = features.to_dtype(self.dtype)?;

        let mut chunks = Vec::new();
        let mut valid = Vec::new();
        let mut at = start;
        while at < end {
            let len = chunk.min(end - at);
            let mut c = x.narrow(1, at, len)?;
            if len < pad_to {
                c = c.pad_with_zeros(D::Minus1, 0, pad_to - len)?;
            }
            chunks.push(c);
            valid.push(conv_output_len(len));
            at += len;
        }
        let batch = Tensor::stack(&chunks, 0)?.unsqueeze(1)?;
        let h = self.conv1.forward(&batch)?.gelu_erf()?;
        let h = self.conv2.forward(&h)?.gelu_erf()?;
        let h = self.conv3.forward(&h)?.gelu_erf()?;
        let (n, c, f, t) = h.dims4()?;
        let h = h
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .reshape((n, t, c * f))?;
        let h = self.conv_out.forward(&h)?;
        let h = h.broadcast_add(&self.positions.narrow(0, 0, t)?)?;

        let mut tokens = Vec::with_capacity(n);
        for (i, &v) in valid.iter().enumerate() {
            tokens.push(h.narrow(0, i, 1)?.squeeze(0)?.narrow(0, 0, v)?);
        }
        let mut h = Tensor::cat(&tokens, 0)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.ln_post.forward(&h)?;
        let h = self.proj1.forward(&h)?.gelu_erf()?;
        Ok(self.proj2.forward(&h)?)
    }
}
