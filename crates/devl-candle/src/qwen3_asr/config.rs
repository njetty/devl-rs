//! Audio encoder configuration for the pinned Qwen3-ASR checkpoint.

use serde_json::Value;

use super::Qwen3AsrError;

/// Hyperparameters of the Qwen3-ASR audio tower and its mel frontend.
///
/// Field names follow `thinker_config.audio_config` in the checkpoint
/// `config.json`. Use [`Qwen3AsrAudioConfig::qwen3_asr_1_7b`] for the pinned
/// checkpoint described in `docs/ARCHITECTURE.md` Section 6.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AsrAudioConfig {
    /// Number of mel bins (`128` for the pinned checkpoint).
    pub num_mel_bins: usize,
    /// Transformer width.
    pub d_model: usize,
    /// Number of transformer layers.
    pub encoder_layers: usize,
    /// Number of attention heads per layer.
    pub encoder_attention_heads: usize,
    /// Feed-forward inner width.
    pub encoder_ffn_dim: usize,
    /// Channel width of the three stride-2 convolutions.
    pub downsample_hidden_size: usize,
    /// Width of the final projection, equal to the decoder hidden size.
    pub output_dim: usize,
    /// Half the conv chunk length in mel frames. A chunk is `2 * n_window` frames.
    pub n_window: usize,
    /// Attention window length in mel frames. Must be a multiple of the chunk length.
    pub n_window_infer: usize,
}

/// Mel frames per second of audio (`16 kHz / hop 160`).
pub const MEL_FRAMES_PER_SECOND: usize = 100;

/// Output length of the three stride-2, kernel-3, padding-1 convolutions for `frames` input frames.
///
/// Matches `_get_feat_extract_output_lengths` for one chunk and is `0` for `0`.
pub fn conv_output_len(frames: usize) -> usize {
    let mut len = frames;
    for _ in 0..3 {
        if len == 0 {
            return 0;
        }
        len = (len - 1) / 2 + 1;
    }
    len
}

impl Qwen3AsrAudioConfig {
    /// Values for `Qwen/Qwen3-ASR-1.7B` at the pinned revision.
    pub fn qwen3_asr_1_7b() -> Self {
        Self {
            num_mel_bins: 128,
            d_model: 1024,
            encoder_layers: 24,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            downsample_hidden_size: 480,
            output_dim: 2048,
            n_window: 50,
            n_window_infer: 800,
        }
    }

    /// Parse and validate `thinker_config.audio_config` from a checkpoint `config.json` string.
    ///
    /// The activation must be `gelu`. Unknown fields are ignored.
    pub fn from_checkpoint_config_json(json: &str) -> Result<Self, Qwen3AsrError> {
        let root: Value = serde_json::from_str(json)
            .map_err(|e| Qwen3AsrError::InvalidConfig(format!("config.json is not JSON: {e}")))?;
        let audio = root
            .get("thinker_config")
            .and_then(|v| v.get("audio_config"))
            .ok_or_else(|| {
                Qwen3AsrError::InvalidConfig("missing thinker_config.audio_config".into())
            })?;
        let field = |name: &str| -> Result<usize, Qwen3AsrError> {
            audio
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    Qwen3AsrError::InvalidConfig(format!("missing integer audio_config.{name}"))
                })
        };
        let activation = audio
            .get("activation_function")
            .and_then(Value::as_str)
            .unwrap_or("gelu");
        if activation != "gelu" {
            return Err(Qwen3AsrError::InvalidConfig(format!(
                "unsupported activation_function {activation:?} (expected \"gelu\")"
            )));
        }
        let cfg = Self {
            num_mel_bins: field("num_mel_bins")?,
            d_model: field("d_model")?,
            encoder_layers: field("encoder_layers")?,
            encoder_attention_heads: field("encoder_attention_heads")?,
            encoder_ffn_dim: field("encoder_ffn_dim")?,
            downsample_hidden_size: field("downsample_hidden_size")?,
            output_dim: field("output_dim")?,
            n_window: field("n_window")?,
            n_window_infer: field("n_window_infer")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check internal consistency of the hyperparameters.
    pub fn validate(&self) -> Result<(), Qwen3AsrError> {
        let bad = |msg: &str| Err(Qwen3AsrError::InvalidConfig(msg.to_string()));
        if self.num_mel_bins != 80 && self.num_mel_bins != 128 {
            return bad("num_mel_bins must be 80 or 128");
        }
        if self.encoder_layers == 0
            || self.d_model == 0
            || self.encoder_ffn_dim == 0
            || self.downsample_hidden_size == 0
            || self.output_dim == 0
        {
            return bad("layer and width parameters must be nonzero");
        }
        if self.encoder_attention_heads == 0
            || !self.d_model.is_multiple_of(self.encoder_attention_heads)
        {
            return bad("d_model must be a nonzero multiple of encoder_attention_heads");
        }
        if !self.d_model.is_multiple_of(2) {
            return bad("d_model must be even for sinusoidal position embeddings");
        }
        if self.n_window == 0 || self.n_window_infer == 0 {
            return bad("n_window and n_window_infer must be nonzero");
        }
        let Some(chunk_frames) = self.n_window.checked_mul(2) else {
            return bad("2 * n_window exceeds the platform limit");
        };
        if !self.n_window_infer.is_multiple_of(chunk_frames) {
            return bad("n_window_infer must be a multiple of 2 * n_window");
        }
        Ok(())
    }

    /// Mel frames per convolution chunk (`2 * n_window`, `100` for the pinned checkpoint).
    pub fn chunk_frames(&self) -> usize {
        2 * self.n_window
    }

    /// Encoder tokens produced by one full chunk (`13` for the pinned checkpoint).
    pub fn tokens_per_chunk(&self) -> usize {
        conv_output_len(self.chunk_frames())
    }

    /// Mel frames per attention window (`n_window_infer`, `800` for the pinned checkpoint).
    pub fn window_frames(&self) -> usize {
        self.n_window_infer
    }

    /// Encoder tokens produced by one full attention window (`104` for the pinned checkpoint).
    pub fn tokens_per_window(&self) -> usize {
        self.tokens_per_chunk() * (self.n_window_infer / self.chunk_frames())
    }

    /// Number of encoder tokens for a clip with `frames` mel frames.
    ///
    /// Equals `tokens_per_chunk * (frames / chunk) + conv_output_len(frames % chunk)`.
    pub fn output_tokens(&self, frames: usize) -> usize {
        self.tokens_per_chunk() * (frames / self.chunk_frames())
            + conv_output_len(frames % self.chunk_frames())
    }
}

/// Stop token ids that end generation upstream: `<|im_end|>` then `<|endoftext|>`.
///
/// `generation_config.json` lists both (`docs/ARCHITECTURE.md` Section 6.5). The decoder never
/// rewrites or maps these ids. Mapping them onto the single
/// [`devl_core::DevlBackend::eos_token`] is an integration decision (D1 in Section 6.10).
pub const STOP_TOKEN_IDS: [u32; 2] = [151_645, 151_643];

/// Whether `token` is one of [`STOP_TOKEN_IDS`].
pub fn is_stop_token(token: u32) -> bool {
    STOP_TOKEN_IDS.contains(&token)
}

/// Hyperparameters of the Qwen3-ASR text decoder (`thinker_config.text_config`).
///
/// Use [`Qwen3AsrTextConfig::qwen3_asr_1_7b`] for the pinned checkpoint described in
/// `docs/ARCHITECTURE.md` Section 6.2.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AsrTextConfig {
    /// Vocabulary size (`151936`).
    pub vocab_size: usize,
    /// Residual stream width (`2048`).
    pub hidden_size: usize,
    /// Gated MLP inner width (`6144`).
    pub intermediate_size: usize,
    /// Number of decoder layers (`28`).
    pub num_hidden_layers: usize,
    /// Query heads per layer (`16`).
    pub num_attention_heads: usize,
    /// Key/value heads per layer (`8`).
    pub num_key_value_heads: usize,
    /// Width of one head (`128`). Independent of `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    /// RMSNorm epsilon (`1e-6`).
    pub rms_norm_eps: f64,
    /// RoPE base (`1e6`).
    pub rope_theta: f64,
    /// Largest supported absolute position count (`65536`).
    pub max_position_embeddings: usize,
}

impl Qwen3AsrTextConfig {
    /// Values for `Qwen/Qwen3-ASR-1.7B` at the pinned revision.
    pub fn qwen3_asr_1_7b() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 65_536,
        }
    }

    /// Parse and validate `thinker_config.text_config` from a checkpoint `config.json` string.
    ///
    /// Requires `hidden_act` `silu`, no attention bias, and no sliding window, which are the
    /// only variants the decoder implements. Unknown fields are ignored.
    pub fn from_checkpoint_config_json(json: &str) -> Result<Self, Qwen3AsrError> {
        let root: Value = serde_json::from_str(json)
            .map_err(|e| Qwen3AsrError::InvalidConfig(format!("config.json is not JSON: {e}")))?;
        let text = root
            .get("thinker_config")
            .and_then(|v| v.get("text_config"))
            .ok_or_else(|| {
                Qwen3AsrError::InvalidConfig("missing thinker_config.text_config".into())
            })?;
        let int = |name: &str| -> Result<usize, Qwen3AsrError> {
            text.get(name)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    Qwen3AsrError::InvalidConfig(format!("missing integer text_config.{name}"))
                })
        };
        let float = |name: &str| -> Result<f64, Qwen3AsrError> {
            text.get(name).and_then(Value::as_f64).ok_or_else(|| {
                Qwen3AsrError::InvalidConfig(format!("missing number text_config.{name}"))
            })
        };
        let act = text
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("silu");
        if act != "silu" {
            return Err(Qwen3AsrError::InvalidConfig(format!(
                "unsupported hidden_act {act:?} (expected \"silu\")"
            )));
        }
        if text
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(Qwen3AsrError::InvalidConfig(
                "attention_bias is not supported".into(),
            ));
        }
        if text.get("sliding_window").is_some_and(|v| !v.is_null()) {
            return Err(Qwen3AsrError::InvalidConfig(
                "sliding_window is not supported".into(),
            ));
        }
        let cfg = Self {
            vocab_size: int("vocab_size")?,
            hidden_size: int("hidden_size")?,
            intermediate_size: int("intermediate_size")?,
            num_hidden_layers: int("num_hidden_layers")?,
            num_attention_heads: int("num_attention_heads")?,
            num_key_value_heads: int("num_key_value_heads")?,
            head_dim: int("head_dim")?,
            rms_norm_eps: float("rms_norm_eps")?,
            rope_theta: float("rope_theta")?,
            max_position_embeddings: int("max_position_embeddings")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check internal consistency of the hyperparameters.
    pub fn validate(&self) -> Result<(), Qwen3AsrError> {
        let bad = |msg: &str| Err(Qwen3AsrError::InvalidConfig(msg.to_string()));
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.max_position_embeddings == 0
        {
            return bad("decoder width, depth, vocabulary and context must be nonzero");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return bad("num_attention_heads must be a nonzero multiple of num_key_value_heads");
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return bad("head_dim must be even and nonzero for rotate-half RoPE");
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0)
            || !(self.rope_theta.is_finite() && self.rope_theta > 0.0)
        {
            return bad("rms_norm_eps and rope_theta must be finite and positive");
        }
        if u32::try_from(self.vocab_size).is_err() {
            return bad("vocab_size exceeds the u32 token id range");
        }
        Ok(())
    }
}
