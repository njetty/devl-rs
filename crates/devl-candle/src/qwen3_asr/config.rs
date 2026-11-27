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
                .map(|v| v as usize)
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
        if !self.n_window_infer.is_multiple_of(self.chunk_frames()) {
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
