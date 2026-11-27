//! [`DevlBackend`] implementation for Qwen3-ASR with reusable audio windows and prefix KV.
//!
//! The backend wires the windowed audio encoder ([`Qwen3AsrAudioEncoder`]) and the causal text
//! decoder ([`Qwen3AsrDecoder`]) into the DEVL protocol. It contains no verification, prefix
//! matching, rollback selection, or scheduling logic: those stay in `devl-core`
//! (`docs/ARCHITECTURE.md` Sections 2, 3, and 6.8 to 6.9).
//!
//! # Prompt layout
//!
//! ```text
//! positions:  0 .. 8 | 9 .. 9+104m-1 | 9+104m .. 9+N-1 | 9+N .. 14+N | generated
//! content:    prefix | complete windows | partial tail   | suffix       | draft / decode
//! cached:     yes      yes                no                no             per pass
//!                                                         ^ P = 15 + N
//! ```
//!
//! `N` audio tokens replace the single `<|audio_pad|>`. Audio embeddings are decoder inputs at
//! consecutive absolute positions, so there is no separate audio KV.
//!
//! # Reuse and invalidation
//!
//! [`Qwen3AsrAudioEncoding`] is an immutable value. It stores the decoder KV of the prompt prefix
//! plus every *final* window, and the audio embeddings of the remaining tail. A window is final
//! when its last mel frame no longer depends on reflect padding
//! ([`samples_to_finalize_frame`]). `encode_audio(pcm, prior)` reuses the longest prefix of the
//! prior windows for which all of the following hold:
//!
//! - the prior came from this backend instance (so weights, prompt, and stop set are equal);
//! - the samples that determine the window are bit-identical to the retained ones;
//! - the global clamp floor is unchanged, or the window's raw minimum lies at or above both the
//!   old and new floor so the clamp is inactive under both.
//!
//! Reuse is prefix-closed, because the KV of a window was computed on top of every earlier one.
//! Everything after the last reused window (new final windows, the tail, suffix, and draft) is
//! recomputed. A rejected prior never causes an error: the encoding is rebuilt from scratch.
//!
//! # Input bound and memory
//!
//! The decoder caches `2 * layers * kv_heads * head_dim` values per token, which is `57344`
//! values (`224 KiB` in `f32`) for the pinned model, and 13 audio tokens per second. These are
//! sizes derived from shapes, not measurements. Prefix prefill runs one 104-token window at a
//! time, so attention scores stay `heads x 104 x length` instead of `heads x length x length`.
//! `verify_draft` forks the cached prefix and appends, which allocates a second copy of the
//! cache while the encoding stays alive, so budget about three times the cache size at peak
//! (encoding, an extension in flight, and one verification pass).
//!
//! [`Qwen3AsrBackendConfig::max_audio_samples`] therefore defaults to
//! [`DEFAULT_MAX_AUDIO_SAMPLES`] (120 s, about 1.6 thousand audio tokens, about 0.35 GiB of `f32`
//! cache per copy). It is a conservative safety cap and not a validated limit: long-context
//! behavior has been exercised only on the clip lengths listed in `docs/ARCHITECTURE.md`
//! Section 6.15. It can be raised up to [`MAX_SAMPLES`] (1200 s, the upstream limit) by callers
//! who have budgeted the memory. Longer input fails with [`Qwen3AsrError::AudioTooLong`].
//!
//! Mel features are recomputed for the whole buffer on every call, which is linear in the
//! clip length, as is the sample comparison used for validation.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Tensor};
use devl_core::{DevlBackend, StepOutput, TokenId, VerifyOutput};

use super::config::STOP_TOKEN_IDS;
use super::decoder::{Qwen3AsrDecoder, argmax_rows};
use super::encoder::Qwen3AsrAudioEncoder;
use super::frontend::{HOP_LENGTH, MAX_SAMPLES, prepare_pcm, samples_to_finalize_frame};
use super::kv_cache::Qwen3AsrKvCache;
use super::{QWEN3_ASR_SAMPLE_RATE_HZ, Qwen3AsrError};

/// Prompt tokens before the audio for the default (empty context, no forced language) prompt:
/// `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>`.
pub const PROMPT_PREFIX_IDS: [u32; 9] = [
    151_644, 8_948, 198, 151_645, 198, 151_644, 872, 198, 151_669,
];

/// The `<|audio_pad|>` token id that the audio embeddings replace.
pub const AUDIO_PAD_ID: u32 = 151_676;

/// Prompt tokens after the audio: `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`.
pub const PROMPT_SUFFIX_IDS: [u32; 6] = [151_670, 151_645, 198, 151_644, 77_091, 198];

/// Canonical id reported by [`DevlBackend::eos_token`]: `<|im_end|>`.
pub const EOS_TOKEN_ID: u32 = 151_645;

/// Default input cap in samples (120 s at 16 kHz). See the module documentation.
pub const DEFAULT_MAX_AUDIO_SAMPLES: usize = 120 * QWEN3_ASR_SAMPLE_RATE_HZ;

/// Configuration of [`Qwen3AsrBackend`].
///
/// [`Default`] is the pinned checkpoint's prompt without a forced language. The prompt fields
/// are public so that tests can use a small vocabulary. Changing them for the real checkpoint
/// is unsupported until verified (`docs/ARCHITECTURE.md` Section 6.5, decision D2).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AsrBackendConfig {
    /// Tokens placed before the audio. Must be nonempty.
    pub prompt_prefix: Vec<u32>,
    /// Token id that the audio embeddings replace. Must not occur in the prompt text.
    pub audio_pad_id: u32,
    /// Tokens placed after the audio. Must be nonempty.
    pub prompt_suffix: Vec<u32>,
    /// Canonical end-of-sequence id returned by [`DevlBackend::eos_token`].
    pub eos_token: u32,
    /// Every id that stops generation. Must contain `eos_token`. The generated id is kept as is.
    pub stop_tokens: Vec<u32>,
    /// Largest accepted clip in samples.
    ///
    /// Must be in `MIN_SAMPLES..=MAX_SAMPLES`. Defaults to [`DEFAULT_MAX_AUDIO_SAMPLES`].
    pub max_audio_samples: usize,
    /// Whether to copy per-position logits to the host in [`VerifyOutput`] and [`StepOutput`].
    pub return_logits: bool,
}

impl Default for Qwen3AsrBackendConfig {
    fn default() -> Self {
        Self {
            prompt_prefix: PROMPT_PREFIX_IDS.to_vec(),
            audio_pad_id: AUDIO_PAD_ID,
            prompt_suffix: PROMPT_SUFFIX_IDS.to_vec(),
            eos_token: EOS_TOKEN_ID,
            stop_tokens: STOP_TOKEN_IDS.to_vec(),
            max_audio_samples: DEFAULT_MAX_AUDIO_SAMPLES,
            return_logits: false,
        }
    }
}

impl Qwen3AsrBackendConfig {
    fn validate(&self, vocab_size: usize) -> Result<(), Qwen3AsrError> {
        let bad = |msg: String| Err(Qwen3AsrError::InvalidConfig(msg));
        if self.prompt_prefix.is_empty() || self.prompt_suffix.is_empty() {
            return bad("prompt_prefix and prompt_suffix must be nonempty".into());
        }
        if self
            .prompt_prefix
            .iter()
            .chain(&self.prompt_suffix)
            .any(|&t| t == self.audio_pad_id)
        {
            return bad("audio_pad_id must not occur in the prompt prefix or suffix".into());
        }
        if !self.stop_tokens.contains(&self.eos_token) {
            return bad("stop_tokens must contain eos_token".into());
        }
        let ids = self
            .prompt_prefix
            .iter()
            .chain(&self.prompt_suffix)
            .chain(&self.stop_tokens)
            .chain([&self.audio_pad_id]);
        if let Some(t) = ids.into_iter().find(|&&t| t as usize >= vocab_size) {
            return bad(format!(
                "token id {t} is outside the vocabulary of {vocab_size}"
            ));
        }
        if self.max_audio_samples < super::frontend::MIN_SAMPLES
            || self.max_audio_samples > MAX_SAMPLES
        {
            return bad(format!(
                "max_audio_samples must be in {}..={MAX_SAMPLES}",
                super::frontend::MIN_SAMPLES
            ));
        }
        Ok(())
    }
}

/// Counters describing how one [`Qwen3AsrAudioEncoding`] was built.
///
/// They make reuse observable in tests and diagnostics. All counts are in attention windows
/// unless stated otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3AsrEncodeStats {
    /// Final windows taken over from the prior encoding together with their prefix KV.
    pub reused_windows: usize,
    /// Final windows encoded and prefilled by this call.
    pub encoded_windows: usize,
    /// Windows after the last final one, encoded by this call and never cached.
    pub tail_windows: usize,
    /// Prompt prefix tokens prefilled by this call (`0` when the prior prefix was reused).
    pub prefilled_prefix_tokens: usize,
}

/// Bookkeeping for one final window.
#[derive(Debug, Clone, Copy)]
struct FinalWindow {
    /// Minimum raw log-mel value over the window's frames.
    min_raw: f32,
    /// Clamp floor in effect when the window was encoded.
    encoded_floor: f32,
}

/// Encoded audio: prefix KV of every final window plus the tail embeddings.
///
/// Immutable. Created by [`DevlBackend::encode_audio`] and valid only for the
/// [`Qwen3AsrBackend`] that created it (other backends ignore it as a prior and reject it in
/// [`DevlBackend::verify_draft`]).
#[derive(Debug, Clone)]
pub struct Qwen3AsrAudioEncoding {
    backend_id: u64,
    /// Samples after the short-clip pad.
    samples: usize,
    audio_tokens: usize,
    floor: f32,
    windows: Vec<FinalWindow>,
    /// Samples `0..anchor_len(windows.len())`, which determine every final window.
    anchor: Arc<[f32]>,
    /// KV of the prompt prefix and the audio tokens of every final window.
    prefix_kv: Qwen3AsrKvCache,
    /// Embeddings `[rows, hidden]` of the audio after the last final window, in decoder dtype.
    tail: Option<Tensor>,
    stats: Qwen3AsrEncodeStats,
}

impl Qwen3AsrAudioEncoding {
    /// Number of audio tokens `N` that replace `<|audio_pad|>` in the prompt.
    pub fn audio_tokens(&self) -> usize {
        self.audio_tokens
    }

    /// Number of samples encoded, after the short-clip tail pad.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Number of final windows whose prefix KV this encoding holds.
    pub fn final_windows(&self) -> usize {
        self.windows.len()
    }

    /// Length of the cached prefix KV (prompt prefix plus the final windows' audio tokens).
    pub fn cached_prefix_len(&self) -> usize {
        self.prefix_kv.len()
    }

    /// Clamp floor of the whole clip this encoding was built from.
    pub fn floor(&self) -> f32 {
        self.floor
    }

    /// How this encoding was built relative to its prior.
    pub fn stats(&self) -> Qwen3AsrEncodeStats {
        self.stats
    }
}

/// Decoder state of one verification pass.
///
/// Created empty by [`DevlBackend::new_kv_cache`], filled by [`DevlBackend::verify_draft`].
/// It owns its tensors and never aliases the encoding's cached prefix mutably.
#[derive(Debug, Clone)]
pub struct Qwen3AsrDecodeState {
    cache: Qwen3AsrKvCache,
    /// Prompt length `P`, or `0` before the first `verify_draft`.
    prompt_len: usize,
}

impl Qwen3AsrDecodeState {
    /// Prompt length `P` recorded by `verify_draft`, `0` before it.
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// Number of cached positions (`P` plus generated tokens fed so far).
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether no position is cached.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

/// Qwen3-ASR [`DevlBackend`] over Candle.
///
/// All methods take `&self`, so one loaded backend can be shared through an `Arc`.
pub struct Qwen3AsrBackend {
    id: u64,
    encoder: Qwen3AsrAudioEncoder,
    decoder: Qwen3AsrDecoder,
    config: Qwen3AsrBackendConfig,
}

impl std::fmt::Debug for Qwen3AsrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrBackend")
            .field("id", &self.id)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Qwen3AsrBackend {
    /// Combine a loaded encoder and decoder.
    ///
    /// Fails when the encoder output width differs from the decoder hidden size or when
    /// `config` is inconsistent with the decoder vocabulary.
    pub fn new(
        encoder: Qwen3AsrAudioEncoder,
        decoder: Qwen3AsrDecoder,
        config: Qwen3AsrBackendConfig,
    ) -> Result<Self, Qwen3AsrError> {
        if encoder.config().output_dim != decoder.config().hidden_size {
            return Err(Qwen3AsrError::InvalidConfig(format!(
                "encoder output width {} differs from decoder hidden size {}",
                encoder.config().output_dim,
                decoder.config().hidden_size
            )));
        }
        config.validate(decoder.config().vocab_size)?;
        Ok(Self {
            id: NEXT_BACKEND_ID.fetch_add(1, Ordering::Relaxed),
            encoder,
            decoder,
            config,
        })
    }

    /// Load the pinned `Qwen/Qwen3-ASR-1.7B` snapshot from a local directory.
    ///
    /// Reads the audio tower and the text decoder with the same checks as
    /// [`Qwen3AsrAudioEncoder::from_checkpoint_dir`] and [`Qwen3AsrDecoder::from_checkpoint_dir`].
    pub fn from_checkpoint_dir(
        dir: &Path,
        dtype: DType,
        device: &Device,
        config: Qwen3AsrBackendConfig,
    ) -> Result<Self, Qwen3AsrError> {
        let encoder = Qwen3AsrAudioEncoder::from_checkpoint_dir(dir, dtype, device)?;
        let decoder = Qwen3AsrDecoder::from_checkpoint_dir(dir, dtype, device)?;
        Self::new(encoder, decoder, config)
    }

    /// Backend configuration.
    pub fn config(&self) -> &Qwen3AsrBackendConfig {
        &self.config
    }

    /// The audio encoder.
    pub fn encoder(&self) -> &Qwen3AsrAudioEncoder {
        &self.encoder
    }

    /// The text decoder.
    pub fn decoder(&self) -> &Qwen3AsrDecoder {
        &self.decoder
    }

    /// Number of final windows among the windows of a clip of `samples` samples.
    fn final_windows(&self, samples: usize) -> usize {
        let w = self.encoder.config().window_frames();
        let first = samples_to_finalize_frame(w - 1);
        if samples < first {
            0
        } else {
            (samples - first) / (HOP_LENGTH * w) + 1
        }
    }

    /// Number of leading samples that determine the first `windows` final windows.
    fn anchor_len(&self, windows: usize) -> usize {
        match windows {
            0 => 0,
            n => {
                let w = self.encoder.config().window_frames();
                samples_to_finalize_frame(w * (n - 1) + w - 1)
            }
        }
    }

    fn prompt_len(&self, audio_tokens: usize) -> usize {
        self.config.prompt_prefix.len() + audio_tokens + self.config.prompt_suffix.len()
    }

    /// Embeddings of the window `index`, converted to the decoder dtype and device.
    fn window_embeds(&self, features: &Tensor, index: usize) -> Result<Tensor, Qwen3AsrError> {
        let e = self.encoder.encode_window(features, index)?;
        Ok(e.to_dtype(self.decoder.dtype())?
            .to_device(self.decoder.device())?)
    }

    fn check_encoding(&self, encoding: &Qwen3AsrAudioEncoding) -> Result<(), Qwen3AsrError> {
        if encoding.backend_id != self.id {
            return Err(Qwen3AsrError::StateMismatch(
                "audio encoding was created by a different backend instance".into(),
            ));
        }
        Ok(())
    }

    fn check_state(
        &self,
        encoding: &Qwen3AsrAudioEncoding,
        kv: &Qwen3AsrDecodeState,
    ) -> Result<(), Qwen3AsrError> {
        self.check_encoding(encoding)?;
        if kv.prompt_len == 0 {
            return Err(Qwen3AsrError::StateMismatch(
                "decoder state has no prompt (call verify_draft first)".into(),
            ));
        }
        if kv.prompt_len != self.prompt_len(encoding.audio_tokens) {
            return Err(Qwen3AsrError::StateMismatch(format!(
                "decoder state prompt length {} does not match the encoding ({})",
                kv.prompt_len,
                self.prompt_len(encoding.audio_tokens)
            )));
        }
        Ok(())
    }

    /// Longest prefix of the prior's final windows that may be reused for `pcm`.
    fn reusable_windows(
        &self,
        prior: Option<&Qwen3AsrAudioEncoding>,
        pcm: &[f32],
        floor: f32,
        mel: &super::frontend::Qwen3AsrMel,
    ) -> usize {
        let Some(prior) = prior.filter(|p| p.backend_id == self.id) else {
            return 0;
        };
        let common = prior
            .anchor
            .iter()
            .zip(pcm)
            .take_while(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        let candidates = prior.windows.len().min(self.final_windows(common));
        let w = self.encoder.config().window_frames();
        let mut reusable = 0;
        for (k, win) in prior.windows.iter().enumerate().take(candidates) {
            let Some(min_raw) = mel.min_raw_in_frames(k * w, (k + 1) * w) else {
                break;
            };
            // The stored minimum is a property of the (identical) samples. Compare bits so that
            // a mismatch can never be hidden by a tolerance.
            if min_raw.to_bits() != win.min_raw.to_bits() {
                break;
            }
            let unchanged_floor = win.encoded_floor == floor;
            let clamp_inactive = win.min_raw >= win.encoded_floor.max(floor);
            if !(unchanged_floor || clamp_inactive) {
                break;
            }
            reusable += 1;
        }
        reusable
    }
}

impl DevlBackend for Qwen3AsrBackend {
    type Error = Qwen3AsrError;
    type AudioEncoding = Qwen3AsrAudioEncoding;
    type KvCache = Qwen3AsrDecodeState;

    fn eos_token(&self) -> TokenId {
        self.config.eos_token
    }

    fn is_eos_token(&self, token: TokenId) -> bool {
        self.config.stop_tokens.contains(&token)
    }

    fn sample_rate_hz(&self) -> u32 {
        QWEN3_ASR_SAMPLE_RATE_HZ as u32
    }

    fn new_kv_cache(&self) -> Result<Self::KvCache, Self::Error> {
        Ok(Qwen3AsrDecodeState {
            cache: self.decoder.new_cache(),
            prompt_len: 0,
        })
    }

    fn encode_audio(
        &self,
        pcm: &[f32],
        prior_encoding: Option<&Self::AudioEncoding>,
    ) -> Result<Self::AudioEncoding, Self::Error> {
        if pcm.len() > self.config.max_audio_samples {
            return Err(Qwen3AsrError::AudioTooLong {
                samples: pcm.len(),
                max_samples: self.config.max_audio_samples,
            });
        }
        let prepared = prepare_pcm(pcm)?;
        let pcm = prepared.as_ref();
        let mel = self.encoder.frontend().compute(pcm)?;
        let floor = mel.floor();
        let frames = mel.frames();
        let cfg = self.encoder.config();
        let per_window = cfg.tokens_per_window();
        let w = cfg.window_frames();
        let audio_tokens = cfg.output_tokens(frames);
        let final_windows = self.final_windows(pcm.len());
        let total_windows = self.encoder.window_count(frames);
        let reused = self.reusable_windows(prior_encoding, pcm, floor, &mel);
        debug_assert!(reused <= final_windows && final_windows <= total_windows);

        let prefix_len = self.config.prompt_prefix.len();
        let mut stats = Qwen3AsrEncodeStats {
            reused_windows: reused,
            encoded_windows: final_windows - reused,
            tail_windows: total_windows - final_windows,
            prefilled_prefix_tokens: 0,
        };
        let mut windows: Vec<FinalWindow> = prior_encoding
            .map(|p| p.windows[..reused].to_vec())
            .unwrap_or_default();

        let mut cache = match prior_encoding {
            Some(prior) if reused > 0 => {
                let mut c = prior.prefix_kv.fork();
                c.truncate(prefix_len + per_window * reused)?;
                c
            }
            _ => {
                let mut c = self.decoder.new_cache();
                let embeds = self.decoder.embed_tokens(&self.config.prompt_prefix)?;
                self.decoder.extend(&embeds, &mut c)?;
                stats.prefilled_prefix_tokens = prefix_len;
                c
            }
        };

        let features = mel.to_tensor(self.encoder.device())?;
        for k in reused..final_windows {
            let embeds = self.window_embeds(&features, k)?;
            if embeds.dim(0)? != per_window {
                return Err(Qwen3AsrError::InvalidInput(format!(
                    "final window {k} produced {} tokens, expected {per_window}",
                    embeds.dim(0)?
                )));
            }
            self.decoder.extend(&embeds, &mut cache)?;
            let min_raw = mel
                .min_raw_in_frames(k * w, (k + 1) * w)
                .ok_or_else(|| Qwen3AsrError::InvalidInput("window outside the clip".into()))?;
            windows.push(FinalWindow {
                min_raw,
                encoded_floor: floor,
            });
        }

        let tail = if total_windows > final_windows {
            let parts = (final_windows..total_windows)
                .map(|k| self.window_embeds(&features, k))
                .collect::<Result<Vec<_>, _>>()?;
            Some(Tensor::cat(&parts, 0)?)
        } else {
            None
        };
        let tail_rows = tail.as_ref().map_or(Ok(0), |t| t.dim(0))?;
        if per_window * final_windows + tail_rows != audio_tokens {
            return Err(Qwen3AsrError::InvalidInput(format!(
                "encoded {} audio tokens but the clip has {audio_tokens}",
                per_window * final_windows + tail_rows
            )));
        }

        let anchor = match prior_encoding {
            Some(p) if reused == final_windows && p.windows.len() == final_windows => {
                Arc::clone(&p.anchor)
            }
            _ => Arc::from(&pcm[..self.anchor_len(final_windows)]),
        };
        debug_assert!(self.anchor_len(final_windows) <= pcm.len());

        Ok(Qwen3AsrAudioEncoding {
            backend_id: self.id,
            samples: pcm.len(),
            audio_tokens,
            floor,
            windows,
            anchor,
            prefix_kv: cache,
            tail,
            stats,
        })
    }

    fn verify_draft(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        draft: &[TokenId],
    ) -> Result<VerifyOutput, Self::Error> {
        self.check_encoding(encoding)?;
        let k = draft.len();
        // The cached prefix is shared. Appending allocates new tensors, so it is never mutated.
        let mut cache = encoding.prefix_kv.fork();

        let mut parts = Vec::with_capacity(3);
        if let Some(tail) = &encoding.tail {
            parts.push(tail.clone());
        }
        parts.push(self.decoder.embed_tokens(&self.config.prompt_suffix)?);
        let prompt_len = self.prompt_len(encoding.audio_tokens);
        if k > 1 {
            parts.push(self.decoder.embed_tokens(&draft[..k - 1])?);
        }
        let embeds = Tensor::cat(&parts, 0)?;
        let rows = k.max(1);
        let logits = self.decoder.prefill(&embeds, &mut cache, rows)?;
        debug_assert_eq!(cache.len(), prompt_len + k.saturating_sub(1));
        let argmax_tokens = argmax_rows(&logits)?;
        let logits = self
            .config
            .return_logits
            .then(|| logits.to_vec2::<f32>())
            .transpose()?;

        *kv = Qwen3AsrDecodeState { cache, prompt_len };
        Ok(VerifyOutput {
            argmax_tokens,
            logits,
        })
    }

    fn rollback_kv(&self, kv: &mut Self::KvCache, accepted_len: usize) -> Result<(), Self::Error> {
        let target = kv.prompt_len + accepted_len;
        if kv.prompt_len == 0 || target > kv.cache.len() {
            return Err(Qwen3AsrError::InvalidRollback {
                requested: accepted_len,
                prompt_len: kv.prompt_len,
                len: kv.cache.len(),
            });
        }
        kv.cache.truncate(target)
    }

    fn decode_step(
        &self,
        encoding: &Self::AudioEncoding,
        kv: &mut Self::KvCache,
        input_token: TokenId,
    ) -> Result<StepOutput, Self::Error> {
        self.check_state(encoding, kv)?;
        let mut next = kv.cache.fork();
        let logits = self.decoder.step_logits(input_token, &mut next)?;
        let token = argmax_rows(&logits.unsqueeze(0)?)?[0];
        let logits = self
            .config
            .return_logits
            .then(|| logits.to_vec1::<f32>())
            .transpose()?;
        kv.cache = next;
        Ok(StepOutput { token, logits })
    }
}
