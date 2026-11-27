//! Deterministic Whisper log-mel spectrogram frontend.
//!
//! Converts mono 16 kHz `f32` PCM audio into fixed 30-second Whisper log-mel
//! tensors of shape `[1, num_mel_bins, 3000]` (`DType::F32`) using
//! `candle_transformers::models::whisper::audio::pcm_to_mel`.

use std::path::Path;

use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{self, Config as WhisperModelConfig};

use crate::CandleBackendError;

/// Expected Whisper input sample rate in Hz (`16_000`).
pub const WHISPER_SAMPLE_RATE_HZ: usize = whisper::SAMPLE_RATE;

/// Maximum number of mono PCM samples in one 30-second Whisper window (`480_000`).
pub const WHISPER_MAX_SAMPLES: usize = whisper::N_SAMPLES;

/// Number of log-mel time frames in one 30-second Whisper window (`3_000`).
pub const WHISPER_N_FRAMES: usize = whisper::N_FRAMES;

/// Number of FFT frequency bins per mel filter row (`1 + N_FFT / 2 = 201`).
pub const WHISPER_N_FFT_BINS: usize = 1 + whisper::N_FFT / 2;

/// Number of mel bins used by Whisper v1 and v2 models (`80`).
pub const WHISPER_MEL_BINS_V1_V2: usize = 80;

/// Number of mel bins used by Whisper v3 and v3-turbo models (`128`).
pub const WHISPER_MEL_BINS_V3: usize = 128;

fn validate_mel_bin_count(num_mel_bins: usize) -> Result<(), CandleBackendError> {
    if num_mel_bins == WHISPER_MEL_BINS_V1_V2 || num_mel_bins == WHISPER_MEL_BINS_V3 {
        Ok(())
    } else {
        Err(CandleBackendError::UnsupportedMelBins { num_mel_bins })
    }
}

/// Validated Whisper mel filter bank stored in row-major `[num_mel_bins, 201]` layout.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFilterBank {
    num_mel_bins: usize,
    filters: Vec<f32>,
}

impl MelFilterBank {
    /// Construct a validated mel filter bank from flat row-major `f32` weights.
    ///
    /// `num_mel_bins` must be `80` (Whisper v1/v2) or `128` (Whisper v3/v3-turbo),
    /// and `filters.len()` must equal `num_mel_bins * WHISPER_N_FFT_BINS`.
    pub fn new(num_mel_bins: usize, filters: Vec<f32>) -> Result<Self, CandleBackendError> {
        validate_mel_bin_count(num_mel_bins)?;
        let expected_len = num_mel_bins * WHISPER_N_FFT_BINS;
        if filters.len() != expected_len {
            return Err(CandleBackendError::InvalidMelFilterDimensions {
                num_mel_bins,
                expected_len,
                actual_len: filters.len(),
            });
        }
        Ok(Self {
            num_mel_bins,
            filters,
        })
    }

    /// Parse a validated mel filter bank from raw little-endian `f32` bytes
    /// (matching Whisper's `melfilters.bytes` / `melfilters128.bytes` format).
    pub fn from_le_bytes(num_mel_bins: usize, bytes: &[u8]) -> Result<Self, CandleBackendError> {
        validate_mel_bin_count(num_mel_bins)?;
        let expected_len = num_mel_bins * WHISPER_N_FFT_BINS;
        let expected_bytes = expected_len * std::mem::size_of::<f32>();
        if bytes.len() != expected_bytes {
            return Err(CandleBackendError::InvalidMelFilterBytes {
                num_mel_bins,
                expected_bytes,
                actual_bytes: bytes.len(),
            });
        }

        let (chunks, remainder) = bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        let filters = chunks.iter().copied().map(f32::from_le_bytes).collect();
        Ok(Self {
            num_mel_bins,
            filters,
        })
    }

    /// Load a validated mel filter bank from a little-endian `f32` binary file on disk.
    pub fn from_le_bytes_file(
        num_mel_bins: usize,
        path: impl AsRef<Path>,
    ) -> Result<Self, CandleBackendError> {
        let bytes = std::fs::read(path)?;
        Self::from_le_bytes(num_mel_bins, &bytes)
    }

    /// Construct a deterministic Slaney-scaled, area-normalized Whisper mel filter
    /// bank for `80` or `128` bins at `16 kHz` with `N_FFT = 400`.
    pub fn slaney(num_mel_bins: usize) -> Result<Self, CandleBackendError> {
        validate_mel_bin_count(num_mel_bins)?;

        const F_MIN: f64 = 0.0;
        const F_MAX: f64 = (WHISPER_SAMPLE_RATE_HZ as f64) / 2.0;
        const F_SP: f64 = 200.0 / 3.0;
        const MIN_LOG_HZ: f64 = 1000.0;
        const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
        let logstep = (6.4_f64).ln() / 27.0;

        let hz_to_mel = |freq: f64| -> f64 {
            if freq >= MIN_LOG_HZ {
                MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep
            } else {
                (freq - F_MIN) / F_SP
            }
        };
        let mel_to_hz = |mel: f64| -> f64 {
            if mel >= MIN_LOG_MEL {
                MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
            } else {
                F_MIN + F_SP * mel
            }
        };

        let mel_min = hz_to_mel(F_MIN);
        let mel_max = hz_to_mel(F_MAX);
        let num_points = num_mel_bins + 2;
        let mel_freqs: Vec<f64> = (0..num_points)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * (i as f64) / ((num_points - 1) as f64);
                mel_to_hz(mel)
            })
            .collect();

        let fft_freqs: Vec<f64> = (0..WHISPER_N_FFT_BINS)
            .map(|k| (k as f64) * (WHISPER_SAMPLE_RATE_HZ as f64) / (whisper::N_FFT as f64))
            .collect();

        let mut filters = vec![0.0_f32; num_mel_bins * WHISPER_N_FFT_BINS];
        for m in 0..num_mel_bins {
            let f_left = mel_freqs[m];
            let f_center = mel_freqs[m + 1];
            let f_right = mel_freqs[m + 2];
            let enorm = 2.0 / (f_right - f_left);
            for (k, &freq) in fft_freqs.iter().enumerate() {
                let lower = (freq - f_left) / (f_center - f_left);
                let upper = (f_right - freq) / (f_right - f_center);
                let weight = lower.min(upper).max(0.0) * enorm;
                filters[m * WHISPER_N_FFT_BINS + k] = weight as f32;
            }
        }

        Self::new(num_mel_bins, filters)
    }

    /// Number of mel bins (`80` or `128`).
    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }

    /// Underlying flat row-major `[num_mel_bins, 201]` filter weights.
    pub fn as_slice(&self) -> &[f32] {
        &self.filters
    }
}

/// Convert mono 16 kHz `f32` PCM samples into a deterministic Whisper log-mel
/// spectrogram tensor of shape `[1, num_mel_bins, 3000]` (`DType::F32`) on `device`.
///
/// - Empty (`&[]`) and short (`< 480_000` samples) inputs are deterministically
///   zero-padded from stream start to a full 30-second window (`3_000` frames).
/// - Inputs exceeding [`WHISPER_MAX_SAMPLES`] (`480_000` samples / 30 s at 16 kHz)
///   return [`CandleBackendError::AudioTooLong`] rather than silently truncating.
/// - Mismatched `model_config.num_mel_bins` and `mel_filters.num_mel_bins()` return
///   [`CandleBackendError::MelBinMismatch`].
pub fn pcm_to_mel_tensor(
    model_config: &WhisperModelConfig,
    mel_filters: &MelFilterBank,
    pcm: &[f32],
    device: &Device,
) -> Result<Tensor, CandleBackendError> {
    if pcm.len() > WHISPER_MAX_SAMPLES {
        return Err(CandleBackendError::AudioTooLong {
            samples: pcm.len(),
            max_samples: WHISPER_MAX_SAMPLES,
        });
    }

    let num_mel_bins = mel_filters.num_mel_bins();
    if model_config.num_mel_bins != num_mel_bins {
        return Err(CandleBackendError::MelBinMismatch {
            model_mel_bins: model_config.num_mel_bins,
            filter_mel_bins: num_mel_bins,
        });
    }

    // `candle_transformers::models::whisper::audio::pcm_to_mel` computes
    // `n_len = samples.len() / HOP_LENGTH` and pads by `pad = 1500` frames.
    // When `samples.len() < HOP_LENGTH` (including `&[]`), `n_len == 0` yields
    // only 1500 frames instead of 3000. Zero-padding sub-hop inputs to one hop
    // (`HOP_LENGTH = 160` samples) guarantees at least `3000` frames while
    // producing bit-identical frames `0..3000` to padding to `480_000` samples.
    let mel = if pcm.len() < whisper::HOP_LENGTH {
        let mut padded = vec![0.0_f32; whisper::HOP_LENGTH];
        padded[..pcm.len()].copy_from_slice(pcm);
        whisper::audio::pcm_to_mel(model_config, &padded, mel_filters.as_slice())
    } else {
        whisper::audio::pcm_to_mel(model_config, pcm, mel_filters.as_slice())
    };

    let total_frames = mel.len() / num_mel_bins;
    debug_assert_eq!(mel.len(), num_mel_bins * total_frames);
    debug_assert!(total_frames >= WHISPER_N_FRAMES);

    let data = if total_frames == WHISPER_N_FRAMES {
        mel
    } else {
        // Inputs longer than 15 s (`> 240_000` samples) cause `pcm_to_mel` to
        // append an extra 1500-frame zero chunk (`total_frames == 4500`). Slice
        // each mel row to the 30-second window (`0..3000`) on host so the
        // resulting tensor is contiguous `[1, num_mel_bins, 3000]`.
        let mut trimmed = Vec::with_capacity(num_mel_bins * WHISPER_N_FRAMES);
        for bin in 0..num_mel_bins {
            let row_start = bin * total_frames;
            trimmed.extend_from_slice(&mel[row_start..row_start + WHISPER_N_FRAMES]);
        }
        trimmed
    };

    Ok(Tensor::from_vec(
        data,
        (1, num_mel_bins, WHISPER_N_FRAMES),
        device,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn test_whisper_config(num_mel_bins: usize) -> WhisperModelConfig {
        WhisperModelConfig {
            num_mel_bins,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            vocab_size: 51_865,
            max_target_positions: 448,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            suppress_tokens: Vec::new(),
        }
    }

    fn sine_wave(len: usize, freq_hz: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / WHISPER_SAMPLE_RATE_HZ as f32;
                0.25 * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    fn tensor_bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn bit_identical_determinism_80_and_128_bins() {
        let pcm = sine_wave(WHISPER_SAMPLE_RATE_HZ * 2, 440.0);

        for num_mel_bins in [WHISPER_MEL_BINS_V1_V2, WHISPER_MEL_BINS_V3] {
            let cfg = test_whisper_config(num_mel_bins);
            let filters = MelFilterBank::slaney(num_mel_bins).unwrap();

            let first = pcm_to_mel_tensor(&cfg, &filters, &pcm, &Device::Cpu).unwrap();
            let second = pcm_to_mel_tensor(&cfg, &filters, &pcm, &Device::Cpu).unwrap();

            assert_eq!(first.dtype(), DType::F32);
            assert_eq!(first.dims(), &[1, num_mel_bins, WHISPER_N_FRAMES]);
            assert_eq!(tensor_bits(&first), tensor_bits(&second));
        }
    }

    #[test]
    fn empty_and_short_audio_pad_deterministically_to_3000_frames() {
        let cfg = test_whisper_config(WHISPER_MEL_BINS_V1_V2);
        let filters = MelFilterBank::slaney(WHISPER_MEL_BINS_V1_V2).unwrap();

        // Empty slice (&[])
        let empty_a = pcm_to_mel_tensor(&cfg, &filters, &[], &Device::Cpu).unwrap();
        let empty_b = pcm_to_mel_tensor(&cfg, &filters, &[], &Device::Cpu).unwrap();
        assert_eq!(empty_a.dtype(), DType::F32);
        assert_eq!(
            empty_a.dims(),
            &[1, WHISPER_MEL_BINS_V1_V2, WHISPER_N_FRAMES]
        );
        assert_eq!(tensor_bits(&empty_a), tensor_bits(&empty_b));

        // Sub-hop clip (< 160 samples) matches manual zero-padding to 30 s.
        let sub_hop = sine_wave(50, 300.0);
        let mut sub_hop_full = vec![0.0_f32; WHISPER_MAX_SAMPLES];
        sub_hop_full[..sub_hop.len()].copy_from_slice(&sub_hop);
        let mel_sub = pcm_to_mel_tensor(&cfg, &filters, &sub_hop, &Device::Cpu).unwrap();
        let mel_sub_full = pcm_to_mel_tensor(&cfg, &filters, &sub_hop_full, &Device::Cpu).unwrap();
        assert_eq!(
            mel_sub.dims(),
            &[1, WHISPER_MEL_BINS_V1_V2, WHISPER_N_FRAMES]
        );
        assert_eq!(tensor_bits(&mel_sub), tensor_bits(&mel_sub_full));

        // Short clip (1 s) matches manual zero-padding to 30 s.
        let short = sine_wave(WHISPER_SAMPLE_RATE_HZ, 520.0);
        let mut short_full = vec![0.0_f32; WHISPER_MAX_SAMPLES];
        short_full[..short.len()].copy_from_slice(&short);
        let mel_short = pcm_to_mel_tensor(&cfg, &filters, &short, &Device::Cpu).unwrap();
        let mel_short_full = pcm_to_mel_tensor(&cfg, &filters, &short_full, &Device::Cpu).unwrap();
        assert_eq!(tensor_bits(&mel_short), tensor_bits(&mel_short_full));

        // Mid-window clip (> 15 s, where candle's pcm_to_mel emits 4500 frames)
        // trims cleanly to 3000 frames and matches full 30 s zero-padded input.
        let mid = sine_wave(WHISPER_SAMPLE_RATE_HZ * 18, 660.0);
        let mut mid_full = vec![0.0_f32; WHISPER_MAX_SAMPLES];
        mid_full[..mid.len()].copy_from_slice(&mid);
        let mel_mid = pcm_to_mel_tensor(&cfg, &filters, &mid, &Device::Cpu).unwrap();
        let mel_mid_full = pcm_to_mel_tensor(&cfg, &filters, &mid_full, &Device::Cpu).unwrap();
        assert_eq!(
            mel_mid.dims(),
            &[1, WHISPER_MEL_BINS_V1_V2, WHISPER_N_FRAMES]
        );
        assert_eq!(tensor_bits(&mel_mid), tensor_bits(&mel_mid_full));
    }

    #[test]
    fn exact_30s_boundary_succeeds_and_above_30s_errors() {
        let cfg = test_whisper_config(WHISPER_MEL_BINS_V3);
        let filters = MelFilterBank::slaney(WHISPER_MEL_BINS_V3).unwrap();

        let exact_30s = vec![0.0_f32; WHISPER_MAX_SAMPLES];
        let mel = pcm_to_mel_tensor(&cfg, &filters, &exact_30s, &Device::Cpu).unwrap();
        assert_eq!(mel.dims(), &[1, WHISPER_MEL_BINS_V3, WHISPER_N_FRAMES]);

        let over_30s = vec![0.0_f32; WHISPER_MAX_SAMPLES + 1];
        let err = pcm_to_mel_tensor(&cfg, &filters, &over_30s, &Device::Cpu).unwrap_err();
        assert!(matches!(
            err,
            CandleBackendError::AudioTooLong {
                samples: 480_001,
                max_samples: 480_000,
            }
        ));
    }

    #[test]
    fn invalid_mel_filter_dimensions_and_mismatches_return_errors() {
        // Unsupported mel bin count.
        let err = MelFilterBank::new(64, vec![0.0; 64 * WHISPER_N_FFT_BINS]).unwrap_err();
        assert!(matches!(
            err,
            CandleBackendError::UnsupportedMelBins { num_mel_bins: 64 }
        ));

        // Wrong f32 element count for 80 bins.
        let err = MelFilterBank::new(WHISPER_MEL_BINS_V1_V2, vec![0.0; 100]).unwrap_err();
        assert!(matches!(
            err,
            CandleBackendError::InvalidMelFilterDimensions {
                num_mel_bins: 80,
                expected_len: 16_080,
                actual_len: 100,
            }
        ));

        // Wrong byte length for from_le_bytes.
        let err = MelFilterBank::from_le_bytes(WHISPER_MEL_BINS_V3, &[0u8; 12]).unwrap_err();
        assert!(matches!(
            err,
            CandleBackendError::InvalidMelFilterBytes {
                num_mel_bins: 128,
                expected_bytes: 102_912,
                actual_bytes: 12,
            }
        ));

        // Round-trip from_le_bytes works bit-identically.
        let slaney80 = MelFilterBank::slaney(WHISPER_MEL_BINS_V1_V2).unwrap();
        let bytes: Vec<u8> = slaney80
            .as_slice()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let loaded = MelFilterBank::from_le_bytes(WHISPER_MEL_BINS_V1_V2, &bytes).unwrap();
        assert_eq!(loaded, slaney80);

        // Model config vs filter bank bin mismatch.
        let cfg128 = test_whisper_config(WHISPER_MEL_BINS_V3);
        let err = pcm_to_mel_tensor(&cfg128, &slaney80, &[], &Device::Cpu).unwrap_err();
        assert!(matches!(
            err,
            CandleBackendError::MelBinMismatch {
                model_mel_bins: 128,
                filter_mel_bins: 80,
            }
        ));
    }
}
