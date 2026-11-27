//! Variable-length log-mel frontend for Qwen3-ASR.
//!
//! Reproduces the `WhisperFeatureExtractor` torch path used by the upstream
//! processor for a single clip: centered reflect-padded STFT with a periodic
//! Hann window, last frame dropped, power spectrum, Slaney mel projection,
//! `log10` with a `1e-10` floor, a clamp at `max - 8.0` where the maximum is
//! taken over every frame of the clip, and `(x + 4) / 4`. There is no padding
//! to 30 seconds and no truncation, so `frames == samples / 160`.

use candle_core::{Device, Tensor};

use super::Qwen3AsrError;
use crate::mel::{MelFilterBank, WHISPER_N_FFT_BINS};

/// Required input sample rate in Hz.
pub const QWEN3_ASR_SAMPLE_RATE_HZ: usize = 16_000;
/// FFT size in samples.
pub const N_FFT: usize = 400;
/// Hop between frames in samples.
pub const HOP_LENGTH: usize = 160;
/// Shortest clip in samples. Shorter input is zero-padded at the tail (0.5 s).
pub const MIN_SAMPLES: usize = 8_000;
/// Longest supported clip in samples (1200 s, the upstream `MAX_ASR_INPUT_SECONDS`).
pub const MAX_SAMPLES: usize = 1_200 * QWEN3_ASR_SAMPLE_RATE_HZ;
/// Distance below the global log-mel maximum at which values are clamped.
pub const LOG_FLOOR_RANGE: f32 = 8.0;
/// Lower bound applied to the mel power before `log10`.
pub const MEL_POWER_FLOOR: f32 = 1e-10;

/// Number of mel frames for `samples` samples (`samples / 160`).
pub fn frame_count(samples: usize) -> usize {
    samples / HOP_LENGTH
}

/// Index of the first sample that makes mel frame `frame` final.
///
/// Frame `i` spans `[160 * i - 200, 160 * i + 200)`, so it no longer depends on
/// reflect padding once at least `160 * i + 200` samples are present.
pub fn samples_to_finalize_frame(frame: usize) -> usize {
    HOP_LENGTH * frame + N_FFT / 2
}

/// Validate `pcm` and apply the short-clip tail pad as a pure function of the input.
///
/// Rejects empty input, samples that are not finite or lie outside `[-1.0, 1.0]`,
/// and clips longer than [`MAX_SAMPLES`]. Clips shorter than [`MIN_SAMPLES`] are
/// zero-padded at the tail to exactly [`MIN_SAMPLES`].
pub fn prepare_pcm(pcm: &[f32]) -> Result<std::borrow::Cow<'_, [f32]>, Qwen3AsrError> {
    if pcm.is_empty() {
        return Err(Qwen3AsrError::EmptyAudio);
    }
    if pcm.len() > MAX_SAMPLES {
        return Err(Qwen3AsrError::AudioTooLong {
            samples: pcm.len(),
            max_samples: MAX_SAMPLES,
        });
    }
    if let Some((index, &value)) = pcm
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || v.abs() > 1.0)
    {
        return Err(Qwen3AsrError::InvalidSample { index, value });
    }
    if pcm.len() < MIN_SAMPLES {
        let mut padded = vec![0.0_f32; MIN_SAMPLES];
        padded[..pcm.len()].copy_from_slice(pcm);
        Ok(std::borrow::Cow::Owned(padded))
    } else {
        Ok(std::borrow::Cow::Borrowed(pcm))
    }
}

/// Raw and normalized log-mel features of one clip.
#[derive(Debug, Clone)]
pub struct Qwen3AsrMel {
    num_mel_bins: usize,
    frames: usize,
    /// `log10(max(mel, 1e-10))` before the global clamp, row-major `[bins, frames]`.
    raw: Vec<f32>,
    max_raw: f32,
}

impl Qwen3AsrMel {
    /// Number of mel bins.
    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }

    /// Number of mel frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Row-major `[bins, frames]` `log10` mel values before clamping and scaling.
    pub fn raw_log_mel(&self) -> &[f32] {
        &self.raw
    }

    /// Maximum raw log-mel value over every bin and frame.
    pub fn max_raw(&self) -> f32 {
        self.max_raw
    }

    /// Clamp floor `max_raw - 8.0`, the value a later chunk of audio can move.
    pub fn floor(&self) -> f32 {
        self.max_raw - LOG_FLOOR_RANGE
    }

    /// Minimum raw log-mel value over bins and frames `start..end`.
    ///
    /// Returns `None` for an empty or out-of-range frame span. A block whose
    /// minimum is at least the floor is unaffected by the clamp.
    pub fn min_raw_in_frames(&self, start: usize, end: usize) -> Option<f32> {
        if start >= end || end > self.frames {
            return None;
        }
        let mut min = f32::INFINITY;
        for bin in 0..self.num_mel_bins {
            let row = &self.raw[bin * self.frames..(bin + 1) * self.frames];
            for &v in &row[start..end] {
                min = min.min(v);
            }
        }
        Some(min)
    }

    /// Normalized features `(max(raw, floor) + 4) / 4`, row-major `[bins, frames]`.
    pub fn normalized(&self) -> Vec<f32> {
        let floor = self.floor();
        self.raw
            .iter()
            .map(|&v| (v.max(floor) + 4.0) / 4.0)
            .collect()
    }

    /// Normalized features as an `F32` tensor of shape `[bins, frames]`.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor, Qwen3AsrError> {
        Ok(Tensor::from_vec(
            self.normalized(),
            (self.num_mel_bins, self.frames),
            device,
        )?)
    }
}

/// Reusable mel frontend holding the filter bank, window, and FFT twiddles.
#[derive(Debug, Clone)]
pub struct Qwen3AsrMelFrontend {
    filters: MelFilterBank,
    window: Vec<f32>,
    fft: MixedRadixFft,
}

impl Qwen3AsrMelFrontend {
    /// Build the frontend for `num_mel_bins` (`80` or `128`) with a Slaney filter bank.
    pub fn new(num_mel_bins: usize) -> Result<Self, Qwen3AsrError> {
        let filters = MelFilterBank::slaney(num_mel_bins)
            .map_err(|e| Qwen3AsrError::InvalidConfig(e.to_string()))?;
        let window = (0..N_FFT)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();
        Ok(Self {
            filters,
            window,
            fft: MixedRadixFft::new(N_FFT),
        })
    }

    /// Number of mel bins.
    pub fn num_mel_bins(&self) -> usize {
        self.filters.num_mel_bins()
    }

    /// Compute log-mel features for `pcm` after validation and short-clip padding.
    pub fn compute(&self, pcm: &[f32]) -> Result<Qwen3AsrMel, Qwen3AsrError> {
        let pcm = prepare_pcm(pcm)?;
        let samples = pcm.len();
        let frames = frame_count(samples);
        let bins = self.num_mel_bins();
        let pad = N_FFT / 2;

        // Reflect padding without edge repeat, as `torch.stft(center=True, pad_mode="reflect")`.
        let mut padded = Vec::with_capacity(samples + 2 * pad);
        padded.extend((1..=pad).rev().map(|i| pcm[i]));
        padded.extend_from_slice(&pcm);
        padded.extend((1..=pad).map(|i| pcm[samples - 1 - i]));

        let mut raw = vec![0.0_f32; bins * frames];
        let mut buf = vec![(0.0_f64, 0.0_f64); N_FFT];
        let mut power = vec![0.0_f32; WHISPER_N_FFT_BINS];
        let filters = self.filters.as_slice();
        for frame in 0..frames {
            let seg = &padded[frame * HOP_LENGTH..frame * HOP_LENGTH + N_FFT];
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = ((seg[i] * self.window[i]) as f64, 0.0);
            }
            let spectrum = self.fft.transform(&buf);
            for (k, p) in power.iter_mut().enumerate() {
                let (re, im) = spectrum[k];
                *p = (re * re + im * im) as f32;
            }
            for bin in 0..bins {
                let row = &filters[bin * WHISPER_N_FFT_BINS..(bin + 1) * WHISPER_N_FFT_BINS];
                let mel: f64 = row
                    .iter()
                    .zip(&power)
                    .map(|(&w, &p)| w as f64 * p as f64)
                    .sum();
                raw[bin * frames + frame] = (mel as f32).max(MEL_POWER_FLOOR).log10();
            }
        }
        let max_raw = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Ok(Qwen3AsrMel {
            num_mel_bins: bins,
            frames,
            raw,
            max_raw,
        })
    }
}

/// Recursive mixed-radix complex FFT for arbitrary composite sizes (`400 = 2^4 * 5^2`).
#[derive(Debug, Clone)]
struct MixedRadixFft {
    n: usize,
    twiddle: Vec<(f64, f64)>,
}

impl MixedRadixFft {
    fn new(n: usize) -> Self {
        let twiddle = (0..n)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (angle.cos(), angle.sin())
            })
            .collect();
        Self { n, twiddle }
    }

    fn transform(&self, input: &[(f64, f64)]) -> Vec<(f64, f64)> {
        debug_assert_eq!(input.len(), self.n);
        let mut out = vec![(0.0, 0.0); self.n];
        self.recurse(input, 1, self.n, &mut out);
        out
    }

    fn recurse(&self, x: &[(f64, f64)], stride: usize, n: usize, out: &mut [(f64, f64)]) {
        if n == 1 {
            out[0] = x[0];
            return;
        }
        let p = (2..=n).find(|f| n.is_multiple_of(*f)).unwrap_or(n);
        let m = n / p;
        for r in 0..p {
            self.recurse(
                &x[r * stride..],
                stride * p,
                m,
                &mut out[r * m..(r + 1) * m],
            );
        }
        // Twiddle for an index `a` at sub-size `n` is `W_N^(a * N / n)`.
        let scale = self.n / n;
        let mut tmp = vec![(0.0, 0.0); p];
        for k in 0..m {
            for (r, t) in tmp.iter_mut().enumerate() {
                let (wr, wi) = self.twiddle[(r * k * scale) % self.n];
                let (yr, yi) = out[r * m + k];
                *t = (yr * wr - yi * wi, yr * wi + yi * wr);
            }
            for q in 0..p {
                let (mut sr, mut si) = (0.0, 0.0);
                for (r, &(tr, ti)) in tmp.iter().enumerate() {
                    let (wr, wi) = self.twiddle[(r * q * m * scale) % self.n];
                    sr += tr * wr - ti * wi;
                    si += tr * wi + ti * wr;
                }
                out[k + q * m] = (sr, si);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(len: usize, hz: f32, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / 16_000.0).sin())
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() {
        let fft = MixedRadixFft::new(N_FFT);
        let x: Vec<(f64, f64)> = (0..N_FFT)
            .map(|i| (((i * 7 + 3) % 11) as f64 - 5.0, ((i * 5) % 13) as f64 - 6.0))
            .collect();
        let y = fft.transform(&x);
        for k in [0usize, 1, 17, 199, 200, 399] {
            let (mut re, mut im) = (0.0, 0.0);
            for (n, &(xr, xi)) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / N_FFT as f64;
                re += xr * a.cos() - xi * a.sin();
                im += xr * a.sin() + xi * a.cos();
            }
            assert!(
                (re - y[k].0).abs() < 1e-9 && (im - y[k].1).abs() < 1e-9,
                "bin {k}"
            );
        }
    }

    #[test]
    fn frame_count_is_samples_over_hop_without_padding() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        for samples in [8_000usize, 8_159, 8_160, 16_000, 128_039, 128_040] {
            let mel = fe.compute(&sine(samples, 440.0, 0.5)).unwrap();
            assert_eq!(mel.frames(), samples / 160, "samples {samples}");
            assert_eq!(mel.raw_log_mel().len(), 128 * mel.frames());
        }
        assert_eq!(frame_count(240_820), 1505);
    }

    #[test]
    fn short_clip_is_zero_padded_at_tail_to_half_second() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        let short = sine(3_000, 300.0, 0.4);
        let mut manual = short.clone();
        manual.resize(MIN_SAMPLES, 0.0);
        let a = fe.compute(&short).unwrap();
        let b = fe.compute(&manual).unwrap();
        assert_eq!(a.frames(), 50);
        assert_eq!(a.raw_log_mel(), b.raw_log_mel());
        let one = fe.compute(&[0.25]).unwrap();
        assert_eq!(one.frames(), 50);
    }

    #[test]
    fn invalid_pcm_is_rejected() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        assert!(matches!(fe.compute(&[]), Err(Qwen3AsrError::EmptyAudio)));
        let mut pcm = sine(9_000, 200.0, 0.3);
        pcm[123] = f32::NAN;
        assert!(matches!(
            fe.compute(&pcm),
            Err(Qwen3AsrError::InvalidSample { index: 123, .. })
        ));
        pcm[123] = f32::INFINITY;
        assert!(matches!(
            fe.compute(&pcm),
            Err(Qwen3AsrError::InvalidSample { .. })
        ));
        pcm[123] = 1.0001;
        assert!(matches!(
            fe.compute(&pcm),
            Err(Qwen3AsrError::InvalidSample { index: 123, .. })
        ));
        pcm[123] = -1.0;
        assert!(fe.compute(&pcm).is_ok());
        let too_long = vec![0.0_f32; MAX_SAMPLES + 1];
        assert!(matches!(
            prepare_pcm(&too_long),
            Err(Qwen3AsrError::AudioTooLong { .. })
        ));
    }

    #[test]
    fn global_floor_clamps_and_normalizes() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        // Loud tone for the first second, near-silence afterwards.
        let mut pcm = sine(32_000, 1_000.0, 0.9);
        for v in &mut pcm[16_000..] {
            *v *= 1e-6;
        }
        let mel = fe.compute(&pcm).unwrap();
        let floor = mel.floor();
        assert_eq!(floor, mel.max_raw() - 8.0);
        let norm = mel.normalized();
        let lo = (floor + 4.0) / 4.0;
        let hi = (mel.max_raw() + 4.0) / 4.0;
        assert!(norm.iter().all(|&v| v >= lo - 1e-6 && v <= hi + 1e-6));
        assert!(
            mel.raw_log_mel().iter().any(|&v| v < floor),
            "silence must fall below the floor"
        );
        assert!(norm.contains(&lo), "clamped values equal the floor");
        assert!(norm.iter().any(|&v| (v - hi).abs() < 1e-6));
    }

    #[test]
    fn later_loud_audio_changes_earlier_normalized_frames() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        let quiet = sine(16_000, 500.0, 0.05);
        let mut loud = quiet.clone();
        loud.extend(sine(16_000, 1_500.0, 0.95));
        let a = fe.compute(&quiet).unwrap();
        let b = fe.compute(&loud).unwrap();
        assert!(b.floor() > a.floor());
        let (na, nb) = (a.normalized(), b.normalized());
        // First 50 frames are final in both passes (frame i needs 160 * i + 200 samples).
        let frames_a = a.frames();
        let changed =
            (0..128).any(|bin| (0..50).any(|f| na[bin * frames_a + f] != nb[bin * b.frames() + f]));
        assert!(changed, "a later maximum must move earlier clamped values");
        // The raw values of final frames do not depend on later audio.
        for bin in 0..128 {
            for f in 0..50 {
                assert_eq!(
                    a.raw_log_mel()[bin * frames_a + f],
                    b.raw_log_mel()[bin * b.frames() + f]
                );
            }
        }
    }

    #[test]
    fn frame_boundary_is_final_at_160_i_plus_200_samples() {
        assert_eq!(samples_to_finalize_frame(0), 200);
        assert_eq!(samples_to_finalize_frame(799), 160 * 799 + 200);
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        let full = sine(40_000, 700.0, 0.6);
        let reference = fe.compute(&full).unwrap();
        let frame = 120usize;
        let need = samples_to_finalize_frame(frame);
        let at_need = fe.compute(&full[..need]).unwrap();
        let before = fe.compute(&full[..need - 1]).unwrap();
        let get = |m: &Qwen3AsrMel, bin: usize| m.raw_log_mel()[bin * m.frames() + frame];
        // At `need` samples the frame equals the full-clip value (up to the near-zero last Hann tap).
        for bin in 0..128 {
            assert!(
                (get(&at_need, bin) - get(&reference, bin)).abs() < 1e-4,
                "bin {bin}"
            );
        }
        // One sample earlier the frame is still reflect padded and differs somewhere.
        assert!((0..128).any(|bin| (get(&before, bin) - get(&reference, bin)).abs() > 1e-6));
    }

    #[test]
    fn tensor_has_bins_by_frames_shape() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        let mel = fe.compute(&sine(20_000, 800.0, 0.5)).unwrap();
        let t = mel.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[128, 125]);
    }

    #[test]
    fn min_raw_in_frames_covers_span_only() {
        let fe = Qwen3AsrMelFrontend::new(128).unwrap();
        let mel = fe.compute(&sine(20_000, 800.0, 0.5)).unwrap();
        assert!(mel.min_raw_in_frames(0, 0).is_none());
        assert!(mel.min_raw_in_frames(0, mel.frames() + 1).is_none());
        let all = mel.min_raw_in_frames(0, mel.frames()).unwrap();
        let part = mel.min_raw_in_frames(10, 20).unwrap();
        assert!(part >= all);
    }
}
