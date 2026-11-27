//! Binary audio framing.
//!
//! Layout of one audio frame (all integers little-endian):
//!
//! | Offset | Size | Field |
//! | :--- | :--- | :--- |
//! | 0 | 1 | frame layout version (the protocol major, `1`) |
//! | 1 | 1 | frame kind (`1` = audio) |
//! | 2 | 8 | `seq` (`u64`) |
//! | 10 | n | PCM payload in the format declared at `Start` |
//!
//! The envelope carries no length: the transport delivers whole messages
//! (for example one WebSocket binary message). Streaming transports must add
//! their own length prefix.

use crate::error::{ErrorCode, StreamError};
use crate::message::{AudioFormat, AudioFrame, PROTOCOL_VERSION, SampleEncoding};

/// Size of the fixed frame header in bytes.
pub const FRAME_HEADER_LEN: usize = 10;

const KIND_AUDIO: u8 = 1;

/// Serialize an audio frame into its binary envelope.
pub fn encode_audio_frame(frame: &AudioFrame) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + frame.payload.len());
    out.push(PROTOCOL_VERSION.major as u8);
    out.push(KIND_AUDIO);
    out.extend_from_slice(&frame.seq.to_le_bytes());
    out.extend_from_slice(&frame.payload);
    out
}

/// Parse a binary envelope into an [`AudioFrame`].
///
/// Checks the header only; payload shape is checked by [`decode_samples`].
pub fn decode_audio_frame(bytes: &[u8]) -> Result<AudioFrame, StreamError> {
    if bytes.len() < FRAME_HEADER_LEN {
        return Err(StreamError::new(
            ErrorCode::MalformedFrame,
            format!(
                "frame is {} bytes, header needs {FRAME_HEADER_LEN}",
                bytes.len()
            ),
        ));
    }
    if u16::from(bytes[0]) != PROTOCOL_VERSION.major {
        return Err(StreamError::new(
            ErrorCode::UnsupportedVersion,
            format!("frame layout version {}", bytes[0]),
        ));
    }
    if bytes[1] != KIND_AUDIO {
        return Err(StreamError::new(
            ErrorCode::MalformedFrame,
            format!("unknown frame kind {}", bytes[1]),
        ));
    }
    let mut seq = [0u8; 8];
    seq.copy_from_slice(&bytes[2..FRAME_HEADER_LEN]);
    Ok(AudioFrame {
        seq: u64::from_le_bytes(seq),
        payload: bytes[FRAME_HEADER_LEN..].to_vec(),
    })
}

/// Decode a payload into mono `f32` samples for `DevlSession::push_audio`.
///
/// Rejects empty payloads, payloads that are not a whole number of samples,
/// payloads over `max_samples` ([`ErrorCode::FrameTooLarge`]), and non-finite
/// `F32Le` values.
pub fn decode_samples(
    format: AudioFormat,
    payload: &[u8],
    max_samples: u64,
) -> Result<Vec<f32>, StreamError> {
    let width = format.encoding.bytes_per_sample();
    if payload.is_empty() {
        return Err(StreamError::new(
            ErrorCode::MalformedFrame,
            "empty audio payload",
        ));
    }
    if !payload.len().is_multiple_of(width) {
        return Err(StreamError::new(
            ErrorCode::MalformedFrame,
            format!(
                "payload of {} bytes is not a multiple of {width}",
                payload.len()
            ),
        ));
    }
    let count = (payload.len() / width) as u64;
    if count > max_samples {
        return Err(StreamError::new(
            ErrorCode::FrameTooLarge,
            format!("frame has {count} samples, limit is {max_samples}"),
        ));
    }
    match format.encoding {
        SampleEncoding::S16Le => Ok(payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
            .collect()),
        SampleEncoding::F32Le => {
            let mut out = Vec::with_capacity(count as usize);
            for b in payload.as_chunks::<4>().0 {
                let v = f32::from_le_bytes(*b);
                if !v.is_finite() {
                    return Err(StreamError::new(
                        ErrorCode::MalformedFrame,
                        "non-finite f32 sample",
                    ));
                }
                out.push(v);
            }
            Ok(out)
        }
    }
}
