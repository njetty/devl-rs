//! JSON encoding of control messages for the WebSocket adapter.
//!
//! Control messages travel as WebSocket text frames, one JSON object per
//! frame, discriminated by a `type` field. Audio travels as binary frames in
//! the envelope defined by [`crate::frame`]. Unknown extra fields on client
//! messages are ignored so minor versions can add optional fields.

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{ErrorCode, StreamError};
use crate::message::{AudioFormat, ClientMessage, ProtocolVersion, SampleEncoding, ServerEvent};

#[derive(Deserialize)]
struct WireVersion {
    major: u16,
    minor: u16,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireEncoding {
    S16le,
    F32le,
}

#[derive(Deserialize)]
struct WireFormat {
    encoding: WireEncoding,
    sample_rate_hz: u32,
    channels: u16,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireClient {
    Start {
        version: WireVersion,
        format: WireFormat,
    },
    Commit {
        frames: u64,
    },
    Cancel,
    Close,
}

/// Parse one client control message from a text frame.
///
/// Anything that is not a well-formed message of a known `type` is
/// [`ErrorCode::MalformedFrame`].
pub fn parse_client_message(text: &str) -> Result<ClientMessage, StreamError> {
    let wire: WireClient = serde_json::from_str(text).map_err(|e| {
        StreamError::new(
            ErrorCode::MalformedFrame,
            format!("invalid control message: {e}"),
        )
    })?;
    Ok(match wire {
        WireClient::Start { version, format } => ClientMessage::Start {
            version: ProtocolVersion {
                major: version.major,
                minor: version.minor,
            },
            format: AudioFormat {
                encoding: match format.encoding {
                    WireEncoding::S16le => SampleEncoding::S16Le,
                    WireEncoding::F32le => SampleEncoding::F32Le,
                },
                sample_rate_hz: format.sample_rate_hz,
                channels: format.channels,
            },
        },
        WireClient::Commit { frames } => ClientMessage::Commit { frames },
        WireClient::Cancel => ClientMessage::Cancel,
        WireClient::Close => ClientMessage::Close,
    })
}

/// Encode a client control message as JSON, for clients and tests.
///
/// Returns `None` for [`ClientMessage::Audio`], which has no JSON form and
/// travels in the binary envelope of [`crate::frame`].
pub fn encode_client_message(msg: &ClientMessage) -> Option<String> {
    let value: Value = match msg {
        ClientMessage::Start { version, format } => json!({
            "type": "start",
            "version": { "major": version.major, "minor": version.minor },
            "format": {
                "encoding": match format.encoding {
                    SampleEncoding::S16Le => "s16le",
                    SampleEncoding::F32Le => "f32le",
                },
                "sample_rate_hz": format.sample_rate_hz,
                "channels": format.channels,
            },
        }),
        ClientMessage::Audio(_) => return None,
        ClientMessage::Commit { frames } => json!({ "type": "commit", "frames": frames }),
        ClientMessage::Cancel => json!({ "type": "cancel" }),
        ClientMessage::Close => json!({ "type": "close" }),
    };
    Some(value.to_string())
}

/// Encode a server event as a JSON text frame body.
pub fn encode_server_event(event: &ServerEvent) -> String {
    let value: Value = match event {
        ServerEvent::Started { version, limits } => json!({
            "type": "started",
            "version": { "major": version.major, "minor": version.minor },
            "limits": {
                "max_frame_samples": limits.max_frame_samples,
                "max_stream_samples": limits.max_stream_samples,
                "initial_credit_samples": limits.initial_credit_samples,
                "max_outstanding_credit_samples": limits.max_outstanding_credit_samples,
            },
        }),
        ServerEvent::Credit { samples } => json!({ "type": "credit", "samples": samples }),
        ServerEvent::Partial {
            revision,
            audio_samples,
            text,
        } => json!({
            "type": "partial",
            "revision": revision,
            "audio_samples": audio_samples,
            "text": text,
        }),
        ServerEvent::Final {
            audio_samples,
            text,
        } => json!({ "type": "final", "audio_samples": audio_samples, "text": text }),
        ServerEvent::Cancelled => json!({ "type": "cancelled" }),
        ServerEvent::Error(e) => json!({
            "type": "error",
            "code": e.code.as_str(),
            "message": e.message,
        }),
    };
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::PROTOCOL_VERSION;

    #[test]
    fn parses_start_and_simple_messages() {
        let start = r#"{"type":"start","version":{"major":1,"minor":0},
            "format":{"encoding":"s16le","sample_rate_hz":16000,"channels":1},"extra":1}"#;
        assert_eq!(
            parse_client_message(start).unwrap(),
            ClientMessage::Start {
                version: PROTOCOL_VERSION,
                format: AudioFormat::mono_16k_s16(),
            }
        );
        assert_eq!(
            parse_client_message(r#"{"type":"commit","frames":3}"#).unwrap(),
            ClientMessage::Commit { frames: 3 }
        );
        assert_eq!(
            parse_client_message(r#"{"type":"cancel"}"#).unwrap(),
            ClientMessage::Cancel
        );
        assert_eq!(
            parse_client_message(r#"{"type":"close"}"#).unwrap(),
            ClientMessage::Close
        );
    }

    #[test]
    fn client_messages_round_trip() {
        for msg in [
            ClientMessage::Start {
                version: PROTOCOL_VERSION,
                format: AudioFormat::mono_16k_f32(),
            },
            ClientMessage::Commit { frames: 7 },
            ClientMessage::Cancel,
            ClientMessage::Close,
        ] {
            let text = encode_client_message(&msg).unwrap();
            assert_eq!(parse_client_message(&text).unwrap(), msg);
        }
        assert!(
            encode_client_message(&ClientMessage::Audio(crate::AudioFrame::from_s16(0, &[1])))
                .is_none()
        );
    }

    #[test]
    fn rejects_malformed_messages() {
        for bad in [
            "",
            "not json",
            "[]",
            r#"{"type":"nope"}"#,
            r#"{"type":"commit"}"#,
            r#"{"type":"commit","frames":-1}"#,
            r#"{"type":"start","version":{"major":1,"minor":0},"format":{"encoding":"mp3","sample_rate_hz":16000,"channels":1}}"#,
        ] {
            let e = parse_client_message(bad).unwrap_err();
            assert_eq!(e.code, ErrorCode::MalformedFrame, "{bad}");
        }
    }

    #[test]
    fn encodes_events() {
        let v: Value = serde_json::from_str(&encode_server_event(&ServerEvent::Error(
            StreamError::new(ErrorCode::SequenceViolation, "gap"),
        )))
        .unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["code"], "sequence_violation");
        let v: Value =
            serde_json::from_str(&encode_server_event(&ServerEvent::Credit { samples: 5 }))
                .unwrap();
        assert_eq!(v["samples"], 5);
    }
}
