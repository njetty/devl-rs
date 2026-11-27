//! Contract validation tests: framing, ordering, limits, cancellation.

use devl_server::*;

fn limits() -> StreamLimits {
    StreamLimits {
        max_frame_samples: 8,
        max_stream_samples: 20,
        initial_credit_samples: 16,
        max_outstanding_credit_samples: 16,
    }
}

fn start_msg() -> ClientMessage {
    ClientMessage::Start {
        version: PROTOCOL_VERSION,
        format: AudioFormat::mono_16k_s16(),
    }
}

fn audio(seq: u64, n: usize) -> ClientMessage {
    ClientMessage::Audio(AudioFrame::from_s16(seq, &vec![100i16; n]))
}

fn started() -> ServerEvent {
    ServerEvent::Started {
        version: PROTOCOL_VERSION,
        limits: limits(),
    }
}

/// Contract in `Streaming` with `Started` emitted.
fn streaming() -> StreamContract {
    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    c.receive(&start_msg()).unwrap();
    c.emit(&started()).unwrap();
    c
}

fn code<T: std::fmt::Debug>(r: Result<T, StreamError>) -> ErrorCode {
    r.unwrap_err().code
}

#[test]
fn happy_path_to_final() {
    let mut c = streaming();
    let Accepted::Audio { seq, samples } = c.receive(&audio(0, 4)).unwrap() else {
        panic!("expected audio");
    };
    assert_eq!((seq, samples.len()), (0, 4));
    assert_eq!(samples[0], 100.0 / 32768.0);
    c.emit(&ServerEvent::Partial {
        revision: 1,
        audio_samples: 4,
        text: "a".into(),
    })
    .unwrap();
    c.receive(&audio(1, 4)).unwrap();
    assert_eq!(
        c.receive(&ClientMessage::Commit { frames: 2 }),
        Ok(Accepted::Commit)
    );
    assert_eq!(c.phase(), StreamPhase::Committing);
    c.emit(&ServerEvent::Final {
        audio_samples: 8,
        text: "ab".into(),
    })
    .unwrap();
    assert_eq!(c.phase(), StreamPhase::Finalized);
    assert_eq!(c.receive(&ClientMessage::Close), Ok(Accepted::Close));
}

#[test]
fn empty_commit_is_legal() {
    let mut c = streaming();
    assert_eq!(
        c.receive(&ClientMessage::Commit { frames: 0 }),
        Ok(Accepted::Commit)
    );
    c.emit(&ServerEvent::Final {
        audio_samples: 0,
        text: String::new(),
    })
    .unwrap();
}

#[test]
fn frame_envelope_round_trips() {
    let f = AudioFrame::from_s16(7, &[1, -2, 3]);
    let bytes = encode_audio_frame(&f);
    assert_eq!(bytes.len(), FRAME_HEADER_LEN + 6);
    assert_eq!(decode_audio_frame(&bytes).unwrap(), f);
}

#[test]
fn malformed_envelopes() {
    assert_eq!(code(decode_audio_frame(&[])), ErrorCode::MalformedFrame);
    assert_eq!(
        code(decode_audio_frame(&[1, 1, 0, 0])),
        ErrorCode::MalformedFrame
    );
    let mut bytes = encode_audio_frame(&AudioFrame::from_s16(0, &[1]));
    bytes[0] = 2;
    assert_eq!(
        code(decode_audio_frame(&bytes)),
        ErrorCode::UnsupportedVersion
    );
    bytes[0] = 1;
    bytes[1] = 9;
    assert_eq!(code(decode_audio_frame(&bytes)), ErrorCode::MalformedFrame);
}

#[test]
fn malformed_payloads() {
    let s16 = AudioFormat::mono_16k_s16();
    let f32f = AudioFormat::mono_16k_f32();
    assert_eq!(code(decode_samples(s16, &[], 8)), ErrorCode::MalformedFrame);
    assert_eq!(
        code(decode_samples(s16, &[0, 0, 0], 8)),
        ErrorCode::MalformedFrame
    );
    assert_eq!(
        code(decode_samples(f32f, &[0; 6], 8)),
        ErrorCode::MalformedFrame
    );
    assert_eq!(
        code(decode_samples(s16, &[0; 18], 8)),
        ErrorCode::FrameTooLarge
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let p = AudioFrame::from_f32(0, &[0.0, bad]).payload;
        assert_eq!(code(decode_samples(f32f, &p, 8)), ErrorCode::MalformedFrame);
    }
    let p = AudioFrame::from_f32(0, &[0.5, -1.0]).payload;
    assert_eq!(decode_samples(f32f, &p, 8).unwrap(), vec![0.5, -1.0]);
    let p = AudioFrame::from_s16(0, &[i16::MIN, i16::MAX]).payload;
    assert_eq!(decode_samples(s16, &p, 8).unwrap()[0], -1.0);
}

#[test]
fn malformed_frame_fails_stream() {
    let mut c = streaming();
    let bad = ClientMessage::Audio(AudioFrame {
        seq: 0,
        payload: vec![1, 2, 3],
    });
    assert_eq!(code(c.receive(&bad)), ErrorCode::MalformedFrame);
    assert_eq!(c.phase(), StreamPhase::Failed);
    c.emit(&ServerEvent::Error(StreamError::new(
        ErrorCode::MalformedFrame,
        "x",
    )))
    .unwrap();
    // Exactly one error event.
    assert!(
        c.emit(&ServerEvent::Error(StreamError::new(
            ErrorCode::Internal,
            "y"
        )))
        .is_err()
    );
    assert_eq!(c.receive(&ClientMessage::Close), Ok(Accepted::Close));
}

#[test]
fn invalid_client_order() {
    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    assert_eq!(
        code(c.receive(&audio(0, 1))),
        ErrorCode::InvalidMessageOrder
    );

    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    assert_eq!(
        code(c.receive(&ClientMessage::Commit { frames: 0 })),
        ErrorCode::InvalidMessageOrder
    );

    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    assert_eq!(
        code(c.receive(&ClientMessage::Cancel)),
        ErrorCode::InvalidMessageOrder
    );

    let mut c = streaming();
    assert_eq!(
        code(c.receive(&start_msg())),
        ErrorCode::InvalidMessageOrder
    );

    // Audio before the server emitted Started.
    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    c.receive(&start_msg()).unwrap();
    assert_eq!(
        code(c.receive(&audio(0, 1))),
        ErrorCode::InvalidMessageOrder
    );

    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    c.receive(&start_msg()).unwrap();
    assert_eq!(
        code(c.receive(&ClientMessage::Commit { frames: 0 })),
        ErrorCode::InvalidMessageOrder
    );

    // Audio and a second commit after commit.
    let mut c = streaming();
    c.receive(&ClientMessage::Commit { frames: 0 }).unwrap();
    assert_eq!(
        code(c.receive(&audio(0, 1))),
        ErrorCode::InvalidMessageOrder
    );
    let mut c = streaming();
    c.receive(&ClientMessage::Commit { frames: 0 }).unwrap();
    assert_eq!(
        code(c.receive(&ClientMessage::Commit { frames: 0 })),
        ErrorCode::InvalidMessageOrder
    );
}

#[test]
fn invalid_server_order() {
    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    assert!(c.emit(&started()).is_err()); // before Start
    c.receive(&start_msg()).unwrap();
    assert!(
        c.emit(&ServerEvent::Partial {
            revision: 0,
            audio_samples: 0,
            text: String::new()
        })
        .is_err()
    ); // before Started
    c.emit(&started()).unwrap();
    assert!(c.emit(&started()).is_err()); // twice
    assert!(
        c.emit(&ServerEvent::Final {
            audio_samples: 0,
            text: String::new()
        })
        .is_err()
    ); // before commit
    c.receive(&ClientMessage::Commit { frames: 0 }).unwrap();
    assert!(
        c.emit(&ServerEvent::Partial {
            revision: 1,
            audio_samples: 0,
            text: String::new()
        })
        .is_err()
    ); // partial after commit
    c.emit(&ServerEvent::Final {
        audio_samples: 0,
        text: String::new(),
    })
    .unwrap();
    assert!(
        c.emit(&ServerEvent::Final {
            audio_samples: 0,
            text: String::new()
        })
        .is_err()
    ); // second terminal
    assert!(c.emit(&ServerEvent::Cancelled).is_err());
}

#[test]
fn partial_revisions_and_coverage() {
    let mut c = streaming();
    c.receive(&audio(0, 4)).unwrap();
    let p = |revision, audio_samples| ServerEvent::Partial {
        revision,
        audio_samples,
        text: String::new(),
    };
    c.emit(&p(3, 4)).unwrap();
    assert!(c.emit(&p(3, 4)).is_err());
    assert!(c.emit(&p(2, 4)).is_err());
    assert!(c.emit(&p(9, 5)).is_err());
    c.emit(&p(9, 2)).unwrap(); // gaps allowed
}

#[test]
fn sequencing_violations() {
    let mut c = streaming();
    assert_eq!(code(c.receive(&audio(1, 1))), ErrorCode::SequenceViolation);

    let mut c = streaming();
    c.receive(&audio(0, 1)).unwrap();
    assert_eq!(code(c.receive(&audio(0, 1))), ErrorCode::SequenceViolation);

    let mut c = streaming();
    c.receive(&audio(0, 1)).unwrap();
    assert_eq!(
        code(c.receive(&ClientMessage::Commit { frames: 2 })),
        ErrorCode::SequenceViolation
    );
}

#[test]
fn version_and_format_negotiation() {
    let mk = |version, format| {
        let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
        c.receive(&ClientMessage::Start { version, format })
    };
    let ok = AudioFormat::mono_16k_f32();
    assert!(mk(PROTOCOL_VERSION, ok).is_ok());
    assert_eq!(
        code(mk(ProtocolVersion { major: 2, minor: 0 }, ok)),
        ErrorCode::UnsupportedVersion
    );
    assert_eq!(
        code(mk(ProtocolVersion { major: 1, minor: 1 }, ok)),
        ErrorCode::UnsupportedVersion
    );
    let mut f = ok;
    f.sample_rate_hz = 48_000;
    assert_eq!(code(mk(PROTOCOL_VERSION, f)), ErrorCode::UnsupportedFormat);
    let mut f = ok;
    f.channels = 2;
    assert_eq!(code(mk(PROTOCOL_VERSION, f)), ErrorCode::UnsupportedFormat);
}

#[test]
fn backpressure_and_credit_release() {
    let mut c = streaming();
    c.receive(&audio(0, 8)).unwrap();
    c.receive(&audio(1, 8)).unwrap();
    assert_eq!(c.credit(), 0);
    // Credit is exhausted: next frame is a violation.
    let mut over = streaming();
    over.receive(&audio(0, 8)).unwrap();
    over.receive(&audio(1, 8)).unwrap();
    assert_eq!(
        code(over.receive(&audio(2, 1))),
        ErrorCode::BackpressureViolation
    );
    // Releasing credit lets the stream continue.
    c.emit(&ServerEvent::Credit { samples: 4 }).unwrap();
    c.receive(&audio(2, 4)).unwrap();
    // Grants beyond the outstanding maximum or of zero are server bugs.
    assert!(c.emit(&ServerEvent::Credit { samples: 17 }).is_err());
    assert!(c.emit(&ServerEvent::Credit { samples: 0 }).is_err());
    let mut c = streaming();
    assert!(c.emit(&ServerEvent::Credit { samples: u64::MAX }).is_err());
    assert_eq!(c.credit(), limits().initial_credit_samples);
}

#[test]
fn stream_length_limit() {
    let mut c = streaming();
    c.receive(&audio(0, 8)).unwrap();
    c.receive(&audio(1, 8)).unwrap();
    c.emit(&ServerEvent::Credit { samples: 8 }).unwrap();
    assert_eq!(code(c.receive(&audio(2, 8))), ErrorCode::StreamTooLong);
}

#[test]
fn frame_size_limit_through_state() {
    let mut c = streaming();
    assert_eq!(code(c.receive(&audio(0, 9))), ErrorCode::FrameTooLarge);
}

#[test]
fn cancel_flow() {
    let mut c = streaming();
    c.receive(&audio(0, 4)).unwrap();
    assert_eq!(c.receive(&ClientMessage::Cancel), Ok(Accepted::Cancel));
    // In-flight messages and duplicates are dropped, not errors.
    assert_eq!(c.receive(&audio(1, 4)), Ok(Accepted::Ignored));
    assert_eq!(
        c.receive(&ClientMessage::Commit { frames: 2 }),
        Ok(Accepted::Ignored)
    );
    assert_eq!(c.receive(&ClientMessage::Cancel), Ok(Accepted::Ignored));
    // Partials are suppressed once cancel is pending.
    assert!(
        c.emit(&ServerEvent::Partial {
            revision: 1,
            audio_samples: 4,
            text: String::new()
        })
        .is_err()
    );
    c.emit(&ServerEvent::Cancelled).unwrap();
    assert_eq!(c.phase(), StreamPhase::Cancelled);
    assert_eq!(c.receive(&ClientMessage::Close), Ok(Accepted::Close));
}

#[test]
fn cancel_during_commit_and_after_terminal() {
    let mut c = streaming();
    c.receive(&ClientMessage::Commit { frames: 0 }).unwrap();
    assert_eq!(c.receive(&ClientMessage::Cancel), Ok(Accepted::Cancel));
    assert!(
        c.emit(&ServerEvent::Final {
            audio_samples: 0,
            text: String::new()
        })
        .is_err()
    );
    c.emit(&ServerEvent::Cancelled).unwrap();

    // Cancel that crossed a Final on the wire is a no-op.
    let mut c = streaming();
    c.receive(&ClientMessage::Commit { frames: 0 }).unwrap();
    c.emit(&ServerEvent::Final {
        audio_samples: 0,
        text: String::new(),
    })
    .unwrap();
    assert_eq!(c.receive(&ClientMessage::Cancel), Ok(Accepted::Ignored));
    assert_eq!(c.phase(), StreamPhase::Finalized);
}

#[test]
fn close_before_terminal_abandons_stream() {
    let mut c = streaming();
    assert_eq!(c.receive(&ClientMessage::Close), Ok(Accepted::Cancel));
    c.emit(&ServerEvent::Cancelled).unwrap();

    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    assert_eq!(c.receive(&ClientMessage::Close), Ok(Accepted::Close));
}

#[test]
fn limits_validation() {
    assert!(StreamLimits::default().validate().is_ok());
    let mut l = limits();
    l.max_frame_samples = 0;
    assert!(StreamContract::new(StreamId(1), l).is_err());
    let mut l = limits();
    l.initial_credit_samples = 4; // below one frame
    assert!(l.validate().is_err());
    let mut l = limits();
    l.max_outstanding_credit_samples = 1;
    assert!(l.validate().is_err());
}

#[test]
fn started_must_echo_limits() {
    let mut c = StreamContract::new(StreamId(1), limits()).unwrap();
    c.receive(&start_msg()).unwrap();
    assert!(
        c.emit(&ServerEvent::Started {
            version: ProtocolVersion { major: 2, minor: 0 },
            limits: limits(),
        })
        .is_err()
    );
    assert_eq!(c.credit(), 0);
    let mut other = limits();
    other.max_frame_samples = 4;
    assert!(
        c.emit(&ServerEvent::Started {
            version: PROTOCOL_VERSION,
            limits: other
        })
        .is_err()
    );
}
