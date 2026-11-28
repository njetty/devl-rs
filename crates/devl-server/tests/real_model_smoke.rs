//! Opt-in real-model smoke tests through both server transports.
//!
//! Each test loads a local checkpoint once, computes the final tokens of a
//! direct `DevlSession` over a clip, then streams the identical PCM through a
//! loopback `HttpServer` and `WsServer` sharing one `Coordinator` over that
//! same backend with the same session settings, and requires the final token
//! ids to be identical.
//!
//! The wire protocol carries text, not token ids. The ids are observed at the
//! coordinator's detokenizer, which receives exactly the verified final ids
//! right before the `final` event is emitted. The test records every call and
//! requires that the only id sequence producing the final text is the direct
//! session's, so a stale partial cannot be mistaken for the final.
//!
//! Run with (see README "Real-model smoke checks"):
//!
//! ```text
//! DEVL_SMOKE_AUDIO=clip.wav DEVL_SMOKE_WHISPER_DIR=whisper-tiny.en \
//!   cargo test --release -p devl-server --features candle --test real_model_smoke \
//!   whisper -- --ignored --nocapture
//! ```
//!
//! `DEVL_SMOKE_QWEN3_ASR_DIR` selects the pinned Qwen3-ASR snapshot for the
//! `qwen3_asr` test. `DEVL_SMOKE_WHISPER_TOKENIZER` overrides the default
//! `<whisper dir>/tokenizer.json`. Weights and clips are never committed.

#[path = "../examples/support/client.rs"]
mod client;
#[path = "../src/bin/devl-server/models.rs"]
mod models;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use client::{Replay, Transport};
use devl_core::{DevlBackend, DevlSession, SessionConfig, TokenId};
use devl_server::{Coordinator, HttpConfig, HttpServer, WsConfig, WsServer};

type Calls = Arc<Mutex<Vec<(Vec<TokenId>, String)>>>;

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must be set for this opt-in test"))
}

fn run_smoke<B>(
    label: &str,
    backend: B,
    detokenize: models::Detokenize,
    session: SessionConfig,
    max_stream_samples: u64,
    samples: &[i16],
) where
    B: DevlBackend + Clone + Send + Sync + 'static,
    B::AudioEncoding: Send,
{
    let pcm: Vec<f32> = samples.iter().map(|s| f32::from(*s) / 32768.0).collect();

    // Direct reference: the whole clip, then commit.
    let started = Instant::now();
    let mut direct = DevlSession::new(backend.clone(), session).unwrap();
    direct.push_audio(&pcm).unwrap();
    let direct_ids = direct.commit().unwrap().tokens.clone();
    let direct_secs = started.elapsed().as_secs_f64();
    drop(direct);
    let direct_text = detokenize(&direct_ids);
    eprintln!(
        "[{label}] direct session: {} ids in {direct_secs:.1} s",
        direct_ids.len()
    );
    eprintln!("[{label}] direct ids: {direct_ids:?}");
    eprintln!("[{label}] direct text: {direct_text:?}");
    assert!(!direct_ids.is_empty());
    assert!(!direct_text.is_empty(), "empty transcript, wrong clip?");

    // Coordinator over the same backend, recording every detokenizer call.
    let calls: Calls = Arc::default();
    let recorder = {
        let calls = Arc::clone(&calls);
        let detokenize = Arc::clone(&detokenize);
        move |ids: &[TokenId]| {
            let text = detokenize(ids);
            calls.lock().unwrap().push((ids.to_vec(), text.clone()));
            text
        }
    };
    let config = models::coordinator_config(session, Some(max_stream_samples), 2, 8, 2_000);
    let coordinator = Arc::new(Coordinator::new(backend, recorder, config).unwrap());
    let http = HttpServer::bind(
        "127.0.0.1:0",
        Arc::clone(&coordinator),
        HttpConfig::default(),
    )
    .unwrap();
    let ws = WsServer::bind("127.0.0.1:0", Arc::clone(&coordinator), WsConfig::default()).unwrap();

    for (transport, addr) in [
        (Transport::Http, http.local_addr().to_string()),
        (Transport::Ws, ws.local_addr().to_string()),
    ] {
        calls.lock().unwrap().clear();
        let started = Instant::now();
        // Real-time pacing lets checkpoint drafts finish before the commit, so the
        // final is verified against a real draft instead of decoded from scratch.
        let replay = Replay {
            realtime: true,
            ..Replay::default()
        };
        let outcome = client::run(transport, &addr, samples, &replay).unwrap();
        let secs = started.elapsed().as_secs_f64();
        assert_eq!(outcome.terminal_count(), 1, "{:?}", outcome.events);
        let fin = outcome.terminal().unwrap();
        assert_eq!(fin["type"], "final", "{fin}");
        assert_eq!(fin["audio_samples"], samples.len(), "{fin}");
        let text = fin["text"].as_str().unwrap();
        assert_eq!(
            text, direct_text,
            "{transport:?} text differs from direct session"
        );

        let observed: Vec<Vec<TokenId>> = {
            let mut ids: Vec<_> = calls
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, t)| t == text)
                .map(|(ids, _)| ids.clone())
                .collect();
            ids.dedup();
            ids
        };
        assert_eq!(
            observed,
            vec![direct_ids.clone()],
            "{transport:?}: final ids differ from the direct session"
        );
        eprintln!(
            "[{label}] {transport:?}: final ids identical to direct ({} ids), {} partials, \
             {} detokenizer calls, {secs:.1} s end to end",
            direct_ids.len(),
            outcome.partials().len(),
            calls.lock().unwrap().len(),
        );
        eprintln!("[{label}] {transport:?}: final text: {text:?}");
    }
    http.shutdown();
    ws.shutdown();
    coordinator.shutdown();
}

#[test]
#[ignore = "needs DEVL_SMOKE_AUDIO and DEVL_SMOKE_WHISPER_DIR (local Whisper checkpoint)"]
fn whisper_final_tokens_match_direct_session_over_http_and_websocket() {
    let samples = client::read_pcm_s16(&env_path("DEVL_SMOKE_AUDIO")).unwrap();
    assert!(
        (1..=models::WHISPER_MAX_STREAM_SAMPLES as usize).contains(&samples.len()),
        "Whisper accepts 1 to {} samples, got {}",
        models::WHISPER_MAX_STREAM_SAMPLES,
        samples.len()
    );
    let dir = env_path("DEVL_SMOKE_WHISPER_DIR");
    let tokenizer = std::env::var_os("DEVL_SMOKE_WHISPER_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("tokenizer.json"));
    let started = Instant::now();
    let (backend, detok) = models::load_whisper(&dir, &tokenizer).unwrap();
    eprintln!(
        "[whisper] loaded in {:.1} s, f32, CPU",
        started.elapsed().as_secs_f64()
    );
    run_smoke(
        "whisper",
        backend,
        detok,
        models::WHISPER_SESSION,
        models::WHISPER_MAX_STREAM_SAMPLES,
        &samples,
    );
}

#[test]
#[ignore = "needs DEVL_SMOKE_AUDIO and DEVL_SMOKE_QWEN3_ASR_DIR (pinned Qwen3-ASR snapshot)"]
fn qwen3_asr_final_tokens_match_direct_session_over_http_and_websocket() {
    let samples = client::read_pcm_s16(&env_path("DEVL_SMOKE_AUDIO")).unwrap();
    let started = Instant::now();
    let (backend, detok) = models::load_qwen3_asr(&env_path("DEVL_SMOKE_QWEN3_ASR_DIR")).unwrap();
    eprintln!(
        "[qwen3-asr] loaded in {:.1} s, f32, CPU",
        started.elapsed().as_secs_f64()
    );
    let max = backend.config().max_audio_samples as u64;
    assert!(
        samples.len() as u64 <= max,
        "clip longer than the {max} sample limit"
    );
    run_smoke(
        "qwen3-asr",
        backend,
        detok,
        models::QWEN3_ASR_SESSION,
        max,
        &samples,
    );
}
