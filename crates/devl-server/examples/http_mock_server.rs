//! HTTP streaming endpoint over a deterministic mock backend.
//!
//! Serves `POST /v1/stream` (see `docs/SERVER_CONTRACT.md`) on the given
//! address so the HTTP framing can be exercised with `http_stream_client` or
//! `curl`. The backend is `MockBackend`: its tokens are synthetic, so the
//! transcript text is a list of token ids, not speech. Loading a real model
//! is outside this example.
//!
//! ```text
//! cargo run -p devl-server --example http_mock_server -- 127.0.0.1:8080
//! ```
//!
//! The server runs until its standard input closes.

use std::io::Read;
use std::sync::Arc;

use devl_core::mock::{MOCK_EOS_TOKEN, MockBackend};
use devl_core::{SchedulerConfig, TokenId};
use devl_server::{Coordinator, CoordinatorConfig, HttpConfig, HttpServer};

fn detokenize(tokens: &[TokenId]) -> String {
    tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> std::io::Result<()> {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    // Checkpoint transcripts at 1 s and 2 s of 16 kHz audio.
    let backend = MockBackend::with_transcripts([
        (16_000, vec![11, 12, MOCK_EOS_TOKEN]),
        (32_000, vec![11, 12, 13, MOCK_EOS_TOKEN]),
    ]);
    let config = CoordinatorConfig {
        scheduler: SchedulerConfig {
            max_concurrent_drafts: 2,
        },
        draft_cadence_samples: 16_000,
        ..CoordinatorConfig::default()
    };
    let coordinator = Arc::new(
        Coordinator::new(backend, detokenize, config)
            .map_err(|e| std::io::Error::other(e.to_string()))?,
    );
    let server = HttpServer::bind(&addr, Arc::clone(&coordinator), HttpConfig::default())?;
    println!("listening on http://{}/v1/stream", server.local_addr());
    println!("close stdin to stop");
    let mut sink = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut sink);
    server.shutdown();
    coordinator.shutdown();
    Ok(())
}
