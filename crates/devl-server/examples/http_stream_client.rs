//! Incremental HTTP client for the `devl-server` streaming endpoint.
//!
//! Streams 16 kHz mono signed 16-bit little-endian PCM to `POST /v1/stream`
//! one frame at a time while printing every event line as it arrives, so
//! partial transcripts appear before the final.
//!
//! ```text
//! cargo run -p devl-server --example http_stream_client -- 127.0.0.1:8080 [audio.s16le]
//! ```
//!
//! With no file, three seconds of silence are sent. Frames are paced at
//! roughly real time to show events interleaving with the upload.

use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use devl_server::http::{REQUEST_CONTENT_TYPE, STREAM_PATH, encode_message_record};
use devl_server::{AudioFormat, AudioFrame, ClientMessage, PROTOCOL_VERSION};
use hyper::Request;
use hyper::body::{Body, Bytes, Frame};
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use tokio::sync::mpsc;

const FRAME_SAMPLES: usize = 8_000;

struct UploadBody(mpsc::Receiver<Bytes>);

impl Body for UploadBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.0.poll_recv(cx).map(|c| c.map(|b| Ok(Frame::data(b))))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    let samples: Vec<i16> = match args.next() {
        Some(path) => std::fs::read(path)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect(),
        None => vec![0; 3 * 16_000],
    };

    let tcp = tokio::net::TcpStream::connect(&addr).await?;
    let (mut sender, conn) = http1::handshake(TokioIo::new(tcp)).await?;
    tokio::spawn(conn);

    let (tx, rx) = mpsc::channel::<Bytes>(16);
    let request = Request::post(STREAM_PATH)
        .header("host", addr.as_str())
        .header("content-type", REQUEST_CONTENT_TYPE)
        .body(UploadBody(rx))?;

    // Upload concurrently with reading the response.
    let upload = tokio::spawn(async move {
        let send = |msg: ClientMessage| {
            let tx = tx.clone();
            async move {
                tx.send(Bytes::from(encode_message_record(&msg)))
                    .await
                    .is_ok()
            }
        };
        if !send(ClientMessage::Start {
            version: PROTOCOL_VERSION,
            format: AudioFormat::mono_16k_s16(),
        })
        .await
        {
            return;
        }
        let mut frames = 0u64;
        for chunk in samples.chunks(FRAME_SAMPLES) {
            let msg = ClientMessage::Audio(AudioFrame::from_s16(frames, chunk));
            if !send(msg).await {
                return;
            }
            frames += 1;
            let ms = u64::try_from(chunk.len()).unwrap_or(0) / 16;
            tokio::time::sleep(Duration::from_millis(ms)).await;
        }
        let _ = send(ClientMessage::Commit { frames }).await;
        // Dropping the last sender ends the request body.
    });

    let response = sender.send_request(request).await?;
    eprintln!("status: {}", response.status());
    let mut body = response.into_body();
    let mut pending = Vec::new();
    while let Some(frame) = std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await {
        let Ok(data) = frame?.into_data() else {
            continue;
        };
        pending.extend_from_slice(&data);
        while let Some(nl) = pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = pending.drain(..=nl).collect();
            print!("{}", String::from_utf8_lossy(&line));
        }
    }
    let _ = upload.await;
    Ok(())
}
