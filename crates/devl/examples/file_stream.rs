//! Replay one WAV through a streaming Whisper session.
mod support;
#[path = "support/tokenizer.rs"]
mod tokenizer;

use std::time::Instant;

use devl::DevlSession;
use serde_json::json;
use support::{
    ArgsSpec, CHUNK_SAMPLES, Inputs, Result, SESSION_CONFIG, commit_record, emit, is_checkpoint,
    load_wav,
};

fn main() -> Result<()> {
    let inputs = Inputs::parse(
        "file_stream",
        ArgsSpec {
            min_files: 1,
            max_files: 1,
            allow_tokenizer: true,
            allow_references: false,
        },
    )?;
    let tokenizer = tokenizer::load(inputs.tokenizer.as_deref())?;
    let pcm = load_wav(&inputs.fixtures[0])?;
    let (backend, model) = inputs.load_backend()?;
    let mut session = DevlSession::new(&backend, SESSION_CONFIG)?;
    let chunks = pcm.chunks(CHUNK_SAMPLES);

    for (index, chunk) in chunks.clone().enumerate() {
        session.push_audio(chunk)?;
        if is_checkpoint(index + 1, chunks.len()) {
            let draft = session.run_draft()?;
            let mut event = json!({"event": "draft", "audio_samples": draft.audio_samples, "tokens": draft.tokens});
            if let Some(tokenizer) = &tokenizer {
                event["transcript"] = json!(tokenizer.decode(&draft.tokens)?);
            }
            emit(&event)?;
        }
    }

    let endpoint = Instant::now();
    let commit = session.commit()?;
    let elapsed = endpoint.elapsed();
    let mut event =
        json!({"event": "final", "model": model, "commit": commit_record(commit, elapsed)});
    if let Some(tokenizer) = &tokenizer {
        event["transcript"] = json!(tokenizer.decode(&commit.tokens)?);
    }
    emit(&event)
}
