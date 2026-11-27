//! Shared Whisper tokenizer loading and decoding. Declared by the examples
//! that accept `--tokenizer` (file_stream and bench_ttfs); multi_stream does
//! not compile this module.
use std::error::Error;
use std::path::Path;

pub(super) struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub(super) fn load(path: &Path) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| format!("{}: {err}", path.display()))?;
        Ok(Self { inner })
    }

    /// Decode token ids into human-readable text, dropping special tokens.
    pub(super) fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn Error + Send + Sync>> {
        let text = self
            .inner
            .decode(tokens, true)
            .map_err(|err| format!("tokenizer decode failed: {err}"))?;
        Ok(text.trim().to_owned())
    }
}

/// Load the tokenizer when `--tokenizer` was given, in the same style as
/// the rest of the shared helpers.
pub(super) fn load(path: Option<&Path>) -> Result<Option<Tokenizer>, Box<dyn Error + Send + Sync>> {
    path.map(Tokenizer::load).transpose()
}
