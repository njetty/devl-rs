//! Forkable and truncatable KV cache for the Qwen3-ASR text decoder.
//!
//! Keys are stored after `q_norm`/`k_norm` and rotary embedding, at the absolute position of
//! each token, and values as projected. The cache length is therefore the only position state:
//! the next token sits at absolute position [`Qwen3AsrKvCache::len`], and truncating to `n`
//! entries restores exactly the state after `n` tokens because the decoder is causal and no
//! cached entry depends on a later token (`docs/ARCHITECTURE.md` Sections 6.6 and 6.9).
//!
//! Tensors are reference counted and never mutated in place. Appending concatenates into new
//! tensors and truncation narrows a view, so [`Qwen3AsrKvCache::fork`] is a cheap clone and the
//! forks cannot observe each other's appends or truncations.

use candle_core::Tensor;

use super::Qwen3AsrError;

/// Cached `[1, kv_heads, len, head_dim]` key and value tensors of one layer.
#[derive(Debug, Clone)]
pub(crate) struct LayerKv {
    pub(crate) k: Tensor,
    pub(crate) v: Tensor,
}

/// Per-layer key/value cache of one decoding pass.
///
/// Create with [`crate::qwen3_asr::Qwen3AsrDecoder::new_cache`]. Cloning and
/// [`fork`](Self::fork) are equivalent.
#[derive(Debug, Clone)]
pub struct Qwen3AsrKvCache {
    layers: Vec<Option<LayerKv>>,
    len: usize,
}

impl Qwen3AsrKvCache {
    /// Empty cache for a decoder with `num_layers` layers.
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            len: 0,
        }
    }

    /// Number of cached positions, which is also the absolute position of the next token.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no position is cached.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of decoder layers this cache was created for.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Independent copy that shares the existing tensors.
    ///
    /// Later appends or truncations on either cache do not affect the other.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Keep the first `len` positions and drop the rest.
    ///
    /// Returns [`Qwen3AsrError::InvalidCacheLength`] when `len` exceeds the cached length. The
    /// cache is unchanged on error. Truncating to the current length is a no-op. The decoder
    /// does not know where the prompt ends, so callers choose a boundary of at least their
    /// prompt length.
    pub fn truncate(&mut self, len: usize) -> Result<(), Qwen3AsrError> {
        if len > self.len {
            return Err(Qwen3AsrError::InvalidCacheLength {
                requested: len,
                len: self.len,
            });
        }
        if len == self.len {
            return Ok(());
        }
        let mut next = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            next.push(match layer {
                Some(kv) if len > 0 => Some(LayerKv {
                    k: kv.k.narrow(2, 0, len)?,
                    v: kv.v.narrow(2, 0, len)?,
                }),
                _ => None,
            });
        }
        self.layers = next;
        self.len = len;
        Ok(())
    }

    pub(crate) fn layer(&self, index: usize) -> Option<&LayerKv> {
        self.layers.get(index).and_then(Option::as_ref)
    }

    /// Replace every layer with `layers` and advance the length by `added`.
    ///
    /// Called once per successful forward pass so a failure midway leaves the cache untouched.
    pub(crate) fn commit(&mut self, layers: Vec<LayerKv>, added: usize) {
        debug_assert_eq!(layers.len(), self.layers.len());
        self.layers = layers.into_iter().map(Some).collect();
        self.len += added;
    }
}
