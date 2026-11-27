//! Selective loading of `thinker.*` tensors from safetensors shards.
//!
//! Only the tensor groups a component needs are read (the audio tower, or the text
//! decoder and head), so neither load touches the other and no `unsafe` memory
//! mapping is needed. Tensors are read
//! with plain file I/O using the safetensors header (`u64` length, JSON table,
//! raw little-endian data).

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::Value;

use super::Qwen3AsrError;

/// Prefix of the audio tower in the checkpoint tensor names.
pub const AUDIO_TOWER_PREFIX: &str = "thinker.audio_tower.";

fn parse_dtype(name: &str) -> Result<DType, Qwen3AsrError> {
    Ok(match name {
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        other => {
            return Err(Qwen3AsrError::InvalidCheckpoint(format!(
                "unsupported tensor dtype {other}"
            )));
        }
    })
}

/// Prefix of the text decoder in the checkpoint tensor names.
pub const TEXT_DECODER_PREFIX: &str = "thinker.model.";

/// Prefix of the language-model head in the checkpoint tensor names.
pub const LM_HEAD_PREFIX: &str = "thinker.lm_head.";

/// Pinned shard file names and byte sizes for `Qwen/Qwen3-ASR-1.7B` at the revision in
/// `docs/ARCHITECTURE.md` Section 6.1.
pub const PINNED_SHARDS: [(&str, u64); 2] = [
    ("model-00001-of-00002.safetensors", 4_220_320_824),
    ("model-00002-of-00002.safetensors", 478_200_688),
];

/// Check that both pinned shards exist in `checkpoint_dir` with the pinned byte sizes.
///
/// This is a cheap guard against a different revision or a truncated download. It is not a
/// content hash. The sha256 values in Section 6.1 remain the authoritative check and are
/// verified with `sha256sum` outside the process.
pub fn verify_pinned_shard_sizes(checkpoint_dir: &Path) -> Result<(), Qwen3AsrError> {
    for (name, expected) in PINNED_SHARDS {
        let path = checkpoint_dir.join(name);
        let size = std::fs::metadata(&path)
            .map_err(|e| {
                Qwen3AsrError::InvalidCheckpoint(format!("pinned shard {}: {e}", path.display()))
            })?
            .len();
        if size != expected {
            return Err(Qwen3AsrError::InvalidCheckpoint(format!(
                "shard {name} is {size} bytes, expected {expected} for the pinned revision"
            )));
        }
    }
    Ok(())
}

/// Read every tensor whose name starts with `prefix` from one safetensors file.
///
/// Tensors are returned in their stored dtype on `device`.
pub fn read_prefixed_tensors(
    path: &Path,
    prefix: &str,
    device: &Device,
) -> Result<HashMap<String, Tensor>, Qwen3AsrError> {
    read_tensors_where(path, |name| name.starts_with(prefix), device)
}

/// Read every tensor whose name satisfies `select` from one safetensors file.
fn read_tensors_where(
    path: &Path,
    select: impl Fn(&str) -> bool,
    device: &Device,
) -> Result<HashMap<String, Tensor>, Qwen3AsrError> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > 100 * 1024 * 1024 || 8 + header_len > file_len {
        return Err(Qwen3AsrError::InvalidCheckpoint(format!(
            "bad safetensors header length {header_len} in {}",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header).map_err(|e| {
        Qwen3AsrError::InvalidCheckpoint(format!("bad safetensors header JSON: {e}"))
    })?;
    let table = header.as_object().ok_or_else(|| {
        Qwen3AsrError::InvalidCheckpoint("safetensors header is not an object".into())
    })?;
    let data_start = 8 + header_len;

    let mut out = HashMap::new();
    for (name, meta) in table {
        if !select(name) {
            continue;
        }
        let bad = |what: &str| Qwen3AsrError::InvalidCheckpoint(format!("tensor {name}: {what}"));
        let dtype = parse_dtype(
            meta.get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("no dtype"))?,
        )?;
        let shape: Vec<usize> = meta
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("no shape"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| bad("bad shape"))
            })
            .collect::<Result<_, _>>()?;
        let offsets = meta
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| bad("no data_offsets"))?;
        let begin = offsets[0].as_u64().ok_or_else(|| bad("bad offset"))?;
        let end = offsets[1].as_u64().ok_or_else(|| bad("bad offset"))?;
        let bytes = shape
            .iter()
            .try_fold(dtype.size_in_bytes(), |bytes, &dim| bytes.checked_mul(dim))
            .ok_or_else(|| bad("tensor byte length overflows"))?;
        let length = end
            .checked_sub(begin)
            .and_then(|length| usize::try_from(length).ok());
        if end > file_len - data_start || length != Some(bytes) {
            return Err(bad("data_offsets inconsistent with shape and file size"));
        }
        let mut buf = vec![0u8; bytes];
        file.seek(SeekFrom::Start(data_start + begin))?;
        file.read_exact(&mut buf)?;
        out.insert(
            name.clone(),
            Tensor::from_raw_buffer(&buf, dtype, &shape, device)?,
        );
    }
    Ok(out)
}

/// Load every tensor under any of `prefixes` into a map, using the shard index to choose files.
///
/// Every indexed tensor matching a prefix must be present in its shard.
fn read_indexed_tensors(
    checkpoint_dir: &Path,
    prefixes: &[&str],
    device: &Device,
) -> Result<HashMap<String, Tensor>, Qwen3AsrError> {
    let index_path = checkpoint_dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_slice(&std::fs::read(&index_path)?).map_err(|e| {
        Qwen3AsrError::InvalidCheckpoint(format!("bad model.safetensors.index.json: {e}"))
    })?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| Qwen3AsrError::InvalidCheckpoint("index has no weight_map".into()))?;
    let wanted = |name: &str| prefixes.iter().any(|p| name.starts_with(p));
    let shards: BTreeSet<&str> = weight_map
        .iter()
        .filter(|(name, _)| wanted(name))
        .filter_map(|(_, shard)| shard.as_str())
        .collect();
    if shards.is_empty() {
        return Err(Qwen3AsrError::InvalidCheckpoint(format!(
            "index lists no tensors under {prefixes:?}"
        )));
    }
    let mut tensors = HashMap::new();
    for shard in shards {
        if shard.contains('/') || shard.contains('\\') || shard.starts_with("..") {
            return Err(Qwen3AsrError::InvalidCheckpoint(format!(
                "shard name {shard:?} must be a plain file name"
            )));
        }
        tensors.extend(read_tensors_where(
            &checkpoint_dir.join(shard),
            wanted,
            device,
        )?);
    }
    if let Some(missing) = weight_map
        .keys()
        .find(|name| wanted(name) && !tensors.contains_key(name.as_str()))
    {
        return Err(Qwen3AsrError::InvalidCheckpoint(format!(
            "index lists tensor {missing} but its shard does not contain it"
        )));
    }
    Ok(tensors)
}

/// Load the audio tower tensors of a checkpoint directory into a [`VarBuilder`]
/// rooted at `thinker.audio_tower`.
///
/// `model.safetensors.index.json` selects the shards that hold audio tower
/// tensors. Stored `bfloat16` weights are converted to `dtype` on access.
pub fn audio_tower_var_builder(
    checkpoint_dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, Qwen3AsrError> {
    let tensors = read_indexed_tensors(checkpoint_dir, &[AUDIO_TOWER_PREFIX], device)?;
    // Names are kept in full so the builder prefix is `thinker.audio_tower`.
    Ok(VarBuilder::from_tensors(tensors, dtype, device).pp("thinker.audio_tower"))
}

/// Load the text decoder and language-model head tensors of a checkpoint directory into a
/// [`VarBuilder`] rooted at `thinker`.
///
/// Only `thinker.model.*` and `thinker.lm_head.*` are read, so the audio tower is never
/// loaded. Stored `bfloat16` weights are converted to `dtype` on access.
pub fn text_decoder_var_builder(
    checkpoint_dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, Qwen3AsrError> {
    let tensors = read_indexed_tensors(
        checkpoint_dir,
        &[TEXT_DECODER_PREFIX, LM_HEAD_PREFIX],
        device,
    )?;
    Ok(VarBuilder::from_tensors(tensors, dtype, device).pp("thinker"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn reads_only_prefixed_tensors_and_converts_bf16() {
        let dir = std::env::temp_dir().join(format!("devl-qwen3-weights-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 1.5f32 is 0x3FC00000, so bf16 is 0x3FC0.
        let bf16_one = 0x3FC0_u16.to_le_bytes();
        let f32_two = 2.0_f32.to_le_bytes();
        write_safetensors(
            &dir.join("a.safetensors"),
            &[
                ("thinker.audio_tower.w", "BF16", vec![1], bf16_one.to_vec()),
                ("thinker.model.other", "F32", vec![1], f32_two.to_vec()),
            ],
        );
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"thinker.audio_tower.w":"a.safetensors","thinker.model.other":"a.safetensors"}}"#,
        )
        .unwrap();
        let vb = audio_tower_var_builder(&dir, DType::F32, &Device::Cpu).unwrap();
        let w = vb.get(1, "w").unwrap();
        assert_eq!(w.dtype(), DType::F32);
        assert_eq!(w.to_vec1::<f32>().unwrap(), vec![1.5]);
        assert!(vb.get(1, "other").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_inconsistent_offsets() {
        let dir =
            std::env::temp_dir().join(format!("devl-qwen3-weights-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            &[(
                "thinker.audio_tower.w",
                "F32",
                vec![4],
                1.0_f32.to_le_bytes().to_vec(),
            )],
        );
        let err = read_prefixed_tensors(&path, AUDIO_TOWER_PREFIX, &Device::Cpu).unwrap_err();
        assert!(matches!(err, Qwen3AsrError::InvalidCheckpoint(_)));

        let header = br#"{"thinker.audio_tower.w":{"dtype":"F32","shape":[1],"data_offsets":[18446744073709551611,18446744073709551615]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        std::fs::write(&path, file).unwrap();
        let err = read_prefixed_tensors(&path, AUDIO_TOWER_PREFIX, &Device::Cpu).unwrap_err();
        assert!(matches!(err, Qwen3AsrError::InvalidCheckpoint(_)));

        write_safetensors(
            &path,
            &[("thinker.audio_tower.w", "F32", vec![usize::MAX, 2], vec![])],
        );
        let err = read_prefixed_tensors(&path, AUDIO_TOWER_PREFIX, &Device::Cpu).unwrap_err();
        assert!(matches!(err, Qwen3AsrError::InvalidCheckpoint(_)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
