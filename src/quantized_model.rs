//! The on-disk quantized model.
//!
//! The model downloads as `model.safetensors` (BF16, 235 MB). Once, after the
//! download, an app calls `quantize_model_file` to write `model.q8.gguf`
//! beside it and can delete the original: the FlowLM transformer's four linear
//! weights per layer (302 MB as f32, the memory-bound part of every frame)
//! go to Q8_0, everything else to F16, which is lossless for BF16 values in
//! range. 130 MB on disk instead of 235, and the quantization is no longer
//! paid at every launch. `PocketTTSModel::load` prefers the GGUF when present
//! and quantizes at load from the safetensors otherwise, so a sideloaded or
//! not-yet-converted pack still works.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};

pub const GGUF_FILE_NAME: &str = "model.q8.gguf";
pub const SAFETENSORS_FILE_NAME: &str = "model.safetensors";

/// The tensors that carry the FlowLM step's memory traffic.
pub fn is_quantized_linear(name: &str) -> bool {
    name.starts_with("flow_lm.transformer.layers.")
        && (name.ends_with(".self_attn.in_proj.weight")
            || name.ends_with(".self_attn.out_proj.weight")
            || name.ends_with(".linear1.weight")
            || name.ends_with(".linear2.weight"))
}

/// The Mimi *encoder* (audio in, latents out) ships in the file but a
/// text-to-speech engine never runs it: 17.6 M parameters, 35 MB, dropped.
pub fn is_unused(name: &str) -> bool {
    name.starts_with("mimi.encoder.") || name.starts_with("mimi.encoder_transformer.") || name.starts_with("mimi.downsample.")
}

/// Converts `source` (safetensors) into `destination` (GGUF). Writes to a
/// temporary file first and renames, so a crash mid-way leaves no half file
/// for the loader to trip over.
pub fn quantize_model_file(source: &Path, destination: &Path) -> Result<()> {
    let started = std::time::Instant::now();
    let tensors = candle_core::safetensors::load(source, &Device::Cpu)?;
    let mut names: Vec<&String> = tensors.keys().filter(|n| !is_unused(n)).collect();
    names.sort();
    let mut quantized: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    let (mut q8_bytes, mut f16_bytes) = (0usize, 0usize);
    for name in names {
        let t = &tensors[name];
        let (dtype, is_q8) = if is_quantized_linear(name) && t.dims().len() == 2 && t.dim(1)? % 32 == 0 {
            (GgmlDType::Q8_0, true)
        } else {
            (GgmlDType::F16, false)
        };
        let q = QTensor::quantize(&t.to_dtype(DType::F32)?, dtype)?;
        let bytes = q.storage_size_in_bytes();
        if is_q8 { q8_bytes += bytes } else { f16_bytes += bytes }
        quantized.push((name.clone(), q));
    }
    let refs: Vec<(&str, &QTensor)> = quantized.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let meta_arch = gguf_file::Value::String("pocket-tts".to_string());
    let meta_quant = gguf_file::Value::String("flow_lm.transformer linears Q8_0, rest F16".to_string());
    let metadata = [("general.architecture", &meta_arch), ("aloud.quantization", &meta_quant)];

    let tmp = destination.with_extension("gguf.partial");
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        gguf_file::write(&mut file, &metadata, &refs)?;
    }
    std::fs::rename(&tmp, destination)?;
    eprintln!(
        "[Quantize] {} -> {}: {} tensors, {:.0} MB Q8_0 + {:.0} MB F16, {:.1}s",
        source.display(),
        destination.display(),
        refs.len(),
        q8_bytes as f64 / 1e6,
        f16_bytes as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Reads a GGUF written by `quantize_model_file`: the Q8_0 tensors go to the
/// quantized-linear registry (see `modules::qlinear`), everything else comes
/// back widened to f32 for the ordinary `VarBuilder`.
pub fn load_gguf(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut plain: HashMap<String, Tensor> = HashMap::with_capacity(content.tensor_infos.len());
    let mut registry: HashMap<String, Arc<QTensor>> = HashMap::new();
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    for name in names {
        let q = content.tensor(&mut file, &name, device)?;
        match q.dtype() {
            GgmlDType::F16 | GgmlDType::F32 => {
                plain.insert(name, q.dequantize(device)?.to_dtype(DType::F32)?);
            },
            _ => {
                registry.insert(name, Arc::new(q));
            },
        }
    }
    eprintln!("[Quantize] loaded {}: {} f32 tensors, {} quantized", path.display(), plain.len(), registry.len());
    crate::modules::qlinear::register(registry);
    Ok(plain)
}
