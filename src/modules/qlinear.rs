//! A Linear whose weight is held quantized (GGML block format) and
//! multiplied with candle's quantized matmul kernels.
//!
//! The FlowLM transformer is 302 MB of f32 weights read once per generated
//! frame at batch size 1, which is memory-bandwidth-bound on a phone. Q8_0
//! reads a quarter of that. `ALOUD_QUANT` selects the format at load time:
//! `Q8_0` (default), `Q4K`, `Q6K`, `Q5K`, `Q4_0`, or `F32` for the original
//! f32 path.
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Tensors already quantized on disk (`quantized_model::load_gguf`), keyed by
/// full name. `QLinear::new` takes its weight from here when present, so the
/// model constructors need not know which file the pack came from. Filled
/// before the model is built and drained as it is; one model loads at a time.
static REGISTRY: Mutex<Option<HashMap<String, Arc<QTensor>>>> = Mutex::new(None);

pub fn register(tensors: HashMap<String, Arc<QTensor>>) {
    *REGISTRY.lock().unwrap() = Some(tensors);
}

fn take(name: &str) -> Option<Arc<QTensor>> {
    REGISTRY.lock().unwrap().as_mut().and_then(|m| m.remove(name))
}

pub fn clear_registry() {
    *REGISTRY.lock().unwrap() = None;
}

#[derive(Debug)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    /// The same weight expanded to f32, used for inputs of more than
    /// one position. See `dense_prefill`.
    dense: Option<Tensor>,
}

/// Whether to keep an f32 copy of each quantized weight for multi-position
/// inputs — the text read-in before a sentence's first frame.
///
/// Q8_0 wins at one position per call because that is bound by reading the
/// weights, and it reads a quarter as much. A read-in of ninety tokens is
/// bound by arithmetic instead, where candle's quantized kernel is a row of
/// dot products and the f32 path is a proper GEMM. The price is the f32 copy:
/// 302 MB for FlowLM. `ALOUD_DENSE_PREFILL=1` turns it on.
pub fn dense_prefill() -> bool {
    matches!(std::env::var("ALOUD_DENSE_PREFILL").as_deref(), Ok("1"))
}

pub fn selected_dtype() -> Option<GgmlDType> {
    match std::env::var("ALOUD_QUANT").as_deref() {
        Ok("F32") | Ok("f32") | Ok("none") => None,
        Ok("Q4_0") => Some(GgmlDType::Q4_0),
        Ok("Q4K") | Ok("Q4_K") => Some(GgmlDType::Q4K),
        Ok("Q5K") | Ok("Q5_K") => Some(GgmlDType::Q5K),
        Ok("Q6K") | Ok("Q6_K") => Some(GgmlDType::Q6K),
        _ => Some(GgmlDType::Q8_0),
    }
}

impl QLinear {
    pub fn new(in_dim: usize, out_dim: usize, with_bias: bool, vb: VarBuilder) -> Result<Self> {
        let full_name = format!("{}.weight", vb.prefix());
        let inner = if let Some(q) = take(&full_name) {
            if q.shape().dims() != [out_dim, in_dim] {
                candle_core::bail!("{full_name}: stored shape {:?}, expected [{out_dim}, {in_dim}]", q.shape());
            }
            QMatMul::from_arc(q)?
        } else {
            let weight = vb.get((out_dim, in_dim), "weight")?;
            match selected_dtype() {
                Some(dtype) => QMatMul::from_qtensor(QTensor::quantize(&weight, dtype)?)?,
                None => QMatMul::Tensor(weight),
            }
        };
        let bias = if with_bias { Some(vb.get(out_dim, "bias")?) } else { None };
        let dense = match &inner {
            QMatMul::QTensor(q) if dense_prefill() => Some(q.dequantize(&q.device())?.t()?.contiguous()?),
            _ => None,
        };
        Ok(Self { inner, bias, dense })
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let positions = if x.rank() == 3 { x.dim(1)? } else { 1 };
        let y = match &self.dense {
            Some(w) if positions > 1 => x.broadcast_matmul(w)?,
            _ => self.inner.forward(x)?,
        };
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}
