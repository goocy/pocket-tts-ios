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

#[derive(Debug)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
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
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let inner = match selected_dtype() {
            Some(dtype) => QMatMul::from_qtensor(QTensor::quantize(&weight, dtype)?)?,
            None => QMatMul::Tensor(weight),
        };
        let bias = if with_bias { Some(vb.get(out_dim, "bias")?) } else { None };
        Ok(Self { inner, bias })
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.inner.forward(x)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}
