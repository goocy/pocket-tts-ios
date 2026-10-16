//! Multi-head attention modules
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::rotary::RotaryEmbedding;

/// Multi-Head Attention with optional KV caching
#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let q_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&mut KVCache>,
        causal_mask: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Apply rotary embeddings BEFORE transpose (RoPE expects [batch, seq, heads, dim])
        let (q, k) = if let Some(rope) = rotary {
            let offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
            rope.forward(&q, &k, offset)?
        } else {
            (q, k)
        };

        // Transpose to [batch, num_heads, seq, head_dim] for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Update KV cache if provided; either way k_t is [batch, heads, head_dim, kv_len]
        let (k_t, v) = if let Some(cache) = kv_cache {
            cache.update(k, v)?
        } else {
            (k.transpose(2, 3)?.contiguous()?, v.contiguous()?)
        };

        // Attention scores: Q @ K^T
        let attn_weights = q.contiguous()?.matmul(&k_t)?;
        let attn_weights = (attn_weights * self.scale as f64)?;

        // Apply causal mask if needed
        let attn_weights = if causal_mask {
            let mask = self.create_causal_mask(seq_len, k_t.dim(3)?, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        // Softmax (use softmax_last_dim for better numerical precision, matches Kyutai's Moshi Rust)
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Weighted sum of values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    fn create_causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j <= i + (kv_len - q_len) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
    }
}

/// KV Cache for efficient autoregressive generation
#[derive(Debug)]
pub struct KVCache {
    /// Keys are cached already transposed, `[batch, heads, head_dim, seq]`,
    /// so `q.matmul(k_t)` reads a contiguous operand every step. Caching
    /// `[batch, heads, seq, head_dim]` and transposing per step gives a
    /// strided view the gemm crate tolerates but Accelerate's sgemm rejects —
    /// and copying it each step cost ~2.5 ms per frame on an A18.
    k_t_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k_t_cache: None,
            v_cache: None,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.k_t_cache.as_ref().map(|k| k.dim(3).unwrap_or(0)).unwrap_or(0)
    }

    /// Appends `k` and `v` (both `[batch, heads, seq, head_dim]`) and returns
    /// the full cache as `(k_t, v)` with `k_t` transposed to
    /// `[batch, heads, head_dim, seq]`; both are contiguous.
    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        // Both inputs contiguous so `cat` takes its contiguous fast path and
        // returns contiguous tensors; with a strided input it would return a
        // transposed view instead, which Accelerate's sgemm rejects.
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let v = v.contiguous()?;
        let (k_out, v_out) = match (&self.k_t_cache, &self.v_cache) {
            (Some(k_cache), Some(v_cache)) => {
                let k_new = Tensor::cat(&[k_cache, &k_t], 3)?.contiguous()?;
                let v_new = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                (k_new, v_new)
            },
            _ => (k_t, v),
        };

        self.k_t_cache = Some(k_out.clone());
        self.v_cache = Some(v_out.clone());

        Ok((k_out, v_out))
    }

    pub fn clear(&mut self) {
        self.k_t_cache = None;
        self.v_cache = None;
    }

    /// Preload the cache with a precomputed state (e.g. a saved voice KV state).
    ///
    /// `k` and `v` must already be in the cache layout `[batch, num_heads, seq, head_dim]`
    /// and (for `k`) post-RoPE, matching what `update()` stores. Subsequent `update()` calls
    /// append along the sequence dimension, and `seq_len()` reflects the preloaded length so
    /// RoPE offsets for following tokens continue from here.
    pub fn set(&mut self, k: Tensor, v: Tensor) {
        self.k_t_cache = k.transpose(2, 3).and_then(|t| t.contiguous()).ok();
        self.v_cache = v.contiguous().ok();
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Causal Self-Attention (convenience wrapper)
pub type CausalSelfAttention = MultiHeadAttention;

/// Fused Multi-Head Attention (Kyutai Pocket style)
/// Uses combined in_proj for Q/K/V instead of separate projections
#[derive(Debug)]
pub struct FusedMultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl FusedMultiHeadAttention {
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // Kyutai Pocket uses in_proj (combined QKV) and out_proj - NO BIAS
        let in_proj = candle_nn::linear_no_bias(hidden_size, hidden_size * 3, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&mut KVCache>,
        causal_mask: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;

        // Combined projection to get Q, K, V together
        let qkv = self.in_proj.forward(x)?;

        // Split into Q, K, V (each has hidden_size dimension)
        let q = qkv.narrow(2, 0, hidden_size)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?;
        let v = qkv.narrow(2, hidden_size * 2, hidden_size)?;

        // Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Apply rotary embeddings BEFORE transpose (RoPE expects [batch, seq, heads, dim])
        let (q, k) = if let Some(rope) = rotary {
            let offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
            rope.forward(&q, &k, offset)?
        } else {
            (q, k)
        };

        // Transpose to [batch, num_heads, seq, head_dim] for attention computation
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Update KV cache if provided; either way k_t is [batch, heads, head_dim, kv_len]
        // and both operands are contiguous (Accelerate's sgemm refuses strided ones).
        let (k_t, v) = if let Some(cache) = kv_cache {
            cache.update(k, v)?
        } else {
            (k.transpose(2, 3)?.contiguous()?, v.contiguous()?)
        };

        // Attention scores: Q @ K^T * scale
        let attn_weights = q.contiguous()?.matmul(&k_t)?;
        let attn_weights = (attn_weights * self.scale as f64)?;

        // Apply causal mask if needed
        let attn_weights = if causal_mask {
            let mask = self.create_causal_mask(seq_len, k_t.dim(3)?, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Weighted sum of values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        let final_output = self.out_proj.forward(&attn_output)?;

        Ok(final_output)
    }

    fn create_causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j <= i + (kv_len - q_len) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
    }
}
