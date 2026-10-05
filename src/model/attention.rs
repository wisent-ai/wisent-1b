//! Multi-head attention whose queries and keys may have different widths, so
//! tokens can read concepts and concepts can read tokens.

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::parts::{dropout, linear_xavier};

/// The widths and head layout of one attention.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub out_dim: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub causal: bool,
}

#[derive(Clone, Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    shape: AttentionShape,
    dropout: f32,
}

impl Attention {
    pub fn new(shape: AttentionShape, dropout: f32, vb: VarBuilder) -> Result<Self> {
        let inner = shape.n_heads * shape.d_head;
        Ok(Self {
            q_proj: linear_xavier(shape.q_dim, inner, false, vb.pp("q_proj"))?,
            k_proj: linear_xavier(shape.kv_dim, inner, false, vb.pp("k_proj"))?,
            v_proj: linear_xavier(shape.kv_dim, inner, false, vb.pp("v_proj"))?,
            out_proj: linear_xavier(inner, shape.out_dim, false, vb.pp("out_proj"))?,
            shape,
            dropout,
        })
    }

    /// `query` (B, Tq, q_dim) attends over `key`/`value` (B, Tkv, kv_dim);
    /// the result is (B, Tq, out_dim).
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, t_q, _) = query.dims3()?;
        let (_, t_kv, _) = key.dims3()?;
        let heads = |proj: &Linear, xs: &Tensor, len: usize| -> Result<Tensor> {
            proj.forward(xs)?
                .reshape((batch, len, self.shape.n_heads, self.shape.d_head))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = heads(&self.q_proj, query, t_q)?;
        let k = heads(&self.k_proj, key, t_kv)?;
        let v = heads(&self.v_proj, value, t_kv)?;

        let scores = (q.matmul(&k.t()?)? / (self.shape.d_head as f64).sqrt())?;
        let scores = match self.shape.causal {
            true => scores.broadcast_add(&causal_mask(t_q, t_kv, query.device())?)?,
            false => scores,
        };
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let weights = dropout(&weights, self.dropout, train)?;

        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, t_q, self.shape.n_heads * self.shape.d_head))?;
        let out = self.out_proj.forward(&out)?;
        dropout(&out, self.dropout, train)
    }
}

/// Zero where a query may see a key, minus infinity where the key lies after it.
fn causal_mask(t_q: usize, t_kv: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t_q)
        .flat_map(|i| (0..t_kv).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (t_q, t_kv), device)
}
