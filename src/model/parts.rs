//! The small pieces both models are built from, initialised the way the
//! reference implementation initialised them.
//!
//! Linear layers follow PyTorch's `nn.Linear` default (weight and bias uniform
//! in ±1/√fan_in), attention projections Xavier-uniform (±√(6/(fan_in+fan_out))),
//! layer norms start at weight one and bias zero.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Init, Linear, VarBuilder};

/// A linear layer with PyTorch's default initialisation.
pub fn linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_dim as f64).sqrt();
    let init = Init::Uniform { lo: -bound, up: bound };
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
    let bias = match bias {
        true => Some(vb.get_with_hints(out_dim, "bias", init)?),
        false => None,
    };
    Ok(Linear::new(weight, bias))
}

/// A linear layer with a Xavier-uniform weight and, if any, a zero bias.
pub fn linear_xavier(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bound = (6.0 / (in_dim + out_dim) as f64).sqrt();
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", Init::Uniform { lo: -bound, up: bound })?;
    let bias = match bias {
        true => Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?),
        false => None,
    };
    Ok(Linear::new(weight, bias))
}

/// A linear layer that starts at exactly zero, weight and bias.
pub fn linear_zero(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", Init::Const(0.0))?;
    let bias = vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// A trainable tensor drawn from N(0, stdev²).
pub fn normal(vb: &VarBuilder, dims: &[usize], name: &str, stdev: f64) -> Result<Tensor> {
    vb.get_with_hints(dims, name, Init::Randn { mean: 0.0, stdev })
}

/// Dropout while training with a non-zero rate; the identity otherwise.
pub fn dropout(xs: &Tensor, rate: f32, train: bool) -> Result<Tensor> {
    match train && rate > 0.0 {
        true => candle_nn::ops::dropout(xs, rate),
        false => Ok(xs.clone()),
    }
}

/// Layer normalisation over the last dimension, built from differentiable
/// tensor operations so it trains.
#[derive(Clone, Debug)]
pub struct Norm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Norm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints(size, "weight", Init::Const(1.0))?,
            bias: vb.get_with_hints(size, "bias", Init::Const(0.0))?,
            eps: eps as f32,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm_slow(xs, &self.weight, &self.bias, self.eps)
    }
}

/// Two linear layers with an exact GELU between them, dropout after the
/// activation and, when `final_dropout`, after the output.
#[derive(Clone, Debug)]
pub struct Mlp {
    first: Linear,
    second: Linear,
    dropout: f32,
    final_dropout: bool,
}

impl Mlp {
    pub fn new(
        dims: (usize, usize, usize),
        dropout: f32,
        final_dropout: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (d_in, hidden, d_out) = dims;
        Ok(Self {
            first: linear(d_in, hidden, true, vb.pp("fc1"))?,
            second: linear(hidden, d_out, true, vb.pp("fc2"))?,
            dropout,
            final_dropout,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let hidden = self.first.forward(xs)?.gelu_erf()?;
        let hidden = dropout(&hidden, self.dropout, train)?;
        let out = self.second.forward(&hidden)?;
        match self.final_dropout {
            true => dropout(&out, self.dropout, train),
            false => Ok(out),
        }
    }
}
