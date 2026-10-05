//! What a concept is in the geometric model: a subspace with a basis and a
//! centroid, a Gaussian state in its coordinates, and the router that decides
//! which tokens each concept reads.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use crate::config::RejConfigV2;
use crate::model::parts::{dropout, linear, normal};

/// Orthonormal rows of `x` (K, rank, d) by modified Gram–Schmidt, batched
/// over K and differentiable. It spans the same subspace as a QR factorisation;
/// a row may differ from Householder QR's by its sign.
pub fn orthonormalize(x: &Tensor) -> Result<Tensor> {
    let rank = x.dim(1)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(rank);
    for index in 0..rank {
        let mut row = x.narrow(1, index, 1)?;
        for done in &rows {
            let projection = (&row * done)?.sum_keepdim(D::Minus1)?;
            row = (&row - done.broadcast_mul(&projection)?)?;
        }
        let norm = row.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        rows.push(row.broadcast_div(&norm)?);
    }
    Tensor::cat(&rows, 1)
}

/// coords (B, K, r) and basis (K, r, d) → (B, K, d): `bkr,krd->bkd`.
pub fn coords_to_ambient(coords: &Tensor, basis: &Tensor) -> Result<Tensor> {
    coords
        .transpose(0, 1)?
        .contiguous()?
        .matmul(basis)?
        .transpose(0, 1)?
        .contiguous()
}

/// x (B, K, d) and basis (K, r, d) → (B, K, r): `bkd,krd->bkr`.
pub fn ambient_to_coords(x: &Tensor, basis: &Tensor) -> Result<Tensor> {
    x.transpose(0, 1)?
        .contiguous()?
        .matmul(&basis.transpose(1, 2)?.contiguous()?)?
        .transpose(0, 1)?
        .contiguous()
}

/// One subspace per concept slot: a basis of `subspace_rank` rows and a centroid.
#[derive(Clone, Debug)]
pub struct SubspaceConceptBank {
    basis_raw: Tensor,
    pub centroid: Tensor,
    normalize: bool,
}

impl SubspaceConceptBank {
    pub fn new(config: &RejConfigV2, vb: VarBuilder) -> Result<Self> {
        let (k, r, dc) = (config.base.n_concepts, config.subspace_rank, config.base.d_concept);
        let std = config.base.initializer_range;
        Ok(Self {
            basis_raw: normal(&vb, &[k, r, dc], "basis_raw", std)?,
            centroid: normal(&vb, &[k, dc], "centroid", std)?,
            normalize: config.normalize_subspace_basis,
        })
    }

    /// The bases (K, r, d_concept), orthonormalised when the configuration says so.
    pub fn basis(&self) -> Result<Tensor> {
        match self.normalize {
            true => orthonormalize(&self.basis_raw),
            false => Ok(self.basis_raw.clone()),
        }
    }

    pub fn project_to_subspace(&self, x: &Tensor, basis: &Tensor) -> Result<Tensor> {
        ambient_to_coords(x, basis)
    }

    pub fn project_from_subspace(&self, coords: &Tensor, basis: &Tensor) -> Result<Tensor> {
        coords_to_ambient(coords, basis)?.broadcast_add(&self.centroid.unsqueeze(0)?)
    }
}

/// A reparameterised draw from N(mean, exp(log_std)²), or the mean itself.
pub fn sample(mean: &Tensor, log_std: &Tensor, deterministic: bool) -> Result<Tensor> {
    match deterministic {
        true => Ok(mean.clone()),
        false => mean + (log_std.exp()? * mean.randn_like(0.0, 1.0)?)?,
    }
}

/// KL(N(mean, std²) ‖ N(0, 1)) summed over every element:
/// ½ (mean² + std² − 1 − 2 log std).
pub fn kl_divergence(mean: &Tensor, log_std: &Tensor) -> Result<Tensor> {
    let variance = log_std.exp()?.sqr()?;
    let inner = ((mean.sqr()? + variance)? - (log_std * 2.0)?)?;
    ((inner - 1.0)? * 0.5)?.sum_all()
}

/// Per-token relevance of each concept: weights (B, K, T) summing to one over T.
#[derive(Clone, Debug)]
pub struct ConceptRouter {
    query: Linear,
    dropout: f32,
}

impl ConceptRouter {
    pub fn new(d_model: usize, n_concepts: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            query: linear(d_model, n_concepts, false, vb.pp("query"))?,
            dropout,
        })
    }

    pub fn forward(&self, tokens: &Tensor, train: bool) -> Result<Tensor> {
        let logits = self.query.forward(tokens)?.transpose(1, 2)?.contiguous()?;
        let weights = candle_nn::ops::softmax(&logits, D::Minus1)?;
        dropout(&weights, self.dropout, train)
    }
}
