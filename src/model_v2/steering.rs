//! Moving named concepts: the learned steering manifold, the caller's
//! geometric controls, and the head that predicts which controls were applied.

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use super::GeometricControls;
use crate::config::RejConfigV2;
use crate::model::parts::{normal, Mlp, Norm};

/// Each named concept owns `n_titan_directions` directions in subspace
/// coordinates; an intensity network over the pooled tokens mixes them into
/// one shift per named concept and layer.
#[derive(Clone, Debug)]
pub struct SteeringManifold {
    directions: Tensor,
    intensity: Mlp,
    named: usize,
    n_directions: usize,
}

impl SteeringManifold {
    pub fn new(config: &RejConfigV2, vb: VarBuilder) -> Result<Self> {
        let named = config.base.n_named_concepts();
        let n_directions = config.n_titan_directions;
        let dims = [named, n_directions, config.subspace_rank];
        Ok(Self {
            directions: normal(&vb, &dims, "directions", config.base.initializer_range)?,
            intensity: Mlp::new(
                (config.base.d_model, config.steering_hidden, named * n_directions),
                config.base.dropout,
                false,
                vb.pp("intensity_mlp"),
            )?,
            named,
            n_directions,
        })
    }

    /// coords (B, K, r), tokens (B, T, d_model) → coords with the named
    /// concepts shifted.
    pub fn forward(&self, coords: &Tensor, tokens: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, k, _) = coords.dims3()?;
        let pooled = tokens.mean(1)?;
        let intensity = self
            .intensity
            .forward(&pooled, train)?
            .reshape((batch, self.named, self.n_directions))?;
        let intensity = candle_nn::ops::softmax(&intensity, D::Minus1)?;
        // bnd,ndr->bnr
        let delta = intensity
            .transpose(0, 1)?
            .contiguous()?
            .matmul(&self.directions)?
            .transpose(0, 1)?;
        let shifted = (coords.narrow(1, 0, self.named)? + delta)?;
        replace_named(&shifted, coords, self.named, k)
    }
}

/// `named_part` (B, n, r) in place of the first n concepts of `all` (B, K, r).
fn replace_named(named_part: &Tensor, all: &Tensor, named: usize, k: usize) -> Result<Tensor> {
    match named == k {
        true => Ok(named_part.clone()),
        false => Tensor::cat(&[named_part, &all.narrow(1, named, k - named)?], 1),
    }
}

/// Apply the caller's controls to the named concepts' mean and log standard
/// deviation (each (B, K, r)).
pub fn apply_controls(
    mean: &Tensor,
    log_std: &Tensor,
    controls: &GeometricControls,
    named: usize,
) -> Result<(Tensor, Tensor)> {
    let k = mean.dim(1)?;
    let mut named_mean = mean.narrow(1, 0, named)?;
    let mut named_log_std = log_std.narrow(1, 0, named)?;
    if let Some(magnitude) = &controls.magnitude {
        named_mean = named_mean.broadcast_mul(&(magnitude.unsqueeze(D::Minus1)? + 1.0)?)?;
    }
    if let Some(direction) = &controls.direction {
        named_mean = named_mean.broadcast_add(direction)?;
    }
    if let Some(uncertainty) = &controls.uncertainty {
        named_log_std = named_log_std.broadcast_add(&uncertainty.unsqueeze(D::Minus1)?)?;
    }
    if let Some(select) = &controls.select {
        let gate = candle_nn::ops::sigmoid(&select.unsqueeze(D::Minus1)?)?;
        named_mean = named_mean.broadcast_mul(&gate)?;
    }
    Ok((
        replace_named(&named_mean, mean, named, k)?,
        replace_named(&named_log_std, log_std, named, k)?,
    ))
}

/// Predicts the named-concept control magnitudes from the concept embeddings
/// pooled over concepts.
#[derive(Clone, Debug)]
pub struct ConceptAlignmentHead {
    norm: Norm,
    net: Mlp,
}

impl ConceptAlignmentHead {
    pub fn new(config: &RejConfigV2, vb: VarBuilder) -> Result<Self> {
        let dc = config.base.d_concept;
        Ok(Self {
            norm: Norm::new(dc, config.base.layer_norm_eps, vb.pp("norm"))?,
            net: Mlp::new(
                (dc, dc, config.base.n_named_concepts()),
                config.base.dropout,
                false,
                vb.pp("net"),
            )?,
        })
    }

    /// (B, K, d_concept) → (B, n_named).
    pub fn forward(&self, concepts: &Tensor, train: bool) -> Result<Tensor> {
        self.net.forward(&self.norm.forward(&concepts.mean(1)?)?, train)
    }
}
