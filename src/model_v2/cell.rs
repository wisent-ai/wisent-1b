//! The non-linear cell that reads tokens into concepts, updates them, and
//! writes them back.

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use super::concepts::coords_to_ambient;
use crate::config::RejConfigV2;
use crate::model::parts::Mlp;

#[derive(Clone, Debug)]
pub struct NonlinearConceptCell {
    read_mlp: Mlp,
    update_mlp: Mlp,
    write_mlp: Mlp,
    manifold_decoder: Option<Mlp>,
}

impl NonlinearConceptCell {
    pub fn new(config: &RejConfigV2, vb: VarBuilder) -> Result<Self> {
        let (d, dc, r) = (config.base.d_model, config.base.d_concept, config.subspace_rank);
        let (h, p) = (config.concept_mlp_hidden, config.base.dropout);
        let manifold_decoder = match config.use_manifold_decoder {
            // The decoder has no dropout in the reference model.
            true => Some(Mlp::new((r, h, dc), 0.0, false, vb.pp("manifold_decoder"))?),
            false => None,
        };
        Ok(Self {
            read_mlp: Mlp::new((d + dc, h, dc), p, false, vb.pp("read_mlp"))?,
            update_mlp: Mlp::new((dc, h, dc), p, false, vb.pp("update_mlp"))?,
            write_mlp: Mlp::new((dc, h, d), p, false, vb.pp("write_mlp"))?,
            manifold_decoder,
        })
    }

    /// Each concept reads every token through the MLP, weighted by the router.
    /// tokens (B, T, d_model), concepts (B, K, d_concept), router (B, K, T)
    /// → update (B, K, d_concept).
    pub fn read(&self, tokens: &Tensor, concepts: &Tensor, router: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, len, d) = tokens.dims3()?;
        let (_, k, dc) = concepts.dims3()?;
        let concept_per_token = concepts.unsqueeze(2)?.broadcast_as((batch, k, len, dc))?.contiguous()?;
        let tokens_per_concept = tokens.unsqueeze(1)?.broadcast_as((batch, k, len, d))?.contiguous()?;
        let pair = Tensor::cat(&[&tokens_per_concept, &concept_per_token], D::Minus1)?;
        let update = self.read_mlp.forward(&pair, train)?;
        router.unsqueeze(D::Minus1)?.broadcast_mul(&update)?.sum(2)
    }

    pub fn update(&self, concepts: &Tensor, train: bool) -> Result<Tensor> {
        concepts + self.update_mlp.forward(concepts, train)?
    }

    /// concepts (B, K, d_concept) → token updates (B, K, d_model).
    pub fn write(&self, concepts: &Tensor, train: bool) -> Result<Tensor> {
        self.write_mlp.forward(concepts, train)
    }

    /// Subspace coordinates (B, K, r) → concept embeddings (B, K, d_concept):
    /// the linear reconstruction plus the centroid, plus the manifold
    /// decoder's curve when the model has one.
    pub fn decode_manifold(&self, coords: &Tensor, basis: &Tensor, centroid: &Tensor, train: bool) -> Result<Tensor> {
        let linear = coords_to_ambient(coords, basis)?.broadcast_add(&centroid.unsqueeze(0)?)?;
        match &self.manifold_decoder {
            None => Ok(linear),
            Some(decoder) => linear + decoder.forward(coords, train)?,
        }
    }
}
