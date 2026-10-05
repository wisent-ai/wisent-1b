//! One layer of the first model, and the fusion that carries the concept
//! state from one decoding step to the next.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Init, Linear, VarBuilder};

use super::attention::{Attention, AttentionShape};
use super::parts::{linear_xavier, linear_zero, Mlp, Norm};
use crate::config::RejConfig;

/// Tokens attend to tokens; concepts read tokens, attend to each other and
/// pass a feed-forward; tokens read concepts through a gate that starts at
/// zero, so a fresh layer is a plain transformer layer.
#[derive(Clone, Debug)]
pub struct RejLayer {
    token_ln1: Norm,
    token_self_attn: Attention,
    token_ln2: Norm,
    token_ffn: Mlp,
    concept_ln1: Norm,
    concept_read_tokens: Attention,
    concept_ln2: Norm,
    concept_self_attn: Attention,
    concept_ln3: Norm,
    concept_ffn: Mlp,
    token_ln3: Norm,
    token_read_concepts: Attention,
    concept_to_token_gate: Tensor,
}

impl RejLayer {
    pub fn new(config: &RejConfig, vb: VarBuilder) -> Result<Self> {
        let (d, dc, eps, p) = (config.d_model, config.d_concept, config.layer_norm_eps, config.dropout);
        let shape = |q_dim, kv_dim, out_dim, n_heads, causal| AttentionShape {
            q_dim,
            kv_dim,
            out_dim,
            n_heads,
            d_head: config.d_head,
            causal,
        };
        let (h, ch) = (config.n_heads, config.concept_heads);
        Ok(Self {
            token_ln1: Norm::new(d, eps, vb.pp("token_ln1"))?,
            token_self_attn: Attention::new(shape(d, d, d, h, true), p, vb.pp("token_self_attn"))?,
            token_ln2: Norm::new(d, eps, vb.pp("token_ln2"))?,
            token_ffn: Mlp::new((d, config.intermediate_size, d), p, true, vb.pp("token_ffn"))?,
            concept_ln1: Norm::new(dc, eps, vb.pp("concept_ln1"))?,
            concept_read_tokens: Attention::new(shape(dc, d, dc, ch, false), p, vb.pp("concept_read_tokens"))?,
            concept_ln2: Norm::new(dc, eps, vb.pp("concept_ln2"))?,
            concept_self_attn: Attention::new(shape(dc, dc, dc, ch, false), p, vb.pp("concept_self_attn"))?,
            concept_ln3: Norm::new(dc, eps, vb.pp("concept_ln3"))?,
            concept_ffn: Mlp::new((dc, config.concept_intermediate_size, dc), p, true, vb.pp("concept_ffn"))?,
            token_ln3: Norm::new(d, eps, vb.pp("token_ln3"))?,
            token_read_concepts: Attention::new(shape(d, dc, d, h, false), p, vb.pp("token_read_concepts"))?,
            concept_to_token_gate: vb.get_with_hints(1, "concept_to_token_gate", Init::Const(0.0))?,
        })
    }

    /// tokens (B, T, d_model), concepts (B, K, d_concept) → both, updated.
    pub fn forward(&self, tokens: &Tensor, concepts: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let normed = self.token_ln1.forward(tokens)?;
        let tokens = (tokens + self.token_self_attn.forward(&normed, &normed, &normed, train)?)?;

        let normed = self.concept_ln1.forward(concepts)?;
        let concepts = (concepts + self.concept_read_tokens.forward(&normed, &tokens, &tokens, train)?)?;

        let normed = self.concept_ln2.forward(&concepts)?;
        let concepts = (&concepts + self.concept_self_attn.forward(&normed, &normed, &normed, train)?)?;

        let normed = self.concept_ln3.forward(&concepts)?;
        let concepts = (&concepts + self.concept_ffn.forward(&normed, train)?)?;

        let normed = self.token_ln3.forward(&tokens)?;
        let read = self.token_read_concepts.forward(&normed, &concepts, &concepts, train)?;
        let tokens = (&tokens + read.broadcast_mul(&self.concept_to_token_gate)?)?;

        let normed = self.token_ln2.forward(&tokens)?;
        let tokens = (&tokens + self.token_ffn.forward(&normed, train)?)?;
        Ok((tokens, concepts))
    }
}

/// Gated fusion of the previous step's final concept state into this step's
/// initial one: `initial + sigmoid(gate([initial, carried])) * value([initial, carried])`.
/// The value projection starts at zero, so the carry changes nothing until it
/// is trained.
#[derive(Clone, Debug)]
pub struct ConceptCarry {
    value: Linear,
    gate: Linear,
}

impl ConceptCarry {
    pub fn new(d_concept: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            value: linear_zero(2 * d_concept, d_concept, vb.pp("value"))?,
            gate: linear_xavier(2 * d_concept, d_concept, true, vb.pp("gate"))?,
        })
    }

    pub fn forward(&self, initial: &Tensor, carried: &Tensor) -> Result<Tensor> {
        let pair = Tensor::cat(&[initial, carried], D::Minus1)?;
        let gate = candle_nn::ops::sigmoid(&self.gate.forward(&pair)?)?;
        initial + (gate * self.value.forward(&pair)?)?
    }
}
