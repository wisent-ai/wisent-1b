//! The first model: tokens and concepts advancing together through the layers.

use candle_core::{DType, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use super::layer::{ConceptCarry, RejLayer};
use super::parts::{normal, Norm};
use crate::config::RejConfig;
use crate::Error;

/// What one forward pass is asked to do.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pass<'a> {
    /// (B, n_named) or (n_named) scalar control magnitudes.
    pub controls: Option<&'a Tensor>,
    /// The final concept state of a previous step, fused into this one's.
    pub concept_state: Option<&'a Tensor>,
    pub return_concept_trace: bool,
    pub return_concept_state: bool,
    pub train: bool,
}

/// What one forward pass produced.
#[derive(Clone, Debug)]
pub struct Output {
    /// (B, T, vocab_size)
    pub logits: Tensor,
    /// One (B, K, d_concept) state per layer, when asked.
    pub concept_trace: Option<Vec<Tensor>>,
    /// The final (B, K, d_concept) state, when asked.
    pub concept_state: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct RejRnm {
    pub config: RejConfig,
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    concept_embeddings: Tensor,
    token_control_embedding: Tensor,
    layers: Vec<RejLayer>,
    concept_carry: Option<ConceptCarry>,
    token_ln: Norm,
    lm_head: Linear,
}

impl RejRnm {
    pub fn new(config: RejConfig, vb: VarBuilder) -> Result<Self, Error> {
        config.validate()?;
        let c = &config;
        let std = c.initializer_range;
        let token_weight = normal(&vb.pp("token_embeddings"), &[c.vocab_size, c.d_model], "weight", std)?;
        let position_weight =
            normal(&vb.pp("position_embeddings"), &[c.max_position_embeddings, c.d_model], "weight", std)?;
        let lm_weight = match c.tie_word_embeddings {
            true => token_weight.clone(),
            false => normal(&vb.pp("lm_head"), &[c.vocab_size, c.d_model], "weight", std)?,
        };
        let layers = (0..c.n_layers)
            .map(|index| RejLayer::new(c, vb.pp(format!("layers.{index}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let concept_carry = match c.carry_concept_state {
            true => Some(ConceptCarry::new(c.d_concept, vb.pp("concept_carry"))?),
            false => None,
        };
        Ok(Self {
            token_embeddings: Embedding::new(token_weight, c.d_model),
            position_embeddings: Embedding::new(position_weight, c.d_model),
            concept_embeddings: normal(&vb, &[c.n_concepts, c.d_concept], "concept_embeddings", std)?,
            token_control_embedding: normal(
                &vb,
                &[c.n_named_concepts(), c.d_model],
                "token_control_embedding",
                std,
            )?,
            layers,
            concept_carry,
            token_ln: Norm::new(c.d_model, c.layer_norm_eps, vb.pp("token_ln"))?,
            lm_head: Linear::new(lm_weight, None),
            config,
        })
    }

    pub fn named_concepts(&self) -> &[String] {
        &self.config.named_concepts
    }

    /// The device the model's variables live on.
    pub fn device(&self) -> &candle_core::Device {
        self.concept_embeddings.device()
    }

    /// `input_ids` (B, T) of u32 token ids.
    pub fn forward(&self, input_ids: &Tensor, pass: Pass<'_>) -> Result<Output, Error> {
        let (batch, len) = input_ids.dims2()?;
        if len > self.config.max_position_embeddings {
            return Err(Error::Control(format!(
                "sequence of {len} tokens exceeds max_position_embeddings {}",
                self.config.max_position_embeddings
            )));
        }
        let device = input_ids.device();
        let positions = Tensor::arange(0u32, len as u32, device)?.unsqueeze(0)?;
        let mut tokens = self
            .token_embeddings
            .forward(input_ids)?
            .broadcast_add(&self.position_embeddings.forward(&positions)?)?;

        let controls = pass.controls.map(|c| self.validate_controls(c)).transpose()?;
        if let Some(controls) = &controls {
            // A direct path from the controls to the token stream, so they
            // reach generation from the first layer.
            let token_control = controls.unsqueeze(1)?.broadcast_matmul(&self.token_control_embedding)?;
            tokens = tokens.broadcast_add(&token_control)?;
        }
        tokens = super::parts::dropout(&tokens, self.config.dropout, pass.train)?;

        let mut concepts = self.initial_concepts(controls.as_ref(), batch)?;
        if let Some(state) = pass.concept_state {
            let carry = self.concept_carry.as_ref().ok_or_else(|| {
                Error::Control(
                    "a concept state was passed but this model was built with \
                     carry_concept_state false, so it has no carry"
                        .into(),
                )
            })?;
            let expected = [batch, self.config.n_concepts, self.config.d_concept];
            if state.dims() != expected {
                return Err(Error::Control(format!(
                    "concept state shape {:?} must be {expected:?}",
                    state.dims()
                )));
            }
            concepts = carry.forward(&concepts, &state.to_device(device)?)?;
        }

        let mut trace = pass.return_concept_trace.then(Vec::new);
        for layer in &self.layers {
            (tokens, concepts) = layer.forward(&tokens, &concepts, pass.train)?;
            if let Some(trace) = &mut trace {
                trace.push(concepts.detach());
            }
        }
        let logits = self.lm_head.forward(&self.token_ln.forward(&tokens)?)?;
        Ok(Output {
            logits,
            concept_trace: trace,
            concept_state: pass.return_concept_state.then_some(concepts),
        })
    }

    /// Controls as (B, n_named) f32, refused when the width is not the
    /// number of named concepts.
    fn validate_controls(&self, controls: &Tensor) -> Result<Tensor, Error> {
        let controls = match controls.rank() {
            1 => controls.unsqueeze(0)?,
            _ => controls.clone(),
        };
        let width = controls.dim(candle_core::D::Minus1)?;
        if controls.rank() != 2 || width != self.config.n_named_concepts() {
            return Err(Error::Control(format!(
                "controls of shape {:?} must be (batch, {})",
                controls.dims(),
                self.config.n_named_concepts()
            )));
        }
        Ok(controls.to_dtype(DType::F32)?)
    }

    /// The learned concept embeddings for each batch row, with the named
    /// slots scaled by the controls.
    fn initial_concepts(&self, controls: Option<&Tensor>, batch: usize) -> Result<Tensor, Error> {
        let (k, dc) = (self.config.n_concepts, self.config.d_concept);
        let concepts = self.concept_embeddings.unsqueeze(0)?.broadcast_as((batch, k, dc))?;
        let Some(controls) = controls else {
            return Ok(concepts.contiguous()?);
        };
        let named = self.config.n_named_concepts();
        let scaled = controls
            .unsqueeze(2)?
            .broadcast_mul(&self.concept_embeddings.narrow(0, 0, named)?.unsqueeze(0)?)?;
        match named == k {
            true => Ok(scaled),
            false => Ok(Tensor::cat(&[&scaled, &concepts.narrow(1, named, k - named)?], 1)?),
        }
    }
}
