//! The geometric model: tokens and concepts advancing together, layer by
//! layer, with the KL and geometry terms a training step reads back.

use candle_core::{Module, Tensor};
use candle_nn::{Embedding, Init, Linear, VarBuilder};

use super::cell::NonlinearConceptCell;
use super::concepts::{kl_divergence, sample, ConceptRouter, SubspaceConceptBank};
use super::steering::{apply_controls, ConceptAlignmentHead, SteeringManifold};
use super::GeometricControls;
use crate::config::RejConfigV2;
use crate::model::parts::{dropout, normal, Mlp, Norm};
use crate::model::{Attention, AttentionShape};
use crate::Error;

#[derive(Clone, Copy, Debug, Default)]
pub struct PassV2<'a> {
    pub controls: Option<&'a GeometricControls>,
    pub return_concept_trace: bool,
    pub return_alignment_pred: bool,
    pub return_concept_embedding: bool,
    /// Use each concept's mean instead of drawing from its Gaussian.
    pub deterministic: bool,
    pub train: bool,
}

/// One layer's concept state, as the trace records it.
#[derive(Clone, Debug)]
pub struct TraceStep {
    pub mean: Tensor,
    pub std: Tensor,
    pub embedding: Tensor,
}

#[derive(Clone, Debug)]
pub struct OutputV2 {
    pub logits: Tensor,
    pub concept_trace: Option<Vec<TraceStep>>,
    /// Mean KL per batch row and layer, when the concepts are probabilistic.
    pub kl_loss: Option<Tensor>,
    /// Mean reconstruction error per layer, when geometry is regularised.
    pub geometry_loss: Option<Tensor>,
    pub predicted_controls: Option<Tensor>,
    pub concept_embedding: Option<Tensor>,
}

#[derive(Clone, Debug)]
struct TokenLayer {
    ln1: Norm,
    self_attn: Attention,
    ln2: Norm,
    ffn: Mlp,
}

#[derive(Clone, Debug)]
pub struct RejRnmV2 {
    pub config: RejConfigV2,
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    concept_bank: SubspaceConceptBank,
    concept_mean_init: Tensor,
    concept_log_std_init: Option<Tensor>,
    router: Option<ConceptRouter>,
    cells: Vec<NonlinearConceptCell>,
    steering: Option<Vec<SteeringManifold>>,
    token_layers: Vec<TokenLayer>,
    concept_to_token_gate: Tensor,
    token_control_embedding: Tensor,
    alignment_head: Option<ConceptAlignmentHead>,
    token_ln: Norm,
    lm_head: Linear,
}

impl RejRnmV2 {
    pub fn new(config: RejConfigV2, vb: VarBuilder) -> Result<Self, Error> {
        config.validate()?;
        let (c, b) = (&config, &config.base);
        let std = b.initializer_range;
        let (k, r, d) = (b.n_concepts, c.subspace_rank, b.d_model);
        let token_weight = normal(&vb.pp("token_embeddings"), &[b.vocab_size, d], "weight", std)?;
        let position_weight = normal(&vb.pp("position_embeddings"), &[b.max_position_embeddings, d], "weight", std)?;
        let lm_weight = match b.tie_word_embeddings {
            true => token_weight.clone(),
            false => normal(&vb.pp("lm_head"), &[b.vocab_size, d], "weight", std)?,
        };
        let n_layers = b.n_layers;
        let cells = layer_paths("concept_cells", n_layers)
            .map(|path| NonlinearConceptCell::new(c, vb.pp(path)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let steering = match c.use_titan_manifold {
            true => Some(
                layer_paths("steering_manifolds", n_layers)
                    .map(|path| SteeringManifold::new(c, vb.pp(path)))
                    .collect::<candle_core::Result<Vec<_>>>()?,
            ),
            false => None,
        };
        let shape = AttentionShape {
            q_dim: d,
            kv_dim: d,
            out_dim: d,
            n_heads: b.n_heads,
            d_head: b.d_head,
            causal: true,
        };
        let token_layers = layer_paths("token_layers", n_layers)
            .map(|path| -> candle_core::Result<TokenLayer> {
                let vb = vb.pp(path);
                Ok(TokenLayer {
                    ln1: Norm::new(d, b.layer_norm_eps, vb.pp("ln1"))?,
                    self_attn: Attention::new(shape, b.dropout, vb.pp("self_attn"))?,
                    ln2: Norm::new(d, b.layer_norm_eps, vb.pp("ln2"))?,
                    ffn: Mlp::new((d, b.intermediate_size, d), b.dropout, true, vb.pp("ffn"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            token_embeddings: Embedding::new(token_weight, d),
            position_embeddings: Embedding::new(position_weight, d),
            concept_bank: SubspaceConceptBank::new(c, vb.pp("concept_bank"))?,
            concept_mean_init: normal(&vb, &[k, r], "concept_mean_init", std)?,
            concept_log_std_init: match c.probabilistic_concepts {
                true => Some(normal(&vb, &[k, r], "concept_log_std_init", c.concept_log_std_init_range)?),
                false => None,
            },
            router: match c.use_concept_router {
                true => Some(ConceptRouter::new(d, k, b.dropout, vb.pp("router"))?),
                false => None,
            },
            cells,
            steering,
            token_layers,
            concept_to_token_gate: vb.get_with_hints(1, "concept_to_token_gate", Init::Const(0.0))?,
            token_control_embedding: normal(&vb, &[b.n_named_concepts(), d], "token_control_embedding", std)?,
            alignment_head: match c.use_concept_alignment {
                true => Some(ConceptAlignmentHead::new(c, vb.pp("concept_alignment_head"))?),
                false => None,
            },
            token_ln: Norm::new(d, b.layer_norm_eps, vb.pp("token_ln"))?,
            lm_head: Linear::new(lm_weight, None),
            config,
        })
    }

    pub fn named_concepts(&self) -> &[String] {
        &self.config.base.named_concepts
    }

    /// The device the model's variables live on.
    pub fn device(&self) -> &candle_core::Device {
        self.concept_mean_init.device()
    }

    /// `input_ids` (B, T) of u32 token ids.
    pub fn forward(&self, input_ids: &Tensor, pass: PassV2<'_>) -> Result<OutputV2, Error> {
        let (c, b) = (&self.config, &self.config.base);
        let (batch, len) = input_ids.dims2()?;
        if len > b.max_position_embeddings {
            return Err(Error::Control(format!(
                "sequence of {len} tokens exceeds max_position_embeddings {}",
                b.max_position_embeddings
            )));
        }
        let device = input_ids.device();
        let positions = Tensor::arange(0u32, len as u32, device)?.unsqueeze(0)?;
        let mut tokens = self
            .token_embeddings
            .forward(input_ids)?
            .broadcast_add(&self.position_embeddings.forward(&positions)?)?;

        let controls = match pass.controls {
            Some(controls) => controls.validated(c)?,
            None => GeometricControls::default(),
        };
        if let Some(magnitude) = &controls.magnitude {
            let token_control = magnitude.unsqueeze(1)?.broadcast_matmul(&self.token_control_embedding)?;
            tokens = tokens.broadcast_add(&token_control)?;
        }
        tokens = dropout(&tokens, b.dropout, pass.train)?;

        let (k, r) = (b.n_concepts, c.subspace_rank);
        let mean = self.concept_mean_init.unsqueeze(0)?.broadcast_as((batch, k, r))?.contiguous()?;
        let log_std = match &self.concept_log_std_init {
            Some(init) => init.unsqueeze(0)?.broadcast_as((batch, k, r))?.contiguous()?,
            None => mean.zeros_like()?,
        };
        let (mut mean, log_std) = apply_controls(&mean, &log_std, &controls, b.n_named_concepts())?;

        let basis = self.concept_bank.basis()?;
        let centroid = &self.concept_bank.centroid;
        let deterministic = pass.deterministic || !c.probabilistic_concepts;
        let gate = candle_nn::ops::sigmoid(&self.concept_to_token_gate)?;
        let mut kl_loss: Option<Tensor> = None;
        let mut geometry_loss: Option<Tensor> = None;
        let mut trace = pass.return_concept_trace.then(Vec::new);
        let mut concept_emb = None;

        for (index, (cell, layer)) in self.cells.iter().zip(&self.token_layers).enumerate() {
            let normed = layer.ln1.forward(&tokens)?;
            tokens = (&tokens + layer.self_attn.forward(&normed, &normed, &normed, pass.train)?)?;
            if let Some(steering) = &self.steering {
                mean = steering[index].forward(&mean, &tokens, pass.train)?;
            }

            let coords = sample(&mean, &log_std, deterministic)?;
            let emb = cell.decode_manifold(&coords, &basis, centroid, pass.train)?;
            let router = match &self.router {
                Some(router) => router.forward(&tokens, pass.train)?,
                None => (Tensor::ones((batch, k, len), candle_core::DType::F32, device)? / len as f64)?,
            };
            let emb = (&emb + cell.read(&tokens, &emb, &router, pass.train)?)?;
            let emb = cell.update(&emb, pass.train)?;

            let centered = emb.broadcast_sub(&centroid.unsqueeze(0)?)?;
            mean = self.concept_bank.project_to_subspace(&centered, &basis)?;
            if c.use_geometry_regularization {
                let rebuilt = self.concept_bank.project_from_subspace(&mean, &basis)?;
                accumulate(&mut geometry_loss, candle_nn::loss::mse(&emb, &rebuilt)?)?;
            }
            if c.probabilistic_concepts {
                accumulate(&mut kl_loss, kl_divergence(&mean, &log_std)?)?;
            }

            let writes = cell.write(&emb, pass.train)?;
            let update = router.transpose(1, 2)?.contiguous()?.matmul(&writes)?;
            tokens = (&tokens + update.broadcast_mul(&gate)?)?;
            let normed = layer.ln2.forward(&tokens)?;
            tokens = (&tokens + layer.ffn.forward(&normed, pass.train)?)?;

            if let Some(trace) = &mut trace {
                trace.push(TraceStep { mean: mean.detach(), std: log_std.exp()?.detach(), embedding: emb.detach() });
            }
            concept_emb = Some(emb);
        }

        let logits = self.lm_head.forward(&self.token_ln.forward(&tokens)?)?;
        let layers = b.n_layers as f64;
        let predicted_controls = match (&self.alignment_head, &concept_emb, pass.return_alignment_pred) {
            (Some(head), Some(emb), true) => Some(head.forward(emb, pass.train)?),
            _ => None,
        };
        Ok(OutputV2 {
            logits,
            concept_trace: trace,
            kl_loss: kl_loss.map(|kl| kl / (batch as f64 * layers)).transpose()?,
            geometry_loss: geometry_loss.map(|geometry| geometry / layers).transpose()?,
            predicted_controls,
            concept_embedding: concept_emb.filter(|_| pass.return_concept_embedding),
        })
    }
}

fn accumulate(total: &mut Option<Tensor>, term: Tensor) -> candle_core::Result<()> {
    *total = Some(match total.take() {
        Some(sum) => (sum + term)?,
        None => term,
    });
    Ok(())
}

/// `name.0` … `name.{n-1}`: the variable path of each layer's part.
fn layer_paths(name: &'static str, n_layers: usize) -> impl Iterator<Item = String> {
    (0..n_layers).map(move |index| format!("{name}.{index}"))
}
