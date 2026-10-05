//! Making the named concepts controllable: a probe that reads a named concept
//! slot and predicts a label for it, and a fine-tuner that teaches the model
//! to keep predicting text under random concept perturbations.

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};

use super::{lm_loss, padded, scalar, OptimiserSettings, Trainer};
use crate::model::parts::{linear, Norm};
use crate::model::{Pass, RejRnm};
use crate::tokenizer::RejTokenizer;
use crate::Error;

/// A text labelled with how strongly it shows one named concept.
#[derive(Clone, Debug)]
pub struct ConceptExample {
    pub concept: String,
    pub text: String,
    /// The target probability that the concept is present.
    pub label: f32,
}

/// A layer norm and a linear map from one named concept's last-layer state to
/// a logit, trained with binary cross-entropy while the model stays fixed.
pub struct ConceptProbe {
    norm: Norm,
    head: Linear,
    trainer: Trainer,
}

impl ConceptProbe {
    pub fn new(model: &RejRnm, settings: OptimiserSettings) -> Result<Self, Error> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, model.device());
        let dc = model.config.d_concept;
        let norm = Norm::new(dc, model.config.layer_norm_eps, vb.pp("norm"))?;
        let head = linear(dc, 1, true, vb.pp("head"))?;
        Ok(Self { norm, head, trainer: Trainer::new(varmap.all_vars(), settings)? })
    }

    /// One update of the probe on `examples`; returns the loss before it.
    pub fn step(&mut self, model: &RejRnm, tokenizer: &RejTokenizer, examples: &[ConceptExample]) -> Result<f32, Error> {
        let device = model.device();
        let named = model.named_concepts();
        let sequences = examples
            .iter()
            .map(|example| tokenizer.encode(&example.text))
            .collect::<Result<Vec<_>, Error>>()?;
        let batch = padded(&sequences, tokenizer.pad_token_id(), device)?;
        let output = model.forward(&batch, Pass { return_concept_trace: true, ..Pass::default() })?;
        let last = output
            .concept_trace
            .and_then(|trace| trace.last().cloned())
            .ok_or_else(|| Error::Training("the model returned no concept trace".into()))?;
        let selected = examples
            .iter()
            .enumerate()
            .map(|(row, example)| -> Result<Tensor, Error> {
                let slot = crate::generate::index_of(named, &example.concept)?;
                Ok(last.i((row, slot))?.unsqueeze(0)?)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let states = Tensor::cat(&selected, 0)?.detach();
        let logits = self.head.forward(&self.norm.forward(&states)?)?.squeeze(1)?;
        let labels: Vec<f32> = examples.iter().map(|example| example.label).collect();
        let labels = Tensor::new(labels, device)?;
        let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;
        self.trainer.step(&loss)?;
        scalar(&loss)
    }
}

/// Fine-tunes a model to respond smoothly to concept controls: each step adds
/// the next-token loss under random magnitudes drawn from N(0, scale²) to the
/// loss without controls.
pub struct ControlFineTuner {
    pub perturbation_scale: f64,
}

impl ControlFineTuner {
    /// One step on `batch` (B, T); returns (loss without controls, loss with them).
    pub fn step(&self, model: &RejRnm, trainer: &mut Trainer, batch: &Tensor) -> Result<(f32, f32), Error> {
        let base = model.forward(batch, Pass { train: true, ..Pass::default() })?;
        let base_loss = lm_loss(&base.logits, batch)?;
        let (rows, _) = batch.dims2()?;
        let controls = random_controls(rows, model.config.n_named_concepts(), self.perturbation_scale, model.device())?;
        let perturbed = model.forward(batch, Pass { controls: Some(&controls), train: true, ..Pass::default() })?;
        let perturbed_loss = lm_loss(&perturbed.logits, batch)?;
        trainer.step(&(&base_loss + &perturbed_loss)?)?;
        Ok((scalar(&base_loss)?, scalar(&perturbed_loss)?))
    }
}

fn random_controls(rows: usize, named: usize, scale: f64, device: &Device) -> Result<Tensor, Error> {
    Ok(Tensor::randn(0f32, scale as f32, (rows, named), device)?)
}
