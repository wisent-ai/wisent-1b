//! Training the geometric model: plain language modelling, with optional
//! random control perturbation; concept alignment against the controls a
//! batch was labelled with; and language-invariant alignment over parallel
//! sentences. The loss weights are the configuration's.

use candle_core::Tensor;
use serde::Serialize;

use super::{lm_loss, scalar, Trainer};
use crate::model_v2::{GeometricControls, OutputV2, PassV2, RejRnmV2};
use crate::Error;

/// One batch of v2 training.
#[derive(Clone, Debug)]
pub enum V2Batch {
    /// Token ids (B, T). With a scale, every row gets random magnitude
    /// controls drawn from N(0, scale²).
    Plain { tokens: Tensor, perturbation_scale: Option<f64> },
    /// Token ids (B, T) and the magnitudes (B, n_named) they were labelled with.
    Aligned { tokens: Tensor, controls: Tensor },
    /// Parallel token ids in two languages and their shared magnitudes.
    Multilingual { first: Tensor, second: Tensor, controls: Tensor },
}

/// The parts of one step's loss; a part the step did not compute is absent.
#[derive(Clone, Debug, Default, Serialize)]
pub struct V2Losses {
    pub total_loss: f32,
    pub lm_loss: f32,
    pub kl_loss: Option<f32>,
    pub geometry_loss: Option<f32>,
    pub align_loss: Option<f32>,
    pub inv_loss: Option<f32>,
}

/// One training step on `batch`.
pub fn train_v2_step(model: &RejRnmV2, trainer: &mut Trainer, batch: &V2Batch) -> Result<V2Losses, Error> {
    let config = &model.config;
    let pass = |tokens: &Tensor, controls: Option<&GeometricControls>, align: bool, emb: bool| {
        model.forward(
            tokens,
            PassV2 {
                controls,
                return_alignment_pred: align,
                return_concept_embedding: emb,
                train: true,
                ..PassV2::default()
            },
        )
    };
    let magnitude = |controls: &Tensor| GeometricControls { magnitude: Some(controls.clone()), ..Default::default() };
    let mut parts = Parts::default();
    match batch {
        V2Batch::Plain { tokens, perturbation_scale } => {
            let controls = match perturbation_scale {
                Some(scale) => {
                    let (rows, _) = tokens.dims2()?;
                    let shape = (rows, config.base.n_named_concepts());
                    Some(magnitude(&Tensor::randn(0f32, *scale as f32, shape, tokens.device())?))
                }
                None => None,
            };
            parts.add_pass(model, &pass(tokens, controls.as_ref(), false, false)?, tokens)?;
        }
        V2Batch::Aligned { tokens, controls } => {
            require(config.use_concept_alignment, "aligned training needs use_concept_alignment")?;
            let output = pass(tokens, Some(&magnitude(controls)), true, false)?;
            parts.add_pass(model, &output, tokens)?;
            parts.align(model, &output, controls)?;
        }
        V2Batch::Multilingual { first, second, controls } => {
            require(
                config.use_language_invariant_concepts,
                "multilingual training needs use_language_invariant_concepts",
            )?;
            let controls_v = magnitude(controls);
            let out_first = pass(first, Some(&controls_v), config.use_concept_alignment, true)?;
            let out_second = pass(second, Some(&controls_v), false, true)?;
            parts.add_pass(model, &out_first, first)?;
            parts.add_pass(model, &out_second, second)?;
            if config.use_concept_alignment {
                parts.align(model, &out_first, controls)?;
            }
            let pooled = |output: &OutputV2| -> Result<Tensor, Error> {
                let emb = output.concept_embedding.as_ref().ok_or_else(|| Error::Training("no concept embedding".into()))?;
                Ok(emb.mean(1)?)
            };
            let invariance = candle_nn::loss::mse(&pooled(&out_first)?, &pooled(&out_second)?)?;
            parts.add(Part::Invariance, invariance, Some(config.language_invariant_weight))?;
        }
    }
    let total = parts.total.ok_or_else(|| Error::Training("the step computed no loss".into()))?;
    trainer.step(&total)?;
    Ok(V2Losses { total_loss: scalar(&total)?, ..parts.losses })
}

/// Train on `batches` until they run out or `num_steps` steps are done.
/// `on_step(step, losses)` runs after every step, numbered from one.
pub fn train_v2(
    model: &RejRnmV2,
    trainer: &mut Trainer,
    batches: impl IntoIterator<Item = Result<V2Batch, Error>>,
    num_steps: usize,
    mut on_step: impl FnMut(usize, &V2Losses) -> Result<(), Error>,
) -> Result<Vec<V2Losses>, Error> {
    let mut history = Vec::new();
    for (index, batch) in batches.into_iter().take(num_steps).enumerate() {
        let losses = train_v2_step(model, trainer, &batch?)?;
        on_step(index + 1, &losses)?;
        history.push(losses);
    }
    Ok(history)
}

fn require(condition: bool, message: &str) -> Result<(), Error> {
    match condition {
        true => Ok(()),
        false => Err(Error::Training(message.into())),
    }
}

/// A part of the v2 loss.
#[derive(Clone, Copy)]
enum Part {
    LanguageModel,
    Kl,
    Geometry,
    Alignment,
    Invariance,
}

/// The weighted sum being built and the unweighted parts it reports.
#[derive(Default)]
struct Parts {
    total: Option<Tensor>,
    losses: V2Losses,
}

impl Parts {
    /// The language-model loss of one pass, which carries no weight, and the
    /// KL and geometry terms the pass reported, weighted by the configuration.
    fn add_pass(&mut self, model: &RejRnmV2, output: &OutputV2, tokens: &Tensor) -> Result<(), Error> {
        self.add(Part::LanguageModel, lm_loss(&output.logits, tokens)?, None)?;
        if let Some(kl) = &output.kl_loss {
            self.add(Part::Kl, kl.clone(), Some(model.config.kl_weight))?;
        }
        if let Some(geometry) = &output.geometry_loss {
            self.add(Part::Geometry, geometry.clone(), Some(model.config.geometry_weight))?;
        }
        Ok(())
    }

    fn align(&mut self, model: &RejRnmV2, output: &OutputV2, controls: &Tensor) -> Result<(), Error> {
        let predicted = output
            .predicted_controls
            .as_ref()
            .ok_or_else(|| Error::Training("the model returned no alignment prediction".into()))?;
        let loss = candle_nn::loss::mse(predicted, controls)?;
        self.add(Part::Alignment, loss, Some(model.config.alignment_weight))
    }

    /// Add `term`, times `weight` when it has one, to the total, and `term`
    /// to its reported part.
    fn add(&mut self, part: Part, term: Tensor, weight: Option<f64>) -> Result<(), Error> {
        let value = scalar(&term)?;
        let slot = match part {
            Part::LanguageModel => {
                self.losses.lm_loss += value;
                None
            }
            Part::Kl => Some(&mut self.losses.kl_loss),
            Part::Geometry => Some(&mut self.losses.geometry_loss),
            Part::Alignment => Some(&mut self.losses.align_loss),
            Part::Invariance => Some(&mut self.losses.inv_loss),
        };
        if let Some(slot) = slot {
            *slot = Some(slot.unwrap_or_default() + value);
        }
        let weighted = match weight {
            Some(weight) => (term * weight)?,
            None => term,
        };
        self.total = Some(match self.total.take() {
            Some(total) => (total + weighted)?,
            None => weighted,
        });
        Ok(())
    }
}
