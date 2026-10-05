//! Training either model.
//!
//! The optimiser is candle's AdamW: the learning rate is the caller's, and
//! beta1, beta2, epsilon and weight decay are candle's declared defaults
//! unless the caller states them. Gradients are clipped only when the caller
//! names a maximum norm. Every step's losses are returned to the caller, who
//! decides what to print and when to save.

mod control;
mod data;
mod first;
mod v2;

pub use control::{ConceptExample, ConceptProbe, ControlFineTuner};
pub use data::{batches, lm_loss, padded, TokenDataset};
pub use first::{train, train_step, CarrySchedule};
pub use v2::{train_v2, train_v2_step, V2Batch, V2Losses};

use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::Error;

/// The optimiser settings a run states.
#[derive(Clone, Copy, Debug)]
pub struct OptimiserSettings {
    pub learning_rate: f64,
    pub beta1: Option<f64>,
    pub beta2: Option<f64>,
    pub eps: Option<f64>,
    pub weight_decay: Option<f64>,
    /// Scale the gradients down to this total norm when they exceed it.
    pub max_grad_norm: Option<f64>,
}

/// AdamW over `vars` and the clipping the run asked for.
pub struct Trainer {
    optimiser: AdamW,
    vars: Vec<Var>,
    max_grad_norm: Option<f64>,
}

impl Trainer {
    pub fn new(vars: Vec<Var>, settings: OptimiserSettings) -> Result<Self, Error> {
        let declared = ParamsAdamW::default();
        let params = ParamsAdamW {
            lr: settings.learning_rate,
            beta1: settings.beta1.unwrap_or(declared.beta1),
            beta2: settings.beta2.unwrap_or(declared.beta2),
            eps: settings.eps.unwrap_or(declared.eps),
            weight_decay: settings.weight_decay.unwrap_or(declared.weight_decay),
        };
        Ok(Self {
            optimiser: AdamW::new(vars.clone(), params)?,
            vars,
            max_grad_norm: settings.max_grad_norm,
        })
    }

    /// Backpropagate `loss`, clip if asked, and update every variable.
    pub fn step(&mut self, loss: &Tensor) -> Result<(), Error> {
        let mut grads = loss.backward()?;
        if let Some(max_norm) = self.max_grad_norm {
            clip_grad_norm(&mut grads, &self.vars, max_norm)?;
        }
        self.optimiser.step(&grads)?;
        Ok(())
    }
}

/// Scale every gradient by max_norm / ‖g‖ when the total norm ‖g‖ exceeds max_norm.
fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<(), Error> {
    let mut total = 0f64;
    for var in vars {
        if let Some(grad) = grads.get(var) {
            total += grad.sqr()?.sum_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
        }
    }
    let norm = total.sqrt();
    if norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / norm;
    for var in vars {
        if let Some(grad) = grads.remove(var) {
            grads.insert(var, (grad * scale)?);
        }
    }
    Ok(())
}

/// A scalar loss as f32.
pub(crate) fn scalar(loss: &Tensor) -> Result<f32, Error> {
    Ok(loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?)
}
