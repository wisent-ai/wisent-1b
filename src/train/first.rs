//! Training the first model: one step, and the loop over batches.

use candle_core::Tensor;

use super::{lm_loss, scalar, Trainer};
use crate::model::{Pass, RejRnm};
use crate::Error;

/// How the cross-step concept carry is trained. Teacher forcing runs a whole
/// sequence in one pass, so nothing in an ordinary step is a previous step and
/// the carry never fires; a carried step runs the batch `passes` times, each
/// pass starting from the concept state the previous one ended on (detached).
/// One step in every `every` is carried.
#[derive(Clone, Copy, Debug)]
pub struct CarrySchedule {
    pub passes: usize,
    pub every: usize,
}

/// One step on `batch` (B, T): the mean next-token loss over `passes` passes.
pub fn train_step(model: &RejRnm, trainer: &mut Trainer, batch: &Tensor, passes: usize) -> Result<f32, Error> {
    if passes == 0 {
        return Err(Error::Training("a step needs at least one pass".into()));
    }
    let carry = model.config.carry_concept_state;
    if passes > 1 && !carry {
        return Err(Error::Training(
            "more than one pass needs carry_concept_state true in the model's configuration".into(),
        ));
    }
    let mut state: Option<Tensor> = None;
    let mut losses = Vec::with_capacity(passes);
    for _ in 0..passes {
        let output = model.forward(
            batch,
            Pass {
                concept_state: state.as_ref(),
                return_concept_state: carry,
                train: true,
                ..Pass::default()
            },
        )?;
        losses.push(lm_loss(&output.logits, batch)?);
        if carry {
            state = output.concept_state.map(|state| state.detach());
        }
    }
    let loss = Tensor::stack(&losses, 0)?.mean_all()?;
    trainer.step(&loss)?;
    scalar(&loss)
}

/// Train on `batches` (each (B, T) u32) until they run out or `num_steps`
/// steps are done. `on_step(step, loss)` runs after every step, numbered from
/// one; it may save a checkpoint or report progress.
pub fn train(
    model: &RejRnm,
    trainer: &mut Trainer,
    batches: impl IntoIterator<Item = Result<Tensor, Error>>,
    num_steps: usize,
    carry: Option<CarrySchedule>,
    mut on_step: impl FnMut(usize, f32) -> Result<(), Error>,
) -> Result<Vec<f32>, Error> {
    if let Some(schedule) = carry {
        if !model.config.carry_concept_state {
            return Err(Error::Training(
                "a carry schedule needs carry_concept_state true in the model's configuration".into(),
            ));
        }
        if schedule.passes < 2 || schedule.every == 0 {
            return Err(Error::Training(
                "a carry schedule needs at least two passes on one step in a positive number".into(),
            ));
        }
    } else if model.config.carry_concept_state {
        return Err(Error::Training(
            "the configuration carries the concept state, so training needs a carry schedule: \
             with one pass the carry is never called and never receives a gradient"
                .into(),
        ));
    }
    let mut losses = Vec::new();
    for (index, batch) in batches.into_iter().take(num_steps).enumerate() {
        let step = index + 1;
        let passes = match carry {
            Some(schedule) if step % schedule.every == 0 => schedule.passes,
            _ => 1,
        };
        let loss = train_step(model, trainer, &batch?, passes)?;
        losses.push(loss);
        on_step(step, loss)?;
    }
    Ok(losses)
}
