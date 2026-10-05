//! Decoding with the first model.

use candle_core::{IndexOp, Tensor};

use super::{named_vector, next_token, window, GenerationOutput, Sampling};
use crate::model::{Pass, RejRnm};
use crate::tokenizer::RejTokenizer;
use crate::Error;

/// Generate up to `max_new_tokens` after `prompt` with a magnitude per named
/// concept in `controls`. Stops early at the tokenizer's end-of-sequence id.
/// With `carry_concept_state`, the concept state each step ends on is fused
/// into the next step's; without it every step rebuilds the concept stream.
pub fn generate(
    model: &RejRnm,
    tokenizer: &RejTokenizer,
    prompt: &str,
    controls: &[(String, f64)],
    max_new_tokens: usize,
    sampling: &Sampling,
    return_concept_trace: bool,
) -> Result<GenerationOutput, Error> {
    let device = model.device().clone();
    let mut ids = tokenizer.encode(prompt)?;
    let named = model.named_concepts().to_vec();
    let controls = match controls.is_empty() {
        true => None,
        false => Some(Tensor::new(named_vector(&named, controls)?, &device)?.unsqueeze(0)?),
    };
    let carry = model.config.carry_concept_state;
    let mut carried: Option<Tensor> = None;
    let mut steps: Vec<Tensor> = Vec::new();

    for _ in 0..max_new_tokens {
        let context = window(&ids, model.config.max_position_embeddings);
        let input = Tensor::new(context, &device)?.unsqueeze(0)?;
        let output = model.forward(
            &input,
            Pass {
                controls: controls.as_ref(),
                concept_state: carried.as_ref(),
                return_concept_trace,
                return_concept_state: carry,
                train: false,
            },
        )?;
        if carry {
            carried = output.concept_state.map(|state| state.detach());
        }
        if let Some(last) = output.concept_trace.and_then(|trace| trace.last().cloned()) {
            steps.push(last);
        }
        let token = next_token(&output.logits, sampling)?;
        ids.push(token);
        if tokenizer.eos_token_id() == Some(token) {
            break;
        }
    }

    Ok(GenerationOutput {
        text: tokenizer.decode(&ids)?,
        concept_trace: match return_concept_trace {
            true => Some(per_concept(&steps, &named)?),
            false => None,
        },
        token_ids: ids,
    })
}

pub(crate) fn per_concept(steps: &[Tensor], named: &[String]) -> Result<Vec<(String, Vec<Vec<f32>>)>, Error> {
    named
        .iter()
        .enumerate()
        .map(|(index, name)| -> Result<(String, Vec<Vec<f32>>), Error> {
            let states = steps
                .iter()
                .map(|step| -> Result<Vec<f32>, Error> { Ok(step.i((0, index))?.to_vec1::<f32>()?) })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok((name.clone(), states))
        })
        .collect()
}
