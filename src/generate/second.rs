//! Decoding with the geometric model: the same prompt, plus controls that
//! name a direction in a concept subspace, an uncertainty or a selection.

use candle_core::Tensor;

use super::first::per_concept;
use super::{index_of, named_vector, next_token, window, GenerationOutput, Sampling};
use crate::model_v2::{GeometricControls, PassV2, RejRnmV2};
use crate::tokenizer::RejTokenizer;
use crate::Error;

/// Controls by concept name. Each list may be empty.
#[derive(Clone, Debug, Default)]
pub struct NamedControls {
    pub magnitude: Vec<(String, f64)>,
    /// A vector of `subspace_rank` coordinates per concept.
    pub direction: Vec<(String, Vec<f64>)>,
    pub uncertainty: Vec<(String, f64)>,
    pub select: Vec<(String, f64)>,
}

/// Generate up to `max_new_tokens` after `prompt` under `controls`. With
/// `deterministic`, each concept takes its mean instead of a draw.
#[allow(clippy::too_many_arguments)]
pub fn generate_v2(
    model: &RejRnmV2,
    tokenizer: &RejTokenizer,
    prompt: &str,
    controls: &NamedControls,
    max_new_tokens: usize,
    sampling: &Sampling,
    return_concept_trace: bool,
    deterministic: bool,
) -> Result<GenerationOutput, Error> {
    let named = model.named_concepts().to_vec();
    let controls = to_tensors(controls, &named, model.config.subspace_rank, model.device())?;
    let mut ids = tokenizer.encode(prompt)?;
    let mut steps: Vec<Tensor> = Vec::new();

    for _ in 0..max_new_tokens {
        let context = window(&ids, model.config.base.max_position_embeddings);
        let input = Tensor::new(context, model.device())?.unsqueeze(0)?;
        let output = model.forward(
            &input,
            PassV2 {
                controls: Some(&controls),
                return_concept_trace,
                deterministic,
                ..PassV2::default()
            },
        )?;
        if let Some(last) = output.concept_trace.and_then(|trace| trace.last().map(|step| step.mean.clone())) {
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

fn to_tensors(
    controls: &NamedControls,
    named: &[String],
    rank: usize,
    device: &candle_core::Device,
) -> Result<GeometricControls, Error> {
    let scalar = |values: &[(String, f64)]| -> Result<Option<Tensor>, Error> {
        match values.is_empty() {
            true => Ok(None),
            false => Ok(Some(Tensor::new(named_vector(named, values)?, device)?.unsqueeze(0)?)),
        }
    };
    let direction = match controls.direction.is_empty() {
        true => None,
        false => {
            let mut grid = vec![0.0f32; named.len() * rank];
            for (name, vector) in &controls.direction {
                if vector.len() != rank {
                    return Err(Error::Control(format!(
                        "direction for '{name}' has {} coordinates; this model's subspace rank is {rank}",
                        vector.len()
                    )));
                }
                let row = index_of(named, name)? * rank;
                for (offset, value) in vector.iter().enumerate() {
                    grid[row + offset] = *value as f32;
                }
            }
            Some(Tensor::from_vec(grid, (1, named.len(), rank), device)?)
        }
    };
    Ok(GeometricControls {
        magnitude: scalar(&controls.magnitude)?,
        direction,
        uncertainty: scalar(&controls.uncertainty)?,
        select: scalar(&controls.select)?,
    })
}
