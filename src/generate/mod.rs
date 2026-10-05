//! Decoding with concept controls: the first model takes a magnitude per
//! named concept, the geometric model a magnitude, direction, uncertainty or
//! selection.
//!
//! How many tokens to generate is the caller's. Sampling uses the logits as
//! they are unless the caller names a temperature, a top-k or a top-p.

mod first;
mod second;

pub use first::generate;
pub use second::{generate_v2, NamedControls};

use candle_core::{IndexOp, Tensor};
use rand::Rng;

use crate::Error;

/// How the next token is chosen.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sampling {
    /// Divide the logits by this before choosing.
    pub temperature: Option<f64>,
    /// Keep only the k most likely tokens.
    pub top_k: Option<usize>,
    /// Keep the smallest set of most likely tokens whose probability reaches p.
    pub top_p: Option<f64>,
    /// Take the most likely token instead of drawing one.
    pub greedy: bool,
}

/// What decoding produced.
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    /// Per named concept, its last-layer state at every generated step.
    pub concept_trace: Option<Vec<(String, Vec<Vec<f32>>)>>,
}

/// The next token from the last position of `logits` (1, T, vocab).
pub(crate) fn next_token(logits: &Tensor, sampling: &Sampling) -> Result<u32, Error> {
    let len = logits.dim(1)?;
    let mut scores: Vec<f32> = logits.i((0, len - 1))?.to_vec1()?;
    if let Some(temperature) = sampling.temperature {
        if temperature <= 0.0 {
            return Err(Error::Control(format!("temperature {temperature} must be above zero")));
        }
        scores.iter_mut().for_each(|score| *score /= temperature as f32);
    }
    if let Some(k) = sampling.top_k.filter(|&k| k > 0 && k < scores.len()) {
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[k - 1];
        scores.iter_mut().filter(|score| **score < kth).for_each(|score| *score = f32::NEG_INFINITY);
    }
    if let Some(p) = sampling.top_p.filter(|&p| p > 0.0) {
        keep_nucleus(&mut scores, p);
    }
    let probabilities = softmax(&scores);
    if sampling.greedy {
        return Ok(argmax(&probabilities));
    }
    let draw: f32 = rand::rng().random();
    let mut cumulative = 0.0;
    for (id, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return Ok(id as u32);
        }
    }
    Ok(argmax(&probabilities))
}

/// Remove every token outside the smallest most-likely set whose cumulative
/// probability exceeds p; the most likely token always stays.
fn keep_nucleus(scores: &mut [f32], p: f64) {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let sorted: Vec<f32> = order.iter().map(|&id| scores[id]).collect();
    let probabilities = softmax(&sorted);
    let mut cumulative = 0.0f64;
    for (rank, &id) in order.iter().enumerate() {
        let before = cumulative;
        cumulative += probabilities[rank] as f64;
        if rank > 0 && before > p {
            scores[id] = f32::NEG_INFINITY;
        }
    }
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(id, _)| id as u32)
}

/// One value per named concept, in the model's order; a name the model does
/// not have is refused.
pub fn named_vector(named: &[String], values: &[(String, f64)]) -> Result<Vec<f32>, Error> {
    let mut vector = vec![0.0f32; named.len()];
    for (name, value) in values {
        let index = index_of(named, name)?;
        vector[index] = *value as f32;
    }
    Ok(vector)
}

/// The position of concept `name` among the model's named concepts.
pub fn index_of(named: &[String], name: &str) -> Result<usize, Error> {
    named.iter().position(|known| known == name).ok_or_else(|| {
        Error::Control(format!("unknown concept '{name}'; this model names {named:?}"))
    })
}

/// The ids the model sees at the next step: the last `max_positions` of `ids`.
pub(crate) fn window(ids: &[u32], max_positions: usize) -> &[u32] {
    &ids[ids.len().saturating_sub(max_positions)..]
}
