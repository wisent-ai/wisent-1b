//! What a training step reads: token sequences cut from a corpus, the
//! padding that makes a batch, and the next-token loss.

use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;

use crate::Error;

/// Sequences of `seq_length + 1` ids cut from a corpus every `seq_length`
/// ids, so each sequence's last id is the next one's first target.
#[derive(Clone, Debug)]
pub struct TokenDataset {
    pub samples: Vec<Vec<u32>>,
}

impl TokenDataset {
    pub fn new(ids: &[u32], seq_length: usize) -> Result<Self, Error> {
        if seq_length == 0 {
            return Err(Error::Training("seq_length must be at least one".into()));
        }
        let samples = (0..ids.len().saturating_sub(seq_length).max(1))
            .step_by(seq_length)
            .map(|start| ids[start..(start + seq_length + 1).min(ids.len())].to_vec())
            .filter(|chunk| chunk.len() >= 2)
            .collect();
        Ok(Self { samples })
    }
}

/// The samples shuffled and grouped `batch_size` at a time; each group is one
/// padded batch.
pub fn batches(samples: &[Vec<u32>], batch_size: usize) -> Result<Vec<Vec<Vec<u32>>>, Error> {
    if batch_size == 0 {
        return Err(Error::Training("batch_size must be at least one".into()));
    }
    let mut order: Vec<&Vec<u32>> = samples.iter().collect();
    order.shuffle(&mut rand::rng());
    Ok(order
        .chunks(batch_size)
        .map(|chunk| chunk.iter().map(|sample| (*sample).clone()).collect())
        .collect())
}

/// `sequences` as one (B, T) u32 tensor, shorter ones padded with `pad`.
/// Sequences of different lengths without a pad id are refused.
pub fn padded(sequences: &[Vec<u32>], pad: Option<u32>, device: &Device) -> Result<Tensor, Error> {
    let width = sequences.iter().map(Vec::len).max().unwrap_or(0);
    let mut flat = Vec::with_capacity(sequences.len() * width);
    for sequence in sequences {
        flat.extend_from_slice(sequence);
        if sequence.len() < width {
            let pad = pad.ok_or_else(|| {
                Error::Training("a batch needs padding but the tokenizer has no pad token".into())
            })?;
            flat.extend(std::iter::repeat_n(pad, width - sequence.len()));
        }
    }
    Ok(Tensor::from_vec(flat, (sequences.len(), width), device)?)
}

/// Mean next-token cross-entropy: position t predicts the id at t + 1.
pub fn lm_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor, Error> {
    let (batch, len, vocab) = logits.dims3()?;
    if len < 2 {
        return Err(Error::Training("a sequence needs at least two ids to have a target".into()));
    }
    let predicted = logits.narrow(1, 0, len - 1)?.reshape((batch * (len - 1), vocab))?;
    let targets = labels.narrow(1, 1, len - 1)?.reshape(batch * (len - 1))?;
    Ok(candle_nn::loss::cross_entropy(&predicted, &targets)?)
}
