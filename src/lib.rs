//! Rej-1B: a representation-native language model with explicit concept control.
//!
//! * [`model::RejRnm`] carries a token stream and a concept stream of vectors.
//! * [`model_v2::RejRnmV2`] makes each concept a subspace with a Gaussian state.
//! * [`train`] trains either model; [`generate`] decodes with concept controls.
//! * [`checkpoint`] saves and loads a model with the configuration it was built from.
//!
//! Every setting a run depends on is stated by the run: the model's shape and
//! loss weights by its configuration file, the optimiser, schedule and
//! sampling by the caller. Nothing here assumes one.

pub mod checkpoint;
pub mod config;
pub mod generate;
pub mod model;
pub mod model_v2;
pub mod tokenizer;
pub mod train;

use std::path::PathBuf;

/// Why an operation did not produce its result.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration refused: {0}")]
    Config(String),
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("tensor operation failed: {0}")]
    Tensor(#[from] candle_core::Error),
    #[error("tokenizer {}: {message}", path.display())]
    Tokenizer { path: PathBuf, message: String },
    #[error("control refused: {0}")]
    Control(String),
    #[error("training refused: {0}")]
    Training(String),
}
