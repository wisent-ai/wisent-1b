//! A model's configuration, read from the JSON file the run declares.
//!
//! Every field is required. The model has no built-in preset and no derived
//! default: a width, a head count, a dropout or a loss weight is the value the
//! configuration file states, and a file missing one is refused by serde with
//! the field's name. `configs/` holds the declared configurations.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model_v2::AllowedControls;
use crate::Error;

/// The first model: a token stream and a concept stream of vectors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RejConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Width of one attention head.
    pub d_head: usize,
    /// Hidden width of the token feed-forward.
    pub intermediate_size: usize,
    pub dropout: f32,
    pub n_concepts: usize,
    pub d_concept: usize,
    /// Heads of the two attentions inside the concept stream.
    pub concept_heads: usize,
    /// Hidden width of the concept feed-forward.
    pub concept_intermediate_size: usize,
    /// The first slots of the concept stream, by name; their count is the
    /// number of named concepts.
    pub named_concepts: Vec<String>,
    /// Fuse the concept state a decoding step ends on into the next step's.
    pub carry_concept_state: bool,
    pub layer_norm_eps: f64,
    /// Standard deviation of the normal initialisation of embeddings.
    pub initializer_range: f64,
    pub tie_word_embeddings: bool,
}

/// The geometric model: concepts are subspaces with a Gaussian state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RejConfigV2 {
    #[serde(flatten)]
    pub base: RejConfig,
    pub subspace_rank: usize,
    pub normalize_subspace_basis: bool,
    pub probabilistic_concepts: bool,
    /// Standard deviation of the initial log standard deviation of concepts.
    pub concept_log_std_init_range: f64,
    pub kl_weight: f64,
    pub concept_mlp_hidden: usize,
    pub use_concept_router: bool,
    pub use_manifold_decoder: bool,
    /// Which control modes a caller may use on this model.
    pub allowed_controls: AllowedControls,
    pub use_concept_alignment: bool,
    pub alignment_weight: f64,
    pub use_titan_manifold: bool,
    pub n_titan_directions: usize,
    /// Hidden width of the steering manifold's intensity network.
    pub steering_hidden: usize,
    pub use_geometry_regularization: bool,
    pub geometry_weight: f64,
    pub use_language_invariant_concepts: bool,
    pub language_invariant_weight: f64,
}

impl RejConfig {
    pub fn n_named_concepts(&self) -> usize {
        self.named_concepts.len()
    }

    /// Refuse a configuration whose parts do not fit together.
    pub fn validate(&self) -> Result<(), Error> {
        if self.n_named_concepts() > self.n_concepts {
            return Err(Error::Config(format!(
                "named_concepts has {} names but n_concepts is {}",
                self.n_named_concepts(),
                self.n_concepts
            )));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(Error::Config(format!(
                "dropout {} is not in [0, 1)",
                self.dropout
            )));
        }
        Ok(())
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let config: Self = read_json(path)?;
        config.validate()?;
        Ok(config)
    }
}

impl RejConfigV2 {
    pub fn validate(&self) -> Result<(), Error> {
        self.base.validate()?;
        if self.base.carry_concept_state {
            return Err(Error::Config(
                "carry_concept_state is implemented for the first model, not v2: v2 carries a \
                 probabilistic concept state and needs its own fusion rule"
                    .into(),
            ));
        }
        if self.subspace_rank > self.base.d_concept {
            return Err(Error::Config(format!(
                "subspace_rank {} exceeds d_concept {}",
                self.subspace_rank, self.base.d_concept
            )));
        }
        Ok(())
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let config: Self = read_json(path)?;
        config.validate()?;
        Ok(config)
    }
}

/// Which model a configuration file describes: a v2 file names a subspace rank.
#[derive(Clone, Debug)]
pub enum AnyConfig {
    V1(RejConfig),
    V2(RejConfigV2),
}

impl AnyConfig {
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let value: serde_json::Value = read_json(path)?;
        if value.get("subspace_rank").is_some() {
            Ok(Self::V2(RejConfigV2::from_file(path)?))
        } else {
            Ok(Self::V1(RejConfig::from_file(path)?))
        }
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|error| {
        Error::Config(format!("{}: {error}", path.display()))
    })
}
