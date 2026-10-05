//! The geometric model: each concept is a rank-r subspace with a basis and a
//! centroid, its state a Gaussian over subspace coordinates. Non-linear cells
//! read, update and write concepts; a router decides which tokens each
//! concept reads; a learned steering manifold shifts named concepts per
//! input; an alignment head predicts the controls a step injected.

mod cell;
mod concepts;
mod network;
mod steering;

use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

pub use cell::NonlinearConceptCell;
pub use concepts::{orthonormalize, ConceptRouter, SubspaceConceptBank};
pub use network::{OutputV2, PassV2, RejRnmV2, TraceStep};
pub use steering::{ConceptAlignmentHead, SteeringManifold};

use crate::config::RejConfigV2;
use crate::Error;

/// A way a caller may move a named concept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMode {
    /// Scales a named concept's mean by (1 + value).
    Magnitude,
    /// Adds a vector in subspace coordinates to a named concept's mean.
    Direction,
    /// Adds to a named concept's log standard deviation.
    Uncertainty,
    /// Multiplies a named concept's mean by sigmoid(value).
    Select,
}

/// Which control modes a model accepts, one flag per mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllowedControls {
    pub magnitude: bool,
    pub direction: bool,
    pub uncertainty: bool,
    pub select: bool,
}

impl AllowedControls {
    pub fn allows(&self, mode: ControlMode) -> bool {
        match mode {
            ControlMode::Magnitude => self.magnitude,
            ControlMode::Direction => self.direction,
            ControlMode::Uncertainty => self.uncertainty,
            ControlMode::Select => self.select,
        }
    }
}

/// Controls for one batch. Each scalar mode is (B, n_named); a direction is
/// (B, n_named, subspace_rank).
#[derive(Clone, Debug, Default)]
pub struct GeometricControls {
    pub magnitude: Option<Tensor>,
    pub direction: Option<Tensor>,
    pub uncertainty: Option<Tensor>,
    pub select: Option<Tensor>,
}

impl GeometricControls {
    /// The controls with every tensor in batch form, f32, refused when a mode
    /// the configuration does not allow is set or a shape does not fit.
    pub fn validated(&self, config: &RejConfigV2) -> Result<Self, Error> {
        let named = config.base.n_named_concepts();
        let scalar = |mode: ControlMode, value: &Option<Tensor>| -> Result<Option<Tensor>, Error> {
            let Some(value) = value else { return Ok(None) };
            allowed(config, mode)?;
            let value = batch_form(value, 1)?;
            if value.rank() != 2 || value.dim(1)? != named {
                return Err(Error::Control(format!(
                    "control {mode:?} of shape {:?} must be (batch, {named})",
                    value.dims()
                )));
            }
            Ok(Some(value))
        };
        let direction = match &self.direction {
            None => None,
            Some(value) => {
                allowed(config, ControlMode::Direction)?;
                let value = batch_form(value, 2)?;
                let rank = config.subspace_rank;
                if value.rank() != 3 || value.dims()[1..] != [named, rank] {
                    return Err(Error::Control(format!(
                        "control Direction of shape {:?} must be (batch, {named}, {rank})",
                        value.dims()
                    )));
                }
                Some(value)
            }
        };
        Ok(Self {
            magnitude: scalar(ControlMode::Magnitude, &self.magnitude)?,
            direction,
            uncertainty: scalar(ControlMode::Uncertainty, &self.uncertainty)?,
            select: scalar(ControlMode::Select, &self.select)?,
        })
    }
}

fn allowed(config: &RejConfigV2, mode: ControlMode) -> Result<(), Error> {
    match config.allowed_controls.allows(mode) {
        true => Ok(()),
        false => Err(Error::Control(format!(
            "control mode {mode:?} is not allowed by this model's allowed_controls {:?}",
            config.allowed_controls
        ))),
    }
}

/// `value` with a batch dimension added when it has only `unbatched_rank` dims.
fn batch_form(value: &Tensor, unbatched_rank: usize) -> Result<Tensor, Error> {
    let value = match value.rank() == unbatched_rank {
        true => value.unsqueeze(0)?,
        false => value.clone(),
    };
    Ok(value.to_dtype(DType::F32)?)
}
