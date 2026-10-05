//! A model together with its variables, the device it runs on, and the files
//! it is saved to.
//!
//! A checkpoint is a directory `checkpoint_step_<N>/` holding `config.json`
//! (the configuration the model was built from) and `model.safetensors` (every
//! trainable variable by name). Loading reads the configuration first, builds
//! the model it describes, then fills its variables. The optimiser's moments
//! are not saved: a run resumed from a checkpoint starts them again.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::ValueEnum;

use crate::config::AnyConfig;
use crate::model::RejRnm;
use crate::model_v2::RejRnmV2;
use crate::Error;

const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Clone, Debug)]
pub enum Model {
    V1(RejRnm),
    V2(RejRnmV2),
}

/// A model, the variables it trains, and its device.
#[derive(Clone)]
pub struct Loaded {
    pub model: Model,
    pub varmap: VarMap,
    pub device: Device,
}

impl Loaded {
    /// A freshly initialised model for `config`.
    pub fn build(config: AnyConfig, device: &Device) -> Result<Self, Error> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = match config {
            AnyConfig::V1(config) => Model::V1(RejRnm::new(config, vb)?),
            AnyConfig::V2(config) => Model::V2(RejRnmV2::new(config, vb)?),
        };
        Ok(Self { model, varmap, device: device.clone() })
    }

    /// The model saved in checkpoint directory `dir`.
    pub fn load(dir: &Path, device: &Device) -> Result<Self, Error> {
        let mut loaded = Self::build(AnyConfig::from_file(&dir.join(CONFIG_FILE))?, device)?;
        loaded.varmap.load(dir.join(WEIGHTS_FILE))?;
        Ok(loaded)
    }

    /// Save as `output_dir/checkpoint_step_<step>/` and return that directory.
    pub fn save(&self, output_dir: &Path, step: usize) -> Result<PathBuf, Error> {
        let dir = output_dir.join(format!("checkpoint_step_{step}"));
        std::fs::create_dir_all(&dir).map_err(|source| Error::Io { path: dir.clone(), source })?;
        let config = match &self.model {
            Model::V1(model) => serde_json::to_string_pretty(&model.config),
            Model::V2(model) => serde_json::to_string_pretty(&model.config),
        }
        .map_err(|e| Error::Config(e.to_string()))?;
        let config_path = dir.join(CONFIG_FILE);
        std::fs::write(&config_path, config).map_err(|source| Error::Io { path: config_path, source })?;
        self.varmap.save(dir.join(WEIGHTS_FILE))?;
        Ok(dir)
    }

    /// The number of trainable values.
    pub fn parameter_count(&self) -> usize {
        self.varmap.all_vars().iter().map(|var| var.elem_count()).sum()
    }

    pub fn named_concepts(&self) -> &[String] {
        match &self.model {
            Model::V1(model) => model.named_concepts(),
            Model::V2(model) => model.named_concepts(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match &self.model {
            Model::V1(model) => model.config.vocab_size,
            Model::V2(model) => model.config.base.vocab_size,
        }
    }

    pub fn max_positions(&self) -> usize {
        match &self.model {
            Model::V1(model) => model.config.max_position_embeddings,
            Model::V2(model) => model.config.base.max_position_embeddings,
        }
    }
}

/// Where tensors live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DeviceChoice {
    Cpu,
    Cuda,
    Metal,
}

/// The device the caller named, or the first GPU this build and machine
/// have, or the CPU.
pub fn device(choice: Option<DeviceChoice>) -> Result<Device, Error> {
    let first = 0;
    Ok(match choice {
        Some(DeviceChoice::Cpu) => Device::Cpu,
        Some(DeviceChoice::Cuda) => Device::new_cuda(first)?,
        Some(DeviceChoice::Metal) => Device::new_metal(first)?,
        None if candle_core::utils::cuda_is_available() => Device::new_cuda(first)?,
        None if candle_core::utils::metal_is_available() => Device::new_metal(first)?,
        None => Device::Cpu,
    })
}
