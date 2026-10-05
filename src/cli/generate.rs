//! `rej-1b generate`.

use std::path::PathBuf;

use clap::Args;

use rej_1b::checkpoint::{device, Loaded, Model};
use rej_1b::generate::{generate, generate_v2, NamedControls, Sampling};
use rej_1b::Error;

use super::inputs::{named_value, named_vector};
use super::Runtime;

#[derive(Args)]
pub struct GenerateArgs {
    /// A checkpoint directory written by `rej-1b train`.
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    prompt: String,
    /// Tokens to generate at most; generation also stops at end-of-sequence.
    #[arg(long)]
    max_new_tokens: usize,
    /// A magnitude per named concept, `name=value`; repeat for more.
    #[arg(long = "control", value_parser = named_value)]
    controls: Vec<(String, f64)>,
    /// v2: a direction in a concept's subspace, `name=v1,v2,...`.
    #[arg(long = "direction", value_parser = named_vector)]
    directions: Vec<(String, Vec<f64>)>,
    /// v2: added to a concept's log standard deviation, `name=value`.
    #[arg(long = "uncertainty", value_parser = named_value)]
    uncertainty: Vec<(String, f64)>,
    /// v2: soft selection of a concept, `name=value`.
    #[arg(long = "select", value_parser = named_value)]
    select: Vec<(String, f64)>,
    /// Divide the logits by this before sampling.
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f64>,
    /// Take the most likely token at every step.
    #[arg(long)]
    greedy: bool,
    /// v2: use each concept's mean instead of a draw.
    #[arg(long)]
    deterministic: bool,
    /// Print each named concept's last-layer state norm, averaged over the steps.
    #[arg(long)]
    trace: bool,
    #[command(flatten)]
    runtime: Runtime,
}

pub fn run(args: GenerateArgs) -> Result<(), Error> {
    let device = device(args.runtime.device)?;
    let loaded = Loaded::load(&args.checkpoint, &device)?;
    let tokenizer = args.runtime.tokenizer(loaded.vocab_size())?;
    let sampling = Sampling {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        greedy: args.greedy,
    };
    let output = match &loaded.model {
        Model::V1(model) => {
            let v2_only = !args.directions.is_empty() || !args.uncertainty.is_empty() || !args.select.is_empty();
            if v2_only || args.deterministic {
                return Err(Error::Control(
                    "--direction, --uncertainty, --select and --deterministic apply to v2 checkpoints".into(),
                ));
            }
            if model.config.carry_concept_state {
                eprintln!("cross-step concept carry: on");
            }
            generate(model, &tokenizer, &args.prompt, &args.controls, args.max_new_tokens, &sampling, args.trace)?
        }
        Model::V2(model) => {
            let controls = NamedControls {
                magnitude: args.controls.clone(),
                direction: args.directions.clone(),
                uncertainty: args.uncertainty.clone(),
                select: args.select.clone(),
            };
            generate_v2(
                model,
                &tokenizer,
                &args.prompt,
                &controls,
                args.max_new_tokens,
                &sampling,
                args.trace,
                args.deterministic,
            )?
        }
    };
    println!("{}", output.text);
    for (name, steps) in output.concept_trace.iter().flatten() {
        let norms: Vec<f32> = steps.iter().map(|state| state.iter().map(|v| v * v).sum::<f32>().sqrt()).collect();
        match norms.is_empty() {
            true => eprintln!("{name}: no steps"),
            false => eprintln!("{name}: mean norm {:.4}", norms.iter().sum::<f32>() / norms.len() as f32),
        }
    }
    Ok(())
}
