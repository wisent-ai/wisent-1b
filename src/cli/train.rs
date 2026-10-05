//! `rej-1b train`.

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use clap::Args;

use rej_1b::checkpoint::{device, Loaded, Model};
use rej_1b::config::AnyConfig;
use rej_1b::generate::named_vector;
use rej_1b::tokenizer::RejTokenizer;
use rej_1b::train::{batches, padded, train, train_v2, CarrySchedule, OptimiserSettings, TokenDataset, Trainer, V2Batch};
use rej_1b::Error;

use super::inputs::{read_labelled, read_text, LabelledLine};
use super::Runtime;

/// What the run trains on; exactly one.
#[derive(Args)]
#[group(required = true, multiple = false)]
struct Source {
    /// A plain-text corpus for language modelling.
    #[arg(long, requires = "seq_length")]
    data: Option<PathBuf>,
    /// JSON Lines of {"text", "controls"} for concept-alignment training (v2).
    #[arg(long)]
    aligned: Option<PathBuf>,
    /// JSON Lines of {"text", "parallel", "controls"} for language-invariant training (v2).
    #[arg(long)]
    multilingual: Option<PathBuf>,
}

#[derive(Args)]
pub struct TrainArgs {
    /// The model's configuration file (see configs/).
    #[arg(long)]
    config: PathBuf,
    #[command(flatten)]
    source: Source,
    /// Ids per training sequence cut from --data.
    #[arg(long)]
    seq_length: Option<usize>,
    #[arg(long)]
    batch_size: usize,
    /// Steps to run; training also stops when one pass over the data ends.
    #[arg(long)]
    num_steps: usize,
    #[arg(long)]
    learning_rate: f64,
    /// AdamW's first-moment decay; candle's declared default without it.
    #[arg(long)]
    beta1: Option<f64>,
    /// AdamW's second-moment decay; candle's declared default without it.
    #[arg(long)]
    beta2: Option<f64>,
    /// AdamW's epsilon; candle's declared default without it.
    #[arg(long)]
    eps: Option<f64>,
    /// AdamW's weight decay; candle's declared default without it.
    #[arg(long)]
    weight_decay: Option<f64>,
    /// Clip the gradients to this total norm; no clipping without it.
    #[arg(long)]
    max_grad_norm: Option<f64>,
    /// Save a checkpoint every this many steps, as well as at the end.
    #[arg(long)]
    save_every: Option<usize>,
    /// Passes on a carried step (first model with carry_concept_state).
    #[arg(long, requires = "carry_every")]
    carry_passes: Option<usize>,
    /// Carry one step in this many.
    #[arg(long, requires = "carry_passes")]
    carry_every: Option<usize>,
    /// Draw random magnitude controls from N(0, scale²) on every plain v2 step.
    #[arg(long)]
    perturbation_scale: Option<f64>,
    #[arg(long)]
    output_dir: PathBuf,
    #[command(flatten)]
    runtime: Runtime,
}

pub fn run(args: TrainArgs) -> Result<(), Error> {
    let device = device(args.runtime.device)?;
    let loaded = Loaded::build(AnyConfig::from_file(&args.config)?, &device)?;
    eprintln!("model parameters: {}", loaded.parameter_count());
    let tokenizer = args.runtime.tokenizer(loaded.vocab_size())?;
    let mut trainer = Trainer::new(
        loaded.varmap.all_vars(),
        OptimiserSettings {
            learning_rate: args.learning_rate,
            beta1: args.beta1,
            beta2: args.beta2,
            eps: args.eps,
            weight_decay: args.weight_decay,
            max_grad_norm: args.max_grad_norm,
        },
    )?;
    let save = |step: usize| -> Result<(), Error> {
        if args.save_every.is_some_and(|every| every > 0 && step % every == 0) {
            eprintln!("saved {}", loaded.save(&args.output_dir, step)?.display());
        }
        Ok(())
    };

    let steps_done = match &loaded.model {
        Model::V1(model) => {
            if args.perturbation_scale.is_some() {
                return Err(Error::Training("--perturbation-scale applies to v2 configurations".into()));
            }
            let corpus = corpus_batches(&args, &tokenizer, &device)?;
            let carry = args
                .carry_passes
                .zip(args.carry_every)
                .map(|(passes, every)| CarrySchedule { passes, every });
            let losses = train(model, &mut trainer, corpus, args.num_steps, carry, |step, loss| {
                println!("{}", serde_json::json!({ "step": step, "loss": loss }));
                save(step)
            })?;
            losses.len()
        }
        Model::V2(model) => {
            if args.carry_passes.is_some() {
                return Err(Error::Training("--carry-passes applies to the first model, not v2".into()));
            }
            let named = model.named_concepts().to_vec();
            let source = &args.source;
            let v2_batches: Vec<Result<V2Batch, Error>> = match (&source.aligned, &source.multilingual) {
                (Some(path), _) => labelled_batches(&read_labelled(path)?, &args, &tokenizer, &named, &device, false)?,
                (_, Some(path)) => labelled_batches(&read_labelled(path)?, &args, &tokenizer, &named, &device, true)?,
                _ => corpus_batches(&args, &tokenizer, &device)?
                    .into_iter()
                    .map(|tokens| tokens.map(|tokens| V2Batch::Plain { tokens, perturbation_scale: args.perturbation_scale }))
                    .collect(),
            };
            let history = train_v2(model, &mut trainer, v2_batches, args.num_steps, |step, losses| {
                println!("{}", serde_json::json!({ "step": step, "losses": losses }));
                save(step)
            })?;
            history.len()
        }
    };
    let dir = loaded.save(&args.output_dir, steps_done)?;
    eprintln!("trained {steps_done} steps; final checkpoint {}", dir.display());
    Ok(())
}

/// The corpus cut into sequences, shuffled and padded into batches.
fn corpus_batches(args: &TrainArgs, tokenizer: &RejTokenizer, device: &Device) -> Result<Vec<Result<Tensor, Error>>, Error> {
    let (Some(path), Some(seq_length)) = (&args.source.data, args.seq_length) else {
        return Err(Error::Training("this model trains on --data with --seq-length".into()));
    };
    let ids = tokenizer.encode(&read_text(path)?)?;
    let dataset = TokenDataset::new(&ids, seq_length)?;
    Ok(batches(&dataset.samples, args.batch_size)?
        .iter()
        .map(|batch| padded(batch, tokenizer.pad_token_id(), device))
        .collect())
}

/// Labelled lines grouped into aligned or multilingual batches.
fn labelled_batches(
    lines: &[LabelledLine],
    args: &TrainArgs,
    tokenizer: &RejTokenizer,
    named: &[String],
    device: &Device,
    multilingual: bool,
) -> Result<Vec<Result<V2Batch, Error>>, Error> {
    if args.batch_size == 0 {
        return Err(Error::Training("batch_size must be at least one".into()));
    }
    let batch = |group: &[LabelledLine]| -> Result<V2Batch, Error> {
        let texts = group.iter().map(|line| tokenizer.encode(&line.text)).collect::<Result<Vec<_>, _>>()?;
        let mut controls = Vec::with_capacity(group.len() * named.len());
        for line in group {
            let values: Vec<(String, f64)> = line.controls.iter().map(|(n, v)| (n.clone(), *v)).collect();
            controls.extend(named_vector(named, &values)?);
        }
        let controls = Tensor::from_vec(controls, (group.len(), named.len()), device)?;
        let tokens = padded(&texts, tokenizer.pad_token_id(), device)?;
        if !multilingual {
            return Ok(V2Batch::Aligned { tokens, controls });
        }
        let parallel = group
            .iter()
            .map(|line| -> Result<Vec<u32>, Error> {
                let text = line.parallel.as_deref().ok_or_else(|| {
                    Error::Training(format!("multilingual line '{}' has no parallel text", line.text))
                })?;
                tokenizer.encode(text)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let second = padded(&parallel, tokenizer.pad_token_id(), device)?;
        Ok(V2Batch::Multilingual { first: tokens, second, controls })
    };
    Ok(lines.chunks(args.batch_size).map(batch).collect())
}
