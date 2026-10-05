//! The command line: `train` and `generate`.
//!
//! Every setting that decides a run is a flag the caller states; the flags
//! without a value are the ones whose absence has a meaning of its own (no
//! clipping, no temperature, no checkpoint before the last).

mod generate;
mod inputs;
mod train;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use rej_1b::checkpoint::DeviceChoice;
use rej_1b::tokenizer::RejTokenizer;
use rej_1b::Error;

#[derive(Parser)]
#[command(name = "rej-1b", about = "Train Rej models and generate with concept controls")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a model from a configuration file.
    Train(train::TrainArgs),
    /// Generate text from a checkpoint.
    Generate(generate::GenerateArgs),
}

/// Where the model runs and how text becomes ids.
#[derive(Args, Clone)]
pub struct Runtime {
    /// cpu, cuda or metal; without it the first GPU this build has, else the CPU.
    #[arg(long, value_enum)]
    pub device: Option<DeviceChoice>,
    /// A Hugging Face tokenizer.json; without it the character tokenizer.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// The token that ends generation, for a tokenizer.json.
    #[arg(long, requires = "tokenizer")]
    pub eos_token: Option<String>,
    /// The token that pads a batch, for a tokenizer.json.
    #[arg(long, requires = "tokenizer")]
    pub pad_token: Option<String>,
}

impl Runtime {
    pub fn tokenizer(&self, vocab_size: usize) -> Result<RejTokenizer, Error> {
        match &self.tokenizer {
            Some(path) => RejTokenizer::pretrained(path, vocab_size, self.eos_token.as_deref(), self.pad_token.as_deref()),
            None => Ok(RejTokenizer::native(vocab_size)),
        }
    }
}

pub fn run() -> Result<(), Error> {
    match Cli::parse().command {
        Command::Train(args) => train::run(args),
        Command::Generate(args) => generate::run(args),
    }
}
