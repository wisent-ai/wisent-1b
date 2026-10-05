use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_core::{Device, IndexOp, Tensor};
use rej_1b::checkpoint::{Loaded, Model};
use rej_1b::model::Pass;
use rej_1b::model_v2::PassV2;
use rej_1b::tokenizer::RejTokenizer;
use serde_json::{json, Value};

pub struct Journey {
    pub root: PathBuf,
    pub config: Value,
    commands: usize,
    passed: bool,
}

fn git(args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .expect("read source identity");
    assert!(
        output.status.success(),
        "git: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

impl Journey {
    pub fn new(name: &str, v2: bool) -> Self {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/cli-journeys")
            .join(format!("{name}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("source.diff"), git(&["diff", "--binary", "HEAD"])).unwrap();
        fs::write(
            root.join("source.json"),
            serde_json::to_vec_pretty(&json!({
                "revision": git(&["rev-parse", "HEAD"]).trim(),
                "working_tree": git(&["status", "--porcelain"]),
                "binary_git_object": git(&["hash-object", env!("CARGO_BIN_EXE_rej-1b")]).trim(),
            }))
            .unwrap(),
        )
        .unwrap();
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
            root.join("Cargo.lock"),
        )
        .unwrap();
        let mut config: Value = serde_json::from_str(if v2 {
            include_str!("../../configs/rej_tiny_v2.json")
        } else {
            include_str!("../../configs/rej_tiny.json")
        })
        .unwrap();
        // Fixture dimensions keep real CPU training small; these are not product defaults.
        config["d_model"] = json!(16);
        config["n_layers"] = json!(1);
        config["n_heads"] = json!(2);
        config["d_head"] = json!(8);
        config["intermediate_size"] = json!(24);
        config["d_concept"] = json!(8);
        config["concept_heads"] = json!(2);
        config["concept_intermediate_size"] = json!(16);
        if v2 {
            config["concept_mlp_hidden"] = json!(16);
            config["steering_hidden"] = json!(16);
        }
        fs::write(root.join("corpus.txt"), "The sky is blue. The grass is green.\n").unwrap();
        Self {
            root,
            config,
            commands: 0,
            passed: false,
        }
    }

    pub fn training(&self, source: &str) -> Command {
        fs::write(
            self.root.join("config.json"),
            serde_json::to_vec_pretty(&self.config).unwrap(),
        )
        .unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_rej-1b"));
        command.current_dir(&self.root).args([
            "train",
            "--config",
            "config.json",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--num-steps",
            "2",
            "--learning-rate",
            "0.001",
            "--save-every",
            "1",
            "--output-dir",
            "checkpoints",
        ]);
        if source == "--data" {
            command.args([source, "corpus.txt", "--seq-length", "4"]);
        } else {
            command.args([source, "labels.jsonl"]);
        }
        command
    }

    pub fn generation(&self, step: usize) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_rej-1b"));
        command
            .current_dir(&self.root)
            .args(["generate", "--device", "cpu", "--checkpoint"])
            .arg(self.checkpoint(step))
            .args(["--prompt", "The", "--max-new-tokens", "1", "--greedy"]);
        if self.config.get("subspace_rank").is_some() {
            command.arg("--deterministic");
        }
        command
    }

    pub fn run(&mut self, mut command: Command, expected: i32) -> Output {
        self.commands += 1;
        let start = Instant::now();
        let output = command.output().expect("launch the real rej-1b binary");
        let prefix = self.root.join(format!("command-{}", self.commands));
        fs::write(prefix.with_extension("stdout"), &output.stdout).unwrap();
        fs::write(prefix.with_extension("stderr"), &output.stderr).unwrap();
        fs::write(
            prefix.with_extension("json"),
            serde_json::to_vec_pretty(&json!({
                "program": command.get_program().to_string_lossy(),
                "args": command.get_args().map(|arg| arg.to_string_lossy()).collect::<Vec<_>>(),
                "cwd": command.get_current_dir(),
                "exit_code": output.status.code(),
                "expected_exit_code": expected,
                "elapsed_ms": start.elapsed().as_millis(),
            }))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            output.status.code(),
            Some(expected),
            "{}: {}",
            self.root.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    pub fn checkpoint(&self, step: usize) -> PathBuf {
        self.root.join("checkpoints").join(format!("checkpoint_step_{step}"))
    }

    pub fn assert_updated_checkpoint(&self) {
        let before =
            candle_core::safetensors::load(self.checkpoint(1).join("model.safetensors"), &Device::Cpu).unwrap();
        let after = candle_core::safetensors::load(self.checkpoint(2).join("model.safetensors"), &Device::Cpu).unwrap();
        assert_eq!(before.len(), after.len());
        for (name, tensor) in &after {
            assert_eq!(tensor.dims(), before[name].dims(), "changed shape: {name}");
            assert!(
                tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite()),
                "non-finite weight: {name}"
            );
        }
        let values = |tensor: &Tensor| tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(
            values(&before["lm_head.weight"]),
            values(&after["lm_head.weight"]),
            "the second optimizer step did not update the language-model head"
        );
        let saved: Value = serde_json::from_slice(&fs::read(self.checkpoint(2).join("config.json")).unwrap()).unwrap();
        assert_eq!(saved, self.config, "checkpoint changed the declared model");
    }

    pub fn assert_greedy_reload(&mut self) {
        let loaded = Loaded::load(&self.checkpoint(2), &Device::Cpu).unwrap();
        let tokenizer = RejTokenizer::native(loaded.vocab_size());
        let mut ids = tokenizer.encode("The").unwrap();
        let tokens = Tensor::new(ids.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let logits = match &loaded.model {
            Model::V1(model) => model.forward(&tokens, Pass::default()).unwrap().logits,
            Model::V2(model) => {
                model
                    .forward(
                        &tokens,
                        PassV2 {
                            deterministic: true,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .logits
            }
        };
        // Read the trained model's logits directly, without using either generation helper.
        let scores = logits.i((0, ids.len() - 1)).unwrap().to_vec1::<f32>().unwrap();
        assert!(scores.iter().all(|v| v.is_finite()));
        let chosen = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        ids.push(chosen);
        let expected = format!("{}\n", tokenizer.decode(&ids).unwrap());
        let output = self.run(self.generation(2), 0);
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            expected,
            "greedy CLI decoding disagrees with the persisted model's highest logit"
        );
    }

    pub fn complete(&mut self) {
        self.passed = true;
    }
}

impl Drop for Journey {
    fn drop(&mut self) {
        let report = json!({ "passed": self.passed && !std::thread::panicking(), "commands": self.commands });
        let _ = fs::write(
            self.root.join("result.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        );
        eprintln!("CLI journey evidence: {}", self.root.display());
    }
}

pub fn steps(output: &Output) -> Vec<Value> {
    let rows: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows.iter().map(|row| row["step"].as_u64().unwrap()).collect::<Vec<_>>(),
        [1, 2]
    );
    rows
}
