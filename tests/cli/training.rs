mod refusals;
mod support;

use std::fs;

use serde_json::{json, Value};
use support::{steps, Journey};

fn language_model_journey(name: &str, v2: bool) {
    let mut journey = Journey::new(name, v2);
    let output = journey.run(journey.training("--data"), 0);
    for row in steps(&output) {
        if v2 {
            let losses = &row["losses"];
            assert_v2_total(losses, &journey.config);
            assert!(losses["kl_loss"].as_f64().unwrap().is_finite());
            assert!(losses["geometry_loss"].as_f64().unwrap().is_finite());
            assert!(losses["align_loss"].is_null());
            assert!(losses["inv_loss"].is_null());
        } else {
            assert!(row["loss"].as_f64().unwrap().is_finite());
        }
    }
    journey.assert_updated_checkpoint();
    journey.assert_greedy_reload();
    journey.complete();
}

#[test]
fn first_model_trains_saves_and_generates_from_its_weights() {
    language_model_journey("v1-language-model", false);
}

#[test]
fn geometric_model_trains_without_fabricated_alignment_measurements() {
    language_model_journey("v2-language-model", true);
}

fn assert_v2_total(losses: &Value, config: &Value) {
    let mut expected = losses["lm_loss"].as_f64().unwrap();
    assert!(expected.is_finite());
    for (loss, weight) in [
        ("kl_loss", "kl_weight"),
        ("geometry_loss", "geometry_weight"),
        ("align_loss", "alignment_weight"),
        ("inv_loss", "language_invariant_weight"),
    ] {
        if let Some(value) = losses[loss].as_f64() {
            assert!(value.is_finite(), "non-finite {loss}");
            expected += value * config[weight].as_f64().unwrap();
        } else {
            assert!(losses[loss].is_null(), "invalid {loss}: {}", losses[loss]);
        }
    }
    let actual = losses["total_loss"].as_f64().unwrap();
    // Each component and each tensor addition is rounded to f32; compare at that precision.
    let tolerance = f64::from(f32::EPSILON) * 16.0 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "reported total {actual} does not include the weighted terms {expected}"
    );
}

fn labelled_journey(multilingual: bool) {
    let mut journey = Journey::new(if multilingual { "v2-multilingual" } else { "v2-aligned" }, true);
    journey.config["use_language_invariant_concepts"] = json!(multilingual);
    let rows = [
        json!({"text": "The sky is blue.", "parallel": "Niebo jest niebieskie.", "controls": {"truthfulness": 1.0}}),
        json!({"text": "Grass is green.", "parallel": "Trawa jest zielona.", "controls": {"truthfulness": -1.0}}),
    ];
    let text = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
    fs::write(journey.root.join("labels.jsonl"), text).unwrap();
    let output = journey.run(
        journey.training(if multilingual { "--multilingual" } else { "--aligned" }),
        0,
    );
    for row in steps(&output) {
        let losses = &row["losses"];
        assert_v2_total(losses, &journey.config);
        assert!(losses["align_loss"].as_f64().unwrap().is_finite());
        if multilingual {
            assert!(losses["inv_loss"].as_f64().unwrap().is_finite());
        } else {
            assert!(losses["inv_loss"].is_null());
        }
    }
    journey.assert_updated_checkpoint();
    journey.assert_greedy_reload();
    journey.complete();
}

#[test]
fn aligned_training_measures_alignment_but_not_invariance() {
    labelled_journey(false);
}

#[test]
fn parallel_text_training_measures_alignment_and_invariance() {
    labelled_journey(true);
}

#[test]
fn carried_training_accepts_a_schedule_that_reaches_the_carry_pass() {
    let mut journey = Journey::new("v1-carry", false);
    journey.config["carry_concept_state"] = json!(true);
    let mut command = journey.training("--data");
    command.args(["--carry-passes", "2", "--carry-every", "1"]);
    let output = journey.run(command, 0);
    for row in steps(&output) {
        assert!(row["loss"].as_f64().unwrap().is_finite());
    }
    journey.assert_updated_checkpoint();
    journey.assert_greedy_reload();
    journey.complete();
}

#[test]
fn exhausted_corpus_reports_and_saves_only_the_completed_steps() {
    let mut journey = Journey::new("v1-data-exhaustion", false);
    // One five-token sequence at stride four, despite a two-step request.
    fs::write(journey.root.join("corpus.txt"), "abcde").unwrap();
    let output = journey.run(journey.training("--data"), 0);
    let rows: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["step"], 1);
    assert!(rows[0]["loss"].as_f64().unwrap().is_finite());
    let loaded = rej_1b::checkpoint::Loaded::load(&journey.checkpoint(1), &candle_core::Device::Cpu).unwrap();
    assert_eq!(
        loaded.vocab_size(),
        journey.config["vocab_size"].as_u64().unwrap() as usize
    );
    assert!(!journey.checkpoint(2).exists(), "claimed a step that never ran");
    journey.complete();
}
