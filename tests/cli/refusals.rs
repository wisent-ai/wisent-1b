use std::fs;

use serde_json::json;

use super::support::Journey;

#[test]
fn carry_without_a_training_schedule_leaves_no_checkpoint() {
    let mut journey = Journey::new("refuse-missing-carry-schedule", false);
    journey.config["carry_concept_state"] = json!(true);
    let output = journey.run(journey.training("--data"), 1);
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("carry") && error.contains("schedule"), "{error}");
    assert!(output.stdout.is_empty(), "refused training reported completed steps");
    assert!(!journey.root.join("checkpoints").exists());
    journey.complete();
}

#[test]
fn missing_parallel_text_is_not_treated_as_a_translation() {
    let mut journey = Journey::new("refuse-missing-parallel-text", true);
    journey.config["use_language_invariant_concepts"] = json!(true);
    fs::write(
        journey.root.join("labels.jsonl"),
        json!({
            "text": "The sky is blue.", "controls": {}
        })
        .to_string(),
    )
    .unwrap();
    let output = journey.run(journey.training("--multilingual"), 1);
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("parallel"), "{error}");
    assert!(output.stdout.is_empty(), "refused training reported completed steps");
    assert!(!journey.root.join("checkpoints").exists());
    journey.complete();
}

#[test]
fn conflicting_training_sources_are_refused_before_training() {
    let mut journey = Journey::new("refuse-conflicting-sources", true);
    let mut command = journey.training("--data");
    command.args(["--aligned", "labels.jsonl"]);
    let output = journey.run(command, 2);
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("--data") && error.contains("--aligned"), "{error}");
    assert!(!journey.root.join("checkpoints").exists());
    journey.complete();
}

#[test]
fn unknown_concept_and_wrong_model_controls_do_not_generate_text() {
    let mut journey = Journey::new("refuse-invalid-controls", false);
    journey.run(journey.training("--data"), 0);
    let mut command = journey.generation(2);
    command.args(["--control", "nonexistent-concept=1"]);
    let output = journey.run(command, 1);
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("nonexistent-concept") && error.contains("truthfulness"),
        "{error}"
    );
    assert!(output.stdout.is_empty(), "unknown control was silently ignored");

    let mut command = journey.generation(2);
    command.args(["--direction", "truthfulness=1,0,0,0"]);
    let output = journey.run(command, 1);
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("--direction") && error.contains("v2"), "{error}");
    assert!(output.stdout.is_empty(), "v2 control was silently ignored by v1");
    journey.complete();
}
