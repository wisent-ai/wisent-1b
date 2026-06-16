"""Tests for RejRNM architecture."""
import torch

from rej_1b.config import rej_tiny_config
from rej_1b.model import RejRNM


def test_forward_shape():
    config = rej_tiny_config()
    model = RejRNM(config)
    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    outputs = model(input_ids)
    assert outputs["logits"].shape == (B, T, config.vocab_size)
    assert outputs["concept_trace"] is None


def test_forward_with_concept_trace():
    config = rej_tiny_config()
    model = RejRNM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids, return_concept_trace=True)
    trace = outputs["concept_trace"]
    assert trace is not None
    assert len(trace) == config.n_layers
    for layer_state in trace:
        assert layer_state.shape == (1, config.n_concepts, config.d_concept)


def test_control_intervention():
    config = rej_tiny_config()
    model = RejRNM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    controls = torch.zeros(1, config.n_named_concepts)
    controls[0, 0] = 2.0  # boost first named concept
    outputs = model(input_ids, controls=controls)
    assert outputs["logits"].shape == (1, 8, config.vocab_size)


def test_unknown_control_raises():
    config = rej_tiny_config()
    model = RejRNM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    # _build_concept_control validates against n_named_concepts, not names.
    controls = torch.zeros(1, config.n_named_concepts + 1)
    try:
        model(input_ids, controls=controls)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_named_concept_labels():
    config = rej_tiny_config()
    model = RejRNM(config)
    assert model.named_concept_labels == config.named_concepts
