"""Tests for controlled generation."""
import torch

from rey_1b.config import rey_tiny_config
from rey_1b.generate import generate, _controls_to_tensor
from rey_1b.model import ReyRNM
from rey_1b.tokenizer import ReyTokenizer


def test_controls_to_tensor():
    named = ["truthfulness", "refusal"]
    controls = {"truthfulness": 1.5, "refusal": -0.5}
    tensor = _controls_to_tensor(controls, named, torch.device("cpu"))
    assert tensor is not None
    assert tensor.shape == (1, 2)
    assert tensor[0, 0].item() == 1.5
    assert tensor[0, 1].item() == -0.5


def test_generate_runs():
    config = rey_tiny_config()
    model = ReyRNM(config)
    tokenizer = ReyTokenizer(vocab_size=config.vocab_size)
    out = generate(
        model,
        tokenizer,
        "hello world",
        max_new_tokens=5,
        do_sample=False,
        device="cpu",
    )
    assert isinstance(out.text, str)
    assert len(out.token_ids) > len(tokenizer.encode("hello world"))


def test_generate_with_trace():
    config = rey_tiny_config()
    model = ReyRNM(config)
    tokenizer = ReyTokenizer(vocab_size=config.vocab_size)
    out = generate(
        model,
        tokenizer,
        "hello",
        controls={"truthfulness": 1.0},
        max_new_tokens=3,
        do_sample=False,
        return_concept_trace=True,
        device="cpu",
    )
    assert out.concept_trace is not None
    assert "truthfulness" in out.concept_trace
    assert len(out.concept_trace["truthfulness"]) == 3
