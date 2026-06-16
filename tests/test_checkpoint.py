"""Tests for checkpoint save/load utilities."""
import os
import tempfile

import torch

from rey_1b.config import rey_tiny_config, rey_tiny_v2_config
from rey_1b.model import ReyRNM
from rey_1b.model_v2 import ReyRNMv2
from rey_1b.utils import save_checkpoint, load_checkpoint


def test_save_load_v1():
    config = rey_tiny_config()
    model = ReyRNM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids)["logits"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(model, None, step=1, output_dir=tmpdir)
        assert os.path.exists(path)

        loaded = load_checkpoint(path, device="cpu")
        assert isinstance(loaded, ReyRNM)
        with torch.no_grad():
            actual = loaded(input_ids)["logits"]
        assert torch.allclose(expected, actual, atol=1e-6)


def test_save_load_v2():
    config = rey_tiny_v2_config()
    model = ReyRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids, deterministic=True)["logits"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(model, None, step=1, output_dir=tmpdir)
        assert os.path.exists(path)

        loaded = load_checkpoint(path, device="cpu")
        assert isinstance(loaded, ReyRNMv2)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids, deterministic=True)["logits"]
        assert torch.allclose(expected, actual, atol=1e-6)
