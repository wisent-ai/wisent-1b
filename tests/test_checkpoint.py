"""Tests for checkpoint save/load utilities."""
import os
import tempfile

import torch

from wisent_1b.config import wisent_tiny_config, wisent_tiny_v2_config
from wisent_1b.model import WisentRNM
from wisent_1b.model_v2 import WisentRNMv2
from wisent_1b.utils import save_checkpoint, load_checkpoint


def test_save_load_v1():
    config = wisent_tiny_config()
    model = WisentRNM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids)["logits"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(model, None, step=1, output_dir=tmpdir)
        assert os.path.exists(path)

        loaded = load_checkpoint(path, device="cpu")
        assert isinstance(loaded, WisentRNM)
        with torch.no_grad():
            actual = loaded(input_ids)["logits"]
        assert torch.allclose(expected, actual, atol=1e-6)


def test_save_load_v2():
    config = wisent_tiny_v2_config()
    model = WisentRNMv2(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids, deterministic=True)["logits"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(model, None, step=1, output_dir=tmpdir)
        assert os.path.exists(path)

        loaded = load_checkpoint(path, device="cpu")
        assert isinstance(loaded, WisentRNMv2)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids, deterministic=True)["logits"]
        assert torch.allclose(expected, actual, atol=1e-6)
