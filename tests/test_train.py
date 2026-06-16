"""Tests for training utilities."""
import torch
from torch.utils.data import DataLoader

from rey_1b.config import rey_tiny_config
from rey_1b.model import ReyRNM
from rey_1b.tokenizer import ReyTokenizer
from rey_1b.train import (
    TokenDataset, collate_fn, compute_lm_loss, train_step, train_step_v2,
)


def test_compute_lm_loss():
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    loss = compute_lm_loss(logits, labels)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_train_step():
    config = rey_tiny_config()
    model = ReyRNM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randint(0, config.vocab_size, (2, 16))
    loss = train_step(model, batch, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0


def test_token_dataset():
    tokenizer = ReyTokenizer(vocab_size=256)
    ids = tokenizer.encode("the cat sat on the mat . " * 10)
    dataset = TokenDataset([ids], seq_length=8)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample.shape[0] <= 9  # seq_length + 1


def test_collate_fn():
    batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    padded = collate_fn(batch, pad_token_id=0)
    assert padded.shape == (2, 3)
    assert padded[1, 2].item() == 0


def test_train_step_v2_with_perturbation():
    from rey_1b.config import rey_tiny_v2_config
    from rey_1b.model_v2 import ReyRNMv2

    config = rey_tiny_v2_config()
    model = ReyRNMv2(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randint(0, config.vocab_size, (2, 16))
    metrics = train_step_v2(
        model, batch, optimizer, torch.device("cpu"),
        perturb_controls=True, perturbation_scale=1.0,
    )
    assert "total_loss" in metrics
    assert "lm_loss" in metrics
    assert "kl_loss" in metrics
    assert metrics["total_loss"] > 0


def test_train_step_v2_aligned():
    from rey_1b.config import rey_tiny_v2_config
    from rey_1b.model_v2 import ReyRNMv2
    from rey_1b.train import train_step_v2_aligned

    config = rey_tiny_v2_config()
    config.use_concept_alignment = True
    model = ReyRNMv2(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_tokens = torch.randint(0, config.vocab_size, (2, 16))
    batch_controls = torch.randn(2, config.n_named_concepts)
    metrics = train_step_v2_aligned(
        model, batch_tokens, batch_controls, optimizer, torch.device("cpu"),
    )
    assert "total_loss" in metrics
    assert "lm_loss" in metrics
    assert "kl_loss" in metrics
    assert "align_loss" in metrics
    assert metrics["total_loss"] > 0


def test_train_step_v2_multilingual():
    from rey_1b.config import rey_tiny_v2_config
    from rey_1b.model_v2 import ReyRNMv2
    from rey_1b.train import train_step_v2_multilingual

    config = rey_tiny_v2_config()
    config.use_language_invariant_concepts = True
    model = ReyRNMv2(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_l1 = torch.randint(0, config.vocab_size, (2, 12))
    batch_l2 = torch.randint(0, config.vocab_size, (2, 14))
    batch_controls = torch.randn(2, config.n_named_concepts)
    metrics = train_step_v2_multilingual(
        model, batch_l1, batch_l2, batch_controls, optimizer, torch.device("cpu"),
    )
    assert "total_loss" in metrics
    assert "lm_loss" in metrics
    assert "kl_loss" in metrics
    assert "geometry_loss" in metrics
    assert "align_loss" in metrics
    assert "inv_loss" in metrics
    assert metrics["total_loss"] > 0
