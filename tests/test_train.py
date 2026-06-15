"""Tests for training utilities."""
import torch
from torch.utils.data import DataLoader

from wisent_1b.config import wisent_tiny_config
from wisent_1b.model import WisentRNM
from wisent_1b.tokenizer import WisentTokenizer
from wisent_1b.train import TokenDataset, collate_fn, compute_lm_loss, train_step


def test_compute_lm_loss():
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    loss = compute_lm_loss(logits, labels)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_train_step():
    config = wisent_tiny_config()
    model = WisentRNM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randint(0, config.vocab_size, (2, 16))
    loss = train_step(model, batch, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0


def test_token_dataset():
    tokenizer = WisentTokenizer(vocab_size=256)
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
