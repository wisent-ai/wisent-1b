"""What a training step reads: a dataset of token sequences, the padding
that makes a batch, and the next-token loss itself."""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# PyTorch cross-entropy ignores targets with this label; progress is logged every few steps.
IGNORE_INDEX = -100
DEFAULT_LOG_EVERY = 10


class TokenDataset(Dataset):
    """Simple dataset that yields sequences of token ids."""

    def __init__(self, token_ids: List[List[int]], seq_length: int):
        self.samples: List[List[int]] = []
        for ids in token_ids:
            for i in range(0, max(1, len(ids) - seq_length), seq_length):
                chunk = ids[i : i + seq_length + 1]
                if len(chunk) < 2:
                    continue
                self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)


def collate_fn(batch: List[torch.Tensor], pad_token_id: int = 0) -> torch.Tensor:
    """Pad a batch of variable-length token sequences."""
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq
    return padded


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross-entropy loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return loss

