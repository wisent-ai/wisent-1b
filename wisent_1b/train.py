"""Causal language modeling training for Wisent-1B."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .model import WisentRNM
from .model_v2 import WisentRNMv2


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
        ignore_index=-100,
    )
    return loss


def train_step(
    model: WisentRNM,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single training step. Returns loss value."""
    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    outputs = model(batch)
    loss = compute_lm_loss(outputs["logits"], batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train(
    model: WisentRNM,
    dataset: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = 10,
    save_every: int | None = None,
    save_fn=None,
) -> List[float]:
    """Train a WisentRNM model on a token dataset.

    Args:
        model: the model to train.
        dataset: iterator yielding token-id batches.
        optimizer: optimizer.
        device: device.
        num_steps: total training steps.
        log_every: how often to print loss.
        save_every: how often to call save_fn(step).
        save_fn: optional callable(step) invoked for checkpointing.

    Returns:
        list of loss values per step.
    """
    model.to(device)
    losses = []
    iterator = iter(dataset)

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        loss = train_step(model, batch, optimizer, device)
        losses.append(loss)
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"Step {step + 1}/{num_steps} | loss: {loss:.4f}")

        if save_every is not None and save_fn is not None and (step + 1) % save_every == 0:
            save_fn(step + 1)

    return losses


def compute_v2_loss(
    model: WisentRNMv2,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute v2 training loss = LM loss + KL regularization.

    Returns:
        total loss, dict of component losses.
    """
    outputs = model(batch)
    lm_loss = compute_lm_loss(outputs["logits"], batch)
    kl_loss = outputs.get("kl_loss", torch.tensor(0.0, device=batch.device))
    total_loss = lm_loss + model.config.kl_weight * kl_loss
    return total_loss, {
        "lm_loss": lm_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
    }


def train_step_v2(
    model: WisentRNMv2,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Single v2 training step. Returns loss components."""
    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    loss, metrics = compute_v2_loss(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return metrics


def train_v2(
    model: WisentRNMv2,
    dataset: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = 10,
    save_every: int | None = None,
    save_fn=None,
) -> List[Dict[str, float]]:
    """Train a WisentRNMv2 model on a token dataset.

    Args:
        model: the v2 model to train.
        dataset: iterator yielding token-id batches.
        optimizer: optimizer.
        device: device.
        num_steps: total training steps.
        log_every: how often to print loss.
        save_every: how often to call save_fn(step).
        save_fn: optional callable(step) invoked for checkpointing.

    Returns:
        list of loss-component dicts per step.
    """
    model.to(device)
    losses = []
    iterator = iter(dataset)

    pbar = tqdm(range(num_steps), desc="Training v2")
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        metrics = train_step_v2(model, batch, optimizer, device)
        losses.append(metrics)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        if log_every > 0 and (step + 1) % log_every == 0:
            msg = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Step {step + 1}/{num_steps} | {msg}")

        if save_every is not None and save_fn is not None and (step + 1) % save_every == 0:
            save_fn(step + 1)

    return losses
