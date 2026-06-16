"""Causal language modeling training for Rey-1B."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .model import ReyRNM
from .model_v2 import ReyRNMv2


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
    model: ReyRNM,
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
    model: ReyRNM,
    dataset: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = 10,
    save_every: int | None = None,
    save_fn=None,
) -> List[float]:
    """Train a ReyRNM model on a token dataset.

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
    model: ReyRNMv2,
    batch: torch.Tensor,
    controls: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute v2 training loss = LM loss + KL regularization.

    Args:
        model: v2 model.
        batch: token ids (B, T).
        controls: optional geometric control dict.

    Returns:
        total loss, dict of component losses.
    """
    outputs = model(batch, controls=controls)
    lm_loss = compute_lm_loss(outputs["logits"], batch)
    kl_loss = outputs.get("kl_loss", torch.tensor(0.0, device=batch.device))
    geometry_loss = outputs.get("geometry_loss", torch.tensor(0.0, device=batch.device))
    total_loss = (
        lm_loss
        + model.config.kl_weight * kl_loss
        + model.config.geometry_weight * geometry_loss
    )
    return total_loss, {
        "lm_loss": lm_loss.item(),
        "kl_loss": kl_loss.item(),
        "geometry_loss": geometry_loss.item(),
        "total_loss": total_loss.item(),
    }


def _random_magnitude_controls(
    batch_size: int,
    n_named_concepts: int,
    scale: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample random magnitude controls for control-perturbation training."""
    return {
        "magnitude": torch.randn(batch_size, n_named_concepts, device=device) * scale,
    }


def train_step_v2(
    model: ReyRNMv2,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    controls: Optional[Dict[str, torch.Tensor]] = None,
    perturb_controls: bool = False,
    perturbation_scale: float = 1.0,
) -> Dict[str, float]:
    """Single v2 training step. Returns loss components.

    Args:
        model: v2 model.
        batch: token ids.
        optimizer: optimizer.
        device: device.
        controls: optional geometric controls.
        perturb_controls: if True, replace/add random magnitude controls.
        perturbation_scale: std of the random control perturbation.
    """
    model.train()
    batch = batch.to(device)
    if perturb_controls:
        controls = _random_magnitude_controls(
            batch.size(0), model.config.n_named_concepts, perturbation_scale, device
        )
    loss, metrics = compute_v2_loss(model, batch, controls=controls)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return metrics


def compute_alignment_loss(
    predicted_controls: torch.Tensor,
    target_controls: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between predicted and target concept control magnitudes."""
    return F.mse_loss(predicted_controls, target_controls)


def compute_language_invariant_loss(
    concept_embedding_l1: torch.Tensor,
    concept_embedding_l2: torch.Tensor,
) -> torch.Tensor:
    """MSE loss aligning concept states of parallel sentences in two languages."""
    pooled_l1 = concept_embedding_l1.mean(dim=1)
    pooled_l2 = concept_embedding_l2.mean(dim=1)
    return F.mse_loss(pooled_l1, pooled_l2)


def train_step_v2_aligned(
    model: ReyRNMv2,
    batch_tokens: torch.Tensor,
    batch_controls: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Single v2 training step with concept-alignment loss.

    Args:
        model: v2 model with use_concept_alignment=True.
        batch_tokens: token ids (B, T).
        batch_controls: target control magnitudes (B, n_named_concepts).
        optimizer: optimizer.
        device: device.

    Returns:
        loss-component dict.
    """
    model.train()
    batch_tokens = batch_tokens.to(device)
    batch_controls = batch_controls.to(device)
    optimizer.zero_grad()

    geo_controls = {"magnitude": batch_controls}
    outputs = model(batch_tokens, controls=geo_controls, return_alignment_pred=True)
    lm_loss = compute_lm_loss(outputs["logits"], batch_tokens)
    kl_loss = outputs.get("kl_loss", torch.tensor(0.0, device=device))
    geometry_loss = outputs.get("geometry_loss", torch.tensor(0.0, device=device))
    align_loss = compute_alignment_loss(outputs["predicted_controls"], batch_controls)

    total_loss = (
        lm_loss
        + model.config.kl_weight * kl_loss
        + model.config.geometry_weight * geometry_loss
        + model.config.alignment_weight * align_loss
    )
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "total_loss": total_loss.item(),
        "lm_loss": lm_loss.item(),
        "kl_loss": kl_loss.item(),
        "geometry_loss": geometry_loss.item(),
        "align_loss": align_loss.item(),
    }


def train_step_v2_multilingual(
    model: ReyRNMv2,
    batch_tokens_l1: torch.Tensor,
    batch_tokens_l2: torch.Tensor,
    batch_controls: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Single v2 training step with language-invariant concept supervision.

    Args:
        model: v2 model with use_language_invariant_concepts=True.
        batch_tokens_l1: token ids in language 1 (B, T1).
        batch_tokens_l2: token ids in language 2 (B, T2).
        batch_controls: target control magnitudes (B, n_named_concepts).
        optimizer: optimizer.
        device: device.

    Returns:
        loss-component dict.
    """
    model.train()
    batch_tokens_l1 = batch_tokens_l1.to(device)
    batch_tokens_l2 = batch_tokens_l2.to(device)
    batch_controls = batch_controls.to(device)
    optimizer.zero_grad()

    geo_controls = {"magnitude": batch_controls}
    out_l1 = model(
        batch_tokens_l1, controls=geo_controls,
        return_alignment_pred=model.config.use_concept_alignment,
        return_concept_embedding=True,
    )
    out_l2 = model(
        batch_tokens_l2, controls=geo_controls,
        return_concept_embedding=True,
    )

    lm_loss = compute_lm_loss(out_l1["logits"], batch_tokens_l1)
    lm_loss = lm_loss + compute_lm_loss(out_l2["logits"], batch_tokens_l2)
    kl_loss = out_l1.get("kl_loss", torch.tensor(0.0, device=device))
    kl_loss = kl_loss + out_l2.get("kl_loss", torch.tensor(0.0, device=device))
    geometry_loss = out_l1.get("geometry_loss", torch.tensor(0.0, device=device))
    geometry_loss = geometry_loss + out_l2.get("geometry_loss", torch.tensor(0.0, device=device))

    total_loss = (
        lm_loss
        + model.config.kl_weight * kl_loss
        + model.config.geometry_weight * geometry_loss
    )

    if model.config.use_concept_alignment:
        align_loss = compute_alignment_loss(out_l1["predicted_controls"], batch_controls)
        total_loss = total_loss + model.config.alignment_weight * align_loss
    else:
        align_loss = torch.tensor(0.0, device=device)

    if model.config.use_language_invariant_concepts:
        inv_loss = compute_language_invariant_loss(
            out_l1["concept_embedding"], out_l2["concept_embedding"]
        )
        total_loss = total_loss + model.config.language_invariant_weight * inv_loss
    else:
        inv_loss = torch.tensor(0.0, device=device)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "total_loss": total_loss.item(),
        "lm_loss": lm_loss.item(),
        "kl_loss": kl_loss.item(),
        "geometry_loss": geometry_loss.item(),
        "align_loss": align_loss.item(),
        "inv_loss": inv_loss.item(),
    }


def train_v2_multilingual(
    model: ReyRNMv2,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = 10,
    save_every: int | None = None,
    save_fn=None,
) -> List[Dict[str, float]]:
    """Train ReyRNMv2 with parallel multilingual concept supervision.

    Args:
        model: v2 model.
        dataset: iterator yielding (tokens_l1, tokens_l2, controls) tuples.
        optimizer: optimizer.
        device: device.
        num_steps: total training steps.
        log_every: how often to print loss.
        save_every: how often to call save_fn(step).
        save_fn: optional callable(step) invoked for checkpointing.

    Returns:
        list of loss-component dicts per step.
    """
    if not model.config.use_language_invariant_concepts:
        raise ValueError(
            "Model must have use_language_invariant_concepts=True for multilingual training."
        )
    model.to(device)
    losses = []
    iterator = iter(dataset)

    pbar = tqdm(range(num_steps), desc="Training v2 multilingual")
    for step in pbar:
        try:
            batch_l1, batch_l2, batch_controls = next(iterator)
        except StopIteration:
            break

        metrics = train_step_v2_multilingual(
            model, batch_l1, batch_l2, batch_controls, optimizer, device
        )
        losses.append(metrics)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        if log_every > 0 and (step + 1) % log_every == 0:
            msg = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Step {step + 1}/{num_steps} | {msg}")

        if save_every is not None and save_fn is not None and (step + 1) % save_every == 0:
            save_fn(step + 1)

    return losses


def train_v2_aligned(
    model: ReyRNMv2,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = 10,
    save_every: int | None = None,
    save_fn=None,
) -> List[Dict[str, float]]:
    """Train ReyRNMv2 with concept-alignment supervision.

    Args:
        model: v2 model.
        dataset: iterator yielding (token_ids, control_magnitudes) tuples.
        optimizer: optimizer.
        device: device.
        num_steps: total training steps.
        log_every: how often to print loss.
        save_every: how often to call save_fn(step).
        save_fn: optional callable(step) invoked for checkpointing.

    Returns:
        list of loss-component dicts per step.
    """
    if not model.config.use_concept_alignment:
        raise ValueError("Model must have use_concept_alignment=True for aligned training.")
    model.to(device)
    losses = []
    iterator = iter(dataset)

    pbar = tqdm(range(num_steps), desc="Training v2 aligned")
    for step in pbar:
        try:
            batch_tokens, batch_controls = next(iterator)
        except StopIteration:
            break

        metrics = train_step_v2_aligned(model, batch_tokens, batch_controls, optimizer, device)
        losses.append(metrics)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        if log_every > 0 and (step + 1) % log_every == 0:
            msg = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Step {step + 1}/{num_steps} | {msg}")

        if save_every is not None and save_fn is not None and (step + 1) % save_every == 0:
            save_fn(step + 1)

    return losses


def train_v2(
    model: ReyRNMv2,
    dataset: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = 10,
    save_every: int | None = None,
    save_fn=None,
    perturb_controls: bool = False,
    perturbation_scale: float = 1.0,
) -> List[Dict[str, float]]:
    """Train a ReyRNMv2 model on a token dataset.

    Args:
        model: the v2 model to train.
        dataset: iterator yielding token-id batches.
        optimizer: optimizer.
        device: device.
        num_steps: total training steps.
        log_every: how often to print loss.
        save_every: how often to call save_fn(step).
        save_fn: optional callable(step) invoked for checkpointing.
        perturb_controls: randomly perturb control magnitudes each step.
        perturbation_scale: std of the random perturbation.

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

        metrics = train_step_v2(
            model,
            batch,
            optimizer,
            device,
            perturb_controls=perturb_controls,
            perturbation_scale=perturbation_scale,
        )
        losses.append(metrics)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        if log_every > 0 and (step + 1) % log_every == 0:
            msg = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Step {step + 1}/{num_steps} | {msg}")

        if save_every is not None and save_fn is not None and (step + 1) % save_every == 0:
            save_fn(step + 1)

    return losses
