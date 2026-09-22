"""The loops: over a plain v2 dataset, over an aligned one, and over a
multilingual pair, each saving on the schedule the caller asked for."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch
from tqdm import tqdm

from ...model_v2 import RejRNMv2
from ..data import DEFAULT_LOG_EVERY
from .steps import (
    train_step_v2,
    train_step_v2_aligned,
    train_step_v2_multilingual,
)

def train_v2_multilingual(
    model: RejRNMv2,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = DEFAULT_LOG_EVERY,
    save_every: int | None = None,
    save_fn=None,
) -> List[Dict[str, float]]:
    """Train RejRNMv2 with parallel multilingual concept supervision.

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
    model: RejRNMv2,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = DEFAULT_LOG_EVERY,
    save_every: int | None = None,
    save_fn=None,
) -> List[Dict[str, float]]:
    """Train RejRNMv2 with concept-alignment supervision.

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
    model: RejRNMv2,
    dataset: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
    log_every: int = DEFAULT_LOG_EVERY,
    save_every: int | None = None,
    save_fn=None,
    perturb_controls: bool = False,
    perturbation_scale: float = 1.0,
) -> List[Dict[str, float]]:
    """Train a RejRNMv2 model on a token dataset.

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
