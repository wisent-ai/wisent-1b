"""One step of v2 training: plain, aligned, and multilingual."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch
from tqdm import tqdm

from ...model_v2 import RejRNMv2
from ..data import DEFAULT_LOG_EVERY, compute_lm_loss
from .losses import (
    _random_magnitude_controls,
    compute_alignment_loss,
    compute_language_invariant_loss,
    compute_v2_loss,
)
def train_step_v2_aligned(
    model: RejRNMv2,
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
    model: RejRNMv2,
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

