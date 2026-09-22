"""What the v2 model is trained against: the language-model loss plus the
geometry and KL terms it reports, and the random controls a step applies."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ...model_v2 import RejRNMv2
from ..data import compute_lm_loss


def compute_v2_loss(
    model: RejRNMv2,
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
    model: RejRNMv2,
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


