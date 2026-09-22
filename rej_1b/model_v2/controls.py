"""What a caller may ask the model to do, checked before it does it.

A control names a mode the configuration allows and carries one value per
named concept — or, for a direction, one vector per named concept in
subspace coordinates. A control that does not fit is refused by name and
shape rather than broadcast into something else."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ..config import RejConfigV2
from .concepts import DIRECTION_SHAPE_DIMENSIONS


def validate_controls(
    config: RejConfigV2,
    controls: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if controls is None:
        return {}
    for mode, tensor in controls.items():
        if mode not in config.control_modes:
            raise ValueError(f"Unknown control mode \x27{mode}\x27. Allowed: {config.control_modes}")
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != config.n_named_concepts and mode != "direction":
            raise ValueError(
                f"Control \x27{mode}\x27 last dim ({tensor.shape[-1]}) must match "
                f"n_named_concepts ({config.n_named_concepts})"
            )
        if mode == "direction" and tuple(tensor.shape)[-DIRECTION_SHAPE_DIMENSIONS:] != (
            config.n_named_concepts,
            config.subspace_rank,
        ):
            raise ValueError(
                f"Control \x27direction\x27 must have shape (B, n_named, subspace_rank) = "
                f"(..., {config.n_named_concepts}, {config.subspace_rank})"
            )
    return controls
