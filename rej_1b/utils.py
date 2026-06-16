"""Utility helpers for Rej-1B."""
from __future__ import annotations

import os
from typing import Dict

import torch

from .config import RejConfig, RejConfigV2
from .model import RejRNM
from .model_v2 import RejRNMv2


def get_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    output_dir: str,
    extra: Dict | None = None,
) -> str:
    """Save a training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        state.update(extra)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, device: torch.device | str = "cpu"):
    """Load a model from a checkpoint (auto-detects v1/v2)."""
    device = torch.device(device)
    state = torch.load(path, map_location=device, weights_only=False)
    config_data = state["config"]
    if config_data.get("subspace_rank") is not None:
        config = RejConfigV2.from_dict(config_data)
        model = RejRNMv2(config).to(device)
    else:
        config = RejConfig.from_dict(config_data)
        model = RejRNM(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model
