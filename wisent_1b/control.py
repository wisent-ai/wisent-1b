"""Concept alignment and control fine-tuning for Wisent-1B.

This module provides helpers to make the named concept plane controllable.
It is intentionally lightweight: full concept alignment would require
contrastive datasets and task-specific heads, which are left to the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WisentRNM


@dataclass
class ConceptExample:
    """A single contrastive example for a named concept."""

    concept_name: str
    text: str
    label: float  # e.g., 1.0 for high concept activation, 0.0 for low


class ConceptHead(nn.Module):
    """Linear probe that reads a concept slot and predicts a concept score."""

    def __init__(self, d_concept: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_concept),
            nn.Linear(d_concept, 1),
        )

    def forward(self, concept_state: torch.Tensor) -> torch.Tensor:
        """Args: (B, K, d_concept) -> (B, K) scores."""
        return self.net(concept_state).squeeze(-1)


def gather_named_concept_states(
    model: WisentRNM,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the named concept states from the last layer.

    Args:
        input_ids: (B, T)

    Returns:
        (B, n_named_concepts, d_concept) concept states.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, return_concept_trace=True)
    last_layer_trace = outputs["concept_trace"][-1]  # (B, K, d_concept)
    return last_layer_trace[:, : model.config.n_named_concepts, :]


def compute_contrastive_concept_loss(
    model: WisentRNM,
    tokenizer,
    examples: List[ConceptExample],
    concept_to_index: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Simple contrastive loss pushing named concepts in the right direction.

    For each example, we want the named concept slot to have a high score when
    label is high and a low score when label is low. A small linear head is
    trained on top of the frozen (or not) concept states.
    """
    concept_head = ConceptHead(model.config.d_concept).to(device)
    optimizer = torch.optim.Adam(concept_head.parameters(), lr=1e-3)

    texts = [ex.text for ex in examples]
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.float32, device=device)
    concept_indices = torch.tensor(
        [concept_to_index[ex.concept_name] for ex in examples], dtype=torch.long, device=device
    )

    token_ids = tokenizer.batch_encode(texts)
    max_len = max(len(ids) for ids in token_ids)
    padded = torch.full((len(token_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, ids in enumerate(token_ids):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    padded = padded.to(device)

    concept_states = gather_named_concept_states(model, padded)  # (B, N, d)
    selected_states = concept_states[torch.arange(len(examples)), concept_indices, :]
    scores = concept_head(selected_states).squeeze(-1)

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss


class ControlFineTuner:
    """Fine-tune a Wisent model to respond smoothly to concept interventions."""

    def __init__(
        self,
        model: WisentRNM,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        perturbation_scale: float = 0.5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.perturbation_scale = perturbation_scale

    def step(self, batch: torch.Tensor) -> Tuple[float, float]:
        """One control fine-tuning step.

        Applies random concept perturbations and ensures the model can still
        predict the original next-token distribution.

        Returns:
            (base_loss, perturbed_loss)
        """
        self.model.train()
        batch = batch.to(self.device)

        # Base forward.
        self.optimizer.zero_grad()
        base_logits = self.model(batch)["logits"]
        base_loss = F.cross_entropy(
            base_logits[:, :-1, :].reshape(-1, base_logits.size(-1)),
            batch[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        # Perturbed forward with random named-concept controls.
        B = batch.size(0)
        random_controls = torch.randn(
            B, self.model.config.n_named_concepts, device=self.device
        ) * self.perturbation_scale

        pert_logits = self.model(batch, controls=random_controls)["logits"]
        pert_loss = F.cross_entropy(
            pert_logits[:, :-1, :].reshape(-1, pert_logits.size(-1)),
            batch[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        loss = base_loss + pert_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return base_loss.item(), pert_loss.item()
