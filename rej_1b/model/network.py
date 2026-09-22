"""The model itself: tokens and concepts advancing together through the
layers, with the concept state carried over depth."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import RejConfig
from .layer import ConceptCarry, RejLayer

class RejRNM(nn.Module):
    """Rej Representation-Native Model."""

    def __init__(self, config: RejConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.token_dropout = nn.Dropout(config.dropout)

        # Learned concept embeddings: one vector per concept slot.
        self.concept_embeddings = nn.Parameter(
            torch.randn(config.n_concepts, config.d_concept) * config.initializer_range
        )

        # Direct token-level control embedding. This guarantees that scalar
        # concept controls reach the token stream even while the deeper concept
        # stream is still learning. It is a stable bootstrap for the RNM
        # architecture and can be annealed or removed in later training stages.
        self.token_control_embedding = nn.Parameter(
            torch.randn(config.n_named_concepts, config.d_model) * config.initializer_range
        )

        self.layers = nn.ModuleList([RejLayer(config) for _ in range(config.n_layers)])

        self.concept_carry = (
            ConceptCarry(config.d_concept) if config.carry_concept_state else None
        )

        self.token_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embeddings.weight, std=self.config.initializer_range)
        nn.init.normal_(self.position_embeddings.weight, std=self.config.initializer_range)
        nn.init.normal_(self.concept_embeddings, std=self.config.initializer_range)
        nn.init.normal_(self.token_control_embedding, std=self.config.initializer_range)
        nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)

    def _validate_controls(self, controls: Optional[torch.Tensor]) -> torch.Tensor:
        """Validate and reshape controls to (B, n_named_concepts)."""
        if controls is None:
            raise ValueError("controls cannot be None for validation")
        if controls.dim() == 1:
            controls = controls.unsqueeze(0)
        if controls.shape[-1] != self.config.n_named_concepts:
            raise ValueError(
                f"controls last dim ({controls.shape[-1]}) must match "
                f"n_named_concepts ({self.config.n_named_concepts})"
            )
        return controls

    def _build_initial_concepts(
        self,
        controls: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the initial concept state for the batch.

        Named concept slots are scaled by the provided scalar controls. Latent
        slots use the learned concept embeddings unchanged.

        Args:
            controls: (B, n_named_concepts) scalar control magnitudes, or None.

        Returns:
            (B, K, d_concept) initial concept state.
        """
        # Start from learned concept embeddings.
        concepts = self.concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1).clone()

        if controls is None:
            return concepts

        controls = self._validate_controls(controls)

        # Scale named concept embeddings by control magnitudes.
        named_embeddings = self.concept_embeddings[: self.config.n_named_concepts]
        scaled_named = controls.unsqueeze(-1) * named_embeddings.unsqueeze(0)
        concepts[:, : self.config.n_named_concepts, :] = scaled_named
        return concepts

    def forward(
        self,
        input_ids: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        return_concept_trace: bool = False,
        concept_state: Optional[torch.Tensor] = None,
        return_concept_state: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (B, T) token indices.
            controls: (B, n_named_concepts) scalar controls, or None.
            return_concept_trace: if True, collect and return concept states per layer.
            concept_state: (B, K, d_concept) final concept state of a previous
                step, fused into this step's initial state. Requires
                `carry_concept_state` on the config.
            return_concept_state: if True, return the final concept state so a
                caller can pass it to the next step.

        Returns:
            dict with keys:
                "logits": (B, T, vocab_size)
                "concept_trace": list of (B, K, d_concept) tensors if requested, else None
                "concept_state": (B, K, d_concept) final state if requested, else None
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        tokens = self.token_embeddings(input_ids) + self.position_embeddings(positions)

        # Add a direct token-level control signal. This is a stable path for
        # concept controls to influence generation from the first layer.
        if controls is not None:
            controls = self._validate_controls(controls)
            token_control = controls.unsqueeze(1) @ self.token_control_embedding
            tokens = tokens + token_control

        tokens = self.token_dropout(tokens)

        # Build initial concept state, applying named concept controls.
        concepts = self._build_initial_concepts(controls, B, device)

        # Fuse in the state the previous step ended on. Refusing an unusable
        # argument here rather than ignoring it: a caller that threads a state
        # through a model built without the carry would otherwise get the
        # no-carry behavior and no way to notice.
        if concept_state is not None:
            if self.concept_carry is None:
                raise ValueError(
                    "concept_state was passed but this model was built with "
                    "carry_concept_state=False, so it has no carry module"
                )
            expected = (B, self.config.n_concepts, self.config.d_concept)
            if tuple(concept_state.shape) != expected:
                raise ValueError(
                    f"concept_state shape {tuple(concept_state.shape)} must be {expected}"
                )
            concepts = self.concept_carry(concepts, concept_state.to(device))

        concept_trace = [] if return_concept_trace else None

        for layer in self.layers:
            tokens, concepts = layer(tokens, concepts)
            if return_concept_trace:
                concept_trace.append(concepts.detach().clone())

        tokens = self.token_ln(tokens)
        logits = self.lm_head(tokens)

        return {
            "logits": logits,
            "concept_trace": concept_trace,
            "concept_state": concepts if return_concept_state else None,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def named_concept_labels(self) -> list:
        return list(self.config.named_concepts)
