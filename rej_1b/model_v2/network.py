"""The model itself: tokens and concepts advancing together, layer by
layer, with the geometry and KL terms a training step reads back."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RejConfigV2
from ..model import FeedForward, FlexibleMultiHeadAttention
from .concepts import (
    ConceptRouter,
    ProbabilisticConceptState,
    SubspaceConceptBank,
)
from .controls import validate_controls
from .dynamics import (
    ConceptAlignmentHead,
    GeometricControl,
    NonlinearConceptCell,
    SteeringManifold,
)


class RejRNMv2(nn.Module):
    """Rej Representation-Native Model v2 with geometric concepts."""

    def __init__(self, config: RejConfigV2):
        super().__init__()
        # RejConfigV2 inherits carry_concept_state from RejConfig, and v2 does
        # not implement it: its concept state is a Gaussian over subspace
        # coordinates, so carrying it across steps is a distribution to
        # propagate rather than a vector to fuse. Refusing is the honest
        # answer; accepting the flag and dropping it would report a carried
        # model that carries nothing.
        if getattr(config, "carry_concept_state", False):
            raise NotImplementedError(
                "carry_concept_state is implemented for RejRNM, not RejRNMv2; "
                "v2 carries a probabilistic concept state and needs its own "
                "fusion rule"
            )
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.token_dropout = nn.Dropout(config.dropout)

        # Subspace concept bank.
        self.concept_bank = SubspaceConceptBank(config)

        # Initial concept state (mean/log_std in subspace coords).
        self.concept_mean_init = nn.Parameter(
            torch.randn(config.n_concepts, config.subspace_rank) * config.initializer_range
        )
        if config.probabilistic_concepts:
            self.concept_log_std_init = nn.Parameter(
                torch.randn(config.n_concepts, config.subspace_rank) * 0.01
            )
        else:
            self.register_parameter("concept_log_std_init", None)

        # Router and non-linear concept cells per layer.
        if config.use_concept_router:
            self.router = ConceptRouter(config.d_model, config.n_concepts, config.dropout)
        else:
            self.router = None

        self.concept_cells = nn.ModuleList(
            [NonlinearConceptCell(config) for _ in range(config.n_layers)]
        )

        if config.use_titan_manifold:
            self.steering_manifolds = nn.ModuleList(
                [SteeringManifold(config) for _ in range(config.n_layers)]
            )
        else:
            self.steering_manifolds = None

        # Token stream layers.
        self.token_ln1s = nn.ModuleList([
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps) for _ in range(config.n_layers)
        ])
        self.token_self_attns = nn.ModuleList([
            FlexibleMultiHeadAttention(
                q_dim=config.d_model, kv_dim=config.d_model, out_dim=config.d_model,
                n_heads=config.n_heads, d_head=config.d_head,
                dropout=config.dropout, causal=True,
            ) for _ in range(config.n_layers)
        ])
        self.token_ln2s = nn.ModuleList([
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps) for _ in range(config.n_layers)
        ])
        self.token_ffns = nn.ModuleList([
            FeedForward(config.d_model, config.intermediate_size, config.d_model, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Concept-to-token gate (learnable, init near 0 for stability).
        self.concept_to_token_gate = nn.Parameter(torch.zeros(1))

        # Direct token-level control embedding (bootstrap path).
        self.token_control_embedding = nn.Parameter(
            torch.randn(config.n_named_concepts, config.d_model) * config.initializer_range
        )

        self.geometric_control = GeometricControl(config)

        if config.use_concept_alignment:
            self.concept_alignment_head = ConceptAlignmentHead(
                config.d_concept, config.n_named_concepts, config.dropout
            )
        else:
            self.concept_alignment_head = None

        self.token_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embeddings.weight, std=self.config.initializer_range)
        nn.init.normal_(self.position_embeddings.weight, std=self.config.initializer_range)
        nn.init.normal_(self.concept_mean_init, std=self.config.initializer_range)
        if self.concept_log_std_init is not None:
            nn.init.normal_(self.concept_log_std_init, std=0.01)
        nn.init.normal_(self.token_control_embedding, std=self.config.initializer_range)
        nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)

    def _validate_controls(self, controls: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return validate_controls(self.config, controls)

    def _build_initial_concept_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build initial mean and log_std in subspace coordinates."""
        mean = self.concept_mean_init.unsqueeze(0).expand(batch_size, -1, -1)
        if self.config.probabilistic_concepts and self.concept_log_std_init is not None:
            log_std = self.concept_log_std_init.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            log_std = torch.zeros_like(mean)
        return mean, log_std

    def forward(
        self,
        input_ids: torch.Tensor,
        controls: Optional[Dict[str, torch.Tensor]] = None,
        return_concept_trace: bool = False,
        return_alignment_pred: bool = False,
        return_concept_embedding: bool = False,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, T)
            controls: dict of control tensors per mode.
            return_concept_trace: if True, collect concept states per layer.
            return_alignment_pred: if True, return concept-alignment predictions.
            return_concept_embedding: if True, return final concept embeddings.
            deterministic: if True, do not sample concept state.

        Returns:
            dict with "logits", optional "concept_trace", optional "kl_loss",
            optional "geometry_loss", optional "predicted_controls",
            and optional "concept_embedding".
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        tokens = self.token_embeddings(input_ids) + self.position_embeddings(positions)

        # Direct token-level control embedding.
        controls = self._validate_controls(controls)
        if "magnitude" in controls:
            mag = controls["magnitude"].unsqueeze(1)  # (B, 1, n_named)
            token_control = mag @ self.token_control_embedding  # (B, 1, d_model)
            tokens = tokens + token_control

        tokens = self.token_dropout(tokens)

        # Initialize concept state.
        concept_mean, concept_log_std = self._build_initial_concept_state(B, device)
        concept_mean, concept_log_std = self.geometric_control(
            concept_mean, concept_log_std, controls
        )

        concept_trace = [] if return_concept_trace else None
        kl_loss = 0.0
        geometry_loss = 0.0

        basis = self.concept_bank.get_basis()
        centroid = self.concept_bank.centroid

        for layer_idx in range(self.config.n_layers):
            cell = self.concept_cells[layer_idx]

            # 1. Token self-attention.
            tokens = tokens + self.token_self_attns[layer_idx](
                self.token_ln1s[layer_idx](tokens),
                self.token_ln1s[layer_idx](tokens),
                self.token_ln1s[layer_idx](tokens),
            )

            # 1b. TITAN-style input-dependent steering manifold.
            if self.steering_manifolds is not None:
                concept_mean = self.steering_manifolds[layer_idx](concept_mean, tokens)

            # 2. Build probabilistic concept state and decode to d_concept.
            concept_state = ProbabilisticConceptState(concept_mean, concept_log_std)
            coords = concept_state.sample(deterministic=deterministic or not self.config.probabilistic_concepts)
            concept_emb = cell.decode_manifold(coords, basis, centroid)  # (B, K, dc)

            # 3. Router (optional).
            if self.router is not None:
                router_weights = self.router(tokens)
            else:
                router_weights = torch.ones(B, self.config.n_concepts, T, device=device) / T

            # 4. Concept read/update.
            read_update = cell.read(tokens, concept_emb, router_weights)
            concept_emb = concept_emb + read_update
            concept_emb = cell.update(concept_emb)

            # 5. Project updated concept embedding back to subspace coords.
            # Subtract centroid and project.
            centered = concept_emb - centroid.unsqueeze(0)
            concept_mean = self.concept_bank.project_to_subspace(centered, basis)
            if self.config.use_geometry_regularization:
                # Biprojection-style regularization: the updated concept embedding
                # should stay close to the subspace manifold it was projected onto.
                reconstructed = self.concept_bank.project_from_subspace(concept_mean, basis)
                geometry_loss = geometry_loss + F.mse_loss(concept_emb, reconstructed)
            if self.config.probabilistic_concepts:
                # Slowly update std; keep it as a learned per-concept parameter for now.
                concept_log_std = concept_log_std
                kl_loss = kl_loss + ProbabilisticConceptState(concept_mean, concept_log_std).kl_divergence()

            # 6. Tokens read from concepts.
            # Each concept writes a d_model update; router weights route it to tokens.
            concept_writes = cell.write(concept_emb)  # (B, K, d_model)
            # (B, T, K) @ (B, K, d_model) -> (B, T, d_model)
            token_update = torch.einsum(
                "btk,bkd->btd", router_weights.transpose(1, 2), concept_writes
            )
            tokens = tokens + torch.sigmoid(self.concept_to_token_gate) * token_update

            # 7. Token FFN.
            tokens = tokens + self.token_ffns[layer_idx](self.token_ln2s[layer_idx](tokens))

            if return_concept_trace:
                concept_trace.append({
                    "mean": concept_mean.detach().clone(),
                    "std": torch.exp(concept_log_std).detach().clone(),
                    "embedding": concept_emb.detach().clone(),
                })

        tokens = self.token_ln(tokens)
        logits = self.lm_head(tokens)

        output = {"logits": logits}
        if return_concept_trace:
            output["concept_trace"] = concept_trace
        if self.config.probabilistic_concepts:
            output["kl_loss"] = kl_loss / (B * self.config.n_layers)
        if self.config.use_geometry_regularization:
            output["geometry_loss"] = geometry_loss / self.config.n_layers
        if return_alignment_pred and self.concept_alignment_head is not None:
            output["predicted_controls"] = self.concept_alignment_head(concept_emb)
        if return_concept_embedding:
            output["concept_embedding"] = concept_emb
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def named_concept_labels(self) -> list:
        return list(self.config.named_concepts)
