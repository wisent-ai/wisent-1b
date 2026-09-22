"""How concepts move: the non-linear cell that reads, updates and writes
them, the learned steering manifold baked into the concept stream, the
controls a caller can apply, and the head that predicts their magnitude."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RejConfigV2

class NonlinearConceptCell(nn.Module):
    """MLP-based read/update/write for concept dynamics."""

    def __init__(self, config: RejConfigV2):
        super().__init__()
        self.config = config
        d = config.d_model
        dc = config.d_concept
        r = config.subspace_rank
        h = config.concept_mlp_hidden

        # Read: tokens -> concept update.
        self.read_mlp = nn.Sequential(
            nn.Linear(d + dc, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, dc),
        )

        # Update: previous concept state -> next concept state.
        self.update_mlp = nn.Sequential(
            nn.Linear(dc, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, dc),
        )

        # Write: concept state -> token update.
        self.write_mlp = nn.Sequential(
            nn.Linear(dc, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, d),
        )

        # Optional manifold decoder: subspace coords -> curved concept embedding.
        if config.use_manifold_decoder:
            self.manifold_decoder = nn.Sequential(
                nn.Linear(r, h),
                nn.GELU(),
                nn.Linear(h, dc),
            )
        else:
            self.manifold_decoder = None

    def read(
        self,
        tokens: torch.Tensor,
        concept_state: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Read token information into concepts via router-weighted MLP.

        Args:
            tokens: (B, T, d_model)
            concept_state: (B, K, d_concept)
            router_weights: (B, K, T)

        Returns:
            update: (B, K, d_concept)
        """
        B, T, _ = tokens.shape
        K = concept_state.size(1)

        # Broadcast concept state to each token.
        concept_per_token = concept_state.unsqueeze(2).expand(B, K, T, -1)  # (B, K, T, dc)
        tokens_per_concept = tokens.unsqueeze(1).expand(B, K, T, -1)  # (B, K, T, d)

        pair = torch.cat([tokens_per_concept, concept_per_token], dim=-1)
        update_per_pair = self.read_mlp(pair)  # (B, K, T, dc)

        # Weighted sum over tokens for each concept.
        weights = router_weights.unsqueeze(-1)  # (B, K, T, 1)
        update = (weights * update_per_pair).sum(dim=2)  # (B, K, dc)
        return update

    def update(self, concept_state: torch.Tensor) -> torch.Tensor:
        return concept_state + self.update_mlp(concept_state)

    def write(self, concept_state: torch.Tensor) -> torch.Tensor:
        return self.write_mlp(concept_state)

    def decode_manifold(self, coords: torch.Tensor, basis: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        """Map subspace coordinates through optional manifold decoder.

        Args:
            coords: (B, K, rank)
            basis: (K, rank, d_concept)
            centroid: (K, d_concept)

        Returns:
            concept_embedding: (B, K, d_concept)
        """
        if self.manifold_decoder is None:
            return torch.einsum("bkr,krd->bkd", coords, basis) + centroid.unsqueeze(0)
        # Decode each coordinate vector through MLP, then reconstruct in ambient space.
        decoded = self.manifold_decoder(coords)  # (B, K, dc)
        linear_part = torch.einsum("bkr,krd->bkd", coords, basis)
        return linear_part + centroid.unsqueeze(0) + decoded


class SteeringManifold(nn.Module):
    """TITAN-style learned steering manifold baked into the concept stream.

    Each named concept owns a set of directions in subspace coordinates. An
    input-dependent intensity network combines those directions per layer,
    adding a context-specific shift to the concept state.
    """

    def __init__(self, config: RejConfigV2):
        super().__init__()
        self.config = config
        self.n_named = config.n_named_concepts
        self.n_directions = config.n_titan_directions
        self.rank = config.subspace_rank

        self.directions = nn.Parameter(
            torch.randn(config.n_named_concepts, config.n_titan_directions, config.subspace_rank)
            * config.initializer_range
        )
        h = max(config.concept_mlp_hidden // 2, 64)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(config.d_model, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.n_named_concepts * config.n_titan_directions),
        )

    def forward(self, concept_coords: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Add input-dependent manifold shift to named concept coordinates.

        Args:
            concept_coords: (B, K, rank)
            tokens: (B, T, d_model)

        Returns:
            updated concept_coords: (B, K, rank)
        """
        B = tokens.size(0)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        intensity = self.intensity_mlp(pooled).view(
            B, self.n_named, self.n_directions
        )
        intensity = F.softmax(intensity, dim=-1)
        # Weighted combination of directions for named concepts.
        delta = torch.einsum("bnd,ndr->bnr", intensity, self.directions)
        concept_coords = concept_coords.clone()
        concept_coords[:, : self.n_named] = concept_coords[:, : self.n_named] + delta
        return concept_coords


class GeometricControl(nn.Module):
    """Maps user controls to subspace-coordinate concept modifications."""

    def __init__(self, config: RejConfigV2):
        super().__init__()
        self.config = config
        self.rank = config.subspace_rank
        self.n_named = config.n_named_concepts

    def forward(
        self,
        concept_mean: torch.Tensor,
        concept_log_std: torch.Tensor,
        controls: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply geometric controls to probabilistic concept coordinates.

        Args:
            concept_mean: (B, K, rank)
            concept_log_std: (B, K, rank)
            controls: dict of tensors each (B, n_named) for modes in config.control_modes.

        Returns:
            new_mean, new_log_std
        """
        mean = concept_mean.clone()
        log_std = concept_log_std.clone()
        n = self.n_named

        if "magnitude" in controls:
            mag = controls["magnitude"].unsqueeze(-1)  # (B, n_named, 1)
            mean[:, :n] = mean[:, :n] * (1.0 + mag)

        if "direction" in controls:
            # direction is a vector in R^rank per named concept.
            dir_vec = controls["direction"]  # (B, n_named, rank)
            mean[:, :n] = mean[:, :n] + dir_vec

        if "uncertainty" in controls:
            unc = controls["uncertainty"].unsqueeze(-1)  # (B, n_named, 1)
            log_std[:, :n] = log_std[:, :n] + unc

        if "select" in controls:
            # Soft selection mask applied to mean.
            sel = controls["select"].unsqueeze(-1)  # (B, n_named, 1)
            mean[:, :n] = mean[:, :n] * torch.sigmoid(sel)

        return mean, log_std


class ConceptAlignmentHead(nn.Module):
    """Predict named-concept control magnitudes from concept embeddings.

    This bakes the Rej-1B concept-alignment idea into the model: the concept
    stream should contain enough information to reconstruct the control labels
    that were injected during training.
    """

    def __init__(self, d_concept: int, n_named_concepts: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_concept),
            nn.Linear(d_concept, d_concept),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_concept, n_named_concepts),
        )

    def forward(self, concept_embedding: torch.Tensor) -> torch.Tensor:
        """Args: concept_embedding (B, K, d_concept) -> predicted_controls (B, n_named)."""
        # Aggregate over concepts and predict control magnitudes.
        pooled = concept_embedding.mean(dim=1)  # (B, d_concept)
        return self.net(pooled)

