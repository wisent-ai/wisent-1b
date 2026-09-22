"""What a concept is in this model: a subspace with a basis and a centroid,
a Gaussian state in its coordinates, and the router that decides which
tokens each concept reads from."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RejConfigV2

# One half, from the closed form of KL(N(mean, std^2) || N(0, 1)).
KL_HALF = 0.5
# A direction control is checked on its last two dimensions: named concepts by subspace rank.
DIRECTION_SHAPE_DIMENSIONS = 2


def orthonormalize(x: torch.Tensor) -> torch.Tensor:
    """Orthonormalize rows of x via QR decomposition.

    Args:
        x: (..., rank, d) where rank <= d.

    Returns:
        orthonormal rows of same shape.
    """
    *batch, rank, d = x.shape
    x2 = x.reshape(-1, rank, d)
    q, _ = torch.linalg.qr(x2.transpose(-2, -1), mode="reduced")
    q = q.transpose(-2, -1)
    return q.reshape(*batch, rank, d)


class SubspaceConceptBank(nn.Module):
    """Bank of concept subspaces: basis + centroid per concept slot."""

    def __init__(self, config: RejConfigV2):
        super().__init__()
        self.config = config
        self.rank = config.subspace_rank
        self.d_concept = config.d_concept
        self.n_concepts = config.n_concepts

        # Raw basis parameters; orthonormalized in forward.
        self.basis_raw = nn.Parameter(
            torch.randn(config.n_concepts, config.subspace_rank, config.d_concept)
            * config.initializer_range
        )
        self.centroid = nn.Parameter(
            torch.randn(config.n_concepts, config.d_concept) * config.initializer_range
        )

    def get_basis(self) -> torch.Tensor:
        """Return orthonormal concept bases (n_concepts, rank, d_concept)."""
        if self.config.normalize_subspace_basis:
            return orthonormalize(self.basis_raw)
        return self.basis_raw

    def project_to_subspace(
        self, x: torch.Tensor, basis: torch.Tensor
    ) -> torch.Tensor:
        """Project d_concept vectors into subspace coordinates.

        Args:
            x: (B, K, d_concept)
            basis: (K, rank, d_concept)

        Returns:
            coords: (B, K, rank)
        """
        # (B, K, d) @ (K, r, d)^T -> (B, K, r)
        return torch.einsum("bkd,krd->bkr", x, basis)

    def project_from_subspace(
        self, coords: torch.Tensor, basis: torch.Tensor
    ) -> torch.Tensor:
        """Map subspace coordinates back to d_concept space.

        Args:
            coords: (B, K, rank)
            basis: (K, rank, d_concept)

        Returns:
            x: (B, K, d_concept)
        """
        # (B, K, r) @ (K, r, d) -> (B, K, d)
        return torch.einsum("bkr,krd->bkd", coords, basis) + self.centroid.unsqueeze(0)


class ProbabilisticConceptState:
    """Gaussian concept state in subspace coordinates."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean  # (B, K, rank)
        self.log_std = log_std  # (B, K, rank)

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def sample(self, deterministic: bool = False) -> torch.Tensor:
        """Reparameterized sample of subspace coordinates."""
        if deterministic:
            return self.mean
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def kl_divergence(self) -> torch.Tensor:
        """KL from N(mean, std^2) to N(0, 1), summed over batch/concepts/rank."""
        var = self.std.pow(2)
        kl = KL_HALF * (self.mean.pow(2) + var - 1.0 - 2.0 * self.log_std)
        return kl.sum()

    def deterministic_coords(self) -> torch.Tensor:
        return self.mean


class ConceptRouter(nn.Module):
    """Per-token router assigning relevance weights to each concept."""

    def __init__(self, d_model: int, n_concepts: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Linear(d_model, n_concepts, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens (B, T, d_model) -> weights (B, K, T)."""
        # (B, T, d) @ (d, K) -> (B, T, K) -> (B, K, T)
        logits = self.query(tokens).transpose(1, 2)
        weights = F.softmax(logits, dim=-1)
        return self.dropout(weights)


