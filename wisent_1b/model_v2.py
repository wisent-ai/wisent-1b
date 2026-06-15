"""Advanced Wisent RNM v2: geometric concepts baked into the architecture.

Concepts are no longer single vectors. Each concept is a subspace (basis + centroid)
optionally carrying a probabilistic state (mean + std). Concept dynamics are non-linear
MLP cells. An input-dependent router decides which concepts read from which tokens.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WisentConfigV2
from .model import FeedForward, FlexibleMultiHeadAttention


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

    def __init__(self, config: WisentConfigV2):
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
        kl = 0.5 * (self.mean.pow(2) + var - 1.0 - 2.0 * self.log_std)
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


class NonlinearConceptCell(nn.Module):
    """MLP-based read/update/write for concept dynamics."""

    def __init__(self, config: WisentConfigV2):
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


class GeometricControl(nn.Module):
    """Maps user controls to subspace-coordinate concept modifications."""

    def __init__(self, config: WisentConfigV2):
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


class WisentRNMv2(nn.Module):
    """Wisent Representation-Native Model v2 with geometric concepts."""

    def __init__(self, config: WisentConfigV2):
        super().__init__()
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
        if controls is None:
            return {}
        for mode, tensor in controls.items():
            if mode not in self.config.control_modes:
                raise ValueError(f"Unknown control mode '{mode}'. Allowed: {self.config.control_modes}")
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != self.config.n_named_concepts and mode != "direction":
                raise ValueError(
                    f"Control '{mode}' last dim ({tensor.shape[-1]}) must match "
                    f"n_named_concepts ({self.config.n_named_concepts})"
                )
            if mode == "direction" and tensor.shape[-2:] != (self.config.n_named_concepts, self.config.subspace_rank):
                raise ValueError(
                    f"Control 'direction' must have shape (B, n_named, subspace_rank) = "
                    f"(..., {self.config.n_named_concepts}, {self.config.subspace_rank})"
                )
        return controls

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
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, T)
            controls: dict of control tensors per mode.
            return_concept_trace: if True, collect concept states per layer.
            deterministic: if True, do not sample concept state.

        Returns:
            dict with "logits", optional "concept_trace", and optional "kl_loss".
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
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def named_concept_labels(self) -> list:
        return list(self.config.named_concepts)
