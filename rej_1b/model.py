"""Rej Representation-Native Model (RNM) architecture."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RejConfig


class FlexibleMultiHeadAttention(nn.Module):
    """Multi-head attention supporting different query/key/value dimensions.

    Args:
        q_dim: dimension of the query input.
        kv_dim: dimension of the key/value input.
        out_dim: dimension of the output projection.
        n_heads: number of attention heads.
        d_head: dimension of each head.
        dropout: dropout probability.
        causal: whether to apply a causal mask.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        out_dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.inner_dim = n_heads * d_head
        self.causal = causal
        self.dropout = dropout

        self.q_proj = nn.Linear(q_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, out_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (B, T_q, q_dim)
            key: (B, T_kv, kv_dim)
            value: (B, T_kv, kv_dim)
            attn_mask: optional additive mask broadcastable to (B, H, T_q, T_kv)

        Returns:
            output: (B, T_q, out_dim)
        """
        B, T_q, _ = query.shape
        _, T_kv, _ = key.shape

        q = self.q_proj(query)  # (B, T_q, H*d)
        k = self.k_proj(key)    # (B, T_kv, H*d)
        v = self.v_proj(value)  # (B, T_kv, H*d)

        q = q.view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)
        # (B, H, T, d)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # (B, H, T_q, T_kv)

        if self.causal and attn_mask is None:
            # Causal mask for self-attention over tokens.
            causal_mask = torch.triu(
                torch.ones(T_q, T_kv, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T_q, d)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.inner_dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """Simple two-layer FFN with GELU activation."""

    def __init__(self, d_in: int, intermediate_size: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class RejLayer(nn.Module):
    """One Rej RNM layer: dual token + concept stream with read/update/write."""

    def __init__(self, config: RejConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        dc = config.d_concept
        h = config.n_heads
        dh = config.d_head

        # Token stream.
        self.token_ln1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.token_self_attn = FlexibleMultiHeadAttention(
            q_dim=d, kv_dim=d, out_dim=d, n_heads=h, d_head=dh,
            dropout=config.dropout, causal=True,
        )
        self.token_ln2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.token_ffn = FeedForward(
            d_in=d,
            intermediate_size=config.intermediate_size,
            d_out=d,
            dropout=config.dropout,
        )

        # Concept stream.
        self.concept_ln1 = nn.LayerNorm(dc, eps=config.layer_norm_eps)
        self.concept_read_tokens = FlexibleMultiHeadAttention(
            q_dim=dc, kv_dim=d, out_dim=dc, n_heads=max(1, h // 2),
            d_head=dh, dropout=config.dropout, causal=False,
        )
        self.concept_ln2 = nn.LayerNorm(dc, eps=config.layer_norm_eps)
        self.concept_self_attn = FlexibleMultiHeadAttention(
            q_dim=dc, kv_dim=dc, out_dim=dc, n_heads=max(1, h // 2),
            d_head=dh, dropout=config.dropout, causal=False,
        )
        self.concept_ln3 = nn.LayerNorm(dc, eps=config.layer_norm_eps)
        self.concept_ffn = FeedForward(
            d_in=dc,
            intermediate_size=4 * dc,
            d_out=dc,
            dropout=config.dropout,
        )

        # Cross-stream: tokens read concepts.
        self.token_ln3 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.token_read_concepts = FlexibleMultiHeadAttention(
            q_dim=d, kv_dim=dc, out_dim=d, n_heads=h, d_head=dh,
            dropout=config.dropout, causal=False,
        )
        # Learned gate for the concept-to-token contribution, initialized to 0.
        # The token stream therefore starts as a standard transformer; the gate
        # grows only if the concept stream actually helps prediction.
        self.concept_to_token_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        tokens: torch.Tensor,
        concepts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one layer.

        Args:
            tokens: (B, T, d_model)
            concepts: (B, K, d_concept)

        Returns:
            tokens: (B, T, d_model)
            concepts: (B, K, d_concept)
        """
        # 1. Tokens attend to tokens.
        tokens = tokens + self.token_self_attn(
            self.token_ln1(tokens),
            self.token_ln1(tokens),
            self.token_ln1(tokens),
        )

        # 2. Concepts read from tokens.
        concepts = concepts + self.concept_read_tokens(
            self.concept_ln1(concepts),
            tokens,
            tokens,
        )

        # 3. Concepts update themselves.
        concepts = concepts + self.concept_self_attn(
            self.concept_ln2(concepts),
            self.concept_ln2(concepts),
            self.concept_ln2(concepts),
        )

        # 4. Concept FFN.
        concepts = concepts + self.concept_ffn(self.concept_ln3(concepts))

        # 5. Tokens read from concepts (gated residual).
        tokens = tokens + self.concept_to_token_gate * self.token_read_concepts(
            self.token_ln3(tokens),
            concepts,
            concepts,
        )

        # 7. Token FFN.
        tokens = tokens + self.token_ffn(self.token_ln2(tokens))

        return tokens, concepts


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
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (B, T) token indices.
            controls: (B, n_named_concepts) scalar controls, or None.
            return_concept_trace: if True, collect and return concept states per layer.

        Returns:
            dict with keys:
                "logits": (B, T, vocab_size)
                "concept_trace": list of (B, K, d_concept) tensors if requested, else None
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
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def named_concept_labels(self) -> list:
        return list(self.config.named_concepts)
