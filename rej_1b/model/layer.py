"""One Rej layer, and how the concept stream is carried from one step of
depth to the next."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RejConfig
from .blocks import FeedForward, FlexibleMultiHeadAttention

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
            intermediate_size=FFN_EXPANSION * dc,
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


class ConceptCarry(nn.Module):
    """Gated fusion of the previous step's concept state into the next step's.

    The concept stream is a state over depth: it is built once from the learned
    concept embeddings, updated layer by layer, and then discarded. Across
    decoding steps nothing survives it but the sampled token, so a `K x
    d_concept` state the model has just paid to compute is thrown away at every
    step boundary. That is the same channel a standard transformer leaves
    narrow, and depth recurrence alone cannot stand in for it: to realize an
    update of the form `c(t+1) = f(c(t), x(t))` the state has to reach the next
    step, not merely a deeper layer of this one.

    The fusion is a gated linear unit over the pair `(initial, carried)`. The
    value projection is zero-initialized, so the carry contributes exactly
    nothing until it is trained: a checkpoint written before this module
    existed produces identical logits with the flag turned on. The gate is not
    zero-initialized, so it receives gradient as soon as the value projection
    moves off zero.
    """

    def __init__(self, d_concept: int):
        super().__init__()
        self.value = nn.Linear(2 * d_concept, d_concept)
        self.gate = nn.Linear(2 * d_concept, d_concept)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, initial: torch.Tensor, carried: torch.Tensor) -> torch.Tensor:
        """Fuse a carried concept state into an initial one.

        Args:
            initial: (B, K, d_concept) state built for this step.
            carried: (B, K, d_concept) final state of the previous step.

        Returns:
            (B, K, d_concept) fused initial state.
        """
        pair = torch.cat([initial, carried], dim=-1)
        return initial + torch.sigmoid(self.gate(pair)) * self.value(pair)


