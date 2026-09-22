"""The pieces every Rej layer is built from: attention that allows
different query and value widths, and the two-layer feed-forward."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# A transformer feed-forward layer widens by this factor, the usual choice since the original design.
FFN_EXPANSION = 4


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


