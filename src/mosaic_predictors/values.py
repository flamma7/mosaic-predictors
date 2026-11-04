"""Next-state values.

v = normalize(W_v LayerNorm(h)).

This module is the only place that decides which features become a memory
value. It applies a standard LayerNorm and a learned projection, then L2
normalizes per head. It does not see actions, AdaLN, or a current/next mix.

Callers pair the result with keys: the value stored at index t is psi(h_{t+1}),
computed by calling this module on the residual at t+1. Retrieval never
builds that pairing itself, so another feature source can replace this class.
"""

from __future__ import annotations

import torch
from torch import nn

_NORM_EPS = 1e-6


class NextStateValues(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        # Affine LayerNorm: learned scale (weight) and shift (bias). No gate.
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.proj = nn.Linear(dim, heads * dim_head, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """psi(z) for each timestep. z is (B, T, D); result is (B, T, H, d)."""
        batch, length, _ = z.shape
        hidden = self.proj(self.norm(z)).view(batch, length, self.heads, self.dim_head)
        hidden_f = hidden.float()
        normalized = hidden_f / (hidden_f.norm(dim=-1, keepdim=True) + _NORM_EPS)
        return normalized.to(dtype=z.dtype)
