"""Persistent memory: dense SwiGLU, width mlp_dim, residual width unchanged.

SwiGLU(x) = W_out (SiLU(W_gate x) * W_up x).

Pre-normalization, action-conditioned AdaLN, and the residual dropout live
in the memory block. That AdaLN emits shift, scale, and a residual gate.
No 2/3-width rounding: mlp_dim is the hidden width.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))
