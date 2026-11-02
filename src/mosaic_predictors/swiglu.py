"""Persistent memory: dense SwiGLU, residual width unchanged.

SwiGLU(x) = W_out (SiLU(W_gate x) * W_up x).

Pre-normalization, action-conditioned AdaLN, and the residual dropout live
in the memory block. That AdaLN emits shift, scale, and a residual gate.
With swiglu_multiple_of unset, mlp_dim is the hidden width. Set it to a
positive int and the width becomes round_up(int(2 * mlp_dim / 3), that).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def swiglu_width(hidden_dim: int, multiple_of: int | None = None) -> int:
    """Hidden width of the three SwiGLU matrices.

    ``multiple_of`` unset keeps ``hidden_dim``. A positive int applies the
    usual shrink, ``int(2 * hidden_dim / 3)``, then rounds up to that
    multiple. 2048 with 256 is 1536.
    """
    if multiple_of is None:
        return hidden_dim
    if (
        isinstance(multiple_of, bool)
        or not isinstance(multiple_of, int)
        or multiple_of < 1
    ):
        raise ValueError(
            f'swiglu_multiple_of must be a positive int or null, got {multiple_of!r}'
        )
    shrunk = int(2 * hidden_dim / 3)
    width = multiple_of * ((shrunk + multiple_of - 1) // multiple_of)
    if width < 1:
        raise ValueError(
            f'SwiGLU width for mlp_dim={hidden_dim} and '
            f'swiglu_multiple_of={multiple_of} is {width}'
        )
    return width


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = swiglu_width(hidden_dim, multiple_of)
        self.w_gate = nn.Linear(dim, self.hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, self.hidden_dim, bias=False)
        self.w_out = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))
