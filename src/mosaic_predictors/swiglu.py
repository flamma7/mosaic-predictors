"""Persistent memory: dense SwiGLU, residual width unchanged.

SwiGLU(x) = W_out (SiLU(W_gate x) * W_up x).

Pre-normalization, action-conditioned AdaLN, and the residual dropout live
in the memory block. That AdaLN emits shift, scale, and a residual gate.
swiglu_scale_hidden false keeps hidden_dim as the SwiGLU width. True applies
round_up(int(2 * hidden_dim / 3), 256), so 2048 becomes 1536.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Round-up multiple used when swiglu_scale_hidden is true.
_SWIGLU_MULTIPLE_OF = 256


def swiglu_width(hidden_dim: int, scale_hidden: bool = False) -> int:
    """Hidden width of the three SwiGLU matrices.

    ``scale_hidden`` false keeps ``hidden_dim``. True applies
    ``int(2 * hidden_dim / 3)`` and rounds up to 256. 2048 becomes 1536.
    """
    if not scale_hidden:
        return hidden_dim
    multiple_of = _SWIGLU_MULTIPLE_OF
    shrunk = int(2 * hidden_dim / 3)
    width = multiple_of * ((shrunk + multiple_of - 1) // multiple_of)
    if width < 1:
        raise ValueError(
            f'SwiGLU width for hidden_dim={hidden_dim} with scaling is {width}'
        )
    return width


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        scale_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = swiglu_width(hidden_dim, scale_hidden)
        self.w_gate = nn.Linear(dim, self.hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, self.hidden_dim, bias=False)
        self.w_out = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))
