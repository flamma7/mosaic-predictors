"""MMv2 gated recurrent keys, Equation (8).

For each head:

    k_tilde_t = W_phi u_t
    g_t       = exp(W_g u_t)            positive, one scalar per head
    lambda_t  = exp(-|W_lambda u_t|)    in (0, 1], one scalar per head
    k_bar_t   = g_t * k_tilde_t + lambda_t * k_bar_{t-1}
    k_t       = k_bar_t / (||k_bar_t||_2 + eps)

k_bar_{-1} = 0 at the start of every window. The gate and decay broadcast
over the head feature axis. The scan is a PyTorch reference (differentiable,
and the same code serves a full window or one cached step).
"""

from __future__ import annotations

import torch
from torch import nn

_EXP_CLIP = 15.0
_NORM_EPS = 1e-6


def normalize_keys(k_bar: torch.Tensor, eps: float = _NORM_EPS) -> torch.Tensor:
    """Unit-norm keys. k_bar is (..., d) and the norm is over the last axis."""
    return k_bar / (k_bar.norm(dim=-1, keepdim=True) + eps)


class GatedKeyExtractor(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.proj_phi = nn.Linear(dim, heads * dim_head, bias=False)
        self.proj_gate = nn.Linear(dim, heads, bias=False)
        self.proj_decay = nn.Linear(dim, heads, bias=False)

    def forward(
        self, u: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan a chunk.

        u: (B, T, D) action-conditioned features.
        state: (B, H, d) unnormalized k_bar from the previous step, or None
        to start from zero. Returned state is always fp32 so a later step
        continues the same accumulation as a single full-window scan.

        Returns normalized keys (B, T, H, d) in u's dtype, and the fp32 state.
        """
        batch, length, _ = u.shape
        heads, dim_head = self.heads, self.dim_head
        k_tilde = self.proj_phi(u).view(batch, length, heads, dim_head).float()
        gate = torch.exp(self.proj_gate(u).float().clamp(max=_EXP_CLIP)).unsqueeze(-1)
        decay = torch.exp(-self.proj_decay(u).float().abs()).unsqueeze(-1)

        if state is None:
            acc = k_tilde.new_zeros(batch, heads, dim_head)
        else:
            acc = state.float()

        keys = []
        for t in range(length):
            acc = gate[:, t] * k_tilde[:, t] + decay[:, t] * acc
            keys.append(normalize_keys(acc))
        stacked = torch.stack(keys, dim=1).to(dtype=u.dtype)
        return stacked, acc
