"""Learnable MMv2 retrieval bandwidth.

beta(n) = b1 * n^alpha + b0, with b0 >= 0, b1 > 0, and alpha in [0, 1].

Parameterization matches the Memory Mosaics reference: positive scales are
stored as logs, the exponent is an absolute value capped at 1, and the
schedule is evaluated in fp32. n is the number of completed memory pairs
the query is allowed to read.
"""

from __future__ import annotations

import torch
from torch import nn

# Same clip the reference uses before exponentiating log-bandwidth.
_MAX_LOG_COEFFICIENT = 10.0


class AdaptiveBandwidth(nn.Module):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.bandwidth_bias = nn.Parameter(torch.empty(n_heads))
        self.bandwidth_scale = nn.Parameter(torch.empty(n_heads))
        self.bandwidth_exponent = nn.Parameter(torch.empty(n_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.bandwidth_bias, 1.5)
        nn.init.constant_(self.bandwidth_scale, 1.5)
        nn.init.constant_(self.bandwidth_exponent, 0.33333)

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(b0, b1, alpha), each shape (n_heads,), in fp32."""
        b0 = torch.exp(self.bandwidth_bias.float().clamp(max=_MAX_LOG_COEFFICIENT))
        b1 = torch.exp(self.bandwidth_scale.float().clamp(max=_MAX_LOG_COEFFICIENT))
        alpha = self.bandwidth_exponent.float().abs().clamp(max=1)
        return b0, b1, alpha

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """beta(n) with shape (..., n_heads, 1) for a count tensor n of shape (...)."""
        b0, b1, alpha = self.coefficients()
        counts = n.to(dtype=torch.float32).clamp_min(0).unsqueeze(-1)
        beta = b1 * torch.pow(counts, alpha) + b0
        return beta.unsqueeze(-1)
