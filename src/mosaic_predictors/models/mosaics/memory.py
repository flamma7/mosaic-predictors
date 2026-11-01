"""Full-window associative retrieval.

For a query at position t, with n completed pairs available:

    y_t = sum_{i < n} softmax_i(beta(n) * k_t · k_i) v_i

n is the number of causally completed pairs, not the configured window
length. n = 0 returns zeros (no softmax over an empty bank). The query
never reads its own unfinished value: callers pass a bank that contains
only pairs whose values are already known, and this module slices that
bank to the first n entries for each query.

One bank per block. No short/long split, no eviction, no positional encoding.
"""

from __future__ import annotations

import torch
from torch import nn

from mosaic_predictors.models.mosaics.bandwidth import AdaptiveBandwidth


class AssociativeMemory(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.bandwidth = AdaptiveBandwidth(heads)
        self.proj_out = nn.Linear(heads * dim_head, dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        bank_keys: torch.Tensor,
        bank_values: torch.Tensor,
        n0: int,
    ) -> torch.Tensor:
        """Read the bank.

        query: (B, T, H, d) keys of the new tokens.
        bank_*: (B, M, H, d) completed pairs visible to this chunk.
        n0: how many of those pairs the first new query may read.
        Query-local index t reads exactly the first n0 + t pairs.
        The bank length must be n0 + T - 1 (every pair the last query can
        read, and nothing beyond it).
        """
        # query is (B, T, H, d): one gated key per frame. TwoRoom training
        # feeds T=3 frames, z0, z1, z2, and starts from an empty cache (n0=0).
        batch, length, heads, dim_head = query.shape
        # bank_keys / bank_values are the completed pairs, (B, M, H, d).
        # For that window the block stores exactly two:
        #   pair 0: key(z0, a0) -> psi(z1)
        #   pair 1: key(z1, a1) -> psi(z2)
        # key(z2, a2) is still pending; psi(z0) is never stored as a value.
        bank_len = bank_keys.shape[1]
        # Query t may read the first n0+t pairs, so the last query needs
        # n0+T-1 of them and the bank stops there. Training: 0+3-1 = 2.
        expected = n0 + length - 1
        if bank_len != expected:
            raise ValueError(
                f'memory bank length {bank_len} != n0 + T - 1 ({expected})'
            )
        # Values are aligned with keys: index i is the same pair in both.
        if bank_values.shape[1] != bank_len:
            raise ValueError('key and value banks have different lengths')

        # A fresh first frame has M=0. There is nothing to attend over, so
        # every query in this chunk gets a zero vector before proj_out.
        if bank_len == 0:
            mixed = query.new_zeros(batch, length, heads, dim_head)
            # (B, T, H, d) -> (B, T, H*d) -> (B, T, hidden_dim).
            return self.proj_out(mixed.reshape(batch, length, heads * dim_head))

        # Pairs visible at each query. Training counts are [0, 1, 2]:
        # z0 sees nothing, z1 sees pair 0, z2 sees both pairs.
        counts = n0 + torch.arange(length, device=query.device)
        # Per-head temperature beta(n) = b1 * n^alpha + b0, shape (T, H, 1).
        # A larger count sharpens the softmax over the longer bank.
        beta = self.bandwidth(counts)
        # Dots and softmax run in fp32; the residual stream may be fp16.
        query_f = query.float()
        keys_f = bank_keys.float()
        values_f = bank_values.float()
        outputs = []
        for step in range(length):
            # Training: step 0 -> 0 pairs, step 1 -> 1, step 2 -> 2.
            # A later rollout step arrives with n0>0 and T=1, so this is
            # just n0 and the single query reads the whole bank.
            count = n0 + step
            if count == 0:
                # z0's prediction has no memory yet.
                outputs.append(query_f.new_zeros(batch, heads, dim_head))
                continue
            # Query key against the first `count` stored keys.
            # (B, H, d) x (B, count, H, d) -> (B, H, count), one score per pair.
            dots = torch.einsum(
                'bhd,bmhd->bhm', query_f[:, step], keys_f[:, :count]
            )
            # beta[step] is (H, 1); broadcast over batch and the pair axis.
            scale = beta[step].view(1, heads, 1)
            # Softmax over stored pairs. At step 1 this is a single pair, so
            # the weight is 1 and the read returns psi(z1). At step 2 the
            # query key(z2, a2) mixes psi(z1) and psi(z2).
            attn = torch.softmax(dots * scale, dim=-1)
            # (B, H, count) x (B, count, H, d) -> (B, H, d).
            outputs.append(torch.einsum('bhm,bmhd->bhd', attn, values_f[:, :count]))
        # Stack the per-frame reads back into (B, T, H, d) and restore dtype.
        mixed = torch.stack(outputs, dim=1).to(dtype=query.dtype)
        # Merge heads, then project to the residual width (B, T, hidden_dim).
        return self.proj_out(mixed.reshape(batch, length, heads * dim_head))
