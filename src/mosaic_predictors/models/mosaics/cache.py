"""Per-candidate incremental memory.

Nothing here is stored on the predictor module. A cache owns, for every
block and every batch row (one CEM candidate):

- the unnormalized gated-key state
- completed (key, value) pairs
- the latest key whose value is still pending the next state

Batch rows do not share storage. reset_cache drops that history.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockCache:
    """Memory state of one predictor block."""

    recurrent: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    pending_key: torch.Tensor | None = None

    @property
    def completed(self) -> int:
        return int(self.keys.shape[1])


@dataclass
class PredictorCache:
    """Memory state of every block, aligned across the batch."""

    blocks: list[BlockCache]
    batch_size: int
    device: torch.device
    dtype: torch.dtype
    max_len: int | None = None

    @property
    def completed(self) -> int:
        return self.blocks[0].completed


def empty_block_cache(
    batch_size: int,
    heads: int,
    dim_head: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BlockCache:
    recurrent = torch.zeros(
        batch_size, heads, dim_head, device=device, dtype=torch.float32
    )
    keys = torch.zeros(
        batch_size, 0, heads, dim_head, device=device, dtype=dtype
    )
    values = torch.zeros(
        batch_size, 0, heads, dim_head, device=device, dtype=dtype
    )
    return BlockCache(recurrent=recurrent, keys=keys, values=values, pending_key=None)


def init_cache(
    batch_size: int,
    n_blocks: int,
    heads: int,
    dim_head: int,
    device: torch.device | str,
    dtype: torch.dtype,
    max_len: int | None = None,
) -> PredictorCache:
    """Fresh window: zero recurrent state and no completed pairs.

    max_len is recorded for the caller. Storage grows by concatenation, so
    candidates never share a writable buffer.
    """
    blocks = [
        empty_block_cache(batch_size, heads, dim_head, device, dtype)
        for _ in range(n_blocks)
    ]
    return PredictorCache(
        blocks=blocks,
        batch_size=batch_size,
        device=torch.device(device),
        dtype=dtype,
        max_len=max_len,
    )


def reset_cache(cache: PredictorCache, heads: int, dim_head: int) -> PredictorCache:
    """Clear every candidate's hypothetical history in this cache object."""
    fresh = init_cache(
        cache.batch_size,
        n_blocks=len(cache.blocks),
        heads=heads,
        dim_head=dim_head,
        device=cache.device,
        dtype=cache.dtype,
        max_len=cache.max_len,
    )
    cache.blocks = fresh.blocks
    return cache
