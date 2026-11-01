"""One residual Memory Mosaics block.

Order, matching the predictor spec:

1. AdaLN on the residual stream, conditioned on the action embedding.
   Shift and scale are zero-initialized. There is no residual gate.
2. Gated recurrent key from that action-conditioned feature.
3. Associative read of completed pairs. Values come from NextStateValues
   applied to this block's incoming residual, shifted so pair t stores
   psi(h_{t+1}).
4. Residual update of the memory output, with LeWM-style dropout.
5. Action-conditioned AdaLN, then SwiGLU persistent memory, and a second
   residual update. This map is separate from the key AdaLN and emits
   shift, scale, and a residual gate, all zero-initialized. The gate
   multiplies the SwiGLU output, so the branch starts closed. The first
   query has an empty memory, so this is the path that lets that action
   affect the prediction.

The value is read from that residual before AdaLN, and the pair is visible
only to later queries. This block's own a_{t+1} is therefore not part of
the value. The residual arriving from an earlier block can already depend
on a_{t+1}, because that block wrote its action-conditioned update into
the same time index.
"""

from __future__ import annotations

import torch
from torch import nn

from mosaic_predictors.models.mosaics.cache import BlockCache, empty_block_cache
from mosaic_predictors.models.mosaics.gated_keys import GatedKeyExtractor
from mosaic_predictors.models.mosaics.memory import AssociativeMemory
from mosaic_predictors.models.mosaics.swiglu import SwiGLU
from mosaic_predictors.models.mosaics.values import NextStateValues


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class MemoryBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim

        self.key_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self.gated_keys = GatedKeyExtractor(dim, heads, dim_head)
        self.values = NextStateValues(dim, heads, dim_head)
        self.memory = AssociativeMemory(dim, heads, dim_head)

        self.swiglu_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.swiglu_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        nn.init.constant_(self.swiglu_modulation[-1].weight, 0)
        nn.init.constant_(self.swiglu_modulation[-1].bias, 0)
        self.swiglu = SwiGLU(dim, mlp_dim)
        self.resid_drop = nn.Dropout(dropout)

    def _empty_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> BlockCache:
        return empty_block_cache(batch, self.heads, self.dim_head, device, dtype)

    def _bank(
        self,
        keys: torch.Tensor,
        states: torch.Tensor,
        cache: BlockCache,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Completed pairs the new queries may read.

        A pending key from the previous step is completed by psi(h[:, 0])
        before the new query runs. Within the chunk, key i is completed by
        psi(h[:, i+1]) and is visible only to later queries. The last new
        key stays pending. `states` is the residual before this block's AdaLN.
        """
        psi = self.values(states)
        pieces_k: list[torch.Tensor] = []
        pieces_v: list[torch.Tensor] = []
        n0 = cache.keys.shape[1]
        if n0:
            pieces_k.append(cache.keys)
            pieces_v.append(cache.values)
        if cache.pending_key is not None:
            pieces_k.append(cache.pending_key.unsqueeze(1))
            pieces_v.append(psi[:, :1])
            n0 += 1
        if keys.shape[1] > 1:
            pieces_k.append(keys[:, :-1])
            pieces_v.append(psi[:, 1:])
        if not pieces_k:
            return keys[:, :0], psi[:, :0], 0
        return torch.cat(pieces_k, dim=1), torch.cat(pieces_v, dim=1), n0

    def forward(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
        cache: BlockCache | None = None,
    ) -> tuple[torch.Tensor, BlockCache]:
        """h, action: (B, T, D). Key t stores psi of h at t+1, before AdaLN."""
        if cache is None:
            cache = self._empty_cache(h.shape[0], h.device, h.dtype)
        if h.shape[0] != cache.recurrent.shape[0]:
            raise ValueError('cache batch does not match the residual stream')

        shift, scale = self.adaLN_modulation(action).chunk(2, dim=-1)
        u = modulate(self.key_norm(h), shift, scale)
        keys, recurrent = self.gated_keys(u, cache.recurrent)
        # Read values before h is updated. Key t gets psi(h_{t+1}) from the
        # residual before this block's AdaLN, so this block's a_{t+1} is not
        # part of the value. A later block's incoming residual already
        # includes earlier blocks' update at that same index.
        bank_k, bank_v, n0 = self._bank(keys, h, cache)
        memory_out = self.memory(keys, bank_k, bank_v, n0)

        h = h + self.resid_drop(memory_out)
        shift_mlp, scale_mlp, gate_mlp = self.swiglu_modulation(action).chunk(3, dim=-1)
        persistent = modulate(self.swiglu_norm(h), shift_mlp, scale_mlp)
        h = h + gate_mlp * self.resid_drop(self.swiglu(persistent))

        updated = BlockCache(
            recurrent=recurrent,
            keys=bank_k,
            values=bank_v,
            pending_key=keys[:, -1],
        )
        return h, updated
