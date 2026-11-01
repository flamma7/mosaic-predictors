"""Six-block action-conditioned Memory Mosaics predictor.

Drop-in shape for LeWM's Predictor.forward(x, c):

    x: (B, T, input_dim) encoder embeddings
    c: (B, T, input_dim) action embeddings already produced by the SWM encoder

No positional embedding. The residual width stays `hidden_dim`; heads of
`dim_head` are an internal projection. Training calls forward without a
cache (one window, empty memory). Rollouts call init_cache / prefill / step
/ reset_cache. Cached steps match a full-window forward when dropout is off.
"""

from __future__ import annotations

import torch
from torch import nn

from mosaic_predictors.models.mosaics.block import MemoryBlock
from mosaic_predictors.models.mosaics.cache import (
    PredictorCache,
    init_cache,
    reset_cache,
)


class MosaicsPredictor(nn.Module):
    def __init__(
        self,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        gated_keys: bool = True,
        action_conditioning: str = 'adaln',
        value_conditioning: str = 'layernorm',
        value_source: str = 'next_state',
        value_interpolation: bool = False,
        memory_type: str = 'full_window',
        adaptive_bandwidth: bool = True,
    ) -> None:
        super().__init__()
        self._check_config(
            gated_keys,
            action_conditioning,
            value_conditioning,
            value_source,
            value_interpolation,
            memory_type,
            adaptive_bandwidth,
        )
        self.num_frames = num_frames
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.dropout_p = dropout
        self.emb_dropout = emb_dropout

        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                MemoryBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, self.output_dim)
            if hidden_dim != self.output_dim
            else nn.Identity()
        )

    @staticmethod
    def _check_config(
        gated_keys: bool,
        action_conditioning: str,
        value_conditioning: str,
        value_source: str,
        value_interpolation: bool,
        memory_type: str,
        adaptive_bandwidth: bool,
    ) -> None:
        if not gated_keys:
            raise ValueError('gated_keys must be true')
        if action_conditioning != 'adaln':
            raise ValueError("action_conditioning must be 'adaln'")
        if value_conditioning != 'layernorm':
            raise ValueError("value_conditioning must be 'layernorm'")
        if value_source != 'next_state':
            raise ValueError("value_source must be 'next_state'")
        if value_interpolation:
            raise ValueError('value_interpolation must be false')
        if memory_type != 'full_window':
            raise ValueError("memory_type must be 'full_window'")
        if not adaptive_bandwidth:
            raise ValueError('adaptive_bandwidth must be true')

    def init_cache(
        self,
        batch_size: int,
        max_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PredictorCache:
        """Independent empty memory for each of batch_size candidates."""
        param = next(self.parameters())
        if device is None:
            device = param.device
        if dtype is None:
            dtype = param.dtype
        return init_cache(
            batch_size,
            n_blocks=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            device=device,
            dtype=dtype,
            max_len=max_len,
        )

    def reset_cache(self, cache: PredictorCache) -> PredictorCache:
        """Drop completed pairs, pending values, and gated recurrent state."""
        return reset_cache(cache, heads=self.heads, dim_head=self.dim_head)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache: PredictorCache | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, PredictorCache]:
        """Predict one next embedding per input step.

        Without a cache, the window starts empty (training). With a cache,
        the same causal read continues from that candidate state and the
        cache object is updated in place.
        """
        if x.ndim != 3 or c.ndim != 3:
            raise ValueError('x and c must have shape (B, T, D)')
        if x.shape[0] != c.shape[0] or x.shape[1] != c.shape[1]:
            raise ValueError('x and c must share batch and time length')
        if cache is not None and cache.batch_size != x.shape[0]:
            raise ValueError(
                f'cache batch {cache.batch_size} does not match x batch {x.shape[0]}'
            )

        h = self.input_proj(self.dropout(x))
        cond = self.cond_proj(c)
        block_in = None if cache is None else cache.blocks
        updated: list = []
        for i, block in enumerate(self.blocks):
            incoming = None if block_in is None else block_in[i]
            # Each block stores psi of its own incoming residual at the next
            # index. The block reads that residual before AdaLN.
            h, block_cache = block(h, cond, incoming)
            if cache is not None:
                updated.append(block_cache)
        out = self.output_proj(self.norm(h))
        if cache is None:
            return out
        cache.blocks = updated
        return out, cache

    def prefill(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache: PredictorCache,
    ) -> tuple[torch.Tensor, PredictorCache]:
        """Run a context window and keep its completed pairs."""
        out = self.forward(x, c, cache)
        assert isinstance(out, tuple)
        return out

    def step(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache: PredictorCache,
    ) -> tuple[torch.Tensor, PredictorCache]:
        """One new state and the action that leaves it.

        x and c are (B, D) or (B, 1, D). Returns the prediction (B, D)
        and the updated cache. The new state completes the previous pair
        before it is used as a query.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)
            c = c.unsqueeze(1)
        if x.shape[1] != 1:
            raise ValueError('step expects a single timestep; use prefill for a chunk')
        out, cache = self.prefill(x, c, cache)
        if squeeze:
            out = out[:, 0]
        return out, cache
