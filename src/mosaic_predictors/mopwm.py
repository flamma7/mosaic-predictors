"""LeWM world model with a Memory Mosaics predictor."""

import torch
from einops import rearrange
from stable_worldmodel.wm.lewm.lewm import LeWM


class MoPWM(LeWM):
    """LeWM world model with a Memory Mosaics predictor.

    encode() and predict() are the LeWM implementations: the same encoder,
    action encoder, projectors, and tensor layout. rollout() keeps LeWM's
    context/candidate action pairing and writes predicted_emb the same way.
    Hypothetical futures use a per-candidate cache. max_memory is how many
    completed pairs that cache keeps; older pairs are dropped once the
    length would pass it. None keeps the full causal history. history_size
    is accepted because planners pass it; it does not set the cap.
    """

    def __init__(self, *args, max_memory: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_memory is not None and max_memory < 0:
            raise ValueError(
                f'max_memory must be non-negative, got {max_memory}'
            )
        self.max_memory = max_memory

    def _project_pred(self, preds: torch.Tensor) -> torch.Tensor:
        """Apply pred_proj the way LeWM.predict does, without a second predictor call."""
        single = preds.ndim == 2
        if single:
            preds = preds.unsqueeze(1)
        batch = preds.size(0)
        flat = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        out = rearrange(flat, '(b t) d -> b t d', b=batch)
        return out[:, 0] if single else out

    def rollout(
        self,
        info,
        action_sequence,
        history_size: int = None,
        max_memory: int | None = None,
    ):
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)
        # LeWM planners pass history_size. The mosaic cap is max_memory:
        # completed (key, value) pairs kept per candidate. The argument
        # overrides the model attribute. None on both keeps every pair.
        _ = history_size
        if max_memory is None:
            max_memory = getattr(self, 'max_memory', None)
        if max_memory is not None and max_memory < 0:
            raise ValueError(
                f'max_memory must be non-negative, got {max_memory}'
            )

        assert 'pixels' in info, 'pixels not in info_dict'
        horizon_ctx = info['pixels'].size(2)
        batch, samples, steps = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                batch, samples, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == horizon_ctx - 1, (
            f'action_history must hold H-1={horizon_ctx - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        # Action paired with context frame k leaves that frame. The current
        # frame (k = H-1) is paired with the first candidate.
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            initial = {
                k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)
            }
            initial = self.encode(initial)
            info['emb'] = (
                initial['emb'].detach().unsqueeze(1).expand(batch, samples, -1, -1)
            )

        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...').contiguous()
        act_past_flat = rearrange(act_past, 'b s ... -> (b s) ...')
        act_cand_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )

        flat = batch * samples
        cache = self.predictor.init_cache(
            flat,
            max_len=max_memory,
            device=emb_init.device,
            dtype=emb_init.dtype,
        )
        ctx_pred, cache = self.predictor.prefill(
            emb_init, all_act_emb[:, :horizon_ctx], cache
        )
        ctx_pred = self._project_pred(ctx_pred)
        emb_list = list(emb_init.unbind(dim=1))
        if steps > 0:
            emb_list.append(ctx_pred[:, -1])
            for t in range(1, steps):
                nxt, cache = self.predictor.step(
                    emb_list[-1], all_act_emb[:, horizon_ctx + t - 1], cache
                )
                emb_list.append(self._project_pred(nxt))
        emb = torch.stack(emb_list, dim=1)
        info['predicted_emb'] = rearrange(
            emb, '(b s) ... -> b s ...', b=batch, s=samples
        )
        return info
