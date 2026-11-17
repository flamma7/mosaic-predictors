"""DINO-WM with optional GRU memory and episode-fixed latent perturbations.

The predictor is the existing causal transformer. When the GRU is on, one
memory token is prepended to the window. It is the hidden state projected
into the token dimension. Perturbations are sampled outside the module and
are not parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from stable_worldmodel.wm.prejepa.module import Attention
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA


def activation_fn(name: str):
    table = {
        'gelu': F.gelu,
        'relu': F.relu,
        'silu': F.silu,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(
            f'Unknown activation {name!r}. Expected one of {sorted(table)}.'
        )
    return table[key]


def require_s_z(value, enabled: bool):
    """Positive global latent RMS. Training does not estimate this."""
    if not enabled:
        return None
    if value is None:
        raise ValueError(
            'perturb.s_z is required when perturbations are enabled. '
            'Run scripts/train/compute_dino_sz.py and set perturb.s_z. '
            'It is not computed during training.'
        )
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'perturb.s_z must be a positive finite number, got {value!r}.'
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f'perturb.s_z must be a positive finite number, got {value}.'
        )
    return value


def require_s_r(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'perturb.s_r must be a positive finite number, got {value!r}.'
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f'perturb.s_r must be a positive finite number, got {value}.'
        )
    return value


def stored_s_r(path):
    """Read a previously resolved ``perturb.s_r`` from a saved config."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    saved = OmegaConf.load(path)
    value = OmegaConf.select(saved, 'perturb.s_r')
    if value is None:
        return None
    try:
        return require_s_r(value)
    except ValueError:
        return None


def shift_actions(action):
    """``a_{t-1}``. The first step of a clip has no previous action."""
    previous = torch.zeros_like(action)
    if action.shape[1] > 1:
        previous[:, 1:] = action[:, :-1]
    return previous


def sample_factors(batch, dim, rank, v1, v2, device, generator=None):
    """``W1 ~ N(0, v1/D)``, ``W2 ~ N(0, v2/r)``, one pair per trajectory."""
    std1 = math.sqrt(float(v1) / dim)
    std2 = math.sqrt(float(v2) / rank)
    w1 = torch.randn(
        batch, rank, dim, device=device, generator=generator
    ) * std1
    w2 = torch.randn(
        batch, dim, rank, device=device, generator=generator
    ) * std2
    return w1, w2


def sample_alpha(batch, alpha, zero_fraction, device, generator=None):
    scales = torch.full((batch,), float(alpha), device=device)
    if zero_fraction <= 0:
        return scales
    drop = torch.rand(batch, device=device, generator=generator) < float(
        zero_fraction
    )
    return scales.masked_fill(drop, 0)


def low_rank_residual(z, w1, w2, activation):
    """``W2 σ(W1 z)`` with one ``(W1, W2)`` held across time and tokens."""
    values = z.float()
    w1 = w1.float()
    w2 = w2.float()
    if w1.ndim == 2:
        hidden = activation(torch.einsum('rd,btnd->btnr', w1, values))
        return torch.einsum('dr,btnr->btnd', w2, hidden)
    hidden = activation(torch.einsum('brd,btnd->btnr', w1, values))
    return torch.einsum('bdr,btnr->btnd', w2, hidden)


def perturb_latents(
    z, w1, w2, alpha, s_z, s_r, eps=1e-8, activation='gelu'
):
    """Scale the residual so its RMS is about ``alpha`` times ``s_z``."""
    residual = low_rank_residual(z, w1, w2, activation_fn(activation))
    if not torch.is_tensor(alpha):
        alpha = torch.full((z.shape[0],), float(alpha), device=z.device)
    alpha = alpha.to(device=z.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
    scale = alpha * (float(s_z) / (float(s_r) + float(eps)))
    return (z.float() + scale * residual).to(dtype=z.dtype)


def residual_rms(
    frames,
    rank,
    v1,
    v2,
    activation='gelu',
    num_matrices=16,
    seed=42,
):
    """RMS of ``W2 σ(W1 z)`` over frames and ``num_matrices`` pairs.

    ``frames`` is ``(M, N, D)`` or ``(M, T, N, D)``. Squared sums stay in
    float64. The matrices are not trained.
    """
    flat = frames.detach().float().reshape(
        -1, frames.shape[-2], frames.shape[-1]
    )
    dim = flat.shape[-1]
    act = activation_fn(activation)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    sum_sq = torch.zeros((), dtype=torch.float64)
    count = 0
    std1 = math.sqrt(float(v1) / dim)
    std2 = math.sqrt(float(v2) / rank)
    for _ in range(int(num_matrices)):
        w1 = torch.randn(rank, dim, generator=generator) * std1
        w2 = torch.randn(dim, rank, generator=generator) * std2
        residual = low_rank_residual(
            flat.unsqueeze(1), w1.to(flat.device), w2.to(flat.device), act
        )
        sum_sq = sum_sq + residual.double().pow(2).sum().cpu()
        count += residual.numel()
    if count == 0:
        raise ValueError('residual RMS received no embeddings')
    return torch.sqrt(sum_sq / count).item()


def latent_rms_from_sums(sum_sq, count):
    if count <= 0:
        raise ValueError('latent RMS received no elements')
    return math.sqrt(float(sum_sq) / float(count))


def deterministic_factors(
    z, seed, rank, v1, v2, alpha, zero_fraction
):
    """Validation matrices, one pair per row, derived from the latent.

    Training draws a new pair on every step. A validation draw has to stay
    the same across runs and not follow the dataloader order, so the seed
    is the run seed mixed with a rounded summary of ``z``. The input and
    the target match because the clip is perturbed once before the split.
    ``zero_fraction`` still turns some rows off, using that same seed.
    """
    batch, _, _, dim = z.shape
    summary = z.detach().float().mean(dim=(1, 2))
    width = min(8, summary.shape[-1])
    keys = summary[:, :width].mul(1e3).round().to(torch.int64).sum(dim=1)
    w1s = []
    w2s = []
    scales = []
    for index in range(batch):
        mix = (int(seed) + int(keys[index].item()) * 13) % (2**31 - 1)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(mix)
        w1, w2 = sample_factors(
            1, dim, rank, v1, v2, device='cpu', generator=generator
        )
        scale = sample_alpha(
            1, alpha, zero_fraction, device='cpu', generator=generator
        )
        w1s.append(w1[0])
        w2s.append(w2[0])
        scales.append(scale[0])
    return (
        torch.stack(w1s).to(z.device),
        torch.stack(w2s).to(z.device),
        torch.stack(scales).to(z.device),
    )


def _single_memory_mask(patches, frames):
    """Frames read the leading memory position and frames up to their own."""
    total = 1 + frames * patches
    mask = torch.zeros(total, total)
    mask[0, 0] = 1
    for frame in range(frames):
        start = 1 + frame * patches
        end = start + patches
        visible = 1 + (frame + 1) * patches
        mask[start:end, :visible] = 1
    return mask.unsqueeze(0).unsqueeze(0)


def install_memory_slots(predictor):
    """Add one leading memory position on the existing predictor.

    The patch positions are copied forward by one slot.
    """
    if getattr(predictor, '_memory_slots', False):
        return
    patches = int(predictor.num_patches)
    frames = int(predictor.num_frames)
    old = predictor.pos_embedding.data
    dim = old.shape[-1]
    updated = old.new_zeros(1, 1 + frames * patches, dim)
    updated[:, 1:] = old
    nn.init.normal_(updated[:, :1], std=0.02)
    predictor.pos_embedding = nn.Parameter(updated)
    mask = _single_memory_mask(patches, frames)
    found = False
    for module in predictor.modules():
        if isinstance(module, Attention):
            module.register_buffer('bias', mask.clone())
            found = True
    if not found:
        raise RuntimeError('CausalPredictor has no attention mask to extend')
    predictor._memory_slots = True


class MeanPool(nn.Module):
    def forward(self, tokens):
        return tokens.mean(dim=2)


class AttentionPool(nn.Module):
    """Learnable queries over visual tokens at a single timestep."""

    def __init__(self, dim, num_queries):
        super().__init__()
        if num_queries < 1:
            raise ValueError(f'num_queries must be >= 1, got {num_queries}')
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * dim**-0.5)

    def weights(self, tokens):
        scale = tokens.shape[-1] ** -0.5
        logits = torch.einsum('kd,btnd->btkn', self.queries, tokens) * scale
        return logits.softmax(dim=-1)

    def forward(self, tokens):
        weights = self.weights(tokens)
        pooled = torch.einsum('btkn,btnd->btkd', weights, tokens)
        return pooled.flatten(-2)


class GRUMemory(nn.Module):
    """One GRU over pooled visual tokens, proprioception, and ``a_{t-1}``."""

    def __init__(
        self,
        visual_dim,
        proprio_dim,
        action_dim,
        token_dim,
        hidden_dim=256,
        input_dim=256,
        num_layers=1,
        pooling='meanpool',
        num_queries=1,
    ):
        super().__init__()
        if pooling not in ('meanpool', 'attention'):
            raise ValueError(
                f"gru.pooling must be 'meanpool' or 'attention', got {pooling!r}"
            )
        self.pooling = pooling
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.num_queries = int(num_queries) if pooling == 'attention' else 1
        if pooling == 'attention':
            self.pool = AttentionPool(self.visual_dim, self.num_queries)
            pooled_dim = self.visual_dim * self.num_queries
        else:
            self.pool = MeanPool()
            pooled_dim = self.visual_dim
        in_dim = pooled_dim + self.proprio_dim + self.action_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, int(input_dim)),
            nn.GELU(),
            nn.Linear(int(input_dim), int(input_dim)),
        )
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.to_token = nn.Linear(int(hidden_dim), int(token_dim))

    def forward(
        self,
        visual,
        proprio,
        action,
        hidden=None,
        previous_actions=None,
        reset_each=False,
    ):
        pooled = self.pool(visual)
        if previous_actions is None:
            previous_actions = shift_actions(action)
        parts = [pooled]
        if self.proprio_dim:
            if proprio is None:
                raise ValueError('GRU expected proprio embeddings')
            parts.append(proprio)
        parts.append(previous_actions)
        inputs = self.input_mlp(torch.cat(parts, dim=-1))
        if reset_each:
            steps = []
            hidden_n = None
            for time in range(inputs.shape[1]):
                step, hidden_n = self.gru(inputs[:, time : time + 1], None)
                steps.append(step)
            sequence = torch.cat(steps, dim=1)
        else:
            sequence, hidden_n = self.gru(inputs, hidden)
        return self.to_token(sequence), hidden_n


def eligible_predictions(model, emb, memory, history, num_preds):
    """Next-latent predictions at every timestep that has a target.

    A clip of ``history + num_preds`` frames is one causal transformer pass,
    matching DINO-WM. Longer clips slide that same window. Each timestep is
    supervised once.
    """
    length = emb.shape[1]
    if length <= num_preds:
        raise ValueError(
            f'Sequence length {length} must exceed num_preds={num_preds}'
        )
    if memory is None and length == history + num_preds:
        return model.predict(emb[:, :history]), emb[:, num_preds:]

    predicted = []
    targets = []
    for frames in range(1, min(history, length)):
        target = frames - 1 + num_preds
        if target >= length:
            break
        window = emb[:, :frames]
        if memory is None:
            step = model.predict(window)[:, -1:]
        else:
            step = model.predict(window, memory=memory[:, frames - 1])[:, -1:]
        predicted.append(step)
        targets.append(emb[:, target : target + 1])

    last_start = length - history - num_preds
    if history <= length and last_start >= 0:
        context = emb[:, : length - num_preds]
        windows = context.unfold(1, history, 1)
        windows = windows.permute(0, 1, 4, 2, 3).contiguous()
        batch, count, frames, patches, dim = windows.shape
        flat = windows.reshape(batch * count, frames, patches, dim)
        if memory is None:
            step = model.predict(flat)[:, -1]
        else:
            mem = memory[:, history - 1 : history - 1 + count]
            mem = mem.reshape(batch * count, -1).contiguous()
            step = model.predict(flat, memory=mem)[:, -1]
        predicted.append(step.reshape(batch, count, patches, dim))
        index = torch.arange(count, device=emb.device) + history - 1 + num_preds
        targets.append(emb[:, index])

    if not predicted:
        raise ValueError(
            f'No eligible prediction in a length-{length} clip '
            f'(history={history}, num_preds={num_preds})'
        )
    return torch.cat(predicted, dim=1), torch.cat(targets, dim=1)


class DinoGRU(PreJEPA):
    """PreJEPA plus an optional GRU context token.

    With the GRU and perturbations off, ``predict`` and ``rollout`` are the
    PreJEPA implementations.
    """

    def __init__(
        self,
        encoder,
        predictor,
        extra_encoders=None,
        decoder=None,
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=True,
        gru_enabled=False,
        gru_hidden=256,
        gru_layers=1,
        gru_input_dim=256,
        gru_pooling='meanpool',
        gru_num_queries=1,
        gru_reset_each_step=False,
        plan_history_len=None,
        visual_dim=None,
        proprio_dim=0,
        action_dim=0,
        val_perturb=False,
        perturb_alpha=0.2,
        perturb_rank=8,
        perturb_activation='gelu',
        perturb_v1=1.0,
        perturb_v2=1.0,
        perturb_zero_fraction=0.25,
        perturb_eps=1e-8,
        perturb_seed=0,
        s_z=None,
        s_r=None,
    ):
        super().__init__(
            encoder,
            predictor,
            extra_encoders=extra_encoders,
            decoder=decoder,
            history_size=history_size,
            num_pred=num_pred,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        self.gru_enabled = bool(gru_enabled)
        self.gru_pooling = str(gru_pooling)
        self.gru_num_queries = int(gru_num_queries)
        self.gru_reset_each_step = bool(gru_reset_each_step)
        self.visual_dim = None if visual_dim is None else int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.val_perturb_enabled = bool(val_perturb)
        self.perturb_alpha = float(perturb_alpha)
        self.perturb_rank = int(perturb_rank)
        self.perturb_activation = str(perturb_activation)
        self.perturb_v1 = float(perturb_v1)
        self.perturb_v2 = float(perturb_v2)
        self.perturb_zero_fraction = float(perturb_zero_fraction)
        self.perturb_eps = float(perturb_eps)
        self.perturb_seed = int(perturb_seed)
        self.s_z = None if s_z is None else float(s_z)
        self.s_r = None if s_r is None else float(s_r)
        self._mem_sig = None
        self._mem_cached = None
        # Episode id -> GRU hidden state after the latest real block-frame.
        # The predictor token is projected from this state at planning time.
        # plan_history_len remains in the signature so older checkpoints load.
        del plan_history_len
        self.plan_history_len = None
        self._live = {}
        if self.gru_enabled:
            if self.visual_dim is None or self.action_dim <= 0:
                raise ValueError(
                    'GRU requires visual_dim and action_dim on the model config'
                )
            token_dim = int(predictor.pos_embedding.shape[-1])
            self.memory = GRUMemory(
                visual_dim=self.visual_dim,
                proprio_dim=self.proprio_dim,
                action_dim=self.action_dim,
                token_dim=token_dim,
                hidden_dim=int(gru_hidden),
                input_dim=int(gru_input_dim),
                num_layers=int(gru_layers),
                pooling=self.gru_pooling,
                num_queries=self.gru_num_queries,
            )
            install_memory_slots(predictor)
        else:
            self.memory = None

    def set_perturbation_scale(self, s_z, s_r):
        self.s_z = None if s_z is None else float(s_z)
        self.s_r = None if s_r is None else float(s_r)

    def reset_persistent_memory(self):
        self._mem_sig = None
        self._mem_cached = None
        self._live = {}
        if hasattr(self, '_init_cached_info'):
            del self._init_cached_info

    def _reset_memory_each_step(self):
        return self.gru_reset_each_step and not self.training

    def _split_latent_streams(self, emb):
        visual = emb[..., : self.visual_dim]
        flat = emb[:, :, 0, :]
        cursor = self.visual_dim
        proprio = None
        action = None
        for key, encoder in self.extra_encoders.items():
            dim = int(encoder.emb_dim)
            piece = flat[..., cursor : cursor + dim]
            if key == 'proprio':
                proprio = piece
            elif key == 'action':
                action = piece
            cursor += dim
        if action is None:
            raise RuntimeError('DinoGRU requires an action encoder')
        return visual, proprio, action

    def memory_sequence(self, visual, proprio, action, hidden=None):
        return self.memory(
            visual,
            proprio,
            action,
            hidden=hidden,
            reset_each=self._reset_memory_each_step(),
        )

    def memory_step(self, frame_emb, prev_action, hidden):
        visual, proprio, _action = self._split_latent_streams(frame_emb)
        reset = self._reset_memory_each_step()
        return self.memory(
            visual,
            proprio,
            prev_action,
            hidden=None if reset else hidden,
            previous_actions=prev_action,
            reset_each=reset,
        )

    def predict(self, embedding, memory=None):
        if not self.gru_enabled:
            if memory is not None:
                raise ValueError('memory was passed but the GRU is disabled')
            return super().predict(embedding)
        if memory is None:
            raise ValueError('GRU predict requires the memory token')
        if memory.ndim == 2:
            memory = memory.unsqueeze(1)
        if memory.ndim != 3 or memory.shape[1] != 1:
            raise ValueError(
                'GRU predict takes one memory token, '
                f'got shape {tuple(memory.shape)}'
            )
        frames, patches = embedding.shape[1], embedding.shape[2]
        flat = rearrange(embedding, 'b t p d -> b (t p) d')
        output = self.predictor(torch.cat([memory, flat], dim=1))
        return rearrange(
            output[:, 1:], 'b (t p) d -> b t p d', t=frames, p=patches
        )

    def _token_from_hidden(self, hidden):
        """Project the top GRU layer into the single predictor token."""
        return self.memory.to_token(hidden[-1])

    def _apply_eval_perturbation(self, encoded):
        if self.s_z is None or self.s_r is None:
            raise ValueError(
                'Validation perturbations need perturb.s_z and perturb.s_r'
            )
        pixels = encoded['pixels_emb']
        w1, w2, alpha = deterministic_factors(
            pixels,
            self.perturb_seed,
            self.perturb_rank,
            self.perturb_v1,
            self.perturb_v2,
            self.perturb_alpha,
            self.perturb_zero_fraction,
        )
        pixels = perturb_latents(
            pixels,
            w1,
            w2,
            alpha,
            self.s_z,
            self.s_r,
            eps=self.perturb_eps,
            activation=self.perturb_activation,
        )
        encoded['pixels_emb'] = pixels
        encoded['emb'] = _retile(self, pixels, encoded)
        return encoded

    def _context_memory(self, emb):
        """GRU tokens for a real context, from a zero state.

        ``emb`` is ``(B, T, P, D)`` after dataset actions have been written
        into the context. The candidate action on the latest frame is not
        an input to this unroll: the GRU sees ``a_{t-1}``.
        """
        visual, proprio, action = self._split_latent_streams(emb)
        return self.memory_sequence(visual, proprio, action, hidden=None)

    def observe_env(self, policy, info):
        """Step the GRU on each env that just recorded a new block-frame.

        ``get_action`` runs this on every environment step. A block-frame is
        a history entry whose distance from the newest frame is a multiple
        of ``action_block``, the same stride the planner uses. The first
        frame of an episode gets a zero previous-action embedding. Later
        frames get the embedding of the action block that arrived with them.
        """
        if not self.gru_enabled:
            return
        buffer = getattr(policy, '_history_buffer', None)
        if buffer is None or 'pixels' not in info:
            return
        action_block = int(policy.cfg.action_block)
        n_envs = buffer.n_envs
        ids = _per_env(info.get('id'), n_envs)
        steps = _per_env(info.get('step_idx'), n_envs)
        if ids is None or steps is None:
            return
        current = set(ids)
        for stale in list(self._live):
            if stale not in current:
                del self._live[stale]

        chosen = []
        fresh = []
        for index in range(n_envs):
            length = len(buffer._buffers[index])
            if length == 0 or (length - 1) % action_block != 0:
                continue
            episode = ids[index]
            step = steps[index]
            state = self._live.get(episode)
            if state is not None and state['step'] == step:
                continue
            start_fresh = (
                length == 1
                or state is None
                or step < state['step']
            )
            chosen.append(index)
            fresh.append(start_fresh)
        if not chosen:
            return

        with torch.no_grad():
            self._absorb_chosen(info, chosen, fresh, ids, steps, buffer, action_block)

    def _absorb_chosen(self, info, chosen, fresh, ids, steps, buffer, action_block):
        encoder = self.extra_encoders['action']
        blocks = []
        for index, start_fresh in zip(chosen, fresh):
            if start_fresh:
                blocks.append(None)
                continue
            blocks.append(
                _previous_action_block(
                    buffer, index, action_block, int(encoder.in_chans)
                )
            )
        self._absorb_frames(info, chosen, fresh, blocks, ids, steps)

    def _absorb_frames(self, info, chosen, fresh, blocks, ids, steps):
        device = next(self.memory.parameters()).device
        dtype = self.memory.gru.weight_ih_l0.dtype
        pixels = _select_rows(_tensor(info['pixels']), chosen).to(device)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        visual = self._encode_image(pixels).to(dtype=dtype)
        proprio = None
        if self.proprio_dim:
            if 'proprio' not in info:
                raise ValueError('GRU observe expected proprio in the observation')
            raw = _select_rows(_tensor(info['proprio']), chosen).to(device)
            if raw.ndim == 2:
                raw = raw.unsqueeze(1)
            proprio = self.extra_encoders['proprio'](raw.float()).to(dtype=dtype)

        encoder = self.extra_encoders['action']
        prev = visual.new_zeros(len(chosen), 1, int(encoder.emb_dim))
        carried = [i for i, block in enumerate(blocks) if block is not None]
        if carried:
            raw = torch.stack([blocks[i] for i in carried]).to(device)
            encoded = encoder(raw.float().unsqueeze(1)).to(dtype=dtype)
            for row, source in enumerate(carried):
                prev[source] = encoded[row]

        groups = {'fresh': [], 'carry': []}
        for row, start_fresh in enumerate(fresh):
            groups['fresh' if start_fresh else 'carry'].append(row)
        outputs = [None] * len(chosen)
        for key, rows in groups.items():
            if not rows:
                continue
            hidden = None
            if key == 'carry':
                hidden = torch.cat(
                    [self._live[ids[chosen[row]]]['hidden'] for row in rows],
                    dim=1,
                ).to(device=device, dtype=dtype)
            index = torch.tensor(rows, device=device)
            _token, hidden_n = self.memory(
                visual.index_select(0, index),
                None if proprio is None else proprio.index_select(0, index),
                prev.index_select(0, index),
                hidden=None if self._reset_memory_each_step() else hidden,
                previous_actions=prev.index_select(0, index),
                reset_each=self._reset_memory_each_step(),
            )
            for slot, row in enumerate(rows):
                outputs[row] = hidden_n[:, slot : slot + 1].detach()

        for row, index in enumerate(chosen):
            self._live[ids[index]] = {
                'step': steps[index],
                'hidden': outputs[row],
            }

    def _live_context(self, ids, steps, device):
        hiddens = []
        for episode, step in zip(ids, steps):
            state = self._live.get(int(episode))
            if state is None or state['step'] != int(step):
                return None
            hiddens.append(state['hidden'])
        return torch.cat(hiddens, dim=1).to(device)

    def _signature(self, info):
        ids = info.get('id')
        steps = info.get('step_idx')
        if ids is None or steps is None:
            return None
        if torch.is_tensor(ids) and ids.ndim > 1:
            ids = ids[:, 0]
        if torch.is_tensor(steps) and steps.ndim > 1:
            steps = steps[:, 0]
        if torch.is_tensor(ids):
            ids = ids.detach().reshape(-1).cpu().tolist()
        else:
            ids = torch.as_tensor(ids).reshape(-1).tolist()
        if torch.is_tensor(steps):
            steps = steps.detach().reshape(-1).cpu().tolist()
        else:
            steps = torch.as_tensor(steps).reshape(-1).tolist()
        return tuple(int(i) for i in ids), tuple(int(s) for s in steps)

    def _predict_window(self, latents, hidden):
        window = latents[:, -self.history_size :]
        if hidden is None:
            return self.predict(window)[:, -1:]
        return self.predict(window, memory=self._token_from_hidden(hidden))[
            :, -1:
        ]

    def rollout(self, info, action_sequence):
        if not self.gru_enabled:
            return super().rollout(info, action_sequence)
        return self._rollout_with_memory(info, action_sequence)

    def _rollout_with_memory(self, info, action_sequence):
        """PreJEPA rollout. Hypothetical steps clone the GRU state.

        Real block-frames update ``_live`` through ``observe_env``. This
        rollout reads that state and does not write it. Candidate actions
        only change the clone. With no live state, the GRU is replayed
        from zero over the frames in ``info``.
        """
        assert 'pixels' in info, 'pixels not in info_dict'
        n_obs = info['pixels'].shape[2]
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                *action_sequence.shape[:2], 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == n_obs - 1, (
            f'action_history must hold n-1={n_obs - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        act_0 = torch.cat([act_past, action_sequence[:, :, :1]], dim=2)
        info['action'] = act_0

        if (
            hasattr(self, '_init_cached_info')
            and torch.equal(self._init_cached_info['id'], info['id'][:, 0])
            and torch.equal(
                self._init_cached_info['step_idx'], info['step_idx'][:, 0]
            )
        ):
            init_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._init_cached_info.items()
            }
        else:
            init_info_dict = {}
            for key, value in info.items():
                if torch.is_tensor(value):
                    init_info_dict[key] = info[key][:, 0]
            init_info_dict = self.encode(
                init_info_dict, pixels_key='pixels', target='emb'
            )
            init_info_dict['emb'] = (
                init_info_dict['emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['emb'].ndim - 1)),
                )
                .clone()
            )
            init_info_dict['pixels_emb'] = (
                init_info_dict['pixels_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['pixels_emb'].ndim - 1)),
                )
            )
            for key in emb_keys:
                init_info_dict[f'{key}_emb'] = (
                    init_info_dict[f'{key}_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_sequence.shape[1],
                        *([-1] * (init_info_dict[f'{key}_emb'].ndim - 1)),
                    )
                )
            init_info_dict = {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in init_info_dict.items()
            }
            self._init_cached_info = init_info_dict

        info['emb'] = init_info_dict['emb']
        info['pixels_emb'] = init_info_dict['pixels_emb']
        for key in emb_keys:
            info[f'{key}_emb'] = init_info_dict[f'{key}_emb']
        info['emb'] = self.replace_action_in_embedding(info['emb'], act_0)

        action_dim = init_info_dict['action_emb'].shape[-1]
        info['action_emb'] = info['emb'][:, :, :n_obs, 0, -action_dim:]

        act_pred = action_sequence[:, :, 1:]
        n_steps = act_pred.shape[2]
        z = info['emb']
        batch, candidates = z.shape[:2]
        z_flat = rearrange(z, 'b n ... -> (b n) ...').clone()
        act_pred_flat = rearrange(act_pred, 'b n ... -> (b n) ...')

        signature = self._signature(info)
        context = info['emb'][:, 0]
        live = None
        if signature is not None:
            live = self._live_context(
                signature[0], signature[1], context.device
            )
        if live is not None:
            ctx_h = live
        elif (
            signature is not None
            and signature == self._mem_sig
            and self._mem_cached is not None
        ):
            ctx_h = self._mem_cached.to(context.device)
        else:
            _tokens, ctx_h = self._context_memory(context)
            ctx_h = ctx_h.detach()
        if signature is not None and (
            live is not None
            or self._mem_cached is None
            or signature != self._mem_sig
        ):
            self._mem_sig = signature
            self._mem_cached = ctx_h.detach()

        hidden = (
            ctx_h.unsqueeze(2)
            .expand(-1, -1, candidates, -1)
            .reshape(ctx_h.shape[0], batch * candidates, ctx_h.shape[-1])
            .clone()
        )

        for time in range(n_steps):
            pred_embed = self._predict_window(z_flat, hidden)
            new_action = act_pred_flat[None, :, time : time + 1, :]
            new_embed = self.replace_action_in_embedding(
                pred_embed.unsqueeze(0), new_action
            )[0]
            _visual, _proprio, prev_action = self._split_latent_streams(
                z_flat[:, -1:]
            )
            z_flat = torch.cat([z_flat, new_embed], dim=1)
            _token, hidden = self.memory_step(new_embed, prev_action, hidden)

        pred_embed = self._predict_window(z_flat, hidden)
        z_flat = torch.cat([z_flat, pred_embed], dim=1)
        z = rearrange(z_flat, '(b n) ... -> b n ...', b=batch, n=candidates)
        info['predicted_emb'] = z

        extra_dims = []
        for key in self.extra_encoders:
            if f'{key}_emb' not in info:
                raise ValueError(f'{key}_emb not in info dict')
            extra_dims.append(info[f'{key}_emb'].shape[-1])
        split = self.split_embedding(z, extra_dims)
        info.update({f'predicted_{k}': v for k, v in split.items()})
        return info

    def rollout_horizon_mse(self, batch, history_size, num_preds, horizons):
        """Open-loop latent MSE. GRU updates use past frames only."""
        for key in self.extra_encoders:
            batch[key] = torch.nan_to_num(batch[key], 0.0)
        encoded = self.encode(batch, target='emb', is_video=False)
        if self.val_perturb_enabled:
            encoded = self._apply_eval_perturbation(encoded)
        emb = encoded['emb']
        pixels_dim = encoded['pixels_emb'].shape[-1]
        action_range = _action_range(encoded, self, pixels_dim)
        visual, proprio, action = self._split_latent_streams(emb)
        _tokens, hidden = self.memory_sequence(
            visual[:, :history_size],
            None if proprio is None else proprio[:, :history_size],
            action[:, :history_size],
        )
        latents = emb[:, :history_size]
        wanted = {int(h) for h in horizons}
        errors = {}
        for step in range(1, max(wanted) + 1):
            pred = self._predict_window(latents, hidden)
            index = (history_size - 1) + step * num_preds
            if index >= emb.shape[1]:
                break
            if step in wanted:
                errors[step] = _obs_mse(
                    pred, emb[:, index : index + 1], action_range
                )
            action_raw = batch['action'][:, index : index + 1]
            pred = self.replace_action_in_embedding(
                pred.unsqueeze(0), action_raw.unsqueeze(0)
            )[0]
            _visual, _proprio, prev_action = self._split_latent_streams(
                latents[:, -1:]
            )
            latents = torch.cat([latents, pred], dim=1)
            _token, hidden = self.memory_step(pred, prev_action, hidden)
        return errors


def _retile(model, pixels, encoded):
    emb = pixels
    patches = pixels.shape[2]
    for key in model.extra_encoders:
        extra = encoded[f'{key}_emb']
        tiled = extra.unsqueeze(2).expand(-1, -1, patches, -1)
        emb = torch.cat([emb, tiled], dim=-1)
    return emb


def _action_range(encoded, model, pixels_dim):
    start = pixels_dim
    found = None
    for key in model.extra_encoders:
        dim = encoded[f'{key}_emb'].shape[-1]
        if key == 'action':
            found = (start, start + dim)
        start += dim
    return found


def _strip_action_dims(tensor, action_range):
    lo, hi = action_range
    return torch.cat([tensor[..., :lo], tensor[..., hi:]], dim=-1)


def attach_live_memory(policy, model):
    """Record each real block-frame into the GRU before planning."""
    if not getattr(model, 'gru_enabled', False):
        return policy

    def _observe(info):
        model.observe_env(policy, info)

    policy.frame_observer = _observe
    return policy


def _tensor(value):
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _per_env(value, n_envs):
    if value is None:
        return None
    value = _tensor(value).detach().cpu()
    if value.numel() == 1 and n_envs == 1:
        return [int(value.reshape(-1)[0])]
    value = value.reshape(n_envs, -1)[:, 0]
    return [int(item) for item in value.tolist()]


def _select_rows(value, indices):
    index = torch.as_tensor(indices, device=value.device)
    return value.index_select(0, index)


def _previous_action_block(buffer, index, action_block, in_chans):
    entries = list(buffer._buffers[index])[-action_block:]
    parts = []
    for entry in entries:
        action = _tensor(entry['action']).detach().reshape(-1).float()
        parts.append(torch.nan_to_num(action, nan=0.0))
    block = torch.cat(parts, dim=0)
    if block.numel() != in_chans:
        raise ValueError(
            f'GRU action block has {block.numel()} values, '
            f'but the action encoder expects {in_chans}.'
        )
    return block


def _obs_mse(pred, target, action_range):
    if action_range is None:
        return F.mse_loss(pred, target)
    return F.mse_loss(
        _strip_action_dims(pred, action_range),
        _strip_action_dims(target, action_range),
    )
