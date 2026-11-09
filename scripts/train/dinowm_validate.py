"""Extra DINO-WM validation: perturbed MPC success and open-loop latent MSE.

Logged next to the regular ``val/*`` metrics:

* ``validate_mpc/<perturbation>`` — planning success rate (percent) on that
  visual shift. The env re-renders the observation and the goal with the
  shift, and CEM minimizes the error between the predicted latent and the
  encoding of that perturbed goal.
* ``validate_mse_<horizon>/<perturbation>`` — open-loop latent MSE at that
  model step. Clips come from the validation split and are long enough for
  ``history_size`` context frames plus ``pred_length`` steps. Each step runs
  the same history window as training, writes the dataset action into the
  predicted frame, and scores the prediction against the encoding of the
  perturbed frame at that step.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

import stable_worldmodel as swm
from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
    WeightedSum,
)
from stable_worldmodel.wrapper.visual import (
    apply_blur,
    apply_dark,
    apply_salt_pepper,
    parse_color,
)

# pygame colors of the PushT render, plus the wall and the background so a
# nearest-color paint does not recolor those. JPEG frames land near these
# values rather than on them.
_PUSHT_COLORS = np.array(
    [
        [65, 105, 225],  # agent, RoyalBlue
        [119, 136, 153],  # block, LightSlateGray
        [144, 238, 144],  # anchor, LightGreen
        [211, 211, 211],  # wall, LightGray
        [255, 255, 255],  # background
    ],
    dtype=np.float32,
)
_COLOR_INDEX = {'agent': 0, 'block': 1, 'anchor': 2}
_RECOLOR_MAX_DIST = 90.0

_PLAN_CALLABLES = [
    {
        'method': '_set_state',
        'args': {'state': {'value': 'state'}},
    },
    {
        'method': '_set_goal_state',
        'args': {'goal_state': {'value': 'goal_state'}},
    },
]


def _strip_action_dims(tensor, action_range):
    """Drop action channels. They condition the predictor, not the loss."""
    lo, hi = action_range
    return torch.cat([tensor[..., :lo], tensor[..., hi:]], dim=-1)


def _action_range(encoded, model, pixels_dim):
    start = pixels_dim
    found = None
    for key in model.extra_encoders:
        dim = encoded[f'{key}_emb'].shape[-1]
        if key == 'action':
            found = (start, start + dim)
        start += dim
    return found


def _obs_mse(pred, target, action_range):
    if action_range is None:
        return F.mse_loss(pred, target)
    return F.mse_loss(
        _strip_action_dims(pred, action_range),
        _strip_action_dims(target, action_range),
    )


def _level(value, default):
    """AdaJEPA flag: off, ``True`` for the paper default, or a number."""
    if value is None or value is False:
        return None
    if value is True:
        return float(default)
    return float(value)


def _mix_seed(seed, name, episode, start):
    name_id = sum(ord(char) for char in name)
    mixed = seed + episode * 10007 + start * 17 + name_id * 13
    return int(mixed % (2**31 - 1))


def _to_hwc(pixels):
    if pixels.ndim != 4:
        raise ValueError(f'Expected a clip (T, ..., H, W), got {tuple(pixels.shape)}')
    if pixels.shape[1] in (1, 3):
        hwc = pixels.permute(0, 2, 3, 1).contiguous()
        return hwc.numpy(), 'chw'
    if pixels.shape[-1] in (1, 3):
        return pixels.contiguous().numpy(), 'hwc'
    raise ValueError(f'Cannot find the channel axis in {tuple(pixels.shape)}')


def _from_hwc(frames, layout):
    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    if layout == 'chw':
        return tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def _recolor(frames, which, paint):
    flat = frames.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(flat[:, None, :] - _PUSHT_COLORS[None], axis=2)
    nearest = dist.argmin(axis=1)
    closest = dist.min(axis=1)
    mask = (nearest == _COLOR_INDEX[which]) & (closest <= _RECOLOR_MAX_DIST)
    out = frames.copy()
    out.reshape(-1, 3)[mask] = paint
    return out


def perturb_clip(pixels, spec, name, seed, episode, start):
    """Apply one visual shift to a uint8 clip. Layout is preserved.

    Color shifts repaint the matching object. Blur, salt-and-pepper, and
    darkening then run on those frames, which is the order the env uses
    (re-render the color, then the frame wrapper).
    """
    hwc, layout = _to_hwc(pixels)
    out = np.ascontiguousarray(hwc)
    for which, key in (
        ('agent', 'agent_color'),
        ('block', 'block_color'),
        ('anchor', 'anchor_color'),
    ):
        color = spec.get(key)
        if color is None or color is False:
            continue
        paint = np.asarray(parse_color(color), dtype=np.uint8)
        out = _recolor(out, which, paint)

    blur = _level(spec.get('blur'), 2.0)
    salt = _level(spec.get('salt_pepper'), 0.01)
    dark = _level(spec.get('dark'), 0.5)
    if blur is None and salt is None and dark is None:
        return _from_hwc(out, layout)

    rng = np.random.RandomState(_mix_seed(seed, name, episode, start))
    painted = out
    out = np.empty_like(painted)
    for t in range(painted.shape[0]):
        frame = painted[t]
        if blur is not None:
            frame = apply_blur(frame, blur)
        if salt is not None:
            frame = apply_salt_pepper(frame, salt, rng)
        if dark is not None:
            frame = apply_dark(frame, dark)
        out[t] = frame
    return _from_hwc(out, layout)


def target_index(history_size, num_preds, horizon):
    """Frame index predicted at an open-loop horizon.

    Horizon 1 is the frame the training forward's last token is trained
    against: ``history_size - 1 + num_preds``. Each later step is one more
    ``num_preds`` jump, with the prediction fed back as context.
    """
    return (history_size - 1) + horizon * num_preds


def rollout_horizon_mse(model, batch, history_size, num_preds, horizons):
    """Open-loop MSE against perturbed encodings, with dataset actions.

    ``batch`` holds one perturbed clip per row. Context is the first
    ``history_size`` encoded frames. Step ``h`` scores the last token of
    ``predict`` on the trailing history window against the encoded frame
    ``target_index(history_size, num_preds, h)``. The dataset action at
    that frame replaces the predicted action channels before the window
    slides.
    """
    for key in model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)

    encoded = model.encode(batch, target='emb', is_video=False)
    emb = encoded['emb']
    pixels_dim = encoded['pixels_emb'].shape[-1]
    action_range = _action_range(encoded, model, pixels_dim)
    z = emb[:, :history_size]
    wanted = set(int(h) for h in horizons)
    errors = {}

    for step in range(1, max(wanted) + 1):
        pred = model.predict(z[:, -history_size:])[:, -1:]
        index = target_index(history_size, num_preds, step)
        if index >= emb.shape[1]:
            break
        if step in wanted:
            errors[step] = _obs_mse(
                pred, emb[:, index : index + 1], action_range
            )
        action = batch['action'][:, index : index + 1]
        pred = model.replace_action_in_embedding(
            pred.unsqueeze(0), action.unsqueeze(0)
        )[0]
        z = torch.cat([z, pred], dim=1)
    return errors


class _RolloutWindows(torch.utils.data.Dataset):
    """Validation clips extended to the rollout length, then perturbed.

    Column normalizers run inside ``source._load_slice`` (the caller points
    ``source.transform`` at them for the duration of the pass). The image
    preprocessor runs after the perturbation, on the uint8 frames.
    """

    def __init__(
        self, source, windows, num_steps, image_preprocess, spec, name, seed
    ):
        self.source = source
        self.windows = windows
        self.num_steps = num_steps
        self.span = num_steps * int(source.frameskip)
        self.image_preprocess = image_preprocess
        self.spec = spec
        self.name = name
        self.seed = seed

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        steps = self.source._load_slice(episode, start, start + self.span)
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        steps['pixels'] = perturb_clip(
            steps['pixels'],
            self.spec,
            self.name,
            self.seed,
            episode,
            start,
        )
        return self.image_preprocess(steps)


def _enabled(perturbations):
    chosen = []
    for name in perturbations:
        spec = perturbations[name]
        if spec is None or not spec.get('enabled', True):
            continue
        chosen.append((str(name), spec))
    return chosen


def _val_windows(dataset, indices, num_steps):
    span = num_steps * int(dataset.frameskip)
    windows = []
    for idx in indices:
        episode, start = dataset.clip_indices[int(idx)]
        episode = int(episode)
        start = int(start)
        if start + span <= int(dataset.lengths[episode]):
            windows.append((episode, start))
    return windows


def _subsample(windows, num_samples, seed):
    if num_samples is None or len(windows) <= int(num_samples):
        return windows
    generator = torch.Generator().manual_seed(int(seed))
    pick = torch.randperm(len(windows), generator=generator)[: int(num_samples)]
    return [windows[int(i)] for i in pick]


def _report(pl_module, metrics):
    """Log through Lightning so these share the validation step record.

    A direct ``logger.log_metrics`` call writes a second Wandb record at the
    same ``trainer/global_step``. Wandb keeps one record per step, and the
    later ``validate/*`` flush replaces it.
    """
    for key, value in metrics.items():
        print(f'[validate_extra] {key}: {value:.6f}')
    if not metrics:
        return
    pl_module.log_dict(
        {key: float(value) for key, value in metrics.items()},
        logger=True,
        on_step=False,
        on_epoch=True,
        add_dataloader_idx=False,
        rank_zero_only=True,
        batch_size=1,
    )


def _planning_objective(use_proprio):
    """Score the perturbed goal encoding, source by source.

    The fused latent contains action channels, and a goal does not
    prescribe an action. Pixels (and proprio, when the model has it) are
    the state the planner matches.
    """
    terms = []
    if use_proprio:
        terms.append(
            (
                1.0,
                GoalMSE(
                    pred_key='predicted_proprio_emb',
                    goal_key='proprio_goal_emb',
                    reduction='mean',
                ),
            )
        )
    terms.append(
        (
            1.0,
            GoalMSE(
                pred_key='predicted_pixels_emb',
                goal_key='pixels_goal_emb',
                reduction='mean',
            ),
        )
    )
    return WeightedSum(terms)


def _eval_wm():
    path = Path(__file__).resolve().parents[1] / 'plan' / 'eval_wm.py'
    spec = importlib.util.spec_from_file_location('dinowm_eval_wm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_transform(img_size, encoder_img_size):
    """Training resize, as a per-frame tensor transform for the planner."""
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    steps = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=int(img_size)),
    ]
    if int(encoder_img_size) != int(img_size):
        steps.append(transforms.Resize(size=int(encoder_img_size)))
    return transforms.Compose(steps)


class ExtraValidationCallback(Callback):
    """Perturbed MPC and rollout MSE after each validation epoch."""

    def __init__(
        self,
        cfg,
        dataset,
        val_indices,
        image_preprocess,
        column_transform,
        cache_dir,
        dataset_name,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.image_preprocess = image_preprocess
        self.column_transform = column_transform
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        extra = cfg.validate_extra
        self.every_n_epochs = int(extra.get('every_n_epochs', 1))
        self.perturbations = _enabled(extra.perturbations)
        self.bf16 = str(cfg.trainer.get('precision', '')).startswith('bf16')

        mse = extra.mse
        self.mse_enabled = bool(mse.get('enabled', True)) and bool(
            self.perturbations
        )
        self.history_size = int(cfg.wm.history_size)
        self.num_preds = int(cfg.wm.num_preds)
        self.pred_length = int(mse.pred_length)
        self.horizons = [int(h) for h in mse.horizons]
        if any(h < 1 or h > self.pred_length for h in self.horizons):
            raise ValueError(
                f'horizons {self.horizons} must lie in '
                f'1..pred_length ({self.pred_length})'
            )
        self.clip_steps = (
            self.history_size + self.pred_length * self.num_preds
        )
        self.seed = int(cfg.seed)
        windows = _val_windows(dataset, val_indices, self.clip_steps)
        self.windows = _subsample(windows, mse.get('num_samples'), self.seed)
        if self.mse_enabled:
            print(
                f'[validate_extra] mse clips {len(self.windows)}'
                f'/{len(windows)} long enough '
                f'(history {self.history_size} + {self.pred_length} steps, '
                f'frameskip {dataset.frameskip})'
            )

        mpc = extra.mpc
        self.mpc_enabled = bool(mpc.get('enabled', True)) and bool(
            self.perturbations
        )
        self._mpc_ready = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        _barrier(trainer)
        if trainer.is_global_zero:
            self._run(trainer, pl_module)
        _barrier(trainer)

    def _run(self, trainer, pl_module):
        model = pl_module.model
        was_training = model.training
        model.eval()
        names = ', '.join(name for name, _ in self.perturbations)
        print(f'[validate_extra] epoch {trainer.current_epoch + 1}: {names}')
        try:
            if self.mse_enabled and self.windows:
                self._run_mse(pl_module, model)
            elif self.mse_enabled:
                print('[validate_extra] no validation clip fits the rollout')
            if self.mpc_enabled:
                self._run_mpc(pl_module, model)
        finally:
            pl_module.model = model
            if hasattr(model, '_init_cached_info'):
                del model._init_cached_info
            model.train(was_training)

    def _run_mse(self, pl_module, model):
        device = next(model.parameters()).device
        use_autocast = self.bf16 and device.type == 'cuda'
        previous = self.dataset.transform
        self.dataset.transform = self.column_transform
        metrics = {}
        try:
            for name, spec in self.perturbations:
                windows = _RolloutWindows(
                    self.dataset,
                    self.windows,
                    self.clip_steps,
                    self.image_preprocess,
                    spec,
                    name,
                    self.seed,
                )
                batch_size = min(
                    int(self.cfg.loader.batch_size), len(windows)
                )
                loader = torch.utils.data.DataLoader(
                    windows,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                    pin_memory=device.type == 'cuda',
                )
                totals = {h: 0.0 for h in self.horizons}
                count = 0
                for batch in loader:
                    batch = {
                        key: value.to(device, non_blocking=True)
                        if torch.is_tensor(value)
                        else value
                        for key, value in batch.items()
                    }
                    taken = batch['pixels'].shape[0]
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        errors = rollout_horizon_mse(
                            model,
                            batch,
                            self.history_size,
                            self.num_preds,
                            self.horizons,
                        )
                    missing = [
                        h for h in self.horizons if h not in errors
                    ]
                    if missing:
                        raise RuntimeError(
                            f'Rollout did not reach horizons {missing}. '
                            f'Clip length is {self.clip_steps}.'
                        )
                    for horizon, value in errors.items():
                        totals[horizon] += float(value) * taken
                    count += taken
                if count == 0:
                    raise RuntimeError(f'No MSE batches for {name}')
                for horizon in self.horizons:
                    metrics[f'validate_mse_{horizon}/{name}'] = (
                        totals[horizon] / count
                    )
        finally:
            self.dataset.transform = previous
        _report(pl_module, metrics)

    def _run_mpc(self, pl_module, model):
        if self._mpc_ready is None:
            self._mpc_ready = self._prepare_mpc(model)
        ready = self._mpc_ready
        metrics = {}
        for name, spec in self.perturbations:
            print(
                f'[validate_extra] mpc {name} '
                f'num_eval={ready["num_eval"]}'
            )
            rate = self._mpc_one(pl_module, model, spec, ready)
            metrics[f'validate_mpc/{name}'] = rate
        _report(pl_module, metrics)

    def _prepare_mpc(self, model):
        mpc = self.cfg.validate_extra.mpc
        num_eval = int(mpc.num_eval)
        eval_budget = int(mpc.eval_budget)
        horizon = int(mpc.horizon)
        action_block = int(mpc.action_block)
        if horizon * action_block > eval_budget:
            raise ValueError(
                'Planning horizon must be smaller than or equal to eval_budget'
            )
        eval_wm = _eval_wm()
        helper_cfg = OmegaConf.create(
            {
                'seed': int(mpc.get('seed', self.seed)),
                'cache_dir': self.cache_dir,
                'dataset': {
                    'keys_to_cache': ['action', 'proprio', 'state'],
                },
                'eval': {'dataset_name': self.dataset_name},
            }
        )
        dataset = eval_wm.get_dataset(helper_cfg, self.dataset_name)
        col_name = eval_wm.episode_col(dataset)
        episodes, _ = np.unique(
            dataset.get_col_data(col_name), return_index=True
        )
        process = {}
        for col in ('action', 'proprio', 'state'):
            scaler = StandardScaler()
            column = dataset.get_col_data(col)
            column = column[~np.isnan(column).any(axis=1)]
            scaler.fit(column)
            process[col] = scaler
            if col != 'action':
                process[f'goal_{col}'] = scaler

        lengths = eval_wm.get_episodes_length(dataset, episodes)
        goal_offset = int(mpc.goal_offset_steps)
        max_start = lengths - goal_offset - 1
        max_start_by_episode = {
            ep_id: max_start[i] for i, ep_id in enumerate(episodes)
        }
        per_row = np.array(
            [
                max_start_by_episode[ep_id]
                for ep_id in dataset.get_col_data(col_name)
            ]
        )
        valid = np.nonzero(dataset.get_col_data('step_idx') <= per_row)[0]
        if len(valid) < num_eval:
            raise ValueError(
                'Not enough episodes with sufficient length for MPC eval.'
            )
        generator = np.random.default_rng(int(mpc.get('seed', self.seed)))
        chosen = np.sort(
            valid[generator.choice(len(valid), size=num_eval, replace=False)]
        )
        device = next(model.parameters()).device
        transform = _image_transform(
            self.cfg.img_size, self.cfg.encoder_img_size
        )
        return {
            'num_eval': num_eval,
            'eval_budget': eval_budget,
            'goal_offset': goal_offset,
            'horizon': horizon,
            'receding_horizon': int(mpc.receding_horizon),
            'action_block': action_block,
            'num_samples': int(mpc.num_samples),
            'n_steps': int(mpc.n_steps),
            'topk': int(mpc.topk),
            'var_scale': float(mpc.var_scale),
            'env_name': str(mpc.env_name),
            'dataset': dataset,
            'process': process,
            'episodes': dataset.get_col_data(col_name)[chosen],
            'start_steps': dataset.get_col_data('step_idx')[chosen],
            'device': device,
            'transform': {'pixels': transform, 'goal': transform},
            'eval_wm': eval_wm,
            'seed': int(mpc.get('seed', self.seed)),
        }

    def _mpc_one(self, pl_module, model, spec, ready):
        if hasattr(model, '_init_cached_info'):
            del model._init_cached_info
        visual_cfg = OmegaConf.create(
            {
                'seed': ready['seed'],
                'visual': {
                    'blur': spec.get('blur', False),
                    'salt_pepper': spec.get('salt_pepper', False),
                    'dark': spec.get('dark', False),
                    'agent_color': spec.get('agent_color'),
                    'block_color': spec.get('block_color'),
                    'anchor_color': spec.get('anchor_color'),
                },
            }
        )
        wrappers, reset_options = ready['eval_wm'].configure_visuals(
            visual_cfg
        )
        world = swm.World(
            ready['env_name'],
            num_envs=ready['num_eval'],
            image_shape=(224, 224),
            max_episode_steps=2 * ready['eval_budget'],
            extra_wrappers=wrappers,
        )
        try:
            objective = _planning_objective(bool(self.cfg.use_proprio))
            cost = ShootingCostEvaluator(model, objective)
            solver = CEMSolver(
                cost=cost,
                batch_size=1,
                num_samples=ready['num_samples'],
                var_scale=ready['var_scale'],
                n_steps=ready['n_steps'],
                topk=ready['topk'],
                device=ready['device'],
                seed=ready['seed'],
            )
            config = swm.PlanConfig(
                horizon=ready['horizon'],
                receding_horizon=ready['receding_horizon'],
                action_block=ready['action_block'],
                history_len=self.history_size,
            )
            policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=config,
                process=ready['process'],
                transform=ready['transform'],
            )
            world.set_policy(policy)
            device_type = ready['device'].type
            with torch.no_grad(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=self.bf16 and device_type == 'cuda',
            ):
                results = world.evaluate(
                    dataset=ready['dataset'],
                    start_steps=ready['start_steps'].tolist(),
                    goal_offset=ready['goal_offset'],
                    eval_budget=ready['eval_budget'],
                    episodes_idx=ready['episodes'].tolist(),
                    callables=_PLAN_CALLABLES,
                    video=None,
                    options=reset_options,
                )
            return float(results['success_rate'])
        finally:
            pl_module.model = model
            if hasattr(model, '_init_cached_info'):
                del model._init_cached_info
            world.close()


def _barrier(trainer):
    if getattr(trainer, 'world_size', 1) > 1:
        trainer.strategy.barrier()
