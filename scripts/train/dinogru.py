"""DINO-GRU training: DINO-WM plus optional GRU memory and latent noise.

Variants share this entry point. ``dinogru.yaml`` is variant D.

    # A baseline
    python scripts/train/dinogru.py gru.enabled=false perturb.enabled=false

    # B perturbations only
    python scripts/train/dinogru.py gru.enabled=false perturb.s_z=S_Z

    # C GRU only
    python scripts/train/dinogru.py perturb.enabled=false

    # D GRU + perturbations, mean pool (default)
    python scripts/train/dinogru.py perturb.s_z=S_Z

    # E attention pool, one GRU
    python scripts/train/dinogru.py gru.pooling=attention perturb.s_z=S_Z

``S_Z`` comes from ``scripts/train/compute_dino_sz.py``. ``s_r`` is read from
the config, reused from a resumed run, or estimated once before the first step.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from torch.nn import functional as F

from dinowm import (
    DINO_COLUMN,
    SaveCkptCallback,
    _embedding_from_cache,
    _n_params,
    _patches_from_cache,
    _resolve_cached_dino,
    _strip_action_dims,
    dinowm_forward,
    get_img_preprocessor,
)
from mosaic_predictors.dinogru import (
    deterministic_factors,
    eligible_predictions,
    perturb_latents,
    require_s_r,
    require_s_z,
    residual_rms,
    sample_alpha,
    sample_factors,
    stored_s_r,
)
from validate_extra import ExtraValidationCallback


def _scale_needed(cfg):
    return bool(cfg.perturb.enabled) or bool(cfg.perturb.val_enabled)


def _clip_length(batch):
    for key in (DINO_COLUMN, 'pixels', 'action', 'proprio'):
        value = batch.get(key)
        if torch.is_tensor(value):
            return int(value.shape[1])
    raise KeyError('batch has no time dimension')


def _perturb_now(stage, cfg):
    if stage == 'train':
        return bool(cfg.perturb.enabled)
    if stage == 'val':
        return bool(cfg.perturb.val_enabled)
    return False


def _draw_perturbation(pixels, cfg, stage):
    rank = int(cfg.perturb.rank)
    dim = int(pixels.shape[-1])
    if stage == 'val':
        return deterministic_factors(
            pixels,
            int(cfg.seed),
            rank,
            float(cfg.perturb.v1),
            float(cfg.perturb.v2),
            float(cfg.perturb.alpha),
            float(cfg.perturb.zero_fraction),
        )
    w1, w2 = sample_factors(
        pixels.shape[0],
        dim,
        rank,
        float(cfg.perturb.v1),
        float(cfg.perturb.v2),
        pixels.device,
    )
    alpha = sample_alpha(
        pixels.shape[0],
        float(cfg.perturb.alpha),
        float(cfg.perturb.zero_fraction),
        pixels.device,
    )
    return w1, w2, alpha


def _retile(model, pixels, encoded):
    emb = pixels
    patches = pixels.shape[2]
    for key in model.extra_encoders:
        extra = encoded[f'{key}_emb']
        tiled = extra.unsqueeze(2).expand(-1, -1, patches, -1)
        emb = torch.cat([emb, tiled], dim=-1)
    return emb


def _observation_loss(output, pred, tgt, model):
    tgt = tgt.detach()
    pixels_dim = output['pixels_emb'].size(-1)
    output['pixels_loss'] = F.mse_loss(
        pred[..., :pixels_dim], tgt[..., :pixels_dim]
    )
    start = pixels_dim
    action_range = None
    for key in model.extra_encoders:
        dim = output[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = (lo, hi)
        else:
            output[f'{key}_loss'] = F.mse_loss(
                pred[..., lo:hi], tgt[..., lo:hi]
            )
        start = hi
    if action_range is None:
        pred_obs, tgt_obs = pred, tgt
    else:
        pred_obs = _strip_action_dims(pred, action_range)
        tgt_obs = _strip_action_dims(tgt, action_range)
    output['loss'] = F.mse_loss(pred_obs, tgt_obs)
    if output['loss'].isnan():
        raise ValueError('NaN loss encountered')
    return output


def dinogru_forward(self, batch, stage, cfg):
    """Next-latent MSE. Short unperturbed clips without a GRU match DINO-WM."""
    history = int(cfg.wm.history_size)
    num_preds = int(cfg.wm.num_preds)
    perturb = _perturb_now(stage, cfg)
    if (
        not self.model.gru_enabled
        and not perturb
        and _clip_length(batch) == history + num_preds
    ):
        return dinowm_forward(self, batch, stage, cfg)

    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)
    if cfg.cached_dino:
        output = _embedding_from_cache(self.model, batch, cfg)
    else:
        output = self.model.encode(batch, target='emb', is_video=False)

    if perturb:
        if cfg.perturb.s_z is None or cfg.perturb.s_r is None:
            raise ValueError(
                'Perturbations need perturb.s_z and a resolved perturb.s_r'
            )
        w1, w2, alpha = _draw_perturbation(
            output['pixels_emb'], cfg, stage
        )
        output['pixels_emb'] = perturb_latents(
            output['pixels_emb'],
            w1,
            w2,
            alpha,
            float(cfg.perturb.s_z),
            float(cfg.perturb.s_r),
            eps=float(cfg.perturb.eps),
            activation=str(cfg.perturb.activation),
        )
        output['emb'] = _retile(self.model, output['pixels_emb'], output)

    memory = None
    if self.model.gru_enabled:
        memory, _hidden = self.model.memory_sequence(
            output['pixels_emb'],
            output.get('proprio_emb'),
            output['action_emb'],
        )
    pred, tgt = eligible_predictions(
        self.model, output['emb'], memory, history, num_preds
    )
    output = _observation_loss(output, pred, tgt, self.model)
    losses = {
        f'{stage}/{key}': value.detach()
        for key, value in output.items()
        if 'loss' in key
    }
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


def _visual_batch(model, batch, cfg, device):
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    if cfg.cached_dino:
        num_patches = (int(cfg.encoder_img_size) // int(cfg.patch_size)) ** 2
        tokens = _patches_from_cache(
            batch[DINO_COLUMN], num_patches, int(cfg.patch_embed_dim)
        )
        return tokens.detach().float()
    for key in model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)
    with torch.no_grad():
        encoded = model.encode(batch, target='emb', is_video=False)
    return encoded['pixels_emb'].detach().float()


def _calibrate_s_r(model, dataset, cfg, device):
    calib = cfg.perturb.calibration
    limit = int(calib.num_samples)
    generator = torch.Generator().manual_seed(int(calib.seed))
    order = torch.randperm(len(dataset), generator=generator)
    seq_len = max(int(cfg.seq_len), 1)
    n_clips = min(len(dataset), (limit + seq_len - 1) // seq_len + 1)
    subset = torch.utils.data.Subset(dataset, order[:n_clips].tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=int(calib.batch_size),
        shuffle=False,
        num_workers=0,
    )
    frames = []
    seen = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            visual = _visual_batch(model, batch, cfg, device)
            flat = visual.reshape(-1, visual.shape[-2], visual.shape[-1])
            if seen + flat.shape[0] > limit:
                flat = flat[: limit - seen]
            frames.append(flat.cpu())
            seen += flat.shape[0]
            if seen >= limit:
                break
    finally:
        model.train(was_training)
    if seen == 0:
        raise RuntimeError('s_r calibration saw no visual embeddings')
    return residual_rms(
        torch.cat(frames, dim=0),
        rank=int(cfg.perturb.rank),
        v1=float(cfg.perturb.v1),
        v2=float(cfg.perturb.v2),
        activation=str(cfg.perturb.activation),
        num_matrices=int(calib.num_matrices),
        seed=int(calib.seed),
    )


def _resolve_s_r(model, dataset, cfg, device, ckpt_path, run_dir):
    if not _scale_needed(cfg):
        return None
    require_s_z(cfg.perturb.s_z, True)
    if cfg.perturb.s_r is not None:
        value = require_s_r(cfg.perturb.s_r)
        print(f'[dinogru] using configured s_r={value:.8f}')
        return value
    if ckpt_path.exists():
        paths = [Path(run_dir) / 'config.yaml']
        run_name = OmegaConf.select(cfg, 'output_model_name')
        if run_name:
            paths.append(
                swm.data.utils.get_cache_dir(sub_folder='checkpoints')
                / str(run_name)
                / 'config.yaml'
            )
        for path in paths:
            resumed = stored_s_r(path)
            if resumed is not None:
                print(f'[dinogru] reusing s_r={resumed:.8f} from {path}')
                return resumed
    if not bool(cfg.perturb.calibrate_s_r):
        raise ValueError(
            'Perturbations are enabled, perturb.s_r is unset, and '
            'perturb.calibrate_s_r is false.'
        )
    value = _calibrate_s_r(model, dataset, cfg, device)
    print(
        f'[dinogru] calibrated s_r={value:.8f} '
        f'on {int(cfg.perturb.calibration.num_samples)} embeddings'
    )
    return value


@hydra.main(version_base=None, config_path='./config', config_name='dinogru')
def run(cfg):
    if _scale_needed(cfg):
        require_s_z(cfg.perturb.s_z, True)
    with open_dict(cfg):
        cfg.data.dataset.num_steps = int(cfg.seq_len)

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    image_preprocess = get_img_preprocessor(
        source='pixels',
        target='pixels',
        image_size=cfg.img_size,
        encoder_image_size=cfg.encoder_img_size,
    )
    use_cache = _resolve_cached_dino(cfg)
    column_transforms = []

    with open_dict(cfg):
        cfg.cached_dino = use_cache
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels') or col == DINO_COLUMN:
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            column_transforms.append(normalizer)

        cfg.model.extra_encoders.modules.action.in_chans = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        if cfg.use_proprio:
            cfg.model.extra_encoders.modules.proprio.in_chans = (
                dataset.get_dim('proprio')
            )
            cfg.model.proprio_dim = int(cfg.proprio_emb_dim)
        else:
            del cfg.model.extra_encoders.modules['proprio']
            cfg.model.proprio_dim = 0
        cfg.model.visual_dim = int(cfg.patch_embed_dim)
        cfg.model.action_dim = int(cfg.action_emb_dim)

    column_transform = spt.data.transforms.Compose(*column_transforms)
    if use_cache:
        dataset.transform = column_transform
        print(
            'Training on cached DINOv2 patch tokens. '
            'MPC and perturbed clips still run the encoder.'
        )
    else:
        dataset.transform = spt.data.transforms.Compose(
            image_preprocess, *column_transforms
        )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.val_loader)

    encoder = hydra.utils.instantiate(cfg.model.encoder)
    hidden = encoder.config.hidden_size
    patch = encoder.config.patch_size
    if hidden != cfg.patch_embed_dim:
        raise ValueError(
            f'Encoder hidden size is {hidden}, expected {cfg.patch_embed_dim}'
        )
    if patch != cfg.patch_size:
        raise ValueError(
            f'Encoder patch size is {patch}, expected {cfg.patch_size}'
        )
    if cfg.encoder_img_size % cfg.patch_size != 0:
        raise ValueError('Encoder input size must be a multiple of patch size')
    num_patches = (cfg.encoder_img_size // cfg.patch_size) ** 2

    extra_dim = cfg.action_emb_dim
    if cfg.use_proprio:
        extra_dim += cfg.proprio_emb_dim
    with open_dict(cfg):
        cfg.model.predictor.num_patches = num_patches
        cfg.model.predictor.dim = hidden + extra_dim

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)
    world_model.train()
    if world_model.backbone.training or any(
        p.requires_grad for p in world_model.backbone.parameters()
    ):
        raise RuntimeError('DINOv2 encoder must stay frozen')
    if world_model.decoder is not None:
        raise RuntimeError('DINO-WM training does not use a decoder')

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    device = torch.device('cpu')
    if torch.cuda.is_available() and str(cfg.trainer.accelerator) != 'cpu':
        device = torch.device('cuda')
    if _scale_needed(cfg):
        world_model = world_model.to(device)
    s_r = _resolve_s_r(
        world_model, train_set, cfg, device, ckpt_path, run_dir
    )
    if s_r is not None:
        with open_dict(cfg):
            cfg.perturb.s_r = float(s_r)
        world_model.set_perturbation_scale(float(cfg.perturb.s_z), float(s_r))

    side = cfg.encoder_img_size // cfg.patch_size
    pred_m = _n_params(world_model.predictor) / 1e6
    train_m = _n_params(world_model, trainable_only=True) / 1e6
    frozen_m = _n_params(world_model.backbone) / 1e6
    pooling = world_model.gru_pooling if world_model.gru_enabled else 'off'
    print(
        f'Patch grid: {num_patches} ({side} x {side}), '
        f'predictor {pred_m:.2f}M params, '
        f'trainable {train_m:.2f}M, '
        f'frozen encoder {frozen_m:.1f}M, '
        f'proprio {"on" if cfg.use_proprio else "off"}, '
        f'seq_len {int(cfg.seq_len)}, '
        f'gru {pooling}, '
        f'perturb {"on" if cfg.perturb.enabled else "off"}, '
        f's_z {cfg.perturb.s_z}, s_r {cfg.perturb.s_r}'
    )

    total_steps = cfg.trainer.max_epochs * len(train)

    def add_opt(modules, lr):
        opt_cfg = dict(cfg.optimizer)
        opt_cfg['lr'] = lr
        return {
            'modules': modules,
            'optimizer': opt_cfg,
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }

    optimizers = {
        'predictor_opt': add_opt('model.predictor', cfg.predictor_lr),
        'action_encoder_opt': add_opt(
            'model.extra_encoders', cfg.action_encoder_lr
        ),
    }
    if world_model.gru_enabled:
        optimizers['memory_opt'] = add_opt('model.memory', cfg.predictor_lr)

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(dinogru_forward, cfg=cfg),
        optim=optimizers,
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    hf_cfg = cfg.get('hf')
    if hf_cfg is not None:
        hf_cfg = OmegaConf.to_container(hf_cfg, resolve=True)

    callbacks = [
        SaveCkptCallback(
            run_name=cfg.output_model_name,
            cfg=cfg.model,
            epoch_interval=1,
            hf_cfg=hf_cfg,
            full_cfg=cfg,
        )
    ]
    extra = cfg.get('validate_extra')
    if extra is not None and extra.get('enabled', False):
        frames_dataset = None
        mse_on = bool(extra.get('mse', {}).get('enabled', True))
        if use_cache and mse_on and 'pixels' not in dataset.column_names:
            frames_dataset = swm.data.load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                num_steps=int(dataset.num_steps),
                frameskip=int(dataset.frameskip),
                keys_to_load=['pixels', 'action', 'proprio', 'state'],
            )
            if frames_dataset.lengths.tolist() != dataset.lengths.tolist():
                raise RuntimeError(
                    'Pixel rows and cached DINO rows have different '
                    'episode lengths, so the train split would not match.'
                )
        callbacks.append(
            ExtraValidationCallback(
                cfg=cfg,
                dataset=dataset,
                val_indices=list(val_set.indices),
                image_preprocess=image_preprocess,
                column_transform=column_transform,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                frames_dataset=frames_dataset,
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()
    return


if __name__ == '__main__':
    run()
