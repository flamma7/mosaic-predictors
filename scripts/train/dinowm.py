import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from torch.nn import functional as F

from validate_extra import ExtraValidationCallback


def get_img_preprocessor(
    source: str,
    target: str,
    image_size: int = 224,
    encoder_image_size: int = 196,
):
    """ImageNet-normalize, resize to the table-12 size, then to the encoder."""
    imagenet_stats = dt.dataset_stats.ImageNet
    steps = [
        dt.transforms.ToImage(
            **imagenet_stats, source=source, target=target
        ),
        dt.transforms.Resize(image_size, source=source, target=target),
    ]
    if encoder_image_size != image_size:
        steps.append(
            dt.transforms.Resize(
                encoder_image_size, source=source, target=target
            )
        )
    return dt.transforms.Compose(*steps)


CONFIG_YAML = 'config.yaml'
# DINOv2 sequence written by precompute_dino.py. One fp16 row per frame:
# CLS, then the patch grid, flattened. Training drops the CLS token.
DINO_COLUMN = 'dino'


def _hf_path_prefix(hf_cfg, run_name):
    """Repo-relative prefix, e.g. ``dinowm/``.

    ``path_prefix`` may already include ``output_model_name``. If it does
    not, ``run_name`` is appended so the on-repo path is
    ``{path_prefix}/{output_model_name}/``.
    """
    prefix = str((hf_cfg or {}).get('path_prefix') or '').strip('/')
    name = run_name or ''
    if name and not prefix.endswith(name):
        prefix = f'{prefix}/{name}' if prefix else name
    return f'{prefix}/' if prefix else ''


class SaveCkptCallback(Callback):
    """Save epoch weights with save_pretrained and upload them to Hugging Face."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        hf_cfg=None,
        full_cfg=None,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.epoch_interval = epoch_interval
        self.hf_cfg = hf_cfg or {}
        self._hf_api = None

    @property
    def _ckpt_dir(self):
        return (
            swm.data.utils.get_cache_dir(sub_folder='checkpoints')
            / self.run_name
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _get_hf_api(self):
        if self._hf_api is not None:
            return self._hf_api

        import logging
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        token = os.environ.get('HF_TOKEN') or self.hf_cfg.get('token')
        self._hf_api = HfApi(token=token)
        return self._hf_api

    def _upload_to_hf(self, weights_file):
        if not self.hf_cfg.get('enabled'):
            return

        repo_id = self.hf_cfg['repo_id']
        prefix = _hf_path_prefix(self.hf_cfg, self.run_name)

        hf_api = self._get_hf_api()
        hf_api.upload_file(
            path_or_fileobj=str(self._ckpt_dir / weights_file),
            path_in_repo=f'{prefix}{weights_file}',
            repo_id=repo_id,
            repo_type='model',
        )
        config_path = self._ckpt_dir / 'config.json'
        if config_path.exists():
            hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo=f'{prefix}config.json',
                repo_id=repo_id,
                repo_type='model',
            )
        print(f'Uploaded {prefix}{weights_file} to {repo_id}')

    def _save_full_config(self):
        if self.full_cfg is None:
            return None
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_dir / CONFIG_YAML
        OmegaConf.save(
            OmegaConf.create(
                OmegaConf.to_container(self.full_cfg, resolve=True)
            ),
            path,
        )
        return CONFIG_YAML

    def _save(self, model, epoch):
        filename = f'weights_epoch_{epoch}.pt'
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=filename,
        )
        self._save_full_config()
        self._upload_to_hf(filename)
        if (self._ckpt_dir / CONFIG_YAML).exists():
            self._upload_to_hf(CONFIG_YAML)


def _strip_action_dims(tensor, action_range):
    """Drop the action channels. They condition the predictor, not the loss."""
    lo, hi = action_range
    return torch.cat([tensor[..., :lo], tensor[..., hi:]], dim=-1)


def _resolve_cached_dino(cfg):
    """Whether the training forward reads precomputed patch tokens.

    ``cached_dino: null`` follows the dataset: on when ``dino`` is loaded.
    Simulator frames and perturbed clips still go through the encoder.
    """
    loaded = DINO_COLUMN in list(cfg.data.dataset.keys_to_load)
    flag = cfg.get('cached_dino')
    if flag is None:
        return loaded
    flag = bool(flag)
    keys = list(cfg.data.dataset.keys_to_load)
    if flag and DINO_COLUMN not in keys:
        raise ValueError(
            'cached_dino=true needs `dino` in data.dataset.keys_to_load'
        )
    if not flag and loaded and 'pixels' not in keys:
        raise ValueError(
            'cached_dino=false needs `pixels` in keys_to_load to encode frames'
        )
    return flag


def _patches_from_cache(tokens, num_patches, embed_dim):
    """Drop the leading CLS token. The cache stores ``[CLS, patches]``."""
    n_tokens = num_patches + 1
    if tokens.ndim == 3:
        flat = tokens.shape[-1]
        expect = n_tokens * embed_dim
        if flat != expect:
            raise ValueError(
                f'Cached dino row is {flat} values, expected {expect} '
                f'(CLS + {num_patches} patches x {embed_dim})'
            )
        tokens = tokens.reshape(*tokens.shape[:2], n_tokens, embed_dim)
    elif tokens.ndim != 4 or tokens.shape[-2:] != (n_tokens, embed_dim):
        raise ValueError(
            'Cached dino tensor must be (B, T, 1+P, D) or (B, T, (1+P)*D), '
            f'got {tuple(tokens.shape)}'
        )
    return tokens[..., 1:, :]


def _embedding_from_cache(model, batch, cfg):
    """Build the predictor input from cached DINOv2 tokens.

    The cache stores the CLS token and the patch grid. The predictor is
    trained on the patches, so the CLS token is dropped here. Action and
    proprio embeddings are still computed; they are trained, and the cache
    does not include them.
    """
    num_patches = (cfg.encoder_img_size // cfg.patch_size) ** 2
    tokens = _patches_from_cache(
        batch[DINO_COLUMN], num_patches, int(cfg.patch_embed_dim)
    )
    pixels_emb = tokens.detach().float()
    n_patches = pixels_emb.shape[2]
    embedding = pixels_emb
    info = {'pixels_emb': pixels_emb}
    for key, encoder in model.extra_encoders.items():
        extra = encoder(batch[key].float())
        info[f'{key}_emb'] = extra
        tiled = extra.unsqueeze(2).expand(-1, -1, n_patches, -1)
        embedding = torch.cat([embedding, tiled], dim=-1)
    info['emb'] = embedding
    return info


def dinowm_forward(self, batch, stage, cfg):
    """Teacher-forced latent MSE on frozen DINOv2 features.

    The clip is ``history_size + num_preds`` frames. The predictor sees the
    first ``history_size`` frames and is trained against the clip shifted by
    ``num_preds``. Action channels are left out of the target.
    """
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)

    if cfg.cached_dino:
        output = _embedding_from_cache(self.model, batch, cfg)
    else:
        output = self.model.encode(batch, target='emb', is_video=False)

    ctx_emb = output['emb'][:, : cfg.wm.history_size]
    pred_emb = self.model.predict(ctx_emb)
    tgt_emb = output['emb'][:, cfg.wm.num_preds :].detach()

    pixels_dim = output['pixels_emb'].size(-1)
    output['pixels_loss'] = F.mse_loss(
        pred_emb[..., :pixels_dim], tgt_emb[..., :pixels_dim]
    )

    start = pixels_dim
    action_range = None
    for key in self.model.extra_encoders:
        dim = output[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = (lo, hi)
        else:
            output[f'{key}_loss'] = F.mse_loss(
                pred_emb[..., lo:hi], tgt_emb[..., lo:hi]
            )
        start = hi

    if action_range is None:
        pred_obs, tgt_obs = pred_emb, tgt_emb
    else:
        pred_obs = _strip_action_dims(pred_emb, action_range)
        tgt_obs = _strip_action_dims(tgt_emb, action_range)

    output['loss'] = F.mse_loss(pred_obs, tgt_obs)
    if output['loss'].isnan():
        raise ValueError('NaN loss encountered')

    losses_dict = {
        f'{stage}/{k}': v.detach()
        for k, v in output.items()
        if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def _n_params(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


@hydra.main(version_base=None, config_path='./config', config_name='dinowm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

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
            # `dino` is the frozen patch grid. Fitting a scaler would read
            # the whole column (hundreds of GB) and then rescale the target.
            if col.startswith('pixels') or col == DINO_COLUMN:
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            column_transforms.append(normalizer)

        # Actions are concatenated across the frameskip. Proprio is subsampled.
        cfg.model.extra_encoders.modules.action.in_chans = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        if cfg.use_proprio:
            cfg.model.extra_encoders.modules.proprio.in_chans = (
                dataset.get_dim('proprio')
            )
        else:
            del cfg.model.extra_encoders.modules['proprio']

    column_transform = spt.data.transforms.Compose(*column_transforms)
    if use_cache:
        # No pixels in this loader. Perturbed MSE opens a second reader.
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

    ##############################
    ##       model / optim      ##
    ##############################

    # EvalOnly keeps the DINOv2 encoder in eval and without gradients,
    # including after Lightning calls train() on the parent.
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

    side = cfg.encoder_img_size // cfg.patch_size
    pred_m = _n_params(world_model.predictor) / 1e6
    train_m = _n_params(world_model, trainable_only=True) / 1e6
    frozen_m = _n_params(world_model.backbone) / 1e6
    print(
        f'Patch grid: {num_patches} ({side} x {side}), '
        f'predictor {pred_m:.2f}M params, '
        f'trainable {train_m:.2f}M, '
        f'frozen encoder {frozen_m:.1f}M, '
        f'proprio {"on" if cfg.use_proprio else "off"}'
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

    # When enabled, proprio lives in extra_encoders and shares this rate.
    optimizers = {
        'predictor_opt': add_opt('model.predictor', cfg.predictor_lr),
        'action_encoder_opt': add_opt(
            'model.extra_encoders', cfg.action_encoder_lr
        ),
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

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

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
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
