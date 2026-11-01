import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained
from mosaic_predictors.models.sigreg_diagnostics import SIGRegCollapseMonitor


class MoPWM(LeWM):
    """LeWM world model with a Memory Mosaics predictor.

    encode() and predict() are the LeWM implementations: the same encoder,
    action encoder, projectors, and tensor layout. rollout() keeps LeWM's
    context/candidate action pairing and writes predicted_emb the same way.
    Hypothetical futures use a per-candidate cache, so the full causal
    history of that rollout stays readable. history_size is accepted because
    planners pass it; memory is not truncated to that window.
    """

    def _project_pred(self, preds: torch.Tensor) -> torch.Tensor:
        """Apply pred_proj the way LeWM.predict does, without a second predictor call."""
        single = preds.ndim == 2
        if single:
            preds = preds.unsqueeze(1)
        batch = preds.size(0)
        flat = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        out = rearrange(flat, '(b t) d -> b t d', b=batch)
        return out[:, 0] if single else out

    def rollout(self, info, action_sequence, history_size: int = None):
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)
        # Retained for the LeWM call signature. The cache keeps every
        # completed pair instead of sliding a window of this length.
        _ = history_size

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
            max_len=horizon_ctx + steps,
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


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


CONFIG_YAML = 'config.yaml'


def _hf_path_prefix(hf_cfg, run_name):
    """Repo-relative prefix, e.g. ``mopwm/lewm/``.

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


class _SigregDiagClose(Callback):
    def on_train_end(self, trainer, pl_module):
        diag = getattr(pl_module, 'sigreg_diagnostics', None)
        if diag is not None:
            diag.close(pl_module)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']  # (B, T, A)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    diag = getattr(self, 'sigreg_diagnostics', None)
    if diag is not None and stage == 'fit':
        diag.update(emb, self)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='mopwm')
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
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

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
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    diag_cfg = cfg.loss.sigreg.get('diagnostics')
    if diag_cfg is not None:
        diag_cfg = OmegaConf.to_container(diag_cfg, resolve=True)
        if diag_cfg.get('enabled', True):
            world_model.sigreg_diagnostics = SIGRegCollapseMonitor(diag_cfg)

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

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
        hf_cfg=hf_cfg,
        full_cfg=cfg,
    )
    callbacks = [save_ckpt_callback]
    if getattr(world_model, 'sigreg_diagnostics', None) is not None:
        callbacks.append(_SigregDiagClose())

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
