"""Run extra validation for checkpoints queued on Hugging Face.

Training with ``validate_extra.on_validation_server=true`` appends one row
per checkpoint to ``validation.csv`` at the repo root. This process claims
a row, downloads that checkpoint plus its ``config.yaml`` and
``config.json``, and runs the same MSE and MPC passes as
``ExtraValidationCallback``. Metrics go to the training W&B run, which
must have been opened with ``wandb.config.settings.mode=shared``.

One process validates one checkpoint at a time. Give each machine its own
``--machine-job-id``. A write always re-reads ``validation.csv`` first.

    python scripts/train/validate_server.py --machine-job-id val-0
"""

from __future__ import annotations

import os

os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import argparse
import re
import time
import traceback

import torch
from omegaconf import OmegaConf

from dinogru import apply_cfg_start_stride
from dinowm import load_training_split
from validate_extra import make_extra_callback
from validation_queue import ValidationQueue, extra_validation_log

_EPOCH = re.compile(r'^weights_epoch_(\d+)\.pt$')
_DEFAULT_REPO = 'flamma77/mosaic_predictors'


class _ModelRef:
    def __init__(self, model):
        self.model = model


def checkpoint_epoch(name):
    match = _EPOCH.fullmatch(str(name))
    if not match:
        raise ValueError(
            f'checkpoint name must be weights_epoch_<n>.pt, got {name}'
        )
    return int(match.group(1))


def download_checkpoint(repo_id, model_name, checkpoint_name):
    from huggingface_hub import hf_hub_download
    import stable_worldmodel as swm

    root = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    root.mkdir(parents=True, exist_ok=True)
    prefix = str(model_name).strip().strip('/')
    for name in (checkpoint_name, 'config.json', 'config.yaml'):
        remote = f'{prefix}/{name}'
        print(f'[validate_server] download {remote}', flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            repo_type='model',
            local_dir=str(root),
        )
    return root / prefix


def log_to_wandb(cfg, job, machine_job_id, metrics, global_step, epoch_index):
    """Log like WandbLogger, at the training epoch's trainer/global_step.

    The internal W&B step only moves forward, and training is already past
    this epoch, so ``step`` is left unset. ``step_sync`` stays off for these
    keys so the late write still plots at the recorded ``trainer/global_step``
    instead of being rejected as a step in the past. ``epoch`` is Lightning's
    0-based ``current_epoch``, added to the same record as the metrics.
    """
    import wandb

    run = wandb.init(
        entity=str(cfg.wandb.config.entity),
        project=str(cfg.wandb.config.project),
        id=str(job['wandb_id']),
        resume='must',
        settings={
            'mode': 'shared',
            'x_primary': False,
            'x_label': f'extra-val-{machine_job_id}',
            'x_update_finish_state': False,
        },
    )
    try:
        run.define_metric('trainer/global_step')
        for pattern in ('validate_*', 'validate_mse_*', 'validate_mpc/*'):
            run.define_metric(
                pattern,
                step_metric='trainer/global_step',
                step_sync=False,
            )
        wandb.log(extra_validation_log(metrics, global_step, epoch_index))
    finally:
        run.finish()


def _global_step(job):
    raw = str(job.get('global_step', '')).strip()
    if raw == '':
        raise RuntimeError(
            f'{job["model_name"]}/{job["checkpoint_name"]} has no '
            'global_step in validation.csv'
        )
    step = int(raw)
    if step < 0:
        raise RuntimeError(f'global_step must be >= 0, got {step}')
    return step


def run_job(repo_id, job):
    """Download one checkpoint and run its config's extra validation."""
    model_name = job['model_name']
    checkpoint_name = job['checkpoint_name']
    global_step = _global_step(job)
    epoch = checkpoint_epoch(checkpoint_name)
    # weights_epoch_N is current_epoch + 1. Lightning's ``epoch`` key is
    # current_epoch, the same record as trainer/global_step.
    epoch_index = epoch - 1
    model_dir = download_checkpoint(repo_id, model_name, checkpoint_name)
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    if cfg.get('validate_extra') is None:
        raise RuntimeError(f'{model_name}/config.yaml has no validate_extra')

    from stable_worldmodel.wm.utils import load_pretrained

    model = load_pretrained(f'{model_name}/{checkpoint_name}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'[validate_server] {checkpoint_name} on {device}', flush=True)

    split = load_training_split(
        cfg,
        before_split=lambda dataset: apply_cfg_start_stride(cfg, dataset),
    )
    callback = make_extra_callback(
        cfg,
        dataset=split.dataset,
        val_indices=list(split.val_set.indices),
        image_preprocess=split.image_preprocess,
        column_transform=split.column_transform,
        cache_dir=split.cache_dir,
        dataset_name=split.dataset_name,
    )
    if callback is None:
        raise RuntimeError(
            f'{model_name}/config.yaml has validate_extra disabled'
        )
    holder = _ModelRef(model)
    metrics = callback.collect_metrics(model, holder, epoch=epoch)
    for key, value in metrics.items():
        print(f'[validate_extra] {key}: {value:.6f}', flush=True)

    if not metrics:
        print('[validate_server] no extra-validation metrics', flush=True)
        return
    if not bool(cfg.wandb.enabled):
        raise RuntimeError(
            f'{model_name}/config.yaml has wandb disabled, so the metrics '
            'cannot join the training run'
        )
    log_to_wandb(
        cfg, job, job['machine_job_id'], metrics, global_step, epoch_index
    )
    print(
        f'[validate_server] logged {len(metrics)} metrics to '
        f'{job["wandb_id"]} at trainer/global_step {global_step}',
        flush=True,
    )


def poll(machine_job_id, repo_id, poll_seconds, once, revision):
    queue = ValidationQueue(repo_id, revision=revision)
    while True:
        job = queue.claim(machine_job_id)
        if job is None:
            print('[validate_server] no free extra-validation job', flush=True)
            if once:
                return
            time.sleep(poll_seconds)
            continue
        print(
            '[validate_server] claimed '
            f'{job["model_name"]}/{job["checkpoint_name"]} '
            f'wandb={job["wandb_id"]}',
            flush=True,
        )
        try:
            run_job(repo_id, job)
            queue.complete(job, machine_job_id)
            print(
                '[validate_server] completed '
                f'{job["model_name"]}/{job["checkpoint_name"]}',
                flush=True,
            )
        except Exception:
            traceback.print_exc()
            print(
                '[validate_server] left '
                f'{job["model_name"]}/{job["checkpoint_name"]} '
                'started and not completed',
                flush=True,
            )
            if once:
                raise SystemExit(1) from None
            time.sleep(poll_seconds)
            continue
        if once:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--machine-job-id',
        required=True,
        help='Id written into validation.csv while this process owns a job.',
    )
    parser.add_argument('--repo-id', default=_DEFAULT_REPO)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument(
        '--once',
        action='store_true',
        help='Claim and run at most one job, then exit.',
    )
    args = parser.parse_args()
    if args.poll_seconds < 0:
        raise SystemExit('--poll-seconds must be >= 0')
    poll(
        machine_job_id=args.machine_job_id,
        repo_id=args.repo_id,
        poll_seconds=args.poll_seconds,
        once=args.once,
        revision=args.revision,
    )


if __name__ == '__main__':
    main()
