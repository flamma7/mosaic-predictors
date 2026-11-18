"""Run extra validation for checkpoints queued on Hugging Face.

Training with ``validate_extra.on_validation_server=true`` appends one row
per checkpoint to ``validation.csv`` at the repo root. This process claims
a row, downloads that checkpoint plus its ``config.yaml`` and
``config.json``, and runs the same MSE and MPC passes as
``ExtraValidationCallback``. Metrics go to the training W&B run, which
must have been opened with ``wandb.config.settings.mode=shared``.

One process validates one checkpoint at a time. Give each machine its own
``--machine-job-id``. A write always re-reads ``validation.csv`` first.

``--override`` changes ``validate_extra`` on the loaded ``config.yaml``
for this process only. ``config.json`` stays the model config.

    python scripts/train/validate_server.py --machine-job-id val-0 \
        --override mse.num_samples=8 --override mpc.num_eval=4

``mse.num_samples`` is how many validation clips the open-loop pass
scores. ``mpc.num_eval`` is how many episodes MPC runs. Per episode,
planning cost is mostly ``mpc.num_samples`` times ``mpc.n_steps``, and
each enabled perturbation repeats both passes.
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
from omegaconf.errors import OmegaConfBaseException

from dinogru import apply_cfg_start_stride
from dinowm import load_training_split
from validate_extra import make_extra_callback
from validation_queue import (
    ValidationQueue,
    define_trainer_step_axis,
    extra_validation_log,
)

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
    this epoch, so ``step`` is left unset. Charts are defined against
    ``trainer/global_step`` (the ``global_step`` column of validation.csv).
    ``step_sync`` stays off so this late write still plots at that value
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
        define_trainer_step_axis(run, cfg, metrics=metrics)
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


def _split_override(item):
    """``mse.num_samples=8`` or ``validate_extra.mse.num_samples=8``."""
    text = str(item).strip()
    if '=' not in text:
        raise ValueError(f'override must be key=value, got {item!r}')
    key, value = text.split('=', 1)
    key = key.strip()
    if key.startswith('validate_extra.'):
        key = key[len('validate_extra.') :]
    if not key or key.startswith('.') or key.endswith('.') or '..' in key:
        raise ValueError(f'override key is empty: {item!r}')
    return key, value


def _require_extra_key(extra, key):
    """Return the node at ``key``. Raise when any path segment is absent.

    ``OmegaConf.select(..., throw_on_missing=True)`` only rejects ``???``
    values. A typo would otherwise be inserted as a new field and ignored
    by the callback.
    """
    node = extra
    for part in key.split('.'):
        if not OmegaConf.is_config(node) or part not in node:
            raise ValueError(f'validate_extra has no {key}')
        node = node[part]
    return node


def apply_validate_extra_overrides(cfg, overrides):
    """Merge ``key=value`` pairs into ``cfg.validate_extra``.

    Keys are relative to ``validate_extra``. A ``validate_extra.`` prefix
    is also accepted. Unknown keys raise, so a typo cannot leave the
    checkpoint config unchanged. The returned config is a new object.
    """
    if not overrides:
        return cfg
    extra = cfg.get('validate_extra')
    if extra is None:
        raise ValueError('config has no validate_extra to override')
    parsed = [_split_override(item) for item in overrides]
    for key, _value in parsed:
        _require_extra_key(extra, key)
    dotted = [f'validate_extra.{key}={value}' for key, value in parsed]
    try:
        patch = OmegaConf.from_dotlist(dotted)
    except OmegaConfBaseException as exc:
        raise ValueError(f'could not parse overrides: {exc}') from exc
    merged = OmegaConf.merge(cfg, patch)
    seen = []
    for key, _value in parsed:
        if key in seen:
            continue
        seen.append(key)
        before = OmegaConf.select(extra, key)
        after = OmegaConf.select(merged.validate_extra, key)
        print(
            f'[validate_server] override validate_extra.{key}: '
            f'{before} -> {after}',
            flush=True,
        )
    return merged


def log_extra_budget(cfg):
    """Print the fields that scale MSE and MPC wall time."""
    extra = cfg.validate_extra
    mse = extra.get('mse') or {}
    mpc = extra.get('mpc') or {}
    names = []
    for name in extra.get('perturbations') or {}:
        spec = extra.perturbations[name]
        if spec is None or not bool(spec.get('enabled', True)):
            continue
        names.append(str(name))
    samples = mse.get('num_samples')
    sample_text = 'all' if samples is None else samples
    print(
        '[validate_server] '
        f'mse num_samples={sample_text} '
        f'pred_length={mse.get("pred_length")} '
        f'enabled={bool(mse.get("enabled", True))}; '
        f'mpc num_eval={mpc.get("num_eval")} '
        f'num_samples={mpc.get("num_samples")} '
        f'n_steps={mpc.get("n_steps")} '
        f'eval_budget={mpc.get("eval_budget")} '
        f'enabled={bool(mpc.get("enabled", True))}; '
        f'perturbations={",".join(names) or "(none)"}',
        flush=True,
    )


def run_job(repo_id, job, overrides=()):
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
    cfg = apply_validate_extra_overrides(cfg, overrides)
    log_extra_budget(cfg)

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


def poll(machine_job_id, repo_id, poll_seconds, once, revision, overrides):
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
            run_job(repo_id, job, overrides)
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
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help=(
            'Override one validate_extra field from the checkpoint '
            'config.yaml. Repeatable. Keys are relative to validate_extra, '
            'for example mse.num_samples=8 or mpc.num_eval=4. '
            'mpc.num_samples and mpc.n_steps scale the CEM loop.'
        ),
    )
    args = parser.parse_args()
    if args.poll_seconds < 0:
        raise SystemExit('--poll-seconds must be >= 0')
    try:
        for item in args.override:
            _split_override(item)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    poll(
        machine_job_id=args.machine_job_id,
        repo_id=args.repo_id,
        poll_seconds=args.poll_seconds,
        once=args.once,
        revision=args.revision,
        overrides=args.override,
    )


if __name__ == '__main__':
    main()
