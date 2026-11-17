"""Download epoch checkpoints and plot planning success.

The yaml lists model names plus the eval seed and env count::

    seed: 42
    num_envs: 50
    models:
      - lewm-pusht

``num_envs`` is passed as ``eval.num_eval`` (``world.num_envs`` follows it).
For each model, ``config.json``, ``config.yaml``, and every
``weights_epoch_*.pt`` are downloaded from ``<hf_repo>/<model>/`` into
``$STABLEWM_HOME/checkpoints/<model>/``. Each checkpoint is scored with
``scripts/plan/eval_wm.py``. The training ``config.yaml`` picks the plan
config (``pusht`` or ``tworoom``) and ``plan_config.history_len``.

Plots land in ``analysis/data/``: ``<model>.png`` for each model, and
``success_rates.png`` with one row per model. Scores are cached in
``analysis/data/success_rates.json`` so a rerun skips finished epochs. Delete that
file, or pass ``--force``, to evaluate again.

Usage:
    python analysis/eval_success.py analysis/models.yaml flamma77/mosaic_predictors
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / 'scripts' / 'plan' / 'eval_wm.py'
EPOCH_RE = re.compile(r'weights_epoch_(\d+)\.pt')
SUCCESS_RE = re.compile(
    r"['\"]success_rate['\"]\s*:\s*"
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)
EVAL_CONFIG_BY_DATASET = {
    'pusht_expert_train.lance': 'pusht',
    'tworoom.h5': 'tworoom',
}


def load_spec(path: Path) -> tuple[int, int, list[str]]:
    with path.open() as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise SystemExit(f'{path} must be a mapping with seed, num_envs, and models')
    try:
        seed = int(spec['seed'])
        num_envs = int(spec['num_envs'])
        models = [str(name) for name in spec['models']]
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f'{path} must contain integer seed, integer num_envs, and a models list'
        ) from exc
    if num_envs < 1:
        raise SystemExit('num_envs must be at least 1')
    if not models:
        raise SystemExit('models is empty')
    for name in models:
        if not name or '/' in name:
            raise SystemExit(f'Invalid model name: {name!r}')
    return seed, num_envs, models


def checkpoint_root() -> Path:
    import stable_worldmodel as swm

    return swm.data.utils.get_cache_dir(sub_folder='checkpoints')


def download_model(repo_id: str, model: str, ckpt_root: Path) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f'{model}/'
    names = []
    for path in list_repo_files(repo_id, repo_type='model'):
        if not path.startswith(prefix):
            continue
        name = path[len(prefix) :]
        if '/' in name:
            continue
        if name in ('config.json', 'config.yaml') or EPOCH_RE.fullmatch(name):
            names.append(name)
    if 'config.json' not in names or 'config.yaml' not in names:
        raise SystemExit(
            f'{repo_id}/{model} is missing config.json or config.yaml'
        )
    epochs = []
    for name in names:
        match = EPOCH_RE.fullmatch(name)
        if match:
            epochs.append(int(match.group(1)))
    if 1 not in epochs:
        raise SystemExit(f'{repo_id}/{model} is missing weights_epoch_1.pt')

    def sort_key(name: str) -> tuple[int, int | str]:
        match = EPOCH_RE.fullmatch(name)
        if match:
            return (1, int(match.group(1)))
        return (0, name)

    model_dir = ckpt_root / model
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(names, key=sort_key):
        print(f'[download] {prefix}{name}', flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=f'{prefix}{name}',
            repo_type='model',
            local_dir=ckpt_root,
        )
    return model_dir


def read_train_config(path: Path) -> dict:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f'{path} is not a mapping')
    return cfg


def eval_config_name(cfg: dict, model: str) -> str:
    data = cfg.get('data')
    dataset = None
    if isinstance(data, dict):
        dataset_cfg = data.get('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = dataset_cfg.get('name')
    if dataset in EVAL_CONFIG_BY_DATASET:
        return EVAL_CONFIG_BY_DATASET[dataset]
    blob = f'{dataset or ""} {model}'.lower()
    if 'pusht' in blob:
        return 'pusht'
    if 'tworoom' in blob:
        return 'tworoom'
    raise SystemExit(
        f'No eval config for {model} (dataset={dataset!r}). '
        'Expected pusht_expert_train.lance or tworoom.h5.'
    )


def history_len(cfg: dict) -> int | None:
    wm = cfg.get('wm')
    if not isinstance(wm, dict) or 'history_size' not in wm:
        return None
    return int(wm['history_size'])


def epoch_numbers(model_dir: Path) -> list[int]:
    epochs = []
    for path in model_dir.glob('weights_epoch_*.pt'):
        match = EPOCH_RE.fullmatch(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def success_rate_from_text(text: str) -> float | None:
    found = SUCCESS_RE.findall(text)
    if not found:
        return None
    return float(found[-1])


def run_eval(
    model: str,
    epoch: int,
    seed: int,
    num_envs: int,
    config_name: str,
    context_len: int | None,
    ckpt_dir: Path,
) -> float:
    policy = f'{model}/weights_epoch_{epoch}.pt'
    cmd = [
        sys.executable,
        '-u',
        str(EVAL_SCRIPT),
        f'--config-name={config_name}',
        f"policy='{policy}'",
        f'seed={seed}',
        f'eval.num_eval={num_envs}',
    ]
    if context_len is not None:
        cmd.append(f'plan_config.history_len={context_len}')
    print('[eval]', ' '.join(cmd), flush=True)

    results_path = ckpt_dir / f'{config_name}_results.txt'
    before = results_path.stat().st_size if results_path.exists() else 0
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        print(line, end='', flush=True)
        chunks.append(line)
    code = proc.wait()
    text = ''.join(chunks)
    if code != 0:
        tail = ''.join(chunks[-40:])
        raise SystemExit(
            f'eval_wm.py exited {code} for {policy}\n{tail}'
        )
    rate = success_rate_from_text(text)
    if rate is None and results_path.exists():
        with results_path.open() as handle:
            handle.seek(before)
            rate = success_rate_from_text(handle.read())
    if rate is None:
        raise SystemExit(f'No success_rate in eval output for {policy}')
    return rate


def load_cache(path: Path, seed: int, num_envs: int, repo: str) -> dict:
    empty = {
        'seed': seed,
        'num_envs': num_envs,
        'hf_repo': repo,
        'models': {},
    }
    if not path.exists():
        return empty
    with path.open() as handle:
        cached = json.load(handle)
    if (
        cached.get('seed') != seed
        or cached.get('num_envs') != num_envs
        or cached.get('hf_repo') != repo
    ):
        print(f'Cached scores in {path} used different settings; starting over.')
        return empty
    cached.setdefault('models', {})
    return cached


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w') as handle:
        json.dump(cache, handle, indent=2)
        handle.write('\n')
    tmp.replace(path)


def series(model_scores: dict) -> tuple[list[int], list[float]]:
    epochs = sorted(int(epoch) for epoch in model_scores)
    rates = [float(model_scores[str(epoch)]) for epoch in epochs]
    return epochs, rates


def _draw(ax, model: str, epochs: list[int], rates: list[float]) -> None:
    ax.plot(epochs, rates, marker='o', linewidth=1.8, color='C0')
    ax.set_title(model)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Success rate (%)')
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def write_plots(output_dir: Path, models: list[str], scores: dict) -> None:
    present = [model for model in models if scores.get(model)]
    if not present:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in present:
        epochs, rates = series(scores[model])
        fig, ax = plt.subplots(figsize=(7, 4))
        _draw(ax, model, epochs, rates)
        fig.tight_layout()
        fig.savefig(output_dir / f'{model}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(
        len(present),
        1,
        figsize=(8, 2.8 * len(present)),
        squeeze=False,
    )
    for ax, model in zip(axes[:, 0], present):
        epochs, rates = series(scores[model])
        _draw(ax, model, epochs, rates)
    fig.tight_layout()
    combined = output_dir / 'success_rates.png'
    fig.savefig(combined, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {combined}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot planning success against checkpoint epoch.',
    )
    parser.add_argument(
        'models_yaml',
        type=Path,
        help='yaml with seed, num_envs, and models',
    )
    parser.add_argument(
        'hf_repo',
        help='Hugging Face model repo, e.g. flamma77/mosaic_predictors',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'data',
        help='directory for the success-rate plots (default: analysis/data/)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='re-evaluate epochs that already have a cached score',
    )
    args = parser.parse_args()

    seed, num_envs, models = load_spec(args.models_yaml)
    ckpt_root = checkpoint_root()
    cache_path = args.output_dir / 'success_rates.json'
    cache = load_cache(cache_path, seed, num_envs, args.hf_repo)
    if args.force:
        cache['models'] = {}
    scores: dict = cache['models']

    print(
        f'Evaluating {len(models)} models from {args.hf_repo}\n'
        f'seed={seed} num_envs={num_envs}\n'
        f'checkpoints -> {ckpt_root}'
    )

    for model in models:
        model_dir = download_model(args.hf_repo, model, ckpt_root)
        cfg = read_train_config(model_dir / 'config.yaml')
        config_name = eval_config_name(cfg, model)
        context = history_len(cfg)
        epochs = epoch_numbers(model_dir)
        print(
            f'[model] {model}: eval config {config_name}, '
            f'history_len={context}, epochs={epochs}'
        )
        model_scores = scores.setdefault(model, {})
        for epoch in epochs:
            key = str(epoch)
            if key in model_scores and not args.force:
                print(
                    f'[skip] {model} epoch {epoch}: '
                    f'success_rate={float(model_scores[key]):.2f}%'
                )
                continue
            rate = run_eval(
                model, epoch, seed, num_envs, config_name, context, model_dir
            )
            model_scores[key] = rate
            save_cache(cache_path, cache)
            print(f'[score] {model} epoch {epoch}: success_rate={rate:.2f}%')
            write_plots(args.output_dir, models, scores)

    write_plots(args.output_dir, models, scores)


if __name__ == '__main__':
    main()
