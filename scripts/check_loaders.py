"""Check that train and val loaders fit in host memory together.

Uses the same dataset split and loader settings as mopwm training.
Exits with an error if both worker sets leave less than MIN_FREE_GB free.

    python scripts/check_loaders.py data=pusht
"""

import os
from pathlib import Path

import hydra
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from stable_worldmodel.data import column_normalizer as get_column_normalizer

# Fail if train and val workers together leave less than this free.
MIN_FREE_GB = 2


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def _gib(num_bytes: int) -> str:
    return f'{num_bytes / (1024 ** 3):.2f} GiB'


def _cgroup_available_bytes() -> int | None:
    """Tightest cgroup memory headroom, so a container limit counts."""
    rel = ''
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            parts = line.split(':', 2)
            if len(parts) == 3 and parts[0] == '0':
                rel = parts[2].lstrip('/')
                break
    except OSError:
        return None

    limits = []
    path = Path('/sys/fs/cgroup') / rel
    root = Path('/sys/fs/cgroup')
    seen = set()
    while path not in seen:
        seen.add(path)
        max_path = path / 'memory.max'
        cur_path = path / 'memory.current'
        if max_path.is_file() and cur_path.is_file():
            raw = max_path.read_text().strip()
            if raw != 'max':
                limits.append(int(raw) - int(cur_path.read_text().strip()))
        if path == root:
            break
        path = path.parent
    if not limits:
        return None
    return min(limits)


def _mem_available_bytes() -> int:
    available = None
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
                break
    if available is None:
        raise RuntimeError('MemAvailable missing from /proc/meminfo')
    cgroup = _cgroup_available_bytes()
    if cgroup is None:
        return available
    return min(available, cgroup)


def _probe_loader_kwargs(loader_cfg):
    """Same batch, workers, and prefetch as the real loader, then exit."""
    kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
    kwargs['shuffle'] = False
    kwargs['persistent_workers'] = False
    if not kwargs.get('num_workers'):
        kwargs.pop('prefetch_factor', None)
    return kwargs


def _prefetch_batches(kwargs) -> int:
    workers = int(kwargs.get('num_workers') or 0)
    prefetch = int(kwargs.get('prefetch_factor') or 1)
    if workers <= 0:
        return 1
    return workers * prefetch


def _fill_loader(loader, batches: int):
    """Pull enough batches to fill the worker prefetch queues."""
    loader_iter = iter(loader)
    got = 0
    try:
        for _ in range(batches):
            next(loader_iter)
            got += 1
    except StopIteration:
        if got == 0:
            raise RuntimeError('loader produced no batches to measure')
    return loader_iter


def _close_loader_iter(loader_iter):
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, '_shutdown_workers', None)
    if shutdown is not None:
        shutdown()


def _assert_val_loader_fits(
    train_set, val_set, train_cfg, val_cfg, min_free_bytes: int
):
    """Start train and val workers together, the way validation does.

    Training leaves persistent workers resident. The first validation pass
    then adds the val workers on top. This probe does that and exits if
    host memory would be below ``min_free_bytes`` once both are up.
    Workers are shut down before this function returns.
    """
    train_kwargs = _probe_loader_kwargs(train_cfg)
    val_kwargs = _probe_loader_kwargs(val_cfg)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)
    train_iter = None
    val_iter = None
    try:
        before = _mem_available_bytes()
        train_iter = _fill_loader(
            train_loader, _prefetch_batches(train_kwargs)
        )
        after_train = _mem_available_bytes()
        try:
            val_iter = _fill_loader(val_loader, _prefetch_batches(val_kwargs))
        except MemoryError as exc:
            raise RuntimeError(
                'val_loader ran out of host memory while train workers '
                'were already up. Lower val_loader.num_workers, '
                'val_loader.prefetch_factor, or val_loader.batch_size.'
            ) from exc
        after_both = _mem_available_bytes()
    finally:
        _close_loader_iter(val_iter)
        _close_loader_iter(train_iter)

    train_cost = max(0, before - after_train)
    val_cost = max(0, after_train - after_both)
    print(
        'Loader memory check: '
        f'train workers {_gib(train_cost)} '
        f'({train_kwargs.get("num_workers", 0)} workers, '
        f'prefetch {train_kwargs.get("prefetch_factor", 1)}, '
        f'batch {train_kwargs["batch_size"]}), '
        f'val workers {_gib(val_cost)} '
        f'({val_kwargs.get("num_workers", 0)} workers, '
        f'prefetch {val_kwargs.get("prefetch_factor", 1)}, '
        f'batch {val_kwargs["batch_size"]}), '
        f'available with both {_gib(after_both)}'
    )
    if after_both < min_free_bytes:
        raise RuntimeError(
            'val_loader does not fit on top of the train loader: '
            f'{_gib(after_both)} available, need at least '
            f'{_gib(min_free_bytes)}. Lower val_loader.num_workers, '
            'val_loader.prefetch_factor, or val_loader.batch_size.'
        )


def _load_splits(cfg):
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
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith('pixels'):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    return train_set, val_set


@hydra.main(
    version_base=None,
    config_path='../src/mosaic_predictors/models/config',
    config_name='mopwm',
)
def run(cfg):
    train_set, val_set = _load_splits(cfg)
    _assert_val_loader_fits(
        train_set,
        val_set,
        cfg.loader,
        cfg.val_loader,
        min_free_bytes=int(MIN_FREE_GB * (1024 ** 3)),
    )


if __name__ == '__main__':
    run()
