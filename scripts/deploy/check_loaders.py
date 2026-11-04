"""Check that train and val loaders fit in host memory together.

Uses the same dataset split and loader settings as mopwm training.
Exits with an error if both worker sets leave less than MIN_FREE_GB free.

    python scripts/check_loaders.py data=pusht
"""

import os
import time
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


def _pipeline_target(loader, kwargs) -> int:
    """Batches a live loader keeps around, besides the one in the step.

    PyTorch will have ``prefetch_factor`` batches in flight per worker.
    A tight read loop never lets that queue fill, which is why a drained
    probe under-counts the footprint training actually reaches.
    """
    workers = int(kwargs.get('num_workers') or 0)
    if workers <= 0:
        return 1
    prefetch = int(kwargs.get('prefetch_factor') or 1)
    target = workers * prefetch
    try:
        n_batches = len(loader)
    except TypeError:
        return target
    if n_batches <= 0:
        raise RuntimeError('loader produced no batches to measure')
    return min(target, n_batches)


def _queued_batches(loader_iter) -> int:
    total = 0
    data_queue = getattr(loader_iter, '_data_queue', None)
    if data_queue is not None:
        try:
            total += data_queue.qsize()
        except (NotImplementedError, OSError):
            pass
    task_info = getattr(loader_iter, '_task_info', None) or {}
    total += sum(1 for info in task_info.values() if len(info) >= 2)
    return total


def _wait_for_queue(loader_iter, target, peak, name):
    """Block until ``target`` completed batches are sitting in the queue."""
    if not hasattr(loader_iter, '_data_queue') or target <= 0:
        peak.sample()
        return
    stable = 0
    last_queued = -1
    last_change = time.monotonic()
    last_print = last_change
    while stable < 2:
        peak.sample()
        queued = _queued_batches(loader_iter)
        now = time.monotonic()
        if queued != last_queued:
            last_queued = queued
            last_change = now
        if now - last_print >= 5:
            print(
                f'Filling {name} prefetch queue ({queued}/{target} batches)',
                flush=True,
            )
            last_print = now
        if queued >= target:
            stable += 1
        else:
            stable = 0
            workers = getattr(loader_iter, '_workers', None) or []
            if workers and any(not worker.is_alive() for worker in workers):
                raise RuntimeError(
                    f'{name} worker exited before the prefetch queue filled '
                    f'({queued}/{target} batches)'
                )
            if now - last_change > 120:
                raise RuntimeError(
                    f'{name} prefetch queue stalled at {queued}/{target} batches'
                )
        time.sleep(0.1)
    peak.sample()


def _saturate_loader(loader, target, peak, name):
    """Fill the prefetch queue and hold one batch, as a training step does."""
    loader_iter = iter(loader)
    held = []
    try:
        if not hasattr(loader_iter, '_data_queue'):
            try:
                held.append(next(loader_iter))
            except StopIteration as exc:
                raise RuntimeError(
                    'loader produced no batches to measure'
                ) from exc
            peak.sample()
            return loader_iter, held

        _wait_for_queue(loader_iter, target, peak, name)
        try:
            held.append(next(loader_iter))
        except StopIteration as exc:
            if not held and _queued_batches(loader_iter) == 0:
                raise RuntimeError(
                    'loader produced no batches to measure'
                ) from exc
        else:
            try:
                n_batches = len(loader)
            except TypeError:
                n_batches = target + 1
            refill = min(target, max(0, n_batches - 1))
            _wait_for_queue(loader_iter, refill, peak, name)
        peak.sample()
        return loader_iter, held
    except BaseException:
        held.clear()
        _close_loader_iter(loader_iter)
        raise


def _close_loader_iter(loader_iter):
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, '_shutdown_workers', None)
    if shutdown is not None:
        shutdown()


class _MemLow:
    """Smallest host-memory availability observed so far."""

    def __init__(self):
        self.available = _mem_available_bytes()

    def sample(self):
        self.available = min(self.available, _mem_available_bytes())


def _assert_val_loader_fits(
    train_set, val_set, train_cfg, val_cfg, min_free_bytes: int
):
    """Start train and val workers together, the way validation does.

    Training leaves persistent workers resident and holds the current
    batch while each worker keeps ``prefetch_factor`` batches queued.
    The first validation pass adds the val workers on top of that.
    This probe fills both queues before measuring, then shuts the
    workers down before returning.
    """
    train_kwargs = _probe_loader_kwargs(train_cfg)
    val_kwargs = _probe_loader_kwargs(val_cfg)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)
    train_target = _pipeline_target(train_loader, train_kwargs)
    val_target = _pipeline_target(val_loader, val_kwargs)
    train_iter = None
    val_iter = None
    train_held = []
    val_held = []
    try:
        before = _mem_available_bytes()
        train_low = _MemLow()
        train_iter, train_held = _saturate_loader(
            train_loader, train_target, train_low, 'train'
        )
        after_train = _mem_available_bytes()
        train_low.sample()
        both_low = _MemLow()
        try:
            val_iter, val_held = _saturate_loader(
                val_loader, val_target, both_low, 'val'
            )
        except MemoryError as exc:
            raise RuntimeError(
                'val_loader ran out of host memory while train workers '
                'were already up. Lower val_loader.num_workers, '
                'val_loader.prefetch_factor, or val_loader.batch_size.'
            ) from exc
        after_both = _mem_available_bytes()
        both_low.sample()
        low = min(train_low.available, both_low.available)
    finally:
        train_held.clear()
        val_held.clear()
        _close_loader_iter(val_iter)
        _close_loader_iter(train_iter)

    train_cost = max(0, before - after_train)
    val_cost = max(0, after_train - after_both)
    print(
        'Loader memory check: '
        f'train workers {_gib(train_cost)} '
        f'({train_kwargs.get("num_workers", 0)} workers, '
        f'prefetch {train_kwargs.get("prefetch_factor", 1)}, '
        f'batch {train_kwargs["batch_size"]}, '
        f'queue {train_target}), '
        f'val workers {_gib(val_cost)} '
        f'({val_kwargs.get("num_workers", 0)} workers, '
        f'prefetch {val_kwargs.get("prefetch_factor", 1)}, '
        f'batch {val_kwargs["batch_size"]}, '
        f'queue {val_target}), '
        f'available with both {_gib(after_both)} '
        f'(low {_gib(low)})'
    )
    if low < min_free_bytes:
        raise RuntimeError(
            'val_loader does not fit on top of the train loader: '
            f'{_gib(low)} available at the peak, need at least '
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
