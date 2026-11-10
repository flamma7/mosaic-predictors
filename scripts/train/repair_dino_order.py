"""Rewrite the on-disk DINOv2 Lance table into episode order.

``precompute_dino.py`` used LanceDB's default parallel writer. That
round-robins row groups across fragments, so a scan of ``episode_idx``
decreases even though each episode is still one contiguous run. This
copies those runs into a new table in ``episode_idx`` order. It does not
re-encode frames.

The new table is written beside the source and needs about as much free
disk as the source takes. After a check that every ``(episode_idx,
step_idx)`` survived, the source directory is renamed to
``<name>.lance.bak`` and the new table takes its place.

Usage on the training machine::

    python scripts/train/repair_dino_order.py

Resume is automatic. ``--overwrite`` drops a partial rewrite and starts
again. Delete the ``.bak`` directory once training has opened the new table.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa


def _datasets_dir() -> Path:
    import stable_worldmodel as swm

    return swm.data.utils.get_cache_dir(sub_folder='datasets')


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--src',
        type=Path,
        default=None,
        help='Shuffled Lance table. Default: '
        '$STABLEWM_HOME/datasets/pusht_expert_train_dino.lance',
    )
    parser.add_argument(
        '--rows-per-commit',
        type=int,
        default=2048,
        help='Frames per Lance commit. A crash loses at most one commit.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete a partial rewrite and start again.',
    )
    return parser.parse_args()


def _dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total


def _runs(episode_idx: np.ndarray, step_idx: np.ndarray):
    """``(episode, start_row, end_row)`` in episode_idx order.

    Each episode must already be one contiguous run. Within a run, row
    order is kept, so frame order inside an episode does not change.
    """
    if len(episode_idx) == 0:
        raise SystemExit('Source table has no rows.')
    if len(episode_idx) != len(step_idx):
        raise SystemExit('episode_idx and step_idx have different lengths.')
    change = np.flatnonzero(np.diff(episode_idx) != 0) + 1
    bounds = np.concatenate([[0], change, [len(episode_idx)]])
    runs = []
    seen = set()
    for index in range(len(bounds) - 1):
        start = int(bounds[index])
        end = int(bounds[index + 1])
        episode = int(episode_idx[start])
        if episode in seen:
            raise SystemExit(
                f'episode_idx {episode} is stored in more than one run. '
                'This script only reorders whole episodes.'
            )
        steps = step_idx[start:end].astype(np.int64, copy=False)
        if len(steps) > 1 and (np.diff(steps) < 0).any():
            raise SystemExit(
                f'episode_idx {episode} has step_idx decreasing inside its '
                'run, so the frames themselves are out of order.'
            )
        seen.add(episode)
        runs.append((episode, start, end))
    runs.sort(key=lambda run: run[0])
    return runs


def _in_order(episode_idx: np.ndarray) -> bool:
    if len(episode_idx) < 2:
        return True
    return not (np.diff(episode_idx.astype(np.int64)) < 0).any()


def _column(dataset, name: str) -> np.ndarray:
    return (
        dataset.to_table(columns=[name])
        .column(name)
        .to_numpy(zero_copy_only=False)
    )


def _prefix_length(dest_dataset, runs, step_idx) -> int:
    """Rows of the sorted episode prefix already stored, or exit."""
    if dest_dataset is None or dest_dataset.count_rows() == 0:
        return 0
    got_episode = _column(dest_dataset, 'episode_idx').astype(np.int64, copy=False)
    got_step = _column(dest_dataset, 'step_idx').astype(np.int64, copy=False)
    if not _in_order(got_episode):
        raise SystemExit(
            'Partial rewrite is itself out of episode order. Re-run with '
            '--overwrite.'
        )
    covered = 0
    for episode, start, end in runs:
        length = end - start
        if covered + length > len(got_episode):
            if covered != len(got_episode):
                raise SystemExit(
                    'Partial rewrite ends in the middle of an episode. '
                    'Re-run with --overwrite.'
                )
            return covered
        if not np.array_equal(
            got_episode[covered : covered + length],
            np.full(length, episode, dtype=np.int64),
        ) or not np.array_equal(
            got_step[covered : covered + length],
            step_idx[start:end].astype(np.int64, copy=False),
        ):
            raise SystemExit(
                'Partial rewrite does not match the source episode order. '
                'Re-run with --overwrite.'
            )
        covered += length
    if covered != len(got_episode):
        raise SystemExit(
            f'Partial rewrite has {len(got_episode)} rows but the matching '
            f'episode prefix is {covered}. Re-run with --overwrite.'
        )
    return covered


def _batches(runs, rows_per_commit: int):
    batch = []
    rows = 0
    for run in runs:
        batch.append(run)
        rows += run[2] - run[1]
        if rows >= rows_per_commit:
            yield batch
            batch = []
            rows = 0
    if batch:
        yield batch


def _indices(batch) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, end, dtype=np.int64) for _, start, end in batch]
    )


def _open_writer(dst: Path, schema: pa.Schema, resume: bool):
    import lancedb

    db = lancedb.connect(str(dst.parent))
    if resume:
        return db.open_table(dst.stem)
    if dst.exists():
        shutil.rmtree(dst)
    return db.create_table(dst.stem, schema=schema)


def _verify(dest_dataset, runs, episode_idx, step_idx):
    got_episode = _column(dest_dataset, 'episode_idx').astype(np.int64, copy=False)
    got_step = _column(dest_dataset, 'step_idx').astype(np.int64, copy=False)
    expect_episode = np.concatenate(
        [episode_idx[start:end] for _, start, end in runs]
    ).astype(np.int64, copy=False)
    expect_step = np.concatenate(
        [step_idx[start:end] for _, start, end in runs]
    ).astype(np.int64, copy=False)
    if len(got_episode) != len(expect_episode) or not np.array_equal(
        got_episode, expect_episode
    ):
        raise SystemExit('Rewritten episode_idx does not match the source.')
    if not np.array_equal(got_step, expect_step):
        raise SystemExit('Rewritten step_idx does not match the source.')
    if not _in_order(got_episode):
        raise SystemExit('Rewritten table is still out of episode order.')


def _swap(src: Path, dst: Path):
    backup = src.with_name(src.name + '.bak')
    if backup.exists():
        raise SystemExit(
            f'{backup} already exists. Move it aside before replacing '
            f'{src.name}.'
        )
    os.rename(src, backup)
    try:
        os.rename(dst, src)
    except Exception:
        print(
            f'Rename of the new table failed. The original is at {backup} '
            f'and the rewrite is at {dst}.'
        )
        raise
    print(f'Replaced {src}')
    print(f'Previous table kept at {backup}')
    print(f'Delete it once training opens the new table: rm -rf {backup}')


def main():
    args = _parse()
    if args.rows_per_commit < 1:
        raise SystemExit('--rows-per-commit must be positive')
    src = args.src or (
        _datasets_dir() / 'pusht_expert_train_dino.lance'
    )
    if not src.is_dir():
        raise SystemExit(f'Missing source table: {src}')
    dst = src.with_name(src.stem + '_ordered.lance')
    if dst.resolve() == src.resolve():
        raise SystemExit('Refusing to write the rewrite on top of the source.')

    import lance

    source = lance.dataset(str(src))
    names = set(source.schema.names)
    for column in ('episode_idx', 'step_idx'):
        if column not in names:
            raise SystemExit(f'{src} has no {column} column')
    episode_idx = _column(source, 'episode_idx')
    step_idx = _column(source, 'step_idx')
    if _in_order(episode_idx):
        print(f'{src} is already in episode order. Nothing to rewrite.')
        return
    runs = _runs(episode_idx, step_idx)
    n_rows = len(episode_idx)
    print(
        f'{src}: {len(runs)} episodes, {n_rows} rows, out of episode order. '
        'Copying runs into episode_idx order.'
    )

    if args.overwrite and dst.exists():
        shutil.rmtree(dst)
        print(f'Removed partial rewrite {dst}')

    resume = dst.exists()
    dest_dataset = lance.dataset(str(dst)) if resume else None
    done_rows = _prefix_length(dest_dataset, runs, step_idx)
    start_at = 0
    counted = 0
    for index, (_, start, end) in enumerate(runs):
        if counted == done_rows:
            start_at = index
            break
        counted += end - start
    else:
        start_at = len(runs)
    if done_rows:
        print(f'Resuming at row {done_rows} ({start_at}/{len(runs)} episodes).')

    if start_at < len(runs):
        src_bytes = _dir_bytes(src)
        dst_bytes = _dir_bytes(dst) if dst.exists() else 0
        need = max(src_bytes - dst_bytes, 0)
        free = shutil.disk_usage(src.parent).free
        print(
            f'Source is {src_bytes / 1e9:.1f} GB. Need about '
            f'{need / 1e9:.1f} GB free for the copy; {free / 1e9:.1f} GB free.'
        )
        if free < need + (1 << 30):
            raise SystemExit(
                'Not enough free disk to write the reordered copy next to '
                'the source. Free space, then re-run. The original table is '
                'unchanged.'
            )
        writer = None
        started = time.perf_counter()
        written_now = 0
        for batch in _batches(runs[start_at:], args.rows_per_commit):
            taken = source.take(_indices(batch).tolist())
            if writer is None:
                writer = _open_writer(dst, taken.schema, resume)
                resume = True
            writer.add(taken, write_parallelism=1)
            n_batch = taken.num_rows
            done_rows += n_batch
            written_now += n_batch
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f'rows {done_rows}/{n_rows}  '
                f'{written_now / elapsed:.0f} rows/s'
            )
            del taken

    rewritten = lance.dataset(str(dst))
    _verify(rewritten, runs, episode_idx, step_idx)
    print(f'Verified {rewritten.count_rows()} rows in episode order.')
    _swap(src, dst)


if __name__ == '__main__':
    main()
