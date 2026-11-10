"""Rewrite the on-disk DINOv2 Lance table into episode order.

``precompute_dino.py`` used LanceDB's default parallel writer. That
round-robins row groups across fragments, so a scan of ``episode_idx``
decreases even though each episode is still one contiguous run. This
copies those runs into a new table in ``episode_idx`` order. It does not
re-encode frames.

Fragments cover a short span of episodes. Once every episode in a
fragment has been copied, that fragment is removed and its files are
deleted, so the rewrite does not need a second copy of the table.

Usage on the training machine::

    python scripts/train/repair_dino_order.py

Resume is automatic. ``--overwrite`` drops a partial rewrite and starts
again; it refuses once source fragments have already been deleted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import timedelta
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
        help='Frames copied per commit. A crash loses at most one commit.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete a partial rewrite and start again.',
    )
    return parser.parse_args()


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total


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


def _scan_runs(episode_idx: np.ndarray, step_idx: np.ndarray):
    """Contiguous ``(episode, start_row, end_row)`` runs, in scan order."""
    if len(episode_idx) == 0:
        return []
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
    return runs


def _fragment_bounds(dataset):
    bounds = []
    offset = 0
    for fragment in dataset.get_fragments():
        count = int(fragment.count_rows())
        bounds.append((fragment.fragment_id, offset, offset + count))
        offset += count
    return bounds


def _locate(runs, bounds):
    """Map each scan run to ``(episode, fragment_id, local_start, local_end)``."""
    if not runs:
        return [], {}
    starts = np.array([start for _, start, _ in bounds], dtype=np.int64)
    located = []
    frag_episodes: dict[int, set[int]] = {}
    for episode, start, end in runs:
        index = int(np.searchsorted(starts, start, side='right') - 1)
        if index < 0 or index >= len(bounds):
            raise SystemExit(f'episode {episode} does not fall in a fragment.')
        frag_id, frag_start, frag_end = bounds[index]
        if end > frag_end:
            raise SystemExit(
                f'episode {episode} spans two fragments and cannot be '
                'reclaimed one fragment at a time.'
            )
        located.append((episode, int(frag_id), start - frag_start, end - frag_start))
        frag_episodes.setdefault(int(frag_id), set()).add(episode)
    located.sort(key=lambda run: run[0])
    return located, frag_episodes


def _plan_path(src: Path) -> Path:
    return src.with_name(src.stem + '_ordered.plan.npz')


def _save_plan(path: Path, episodes: np.ndarray, lengths: np.ndarray):
    np.savez(path, episodes=episodes.astype(np.int64), lengths=lengths.astype(np.int64))


def _load_plan(path: Path):
    with np.load(path) as data:
        return data['episodes'].astype(np.int64), data['lengths'].astype(np.int64)


def _open_writer(dst: Path, schema: pa.Schema, resume: bool):
    import lancedb

    db = lancedb.connect(str(dst.parent))
    if resume:
        return db.open_table(dst.stem)
    if dst.exists():
        shutil.rmtree(dst)
    return db.create_table(dst.stem, schema=schema)


def _read_runs(dataset, batch):
    fragments = {fragment.fragment_id: fragment for fragment in dataset.get_fragments()}
    tables = []
    for episode, frag_id, local_start, local_end in batch:
        fragment = fragments.get(frag_id)
        if fragment is None:
            raise SystemExit(
                f'Fragment {frag_id} for episode {episode} is missing. '
                'The source table changed.'
            )
        tables.append(
            fragment.scanner(offset=local_start, limit=local_end - local_start).to_table()
        )
    return pa.concat_tables(tables)


def _expect_episodes(batch) -> np.ndarray:
    parts = [
        np.full(local_end - local_start, episode, dtype=np.int64)
        for episode, _, local_start, local_end in batch
    ]
    return np.concatenate(parts)


def _drop_fragments(src: Path, drop_ids: set[int]):
    import lance

    dataset = lance.dataset(str(src))
    keep = [
        fragment.metadata
        for fragment in dataset.get_fragments()
        if fragment.fragment_id not in drop_ids
    ]
    operation = lance.LanceOperation.Overwrite(dataset.lance_schema, keep)
    dataset = lance.LanceDataset.commit(str(src), operation)
    stats = dataset.cleanup_old_versions(
        older_than=timedelta(seconds=0),
        delete_unverified=True,
        error_if_tagged_old_versions=False,
    )
    print(f'Removed {len(drop_ids)} source fragments. {stats}')
    return lance.dataset(str(src))


def _dest_episode_lengths(dataset):
    if dataset is None or dataset.count_rows() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    episode_idx = _column(dataset, 'episode_idx').astype(np.int64, copy=False)
    if not _in_order(episode_idx):
        raise SystemExit(
            'Partial rewrite is itself out of episode order. Re-run with '
            '--overwrite only if the source still has every episode.'
        )
    runs = _scan_runs(episode_idx, _column(dataset, 'step_idx'))
    episodes = np.array([episode for episode, _, _ in runs], dtype=np.int64)
    lengths = np.array([end - start for _, start, end in runs], dtype=np.int64)
    if len(episodes) != len(np.unique(episodes)):
        raise SystemExit('Partial rewrite repeats an episode.')
    return episodes, lengths


def _check_dest_prefix(dest_episodes, dest_lengths, plan_episodes, plan_lengths):
    count = len(dest_episodes)
    if count == 0:
        return
    if not np.array_equal(dest_episodes, plan_episodes[:count]):
        raise SystemExit(
            'Partial rewrite is not a prefix of the planned episode order. '
            'Re-run with --overwrite only if the source still has every episode.'
        )
    if not np.array_equal(dest_lengths, plan_lengths[:count]):
        raise SystemExit(
            'Partial rewrite episode lengths do not match the plan.'
        )


def _batches(runs, rows_per_commit: int):
    batch = []
    rows = 0
    for run in runs:
        batch.append(run)
        rows += run[3] - run[2]
        if rows >= rows_per_commit:
            yield batch
            batch = []
            rows = 0
    if batch:
        yield batch


def _swap(src: Path, dst: Path, plan: Path):
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
    plan.unlink(missing_ok=True)
    print(f'Replaced {src}')
    print(f'Old manifest kept at {backup}')
    print(f'Remove it with: rm -rf {backup}')


def main():
    args = _parse()
    if args.rows_per_commit < 1:
        raise SystemExit('--rows-per-commit must be positive')
    src = args.src or (_datasets_dir() / 'pusht_expert_train_dino.lance')
    if not src.is_dir():
        raise SystemExit(f'Missing source table: {src}')
    dst = src.with_name(src.stem + '_ordered.lance')
    plan_file = _plan_path(src)
    if dst.resolve() == src.resolve():
        raise SystemExit('Refusing to write the rewrite on top of the source.')

    import lance

    source = lance.dataset(str(src))
    names = set(source.schema.names)
    for column in ('episode_idx', 'step_idx'):
        if column not in names:
            raise SystemExit(f'{src} has no {column} column')

    source_episode = _column(source, 'episode_idx') if source.count_rows() else np.array([])
    source_step = _column(source, 'step_idx') if source.count_rows() else np.array([])
    source_ids_now = set(int(episode) for episode in np.unique(source_episode)) if len(source_episode) else set()
    if args.overwrite and plan_file.exists():
        planned, _ = _load_plan(plan_file)
        gone = [int(episode) for episode in planned if int(episode) not in source_ids_now]
        if gone:
            raise SystemExit(
                '--overwrite would delete episodes that were already removed '
                f'from the source (first id: {gone[0]}).'
            )
    if args.overwrite:
        if dst.exists():
            shutil.rmtree(dst)
            print(f'Removed partial rewrite {dst}')
        plan_file.unlink(missing_ok=True)
    if source.count_rows() and _in_order(source_episode) and not dst.exists():
        print(f'{src} is already in episode order. Nothing to rewrite.')
        return

    scan_runs = _scan_runs(source_episode, source_step)
    located, frag_episodes = _locate(scan_runs, _fragment_bounds(source))
    source_ids = np.array([episode for episode, _, _, _ in located], dtype=np.int64)

    dest_dataset = lance.dataset(str(dst)) if dst.exists() else None
    dest_episodes, dest_lengths = _dest_episode_lengths(dest_dataset)

    if plan_file.exists():
        plan_episodes, plan_lengths = _load_plan(plan_file)
    else:
        if len(dest_episodes) and len(np.intersect1d(dest_episodes, source_ids)):
            raise SystemExit(
                'A partial rewrite exists but its plan file does not. '
                'Re-run with --overwrite if the source still has every episode.'
            )
        by_id = {episode: local_end - local_start for episode, _, local_start, local_end in located}
        for episode, length in zip(dest_episodes, dest_lengths):
            by_id.setdefault(int(episode), int(length))
        plan_episodes = np.array(sorted(by_id), dtype=np.int64)
        plan_lengths = np.array([by_id[int(episode)] for episode in plan_episodes], dtype=np.int64)
        _save_plan(plan_file, plan_episodes, plan_lengths)

    _check_dest_prefix(dest_episodes, dest_lengths, plan_episodes, plan_lengths)
    copied = set(int(episode) for episode in dest_episodes)
    missing = [int(episode) for episode in plan_episodes if int(episode) not in copied and int(episode) not in set(source_ids.tolist())]
    if missing:
        raise SystemExit(
            f'{len(missing)} episodes are in neither the source nor the '
            f'partial rewrite (first missing id: {missing[0]}).'
        )
    remaining = [run for run in located if run[0] not in copied]
    n_rows = int(plan_lengths.sum())
    done_rows = int(dest_lengths.sum()) if len(dest_lengths) else 0
    print(
        f'{src}: {len(plan_episodes)} episodes, {n_rows} rows. '
        f'{len(copied)} episodes already copied. '
        'Source fragments are deleted once their episodes are in the new table.'
    )
    if done_rows:
        print(f'Resuming at row {done_rows}.')

    writer = None
    resume = dest_dataset is not None and done_rows > 0
    started = time.perf_counter()
    written_now = 0
    src_bytes = _dir_bytes(src)
    bytes_per_row = src_bytes / max(int(source.count_rows()), 1)
    ready = {
        frag_id
        for frag_id, episodes in frag_episodes.items()
        if episodes and episodes <= copied
    }
    if ready:
        _drop_fragments(src, ready)
        for frag_id in ready:
            del frag_episodes[frag_id]

    for batch in _batches(remaining, args.rows_per_commit):
        batch_rows = sum(local_end - local_start for _, _, local_start, local_end in batch)
        free = shutil.disk_usage(src.parent).free
        need = int(batch_rows * bytes_per_row * 1.5) + (1 << 30)
        if free < need:
            raise SystemExit(
                f'Need about {need / 1e9:.1f} GB free for the next '
                f'{batch_rows} rows; {free / 1e9:.1f} GB free. '
                f'{done_rows} rows are already in the new table. Re-run to continue.'
            )
        source = lance.dataset(str(src))
        taken = _read_runs(source, batch)
        got = taken.column('episode_idx').to_numpy()
        if not np.array_equal(got.astype(np.int64, copy=False), _expect_episodes(batch)):
            raise SystemExit('Copied rows are not the expected episodes.')
        if writer is None:
            writer = _open_writer(dst, taken.schema, resume)
            resume = True
        writer.add(taken, write_parallelism=1)
        done_rows += taken.num_rows
        written_now += taken.num_rows
        copied.update(episode for episode, _, _, _ in batch)
        del taken
        ready = {
            frag_id
            for frag_id, episodes in frag_episodes.items()
            if episodes and episodes <= copied
        }
        if ready:
            _drop_fragments(src, ready)
            for frag_id in ready:
                del frag_episodes[frag_id]
        elapsed = max(time.perf_counter() - started, 1e-6)
        print(
            f'rows {done_rows}/{n_rows}  '
            f'{written_now / elapsed:.0f} rows/s  '
            f'{shutil.disk_usage(src.parent).free / 1e9:.1f} GB free'
        )

    rewritten = lance.dataset(str(dst))
    got_episodes, got_lengths = _dest_episode_lengths(rewritten)
    if not np.array_equal(got_episodes, plan_episodes) or not np.array_equal(
        got_lengths, plan_lengths
    ):
        raise SystemExit('Rewritten table does not match the episode plan.')
    print(f'Verified {rewritten.count_rows()} rows in episode order.')
    _swap(src, dst, plan_file)


if __name__ == '__main__':
    main()
