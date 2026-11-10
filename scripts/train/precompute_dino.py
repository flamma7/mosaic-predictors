"""Precompute DINOv2 patch tokens for the PushT Lance dataset.

The source file stores JPEG frames. This writes a second Lance table with
those columns unchanged plus ``dino``: one fp16 DINOv2 sequence per
frame, CLS then the patch grid, flattened ``(1 + num_patches) *
embed_dim``. Training drops the CLS token. Row order and ``episode_idx`` match
the source. PushT does not store a train/test split; ``dinowm.py`` splits
with ``random_split`` and a fixed seed, which depends only on episode
lengths, so the copy keeps that split.

Train on the cache with::

    python scripts/train/dinowm.py data=pusht_dino

MPC and the perturbed open-loop MSE still run the encoder. The cache is
only the clean dataset frames.

Output lands in ``$STABLEWM_HOME/datasets/pusht_expert_train_dino.lance``
(or ``~/.stable_worldmodel/datasets`` when ``STABLEWM_HOME`` is unset).
The job resumes by skipping episodes whose row count already matches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_REPO = 'flamma77/pusht_expert_train_dino'


def _datasets_dir() -> Path:
    import stable_worldmodel as swm

    return swm.data.utils.get_cache_dir(sub_folder='datasets')


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--src',
        type=Path,
        default=None,
        help='Source Lance table. Default: STABLEWM_HOME/datasets/'
        'pusht_expert_train.lance',
    )
    parser.add_argument(
        '--dst',
        type=Path,
        default=None,
        help='Destination Lance table. Default: STABLEWM_HOME/datasets/'
        'pusht_expert_train_dino.lance',
    )
    parser.add_argument('--backbone', default='dinov2_small')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--encoder-img-size', type=int, default=196)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--embed-dim', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--episodes-per-commit',
        type=int,
        default=32,
        help='Episodes per Lance commit. A crash loses at most one commit.',
    )
    parser.add_argument(
        '--limit-episodes',
        type=int,
        default=None,
        help='Stop after this many source episodes. A later run without '
        'the limit continues. Refuses --upload.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Drop the destination table and write it from scratch.',
    )
    parser.add_argument(
        '--upload',
        nargs='?',
        const=DEFAULT_REPO,
        default=None,
        help='Upload to this Hugging Face dataset repo when the table is '
        f'complete. Bare --upload uses {DEFAULT_REPO}.',
    )
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def _table_names(db) -> list[str]:
    listed = db.list_tables()
    if hasattr(listed, 'tables'):
        return list(listed.tables)
    return list(listed)


def _connect(lance_path: Path):
    import lancedb

    lance_path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(lance_path.parent)), lance_path.stem


def _as_table(taken):
    if isinstance(taken, pa.Table):
        return taken
    return pa.Table.from_batches([taken])


def _episode_ids(table) -> np.ndarray:
    column = table.to_lance().to_table(columns=['episode_idx']).column(
        'episode_idx'
    )
    return column.to_numpy()


def _spans(episode_idx: np.ndarray):
    """Contiguous ``(episode_idx, start_row, end_row)`` runs, in row order."""
    if len(episode_idx) == 0:
        raise RuntimeError('Source dataset has no rows.')
    change = np.flatnonzero(np.diff(episode_idx)) + 1
    bounds = np.concatenate([[0], change, [len(episode_idx)]])
    spans = []
    seen = set()
    for index in range(len(bounds) - 1):
        start = int(bounds[index])
        end = int(bounds[index + 1])
        episode = int(episode_idx[start])
        if episode in seen:
            raise RuntimeError(
                f'episode_idx {episode} is not stored contiguously. '
                'The train split follows row order, so this file cannot '
                'be copied as-is.'
            )
        seen.add(episode)
        spans.append((episode, start, end))
    return spans


def _counts(table) -> dict[int, int]:
    ids = _episode_ids(table)
    if len(ids) == 0:
        return {}
    unique, counts = np.unique(ids, return_counts=True)
    return {int(episode): int(count) for episode, count in zip(unique, counts)}


def _resume_at(spans, counts):
    """First span index still to write, and episode ids in the incomplete tail."""
    for index, (episode, start, end) in enumerate(spans):
        have = counts.get(episode, 0)
        need = end - start
        if have == need:
            continue
        if have > need:
            raise RuntimeError(
                f'Episode {episode} has {have} cached rows but the source '
                f'has {need}. Re-run with --overwrite.'
            )
        delete = [
            later
            for later, _, _ in spans[index:]
            if counts.get(later, 0)
        ]
        return index, delete
    return len(spans), []


def _delete_episodes(table, episode_ids):
    ids = sorted({int(episode) for episode in episode_ids})
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        listed = ', '.join(str(episode) for episode in chunk)
        table.delete(f'episode_idx IN ({listed})')


def _decode_jpegs(blobs) -> torch.Tensor:
    from torchvision.io import ImageReadMode, decode_jpeg

    tensors = []
    for blob in blobs:
        raw = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
        tensors.append(torch.frombuffer(raw, dtype=torch.uint8))
    return torch.stack(decode_jpeg(tensors, mode=ImageReadMode.RGB))


def _fp16_fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = np.ascontiguousarray(
        values.reshape(values.shape[0], -1), dtype=np.float16
    )
    width = flat.shape[1]
    data = pa.array(flat.reshape(-1), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(data, width)


def _dino_metadata(args, num_patches: int) -> dict[bytes, bytes]:
    values = {
        'backbone': args.backbone,
        'img_size': args.img_size,
        'encoder_img_size': args.encoder_img_size,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_patches': num_patches,
        'num_tokens': num_patches + 1,
        'dtype': 'float16',
        'layout': 'cls_then_row_major_patches',
    }
    return {key.encode(): str(value).encode() for key, value in values.items()}


def _with_dino(table: pa.Table, tokens: np.ndarray, metadata: dict):
    column = _fp16_fixed_list(tokens)
    out = table.append_column('dino', column)
    fields = []
    for field in out.schema:
        if field.name == 'dino':
            field = field.with_metadata(metadata)
        fields.append(field)
    schema = pa.schema(fields)
    return out.cast(schema)


def _load_encoder(name: str, device: torch.device):
    from stable_worldmodel.wm.prejepa.module import create_backbone

    encoder = create_backbone(name)
    encoder.eval()
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def _check_preprocess(preprocessor, frames: torch.Tensor):
    """Batched resize must match the per-frame path the trainer uses."""
    if frames.shape[0] < 2:
        return
    batch = preprocessor({'pixels': frames[:2]})['pixels']
    solo = torch.stack(
        [
            preprocessor({'pixels': frames[index : index + 1]})['pixels'][0]
            for index in range(2)
        ]
    )
    if batch.shape != solo.shape or not torch.allclose(batch, solo):
        raise RuntimeError(
            'Batched image preprocess does not match per-frame preprocess.'
        )


def _encode_frames(encoder, preprocessor, frames, device, num_patches, embed_dim):
    images = preprocessor({'pixels': frames})['pixels']
    images = images.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16
    ):
        output = encoder(images, interpolate_pos_encoding=True)
        hidden = output.last_hidden_state
    n_tokens = num_patches + 1
    if hidden.shape[1] != n_tokens or hidden.shape[2] != embed_dim:
        raise RuntimeError(
            f'Encoder returned {tuple(hidden.shape)}, expected '
            f'(N, {n_tokens}, {embed_dim}) with the CLS token kept.'
        )
    return hidden.to(dtype=torch.float16).cpu().numpy()


def _encode_episode(
    encoder, preprocessor, frames, device, batch_size, num_patches, embed_dim
):
    chunks = []
    for start in range(0, frames.shape[0], batch_size):
        chunks.append(
            _encode_frames(
                encoder,
                preprocessor,
                frames[start : start + batch_size],
                device,
                num_patches,
                embed_dim,
            )
        )
    return np.concatenate(chunks, axis=0)


def _copy_episodes_table(src_db, dst_db, src_stem: str, dst_stem: str):
    src_name = f'{src_stem}_episodes'
    dst_name = f'{dst_stem}_episodes'
    if src_name not in _table_names(src_db):
        return
    if dst_name in _table_names(dst_db):
        return
    data = src_db.open_table(src_name).to_arrow()
    dst_db.create_table(dst_name, data)
    print(f'Copied episode side table {src_name} -> {dst_name}')


def _flush(dst_db, dst_stem, held_tables, held_tokens, metadata, dest):
    tokens = np.concatenate(held_tokens, axis=0)
    table = _with_dino(pa.concat_tables(held_tables), tokens, metadata)
    if table.num_rows != tokens.shape[0]:
        raise RuntimeError('DINO rows do not match the copied frame rows.')
    # Default parallelism round-robins Arrow row groups (one per episode
    # here) into separate fragments, so a scan of episode_idx decreases.
    # One writer keeps the source order.
    if dest is None:
        dest = dst_db.create_table(dst_stem, schema=table.schema)
    dest.add(table, write_parallelism=1)
    held_tables.clear()
    held_tokens.clear()
    return dest


def _upload_command(path: Path, repo_id: str) -> str:
    lines = [
        'hf upload'
        f' {repo_id}'
        f' {path}'
        f' {path.name}'
        ' --repo-type dataset'
    ]
    side = path.parent / f'{path.stem}_episodes.lance'
    if side.is_dir():
        lines.append(
            'hf upload'
            f' {repo_id}'
            f' {side}'
            f' {side.name}'
            ' --repo-type dataset'
        )
    return '\n'.join(lines)


def _upload(path: Path, repo_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type='dataset', exist_ok=True)
    print(f'Uploading {path} to {repo_id}')
    api.upload_folder(
        repo_id=repo_id,
        repo_type='dataset',
        folder_path=str(path),
        path_in_repo=path.name,
    )
    side = path.parent / f'{path.stem}_episodes.lance'
    if side.is_dir():
        print(f'Uploading {side} to {repo_id}')
        api.upload_folder(
            repo_id=repo_id,
            repo_type='dataset',
            folder_path=str(side),
            path_in_repo=side.name,
        )
    print(f'Uploaded dataset to https://huggingface.co/datasets/{repo_id}')


def main():
    args = _parse()
    if args.encoder_img_size % args.patch_size != 0:
        raise SystemExit('encoder image size must be a multiple of patch size')
    if args.batch_size < 1 or args.episodes_per_commit < 1:
        raise SystemExit('batch size and episodes per commit must be positive')
    if args.upload and args.limit_episodes is not None:
        raise SystemExit('--upload refuses a --limit-episodes table')

    datasets = _datasets_dir()
    src_path = args.src or (datasets / 'pusht_expert_train.lance')
    dst_path = args.dst or (datasets / 'pusht_expert_train_dino.lance')
    if src_path.resolve() == dst_path.resolve():
        raise SystemExit('source and destination are the same path')
    if not src_path.exists():
        raise SystemExit(f'Missing source dataset: {src_path}')

    num_patches = (args.encoder_img_size // args.patch_size) ** 2
    from dinowm import DINO_COLUMN, get_img_preprocessor

    if DINO_COLUMN != 'dino':
        raise SystemExit(
            f'dinowm.DINO_COLUMN is {DINO_COLUMN!r}, this writer emits "dino"'
        )

    src_db, src_stem = _connect(src_path)
    dst_db, dst_stem = _connect(dst_path)
    if args.overwrite:
        for name in (dst_stem, f'{dst_stem}_episodes'):
            if name in _table_names(dst_db):
                dst_db.drop_table(name)
                print(f'Dropped {name}')

    src_table = src_db.open_table(src_stem)
    if 'pixels' not in src_table.schema.names:
        raise SystemExit(f'{src_path} has no pixels column')
    if 'dino' in src_table.schema.names:
        raise SystemExit(f'{src_path} already has a dino column')

    episode_idx = _episode_ids(src_table)
    spans = _spans(episode_idx)
    if args.limit_episodes is not None:
        spans = spans[: args.limit_episodes]
    n_rows = sum(end - start for _, start, end in spans)
    n_tokens = num_patches + 1
    token_bytes = n_rows * n_tokens * args.embed_dim * 2
    print(
        f'Source {src_path}: {len(spans)} episodes, {n_rows} frames. '
        f'DINO sequence {n_tokens} x {args.embed_dim} fp16 '
        f'(CLS + {num_patches} patches) '
        f'is {token_bytes / 1e9:.1f} GB plus the copied columns.'
    )
    print(
        'PushT has no split column. Episode order is copied, so '
        'dinowm random_split(seed=3072, train_split=0.9) is unchanged.'
    )

    dest = None
    if dst_stem in _table_names(dst_db):
        dest = dst_db.open_table(dst_stem)
        width = dest.schema.field('dino').type.list_size
        expect = (num_patches + 1) * args.embed_dim
        if width != expect:
            raise SystemExit(
                f'Existing dino column has width {width}, this run writes '
                f'{expect} (CLS + {num_patches} patches x {args.embed_dim}). '
                'Re-run with --overwrite.'
            )
        start_at, delete = _resume_at(spans, _counts(dest))
        if delete:
            print(f'Deleting {len(delete)} partial or trailing episodes')
            _delete_episodes(dest, delete)
    else:
        start_at = 0

    _copy_episodes_table(src_db, dst_db, src_stem, dst_stem)

    if start_at >= len(spans):
        print(f'{dst_path} already has every requested episode.')
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise SystemExit(
                'CUDA is not available. Pass --device or use a GPU.'
            )
        encoder = _load_encoder(args.backbone, device)
        hidden = int(encoder.config.hidden_size)
        patch = int(encoder.config.patch_size)
        if hidden != args.embed_dim or patch != args.patch_size:
            raise SystemExit(
                f'{args.backbone} is hidden {hidden}, patch {patch}. '
                f'Expected hidden {args.embed_dim}, patch {args.patch_size}.'
            )
        preprocessor = get_img_preprocessor(
            source='pixels',
            target='pixels',
            image_size=args.img_size,
            encoder_image_size=args.encoder_img_size,
        )
        metadata = _dino_metadata(args, num_patches)
        src_lance = src_table.to_lance()
        held_tables = []
        held_tokens = []
        checked = False
        written = sum(end - start for _, start, end in spans[:start_at])
        written_now = 0
        started = time.perf_counter()
        for done, (episode, start, end) in enumerate(
            spans[start_at:], start=start_at
        ):
            taken = _as_table(src_lance.take(list(range(start, end))))
            frames = _decode_jpegs(taken.column('pixels').to_pylist())
            if not checked and frames.shape[0] >= 2:
                _check_preprocess(preprocessor, frames)
                checked = True
            tokens = _encode_episode(
                encoder,
                preprocessor,
                frames,
                device,
                args.batch_size,
                num_patches,
                args.embed_dim,
            )
            if tokens.shape[0] != taken.num_rows:
                raise RuntimeError(
                    f'Episode {episode}: encoded {tokens.shape[0]} frames, '
                    f'source has {taken.num_rows}.'
                )
            held_tables.append(taken)
            held_tokens.append(tokens)
            full = len(held_tables) >= args.episodes_per_commit
            last = done + 1 == len(spans)
            if full or last:
                n_held = sum(table.num_rows for table in held_tables)
                dest = _flush(
                    dst_db,
                    dst_stem,
                    held_tables,
                    held_tokens,
                    metadata,
                    dest,
                )
                written += n_held
                written_now += n_held
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f'episodes {done + 1}/{len(spans)}  '
                    f'rows {written}/{n_rows}  '
                    f'{written_now / elapsed:.1f} frames/s'
                )

    if dest is None and dst_stem in _table_names(dst_db):
        dest = dst_db.open_table(dst_stem)
    if dest is not None:
        got_ids = _episode_ids(dest)
        got = len(got_ids)
        print(f'Wrote {dst_path}: {got} rows, {len(spans)} episodes requested.')
        if args.limit_episodes is None and got != len(episode_idx):
            raise SystemExit(
                f'Destination has {got} rows, source has {len(episode_idx)}.'
            )
        if len(got_ids) > 1 and (np.diff(got_ids.astype(np.int64)) < 0).any():
            raise SystemExit(
                'Destination episode_idx decreases. The Lance write did not '
                'keep source order.'
            )

    command = _upload_command(dst_path, args.upload or DEFAULT_REPO)
    print('Upload with:')
    print(command)
    if args.upload:
        if args.limit_episodes is not None:
            raise SystemExit('refusing to upload a partial table')
        _upload(dst_path, args.upload)


if __name__ == '__main__':
    main()
