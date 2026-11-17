"""Offline global RMS of the visual DINO tokens perturbations are added to.

Uses the PushT training split from a DINO-WM / DINO-GRU config. Cached
``dino`` rows are read when that column is loaded; otherwise the frozen
encoder runs. Proprioception and other non-visual tokens are excluded.
The value is printed for ``perturb.s_z`` and the config file is not edited.

Example::

    python scripts/train/compute_dino_sz.py --override data=pusht_dino \\
        --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dinowm  # noqa: E402
from mosaic_predictors.dinogru import latent_rms_from_sums  # noqa: E402


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=Path,
        default=SCRIPT_DIR / 'config' / 'dinogru.yaml',
        help='Hydra config to compose. Defaults are resolved from its directory.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Stop after this many frames. Default: every training-split frame.',
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        help='Extra Hydra override, e.g. --override data=pusht_dino',
    )
    return parser.parse_args()


def _compose(path: Path, overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None, config_dir=str(path.parent.resolve())
    ):
        return compose(config_name=path.stem, overrides=list(overrides))


def _visuals(model, batch, cfg, num_patches):
    if cfg.cached_dino:
        tokens = dinowm._patches_from_cache(
            batch[dinowm.DINO_COLUMN],
            num_patches,
            int(cfg.patch_embed_dim),
        )
        return tokens.detach().float()
    for key in model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)
    encoded = model.encode(batch, target='emb', is_video=False)
    return encoded['pixels_emb'].detach().float()


def main():
    args = _parse()
    cfg = _compose(args.config, args.override)
    use_cache = dinowm._resolve_cached_dino(cfg)

    import os

    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from omegaconf import OmegaConf, open_dict
    from stable_worldmodel.data import column_normalizer as get_column_normalizer

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    image_preprocess = dinowm.get_img_preprocessor(
        source='pixels',
        target='pixels',
        image_size=cfg.img_size,
        encoder_image_size=cfg.encoder_img_size,
    )
    column_transforms = []
    with open_dict(cfg):
        cfg.cached_dino = use_cache
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels') or col == dinowm.DINO_COLUMN:
                continue
            column_transforms.append(get_column_normalizer(dataset, col, col))
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
        dataset.transform = column_transform
    else:
        dataset.transform = spt.data.transforms.Compose(
            image_preprocess, *column_transforms
        )

    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_set, _val = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device(args.device)
    num_patches = (int(cfg.encoder_img_size) // int(cfg.patch_size)) ** 2
    model = None
    if not use_cache:
        with open_dict(cfg):
            cfg.model.predictor.num_patches = num_patches
            extra = int(cfg.action_emb_dim)
            if cfg.use_proprio:
                extra += int(cfg.proprio_emb_dim)
            cfg.model.predictor.dim = int(cfg.patch_embed_dim) + extra
            cfg.model.proprio_dim = (
                int(cfg.proprio_emb_dim) if cfg.use_proprio else 0
            )
        encoder = hydra.utils.instantiate(cfg.model.encoder)
        model = hydra.utils.instantiate(cfg.model, encoder=encoder)
        model = model.to(device).eval()

    sum_sq = 0.0
    count = 0
    seen_frames = 0
    limit = None if args.limit is None else int(args.limit)
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            visual = _visuals(model, batch, cfg, num_patches)
        flat = visual.reshape(-1, *visual.shape[-2:]).double()
        if limit is not None and seen_frames + flat.shape[0] > limit:
            flat = flat[: limit - seen_frames]
        sum_sq += flat.pow(2).sum().item()
        count += flat.numel()
        seen_frames += flat.shape[0]
        if seen_frames == 0 or seen_frames % 1024 < flat.shape[0]:
            print(
                f'[s_z] frames {seen_frames}  running {math.sqrt(sum_sq / count):.6f}',
                flush=True,
            )
        if limit is not None and seen_frames >= limit:
            break

    if count == 0:
        raise RuntimeError('No visual embeddings were read from the training split')
    s_z = latent_rms_from_sums(sum_sq, count)
    print(f's_z {s_z:.8f}')
    print('Set perturb.s_z to this value in dinogru.yaml.')
    return s_z


if __name__ == '__main__':
    main()
