"""Smoke tests for DINO-GRU variants A-E."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from stable_worldmodel.wm.prejepa.module import CausalPredictor, Embedder
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'train'))

from mosaic_predictors.dinogru import (  # noqa: E402
    AttentionPool,
    DinoGRU,
    eligible_predictions,
    latent_rms_from_sums,
    perturb_latents,
    require_s_z,
    residual_rms,
    sample_factors,
    shift_actions,
    stored_s_r,
)


def _load_train():
    spec = importlib.util.spec_from_file_location(
        'dinogru_train', ROOT / 'scripts' / 'train' / 'dinogru.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokens:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class TinyDino(nn.Module):
    def __init__(self, dim, patches):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.proj = nn.Linear(3, dim)

    def forward(self, pixels, interpolate_pos_encoding=False):
        color = pixels.mean(dim=(2, 3))
        patch = self.proj(color)
        tokens = patch[:, None, :].expand(-1, self.patches, -1).contiguous()
        cls = torch.zeros(
            tokens.shape[0], 1, self.dim, device=tokens.device, dtype=tokens.dtype
        )
        return _Tokens(torch.cat([cls, tokens], dim=1))


def _parts(seed, dropout=0.0, frames=3):
    torch.manual_seed(seed)
    visual, patches = 8, 4
    proprio, action = 4, 4
    encoder = TinyDino(visual, patches)
    predictor = CausalPredictor(
        num_patches=patches,
        num_frames=frames,
        dim=visual + proprio + action,
        depth=1,
        heads=4,
        mlp_dim=32,
        dim_head=8,
        dropout=dropout,
        emb_dropout=0.0,
        pool='mean',
    )
    extras = nn.ModuleDict(
        {
            'proprio': Embedder(in_chans=3, emb_dim=proprio),
            'action': Embedder(in_chans=4, emb_dim=action),
        }
    )
    return encoder, predictor, extras


def make_model(
    seed=0,
    gru=False,
    pooling='meanpool',
    num_queries=1,
    dropout=0.0,
    plan_history_len=64,
):
    encoder, predictor, extras = _parts(seed, dropout=dropout)
    return DinoGRU(
        encoder,
        predictor,
        extra_encoders=extras,
        history_size=3,
        num_pred=1,
        gru_enabled=gru,
        gru_hidden=16,
        gru_layers=1,
        gru_input_dim=16,
        gru_pooling=pooling,
        gru_num_queries=num_queries,
        plan_history_len=plan_history_len,
        visual_dim=8,
        proprio_dim=4,
        action_dim=4,
    )


class _Holder:
    def __init__(self, model):
        self.model = model

    def log_dict(self, values, **_kwargs):
        self.logs = values


def _batch(frames, batch=2):
    return {
        'pixels': torch.randn(batch, frames, 3, 8, 8),
        'action': torch.randn(batch, frames, 4),
        'proprio': torch.randn(batch, frames, 3),
    }


def _cfg(perturb=False, s_z=1.0, s_r=1.0, zero_fraction=0.0):
    return OmegaConf.create(
        {
            'wm': {'history_size': 3, 'num_preds': 1},
            'cached_dino': False,
            'seed': 0,
            'seq_len': 16,
            'perturb': {
                'enabled': perturb,
                'val_enabled': False,
                'alpha': 0.2,
                'rank': 4,
                'activation': 'gelu',
                'v1': 1.0,
                'v2': 1.0,
                'zero_fraction': zero_fraction,
                'eps': 1e-8,
                's_z': s_z,
                's_r': s_r,
                'calibrate_s_r': False,
            },
        }
    )


class DinoGRUTests(unittest.TestCase):
    def test_variant_a_matches_dinowm_forward(self):
        train = _load_train()
        left = make_model(seed=1, gru=False, dropout=0.1)
        encoder, predictor, extras = _parts(seed=2, dropout=0.1)
        right = PreJEPA(
            encoder,
            predictor,
            extra_encoders=extras,
            history_size=3,
            num_pred=1,
        )
        right.load_state_dict(left.state_dict())
        batch = _batch(4)
        cfg = _cfg(perturb=False)
        hold_l = _Holder(left)
        hold_r = _Holder(right)
        torch.manual_seed(5)
        out_l = train.dinogru_forward(
            hold_l, copy.deepcopy(batch), 'train', cfg
        )
        torch.manual_seed(5)
        out_r = train.dinowm_forward(
            hold_r, copy.deepcopy(batch), 'train', cfg
        )
        self.assertTrue(torch.equal(out_l['loss'], out_r['loss']))
        self.assertTrue(torch.equal(out_l['pixels_loss'], out_r['pixels_loss']))
        self.assertEqual(set(hold_l.logs), set(hold_r.logs))
        self.assertIsNone(left.plan_history_len)
        self.assertFalse(getattr(left.predictor, '_memory_slots', False))

    def test_short_window_matches_single_predict(self):
        model = make_model(gru=False)
        model.eval()
        emb = torch.randn(2, 4, 4, 16)
        pred, tgt = eligible_predictions(model, emb, None, 3, 1)
        manual = model.predict(emb[:, :3])
        self.assertTrue(torch.allclose(pred, manual))
        self.assertTrue(torch.allclose(tgt, emb[:, 1:]))

    def test_sequence_16_alignment(self):
        model = make_model(gru=True, pooling='meanpool')
        model.eval()
        emb = torch.randn(2, 16, 4, 16)
        visual = emb[..., :8]
        proprio = torch.randn(2, 16, 4)
        action = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(
            2, 16, 4
        ).contiguous()
        memory, _ = model.memory_sequence(visual, proprio, action)
        pred, tgt = eligible_predictions(model, emb, memory, 3, 1)
        self.assertEqual(pred.shape[1], 15)
        self.assertTrue(
            torch.allclose(
                pred[:, -1],
                model.predict(emb[:, 12:15], memory=memory[:, 14])[:, -1],
            )
        )
        self.assertTrue(torch.allclose(tgt[:, -1], emb[:, 15]))
        self.assertTrue(torch.allclose(tgt[:, 0], emb[:, 1]))
        base, _ = model.memory_sequence(visual, proprio, action)
        dropped_future = action.clone()
        dropped_future[:, 14:] = 0
        future, _ = model.memory_sequence(visual, proprio, dropped_future)
        self.assertTrue(torch.allclose(base[:, 14], future[:, 14]))
        dropped_prev = action.clone()
        dropped_prev[:, 13] = 0
        prev, _ = model.memory_sequence(visual, proprio, dropped_prev)
        self.assertFalse(torch.allclose(base[:, 14], prev[:, 14]))
        shifted = shift_actions(action)
        self.assertEqual(float(shifted[:, 0].abs().sum()), 0.0)
        self.assertTrue(torch.equal(shifted[:, 1:], action[:, :-1]))

    def test_perturbation_is_per_trajectory_and_spares_proprio(self):
        train = _load_train()
        model = make_model(gru=False)
        model.eval()
        batch = _batch(4)
        clean = train.dinogru_forward(
            _Holder(model), copy.deepcopy(batch), 'train', _cfg(perturb=False)
        )
        noisy = train.dinogru_forward(
            _Holder(model),
            copy.deepcopy(batch),
            'train',
            _cfg(perturb=True, zero_fraction=0.0),
        )
        self.assertTrue(torch.equal(clean['proprio_emb'], noisy['proprio_emb']))
        self.assertFalse(torch.allclose(clean['pixels_emb'], noisy['pixels_emb']))
        proprio = noisy['emb'][:, :, 0, 8:12]
        self.assertTrue(torch.allclose(proprio, noisy['proprio_emb']))
        w1, w2 = sample_factors(4, 8, 4, 1.0, 1.0, 'cpu')
        self.assertFalse(torch.allclose(w1[0], w1[1]))
        latent = torch.randn(4, 6, 5, 8)
        alpha = torch.full((4,), 0.2)
        mixed = perturb_latents(latent, w1, w2, alpha, 1.0, 1.0)
        one = perturb_latents(latent[:, :1], w1, w2, alpha, 1.0, 1.0)
        self.assertTrue(torch.allclose(mixed[:, :1], one, atol=1e-5))
        later = perturb_latents(latent[:, 4:5], w1, w2, alpha, 1.0, 1.0)
        self.assertTrue(torch.allclose(mixed[:, 4:5], later, atol=1e-5))
        zeros = torch.zeros(4)
        plain = perturb_latents(latent, w1, w2, zeros, 1.0, 1.0)
        self.assertTrue(torch.allclose(plain, latent))

    def test_relative_perturbation_rms(self):
        torch.manual_seed(0)
        dim, tokens, rank = 32, 16, 8
        calib = torch.randn(512, tokens, dim)
        s_z = latent_rms_from_sums(
            calib.double().pow(2).sum().item(), calib.numel()
        )
        s_r = residual_rms(
            calib, rank, 1.0, 1.0, num_matrices=32, seed=42
        )
        again = residual_rms(
            calib, rank, 1.0, 1.0, num_matrices=32, seed=42
        )
        self.assertEqual(s_r, again)
        latent = torch.randn(128, 2, tokens, dim)
        w1, w2 = sample_factors(128, dim, rank, 1.0, 1.0, 'cpu')
        alpha = torch.full((128,), 0.2)
        alpha[:32] = 0
        mixed = perturb_latents(latent, w1, w2, alpha, s_z, s_r)
        self.assertTrue(torch.allclose(mixed[:32], latent[:32]))
        delta = (mixed[32:] - latent[32:]).float()
        base = latent[32:].float()
        ratio = delta.pow(2).mean().sqrt() / base.pow(2).mean().sqrt()
        self.assertLess(abs(float(ratio) - 0.2), 0.05, msg=f'ratio={float(ratio)}')

    def test_gru_is_causal_and_resettable(self):
        model = make_model(gru=True, pooling='meanpool')
        model.train()
        visual = torch.randn(2, 8, 4, 8, requires_grad=True)
        proprio = torch.randn(2, 8, 4)
        action = torch.randn(2, 8, 4)
        memory, _ = model.memory_sequence(visual, proprio, action)
        memory[:, -1].sum().backward()
        self.assertGreater(float(visual.grad[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(model.memory.gru.weight_hh_l0.grad.abs().sum()), 0.0)
        visual2 = torch.randn(2, 8, 4, 8, requires_grad=True)
        memory2, _ = model.memory_sequence(visual2, proprio, action)
        memory2[:, 2].sum().backward()
        self.assertEqual(float(visual2.grad[:, 3:].abs().sum()), 0.0)
        self.assertGreater(float(visual2.grad[:, :3].abs().sum()), 0.0)
        prefix, _ = model.memory_sequence(
            visual2[:, :3].detach(), proprio[:, :3], action[:, :3]
        )
        full, _ = model.memory_sequence(
            visual2.detach(), proprio, action
        )
        self.assertTrue(torch.allclose(full[:, :3], prefix, atol=1e-5))

        model.gru_reset_each_step = True
        model.eval()
        reset, _ = model.memory_sequence(visual2.detach(), proprio, action)
        carried, _ = model.memory(
            visual2.detach(),
            proprio,
            action,
            reset_each=False,
        )
        self.assertFalse(torch.allclose(reset[:, -1], carried[:, -1]))
        previous = shift_actions(action)
        step, _ = model.memory(
            visual2.detach()[:, -1:],
            proprio[:, -1:],
            previous[:, -1:],
            previous_actions=previous[:, -1:],
            reset_each=True,
        )
        self.assertTrue(torch.allclose(reset[:, -1:], step, atol=1e-5))

    def test_attention_pool_queries_and_single_gru(self):
        tokens = torch.randn(2, 5, 6, 8)
        pool = AttentionPool(8, 3)
        weights = pool.weights(tokens)
        self.assertTrue(
            torch.allclose(weights.sum(-1), torch.ones(2, 5, 3), atol=1e-5)
        )
        edited = tokens.clone()
        edited[:, 3:] = 0
        self.assertTrue(
            torch.allclose(pool.weights(tokens)[:, :3], pool.weights(edited)[:, :3])
        )
        for queries in (1, 3):
            model = make_model(
                gru=True, pooling='attention', num_queries=queries, seed=queries
            )
            grus = [module for module in model.modules() if isinstance(module, nn.GRU)]
            self.assertEqual(len(grus), 1)
            self.assertEqual(
                model.memory.input_mlp[0].in_features, queries * 8 + 4 + 4
            )
            self.assertEqual(model.memory.gru.input_size, 16)
        model = make_model(gru=True, pooling='attention', num_queries=2)
        model.train()
        visual = torch.randn(2, 3, 4, 8)
        proprio = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 4)
        emb = torch.randn(2, 3, 4, 16)
        memory, _ = model.memory_sequence(visual, proprio, action)
        model.predict(emb, memory=memory[:, -1]).sum().backward()
        self.assertIsNotNone(model.memory.pool.queries.grad)
        self.assertGreater(float(model.memory.pool.queries.grad.abs().sum()), 0.0)
        mean = make_model(gru=True, pooling='meanpool')
        pooled = mean.memory.pool(tokens)
        self.assertTrue(torch.allclose(pooled, tokens.mean(dim=2)))

    def test_memory_token_is_causal(self):
        model = make_model(gru=True)
        bias = model.predictor.transformer.layers[0][0].bias[0, 0]
        patches = 4
        frame1 = 1 + patches
        self.assertEqual(model.predictor.pos_embedding.shape[1], 1 + 3 * patches)
        self.assertEqual(float(bias[0, 0]), 1.0)
        self.assertEqual(float(bias[0, 1]), 0.0)
        self.assertEqual(float(bias[1, 0]), 1.0)
        self.assertEqual(float(bias[1, frame1]), 0.0)
        self.assertEqual(float(bias[frame1, 0]), 1.0)
        self.assertEqual(float(bias[frame1, 1]), 1.0)
        emb = torch.randn(2, 3, patches, 16)
        token = torch.randn(2, 16)
        self.assertEqual(model.predict(emb, memory=token).shape, emb.shape)

    def test_rollout_keeps_real_memory(self):
        model = make_model(gru=True)
        model.eval()

        def episode(episode_id, scale):
            pixels = torch.randn(2, 3, 3, 3, 8, 8) + scale
            info = {
                'pixels': pixels,
                'proprio': torch.randn(2, 3, 3, 3),
                'action': torch.zeros(2, 3, 3, 4),
                'id': torch.full((2, 3), episode_id),
                'step_idx': torch.zeros(2, 3),
                'action_history': torch.randn(2, 3, 2, 4),
            }
            return info

        actions_a = torch.randn(2, 3, 2, 4)
        actions_b = torch.randn(2, 3, 2, 4)
        info = episode(7, 0.0)
        model.rollout(info, actions_a)
        hidden = model._mem_cached.clone()
        model.rollout(info, actions_b)
        self.assertTrue(torch.equal(hidden, model._mem_cached))
        recomputed, recomputed_h = model._context_memory(info['emb'][:, 0])
        self.assertTrue(torch.allclose(recomputed_h, model._mem_cached))
        self.assertEqual(recomputed.shape[1], 3)
        other = episode(9, 3.0)
        model.rollout(other, actions_a)
        self.assertEqual(model._mem_sig[0][0], 9)
        self.assertFalse(torch.allclose(hidden, model._mem_cached))
        fresh, fresh_h = model._context_memory(other['emb'][:, 0])
        self.assertTrue(torch.allclose(fresh_h, model._mem_cached))
        self.assertEqual(fresh.shape[1], 3)

    def test_live_block_matches_full_unroll(self):
        from types import SimpleNamespace

        from stable_worldmodel.buffer import HistoryBuffer

        model = make_model(gru=True)
        model.eval()
        self.assertIsNone(model.plan_history_len)
        action_block = 2
        raw = model.extra_encoders['action'].in_chans // action_block
        buffer = HistoryBuffer(
            n_envs=1,
            max_len=20,
            action_block=action_block,
            block_keys=('action',),
        )
        policy = SimpleNamespace(
            cfg=SimpleNamespace(action_block=action_block),
            _history_buffer=buffer,
        )
        seen_pixels = []
        seen_proprio = []
        seen_actions = []
        for step in range(5):
            pixels = torch.randn(1, 1, 3, 8, 8)
            proprio = torch.randn(1, 1, 3)
            action = torch.randn(1, raw)
            buffer.append(
                {'pixels': pixels, 'proprio': proprio, 'action': action}
            )
            before = model._live.get(7)
            model.observe_env(
                policy,
                {
                    'pixels': pixels,
                    'proprio': proprio,
                    'action': action,
                    'id': torch.tensor([7]),
                    'step_idx': torch.tensor([step]),
                },
            )
            seen_actions.append(action.clone())
            if step % action_block == 0:
                seen_pixels.append(pixels.clone())
                seen_proprio.append(proprio.clone())
                self.assertEqual(model._live[7]['step'], step)
            else:
                self.assertEqual(model._live[7]['step'], before['step'])

        visual = model._encode_image(torch.cat(seen_pixels, dim=1))
        proprio = model.extra_encoders['proprio'](
            torch.cat(seen_proprio, dim=1).float()
        )
        action_encoder = model.extra_encoders['action']
        prev = torch.zeros(1, len(seen_pixels), action_encoder.emb_dim)
        for frame, step in enumerate(range(0, 5, action_block)):
            if step == 0:
                continue
            block = torch.cat(
                [
                    seen_actions[t].reshape(-1)
                    for t in range(step - action_block + 1, step + 1)
                ]
            )
            prev[:, frame] = action_encoder(block.view(1, 1, -1).float())[:, 0]
        tokens, hidden = model.memory(
            visual, proprio, prev, previous_actions=prev
        )
        self.assertNotIn('tokens', model._live[7])
        self.assertTrue(torch.allclose(hidden, model._live[7]['hidden']))
        projected = model.memory.to_token(hidden[-1])
        self.assertTrue(torch.allclose(projected, tokens[:, -1]))

        saved = model._live[7]['hidden'].clone()
        info = {
            'pixels': torch.randn(1, 1, 3, 3, 8, 8),
            'proprio': torch.randn(1, 1, 3, 3),
            'action': torch.zeros(1, 1, 3, 4),
            'id': torch.tensor([[7]]),
            'step_idx': torch.tensor([[4]]),
            'action_history': torch.randn(1, 1, 2, 4),
        }
        model.rollout(info, torch.randn(1, 1, 2, 4))
        self.assertTrue(torch.allclose(saved, model._mem_cached))
        replay, replay_h = model._context_memory(info['emb'][:, 0])
        self.assertFalse(torch.allclose(replay_h, model._mem_cached))
        self.assertEqual(replay.shape[1], 3)
        model.rollout(info, torch.randn(1, 1, 2, 4))
        self.assertTrue(torch.equal(saved, model._live[7]['hidden']))

    def test_checkpoint_and_s_r_reuse(self):
        model = make_model(gru=True, pooling='attention', num_queries=2)
        copy = make_model(gru=True, pooling='attention', num_queries=2, seed=1)
        copy.load_state_dict(model.state_dict())
        self.assertTrue(
            any(name.startswith('memory.gru.') for name in model.state_dict())
        )
        left = model.memory.to_token.weight.detach().clone()
        self.assertTrue(torch.equal(left, copy.memory.to_token.weight))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / 'config.yaml'
            OmegaConf.save(
                OmegaConf.create({'perturb': {'s_r': 1.25}}), cfg_path
            )
            self.assertEqual(stored_s_r(cfg_path), 1.25)
            self.assertIsNone(stored_s_r(root / 'missing.yaml'))
            train = _load_train()
            ckpt = root / 'dinogru_weights.ckpt'
            ckpt.write_bytes(b'0')
            cfg = _cfg(perturb=True, s_z=0.5, s_r=None)
            cfg.perturb.s_r = None
            resolved = train._resolve_s_r(
                None, None, cfg, torch.device('cpu'), ckpt, root
            )
            self.assertEqual(resolved, 1.25)
        with self.assertRaises(ValueError):
            require_s_z(None, True)
        with self.assertRaises(ValueError):
            require_s_z(0.0, True)
        self.assertIsNone(require_s_z(None, False))
        self.assertAlmostEqual(latent_rms_from_sums(9 + 16, 2), math.sqrt(12.5))

    def test_config_variants(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_dir = str((ROOT / 'scripts' / 'train' / 'config').resolve())
        cases = {
            'A': ['gru.enabled=false', 'perturb.enabled=false'],
            'B': ['gru.enabled=false', 'perturb.enabled=true'],
            'C': [
                'gru.enabled=true',
                'perturb.enabled=false',
                'gru.pooling=meanpool',
            ],
            'D': [],
            'E': ['gru.pooling=attention', 'gru.num_queries=2'],
        }
        expected = {
            'A': (False, False, 'meanpool'),
            'B': (False, True, 'meanpool'),
            'C': (True, False, 'meanpool'),
            'D': (True, True, 'meanpool'),
            'E': (True, True, 'attention'),
        }
        for name, overrides in cases.items():
            GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name='dinogru', overrides=overrides)
            gru, perturb, pooling = expected[name]
            self.assertEqual(bool(cfg.gru.enabled), gru, name)
            self.assertEqual(bool(cfg.perturb.enabled), perturb, name)
            self.assertEqual(str(cfg.gru.pooling), pooling, name)
            self.assertEqual(int(cfg.seq_len), 16, name)
            self.assertEqual(int(cfg.data.dataset.num_steps), 16, name)
            self.assertEqual(int(cfg.wm.history_size), 3, name)
            self.assertEqual(int(cfg.gru.hidden_dim), 256, name)
            self.assertEqual(float(cfg.perturb.alpha), 0.2, name)
            self.assertEqual(int(cfg.perturb.rank), 8, name)
            self.assertEqual(float(cfg.perturb.zero_fraction), 0.25, name)
            self.assertEqual(int(cfg.perturb.calibration.num_samples), 1024)
        self.assertEqual(int(cfg.gru.num_queries), 2)


if __name__ == '__main__':
    unittest.main()
