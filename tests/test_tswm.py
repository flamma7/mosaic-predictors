"""Encoder pooling, stop-grad, and LeWM rollout shape for TSWM."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from transformer_predictors.tswm import TSWM, temporal_straightening_losses
from stable_worldmodel.wm.lewm.module import Embedder, Predictor
from stable_worldmodel.wm.loss import TemporalStraighteningLoss


class _Logits:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _Hidden:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class TinyResNet(nn.Module):
    def __init__(self, dim: int, pixels: int = 3 * 8 * 8) -> None:
        super().__init__()
        self.lin = nn.Linear(pixels, dim)

    def forward(self, pixels):
        flat = pixels.reshape(pixels.shape[0], -1).to(self.lin.weight.dtype)
        return _Logits(self.lin(flat))


class TinyViT(nn.Module):
    def __init__(self, dim: int, pixels: int = 3 * 8 * 8) -> None:
        super().__init__()
        self.lin = nn.Linear(pixels, dim)

    def forward(self, pixels, interpolate_pos_encoding=False):
        if not interpolate_pos_encoding:
            raise AssertionError('ViT encode should request pos interpolation')
        flat = pixels.reshape(pixels.shape[0], -1).to(self.lin.weight.dtype)
        tokens = self.lin(flat).unsqueeze(1)
        pad = torch.zeros(
            tokens.shape[0], 2, tokens.shape[-1], device=tokens.device
        )
        return _Hidden(torch.cat([tokens, pad], dim=1))


def toy_world(encoder, dim=16, action_dim=4) -> TSWM:
    predictor = Predictor(
        num_frames=3,
        depth=2,
        heads=4,
        mlp_dim=32,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    model = TSWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=Embedder(
            input_dim=action_dim, smoothed_dim=action_dim, emb_dim=dim
        ),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        stop_grad=True,
        interpolate_pos_encoding=isinstance(encoder, TinyViT),
    )
    model.eval()
    return model


class EncodeTest(unittest.TestCase):
    def test_resnet_logits_and_predict_shapes(self):
        model = toy_world(TinyResNet(16))
        pixels = torch.randn(2, 4, 3, 8, 8)
        action = torch.randn(2, 4, 4)
        info = model.encode({'pixels': pixels, 'action': action})
        self.assertEqual(tuple(info['emb'].shape), (2, 4, 16))
        self.assertEqual(tuple(info['act_emb'].shape), (2, 4, 16))
        pred = model.predict(info['emb'][:, :3], info['act_emb'][:, :3])
        self.assertEqual(tuple(pred.shape), (2, 3, 16))

    def test_vit_uses_cls_token(self):
        model = toy_world(TinyViT(16))
        pixels = torch.randn(2, 4, 3, 8, 8)
        info = model.encode({'pixels': pixels, 'action': torch.randn(2, 4, 4)})
        flat = pixels.reshape(8, -1)
        expected = model.projector(model.encoder.lin(flat)).reshape(2, 4, 16)
        self.assertTrue(torch.allclose(info['emb'], expected))

    def test_rollout_shape(self):
        model = toy_world(TinyResNet(16))
        info = {
            'pixels': torch.randn(2, 3, 3, 3, 8, 8),
            'action_history': torch.randn(2, 3, 2, 4),
        }
        out = model.rollout(info, torch.randn(2, 3, 2, 4))
        # H context frames plus T candidate steps.
        self.assertEqual(tuple(out['predicted_emb'].shape), (2, 3, 5, 16))


class LossTest(unittest.TestCase):
    def test_stop_grad_blocks_target_branch(self):
        curvature = TemporalStraighteningLoss()
        emb = torch.randn(4, 4, 8, requires_grad=True)
        pred = torch.randn(4, 3, 8, requires_grad=True)
        stopped = temporal_straightening_losses(
            emb,
            pred,
            emb[:, 1:],
            stop_grad=True,
            curvature=curvature,
            curvature_weight=0.0,
        )
        stopped['pred_loss'].backward()
        self.assertIsNone(emb.grad)

        live = emb.detach().clone().requires_grad_(True)
        open_loss = temporal_straightening_losses(
            live,
            pred.detach(),
            live[:, 1:],
            stop_grad=False,
            curvature=curvature,
            curvature_weight=0.0,
        )
        open_loss['pred_loss'].backward()
        self.assertGreater(live.grad[:, 1:].abs().sum().item(), 0.0)
        self.assertEqual(live.grad[:, :1].abs().sum().item(), 0.0)

    def test_curvature_trains_the_embedding(self):
        emb = torch.randn(4, 4, 8, requires_grad=True)
        pred = torch.randn(4, 3, 8)
        out = temporal_straightening_losses(
            emb,
            pred,
            emb[:, 1:],
            stop_grad=True,
            curvature=TemporalStraighteningLoss(),
            curvature_weight=0.1,
        )
        out['loss'].backward()
        self.assertGreater(emb.grad.abs().sum().item(), 0.0)

    def test_short_clip_has_zero_curvature(self):
        emb = torch.randn(2, 2, 8, requires_grad=True)
        pred = torch.randn(2, 2, 8, requires_grad=True)
        out = temporal_straightening_losses(
            emb,
            pred,
            emb,
            stop_grad=True,
            curvature=TemporalStraighteningLoss(),
            curvature_weight=1.0,
        )
        self.assertEqual(out['curvature_loss'].item(), 0.0)
        out['loss'].backward()
        self.assertIsNone(emb.grad)


if __name__ == '__main__':
    unittest.main()
