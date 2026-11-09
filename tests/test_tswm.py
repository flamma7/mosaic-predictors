"""Scratch ResNet, causal ViT predictor, and stop-grad for TSWM."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from stable_worldmodel.wm.loss import TemporalStraighteningLoss
from stable_worldmodel.wm.prejepa.module import CausalPredictor
from transformer_predictors.tswm import (
    ActionEncoder,
    SmallResNetGeM,
    TSWM,
    temporal_straightening_losses,
)


def toy_world(dim=16, action_dim=4, action_emb=4) -> TSWM:
    predictor = CausalPredictor(
        num_patches=1,
        num_frames=3,
        dim=dim + action_emb,
        depth=2,
        heads=4,
        mlp_dim=32,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
        pool='mean',
    )
    model = TSWM(
        encoder=SmallResNetGeM(dim=dim, gem_p=3.0),
        predictor=predictor,
        action_encoder=ActionEncoder(input_dim=action_dim, emb_dim=action_emb),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        stop_grad=True,
    )
    model.eval()
    return model


class EncoderTest(unittest.TestCase):
    def test_matches_scratch_global_resnet(self):
        net = SmallResNetGeM(dim=384, gem_p=3.0)
        blocks = [net.rb1, net.rb2, net.rb3, net.rb4, net.rb5]
        channels = [(3, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
        self.assertEqual(
            [(block.input_dim, block.output_dim) for block in blocks],
            channels,
        )
        self.assertTrue(all(block.resample == 'down' for block in blocks))
        self.assertAlmostEqual(net.gem.p.detach().item(), 3.0)
        projection = list(net.projection)
        self.assertIsInstance(projection[0], nn.Linear)
        self.assertEqual(projection[0].in_features, 512)
        self.assertEqual(projection[0].out_features, 384)
        self.assertIsInstance(projection[1], nn.GELU)
        self.assertIsInstance(projection[2], nn.Linear)
        self.assertEqual(
            (projection[2].in_features, projection[2].out_features), (384, 384)
        )
        self.assertIsInstance(projection[3], nn.LayerNorm)
        image = torch.randn(2, 3, 32, 32)
        self.assertEqual(tuple(net(image).shape), (2, 1, 384))

    def test_predictor_matches_causal_vit(self):
        # temporal-straightening conf/predictor/vit.yaml, global token,
        # plus the 10-d action embedding concatenated on the feature axis.
        predictor = CausalPredictor(
            num_patches=1,
            num_frames=3,
            dim=384 + 10,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
            pool='mean',
        )
        self.assertEqual(tuple(predictor.pos_embedding.shape), (1, 3, 394))
        self.assertEqual(predictor.dropout.p, 0.0)
        self.assertEqual(len(predictor.transformer.layers), 6)
        attention, feedforward = predictor.transformer.layers[0]
        self.assertEqual(attention.heads, 16)
        self.assertEqual(attention.to_qkv.out_features, 16 * 64 * 3)
        self.assertEqual(attention.dropout.p, 0.1)
        self.assertEqual(feedforward.net[1].out_features, 2048)
        self.assertIsInstance(feedforward.net[2], nn.GELU)
        self.assertEqual(feedforward.net[3].p, 0.1)

    def test_encode_predict_and_rollout_shapes(self):
        model = toy_world()
        pixels = torch.randn(2, 4, 3, 32, 32)
        action = torch.randn(2, 4, 4)
        info = model.encode({'pixels': pixels, 'action': action})
        self.assertEqual(tuple(info['emb'].shape), (2, 4, 16))
        self.assertEqual(tuple(info['act_emb'].shape), (2, 4, 4))
        pred = model.predict(info['emb'][:, :3], info['act_emb'][:, :3])
        self.assertEqual(tuple(pred.shape), (2, 3, 16))
        rollout = model.rollout(
            {
                'pixels': torch.randn(2, 3, 3, 3, 32, 32),
                'action_history': torch.randn(2, 3, 2, 4),
            },
            torch.randn(2, 3, 2, 4),
        )
        self.assertEqual(tuple(rollout['predicted_emb'].shape), (2, 3, 5, 16))


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
