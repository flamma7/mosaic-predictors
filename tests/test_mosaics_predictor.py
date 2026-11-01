"""Architecture, mathematics, causality, cache, and planning checks for MoPWM.

Run from the repo root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch
from torch import nn

from mosaic_predictors.models.mosaics.bandwidth import AdaptiveBandwidth
from mosaic_predictors.models.mosaics.gated_keys import GatedKeyExtractor
from mosaic_predictors.models.mosaics.memory import AssociativeMemory
from mosaic_predictors.models.mosaics.predictor import MosaicsPredictor
from mosaic_predictors.models.mosaics.values import NextStateValues
from mosaic_predictors.models.mopwm import MoPWM


CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / 'src'
    / 'mosaic_predictors'
    / 'models'
    / 'config'
)


def activate(predictor: MosaicsPredictor, std: float = 0.05) -> None:
    """Give actions a real path into the key AdaLN and the SwiGLU AdaLN.

    At construction both maps are zero. Keys then see the plain key norm,
    and the SwiGLU residual gate is closed.
    """
    generator = torch.Generator().manual_seed(0)
    for block in predictor.blocks:
        for linear in (block.adaLN_modulation[-1], block.swiglu_modulation[-1]):
            with torch.no_grad():
                linear.weight.normal_(0, std, generator=generator)
                linear.bias.zero_()


def small_predictor(**overrides) -> MosaicsPredictor:
    kwargs = dict(
        num_frames=3,
        depth=2,
        heads=4,
        mlp_dim=64,
        input_dim=32,
        hidden_dim=32,
        output_dim=32,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    kwargs.update(overrides)
    predictor = MosaicsPredictor(**kwargs)
    activate(predictor)
    predictor.eval()
    return predictor


class EncOut:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class TinyEncoder(nn.Module):
    def __init__(self, dim: int, pixels: int = 3 * 8 * 8) -> None:
        super().__init__()
        self.lin = nn.Linear(pixels, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        flat = pixels.reshape(pixels.shape[0], -1).to(self.lin.weight.dtype)
        return EncOut(self.lin(flat).unsqueeze(1))


class TinyActionEncoder(nn.Module):
    def __init__(self, action_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(action_dim, emb_dim)
        self.inputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return self.lin(x)


def toy_world(dim=16, action_dim=3, depth=2) -> MoPWM:
    predictor = MosaicsPredictor(
        num_frames=3,
        depth=depth,
        heads=4,
        mlp_dim=32,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    activate(predictor)
    model = MoPWM(
        encoder=TinyEncoder(dim),
        predictor=predictor,
        action_encoder=TinyActionEncoder(action_dim, dim),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )
    model.eval()
    return model


class ArchitectureTest(unittest.TestCase):
    def test_full_width_predictor(self):
        predictor = MosaicsPredictor(
            num_frames=3,
            depth=6,
            heads=16,
            mlp_dim=2048,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
            gated_keys=True,
            action_conditioning='adaln',
            value_conditioning='layernorm',
            value_source='next_state',
            value_interpolation=False,
            memory_type='full_window',
            adaptive_bandwidth=True,
        )
        self.assertEqual(len(predictor.blocks), 6)
        self.assertFalse(hasattr(predictor, 'pos_embedding'))
        self.assertIsInstance(predictor.input_proj, nn.Identity)
        self.assertEqual(predictor.dropout.p, 0.0)
        for block in predictor.blocks:
            self.assertEqual(block.heads, 16)
            self.assertEqual(block.dim_head, 64)
            self.assertEqual(block.dim, 192)
            self.assertEqual(block.gated_keys.proj_phi.out_features, 16 * 64)
            self.assertEqual(block.gated_keys.proj_gate.out_features, 16)
            self.assertEqual(block.gated_keys.proj_decay.out_features, 16)
            self.assertEqual(block.memory.proj_out.in_features, 16 * 64)
            self.assertEqual(block.memory.proj_out.out_features, 192)
            self.assertEqual(block.swiglu.hidden_dim, 2048)
            self.assertEqual(block.swiglu.w_out.out_features, 192)
            self.assertEqual(block.resid_drop.p, 0.1)
            self.assertFalse(block.key_norm.elementwise_affine)
            self.assertFalse(block.swiglu_norm.elementwise_affine)
            self.assertTrue(block.values.norm.elementwise_affine)
            self.assertEqual(
                tuple(block.adaLN_modulation[-1].weight.shape), (2 * 192, 192)
            )
            self.assertEqual(block.adaLN_modulation[-1].weight.abs().sum().item(), 0.0)
            self.assertEqual(block.adaLN_modulation[-1].bias.abs().sum().item(), 0.0)
            self.assertEqual(
                tuple(block.swiglu_modulation[-1].weight.shape), (3 * 192, 192)
            )
            self.assertEqual(block.swiglu_modulation[-1].weight.abs().sum().item(), 0.0)
            self.assertEqual(block.swiglu_modulation[-1].bias.abs().sum().item(), 0.0)
        # Independent value projections per block.
        first = predictor.blocks[0].values.proj.weight.data_ptr()
        second = predictor.blocks[1].values.proj.weight.data_ptr()
        self.assertNotEqual(first, second)
        joined = ' '.join(predictor.state_dict())
        self.assertNotIn('pos_embed', joined)
        self.assertNotIn('interp', joined)

        x = torch.randn(2, 4, 192)
        y = predictor(x, torch.randn(2, 4, 192))
        self.assertEqual(tuple(y.shape), (2, 4, 192))
        y.sum().backward()
        # The residual gate is zero, so SwiGLU weights are disconnected.
        # The gate rows of the modulation are the parameters that receive
        # gradient and can open the branch.
        block0 = predictor.blocks[0]
        self.assertEqual(block0.swiglu.w_out.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(block0.swiglu_modulation[-1].weight.grad.abs().sum(), 0)

    def test_rejects_other_memory_designs(self):
        with self.assertRaises(ValueError):
            MosaicsPredictor(
                num_frames=3,
                depth=1,
                heads=2,
                mlp_dim=8,
                input_dim=4,
                hidden_dim=4,
                dim_head=2,
                value_interpolation=True,
            )

    def test_any_window_length(self):
        predictor = small_predictor()
        for length in (1, 2, 3, 5, 7):
            y = predictor(torch.randn(2, length, 32), torch.randn(2, length, 32))
            self.assertEqual(tuple(y.shape), (2, length, 32))


class GatedKeyTest(unittest.TestCase):
    def reference(self, module, u, state=None):
        """Independent transcription of the Section 5 recurrence."""
        batch, length, _ = u.shape
        heads, dim_head = module.heads, module.dim_head
        weight_phi = module.proj_phi.weight.detach().float()
        weight_gate = module.proj_gate.weight.detach().float()
        weight_decay = module.proj_decay.weight.detach().float()
        if state is None:
            acc = torch.zeros(batch, heads, dim_head)
        else:
            acc = state.detach().float().clone()
        keys = []
        for t in range(length):
            step = u[:, t].float()
            k_tilde = (step @ weight_phi.T).view(batch, heads, dim_head)
            gate = torch.exp((step @ weight_gate.T).clamp(max=15)).unsqueeze(-1)
            decay = torch.exp(-(step @ weight_decay.T).abs()).unsqueeze(-1)
            acc = gate * k_tilde + decay * acc
            keys.append(acc / (acc.norm(dim=-1, keepdim=True) + 1e-6))
        return torch.stack(keys, dim=1), acc

    def test_matches_explicit_recurrence(self):
        torch.manual_seed(0)
        module = GatedKeyExtractor(dim=12, heads=3, dim_head=5)
        u = torch.randn(4, 6, 12)
        keys, state = module(u)
        ref_keys, ref_state = self.reference(module, u)
        torch.testing.assert_close(keys.float(), ref_keys)
        torch.testing.assert_close(state, ref_state)
        # Starts from k_bar_{-1} = 0, so the first key ignores decay.
        step0 = u[:, 0].float()
        k0 = (step0 @ module.proj_phi.weight.detach().float().T).view(4, 3, 5)
        g0 = torch.exp(
            (step0 @ module.proj_gate.weight.detach().float().T).clamp(max=15)
        ).unsqueeze(-1)
        expected0 = g0 * k0
        expected0 = expected0 / (expected0.norm(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(keys[:, 0].float(), expected0)

    def test_gate_positive_and_decay_in_unit_interval(self):
        module = GatedKeyExtractor(dim=8, heads=2, dim_head=4)
        u = torch.randn(2, 5, 8) * 3
        gate = torch.exp(module.proj_gate(u).clamp(max=15))
        decay = torch.exp(-module.proj_decay(u).abs())
        self.assertTrue(torch.all(gate > 0))
        self.assertTrue(torch.all(decay > 0))
        self.assertTrue(torch.all(decay <= 1))

    def test_sequence_matches_steps_and_batch_rows(self):
        torch.manual_seed(1)
        module = GatedKeyExtractor(dim=10, heads=2, dim_head=4)
        u = torch.randn(3, 5, 10)
        full, full_state = module(u)
        state = None
        pieces = []
        for t in range(u.shape[1]):
            piece, state = module(u[:, t : t + 1], state)
            pieces.append(piece)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full)
        torch.testing.assert_close(state, full_state)

        left, _ = module(u[:1])
        right, _ = module(u[1:2])
        torch.testing.assert_close(full[:1], left)
        torch.testing.assert_close(full[1:2], right)
        # A different second row does not change the first row's keys.
        u2 = u.clone()
        u2[1:].normal_()
        mixed, _ = module(u2)
        torch.testing.assert_close(mixed[:1], left)
        self.assertFalse(torch.allclose(mixed[1:], full[1:]))


class ValueTest(unittest.TestCase):
    def test_positionwise_layernorm_only(self):
        values = NextStateValues(dim=16, heads=4, dim_head=8)
        self.assertIsInstance(values.norm, nn.LayerNorm)
        self.assertFalse(any('ada' in name.lower() for name, _ in values.named_modules()))
        names = [name for name, _ in values.named_parameters()]
        self.assertFalse(any('interp' in name or 'gamma' in name for name in names))

        z = torch.randn(2, 4, 16, requires_grad=True)
        psi = values(z)
        self.assertEqual(tuple(psi.shape), (2, 4, 4, 8))
        # Unit norm per head.
        torch.testing.assert_close(
            psi.norm(dim=-1), torch.ones(2, 4, 4), atol=1e-5, rtol=1e-5
        )
        psi[:, 2].sum().backward()
        self.assertEqual(z.grad[:, :2].abs().sum().item(), 0.0)
        self.assertEqual(z.grad[:, 3:].abs().sum().item(), 0.0)
        self.assertGreater(z.grad[:, 2].abs().sum().item(), 0.0)


class BandwidthTest(unittest.TestCase):
    def test_schedule_and_constraints(self):
        bandwidth = AdaptiveBandwidth(4)
        b0, b1, alpha = bandwidth.coefficients()
        self.assertTrue(torch.all(b0 > 0))
        self.assertTrue(torch.all(b1 > 0))
        self.assertTrue(torch.all((alpha > 0) & (alpha < 1)))
        torch.testing.assert_close(b0, torch.full((4,), math.exp(1.5)))
        torch.testing.assert_close(alpha, torch.full((4,), 0.33333))

        n = torch.tensor([1.0, 2.0, 4.0])
        beta = bandwidth(n).squeeze(-1)
        expected = b1 * torch.pow(n.unsqueeze(-1), alpha) + b0
        torch.testing.assert_close(beta, expected)
        # Sharpens as the number of stored pairs grows.
        self.assertTrue(torch.all(beta[1] > beta[0]))
        self.assertTrue(torch.all(beta[2] > beta[1]))

        with torch.no_grad():
            bandwidth.bandwidth_exponent.fill_(5.0)
        _, _, capped = bandwidth.coefficients()
        torch.testing.assert_close(capped, torch.ones(4))


class RetrievalTest(unittest.TestCase):
    def test_tworoom_counts_and_older_pairs(self):
        torch.manual_seed(0)
        memory = AssociativeMemory(dim=4, heads=1, dim_head=4)
        with torch.no_grad():
            memory.proj_out.weight.copy_(torch.eye(4))
            memory.bandwidth.bandwidth_bias.fill_(math.log(30.0))
            memory.bandwidth.bandwidth_scale.fill_(-8.0)

        k0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        k1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        v0 = torch.tensor([0.0, 0.0, 1.0, 0.0])
        v1 = torch.tensor([0.0, 0.0, 0.0, 1.0])
        bank_k = torch.stack([k0, k1]).view(1, 2, 1, 4)
        bank_v = torch.stack([v0, v1]).view(1, 2, 1, 4).requires_grad_(True)
        # Three queries, n0 = 0: 0 pairs, then 1, then both. Bank length 2.
        query = torch.stack([k0, k1, k0]).view(1, 3, 1, 4)
        out = memory(query, bank_k, bank_v, n0=0)
        torch.testing.assert_close(out[:, 0], torch.zeros(1, 4))
        torch.testing.assert_close(out[:, 1], v0.view(1, 4), atol=1e-5, rtol=1e-5)
        # Final query matches the older key and still retrieves its value.
        torch.testing.assert_close(out[:, 2], v0.view(1, 4), atol=1e-4, rtol=1e-4)

        out[:, 1].sum().backward()
        self.assertGreater(bank_v.grad[:, 0].abs().sum().item(), 0.0)
        self.assertEqual(bank_v.grad[:, 1].abs().sum().item(), 0.0)

        empty = memory(query[:, :1], bank_k[:, :0], bank_v[:, :0], n0=0)
        torch.testing.assert_close(empty, torch.zeros_like(empty))


class PredictorBehaviorTest(unittest.TestCase):
    def test_future_does_not_change_earlier_predictions(self):
        torch.manual_seed(0)
        predictor = small_predictor()
        x = torch.randn(2, 4, 32)
        a = torch.randn(2, 4, 32)
        base = predictor(x, a)
        for t in range(3):
            x2 = x.clone()
            a2 = a.clone()
            x2[:, t + 1 :].normal_()
            a2[:, t + 1 :].normal_()
            changed = predictor(x2, a2)
            torch.testing.assert_close(changed[:, : t + 1], base[:, : t + 1])
            self.assertFalse(torch.allclose(changed[:, t + 1 :], base[:, t + 1 :]))

        # Position 2 is the first query that reads two pairs. A single pair
        # has attention weight 1, so its key, and therefore the action, cannot
        # move that read. Squared error avoids the final LayerNorm sum, which
        # does not depend on the residual when the scale is all ones.
        x_grad = x.clone().requires_grad_(True)
        a_grad = a.clone().requires_grad_(True)
        pred = predictor(x_grad, a_grad)
        pred[:, 2].pow(2).sum().backward()
        self.assertEqual(x_grad.grad[:, 3:].abs().sum().item(), 0.0)
        self.assertEqual(a_grad.grad[:, 3:].abs().sum().item(), 0.0)
        self.assertGreater(x_grad.grad[:, :3].abs().sum().item(), 0.0)
        self.assertGreater(a_grad.grad[:, :3].abs().sum().item(), 0.0)

    def test_swiglu_adaln_conditions_the_first_step(self):
        """An empty memory cannot use the action. SwiGLU AdaLN can."""
        torch.manual_seed(0)
        predictor = MosaicsPredictor(
            num_frames=3,
            depth=2,
            heads=4,
            mlp_dim=64,
            input_dim=32,
            hidden_dim=32,
            output_dim=32,
            dim_head=8,
            dropout=0.0,
            emb_dropout=0.0,
        )
        predictor.eval()
        x = torch.randn(2, 1, 32)
        a1 = torch.randn(2, 1, 32)
        a2 = torch.randn(2, 1, 32)
        # A zero gate closes the branch, so the action cannot move y.
        torch.testing.assert_close(predictor(x, a1), predictor(x, a2))
        with torch.no_grad():
            predictor.blocks[0].swiglu_modulation[-1].weight.normal_(0, 0.05)
        self.assertFalse(torch.allclose(predictor(x, a1), predictor(x, a2)))

    def test_batch_rows_are_independent(self):
        torch.manual_seed(0)
        predictor = small_predictor()
        x = torch.randn(2, 4, 32)
        a = torch.randn(2, 4, 32)
        both = predictor(x, a)
        torch.testing.assert_close(both[:1], predictor(x[:1], a[:1]))
        torch.testing.assert_close(both[1:], predictor(x[1:], a[1:]))

    def test_tworoom_pair_alignment(self):
        """states [z0,z1,z2], actions [a0,a1,a2]: 0, then 1, then 2 pairs.

        Block 0 stores psi(z1), psi(z2). Each later block stores psi of its
        incoming residual at the next index, not psi(z). psi of the current
        index is not the value for that key, and the target z3 is not an input.
        """
        torch.manual_seed(0)
        predictor = small_predictor()
        z = torch.randn(2, 3, 32)
        a = torch.randn(2, 3, 32)
        cache = predictor.init_cache(2, max_len=3, device=z.device, dtype=z.dtype)
        step0, cache = predictor.step(z[:, 0], a[:, 0], cache)
        self.assertEqual(cache.completed, 0)
        self.assertIsNotNone(cache.blocks[0].pending_key)
        self.assertEqual(tuple(step0.shape), (2, 32))

        _, cache = predictor.step(z[:, 1], a[:, 1], cache)
        self.assertEqual(cache.completed, 1)
        _, cache = predictor.step(z[:, 2], a[:, 2], cache)
        self.assertEqual(cache.completed, 2)

        fresh = predictor.init_cache(2, device=z.device, dtype=z.dtype)
        window, fresh = predictor.prefill(z, a, fresh)
        self.assertEqual(fresh.completed, 2)
        self.assertEqual(tuple(window.shape), (2, 3, 32))
        residual = z
        for layer, (block, state) in enumerate(zip(predictor.blocks, fresh.blocks)):
            psi = block.values(residual)
            torch.testing.assert_close(state.values[:, 0], psi[:, 1])
            torch.testing.assert_close(state.values[:, 1], psi[:, 2])
            self.assertFalse(torch.allclose(state.values[:, 0], psi[:, 0]))
            self.assertEqual(state.completed, 2)
            if layer > 0:
                raw = block.values(z)
                self.assertFalse(torch.allclose(state.values, raw[:, 1:]))
            residual, _ = block(residual, a)
        # Prefill predictions match the stepwise queries at every position.
        torch.testing.assert_close(window[:, 0], step0)


class CacheParityTest(unittest.TestCase):
    def test_chunks_match_full_window(self):
        torch.manual_seed(0)
        predictor = small_predictor()
        x = torch.randn(2, 6, 32)
        a = torch.randn(2, 6, 32)
        full = predictor(x, a)

        cache = predictor.init_cache(2, max_len=6, device=x.device, dtype=x.dtype)
        pre, cache = predictor.prefill(x[:, :3], a[:, :3], cache)
        one, cache = predictor.step(x[:, 3], a[:, 3], cache)
        rest, cache = predictor.prefill(x[:, 4:], a[:, 4:], cache)
        got = torch.cat([pre, one.unsqueeze(1), rest], dim=1)
        torch.testing.assert_close(got, full, atol=1e-5, rtol=1e-5)

        # Same split done as single steps.
        cache = predictor.init_cache(2, device=x.device, dtype=x.dtype)
        pieces = []
        for t in range(x.shape[1]):
            piece, cache = predictor.step(x[:, t], a[:, t], cache)
            pieces.append(piece)
        stepped = torch.stack(pieces, dim=1)
        torch.testing.assert_close(stepped, full, atol=1e-5, rtol=1e-5)
        self.assertEqual(cache.completed, x.shape[1] - 1)

    def test_reset_clears_history(self):
        torch.manual_seed(0)
        predictor = small_predictor()
        hist_x = torch.randn(2, 4, 32)
        hist_a = torch.randn(2, 4, 32)
        new_x = torch.randn(2, 32)
        new_a = torch.randn(2, 32)
        cache = predictor.init_cache(2, device=hist_x.device, dtype=hist_x.dtype)
        _, cache = predictor.prefill(hist_x, hist_a, cache)
        self.assertGreater(cache.blocks[0].recurrent.abs().sum().item(), 0.0)
        continued, _ = predictor.step(new_x, new_a, cache)

        predictor.reset_cache(cache)
        self.assertEqual(cache.completed, 0)
        self.assertIsNone(cache.blocks[0].pending_key)
        self.assertEqual(cache.blocks[0].recurrent.abs().sum().item(), 0.0)
        reset_out, _ = predictor.step(new_x, new_a, cache)
        fresh = predictor(new_x.unsqueeze(1), new_a.unsqueeze(1))[:, 0]
        torch.testing.assert_close(reset_out, fresh, atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(continued, reset_out))


class TrainingIntegrationTest(unittest.TestCase):
    def test_encode_predict_and_losses(self):
        torch.manual_seed(0)
        dim = 16
        model = toy_world(dim=dim, action_dim=3, depth=2)
        model.train()
        batch = 8
        pixels = torch.randn(batch, 4, 3, 8, 8)
        action = torch.randn(batch, 4, 3)
        info = model.encode({'pixels': pixels, 'action': action})
        emb = info['emb']
        self.assertEqual(tuple(emb.shape), (batch, 4, dim))
        self.assertEqual(tuple(info['act_emb'].shape), (batch, 4, dim))

        ctx_len, n_preds = 3, 1
        pred = model.predict(emb[:, :ctx_len], info['act_emb'][:, :ctx_len])
        target = emb[:, n_preds:]
        self.assertEqual(tuple(pred.shape), tuple(target.shape))

        from stable_worldmodel.wm.loss import SIGReg

        sigreg = SIGReg(knots=17, num_proj=32)
        pred_loss = (pred - target).pow(2).mean()
        sig_loss = sigreg(emb.transpose(0, 1))
        loss = pred_loss + 0.1 * sig_loss
        loss.backward()
        encoder_grad = sum(p.grad.abs().sum() for p in model.encoder.parameters())
        predictor_grad = sum(
            p.grad.abs().sum()
            for p in model.predictor.parameters()
            if p.grad is not None
        )
        self.assertGreater(encoder_grad, 0)
        self.assertGreater(predictor_grad, 0)
        self.assertGreater(
            model.predictor.blocks[0].adaLN_modulation[-1].weight.grad.abs().sum(),
            0,
        )
        self.assertGreater(
            model.predictor.blocks[0].swiglu_modulation[-1].weight.grad.abs().sum(),
            0,
        )

    def test_overfit_synthetic_window(self):
        torch.manual_seed(0)
        predictor = MosaicsPredictor(
            num_frames=3,
            depth=2,
            heads=4,
            mlp_dim=64,
            input_dim=32,
            hidden_dim=32,
            output_dim=32,
            dim_head=8,
            dropout=0.0,
            emb_dropout=0.0,
        )
        x = torch.randn(16, 4, 32)
        a = torch.randn(16, 4, 32)
        # A fixed next-state map the position-wise path can memorize.
        target = torch.tanh(x[:, 1:] + 0.3 * a[:, :3])
        optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-3)
        predictor.train()

        def loss_of():
            pred = predictor(x[:, :3], a[:, :3])
            return (pred - target).pow(2).mean()

        initial = loss_of().item()
        for _ in range(80):
            optimizer.zero_grad()
            loss_of().backward()
            optimizer.step()
        predictor.eval()
        final = loss_of().item()
        self.assertLess(final, initial * 0.5)


class PlanningTest(unittest.TestCase):
    def _info(self, batch, samples, hist, action_dim, emb=None):
        info = {'pixels': torch.randn(batch, samples, hist, 3, 8, 8)}
        if emb is not None:
            info['emb'] = emb
        return info

    def test_rollout_matches_full_window_predict(self):
        torch.manual_seed(0)
        model = toy_world()
        model.eval()
        batch, samples, hist, horizon, dim, action_dim = 2, 3, 3, 4, 16, 3
        emb = torch.randn(batch, samples, hist, dim)
        past = torch.randn(batch, samples, hist - 1, action_dim)
        candidates = torch.randn(batch, samples, horizon, action_dim)
        info = self._info(batch, samples, hist, action_dim, emb=emb)
        info['action_history'] = past
        out = model.rollout(dict(info), candidates)['predicted_emb']

        flat_emb = emb.reshape(batch * samples, hist, dim)
        flat_act = torch.cat([past, candidates], dim=2).reshape(
            batch * samples, hist - 1 + horizon, action_dim
        )
        all_act = model.action_encoder(flat_act)
        frames = list(flat_emb.unbind(dim=1))
        first = model.predict(torch.stack(frames, dim=1), all_act[:, :hist])[:, -1]
        frames.append(first)
        for t in range(1, horizon):
            window = torch.stack(frames, dim=1)
            nxt = model.predict(window, all_act[:, : hist + t])[:, -1]
            frames.append(nxt)
        expected = torch.stack(frames, dim=1).reshape(batch, samples, hist + horizon, dim)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[:, :, :hist], emb)

    def test_candidates_are_independent_and_repeatable(self):
        torch.manual_seed(0)
        model = toy_world()
        model.eval()
        batch, samples, hist, horizon = 1, 2, 1, 3
        pixels = torch.randn(batch, samples, hist, 3, 8, 8)
        # Same context image for every candidate; actions differ.
        pixels = pixels[:, :1].expand(batch, samples, hist, 3, 8, 8).contiguous()
        candidates = torch.randn(batch, samples, horizon, 3)
        info = {'pixels': pixels}
        out = model.rollout(dict(info), candidates)['predicted_emb']
        self.assertEqual(tuple(out.shape), (batch, samples, hist + horizon, 16))
        self.assertFalse(torch.allclose(out[:, 0, hist + 1 :], out[:, 1, hist + 1 :]))

        for index in range(samples):
            alone_pixels = pixels[:, index : index + 1]
            alone_act = candidates[:, index : index + 1]
            alone = model.rollout(
                {'pixels': alone_pixels.clone()}, alone_act
            )['predicted_emb']
            torch.testing.assert_close(out[:, index : index + 1], alone)

        again = model.rollout(
            {'pixels': pixels.clone()}, candidates.clone()
        )['predicted_emb']
        torch.testing.assert_close(again, out)

    def test_action_history_contract(self):
        torch.manual_seed(0)
        model = toy_world()
        model.eval()
        emb = torch.randn(2, 2, 3, 16)
        past = torch.randn(2, 2, 2, 3)
        candidates = torch.randn(2, 2, 3, 3)
        info = {'pixels': torch.randn(2, 2, 3, 3, 8, 8), 'emb': emb, 'action_history': past}
        out_a = model.rollout(dict(info), candidates)['predicted_emb']
        info_b = dict(info)
        info_b['action_history'] = torch.zeros_like(past)
        out_b = model.rollout(info_b, candidates)['predicted_emb']
        torch.testing.assert_close(out_a[:, :, :3], emb)
        self.assertFalse(torch.allclose(out_a[:, :, 3:], out_b[:, :, 3:]))
        # The last call encoded info_b. The call before it encoded `past`.
        seen = model.action_encoder.inputs[-2]
        expected = torch.cat([past, candidates], dim=2).reshape(4, 5, 3)
        torch.testing.assert_close(seen, expected)

        bad = {'pixels': torch.randn(2, 2, 3, 3, 8, 8), 'emb': emb}
        bad['action_history'] = torch.randn(2, 2, 1, 3)
        with self.assertRaises(AssertionError):
            model.rollout(bad, candidates)

    def test_shooting_cost(self):
        from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
        from stable_worldmodel.protocols import Dynamics

        torch.manual_seed(0)
        model = toy_world()
        model.eval()
        self.assertIsInstance(model, Dynamics)
        batch, samples, hist, horizon = 2, 3, 1, 3
        info = {
            'pixels': torch.randn(batch, samples, hist, 3, 8, 8),
            'goal': torch.randn(batch, samples, 1, 3, 8, 8),
            'action': torch.randn(batch, samples, hist, 3),
        }
        candidates = torch.randn(batch, samples, horizon, 3)
        cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(info, candidates)
        self.assertEqual(tuple(cost.shape), (batch, samples))
        self.assertTrue(torch.isfinite(cost).all())

    def test_candidate_action_grads_do_not_cross(self):
        torch.manual_seed(0)
        model = toy_world()
        dim = 16
        model.pred_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        # BatchNorm1d inside Sequential sees (B*T, D) from _project_pred. Eval
        # uses running stats, so chunked rollout matches a full-window predict.
        model.eval()
        emb = torch.randn(1, 2, 3, dim)
        past = torch.randn(1, 2, 2, 3)
        candidates = torch.randn(1, 2, 3, 3, requires_grad=True)
        info = {
            'pixels': torch.randn(1, 2, 3, 3, 8, 8),
            'emb': emb,
            'action_history': past,
        }
        predicted = model.rollout(info, candidates)['predicted_emb']
        predicted[:, 0].sum().backward()
        self.assertGreater(candidates.grad[:, 0].abs().sum().item(), 0.0)
        self.assertEqual(candidates.grad[:, 1].abs().sum().item(), 0.0)


class HydraTest(unittest.TestCase):
    def test_config_instantiates_mopwm(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(
                config_dir=str(CONFIG_DIR), version_base=None
            ):
                cfg = compose(
                    config_name='mopwm',
                    overrides=['model.action_encoder.input_dim=10'],
                )
            model = instantiate(cfg.model)
        finally:
            GlobalHydra.instance().clear()

        self.assertIsInstance(model, MoPWM)
        predictor = model.predictor
        self.assertIsInstance(predictor, MosaicsPredictor)
        self.assertEqual(predictor.depth, 6)
        self.assertEqual(predictor.heads, 16)
        self.assertEqual(predictor.dim_head, 64)
        self.assertEqual(predictor.mlp_dim, 2048)
        self.assertEqual(predictor.hidden_dim, 192)
        self.assertEqual(predictor.num_frames, 3)
        self.assertEqual(predictor.dropout_p, 0.1)
        self.assertEqual(predictor.emb_dropout, 0.0)


if __name__ == '__main__':
    unittest.main()
