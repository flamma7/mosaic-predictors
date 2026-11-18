"""Overrides applied to a checkpoint's validate_extra config."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'train'))

from validate_server import (  # noqa: E402
    apply_validate_extra_overrides,
    log_extra_budget,
)


def _cfg():
    return OmegaConf.create(
        {
            'seed': 7,
            'wandb': {'enabled': True, 'config': {'id': 'run-1'}},
            'validate_extra': {
                'enabled': True,
                'mse': {
                    'enabled': True,
                    'num_samples': 64,
                    'pred_length': 25,
                },
                'mpc': {
                    'enabled': True,
                    'num_eval': 50,
                    'num_samples': 300,
                    'n_steps': 30,
                    'eval_budget': 50,
                    'seed': '${seed}',
                },
                'perturbations': {
                    'none': {'enabled': True},
                    'blur': {'enabled': True, 'blur': True},
                },
            },
        }
    )


class ValidateExtraOverrideTests(unittest.TestCase):
    def test_empty_overrides_keep_the_loaded_config(self):
        cfg = _cfg()
        self.assertIs(apply_validate_extra_overrides(cfg, ()), cfg)

    def test_relative_and_prefixed_keys_change_only_those_fields(self):
        merged = apply_validate_extra_overrides(
            _cfg(),
            [
                'mse.num_samples=8',
                'validate_extra.mpc.num_eval=4',
                'mpc.num_samples=32',
                'mpc.n_steps=5',
                'perturbations.blur.enabled=false',
            ],
        )
        extra = merged.validate_extra
        self.assertEqual(int(extra.mse.num_samples), 8)
        self.assertEqual(int(extra.mse.pred_length), 25)
        self.assertEqual(int(extra.mpc.num_eval), 4)
        self.assertEqual(int(extra.mpc.num_samples), 32)
        self.assertEqual(int(extra.mpc.n_steps), 5)
        self.assertEqual(int(extra.mpc.eval_budget), 50)
        self.assertEqual(int(extra.mpc.seed), 7)
        self.assertFalse(bool(extra.perturbations.blur.enabled))
        self.assertTrue(bool(extra.perturbations.none.enabled))
        self.assertEqual(str(merged.wandb.config.id), 'run-1')

    def test_null_num_samples_means_every_clip(self):
        merged = apply_validate_extra_overrides(
            _cfg(), ['mse.num_samples=null']
        )
        self.assertIsNone(merged.validate_extra.mse.num_samples)

    def test_later_override_wins(self):
        merged = apply_validate_extra_overrides(
            _cfg(), ['mpc.num_eval=4', 'mpc.num_eval=2']
        )
        self.assertEqual(int(merged.validate_extra.mpc.num_eval), 2)

    def test_unknown_key_raises(self):
        cfg = _cfg()
        for item in ('num_samples=8', 'mse.num_sample=8', 'mpc.num_evals=4'):
            with self.assertRaises(ValueError) as caught:
                apply_validate_extra_overrides(cfg, [item])
            self.assertIn('has no', str(caught.exception))
        self.assertFalse('num_samples' in cfg.validate_extra)
        self.assertEqual(int(cfg.validate_extra.mse.num_samples), 64)

    def test_missing_equals_raises(self):
        with self.assertRaises(ValueError):
            apply_validate_extra_overrides(_cfg(), ['mpc.num_eval'])

    def test_budget_log_names_the_runtime_fields(self):
        merged = apply_validate_extra_overrides(
            _cfg(), ['perturbations.blur.enabled=false']
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            log_extra_budget(merged)
        text = buffer.getvalue()
        self.assertIn('mse num_samples=64', text)
        self.assertIn('mpc num_eval=50', text)
        self.assertIn('num_samples=300', text)
        self.assertIn('n_steps=30', text)
        self.assertIn('perturbations=none', text)
        self.assertNotIn('blur', text)


if __name__ == '__main__':
    unittest.main()
