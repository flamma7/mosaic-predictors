"""Queue protocol for remote extra validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'train'))

from validation_queue import (  # noqa: E402
    StaleValidationCsv,
    ValidationQueue,
    claim_job,
    complete_job,
    dumps,
    enqueue_job,
    extra_validation_log,
    loads,
    logged_global_step,
    queue_this_epoch,
)


def _job(
    checkpoint,
    wandb_id='run-1',
    model='dinogru',
    started='false',
    machine='',
    completed='false',
    global_step='40',
):
    return {
        'model_name': model,
        'checkpoint_name': checkpoint,
        'wandb_id': wandb_id,
        'global_step': global_step,
        'started': started,
        'machine_job_id': machine,
        'completed': completed,
    }


class MemoryQueue(ValidationQueue):
    def __init__(self, rows=None):
        super().__init__('mem/repo', api=object())
        self.rows = [dict(row) for row in (rows or [])]
        self.revision_n = 0
        self.sha = f'{self.revision_n:040x}'
        self.reads = 0
        self.commits = 0
        self.fail_once = False
        self.intruder = None

    def read(self):
        self.reads += 1
        return [dict(row) for row in self.rows], self.sha

    def _advance(self, rows):
        self.revision_n += 1
        self.sha = f'{self.revision_n:040x}'
        self.rows = [dict(row) for row in rows]

    def commit(self, rows, parent, message):
        if self.fail_once:
            self.fail_once = False
            if self.intruder is not None:
                self._advance(self.intruder)
            raise StaleValidationCsv('parent commit does not match')
        if parent != self.sha:
            raise StaleValidationCsv('parent commit does not match')
        self.commits += 1
        self._advance(rows)


class ValidationQueueTests(unittest.TestCase):
    def test_round_trip_normalizes_bools(self):
        text = dumps(
            [
                _job(
                    'weights_epoch_1.pt',
                    started='TRUE',
                    machine='val-0',
                    completed='no',
                )
            ]
        )
        self.assertTrue(text.startswith('model_name,checkpoint_name,wandb_id,'))
        rows = loads(text)
        self.assertEqual(rows[0]['started'], 'true')
        self.assertEqual(rows[0]['completed'], 'false')
        self.assertEqual(rows[0]['machine_job_id'], 'val-0')
        self.assertEqual(loads(dumps(rows)), rows)

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            loads('model_name,checkpoint_name\n dinogru,weights_epoch_1.pt\n')

    def test_enqueue_is_idempotent_and_does_not_edit_the_input(self):
        original = [_job('weights_epoch_1.pt')]
        snapshot = [dict(row) for row in original]
        rows, added = enqueue_job(
            original, 'dinogru', 'weights_epoch_1.pt', 'run-1', 40
        )
        self.assertFalse(added)
        self.assertEqual(original, snapshot)
        rows, added = enqueue_job(
            rows, 'dinogru', 'weights_epoch_2.pt', 'run-1', 80
        )
        self.assertTrue(added)
        self.assertEqual(
            [row['checkpoint_name'] for row in rows],
            ['weights_epoch_1.pt', 'weights_epoch_2.pt'],
        )
        self.assertEqual(rows[1]['started'], 'false')
        self.assertEqual(rows[1]['machine_job_id'], '')
        self.assertEqual(rows[1]['global_step'], '80')
        self.assertEqual(rows[0]['global_step'], '40')

    def test_claim_takes_the_first_free_row_only(self):
        original = [
            _job('weights_epoch_1.pt'),
            _job('weights_epoch_2.pt', started='true', machine='other'),
        ]
        snapshot = [dict(row) for row in original]
        rows, job, dirty = claim_job(original, 'val-0')
        self.assertTrue(dirty)
        self.assertEqual(original, snapshot)
        self.assertEqual(job['checkpoint_name'], 'weights_epoch_1.pt')
        self.assertEqual(job['machine_job_id'], 'val-0')
        self.assertEqual(job['global_step'], '40')
        self.assertEqual(rows[1]['machine_job_id'], 'other')
        self.assertEqual(rows[1]['started'], 'true')

    def test_claim_resumes_our_unfinished_row_without_a_write(self):
        rows = [
            _job(
                'weights_epoch_3.pt',
                started='true',
                machine='val-0',
            )
        ]
        _, job, dirty = claim_job(rows, 'val-0')
        self.assertFalse(dirty)
        self.assertEqual(job['checkpoint_name'], 'weights_epoch_3.pt')
        _, job, dirty = claim_job(
            [_job('weights_epoch_3.pt', started='true', machine='val-0', completed='true')],
            'val-0',
        )
        self.assertIsNone(job)
        self.assertFalse(dirty)

    def test_complete_keeps_other_rows_and_refuses_a_foreign_owner(self):
        mine = _job('weights_epoch_1.pt', started='true', machine='val-0')
        other = _job(
            'weights_epoch_2.pt', started='true', machine='val-1'
        )
        rows, dirty = complete_job([mine, other], mine, 'val-0')
        self.assertTrue(dirty)
        self.assertEqual(rows[0]['completed'], 'true')
        self.assertEqual(rows[1]['machine_job_id'], 'val-1')
        self.assertEqual(rows[1]['completed'], 'false')
        with self.assertRaises(RuntimeError):
            complete_job(rows, other, 'val-0')

    def test_two_servers_claim_different_jobs(self):
        queue = MemoryQueue(
            [_job('weights_epoch_1.pt'), _job('weights_epoch_2.pt')]
        )
        first = queue.claim('val-0')
        second = queue.claim('val-1')
        self.assertEqual(first['checkpoint_name'], 'weights_epoch_1.pt')
        self.assertEqual(second['checkpoint_name'], 'weights_epoch_2.pt')
        self.assertEqual(queue.rows[0]['machine_job_id'], 'val-0')
        self.assertEqual(queue.rows[1]['machine_job_id'], 'val-1')
        self.assertIsNone(queue.claim('val-2'))

    def test_stale_claim_rereads_before_writing(self):
        queue = MemoryQueue(
            [_job('weights_epoch_1.pt'), _job('weights_epoch_2.pt')]
        )
        queue.fail_once = True
        queue.intruder = [
            _job(
                'weights_epoch_1.pt',
                started='true',
                machine='val-1',
            ),
            _job('weights_epoch_2.pt'),
        ]
        claimed = queue.claim('val-0')
        self.assertGreaterEqual(queue.reads, 2)
        self.assertEqual(claimed['checkpoint_name'], 'weights_epoch_2.pt')
        self.assertEqual(queue.rows[0]['machine_job_id'], 'val-1')
        self.assertEqual(queue.rows[1]['machine_job_id'], 'val-0')
        self.assertEqual(queue.rows[1]['started'], 'true')

    def test_complete_rereads_and_does_not_write_on_conflict(self):
        mine = _job('weights_epoch_1.pt', started='true', machine='val-0')
        queue = MemoryQueue([mine])
        queue.complete(mine, 'val-0')
        self.assertEqual(queue.rows[0]['completed'], 'true')
        self.assertGreaterEqual(queue.reads, 1)
        foreign = _job('weights_epoch_1.pt', started='true', machine='val-1')
        queue.rows = [foreign]
        with self.assertRaises(RuntimeError):
            queue.complete(foreign, 'val-0')
        self.assertEqual(queue.rows[0]['machine_job_id'], 'val-1')
        self.assertEqual(queue.rows[0]['completed'], 'false')

    def test_epoch_end_step_matches_the_validation_record(self):
        self.assertTrue(queue_this_epoch(1, 1))
        self.assertFalse(queue_this_epoch(4, 5))
        self.assertTrue(queue_this_epoch(5, 5))
        # Counter has already moved past the step used for val/*.
        self.assertEqual(logged_global_step(1), 0)
        self.assertEqual(logged_global_step(1000), 999)
        with self.assertRaises(ValueError):
            logged_global_step(0)
        payload = extra_validation_log(
            {'validate_mse_5/none': 0.25, 'validate_mpc/none': 80.0},
            999,
            3,
        )
        self.assertEqual(payload['trainer/global_step'], 999)
        self.assertEqual(payload['epoch'], 3)
        self.assertEqual(payload['validate_mse_5/none'], 0.25)
        self.assertNotIn('step', payload)

    def test_configs_point_extra_validation_at_the_server(self):
        for name in ('dinogru.yaml', 'dinowm.yaml'):
            text = (ROOT / 'scripts' / 'train' / 'config' / name).read_text()
            self.assertIn('on_validation_server: false', text)
        launcher = (
            ROOT / 'scripts' / 'train' / 'config' / 'launcher' / 'local.yaml'
        ).read_text()
        self.assertIn('mode: shared', launcher)

    def test_remote_training_requires_and_forces_shared_wandb(self):
        from omegaconf import OmegaConf

        from dinowm import wandb_logger_config
        from validate_extra import extra_runs_locally, require_remote_extra

        cfg = OmegaConf.create(
            {
                'wandb': {
                    'enabled': True,
                    'config': {
                        'id': 'job-1',
                        'entity': 'lab',
                        'project': 'jepa',
                        'settings': {'mode': 'online'},
                    },
                },
                'hf': {'enabled': True, 'repo_id': 'lab/models'},
                'validate_extra': {
                    'enabled': True,
                    'on_validation_server': True,
                },
            }
        )
        self.assertFalse(extra_runs_locally(cfg))
        with self.assertRaises(RuntimeError):
            require_remote_extra(cfg)
        cfg.wandb.config.settings.mode = 'shared'
        require_remote_extra(cfg)
        settings = wandb_logger_config(cfg)['settings']
        self.assertEqual(settings['mode'], 'shared')
        self.assertTrue(settings['x_primary'])
        self.assertTrue(settings['x_update_finish_state'])
        self.assertEqual(settings['x_label'], 'train')


if __name__ == '__main__':
    unittest.main()
