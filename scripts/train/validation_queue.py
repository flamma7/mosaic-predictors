"""validation.csv on the Hugging Face model repo.

Each row is one extra-validation job. Training appends a row when it
uploads a checkpoint. A validation server claims the first row with
``started`` false, writes its ``machine_job_id``, and sets ``completed``
after MSE and MPC.

Several servers can poll the same file. Every write re-reads the file and
commits only if the repo is still at that revision, so a stale copy cannot
overwrite another machine's claim.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path

VALIDATION_CSV = 'validation.csv'
COLUMNS = (
    'model_name',
    'checkpoint_name',
    'wandb_id',
    # Lightning's ``trainer/global_step`` for this epoch. W&B charts use
    # this same integer as their x-axis.
    'global_step',
    'started',
    'machine_job_id',
    'completed',
)
_TRUE = {'true', '1', 'yes'}


class StaleValidationCsv(Exception):
    """The repo moved between the read and the commit."""


def queue_this_epoch(epoch, every_n_epochs):
    """Match ExtraValidationCallback: epoch numbers start at 1."""
    every = int(every_n_epochs)
    if every < 1:
        raise ValueError(f'every_n_epochs must be >= 1, got {every}')
    return int(epoch) % every == 0


def logged_global_step(batches_that_stepped):
    """``trainer/global_step`` Lightning logs for this epoch's metrics.

    ``on_train_epoch_end`` runs after the batch loop has incremented
    ``_batches_that_stepped``. Validation already logged at the previous
    value, and Lightning subtracts one again before the epoch-end train
    metrics. That shared value is what ``WandbLogger`` stores as
    ``trainer/global_step``.
    """
    stepped = int(batches_that_stepped)
    if stepped < 1:
        raise ValueError(
            'No logger step to record. _batches_that_stepped is '
            f'{stepped} at epoch end.'
        )
    return stepped - 1


def extra_validation_log(metrics, global_step, epoch):
    """One W&B record in the shape Lightning uses for extra validation.

    ``WandbLogger`` does not pass ``step``. It writes the logger counter
    into ``trainer/global_step``, and the connector adds 0-based ``epoch``.
    Charts use ``trainer/global_step`` as the x-axis.
    """
    step = int(global_step)
    epoch = int(epoch)
    if step < 0 or epoch < 0:
        raise ValueError(
            f'trainer/global_step and epoch must be >= 0, got {step}, {epoch}'
        )
    payload = {}
    for key, value in metrics.items():
        name = str(key)
        if name in ('trainer/global_step', 'epoch'):
            raise ValueError(f'{name} is the log axis, not an extra metric')
        payload[name] = float(value)
    payload['trainer/global_step'] = step
    payload['epoch'] = epoch
    return payload


def mse_horizons(cfg):
    """Horizons from ``validate_extra.mse.horizons``, or none."""
    if cfg is None or not hasattr(cfg, 'get'):
        return []
    extra = cfg.get('validate_extra')
    if extra is None or not hasattr(extra, 'get'):
        return []
    mse = extra.get('mse')
    if mse is None or not hasattr(mse, 'get'):
        return []
    horizons = mse.get('horizons') or []
    return [int(horizon) for horizon in horizons]


def trainer_step_patterns(horizons):
    """W&B globs that must use Lightning's ``trainer/global_step``.

    A ``*`` pattern does not match ``/``. Lightning's default ``*`` therefore
    leaves ``fit/loss``, ``validate/loss``, and ``validate_mse_<h>/<name>``
    on the internal Step axis. ``validate_mpc/*`` was the one pattern that
    already included the slash.
    """
    patterns = ['fit/*', 'validate/*', 'hparams/*', 'validate_mpc/*']
    seen = set(patterns)
    for horizon in horizons:
        pattern = f'validate_mse_{int(horizon)}/*'
        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    return patterns


def section_glob(metric_name):
    """``validate_mse_5/none`` -> ``validate_mse_5/*``.

    The star stays in the last path segment. ``fit/loss`` -> ``fit/*``.
    """
    name = str(metric_name)
    if '/' not in name or name in ('trainer/global_step',):
        return None
    prefix, _slash, _leaf = name.rpartition('/')
    if not prefix or prefix == 'trainer':
        return None
    return f'{prefix}/*'


def define_trainer_step_axis(run, cfg=None, metrics=None, step_sync=False):
    """Plot training and extra-validation metrics against ``trainer/global_step``.

    ``step_sync`` stays off. The step is written in the same record, and a
    validation server logs that record after training has moved past it.
    """
    define = getattr(run, 'define_metric', None)
    if not callable(define):
        return []
    define('trainer/global_step')
    patterns = trainer_step_patterns(mse_horizons(cfg))
    seen = set(patterns)
    for name in metrics or ():
        pattern = section_glob(name)
        if pattern and pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    for pattern in patterns:
        define(
            pattern,
            step_metric='trainer/global_step',
            step_sync=step_sync,
        )
    return patterns


def _is_true(value):
    return str(value).strip().lower() in _TRUE


def dumps(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(COLUMNS), lineterminator='\n'
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: '' if row.get(column) is None else str(row.get(column))
                for column in COLUMNS
            }
        )
    return buffer.getvalue()


def loads(text):
    if text is None or not str(text).strip():
        return []
    reader = csv.DictReader(io.StringIO(text))
    names = list(reader.fieldnames or [])
    missing = [column for column in COLUMNS if column not in names]
    if missing:
        raise ValueError(f'validation.csv is missing columns {missing}')
    rows = []
    for raw in reader:
        if raw is None:
            continue
        if not any(str(raw.get(column) or '').strip() for column in COLUMNS):
            continue
        row = {}
        for column in COLUMNS:
            value = str(raw.get(column) or '').strip()
            if column in ('started', 'completed'):
                value = 'true' if _is_true(value) else 'false'
            row[column] = value
        rows.append(row)
    return rows


def _copy(rows):
    return [dict(row) for row in rows]


def _same_job(row, job):
    return (
        row['model_name'] == job['model_name']
        and row['checkpoint_name'] == job['checkpoint_name']
        and row['wandb_id'] == job['wandb_id']
    )


def enqueue_job(rows, model_name, checkpoint_name, wandb_id, global_step):
    """Append a not-started row. Returns ``(rows, added)``."""
    model_name = str(model_name).strip().strip('/')
    checkpoint_name = str(checkpoint_name).strip()
    wandb_id = str(wandb_id).strip()
    step = int(global_step)
    if not model_name or not checkpoint_name or not wandb_id:
        raise ValueError(
            'model_name, checkpoint_name, and wandb_id are required'
        )
    if step < 0:
        raise ValueError(f'global_step must be >= 0, got {step}')
    for row in rows:
        if (
            row['model_name'] == model_name
            and row['checkpoint_name'] == checkpoint_name
            and row['wandb_id'] == wandb_id
        ):
            return _copy(rows), False
    added = {
        'model_name': model_name,
        'checkpoint_name': checkpoint_name,
        'wandb_id': wandb_id,
        'global_step': str(step),
        'started': 'false',
        'machine_job_id': '',
        'completed': 'false',
    }
    return [*_copy(rows), added], True


def claim_job(rows, machine_job_id):
    """Claim the next free row, or resume one this machine already owns.

    Returns ``(rows, job or None, dirty)``. ``dirty`` is false when the
    file does not need a write (nothing to claim, or the resume row is
    already marked).
    """
    machine_job_id = str(machine_job_id).strip()
    if not machine_job_id:
        raise ValueError('machine_job_id is required')
    rows = _copy(rows)
    for row in rows:
        if (
            row['machine_job_id'] == machine_job_id
            and _is_true(row['started'])
            and not _is_true(row['completed'])
        ):
            return rows, dict(row), False
    for row in rows:
        if _is_true(row['started']) or _is_true(row['completed']):
            continue
        row['started'] = 'true'
        row['machine_job_id'] = machine_job_id
        row['completed'] = 'false'
        return rows, dict(row), True
    return rows, None, False


def complete_job(rows, job, machine_job_id):
    """Mark the claimed row completed. Refuses a row owned by someone else."""
    machine_job_id = str(machine_job_id).strip()
    rows = _copy(rows)
    for row in rows:
        if not _same_job(row, job):
            continue
        owner = row['machine_job_id']
        if owner != machine_job_id:
            raise RuntimeError(
                f'{job["model_name"]}/{job["checkpoint_name"]} is owned by '
                f'{owner or "nobody"}, not {machine_job_id}'
            )
        if _is_true(row['completed']):
            return rows, False
        row['started'] = 'true'
        row['completed'] = 'true'
        row['machine_job_id'] = machine_job_id
        return rows, True
    raise RuntimeError(
        f'{job["model_name"]}/{job["checkpoint_name"]} is no longer in '
        'validation.csv'
    )


def _stale_commit(exc):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    if status in (409, 412):
        return True
    body = ''
    if response is not None:
        try:
            body = response.text or ''
        except Exception:
            body = ''
    blob = f'{exc} {body}'.lower()
    return 'parent' in blob and 'commit' in blob


class ValidationQueue:
    """Read-modify-write ``validation.csv`` against one model repo."""

    def __init__(self, repo_id, api=None, revision='main', max_attempts=8):
        self.repo_id = repo_id
        self.revision = revision
        self.max_attempts = int(max_attempts)
        self._api = api

    @property
    def api(self):
        if self._api is None:
            import os

            from huggingface_hub import HfApi

            token = os.environ.get('HF_TOKEN')
            self._api = HfApi(token=token)
        return self._api

    def read(self):
        """Fresh ``(rows, commit_sha)`` for ``revision``."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        info = self.api.repo_info(
            self.repo_id, repo_type='model', revision=self.revision
        )
        sha = info.sha
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=VALIDATION_CSV,
                repo_type='model',
                revision=sha,
                token=self.api.token,
            )
            text = Path(path).read_text(encoding='utf-8')
        except EntryNotFoundError:
            text = ''
        return loads(text), sha

    def commit(self, rows, parent, message):
        """Write ``rows`` only if ``revision`` still points at ``parent``."""
        from huggingface_hub.errors import HfHubHTTPError

        try:
            self.api.upload_file(
                path_or_fileobj=dumps(rows).encode('utf-8'),
                path_in_repo=VALIDATION_CSV,
                repo_id=self.repo_id,
                repo_type='model',
                revision=self.revision,
                parent_commit=parent,
                commit_message=message,
            )
        except HfHubHTTPError as exc:
            if _stale_commit(exc):
                raise StaleValidationCsv(str(exc)) from exc
            raise

    def update(self, mutator, message):
        """Re-read, then commit when ``mutator`` returns new rows.

        ``mutator(rows)`` returns a new row list to write, or ``None`` to
        leave the file alone. A stale parent commit retries from a new read.
        """
        last = None
        for attempt in range(1, self.max_attempts + 1):
            rows, parent = self.read()
            updated = mutator(rows)
            if updated is None:
                return rows
            try:
                self.commit(updated, parent, message)
            except StaleValidationCsv as exc:
                last = exc
                time.sleep(min(2.0, 0.05 * attempt))
                continue
            return updated
        raise RuntimeError(
            f'validation.csv on {self.repo_id} changed {self.max_attempts} '
            'times before the write landed'
        ) from last

    def enqueue(self, model_name, checkpoint_name, wandb_id, global_step):
        state = {'added': False}

        def mutator(rows):
            new_rows, added = enqueue_job(
                rows, model_name, checkpoint_name, wandb_id, global_step
            )
            state['added'] = added
            if not added:
                return None
            return new_rows

        self.update(
            mutator,
            f'queue extra validation {model_name}/{checkpoint_name}',
        )
        return state['added']

    def claim(self, machine_job_id):
        state = {'job': None}

        def mutator(rows):
            new_rows, job, dirty = claim_job(rows, machine_job_id)
            state['job'] = None if job is None else dict(job)
            if not dirty:
                return None
            return new_rows

        self.update(
            mutator, f'claim extra validation for {machine_job_id}'
        )
        return state['job']

    def complete(self, job, machine_job_id):
        def mutator(rows):
            new_rows, dirty = complete_job(rows, job, machine_job_id)
            if not dirty:
                return None
            return new_rows

        self.update(
            mutator,
            'complete extra validation '
            f'{job["model_name"]}/{job["checkpoint_name"]}',
        )
