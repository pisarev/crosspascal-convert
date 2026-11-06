# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Schedule and check MetaSchedule tuning for CPU models.

Count the distinct lowered tasks before choosing a budget. The scheduler
can leave tasks unmeasured if the budget is too small. Use a tuned CPU
target only when the resulting database covers every task.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

DEFAULT_TRIALS_PER_TASK = 32

MINIMUM_TRIALS_PER_TASK = 8


def logical_cores() -> int:
    """Logical cores of this machine, never less than one."""
    return max(1, os.cpu_count() or 1)


def cpu_target(cores: Optional[int] = None, mcpu: Optional[str] = None):
    """Select a CPU target for tuning or for a measured tuning database."""
    import tvm

    spec: Dict[str, object] = {'kind': 'llvm', 'num-cores': cores or logical_cores()}
    if mcpu:
        spec['mcpu'] = mcpu
    return tvm.target.Target(spec)


def _pipeline_head(mod, target):
    """The part of ``static_shape_tuning`` that runs before the tuning itself.

    Copied from the pipeline's own source rather than reinvented: two different
    readings of the same lowering would drift apart silently, and the task count
    would stop matching what the scheduler actually sees.
    """
    import tvm
    from tvm.relax import transform
    from tvm.relax.pipeline import zero_pipeline

    with tvm.target.Target(target):
        return tvm.transform.Sequential([
            transform.DecomposeOpsForInference(),
            transform.CanonicalizeBindings(),
            zero_pipeline(),
        ])(mod)


def count_tasks(mod, target) -> Tuple[int, List[str]]:
    """Count distinct lowered tasks and return their names.

    Structurally identical functions share a tuning task.
    """
    import tvm

    lowered = _pipeline_head(mod, target)
    functions = [(gv.name_hint, f) for gv, f in lowered.functions_items()
                 if f.__class__.__name__ == 'PrimFunc']

    distinct: List[Tuple[str, object]] = []
    for name, f in functions:
        for _, seen in distinct:
            try:
                if tvm.ir.structural_equal(f, seen):
                    break
            except Exception:
                continue
        else:
            distinct.append((name, f))
    return len(distinct), [name for name, _ in distinct]


def budget_for(tasks: int, per_task: int = DEFAULT_TRIALS_PER_TASK) -> int:
    """Allocate the same trial budget to each lowered task."""
    if tasks < 1:
        raise ValueError('a model with no tuning tasks cannot be tuned')
    if per_task < MINIMUM_TRIALS_PER_TASK:
        raise ValueError(
            'trials per task must be at least %d, got %d: a smaller figure '
            'leaves tasks untuned, and that is worse than not tuning at all'
            % (MINIMUM_TRIALS_PER_TASK, per_task))
    return tasks * per_task


def database_files(work_dir: str) -> Tuple[str, str]:
    """Where the tuning pipeline keeps its database."""
    return (os.path.join(work_dir, 'database_workload.json'),
            os.path.join(work_dir, 'database_tuning_record.json'))

_SENTINEL_RUN_SECS = 1e10


def _record_has_real_trial(row: object) -> bool:
    """True when a tuning record carries at least one real run time.

    A record's run times live at ``row[1][1]``. A candidate that built but was
    never timed still gets a record - with every run time set to the sentinel -
    so a record existing is not proof its task was measured. Only a positive
    time below the sentinel counts as a real trial.
    """
    try:
        run_secs = row[1][1]
    except (IndexError, TypeError, KeyError):
        return False
    if not isinstance(run_secs, list) or not run_secs:
        return False
    return any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and 0 < value < _SENTINEL_RUN_SECS
        for value in run_secs
    )


def coverage(work_dir: str, expected_tasks: int) -> Dict[str, object]:
    """Check that the database contains a measured record for each task."""
    import json

    workload_path, record_path = database_files(work_dir)
    result: Dict[str, object] = {
        'work_dir': work_dir,
        'expected': expected_tasks,
        'workloads': 0,
        'tuned': 0,
        'untuned': 0,
        'complete': False,
        'reason': '',
    }
    if not os.path.isfile(workload_path) or not os.path.isfile(record_path):
        result['reason'] = 'the tuning database is missing from %s' % work_dir
        return result

    workloads = []
    with open(workload_path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            workloads.append(row[0] if isinstance(row, list) and row else None)

    by_index = {index: key for index, key in enumerate(workloads)}

    measured = set()
    with open(record_path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not (isinstance(row, list) and row):
                continue
            if not _record_has_real_trial(row):
                continue
            reference = row[0]
            if isinstance(reference, int) and reference in by_index:
                measured.add(by_index[reference])
            else:
                measured.add(reference)

    known = set(k for k in workloads if k is not None)
    result['workloads'] = len(workloads)

    if expected_tasks > 0 and len(workloads) != expected_tasks:
        result['reason'] = (
            'the database holds %d tasks but this model has %d - it was built '
            'for something else' % (len(workloads), expected_tasks))
        return result
    result['tuned'] = len(measured & known)
    result['untuned'] = max(0, len(known) - len(measured & known))
    result['complete'] = bool(known) and result['untuned'] == 0
    if not workloads:
        result['reason'] = 'the database holds no workloads at all'
    elif result['untuned']:
        result['reason'] = ('%d of %d tasks received no trial; under an mcpu '
                            'target those are slower than an untuned build'
                            % (result['untuned'], len(workloads)))
    return result

WINDOWS_ESCAPE = 'PASCALVISION_ALLOW_WINDOWS_TUNING'

TUNING_PACKAGES = ('xgboost', 'psutil', 'cloudpickle')


def refuse_without_packages() -> None:
    """Raise if the tuning dependencies are not installed."""
    import importlib

    missing = []
    for package in TUNING_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            'missing tuning dependencies: %s. Install the tuning extra: '
            'pip install "pascalvision-convert[tuning]"' % ', '.join(missing))


def refuse_on_windows() -> None:
    """Raise unless tuning can be expected to survive on this platform."""
    import platform

    if platform.system() != 'Windows':
        return
    if os.environ.get(WINDOWS_ESCAPE) == '1':
        return
    raise RuntimeError(
        'CPU tuning is disabled on Windows for this TVM build. Tune on Linux, '
        'or set %s=1 to opt into experimental Windows tuning.' % WINDOWS_ESCAPE)


def apply_pipeline(target, work_dir: str):
    """Apply an existing tuning database without running new trials."""
    from tvm import relax

    return relax.get_pipeline(
        'static_shape_tuning',
        total_trials=0,
        target=target,
        work_dir=work_dir,
        max_trials_per_task=None,
    )


def tuning_pipeline(total_trials: int, target, work_dir: str, per_task: int):
    """The tuning pipeline with both limits set explicitly.

    Both, always. Leaving ``max_trials_per_task`` unset makes it default to the
    whole budget, and the pipeline's own documentation calls that out as the
    way to skip tasks entirely.
    """
    from tvm import relax

    return relax.get_pipeline(
        'static_shape_tuning',
        total_trials=total_trials,
        target=target,
        work_dir=work_dir,
        max_trials_per_task=per_task,
    )
