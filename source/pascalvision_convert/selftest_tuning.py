from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import List, Tuple

from . import tuning

TASKS = 6


def _write(path: str, rows: List[object]) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def _database(work_dir: str, tuned_tasks: List[int],
              workloads: int = TASKS, unmeasured_tasks: List[int] = ()) -> None:
    os.makedirs(work_dir, exist_ok=True)
    workload_path, record_path = tuning.database_files(work_dir)
    _write(workload_path, [['hash-%d' % i, 'payload'] for i in range(workloads)])

    records = [[i, [['trace'], [0.001], {}, []]] for i in tuned_tasks]
    records += [[i, [['trace'], [1e10], {}, []]] for i in unmeasured_tasks]
    _write(record_path, records)


def _case(name: str, work_dir: str, expected: int,
          want_complete: bool, want_untuned: int) -> bool:
    report = tuning.coverage(work_dir, expected)
    ok = (report['complete'] is want_complete
          and report['untuned'] == want_untuned)
    print('  %-38s complete=%-5s untuned=%-2s -> %s'
          % (name, report['complete'], report['untuned'],
             'AS EXPECTED' if ok else 'WRONG'))
    if not ok:
        print('      %s' % json.dumps(report, ensure_ascii=False))
    return ok


def _refusal_cases(root: str) -> Tuple[int, int]:
    from . import convert

    good = bad = 0

    holed = os.path.join(root, 'refusal-holed')
    _database(holed, [i for i in range(TASKS) if i != 2])
    module = os.path.join(holed, 'model.dll')
    with open(module, 'wb') as handle:
        handle.write(b'not a real module, but a real file')
    with open(module + '.inputs', 'w', encoding='utf-8') as handle:
        handle.write('input')
    request = convert.TuningRequest(TASKS, 32, TASKS * 32, holed)
    try:
        convert._refuse_incomplete_tuning(request, module)
    except RuntimeError:
        gone = not os.path.exists(module)
        print('  %-38s refused, module %s -> %s'
              % ('undertuned build', 'deleted' if gone else 'LEFT BEHIND',
                 'AS EXPECTED' if gone else 'WRONG'))
        good += gone
        bad += not gone
    else:
        print('  %-38s went through -> WRONG' % 'undertuned build')
        bad += 1

    full = os.path.join(root, 'refusal-full')
    _database(full, list(range(TASKS)))
    module = os.path.join(full, 'model.dll')
    with open(module, 'wb') as handle:
        handle.write(b'not a real module, but a real file')
    request = convert.TuningRequest(TASKS, 32, TASKS * 32, full)
    try:
        convert._refuse_incomplete_tuning(request, module)
    except RuntimeError as exc:
        print('  %-38s refused a good build -> WRONG (%s)'
              % ('fully tuned build', exc))
        bad += 1
    else:
        kept = os.path.exists(module)
        print('  %-38s accepted, module %s -> %s'
              % ('fully tuned build', 'kept' if kept else 'DELETED',
                 'AS EXPECTED' if kept else 'WRONG'))
        good += kept
        bad += not kept
    return good, bad


def _platform_cases() -> Tuple[int, int]:
    import platform

    good = bad = 0
    if platform.system() != 'Windows':
        print('  %-38s not Windows -> skipped, and said so'
              % 'windows refusal')
        return 0, 0

    os.environ.pop(tuning.WINDOWS_ESCAPE, None)
    try:
        tuning.refuse_on_windows()
    except RuntimeError:
        print('  %-38s refused -> AS EXPECTED' % 'tuning on Windows')
        good += 1
    else:
        print('  %-38s went through -> WRONG' % 'tuning on Windows')
        bad += 1

    os.environ[tuning.WINDOWS_ESCAPE] = '1'
    try:
        tuning.refuse_on_windows()
    except RuntimeError:
        print('  %-38s still refused -> WRONG' % 'escape hatch set')
        bad += 1
    else:
        print('  %-38s allowed -> AS EXPECTED' % 'escape hatch set')
        good += 1
    finally:
        os.environ.pop(tuning.WINDOWS_ESCAPE, None)
    return good, bad


def _input_name_cases(root: str) -> Tuple[int, int]:
    from . import convert

    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print('  %-38s onnx not installed -> skipped, and said so'
              % 'unknown input name')
        return 0, 0

    put = os.path.join(root, 'tiny.onnx')
    entry = helper.make_tensor_value_info('pixel_values', TensorProto.FLOAT,
                                         [1, 3, 8, 8])
    output_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 1, 1])
    graph = helper.make_graph([helper.make_node('GlobalAveragePool',
                                               ['pixel_values'], ['y'])],
                             'tiny', [entry], [output_info])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 9
    onnx.save(model, put)
    loaded = onnx.load(put)

    good = bad = 0
    try:
        convert._refuse_unknown_input_names(loaded, {'image': (1, 3, 8, 8)})
    except ValueError as exc:
        named_in_error = 'pixel_values' in str(exc) and 'image' in str(exc)
        print('  %-38s refused%s -> %s'
              % ('shape for an input that does not exist',
                 ', and named both' if named_in_error else ', but named neither',
                 'AS EXPECTED' if named_in_error else 'WRONG'))
        good += named_in_error
        bad += not named_in_error
    else:
        print('  %-38s went through -> WRONG' % 'shape for a missing input')
        bad += 1

    try:
        convert._refuse_unknown_input_names(loaded,
                                            {'pixel_values': (1, 3, 8, 8)})
    except ValueError as exc:
        print('  %-38s refused a valid name -> WRONG (%s)'
              % ('shape for the real input', exc))
        bad += 1
    else:
        print('  %-38s accepted -> AS EXPECTED' % 'shape for the real input')
        good += 1
    return good, bad


def _free_dimension_cases(root):
    from . import convert

    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print('  %-38s onnx not installed -> skipped, and said so'
              % 'free dimensions')
        return 0, 0

    def model(shape):
        entry = helper.make_tensor_value_info('pixel_values', TensorProto.FLOAT,
                                             shape)
        output_info = helper.make_tensor_value_info('y', TensorProto.FLOAT,
                                              [1, 3, 1, 1])
        graph = helper.make_graph([helper.make_node('GlobalAveragePool',
                                                   ['pixel_values'], ['y'])],
                                 'tiny', [entry], [output_info])
        m = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid('', 13)])
        m.ir_version = 9
        put = os.path.join(root, 'free.onnx')
        onnx.save(m, put)
        return onnx.load(put)


    dynamic_model = model(['batch', 3, 'height', 'width'])
    static_model = model([1, 3, 8, 8])

    good = bad = 0
    try:
        convert._refuse_free_dimensions(dynamic_model, None)
    except ValueError as exc:
        text = str(exc)
        named_in_error = 'pixel_values' in text and '0' in text and '2' in text
        print('  %-38s refused%s -> %s'
              % ('free axes, no shape given',
                 ', and named them' if named_in_error else ', but named them badly',
                 'AS EXPECTED' if named_in_error else 'WRONG'))
        good += named_in_error
        bad += not named_in_error
    else:
        print('  %-38s went through -> WRONG' % 'free axes, no shape given')
        bad += 1

    try:
        convert._refuse_free_dimensions(dynamic_model,
                                        {'pixel_values': (1, 3, 640, 640)})
    except ValueError as exc:
        print('  %-38s refused a named shape -> WRONG (%s)'
              % ('free axes, shape given', exc))
        bad += 1
    else:
        print('  %-38s accepted -> AS EXPECTED' % 'free axes, shape given')
        good += 1

    try:
        convert._refuse_free_dimensions(dynamic_model,
                                        {'pixel_values': (-1, -1, -1, -1)})
    except ValueError as exc:
        named_in_error = '-1' in str(exc)
        print('  %-38s refused%s -> %s'
              % ('the model own free shape echoed back',
                 ', and quoted it' if named_in_error else ', but quoted nothing',
                 'AS EXPECTED' if named_in_error else 'WRONG'))
        good += named_in_error
        bad += not named_in_error
    else:
        print('  %-38s went through -> WRONG' % 'free shape echoed back')
        bad += 1

    try:
        convert._refuse_free_dimensions(static_model, None)
    except ValueError as exc:
        print('  %-38s refused a fixed model -> WRONG (%s)'
              % ('no free axes at all', exc))
        bad += 1
    else:
        print('  %-38s accepted -> AS EXPECTED' % 'no free axes at all')
        good += 1
    return good, bad


def _budget_cases() -> Tuple[int, int]:
    good = bad = 0
    checks = [
        ('53 tasks x 32 = 1696', lambda: tuning.budget_for(53, 32), 1696),
        ('89 tasks x 32 = 2848', lambda: tuning.budget_for(89, 32), 2848),
        ('28 tasks x 64 = 1792', lambda: tuning.budget_for(28, 64), 1792),
    ]
    for name, call, expected in checks:
        try:
            value = call()
        except Exception as exc:  # noqa: BLE001 - reporting a self-test
            print('  %-38s raised %s -> WRONG' % (name, exc))
            bad += 1
            continue
        ok = value == expected
        print('  %-38s %d -> %s' % (name, value, 'AS EXPECTED' if ok else 'WRONG'))
        good += ok
        bad += not ok

    refusals = [
        ('no tasks at all is refused', lambda: tuning.budget_for(0, 32)),
        ('four trials per task is refused', lambda: tuning.budget_for(53, 4)),
    ]
    for name, call in refusals:
        try:
            call()
        except ValueError:
            print('  %-38s refused -> AS EXPECTED' % name)
            good += 1
        else:
            print('  %-38s went through -> WRONG' % name)
            bad += 1
    return good, bad


def main() -> int:
    root = tempfile.mkdtemp(prefix='selftest-tuning-')
    passed = failed = 0
    try:
        print('COVERAGE GATE')
        full = os.path.join(root, 'full')
        _database(full, list(range(TASKS)))
        passed += _case('every task tuned', full, TASKS, True, 0)

        holed = os.path.join(root, 'holed')
        _database(holed, [i for i in range(TASKS) if i != 3])
        passed += _case('one task without a single trial', holed, TASKS, False, 1)

        empty = os.path.join(root, 'empty')
        _database(empty, [])
        passed += _case('no records at all', empty, TASKS, False, TASKS)

        missing = os.path.join(root, 'missing')
        os.makedirs(missing)
        passed += _case('no database on disk', missing, TASKS, False, 0)

        nothing = os.path.join(root, 'nothing')
        _database(nothing, [], workloads=0)
        passed += _case('database with zero workloads', nothing, TASKS, False, 0)

        alien = os.path.join(root, 'alien')
        _database(alien, list(range(TASKS + 3)), workloads=TASKS + 3)
        passed += _case('database of a different model', alien, TASKS, False, 0)

        sentinel = os.path.join(root, 'sentinel')
        _database(sentinel, [], unmeasured_tasks=list(range(TASKS)))
        passed += _case('every record is the unmeasured sentinel', sentinel,
                        TASKS, False, TASKS)

        print('')
        print('REFUSAL DELETES THE MODULE')
        good, bad = _refusal_cases(root)
        passed += good
        failed += bad

        print('')
        print('INPUT NAMES')
        good, bad = _input_name_cases(root)
        passed += good
        failed += bad
        expected_names = 2 if good + bad else 0

        print('')
        print('FREE DIMENSIONS')
        good, bad = _free_dimension_cases(root)
        passed += good
        failed += bad
        expected_free = 4 if good + bad else 0

        print('')
        print('PLATFORM')
        good, bad = _platform_cases()
        passed += good
        failed += bad
        expected_extra = 2 if good + bad else 0

        print('')
        print('BUDGET')
        good, bad = _budget_cases()
        passed += good
        failed += bad
    finally:
        shutil.rmtree(root, ignore_errors=True)

    expected_checks = 14 + expected_extra + expected_names + expected_free
    print('')
    print('RESULT: %d of %d passed' % (passed, expected_checks))
    return 0 if failed == 0 and passed == expected_checks else 1


if __name__ == '__main__':
    sys.exit(main())
