# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Destination-side bounded RAW ONNX CPU numerical comparison.

This is not a Pascal Facade UAT or a destination VERIFIED passport. A positive
result records an actual successful return from the existing child verifier.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from threading import Thread, Lock

from . import bundle_download, destination_preflight
from .recipe_file import read_recipe
from .llm_build_profile import console_python

MAX_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
MAX_CHILD_OUTPUT = 512 * 1024
CHILD_TIMEOUT_SECONDS = 60


class NativeRunnerRefused(ValueError):
    pass


def _regular(path, kind):
    path = Path(os.path.abspath(os.fspath(path)))
    for node in (path, *path.parents):
        info = os.lstat(node)
        if node.is_symlink() or getattr(node, 'is_junction', lambda: False)() or (
                getattr(info, 'st_file_attributes', 0) & 0x400):
            raise NativeRunnerRefused('linked/reparse %s path' % kind)
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise NativeRunnerRefused('%s must be a unique regular file' % kind)
    return path, info


def _sha(path, limit):
    info = os.stat(path)
    if not 0 < info.st_size <= limit:
        raise NativeRunnerRefused('original ONNX outside byte limit')
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            part = stream.read(1024 * 1024)
            if not part:
                break
            h.update(part)
    after = os.stat(path)
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise NativeRunnerRefused('original ONNX changed during SHA')
    return h.hexdigest()


def _bundle_snapshot(root):
    entries = bundle_download.collect_bundle_files(str(root), allow_download_info=True)
    if len(entries) > 512:
        raise NativeRunnerRefused('bundle member quota exceeded')
    result = []
    total = 0
    for name, path, size in entries:
        total += size
        if total > 8 * 1024 * 1024 * 1024:
            raise NativeRunnerRefused('bundle byte quota exceeded')
        result.append((name, size, _sha(Path(path), 8 * 1024 * 1024 * 1024)))
    return tuple(result)


def _bounded_existing_verifier(source, module):
    """Run the same existing verify module, without its unbounded capture_output."""
    cmd = [console_python(), '-B', '-m', 'pascalvision_convert.verify',
           str(source), str(module), '', 'cpu']
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [part for part in sys.path if part] + [env.get('PYTHONPATH', '')])
    child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             shell=False, env=env)
    data = [bytearray(), bytearray()]
    guard = Lock()
    faults = []
    def drain(index, stream):
        try:
            while True:
                block = stream.read(8192)
                if not block:
                    return
                with guard:
                    if len(data[0]) + len(data[1]) + len(block) > MAX_CHILD_OUTPUT:
                        faults.append('numeric child output quota exceeded')
                        child.kill()
                        return
                    data[index].extend(block)
        except (OSError, ValueError):
            faults.append('numeric child stream failed')
            child.kill()
    readers = [Thread(target=drain, args=(0, child.stdout), daemon=True),
               Thread(target=drain, args=(1, child.stderr), daemon=True)]
    for thread in readers:
        thread.start()
    try:
        code = child.wait(timeout=CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        child.kill()
        child.wait()
        raise NativeRunnerRefused('numeric child timeout') from exc
    finally:
        for thread in readers:
            thread.join(timeout=1)
    if faults or any(thread.is_alive() for thread in readers) or code != 0:
        raise NativeRunnerRefused('numeric child failed, nonzero or output over quota')

    if data[1]:
        raise NativeRunnerRefused('numeric child stderr is nonempty; no clean '
                                  'destination numerical acceptance')
    output = bytes(data[0]).decode('utf-8', 'replace')
    marker = output.rfind(chr(2))
    if marker < 0:
        raise NativeRunnerRefused('numeric child returned no structured report')
    try:
        report = json.loads(output[marker + 1:])
    except (ValueError, UnicodeError) as exc:
        raise NativeRunnerRefused('invalid numeric child report') from exc
    if (type(report) is not dict or report.get('state') != 'done'
            or report.get('name') != 'Comparison with the reference'
            or type(report.get('lines')) is not list
            or type(report.get('numbers')) is not dict):
        raise NativeRunnerRefused('numeric child did not report a complete PASS')
    numbers = report['numbers']
    reference = str(numbers.get('reference') or '').lower()
    if 'onnxruntime' not in reference:
        raise NativeRunnerRefused('numeric child did not use independent ONNX Runtime')
    if not (any(key.startswith('smooth.') for key in numbers) and
            any(key.startswith('flat.') for key in numbers) or
            numbers.get('multi.smooth') is True and
            numbers.get('multi.flat') is True):
        raise NativeRunnerRefused('numeric child lacks nonvacuous smooth/flat evidence')
    return report


def run_cpu_numeric(bundle, original_onnx, *, max_source_bytes=MAX_SOURCE_BYTES):
    """Refuse before verifier unless destination/recipe/original agree exactly."""
    if type(max_source_bytes) is not int or not 1 <= max_source_bytes <= MAX_SOURCE_BYTES:
        raise NativeRunnerRefused('invalid bounded source budget')
    root = Path(os.path.abspath(os.fspath(bundle)))
    source, identity = _regular(original_onnx, 'source')
    if source.suffix.lower() != '.onnx':
        raise NativeRunnerRefused('only caller-selected original ONNX is supported')
    if root == source or root in source.parents or source in root.parents:
        raise NativeRunnerRefused('source and bundle must not overlap')
    before = _sha(source, max_source_bytes)

    try:
        from .recipe import _unheld_onnx_external_data
        if _unheld_onnx_external_data(str(source)):
            raise NativeRunnerRefused('external ONNX tensor data/sidecar is not '
                                      'approved for destination numeric oracle')
    except (ImportError, OSError, ValueError) as exc:
        if isinstance(exc, NativeRunnerRefused):
            raise
        raise NativeRunnerRefused('cannot inspect original ONNX external data; '
                                  'no unbounded sidecar fallback') from exc
    report = destination_preflight.inspect_bundle(
        str(root), destination_preflight.inspect_current_machine())
    if (report.get('status') != 'STATIC_PROFILE_MATCH_NOT_EXECUTED' or
            report.get('acceptance_status') != 'NOT_CHECKED' or
            report.get('target', {}).get('device') != 'cpu' or
            report['target'].get('os') != 'windows' or
            report['target'].get('arch') != 'x86_64' or
            report.get('facade') != 'RAW_TENSORS_NO_FACADE_PASSPORT'):
        raise NativeRunnerRefused('destination RAW CPU static preflight refused')
    recipe = read_recipe(str(root / 'recipe.json'))
    if (recipe.get('target') != 'cpu' or
            not isinstance(recipe.get('source_hash'), str) or
            not re.fullmatch('[0-9a-fA-F]{64}', recipe['source_hash']) or
            recipe['source_hash'].lower() != before or
            recipe.get('source_closure') is not None or
            recipe.get('paddle_selected_source') is not None or
            (isinstance(recipe.get('build_plan'), dict) and
             (recipe['build_plan'].get('route') != 'raw' or
              recipe['build_plan'].get('cpu_codegen') is not None))):
        raise NativeRunnerRefused('original ONNX hash or RAW recipe differs')
    module_name = report['model']['file']
    if (not isinstance(module_name, str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,200}\.cpu\.dll', module_name)):
        raise NativeRunnerRefused('unsupported RAW CPU module name')
    module, _ = _regular(root / module_name, 'module')
    if module.parent != root:
        raise NativeRunnerRefused('module outside bundle')
    bundle_before = _bundle_snapshot(root)
    _bounded_existing_verifier(source, module)
    if _sha(source, max_source_bytes) != before or (
            os.stat(source).st_dev, os.stat(source).st_ino) != (
            identity.st_dev, identity.st_ino):
        raise NativeRunnerRefused('original ONNX changed during native verification')
    after_report = destination_preflight.inspect_bundle(
        str(root), destination_preflight.inspect_current_machine())
    if after_report.get('status') != 'STATIC_PROFILE_MATCH_NOT_EXECUTED' or (
            after_report.get('model') != report.get('model') or
            after_report.get('runtime_libraries') != report.get('runtime_libraries') or
            _bundle_snapshot(root) != bundle_before):
        raise NativeRunnerRefused('bundle changed during numerical verification')
    return {'status': 'TVM_CPU_NUMERIC_PASS',
            'acceptance_status': 'NOT_CHECKED',
            'verification_status': 'NOT_CHECKED',
            'pascal_facade_uat': 'NOT_RUN',
            'runtime_provenance': 'NOT_ATTESTED_BUNDLE_LIBS',
            'original_onnx_sha256': before,
            'native_child_verifier_state': 'done',
            'verified_on_destination': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--original-onnx', required=True)
    args = parser.parse_args(argv)
    try:
        result = run_cpu_numeric(args.bundle, args.original_onnx)
    except (NativeRunnerRefused, OSError, ValueError) as exc:
        print(json.dumps({'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
                          'reason': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
