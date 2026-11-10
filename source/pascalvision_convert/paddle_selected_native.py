# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import time

from . import (
    bundle_atomic, hf_held_stage, omdet_public_candidate,
    recipe_file, source_lock,
)


SCHEMA = 'crosspascal.step08.paddle-selected-native-consumer.v1'
_BLOCK = 1024 * 1024
_MAX_MODEL_BYTES = 512 * 1024 * 1024
_MAX_FILES = 40
_MAX_TOTAL_BYTES = 2 * 1024**3
_MAX_METADATA = 2 * 1024 * 1024
_MAX_IMAGE = 32 * 1024 * 1024
_MAX_DICTIONARY = 8 * 1024 * 1024
_NATIVE_RAM = 5 * 1024**3
_FULL_REC_DICT_SHA = (
    '1ea29636956177e400af712d9782e7693f3fb25f98617bed10479d2965a836fd')
_NATIVE_NAME = 'PaddleOcrSelectedConsumer.exe'
_RUNTIMES = ('tvm_runtime.dll', 'tvm_ffi.dll', 'tvm_runtime_extra.dll')
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\Z')
_CTC_ENV = {

    'PPOCR_KEEP_BGR': '1',
    'PPOCR_NORM': '2',
    'PPOCR_CTC_BLANK_ID': '0',
    'PPOCR_CTC_TEMPERATURE': '0',
    'PPOCR_CTC_LOGIT_TEMP': '1.0',
    'PPOCR_CTC_BLANK_MARGIN_ON': '0',
    'PPOCR_SELECTED_SERVER_REC_CLASSES': '18385',
    'PPOCR_SELECTED_SERVER_DET_NORMALIZATION': 'vendor-bgr-v5',
}


class SelectedNativePaddleError(ValueError):
    pass


def _deny(message):
    raise SelectedNativePaddleError('selected Paddle native OCR refused: ' + message)


def _plain_file(path, *, max_bytes=_MAX_MODEL_BYTES):
    candidate = Path(path)
    if (not candidate.is_absolute() or str(candidate) != os.path.abspath(candidate)
            or any(ord(c) > 127 for c in str(candidate))
            or os.path.normcase(os.path.realpath(candidate)) !=
            os.path.normcase(str(candidate))):
        _deny('native source must have a canonical, absolute, local ASCII path')
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise SelectedNativePaddleError('selected file inaccessible: %s' % candidate) from exc
    if (not stat.S_ISREG(info.st_mode) or
            getattr(info, 'st_file_attributes', 0) & 0x400 or
            info.st_nlink != 1 or not 0 < info.st_size <= max_bytes):
        _deny('selected source is linked, empty or exceeds byte quota: %s' % candidate)
    return candidate, info


def _hash(path, *, max_bytes=_MAX_MODEL_BYTES):
    candidate, info = _plain_file(path, max_bytes=max_bytes)
    digest = hashlib.sha256()
    size = 0
    with candidate.open('rb') as handle:
        first = os.fstat(handle.fileno())
        if (first.st_dev, first.st_ino, first.st_size) != (
                info.st_dev, info.st_ino, info.st_size):
            _deny('selected file changed before full SHA: %s' % candidate)
        while size < info.st_size:
            part = handle.read(min(_BLOCK, info.st_size - size))
            if not part:
                _deny('selected source short read during SHA: %s' % candidate)
            digest.update(part)
            size += len(part)
        if handle.read(1):
            _deny('selected file grew during full SHA: %s' % candidate)
        last = os.fstat(handle.fileno())
    again = _plain_file(candidate, max_bytes=max_bytes)[1]
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if key(last) != key(info) or key(again) != key(info):
        _deny('selected file identity changed during SHA: %s' % candidate)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def _row(role, path, *, relative=None, max_bytes=_MAX_MODEL_BYTES):
    record = _hash(path, max_bytes=max_bytes)
    return {'role': role, 'source': str(path), 'relative': relative,
            'bytes': record['bytes'], 'sha256': record['sha256']}


def _json(path):
    file, info = _plain_file(path, max_bytes=_MAX_METADATA)
    with file.open('rb') as handle:
        raw = handle.read(_MAX_METADATA + 1)
    if len(raw) != info.st_size:
        _deny('selected JSON receipt changed during read')

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _deny('duplicate selected metadata key: %s' % key)
            result[key] = value
        return result

    try:
        record = json.loads(raw.decode('utf-8'),
                            object_pairs_hook=unique_pairs,
                            parse_constant=lambda _: _deny('nonfinite JSON'))
    except (UnicodeError, ValueError, RecursionError, TypeError) as exc:
        raise SelectedNativePaddleError('malformed selected JSON receipt') from exc
    if type(record) is not dict:
        _deny('selected receipt must be a JSON object')
    return record


def _directory(path, name, api):
    try:
        return hf_held_stage._directory(path, name, api)
    except (OSError, ValueError, hf_held_stage.HfHeldStageError) as exc:
        raise SelectedNativePaddleError(
            'selected %s must be a canonical local NTFS directory: %s'
            % (name, exc)) from exc


def _compiled_bundle(directory, role, api):
    root = _directory(directory, role + ' compiled bundle', api)
    recipe = _json(root / 'recipe.json')
    provenance = _json(root / 'import-provenance.json')
    try:
        validated = recipe_file.read_recipe(root / 'recipe.json')
    except (ValueError, TypeError, OSError, KeyError) as exc:
        raise SelectedNativePaddleError(
            role + ' compiled recipe/BuildPlan is not valid: %s' % exc) from exc
    if validated != recipe:
        _deny(role + ' compiled recipe changed between bounded reads')
    if (recipe.get('recipe_version') != 2 or
            recipe.get('target') != 'cpu' or
            type(recipe.get('build_plan')) is not dict or
            recipe['build_plan'].get('route') != 'raw' or
            recipe.get('target_profile') !=
            recipe['build_plan'].get('target_profile')):
        _deny(role + ' recipe lacks a coherent selected CPU BuildPlan')
    target = recipe['build_plan']['target_profile']
    deployment = target.get('deployment') if type(target) is dict else None
    if (type(deployment) is not dict or
            (deployment.get('os'), deployment.get('arch'),
             deployment.get('device')) != ('windows', 'x86_64', 'cpu')):
        _deny(role + ' compiled target must be Windows x86_64 CPU')
    if type(recipe.get('source')) is not str:
        _deny(role + ' original selected Paddle owner source absent')
    try:
        approved = recipe_file.validate_paddle_selected_source(
            recipe.get('paddle_selected_source'), source=recipe['source'])
    except (ValueError, TypeError, OSError, KeyError) as exc:
        raise SelectedNativePaddleError(
            role + ' original Paddle closure malformed: %s' % exc) from exc
    if approved['role'] != role or provenance.get('selected_source_closure') != approved:
        _deny(role + ' source approval differs from compiler provenance')
    source_onnx_sha = provenance.get('onnx_sha256')
    if (type(source_onnx_sha) is not str or
            not _HEX.fullmatch(source_onnx_sha)):
        _deny(role + ' exporter has no selected original ONNX SHA')
    exporter = provenance.get('compiler_held_source')
    if (type(exporter) is not dict or
            exporter.get('status') !=
            'WIN32_PADDLE_SELECTED_HELD_CANDIDATE_NOT_RELEASE_GRADE' or
            exporter.get('source_tree_sha256') != approved['tree_sha256'] or
            exporter.get('verification_status') != 'NOT_CHECKED' or
            exporter.get('release_grade_immutable') is not False or
            exporter.get('retained_after_context') is not True):
        _deny(role + ' original selected Paddle exporter stage was not held')
    holder = provenance.get('compiler_window_held_exported_onnx')
    if (type(holder) is not dict or
            holder.get('status') != 'POST_LOCK_AUDIT_PASSED_NOT_RELEASE_GRADE' or
            holder.get('exporter_sha256') != source_onnx_sha or
            holder.get('staged_model_sha256') != source_onnx_sha or
            type(holder.get('bytes')) is not int or
            not 0 < holder['bytes'] <= _MAX_MODEL_BYTES or
            holder.get('held_during_pipeline_run') is not True or
            holder.get('pipeline_sound') is not True or
            holder.get('verification_status') != 'NOT_CHECKED' or
            holder.get('release_grade_immutable') is not False or
            holder.get('retained_after_context') is not True):
        _deny(role + ' lacks sound selected compiler-window native custody')
    name = recipe.get('name')
    if type(name) is not str or not _NAME.fullmatch(name) or name.endswith('.'):
        _deny(role + ' compiled native module has an unsafe basename')
    dll = root / (name + '.cpu.dll')
    sidecar = root / (name + '.cpu.dll.inputs')
    expected = (b'0\tx\t1,3,960,960\n' if role == 'det' else
                b'0\tx\t1,3,48,320\n')
    _plain_file(sidecar, max_bytes=256)
    if sidecar.read_bytes() not in (expected, expected.replace(b'\n', b'\r\n')):
        _deny(role + ' compiled Pascal single-input ABI differs from PP-OCRv5')
    rows = [

        _row(role + '_dll', dll, relative='models/' + role + '.cpu.dll'),
        _row(role + '_inputs', sidecar,
             relative='models/' + role + '.cpu.dll.inputs', max_bytes=256),
        _row(role + '_recipe', root / 'recipe.json',
             relative='audit/' + role + '-recipe.json', max_bytes=_MAX_METADATA),
        _row(role + '_provenance', root / 'import-provenance.json',
             relative='audit/' + role + '-provenance.json',
             max_bytes=_MAX_METADATA),
    ]
    if role == 'rec':
        dictionary = approved['dictionary']
        consumer = provenance.get('compiler_window_held_rec_dictionary')
        if (type(consumer) is not dict or
                consumer.get('schema') !=
                'crosspascal.paddle-selected-rec-consumer-dictionary.v1' or
                consumer.get('status') !=
                'POST_WIN32_HELD_SELECTED_REC_DICTIONARY_NOT_MODEL_UAT' or
                consumer.get('held_through_pipeline_run') is not True or
                consumer.get('source_tree_sha256') != approved['tree_sha256'] or
                consumer.get('relative_path') != dictionary['path'] or
                consumer.get('sha256') != dictionary['sha256'] or
                consumer.get('bytes') != dictionary['bytes'] or
                consumer.get('native_pascal_ctc') != 'NOT_RUN' or
                consumer.get('verification_status') != 'NOT_CHECKED' or
                consumer.get('release_grade_immutable') is not False or
                consumer.get('retained_after_context') is not True or
                dictionary['sha256'] != _FULL_REC_DICT_SHA):
            _deny('selected Server Rec lacks exact full UTF-8 CTC dictionary receipt')
        dictionary_path = root.joinpath(*dictionary['path'].split('/'))
        row = _row('dictionary', dictionary_path,
                   relative=dictionary['path'], max_bytes=_MAX_DICTIONARY)
        if (row['bytes'], row['sha256']) != (
                dictionary['bytes'], dictionary['sha256']):
            _deny('physical compiled recognition dictionary differs from source')
        rows.append(row)
    elif approved['dictionary'] is not None or provenance.get(
            'compiler_window_held_rec_dictionary') is not None:
        _deny('selected Det bundle may not carry a Rec dictionary')
    runtime = {}
    for library in _RUNTIMES:
        row = _row(role + '_runtime', root / library,
                   relative=(library if role == 'det' else None))
        runtime[library] = row
        rows.append(row)
    if (root / 'tvm_runtime_cuda.dll').exists():
        _deny('selected CPU OCR bundle includes an unapproved CUDA runtime')
    return approved, recipe, rows, runtime


def _png(path):
    image, _ = _plain_file(path, max_bytes=_MAX_IMAGE)
    with image.open('rb') as handle:
        head = handle.read(29)
    if (len(head) != 29 or head[:8] != b'\x89PNG\r\n\x1a\n' or
            head[12:16] != b'IHDR' or
            struct.unpack('>I', head[8:12])[0] != 13 or
            tuple(head[24:29]) != (8, 2, 0, 0, 0)):
        _deny('native selected image must be a true noninterlaced RGB8 PNG')
    w, h = struct.unpack('>II', head[16:24])
    if not 2 <= w <= 4096 or not 2 <= h <= 4096 or w*h > 8*1024**2:
        _deny('selected PNG width/height/pixel quota exceeded')
    return [w, h]


def scan_selected_native(det_bundle, rec_bundle, runner, crosscv, image):
    if not source_lock._is_windows():
        _deny('Win32 local NTFS required, no WSL/unlocked fallback')
    api = source_lock._Kernel32()
    det, det_recipe, det_rows, det_runtime = _compiled_bundle(
        det_bundle, 'det', api)
    rec, rec_recipe, rec_rows, rec_runtime = _compiled_bundle(
        rec_bundle, 'rec', api)
    if os.path.normcase(det_bundle) == os.path.normcase(rec_bundle):
        _deny('Det and Rec must come from distinct selected compiled bundles')
    if det_recipe['target_profile'] != rec_recipe['target_profile']:
        _deny('Det and Rec selected target/CPU ABI profiles differ')
    if any((det_runtime[name]['bytes'], det_runtime[name]['sha256']) !=
           (rec_runtime[name]['bytes'], rec_runtime[name]['sha256'])
           for name in _RUNTIMES):
        _deny('Det and Rec compiled against different TVM runtime bytes')
    if Path(runner).name != _NATIVE_NAME or Path(crosscv).name != 'crosscv.dll':
        _deny('selected native Pascal runner/CrossCV must have exact filenames')
    shape = _png(image)
    rows = det_rows + rec_rows + [
        _row('pascal_runner', Path(runner), relative=_NATIVE_NAME),
        _row('crosscv', Path(crosscv), relative='crosscv.dll'),
        _row('image', Path(image), relative='image.png', max_bytes=_MAX_IMAGE),
    ]
    total = sum(row['bytes'] for row in rows)
    if len(rows) > _MAX_FILES or total > _MAX_TOTAL_BYTES:
        _deny('selected native Pascal family byte/member quota exceeded')
    if len({row['source'].casefold() for row in rows}) != len(rows):
        _deny('one selected native resource has more than one source role')
    approved = {
        'schema': SCHEMA,
        'det_source_tree_sha256': det['tree_sha256'],
        'rec_source_tree_sha256': rec['tree_sha256'],
        'target_profile': det_recipe['target_profile'],
        'png_shape': shape,
        'members': sorted(rows, key=lambda row: (row['role'], row['source'])),
    }
    checksum = hashlib.sha256(json.dumps(
        approved, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    return {**approved, 'tree_sha256': checksum,
            'status': 'SCANNED_NOT_LOCKED',
            'verification_status': 'NOT_CHECKED',
            'native_pascal_ctc': 'NOT_RUN',
            'independent_ort': 'NOT_RUN',
            'human_ground_truth': 'NOT_ATTESTED',
            'release_grade_immutable': False}


def _native_resource_gate():
    if not source_lock._is_windows():
        _deny('native Pascal requires Win32, no WSL/substitute CPU process')

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    info = MemoryStatus()
    info.dwLength = ctypes.sizeof(info)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(info)):
        _deny('current native physical memory measurement failed')
    if info.ullAvailPhys < _NATIVE_RAM:
        _deny('actual trained Pascal Det/Rec/CTC requires >=5GiB free RAM')
    try:
        tasklist = source_lock._trusted_tasklist_path()
        processes = subprocess.run(
            [str(tasklist), '/NH', '/FO', 'CSV'],
            capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=15, check=False, cwd=str(tasklist.parent))
    except (OSError, source_lock.SourceLockUnavailable, subprocess.TimeoutExpired) as exc:
        raise SelectedNativePaddleError(
            'cannot establish competing compute/backup absence') from exc
    if processes.returncode != 0:
        _deny('current competing process enumeration returned nonzero')
    competing = {
        'python.exe', 'pythonw.exe', 'pytest.exe', 'backup.exe',
        'robocopy.exe', 'dcc64.exe',
    }
    for row in csv.reader(processes.stdout.splitlines()):
        if len(row) >= 2 and row[0].strip().lower() in competing:
            pid = row[1].strip()
            if pid.isdecimal() and int(pid) != os.getpid():
                _deny('another Python/backup/compiler process is active: '
                      + row[0])
    return info.ullAvailPhys


def _copy_member(source, destination, *, size, digest):
    got = hashlib.sha256()
    count = 0
    with Path(source).open('rb') as owner, Path(destination).open('xb') as target:
        while count < size:
            block = owner.read(min(_BLOCK, size - count))
            if not block:
                _deny('private copy source truncated before completion')
            target.write(block)
            got.update(block)
            count += len(block)
        if owner.read(1):
            _deny('private source grew during bounded native stage copy')
        target.flush()
        os.fsync(target.fileno())
    if count != size or got.hexdigest() != digest:
        _deny('private runnable copy differs from Preview-approved bytes')


def _parse_ctc_result(output, *, width, height):
    path, info = _plain_file(output, max_bytes=_MAX_METADATA)
    with path.open('rb') as handle:
        content = handle.read(_MAX_METADATA + 1)
    if len(content) != info.st_size:
        _deny('native Pascal OCR output changed during complete read')
    try:
        lines = content.decode('utf-8').splitlines()
        if (len(lines) < 3 or
                lines[0] != 'NATIVE_PASCAL_DET_REC_CTC_EXECUTED_NOT_ORACLE' or
                not re.fullmatch(r'lines=[1-9][0-9]*', lines[1])):
            _deny('native Pascal subprocess has no genuine nonempty CTC receipt')
        count = int(lines[1].split('=', 1)[1])
        if count != len(lines) - 2 or count > 4096:
            _deny('native Pascal OCR line count/row extent inconsistent')
        recognized = []
        for sequence, line in enumerate(lines[2:]):
            columns = line.split('\t')
            if len(columns) != 4 or columns[0] != str(sequence):
                _deny('actual Pascal CTC row/index malformed')
            box = [int(x) for x in columns[1].split(',')]
            score = float(columns[3])
            if (len(box) != 4 or not 0 <= box[0] < box[2] <= width or
                    not 0 <= box[1] < box[3] <= height or
                    not math.isfinite(score) or not 0 <= score <= 1 or
                    not columns[2].strip()):
                _deny('Pascal CTC row/text/box/score differs from selected PNG')
            recognized.append({'box': box, 'text': columns[2], 'score': score})
    except (ValueError, OverflowError, UnicodeError) as exc:
        if isinstance(exc, SelectedNativePaddleError):
            raise
        raise SelectedNativePaddleError(
            'actual Pascal OCR result not parseable as bounded UTF-8 text') from exc
    return hashlib.sha256(content).hexdigest(), recognized


def _run_native_bounded(command, *, stage, environment, result_path, native_log):
    deadline_seconds = 900
    with native_log.open('xb') as log:
        child = subprocess.Popen(
            command, cwd=str(stage), env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
        try:
            end = time.monotonic() + deadline_seconds
            while True:
                if os.fstat(log.fileno()).st_size > _MAX_METADATA:
                    _deny('selected native stdout/stderr exceeded bounded log quota')
                if os.path.lexists(result_path):
                    size = os.lstat(result_path).st_size
                    if size > _MAX_METADATA:
                        _deny('selected native result exceeded output byte quota')
                code = child.poll()
                if code is not None:
                    return code
                if time.monotonic() >= end:
                    _deny('selected native Pascal process exceeded 900-second cap')
                time.sleep(0.10)
        finally:
            if child.poll() is None:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
            child.wait(timeout=15)
            log.flush()
            os.fsync(log.fileno())


def run_selected_native(det_bundle, rec_bundle, runner, crosscv, image,
                        *, approved_tree_sha256, into):
    if type(approved_tree_sha256) is not str or not _HEX.fullmatch(
            approved_tree_sha256):
        _deny('explicit prior lowercase Preview tree SHA-256 required')
    _native_resource_gate()
    snapshot = scan_selected_native(
        det_bundle, rec_bundle, runner, crosscv, image)
    if snapshot['tree_sha256'] != approved_tree_sha256:
        _deny('real selected native source differs from earlier approved Preview')
    try:
        stage = Path(bundle_atomic._local_absolute(into))
    except (OSError, ValueError, bundle_atomic.UnsafeAtomicBundle) as exc:
        raise SelectedNativePaddleError(
            'selected native output must be a canonical new local folder') from exc
    if any(ord(c) > 127 for c in str(stage)):
        _deny('selected Pascal native output must use ASCII path')
    if os.path.lexists(stage):
        _deny('selected native result folder occupied; no overwrite')
    api = source_lock._Kernel32()
    parent = _directory(str(stage.parent), 'new native output parent', api)
    selected = snapshot['members']
    root_sources = (Path(det_bundle), Path(rec_bundle))
    if any(stage == root or root in stage.parents for root in root_sources):
        _deny('selected native output may not be inside original compiled roots')
    for row in selected:
        owner = Path(row['source'])
        if owner == stage or stage == owner.parent:
            _deny('private native output overlaps original selected owner file')
    files_to_copy = [(row['relative'], row['bytes'], row['sha256'])
                     for row in selected if row['relative'] is not None]
    required = hf_held_stage._required_stage_space(
        parent, files_to_copy, ('audit', 'dictionary', 'models'))
    if shutil.disk_usage(parent).free < required:
        _deny('insufficient NTFS free disk for retained selected native closure')

    stage_identity = None
    receipt = None
    pending_advisory = None
    with ExitStack() as holds:
        held_names = set()
        for parent_dir in [parent] + [
                Path(row['source']).parent for row in selected]:
            for directory in reversed((parent_dir, *parent_dir.parents)):
                if directory not in held_names:
                    holds.enter_context(
                        omdet_public_candidate._hold_plain_ntfs_directory(directory))
                    held_names.add(directory)
        owner_handles = []
        for row in selected:
            path = Path(row['source'])
            api.assert_local_ntfs(path)
            handle = api.open(path)
            holds.callback(api.close, handle)
            identity = api.identity(handle)
            if identity.file_bytes != row['bytes']:
                _deny('native Win32 owner FileID size differs before child')
            hf_held_stage._native_check(
                api, path, handle, identity, row['bytes'], row['sha256'], _BLOCK)
            owner_handles.append((path, handle, identity, row))

        if os.path.lexists(stage):
            _deny('selected native output name replaced before exclusive mkdir')
        stage.mkdir(exist_ok=False)
        stamp = bundle_atomic._guard_stat(str(stage), directory=True)
        stage_identity = (stamp.st_dev, stamp.st_ino)
        holds.enter_context(
            omdet_public_candidate._hold_plain_ntfs_directory(stage))
        for leaf in ('audit', 'dictionary', 'models'):
            subdir = stage / leaf
            subdir.mkdir(exist_ok=False)
            holds.enter_context(
                omdet_public_candidate._hold_plain_ntfs_directory(subdir))
        copies = []
        for source, _handle, _identity, row in owner_handles:
            if row['relative'] is None:
                continue
            destination = stage.joinpath(*row['relative'].split('/'))
            _copy_member(source, destination, size=row['bytes'],
                         digest=row['sha256'])
            handle = api.open(destination)
            holds.callback(api.close, handle)
            identity = api.identity(handle)
            hf_held_stage._native_check(
                api, destination, handle, identity,
                row['bytes'], row['sha256'], _BLOCK)
            copies.append((destination, handle, identity, row))

        det_module = stage.joinpath(*next(
            row['relative'] for row in selected
            if row['role'] == 'det_dll').split('/'))
        rec_module = stage.joinpath(*next(
            row['relative'] for row in selected
            if row['role'] == 'rec_dll').split('/'))
        chosen_dict = next(row for row in selected
                           if row['role'] == 'dictionary')
        dictionary = stage.joinpath(*chosen_dict['relative'].split('/'))
        result_path = stage / 'ocr-result.txt'
        native_log = stage / 'native.log'
        command = [
            str(stage / _NATIVE_NAME),
            '--det', str(det_module), '--rec', str(rec_module),
            '--dict', str(dictionary),
            '--image', str(stage / 'image.png'),
            '--out', str(result_path),
        ]
        environment = {
            key: value for key, value in os.environ.items()
            if not key.upper().startswith('PPOCR_') and
            key.upper() not in (
                'PADDLE_OCR_DICT', 'PADDLE_OCR_INFER_YML',
                'TVM_RUNTIME_LIB',
            )
        }
        environment.update(_CTC_ENV)
        environment['PADDLE_OCR_DICT'] = str(dictionary)
        _native_resource_gate()
        outcome = _run_native_bounded(
            command, stage=stage, environment=environment,
            result_path=result_path, native_log=native_log)
        if outcome != 0:
            _deny('actual selected Pascal Det->Rec->CTC process exited nonzero')
        result_sha, lines = _parse_ctc_result(
            result_path, width=snapshot['png_shape'][0],
            height=snapshot['png_shape'][1])
        actual_log = _plain_file(native_log, max_bytes=_MAX_METADATA)[0].read_bytes()
        if b'NATIVE_PASCAL_DET_REC_CTC_EXECUTED_NOT_ORACLE' not in actual_log:
            _deny('native subprocess did not emit its real CTC success marker')

        for source, handle, identity, row in owner_handles:
            hf_held_stage._native_check(
                api, source, handle, identity, row['bytes'], row['sha256'], _BLOCK)
        for destination, handle, identity, row in copies:
            hf_held_stage._native_check(
                api, destination, handle, identity,
                row['bytes'], row['sha256'], _BLOCK)
        receipt = {
            'schema': SCHEMA,
            'approved_tree_sha256': approved_tree_sha256,

            'status': 'SELECTED_NATIVE_EXE_EXIT0_CTC_FORMAT_NOT_ATTESTED',
            'native_pascal_ctc': 'SELF_REPORTED_SELECTED_EXE_NOT_ATTESTED',
            'native_process_exit': 0,
            'result_sha256': result_sha,
            'lines': lines,
            'selected_runtime_status': 'WIN32_HELD_SOURCE_AND_PRIVATE_KIT',
            'verification_status': 'NOT_CHECKED',
            'independent_original_ort': 'NOT_RUN',
            'human_ground_truth': 'NOT_ATTESTED',
            'native_binary_authority': 'NOT_INDEPENDENTLY_ATTESTED',
            'facade_manifest_issued': False,
            'download_ready': False,
            'release_grade_immutable': False,
        }

        pending_advisory = (stage / 'advisory-native-ctc.json').open('xb')

    try:
        if pending_advisory is None or receipt is None:
            _deny('selected native output holder did not complete')
        current = bundle_atomic._guard_stat(str(stage), directory=True)
        if (current.st_dev, current.st_ino) != stage_identity:
            _deny('retained native stage identity changed before result issuance')
        if _hash(result_path, max_bytes=_MAX_METADATA)['sha256'] != receipt['result_sha256']:
            _deny('native output bytes changed after Win32 finalizers')
        body = (json.dumps(receipt, ensure_ascii=False, indent=2,
                           sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        if len(body) > _MAX_METADATA:
            _deny('native selected advisory receipt exceeds byte quota')
        pending_advisory.write(body)
        pending_advisory.flush()
        os.fsync(pending_advisory.fileno())
    finally:
        if pending_advisory is not None:
            pending_advisory.close()
    return receipt


def _parse(argv, *, is_run):
    parser = argparse.ArgumentParser(
        prog=('pascalvision-convert paddle-selected-native-run' if is_run else
              'pascalvision-convert paddle-selected-native-preview'))
    for flag in ('--det-bundle', '--rec-bundle', '--runner',
                 '--crosscv', '--image'):
        parser.add_argument(flag, required=True)
    if is_run:
        parser.add_argument('--approved-tree-sha256', required=True)
        parser.add_argument('--into', required=True)
    return parser.parse_args(argv)


def preview_main(argv=None):
    try:
        args = _parse(argv, is_run=False)
    except SystemExit as error:
        return 0 if error.code == 0 else 1
    try:
        result = scan_selected_native(
            args.det_bundle, args.rec_bundle, args.runner,
            args.crosscv, args.image)
        print(json.dumps(result, sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False))
        return 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print('PADDLE SELECTED NATIVE PREVIEW REFUSED: %s' % error,
              file=sys.stderr)
        return 1


def run_main(argv=None):
    try:
        args = _parse(argv, is_run=True)
    except SystemExit as error:
        return 0 if error.code == 0 else 1
    try:
        result = run_selected_native(
            args.det_bundle, args.rec_bundle, args.runner,
            args.crosscv, args.image,
            approved_tree_sha256=args.approved_tree_sha256, into=args.into)
        print(json.dumps(result, sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False))
        return 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.TimeoutExpired) as error:
        print('PADDLE SELECTED NATIVE RUN REFUSED: %s' % error,
              file=sys.stderr)
        return 1
