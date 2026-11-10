# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys

from . import onnx_closure, source_freeze, source_lock
from .llm_build_profile import console_python


SCHEMA = 'crosspascal.paddleocr.original-ort-raw-capture'
SCHEMA_VERSION = 1
STATUS = 'ORT_RAW_CAPTURE_ONLY'

MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_INPUT_ELEMENTS = 8_000_000
MAX_OUTPUT_ELEMENTS = 8_000_000
MAX_OUTPUT_TENSORS = 8
MAX_INPUT_BYTES = MAX_INPUT_ELEMENTS * 4
MAX_OUTPUT_BYTES = MAX_OUTPUT_ELEMENTS * 4
MAX_CAPTURE_JSON_BYTES = 64 * 1024
MAX_CHILD_SECONDS = 120
MIN_AVAILABLE_RAM_BYTES = 5 * 1024 ** 3

MAX_CHILD_PROCESS_BYTES = 4 * 1024 ** 3

MAX_SHAPE_INFERENCE_MODEL_BYTES = 64 * 1024 * 1024
READ_BLOCK_BYTES = 1024 * 1024
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_POSITIVE = re.compile(r'[0-9]+\Z')
_MODEL_COPY = 'model.onnx'
_INPUT_COPY = 'input.f32le'
_CAPTURE = 'capture.json'
_REFUSAL = 'child-refusal.json'
_OWNER_TICKET = 'parent-owned-stage.json'
_OWNER_SCHEMA = 'crosspascal.ocr.private-parent-stage'
_ACTIVE_CHILD_JOB = None


class CaptureRefused(ValueError):
    pass


def _absolute(raw, label):
    try:
        path = source_freeze._absolute_path(raw, label)
        source_freeze._safe_name(path.name)
        return path
    except (OSError, TypeError, ValueError, source_freeze.SourceFreezeError) as exc:
        raise CaptureRefused('%s must be a safe local file path: %s'
                             % (label, exc)) from exc


def _safe_path(path, *, missing_leaf=False):
    try:
        source_freeze._safe_ancestors(path, leaf_may_be_missing=missing_leaf)
    except (OSError, source_freeze.SourceFreezeError) as exc:
        raise CaptureRefused('linked, redirected or missing path %s: %s'
                             % (path, exc)) from exc


def _regular(path, label, *, limit, expected_bytes=None):
    _safe_path(path)
    info = source_freeze._lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CaptureRefused('%s must be a unique regular file' % label)
    if not 0 < info.st_size <= limit:
        raise CaptureRefused('%s exceeds its nonzero %d-byte admission limit'
                             % (label, limit))
    if expected_bytes is not None and info.st_size != expected_bytes:
        raise CaptureRefused('%s has %d bytes, declared shape needs %d'
                             % (label, info.st_size, expected_bytes))
    return info


def _identity(info):
    return source_freeze._descriptor_identity(info) + (info.st_nlink,)


def _sha_file(path, label, *, limit, expected_bytes=None):
    before = _regular(path, label, limit=limit, expected_bytes=expected_bytes)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    digest = hashlib.sha256()
    size = 0
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            if _identity(os.fstat(fd)) != _identity(before):
                raise CaptureRefused('%s FileID/size changed before SHA' % label)
            while True:
                block = stream.read(min(READ_BLOCK_BYTES, limit - size + 1))
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise CaptureRefused('%s grew beyond the SHA byte limit' % label)
                digest.update(block)
            if (_identity(os.fstat(fd)) != _identity(before)
                    or _identity(_regular(path, label, limit=limit,
                                          expected_bytes=expected_bytes))
                    != _identity(before)):
                raise CaptureRefused('%s changed during SHA' % label)
    finally:
        os.close(fd)
    if size != before.st_size or (expected_bytes is not None
                                  and size != expected_bytes):
        raise CaptureRefused('%s changed byte length during SHA' % label)
    return digest.hexdigest(), size


def _parse_shape(raw):
    if not isinstance(raw, str):
        raise CaptureRefused('--shape must be 1,3,H,W')
    parts = raw.split(',')
    if len(parts) != 4 or any(not _POSITIVE.fullmatch(s) or len(s) > 5
                              for s in parts):
        raise CaptureRefused('--shape must have four positive decimal dimensions')
    dims = [int(s) for s in parts]
    if (dims[:2] != [1, 3] or not 1 <= dims[2] <= 4096
            or not 1 <= dims[3] <= 4096
            or dims[0] * dims[1] * dims[2] * dims[3] > MAX_INPUT_ELEMENTS):
        raise CaptureRefused('--shape must be bounded NCHW [1,3,H,W]')
    return dims


def _available_ram_bytes():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        call = kernel.GlobalMemoryStatusEx
        call.argtypes = [ctypes.POINTER(MemoryStatus)]
        call.restype = ctypes.c_int
        if not call(ctypes.byref(status)):
            raise CaptureRefused('cannot measure available physical RAM')
        return status.ullAvailPhys
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError) as exc:
        raise CaptureRefused('cannot measure available physical RAM') from exc


def _require_heavy_memory():
    if _available_ram_bytes() < MIN_AVAILABLE_RAM_BYTES:
        raise CaptureRefused('original trained ONNX CPU inference requires at '
                             'least 5 GiB currently available physical RAM')


def _install_child_job_limit():
    global _ACTIVE_CHILD_JOB
    if os.name != 'nt' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise CaptureRefused('bounded ORT inference requires 64-bit Windows Job Objects')
    if _ACTIVE_CHILD_JOB is not None:
        raise CaptureRefused('a child memory job was already installed')
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_longlong),
            ('PerJobUserTimeLimit', ctypes.c_longlong),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class IOCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', BasicLimits), ('IoInfo', IOCounters),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    if (ctypes.sizeof(BasicLimits) != 64
            or ctypes.sizeof(IOCounters) != 48
            or ctypes.sizeof(ExtendedLimits) != 144):
        raise CaptureRefused('unproved Windows Job Object 64-bit ABI layout')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise CaptureRefused('Win32 child Job Object could not be created')
    try:
        constraints = ExtendedLimits()


        constraints.BasicLimitInformation.LimitFlags = 0x100 | 0x200 | 0x2000
        constraints.ProcessMemoryLimit = MAX_CHILD_PROCESS_BYTES
        constraints.JobMemoryLimit = MAX_CHILD_PROCESS_BYTES
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(constraints),
                                              ctypes.sizeof(constraints)):
            raise CaptureRefused('Win32 child memory limit installation refused')
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise CaptureRefused('child cannot join bounded Job Object; '
                                 'existing incompatible job or access rights')
    except BaseException:
        kernel.CloseHandle(job)
        raise
    _ACTIVE_CHILD_JOB = job


def _copy_exact(source, target, label, *, sha, size, limit):
    before = _regular(source, label, limit=limit, expected_bytes=size)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    digest = hashlib.sha256()
    copied = 0
    fd = os.open(source, flags)
    try:
        with os.fdopen(fd, 'rb', closefd=False) as incoming:
            if _identity(os.fstat(fd)) != _identity(before):
                raise CaptureRefused('%s changed before private copy' % label)
            with target.open('xb') as output:
                while True:
                    block = incoming.read(min(READ_BLOCK_BYTES, size - copied + 1))
                    if not block:
                        break
                    copied += len(block)
                    if copied > size:
                        raise CaptureRefused('%s grew during private copy' % label)
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if (_identity(os.fstat(fd)) != _identity(before)
                    or _identity(_regular(source, label, limit=limit,
                                          expected_bytes=size))
                    != _identity(before)):
                raise CaptureRefused('%s changed during private copy' % label)
    finally:
        os.close(fd)
    if copied != size or digest.hexdigest() != sha:
        raise CaptureRefused('%s copy differs from approved original SHA' % label)
    copied_sha, copied_size = _sha_file(target, 'private ' + label,
                                        limit=limit, expected_bytes=size)
    if (copied_sha, copied_size) != (sha, size):
        raise CaptureRefused('%s private copy differs from original bytes' % label)


@contextmanager
def _held_files(records):
    if os.name != 'nt':
        yield
        return
    kernel = source_lock._Kernel32()
    held = []
    with ExitStack() as handles:
        for path, expected_sha, expected_size in records:
            kernel.assert_local_ntfs(path)
            handle = kernel.open(path)
            handles.callback(kernel.close, handle)
            held.append((path, handle, expected_sha, expected_size,
                         kernel.identity(handle)))

            digest, count = _hash_held_win32(kernel, handle, expected_size)
            if (digest, count) != (expected_sha, expected_size):
                raise CaptureRefused('locked Win32 source differs from approved SHA')
        yield
        for path, handle, expected_sha, expected_size, identity in held:
            if kernel.identity(handle) != identity:
                raise CaptureRefused('locked Win32 source identity changed')
            digest, count = _hash_held_win32(kernel, handle, expected_size)
            if (digest, count) != (expected_sha, expected_size):
                raise CaptureRefused('locked Win32 bytes changed during inference')


@contextmanager
def _fresh_held_destination(target):
    if os.name != 'nt':
        _safe_path(target, missing_leaf=True)
        target.mkdir(mode=0o700, exist_ok=False)
        yield
        return
    from .omdet_public_candidate import (
        OmDetPublicCandidateError, _hold_plain_ntfs_directory,
    )
    parent = target.parent
    if (os.path.normcase(os.path.abspath(str(parent))) !=
            os.path.normcase(os.path.realpath(str(parent)))):
        raise CaptureRefused('capture destination has redirected/short-name ancestors')
    try:
        with ExitStack() as holders:
            for ancestor in reversed((parent, *parent.parents)):
                holders.enter_context(_hold_plain_ntfs_directory(ancestor))
            if os.path.lexists(target):
                raise CaptureRefused('capture destination occupied during NTFS hold')
            target.mkdir(mode=0o700, exist_ok=False)
            holders.enter_context(_hold_plain_ntfs_directory(target))
            yield
    except OmDetPublicCandidateError as exc:
        raise CaptureRefused('held NTFS capture namespace refused: %s' % exc) from exc


def _hash_held_win32(kernel, handle, expected_bytes):
    if (type(expected_bytes) is not int or
            not 0 < expected_bytes <= max(MAX_MODEL_BYTES, MAX_INPUT_BYTES)):
        raise CaptureRefused('invalid original ONNX Win32 held byte admission')
    if not kernel.kernel.SetFilePointerEx(handle, 0, None, source_lock.FILE_BEGIN):
        raise CaptureRefused('Win32 locked-file SHA seek failed')
    buffer = ctypes.create_string_buffer(READ_BLOCK_BYTES)
    read = kernel.wt.DWORD()
    digest = hashlib.sha256()
    length = 0
    while True:
        if not kernel.kernel.ReadFile(handle, ctypes.byref(buffer),
                                      READ_BLOCK_BYTES, ctypes.byref(read), None):
            raise CaptureRefused('Win32 locked-file SHA read failed')
        if not read.value:
            break
        length += read.value
        if length > expected_bytes:
            raise CaptureRefused('Win32 held file grew beyond admitted byte count')
        digest.update(buffer.raw[:read.value])
    if length != expected_bytes:
        raise CaptureRefused('Win32 held file size differs from approved bytes')
    return digest.hexdigest(), length


def _no_attestation():
    return {
        'status': STATUS,
        'acceptance_status': 'NOT_CHECKED',
        'composite_verification': 'NOT_CHECKED',
        'native_composite_oracle': 'NOT_RUN',
        'human_ground_truth': 'NOT_ATTESTED',
        'facade_manifest_issued': False,
        'ready_issued': False,
    }


def _check_private_root(root):
    _safe_path(root)
    if not stat.S_ISDIR(source_freeze._lstat(root).st_mode):
        raise CaptureRefused('private capture destination is not a directory')
    for forbidden in ('manifest.json', 'manifest.txt', 'READY',
                      source_freeze.READY_MARKER):
        if os.path.lexists(root / forbidden):
            raise CaptureRefused('root Facade/READY marker must never be written')


def _strict_json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CaptureRefused('duplicate capture JSON key: ' + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise CaptureRefused('non-finite JSON constant ' + value)

    try:
        return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
                          parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureRefused('invalid capture JSON: %s' % exc) from exc


def _ticket_identity(root):
    _check_private_root(root)
    details = source_freeze._lstat(root)
    if not stat.S_ISDIR(details.st_mode):
        raise CaptureRefused('parent-created stage is not a plain directory')
    return [int(details.st_dev), int(details.st_ino)]


def _issue_parent_ticket(root, secret, model_sha, input_sha, shape, input_name):
    if not _HEX64.fullmatch(secret):
        raise CaptureRefused('invalid parent stage capability')
    payload = {
        'schema': _OWNER_SCHEMA, 'version': 1, 'capability': secret,
        'identity': _ticket_identity(root),
        'model_sha256': model_sha, 'input_sha256': input_sha,
        'shape': list(shape), 'input_name': input_name,
        'status': 'PRIVATE_PARENT_STAGE_ONLY_NOT_ATTESTED',
    }
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False)
               + '\n').encode('utf-8')
    if len(encoded) > 4096:
        raise CaptureRefused('parent stage capability metadata exceeded quota')
    with (root / _OWNER_TICKET).open('xb') as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def _admit_parent_ticket(root, secret, model_sha, input_sha, shape, input_name,
                         *, before_inference=True):
    if not isinstance(secret, str) or not _HEX64.fullmatch(secret):
        raise CaptureRefused('missing/invalid exclusive parent-stage capability')
    path = root / _OWNER_TICKET
    _regular(path, 'parent stage ticket', limit=4096)
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size > 4096:
            raise CaptureRefused('parent-stage ticket exceeded bounded size')
        contents = stream.read(4097)
    if len(contents) > 4096:
        raise CaptureRefused('parent-stage ticket grew during bounded read')
    record = _strict_json(contents)
    expected = {
        'schema': _OWNER_SCHEMA, 'version': 1, 'capability': secret,
        'identity': _ticket_identity(root),
        'model_sha256': model_sha, 'input_sha256': input_sha,
        'shape': list(shape), 'input_name': input_name,
        'status': 'PRIVATE_PARENT_STAGE_ONLY_NOT_ATTESTED',
    }
    if type(record) is not dict or record != expected:
        raise CaptureRefused('parent stage ticket is not bound to these model/input bytes')


    if before_inference:
        admitted = {_OWNER_TICKET, _MODEL_COPY, _INPUT_COPY}
        if {item.name for item in root.iterdir()} != admitted:
            raise CaptureRefused('parent stage contains unapproved/missing pre-run members')


def _verify_capture(root, model_sha, input_sha, shape):
    _check_private_root(root)
    path = root / _CAPTURE
    _regular(path, 'capture.json', limit=MAX_CAPTURE_JSON_BYTES)
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size > MAX_CAPTURE_JSON_BYTES:
            raise CaptureRefused('capture JSON exceeded read limit')
        payload = stream.read(MAX_CAPTURE_JSON_BYTES + 1)
    if len(payload) > MAX_CAPTURE_JSON_BYTES:
        raise CaptureRefused('capture JSON grew beyond read limit')
    capture = _strict_json(payload)
    if not isinstance(capture, dict):
        raise CaptureRefused('capture JSON must be an object')
    expected = dict(_no_attestation(), schema=SCHEMA,
                    schema_version=SCHEMA_VERSION)
    if any(capture.get(key) != value or type(capture.get(key)) is not type(value)
           for key, value in expected.items()):
        raise CaptureRefused('capture has an invalid status/attestation contract')
    model, actual_input = capture.get('original_onnx'), capture.get('input')
    declared_input_bytes = 4 * shape[0] * shape[1] * shape[2] * shape[3]
    if (not isinstance(model, dict) or not isinstance(actual_input, dict)
            or model.get('file') != _MODEL_COPY
            or model.get('sha256') != model_sha
            or type(model.get('bytes')) is not int
            or not 0 < model['bytes'] <= MAX_MODEL_BYTES
            or model.get('external_data') is not False
            or actual_input.get('file') != _INPUT_COPY
            or actual_input.get('sha256') != input_sha
            or actual_input.get('shape') != shape
            or actual_input.get('dtype') != 'float32le'
            or actual_input.get('bytes') != declared_input_bytes
            or not isinstance(actual_input.get('name'), str)
            or not actual_input['name']
            or capture.get('execution_provider') != 'CPUExecutionProvider'
            or capture.get('active_providers') != ['CPUExecutionProvider']):
        raise CaptureRefused('CPU provider/source SHA or exact input contract changed')
    model_measured, model_bytes = _sha_file(root / _MODEL_COPY,
                                            'original ONNX private copy',
                                            limit=MAX_MODEL_BYTES,
                                            expected_bytes=model['bytes'])
    if (model_measured, model_bytes) != (model_sha, model['bytes']):
        raise CaptureRefused('private original ONNX metadata differs from source')
    if _sha_file(root / _INPUT_COPY, 'private preprocessed input',
                 limit=MAX_INPUT_BYTES,
                 expected_bytes=declared_input_bytes)[0] != input_sha:
        raise CaptureRefused('private input differs from captured SHA')
    expected_names = {_MODEL_COPY, _INPUT_COPY, _CAPTURE}

    if os.path.lexists(root / _OWNER_TICKET):
        expected_names.add(_OWNER_TICKET)
    outputs = capture.get('outputs')
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= MAX_OUTPUT_TENSORS:
        raise CaptureRefused('capture lacks bounded real ONNX float32 outputs')
    total = 0
    for index, item in enumerate(outputs):
        filename = 'output-%02d.f32le' % index
        if (not isinstance(item, dict) or item.get('file') != filename
                or item.get('dtype') != 'float32le'
                or not isinstance(item.get('name'), str)
                or not item['name'] or len(item['name']) > 256
                or not isinstance(item.get('shape'), list)
                or not 1 <= len(item['shape']) <= 5
                or any(type(dim) is not int or dim < 1
                       for dim in item['shape'])):
            raise CaptureRefused('invalid ORT output metadata')
        count = 1
        for dim in item['shape']:
            count *= dim
            if count > MAX_OUTPUT_ELEMENTS:
                raise CaptureRefused('ORT output exceeded element quota')
        total += count * 4
        if total > MAX_OUTPUT_BYTES or item.get('bytes') != count * 4:
            raise CaptureRefused('ORT outputs exceeded aggregate byte quota')
        digest, size = _sha_file(root / filename, 'ORT output',
                                 limit=MAX_OUTPUT_BYTES,
                                 expected_bytes=count * 4)
        if (digest, size) != (item.get('sha256'), count * 4):
            raise CaptureRefused('ORT output SHA/size differs from capture')
        expected_names.add(filename)
    actual = set()
    for index, item in enumerate(root.iterdir()):
        if index >= len(expected_names):
            raise CaptureRefused('private capture has excessive/unexpected members')
        actual.add(item.name)
        _regular(item, 'capture member', limit=max(MAX_MODEL_BYTES,
                                                  MAX_INPUT_BYTES,
                                                  MAX_OUTPUT_BYTES,
                                                  MAX_CAPTURE_JSON_BYTES))
    if actual != expected_names:
        raise CaptureRefused('private capture contains unexpected/missing members')
    capture_sha, capture_bytes = _sha_file(path, 'capture JSON',
                                           limit=MAX_CAPTURE_JSON_BYTES,
                                           expected_bytes=len(payload))
    if capture_bytes != len(payload) or capture_sha != hashlib.sha256(payload).hexdigest():
        raise CaptureRefused('capture JSON changed during final review')
    return capture, capture_sha


def _child_reason(root):
    path = root / _REFUSAL
    if not os.path.lexists(path):
        return 'child produced no bounded refusal detail'
    try:
        _regular(path, 'child refusal', limit=4096)
        with path.open('rb') as stream:
            payload = stream.read(4097)
        if len(payload) > 4096:
            raise CaptureRefused('child refusal note exceeds size limit')
        row = _strict_json(payload)
        if (not isinstance(row, dict) or row.get('status') != 'REFUSED'
                or not isinstance(row.get('reason'), str)
                or len(row['reason']) > 1024):
            raise CaptureRefused('invalid child refusal note')
        return row['reason']
    except (OSError, CaptureRefused):
        return 'child refusal note is invalid'


def _specialize_static_output_shapes(model, input_name, shape, *,
                                     source_bytes):
    import onnx

    if (type(source_bytes) is not int or not
            0 < source_bytes <= MAX_SHAPE_INFERENCE_MODEL_BYTES):
        raise CaptureRefused('dynamic ONNX shape proof exceeds 64 MiB model quota')
    if (not isinstance(model, onnx.ModelProto)
            or type(input_name) is not str or not input_name
            or type(shape) is not list or len(shape) != 4
            or shape[:2] != [1, 3]
            or any(type(axis) is not int or not 1 <= axis <= 4096
                   for axis in shape)
            or shape[0] * shape[1] * shape[2] * shape[3] > MAX_INPUT_ELEMENTS):
        raise CaptureRefused('dynamic shape proof requires an exact static [1,3,H,W] input')
    if len(model.graph.input) != 1:
        raise CaptureRefused('dynamic shape proof requires exactly one ONNX graph input')
    selected = model.graph.input[0]
    if (selected.name != input_name
            or not selected.type.HasField('tensor_type')
            or selected.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or len(selected.type.tensor_type.shape.dim) != 4):
        raise CaptureRefused('dynamic shape proof input name/type/rank disagrees')
    if not 1 <= len(model.graph.output) <= MAX_OUTPUT_TENSORS:
        raise CaptureRefused('dynamic shape proof has an unsupported output count')
    names = [item.name for item in model.graph.output]
    if (len(set(names)) != len(names)
            or any(not item or len(item) > 256 for item in names)):
        raise CaptureRefused('dynamic shape proof has unnamed/duplicate outputs')

    specialized = onnx.ModelProto()
    specialized.CopyFrom(model)
    axes = specialized.graph.input[0].type.tensor_type.shape.dim
    for index, (axis, selected_size) in enumerate(zip(axes, shape)):
        if axis.HasField('dim_value') and axis.dim_value not in (0, selected_size):
            raise CaptureRefused(
                'dynamic shape proof input axis %d contradicts original graph' % index)
        axis.ClearField('dim_param')
        axis.dim_value = selected_size
    try:
        inferred = onnx.shape_inference.infer_shapes(
            specialized, strict_mode=True, data_prop=True)
    except Exception as exc:
        raise CaptureRefused('original ONNX dynamic output could not be statically '
                             'proven: %s' % exc) from exc
    if [item.name for item in inferred.graph.output] != names:
        raise CaptureRefused('ONNX shape proof changed output identity/order')
    outputs = {}
    total = 0
    for before, after in zip(model.graph.output, inferred.graph.output):
        if (not before.type.HasField('tensor_type')
                or before.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or not after.type.HasField('tensor_type')
                or after.type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
            raise CaptureRefused('dynamic shape proof requires float32 output tensors')
        dims = after.type.tensor_type.shape.dim
        if not 1 <= len(dims) <= 5:
            raise CaptureRefused('ONNX shape proof has unsupported output rank')
        original = before.type.tensor_type.shape.dim
        if len(original) != len(dims):
            raise CaptureRefused('ONNX shape proof changed original output rank')
        count = 1
        resolved = []
        for axis, source_axis in zip(dims, original):
            if (not axis.HasField('dim_value')
                    or not 1 <= axis.dim_value <= MAX_OUTPUT_ELEMENTS
                    or (source_axis.HasField('dim_value')
                        and source_axis.dim_value > 0
                        and source_axis.dim_value != axis.dim_value)
                    or count > MAX_OUTPUT_ELEMENTS // axis.dim_value):
                raise CaptureRefused('ONNX symbolic output remains unbounded/contradictory')
            count *= axis.dim_value
            resolved.append(int(axis.dim_value))
        total += count * 4
        if total > MAX_OUTPUT_BYTES:
            raise CaptureRefused('ONNX shape proof exceeds aggregate output byte cap')
        outputs[before.name] = resolved
    return outputs


def _admit_declared_ort_outputs(metadata, *, inferred_shapes=None):
    if not isinstance(metadata, list) or not 1 <= len(metadata) <= MAX_OUTPUT_TENSORS:
        raise CaptureRefused('ORT declared output count is outside bounded quota')
    if inferred_shapes is not None:
        if (type(inferred_shapes) is not dict or
                set(inferred_shapes) != {getattr(item, 'name', None)
                                        for item in metadata} or
                len(inferred_shapes) != len(metadata)):
            raise CaptureRefused('inferred output semantic names differ from ORT session')
    shapes = []
    total_bytes = 0
    for index, entry in enumerate(metadata):
        declared = getattr(entry, 'shape', None)
        dims = declared
        if inferred_shapes is not None:
            resolved = inferred_shapes[getattr(entry, 'name', None)]
            if (type(resolved) is not list or
                    not isinstance(declared, (tuple, list)) or
                    len(resolved) != len(declared) or
                    any((type(a) is not int and a is not None
                         and (type(a) is not str or len(a) > 256))
                        or (type(a) is int and a != b)
                        for a, b in zip(declared, resolved))):
                raise CaptureRefused('inferred output contradicts ORT declared axes')
            dims = resolved
        if (not isinstance(dims, (tuple, list))
                or not 1 <= len(dims) <= 5):
            raise CaptureRefused('ORT output %d has invalid/dynamic rank' % index)
        count = 1
        for dim in dims:
            if type(dim) is not int or not 1 <= dim <= MAX_OUTPUT_ELEMENTS:
                raise CaptureRefused('ORT output %d has dynamic/unbounded shape' % index)
            if count > MAX_OUTPUT_ELEMENTS // dim:
                raise CaptureRefused('ORT output %d exceeds static element quota' % index)
            count *= dim
        total_bytes += count * 4
        if total_bytes > MAX_OUTPUT_BYTES:
            raise CaptureRefused('ORT declared outputs exceed aggregate byte quota')
        shapes.append(list(dims))
    return shapes


def _child_inference(root, shape, model_sha, input_sha, input_name):
    _require_heavy_memory()
    _check_private_root(root)
    if os.path.lexists(root / _CAPTURE):
        raise CaptureRefused('capture.json already exists')
    model_file, input_file = root / _MODEL_COPY, root / _INPUT_COPY
    expected_input_size = 4 * shape[0] * shape[1] * shape[2] * shape[3]
    if _sha_file(model_file, 'private original model', limit=MAX_MODEL_BYTES)[0] != model_sha:
        raise CaptureRefused('original model copy SHA changed before ORT')
    if _sha_file(input_file, 'private preprocessed input',
                 limit=MAX_INPUT_BYTES,
                 expected_bytes=expected_input_size)[0] != input_sha:
        raise CaptureRefused('input copy SHA changed before ORT')

    import onnx

    try:
        graph = onnx.load(str(model_file), load_external_data=False)
        if onnx_closure._external_tensors(graph):
            raise CaptureRefused('external_data ONNX weights are out of scope')
    except CaptureRefused:
        raise
    except Exception as exc:
        raise CaptureRefused('cannot prove self-contained original ONNX: %s'
                             % exc) from exc
    inferred_shapes = None
    if any(
            not axis.HasField('dim_value') or axis.dim_value <= 0
            for entry in graph.graph.output
            for axis in entry.type.tensor_type.shape.dim):
        graph_input = (input_name if input_name is not None else
                       graph.graph.input[0].name if len(graph.graph.input) == 1
                       else '')
        inferred_shapes = _specialize_static_output_shapes(
            graph, graph_input, shape,
            source_bytes=_regular(model_file, 'private original ONNX',
                                  limit=MAX_MODEL_BYTES).st_size)
    del graph

    import numpy as np
    import onnxruntime as ort
    if sys.byteorder != 'little' or 'CPUExecutionProvider' not in ort.get_available_providers():
        raise CaptureRefused('little-endian ONNX Runtime CPU provider is required')
    array = np.fromfile(str(input_file), dtype=np.dtype('<f4'))
    if array.size != expected_input_size // 4 or not bool(np.isfinite(array).all()):
        raise CaptureRefused('preprocessed float32 input length/nonfinite mismatch')
    array = array.reshape(shape)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(model_file), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    session.disable_fallback()
    if session.get_providers() != ['CPUExecutionProvider']:
        raise CaptureRefused('ORT silently enabled another execution provider')
    declared = session.get_inputs()
    if len(declared) != 1 or declared[0].type != 'tensor(float)':
        raise CaptureRefused('ORT model must have exactly one float32 input')
    selected = declared[0]
    if input_name is not None and selected.name != input_name:
        raise CaptureRefused('--input-name differs from original ONNX tensor name')
    if (len(selected.shape) != len(shape) or
            any(type(dim) is int and dim > 0 and dim != actual
                for dim, actual in zip(selected.shape, shape))):
        raise CaptureRefused('requested input shape disagrees with ORT model')
    output_meta = session.get_outputs()
    if not 1 <= len(output_meta) <= MAX_OUTPUT_TENSORS:
        raise CaptureRefused('model declares unsupported number of outputs')
    output_names = [entry.name for entry in output_meta]
    if (len(set(output_names)) != len(output_names)
            or any(not isinstance(name, str) or not name or len(name) > 256
                   or any(ord(ch) < 32 for ch in name)
                   for name in output_names)):
        raise CaptureRefused('invalid/duplicate ORT output name')
    for entry in output_meta:
        if entry.type != 'tensor(float)':
            raise CaptureRefused('non-float32 ORT output is out of scope')
    admitted_shapes = _admit_declared_ort_outputs(
        output_meta, inferred_shapes=inferred_shapes)
    raw_outputs = session.run(output_names, {selected.name: array})
    if len(raw_outputs) != len(output_names):
        raise CaptureRefused('ORT returned an incomplete output tensor set')
    outputs = []
    total = 0
    for index, (name, value) in enumerate(zip(output_names, raw_outputs)):
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32'):
            raise CaptureRefused('ORT returned a non-float32 numpy output')
        dimensions = list(value.shape)
        if dimensions != admitted_shapes[index]:
            raise CaptureRefused('ORT output shape changed from bounded declaration')
        if (not 1 <= len(dimensions) <= 5
                or any(type(dimension) is not int or dimension < 1
                       for dimension in dimensions)
                or not 0 < value.size <= MAX_OUTPUT_ELEMENTS):
            raise CaptureRefused('ORT output has empty/oversized tensor shape')
        total += value.size * 4
        if total > MAX_OUTPUT_BYTES:
            raise CaptureRefused('aggregate ORT outputs exceed bounded byte quota')
        if not bool(np.isfinite(value).all()):
            raise CaptureRefused('ORT output contains NaN/Inf; cannot form OCR evidence')
        encoded = np.ascontiguousarray(value, dtype=np.dtype('<f4')).tobytes(order='C')
        if len(encoded) != value.size * 4:
            raise CaptureRefused('ORT output tensor byte encoding is inconsistent')
        filename = 'output-%02d.f32le' % index
        with (root / filename).open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        digest, length = _sha_file(root / filename, 'captured ORT output',
                                   limit=MAX_OUTPUT_BYTES,
                                   expected_bytes=len(encoded))
        if digest != hashlib.sha256(encoded).hexdigest():
            raise CaptureRefused('ORT output changed after exclusive write')
        outputs.append({'name': name, 'file': filename, 'shape': dimensions,
                        'dtype': 'float32le', 'bytes': length, 'sha256': digest})
    if _sha_file(model_file, 'private original model', limit=MAX_MODEL_BYTES)[0] != model_sha:
        raise CaptureRefused('ONNX model bytes changed during inference')
    if _sha_file(input_file, 'private preprocessed input',
                 limit=MAX_INPUT_BYTES,
                 expected_bytes=expected_input_size)[0] != input_sha:
        raise CaptureRefused('preprocessed input bytes changed during inference')
    payload = dict(_no_attestation(), schema=SCHEMA,
                   schema_version=SCHEMA_VERSION,
                   execution_provider='CPUExecutionProvider',
                   active_providers=list(session.get_providers()),
                   onnxruntime_version=str(ort.__version__),
                   original_onnx={
                       'file': _MODEL_COPY, 'sha256': model_sha,
                       'bytes': _sha_file(model_file, 'model', limit=MAX_MODEL_BYTES)[1],
                       'external_data': False},
                   input={
                       'name': selected.name, 'file': _INPUT_COPY,
                       'dtype': 'float32le', 'shape': shape,
                       'bytes': expected_input_size, 'sha256': input_sha},
                   outputs=outputs,
                   evidence_limit=('Actual CPU ONNX Runtime raw tensor capture only. '
                                   'No image preprocessing or OCR native Pascal '
                                   'inference, postprocessing, independent human '
                                   'ground truth, composite oracle or READY.'))
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False,
                          allow_nan=False) + '\n').encode('utf-8')
    if len(encoded) > MAX_CAPTURE_JSON_BYTES:
        raise CaptureRefused('capture metadata exceeded byte quota')
    with (root / _CAPTURE).open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _parent_run(model_arg, input_arg, shape_arg, into_arg, input_name):
    shape = _parse_shape(shape_arg)
    if input_name is not None and (not input_name or len(input_name) > 256
                                   or any(ord(ch) < 32 for ch in input_name)):
        raise CaptureRefused('--input-name must be a bounded exact ONNX tensor name')
    model = _absolute(model_arg, 'original ONNX')
    input_file = _absolute(input_arg, 'raw input')
    target = _absolute(into_arg, 'private destination')
    if model.suffix.lower() != '.onnx' or input_file.suffix.lower() != '.f32le':
        raise CaptureRefused('requires selected original .onnx and .f32le files')
    if model == input_file or model == target or input_file == target:
        raise CaptureRefused('source, input and destination paths must be distinct')
    if target in model.parents or target in input_file.parents:
        raise CaptureRefused('output cannot be an ancestor of either input file')
    if os.path.lexists(target):
        raise CaptureRefused('capture destination exists; fresh --into is required')
    _safe_path(target, missing_leaf=True)
    if not stat.S_ISDIR(source_freeze._lstat(target.parent).st_mode):
        raise CaptureRefused('private output parent is not a directory')
    expected_input_size = 4 * shape[0] * shape[1] * shape[2] * shape[3]
    model_info = _regular(model, 'original ONNX', limit=MAX_MODEL_BYTES)
    _regular(input_file, 'input f32le', limit=MAX_INPUT_BYTES,
             expected_bytes=expected_input_size)
    if sys.version_info < (3, 11):
        raise CaptureRefused('ocr-ort-raw child requires Python 3.11+ for '
                             'safe -P import-path isolation')


    _require_heavy_memory()
    reserve = (model_info.st_size + expected_input_size + MAX_OUTPUT_BYTES
               + 32 * 1024 * 1024)
    if shutil.disk_usage(target.parent).free < reserve:
        raise CaptureRefused('not enough disk headroom for bounded raw capture')
    model_sha, model_size = _sha_file(model, 'original ONNX',
                                      limit=MAX_MODEL_BYTES)
    input_sha, input_size = _sha_file(input_file, 'input f32le',
                                      limit=MAX_INPUT_BYTES,
                                      expected_bytes=expected_input_size)
    originals = [(model, model_sha, model_size),
                 (input_file, input_sha, input_size)]
    with _held_files(originals), _fresh_held_destination(target):
        _check_private_root(target)
        _copy_exact(model, target / _MODEL_COPY, 'original ONNX',
                    sha=model_sha, size=model_size, limit=MAX_MODEL_BYTES)
        _copy_exact(input_file, target / _INPUT_COPY, 'input f32le',
                    sha=input_sha, size=input_size, limit=MAX_INPUT_BYTES)
        parent_secret = secrets.token_hex(32)
        _issue_parent_ticket(target, parent_secret, model_sha, input_sha,
                             shape, input_name)
        stages = [(target / _MODEL_COPY, model_sha, model_size),
                  (target / _INPUT_COPY, input_sha, input_size)]
        with _held_files(stages):
            command = [console_python(), '-B', '-P', '-s', '-m',
                       'pascalvision_convert.ocr_ort_raw_capture',
                       '__cpu_child', '--into', str(target),
                       '--model-sha256', model_sha,
                       '--input-sha256', input_sha,
                       '--shape', ','.join(map(str, shape)),
                       '--private-token', parent_secret]
            if input_name is not None:
                command += ['--input-name', input_name]


            trusted_root = str(Path(__file__).resolve().parents[1])
            env = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith('PYTHON')}
            env['PYTHONPATH'] = trusted_root
            env['PYTHONNOUSERSITE'] = '1'
            try:
                finished = subprocess.run(
                    command, shell=False, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=trusted_root, env=env, check=False,
                    timeout=MAX_CHILD_SECONDS)
            except subprocess.TimeoutExpired as exc:
                raise CaptureRefused('original ORT CPU child timed out after %ds'
                                     % MAX_CHILD_SECONDS) from exc
            if finished.returncode != 0:
                raise CaptureRefused('original ORT CPU child refused/failed '
                                     '(exit=%d): %s; no numeric acceptance'
                                     % (finished.returncode, _child_reason(target)))
            capture, digest = _verify_capture(target, model_sha, input_sha, shape)
            _admit_parent_ticket(target, parent_secret, model_sha, input_sha,
                                 shape, input_name, before_inference=False)
            for source, sha, count in originals + stages:
                if _sha_file(source, 'pinned model/input',
                             limit=MAX_MODEL_BYTES if source in (model, target / _MODEL_COPY)
                             else MAX_INPUT_BYTES,
                             expected_bytes=count) != (sha, count):
                    raise CaptureRefused('original/private tensor input changed')
    return dict(_no_attestation(), capture_file=str(target / _CAPTURE),
                capture_sha256=digest, original_onnx_sha256=model_sha,
                input_sha256=input_sha, outputs=capture['outputs'])


def _child_main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--into', required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--shape', required=True)
    parser.add_argument('--input-name')
    parser.add_argument('--private-token')
    args = parser.parse_args(argv)
    root = None
    owned = False
    try:
        if not (_HEX64.fullmatch(args.model_sha256)
                and _HEX64.fullmatch(args.input_sha256)):
            raise CaptureRefused('child SHA pins must be exact lowercase SHA-256')
        root = _absolute(args.into, 'private destination')
        shape = _parse_shape(args.shape)
        _admit_parent_ticket(root, args.private_token, args.model_sha256,
                             args.input_sha256, shape, args.input_name)
        owned = True
        _require_heavy_memory()
        _install_child_job_limit()
        _child_inference(root, shape,
                         args.model_sha256, args.input_sha256, args.input_name)
        return 0
    except (CaptureRefused, OSError, ValueError, ImportError, RuntimeError) as exc:
        if owned and root is not None:
            try:
                _check_private_root(root)
                reason = str(exc)[:1024]
                report = json.dumps({'status': 'REFUSED', 'reason': reason},
                                    ensure_ascii=False).encode('utf-8')
                if len(report) <= 4096:
                    with (root / _REFUSAL).open('xb') as stream:
                        stream.write(report)
            except (OSError, CaptureRefused, ValueError):
                pass
        return 1


def main(argv=None) -> int:
    incoming = list(argv) if argv is not None else list(sys.argv[1:])
    if incoming and incoming[0] == '__cpu_child':
        return _child_main(incoming[1:])
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert ocr-ort-raw',
        description=('Capture one actual ORIGINAL ONNX Runtime CPU input/output '
                     'as SHA-pinned f32le evidence; advisory only, not Pascal '
                     'Facade/Det+Rec verification.'))
    parser.add_argument('--model', required=True, help='original single-file ONNX model')
    parser.add_argument('--input', required=True, help='preprocessed float32le tensor')
    parser.add_argument('--shape', required=True, help='bounded 1,3,H,W C-order tensor')
    parser.add_argument('--into', required=True, help='fresh exclusive destination folder')
    parser.add_argument('--input-name', help='exact original ONNX tensor name')
    args = parser.parse_args(incoming)
    try:
        result = _parent_run(args.model, args.input, args.shape,
                             args.into, args.input_name)
    except (CaptureRefused, OSError, ValueError, ImportError,
            source_lock.SourceLockError, source_freeze.SourceFreezeError) as exc:
        print(json.dumps({
            'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
            'native_composite_oracle': 'NOT_RUN',
            'human_ground_truth': 'NOT_ATTESTED',
            'facade_manifest_issued': False, 'ready_issued': False,
            'reason': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))

    return 2


if __name__ == '__main__':
    raise SystemExit(main())
