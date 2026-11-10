# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

from . import hf_held_stage, source_freeze, source_lock

MAX_EXPORTED_ONNX_BYTES = 512 * 1024 * 1024
MAX_BLOCK_BYTES = 1024 * 1024
LARGE_PROTO_MIN_FREE_RAM = 5 * 1024 ** 3
MAX_PROTO_MESSAGES = 1000000
MAX_PROTO_DEPTH = 64
_PREFIX = '.crosspascal-exported-onnx-held-'


class ExportedOnnxHoldError(ValueError):
    pass


class ExportedOnnxHoldUnavailable(ExportedOnnxHoldError):
    pass


class ExportedOnnxHoldChanged(ExportedOnnxHoldError):
    pass


@dataclass(frozen=True)
class HeldExportedOnnx:
    compiler_input: str
    source_exported_onnx: str
    original_export_sha256: str
    staged_model_sha256: str
    bytes: int
    retained_stage: str
    status: str = 'WIN32_EXPORTED_ONNX_COMPILER_HELD_NOT_RELEASE_GRADE'
    verification_status: str = 'NOT_CHECKED'
    release_grade_immutable: bool = False
    retained_after_context: bool = True


def _approved_sha(value):
    if (type(value) is not str or len(value) != 64 or
            any(ch not in '0123456789abcdef' for ch in value)):
        raise ExportedOnnxHoldError('exact lowercase exported-ONNX SHA-256 required')
    return value


def _regular(path, api, *, expected_parent=None):
    source_freeze._safe_ancestors(path)
    info = source_freeze._lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise ExportedOnnxHoldError('exported compiler input must be regular ONNX')
    if path.suffix.lower() != '.onnx' or path.name != source_freeze._safe_name(path.name):
        raise ExportedOnnxHoldError('unsafe exported ONNX filename')
    if (not 0 < info.st_size <= MAX_EXPORTED_ONNX_BYTES or
            getattr(info, 'st_nlink', 1) != 1):
        raise ExportedOnnxHoldError('exported ONNX size/link-count unsupported')
    api.assert_local_ntfs(path)
    if expected_parent is not None and path.parent != expected_parent:
        raise ExportedOnnxHoldError('exported ONNX path left expected output directory')
    return info


def _single_hardlink(api, handle):
    info = api.FileInfo()
    if not api.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise ExportedOnnxHoldError(
            'GetFileInformationByHandle(link count) failed winerror=%d' %
            ctypes.get_last_error())
    if info.nNumberOfLinks != 1:
        raise ExportedOnnxHoldError('exported ONNX hardlink aliases not supported')


def _format_check(path, *, expected_bytes):
    _admit_protobuf_ram(expected_bytes)
    try:
        import onnx
        model = onnx.load_model(str(path), load_external_data=False)
    except (ImportError, OSError, ValueError, RuntimeError, TypeError,
            RecursionError) as exc:
        raise ExportedOnnxHoldError(
            'exported ONNX protobuf unavailable or invalid: %s' % exc) from exc
    if (not model.graph or not model.graph.input or not model.graph.output or
            model.training_info or model.functions):
        raise ExportedOnnxHoldError('unsupported ONNX graph/function dependency')


    stack = [(model, 0)]
    messages = 0
    while stack:
        message, depth = stack.pop()
        messages += 1
        if messages > MAX_PROTO_MESSAGES or depth > MAX_PROTO_DEPTH:
            raise ExportedOnnxHoldError('exported ONNX nested protobuf too complex')
        if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
            if (int(message.data_location) == onnx.TensorProto.EXTERNAL or
                    bool(message.external_data)):
                raise ExportedOnnxHoldError(
                    'exported ONNX external_data not held: refusing until '
                    'full selected sidecar custody is implemented')
        if message.DESCRIPTOR.full_name == 'onnx.NodeProto':
            if message.domain not in ('', 'ai.onnx', 'ai.onnx.ml'):
                raise ExportedOnnxHoldError(
                    'custom ONNX operator domain may require unheld external '
                    'plugins and is not supported by the exported holder')
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                repeated = getattr(field, 'is_repeated', None)
                if repeated is None:
                    repeated = field.label == field.LABEL_REPEATED
                remaining = MAX_PROTO_MESSAGES - messages - len(stack)
                if repeated:
                    if len(value) > remaining:
                        raise ExportedOnnxHoldError(
                            'exported ONNX nested protobuf too complex')
                    stack.extend((child, depth + 1) for child in value)
                else:
                    if remaining < 1:
                        raise ExportedOnnxHoldError(
                            'exported ONNX nested protobuf too complex')
                    stack.append((value, depth + 1))
    return messages


def _copy_handle_exact(api, handle, dest, size, block_bytes):
    if not api.kernel.SetFilePointerEx(
            handle, 0, None, source_lock.FILE_BEGIN):
        raise ExportedOnnxHoldChanged('cannot seek exported Win32 input')
    buf = ctypes.create_string_buffer(block_bytes)
    copied = 0
    digest = hashlib.sha256()
    count = api.wt.DWORD()
    with dest.open('xb') as output:
        while copied < size:
            requested = min(block_bytes, size - copied)
            if not api.kernel.ReadFile(handle, ctypes.byref(buf), requested,
                                       ctypes.byref(count), None):
                raise ExportedOnnxHoldChanged('exported ONNX Win32 source read failed')
            if count.value == 0 or count.value > requested:
                raise ExportedOnnxHoldChanged('exported ONNX shortened/overread')
            chunk = buf.raw[:count.value]
            output.write(chunk)
            digest.update(chunk)
            copied += count.value
        if not api.kernel.ReadFile(handle, ctypes.byref(buf), 1,
                                   ctypes.byref(count), None) or count.value:
            raise ExportedOnnxHoldChanged('exported ONNX grew beyond approved extent')
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def _exact_stage(root, root_identity, leaf):
    hf_held_stage._assert_owned_root(root, root_identity)


    with os.scandir(root) as children:
        one = next(children, None)
        second = next(children, None)
    if one is None or second is not None or one.name != leaf.name:
        raise ExportedOnnxHoldChanged(
            'exported private stage contains unexpected or missing members')
    _regular(leaf, source_lock._Kernel32(), expected_parent=root)


def _admit_protobuf_ram(model_bytes):
    if model_bytes <= 64 * 1024 * 1024:
        return

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

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ExportedOnnxHoldUnavailable(
            'cannot measure free physical RAM before large ONNX parse')
    if status.ullAvailPhys < LARGE_PROTO_MIN_FREE_RAM:
        raise ExportedOnnxHoldUnavailable(
            'large exported ONNX protobuf materialization requires >=5 GiB '
            'free physical RAM; no low-memory native fallback')


def await_post_wsl_ram_admission(model_bytes, *, attempts=13, interval=5):
    if type(model_bytes) is not int or not 0 < model_bytes <= MAX_EXPORTED_ONNX_BYTES:
        raise ExportedOnnxHoldError('invalid selected exported ONNX byte count')
    if type(attempts) is not int or not 1 <= attempts <= 13:
        raise ExportedOnnxHoldError('invalid bounded RAM admission attempts')
    if type(interval) not in (int, float) or not 0 <= interval <= 5:
        raise ExportedOnnxHoldError('invalid bounded RAM admission interval')
    for index in range(attempts):
        try:
            _admit_protobuf_ram(model_bytes)
            return
        except ExportedOnnxHoldUnavailable as exc:
            if ('requires >=5 GiB free physical RAM' not in str(exc) or
                    index + 1 == attempts):
                raise
            time.sleep(interval)


@contextmanager
def hold_private_exported_onnx(exported_path, *, stage_parent,
                               expected_sha256, expected_output_directory,
                               block_bytes=MAX_BLOCK_BYTES):
    if not source_lock._is_windows():
        raise ExportedOnnxHoldUnavailable('native Win32 required; no unlocked fallback')
    if type(block_bytes) is not int or not 1 <= block_bytes <= MAX_BLOCK_BYTES:
        raise ExportedOnnxHoldError('block_bytes outside bounded IO contract')
    sha = _approved_sha(expected_sha256)
    api = source_lock._Kernel32()
    output_root = hf_held_stage._directory(
        expected_output_directory, 'selected ONNX output', api)
    parent = hf_held_stage._directory(stage_parent, 'exported ONNX held parent', api)
    if parent == output_root or output_root in parent.parents:
        raise ExportedOnnxHoldError(
            'exported ONNX private stage must not be inside public output')
    source = Path(os.fspath(exported_path))
    if str(source) != os.path.abspath(str(source)):
        raise ExportedOnnxHoldError('exported ONNX must be canonical absolute path')
    info = _regular(source, api, expected_parent=output_root)
    size = info.st_size
    _admit_protobuf_ram(size)
    required = hf_held_stage._required_stage_space(
        parent, ((source.name, size, sha),), ())
    if shutil.disk_usage(parent).free < required:
        raise ExportedOnnxHoldUnavailable('insufficient NTFS stage space/headroom')
    with ExitStack() as handles:
        original_handle = api.open(source)
        handles.callback(api.close, original_handle)
        original_identity = api.identity(original_handle)
        _single_hardlink(api, original_handle)
        hf_held_stage._native_check(
            api, source, original_handle, original_identity, size, sha,
            block_bytes)
        _format_check(source, expected_bytes=size)


        stage = Path(tempfile.mkdtemp(prefix=_PREFIX, dir=parent))
        root_identity = hf_held_stage._object_id(source_freeze._lstat(stage))
        output_path = stage / source.name
        hf_held_stage._assert_owned_root(stage, root_identity)
        copied_sha = _copy_handle_exact(
            api, original_handle, output_path, size, block_bytes)
        if copied_sha != sha:
            raise ExportedOnnxHoldChanged('private exported ONNX copy differs from exporter SHA')
        _exact_stage(stage, root_identity, output_path)
        staged_handle = api.open(output_path)
        handles.callback(api.close, staged_handle)
        staged_identity = api.identity(staged_handle)
        _single_hardlink(api, staged_handle)
        hf_held_stage._native_check(
            api, output_path, staged_handle, staged_identity, size, sha,
            block_bytes)
        hf_held_stage._native_check(
            api, source, original_handle, original_identity, size, sha,
            block_bytes)
        _format_check(output_path, expected_bytes=size)
        ticket = HeldExportedOnnx(
            str(output_path), str(source), sha, sha, size, str(stage))
        try:
            yield ticket
        except BaseException:
            raise
        else:
            hf_held_stage._native_check(
                api, output_path, staged_handle, staged_identity, size, sha,
                block_bytes)
            hf_held_stage._native_check(
                api, source, original_handle, original_identity, size, sha,
                block_bytes)
            _exact_stage(stage, root_identity, output_path)
            _format_check(output_path, expected_bytes=size)
