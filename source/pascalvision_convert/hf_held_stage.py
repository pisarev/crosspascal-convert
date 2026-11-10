# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

from contextlib import ExitStack, contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

from . import hf_closure, source_freeze, source_lock


MAX_HF_SELECTED_BYTES = hf_closure.MAX_TOTAL_BYTES
MAX_HF_SELECTED_FILES = 512
MAX_IO_BLOCK_BYTES = 1024 * 1024
MIN_DISK_HEADROOM = 256 * 1024 * 1024
_STAGE_PREFIX = '.crosspascal-hf-stage-'


class HfHeldStageError(ValueError):
    pass


class HfHeldStageUnavailable(HfHeldStageError):
    pass


class HfHeldStageChanged(HfHeldStageError):
    pass


@dataclass(frozen=True)
class HeldHfStage:
    staged_directory: str
    compiler_input: str
    source_tree_sha256: str
    model_key: str
    file_count: int
    total_bytes: int
    status: str = 'WIN32_HF_SELECTED_PRIVATE_STAGE_CANDIDATE_NOT_RELEASE_GRADE'
    verification_status: str = 'NOT_CHECKED'
    release_grade_immutable: bool = False
    retained_after_context: bool = True


def _limits(max_total_bytes, max_selected_files, block_bytes):
    if (type(max_total_bytes) is not int or
            not 1 <= max_total_bytes <= MAX_HF_SELECTED_BYTES):
        raise HfHeldStageError('HF stage byte quota must be 1..16GiB')
    if (type(max_selected_files) is not int or
            not 1 <= max_selected_files <= MAX_HF_SELECTED_FILES):
        raise HfHeldStageError('HF stage selected-file quota must be 1..512')
    if (type(block_bytes) is not int or
            not 1 <= block_bytes <= MAX_IO_BLOCK_BYTES):
        raise HfHeldStageError('HF stage block_bytes must be 1..1048576')


def _approved_sha(value):
    if (type(value) is not str or len(value) != 64 or
            any(char not in '0123456789abcdefABCDEF' for char in value)):
        raise HfHeldStageError('exact approved HF source-tree SHA-256 required')
    return value.lower()


def _directory(path, label, api):
    try:
        raw = os.fspath(path)
        if (type(raw) is not str or not raw or '\x00' in raw or
                raw.startswith(('\\\\', '//')) or
                not os.path.isabs(raw) or
                any(piece in ('.', '..') for piece in
                    raw.replace('\\', '/').split('/'))):
            raise HfHeldStageError(
                '%s must be a canonical absolute local directory, no UNC or aliases'
                % label)
        base = source_freeze._absolute_path(raw, label)


        canonical = os.path.realpath(str(base))
        if os.path.normcase(os.path.normpath(canonical)) != (
                os.path.normcase(os.path.normpath(str(base)))):
            raise HfHeldStageError('%s resolves through an alias' % label)
        source_freeze._safe_ancestors(base)
        info = source_freeze._lstat(base)
        if not stat.S_ISDIR(info.st_mode):
            raise HfHeldStageError('%s must be an existing regular directory'
                                   % label)
        api.assert_local_ntfs(base)
    except (OSError, TypeError, ValueError,
            source_freeze.SourceFreezeError,
            source_lock.SourceLockError) as error:
        if isinstance(error, HfHeldStageError):
            raise
        raise HfHeldStageUnavailable(
            'unsafe/unavailable local NTFS %s: %s' % (label, error)) from error
    return base


def _same_object(info, identity):
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)) == identity


def _object_id(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def _assert_owned_root(root, identity):
    try:
        source_freeze._safe_ancestors(root)
        current = source_freeze._lstat(root)
    except (OSError, source_freeze.SourceFreezeError) as error:
        raise HfHeldStageChanged(
            'private HF stage root disappeared or was redirected: %s'
            % error) from error
    if not stat.S_ISDIR(current.st_mode) or not _same_object(current, identity):
        raise HfHeldStageChanged('private HF stage root object was replaced')


def _select_files(report, *, max_total_bytes, max_selected_files):
    rows = report['entries']
    if (not isinstance(rows, list) or not rows or
            len(rows) > max_selected_files or
            report['file_count'] != len(rows) or
            report['total_bytes'] > max_total_bytes):
        raise HfHeldStageError(
            'selected HF resource count or total bytes exceed pilot quota')
    result = []
    total = 0
    for row in rows:
        relative = hf_closure._relative(row['path'])
        size, digest = row['bytes'], _approved_sha(row['sha256'])
        if type(size) is not int or size <= 0:
            raise HfHeldStageError('invalid selected HF member byte length')
        total += size
        if total > max_total_bytes:
            raise HfHeldStageError('selected HF byte quota exceeded')
        result.append((relative, size, digest))
    if total != report['total_bytes']:
        raise HfHeldStageError('HF selected source inventory total differs')
    return tuple(result)


def _directories(files):
    names = set()
    for relative, _, _ in files:
        parts = relative.split('/')
        for count in range(1, len(parts)):
            names.add('/'.join(parts[:count]))
    return tuple(sorted(names, key=lambda x: (x.count('/'), x)))


def _copy_selected_file(origin, stage, relative, size, sha, block_bytes):
    destination = stage.joinpath(*relative.split('/'))
    source_freeze._safe_ancestors(destination, leaf_may_be_missing=True)


    with hf_closure._opened_regular(
            origin, relative, max_bytes=max(size, 1)) as (src, info):
        if info.st_nlink != 1:
            raise HfHeldStageError(
                'HF selected source has a hardlink alias: ' + relative)
        if info.st_size != size:
            raise HfHeldStageChanged(
                'selected original HF member changed before copy: ' + relative)
        copied = hashlib.sha256()
        amount = 0


        with destination.open('xb') as dest:
            while True:
                piece = src.read(block_bytes)
                if not piece:
                    break
                amount += len(piece)
                if amount > size:
                    raise HfHeldStageChanged(
                        'original HF member grew during private copy: ' + relative)
                copied.update(piece)
                dest.write(piece)
            dest.flush()
            os.fsync(dest.fileno())
        if amount != size or copied.hexdigest() != sha:
            raise HfHeldStageChanged(
                'selected original HF bytes differ from approved SHA: ' + relative)


def _audit_exact_private_tree(stage, root_id, files, directories):
    _assert_owned_root(stage, root_id)
    expected_files = {item[0]: item[1] for item in files}
    expected_dirs = set(directories)
    seen_files = set()
    seen_dirs = set()


    max_entries = len(expected_files) + len(expected_dirs)

    def walk(folder, prefix=''):
        with os.scandir(folder) as candidates:
            for member in candidates:
                if len(seen_files) + len(seen_dirs) >= max_entries:
                    raise HfHeldStageChanged(
                        'private HF stage exceeds bounded selected-entry count')
                child = Path(member.path)
                relative = prefix + member.name
                try:
                    hf_closure._relative(relative)
                    source_freeze._safe_ancestors(child)
                    info = source_freeze._lstat(child)
                except (OSError, ValueError, source_freeze.SourceFreezeError,
                        hf_closure.HfClosureError) as error:
                    raise HfHeldStageChanged(
                        'invalid private HF stage member %r: %s' %
                        (relative, error)) from error
                if stat.S_ISDIR(info.st_mode):
                    if relative not in expected_dirs:
                        raise HfHeldStageChanged(
                            'unexpected directory inside private HF stage: ' + relative)
                    seen_dirs.add(relative)
                    walk(child, relative + '/')
                elif stat.S_ISREG(info.st_mode):
                    if relative not in expected_files:
                        raise HfHeldStageChanged(
                            'unexpected file inside private HF stage: ' + relative)
                    if info.st_size != expected_files[relative]:
                        raise HfHeldStageChanged(
                            'private HF stage selected member changed size: ' + relative)
                    if info.st_nlink != 1:
                        raise HfHeldStageChanged(
                            'hardlink alias of a private HF staged member: ' + relative)
                    seen_files.add(relative)
                else:
                    raise HfHeldStageChanged(
                        'special object inside private HF stage: ' + relative)

    walk(stage)
    if seen_files != set(expected_files) or seen_dirs != expected_dirs:
        raise HfHeldStageChanged('private HF stage selected paths are missing')
    _assert_owned_root(stage, root_id)


def _audit_hf_bytes(stage, expected, max_total_bytes):
    try:
        actual = hf_closure.scan_selected_hf_source(
            stage, expected_model_key=expected['model_key'],
            max_total_bytes=max_total_bytes)
    except (OSError, ValueError, TypeError, hf_closure.HfClosureError) as error:
        raise HfHeldStageChanged(
            'private HF stage selected data invalid: %s' % error) from error
    if (actual['tree_sha256'] != expected['tree_sha256'] or
            actual['entries'] != expected['entries'] or
            actual['layout'] != expected['layout'] or
            actual['file_count'] != expected['file_count'] or
            actual['total_bytes'] != expected['total_bytes']):
        raise HfHeldStageChanged(
            'private HF stage selected SHA/tree differs from approved source')


def _hash_win32_handle(api, handle, *, block_bytes, expected_bytes):
    if not api.kernel.SetFilePointerEx(
            handle, 0, None, source_lock.FILE_BEGIN):
        raise HfHeldStageChanged(
            'Win32 HF handle seek failed winerror=%d' % ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(block_bytes)
    count = api.wt.DWORD()
    digest = hashlib.sha256()
    total = 0
    while True:
        if not api.kernel.ReadFile(handle, ctypes.byref(buffer), block_bytes,
                                   ctypes.byref(count), None):
            raise HfHeldStageChanged(
                'Win32 HF handle read failed winerror=%d' %
                ctypes.get_last_error())
        if not count.value:
            break
        total += count.value
        if total > expected_bytes:
            raise HfHeldStageChanged(
                'selected HF handle exceeded approved byte count')
        digest.update(buffer.raw[:count.value])
    if total != expected_bytes:
        raise HfHeldStageChanged(
            'selected HF handle differs from approved byte count')
    return digest.hexdigest(), total


def _native_check(api, path, handle, identity, size, sha, block_bytes):
    if api.identity(handle) != identity:
        raise HfHeldStageChanged(
            'selected HF Win32 FileID/ChangeTime changed while held: ' + str(path))
    try:
        source_lock._selected_path_matches_handle(path, identity, size)
    except (ValueError, OSError, source_lock.SourceLockError) as error:
        raise HfHeldStageChanged(
            'selected HF staged pathname no longer matches native FileID: %s'
            % error) from error
    actual, amount = _hash_win32_handle(
        api, handle, block_bytes=block_bytes, expected_bytes=size)
    if amount != size or actual != sha:
        raise HfHeldStageChanged(
            'selected HF private stage SHA changed while locked: ' + str(path))
    if api.identity(handle) != identity:
        raise HfHeldStageChanged(
            'selected HF Win32 metadata changed during SHA read: ' + str(path))


def _required_stage_space(parent, files, directories):
    from ctypes import wintypes

    disk = ctypes.WinDLL('kernel32', use_last_error=True)
    disk.GetDiskFreeSpaceW.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD)]
    disk.GetDiskFreeSpaceW.restype = wintypes.BOOL
    sectors = wintypes.DWORD()
    sector_bytes = wintypes.DWORD()
    free_clusters = wintypes.DWORD()
    total_clusters = wintypes.DWORD()
    if not disk.GetDiskFreeSpaceW(
            parent.drive + '\\', ctypes.byref(sectors),
            ctypes.byref(sector_bytes), ctypes.byref(free_clusters),
            ctypes.byref(total_clusters)):
        raise HfHeldStageUnavailable(
            'cannot measure NTFS cluster size for HF stage disk preflight: '
            'winerror=%d' % ctypes.get_last_error())
    cluster = sectors.value * sector_bytes.value
    if not 512 <= cluster <= 64 * 1024 * 1024:
        raise HfHeldStageUnavailable(
            'NTFS cluster-size measurement is implausible; refuse HF stage')
    allocated_files = sum(
        ((size + cluster - 1) // cluster) * cluster
        for _, size, _ in files)


    return (allocated_files + (len(directories) + 1) * cluster +
            MIN_DISK_HEADROOM)


@contextmanager
def hold_private_selected_hf_stage(source_directory, *, stage_parent,
                                   expected_tree_sha256,
                                   expected_model_key=None,
                                   max_total_bytes=MAX_HF_SELECTED_BYTES,
                                   max_selected_files=MAX_HF_SELECTED_FILES,
                                   block_bytes=MAX_IO_BLOCK_BYTES):
    if not source_lock._is_windows():
        raise HfHeldStageUnavailable(
            'native Win32 NTFS required for HF staged handles; no WSL/fallback')
    _limits(max_total_bytes, max_selected_files, block_bytes)
    approved_sha = _approved_sha(expected_tree_sha256)
    api = source_lock._Kernel32()
    original = _directory(source_directory, 'HF source directory', api)
    parent = _directory(stage_parent, 'HF private stage parent', api)
    if original == parent or original in parent.parents:
        raise HfHeldStageError(
            'private HF stage parent cannot be inside the original source')
    try:
        first = hf_closure.scan_selected_hf_source(
            original, expected_model_key=expected_model_key,
            max_total_bytes=max_total_bytes)
    except hf_closure.HfClosureError as error:
        raise HfHeldStageError(
            'unsupported/over-budget selected HF source: %s' % error) from error
    if first['tree_sha256'] != approved_sha:
        raise HfHeldStageChanged('original HF selected source differs from approved SHA')
    files = _select_files(
        first, max_total_bytes=max_total_bytes,
        max_selected_files=max_selected_files)
    directories = _directories(files)
    required = _required_stage_space(parent, files, directories)
    if shutil.disk_usage(parent).free < required:
        raise HfHeldStageUnavailable(
            'not enough free NTFS space for cluster-allocated HF stage '
            'and retained-copy headroom')


    stage = Path(tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=parent))
    root_id = _object_id(source_freeze._lstat(stage))
    for relative in directories:
        _assert_owned_root(stage, root_id)
        folder = stage.joinpath(*relative.split('/'))
        source_freeze._safe_ancestors(folder, leaf_may_be_missing=True)
        folder.mkdir(exist_ok=False)
    for relative, size, sha in files:
        _assert_owned_root(stage, root_id)
        _copy_selected_file(original, stage, relative, size, sha, block_bytes)


    try:
        second = hf_closure.scan_selected_hf_source(
            original, expected_model_key=first['model_key'],
            max_total_bytes=max_total_bytes)
    except hf_closure.HfClosureError as error:
        raise HfHeldStageChanged(
            'original selected HF source changed during private copy: %s'
            % error) from error
    if second['tree_sha256'] != approved_sha or second['entries'] != first['entries']:
        raise HfHeldStageChanged(
            'original HF selected bytes changed during private snapshot')
    _audit_exact_private_tree(stage, root_id, files, directories)
    _audit_hf_bytes(stage, first, max_total_bytes)

    with ExitStack() as handles:
        opened = []
        for relative, size, sha in files:
            path = stage.joinpath(*relative.split('/'))
            _assert_owned_root(stage, root_id)
            api.assert_local_ntfs(path)
            handle = api.open(path)
            handles.callback(api.close, handle)
            identity = api.identity(handle)
            if identity.file_bytes != size:
                raise HfHeldStageChanged(
                    'selected HF native file differs in size: ' + relative)
            _native_check(api, path, handle, identity, size, sha, block_bytes)
            opened.append((path, handle, identity, size, sha))

        _audit_exact_private_tree(stage, root_id, files, directories)
        _audit_hf_bytes(stage, first, max_total_bytes)
        for path, handle, identity, size, sha in opened:
            _native_check(api, path, handle, identity, size, sha, block_bytes)

        ticket = HeldHfStage(
            staged_directory=str(stage), compiler_input=str(stage),
            source_tree_sha256=approved_sha,
            model_key=first['model_key'], file_count=len(files),
            total_bytes=first['total_bytes'])
        try:
            yield ticket
        except BaseException:
            raise
        else:
            for path, handle, identity, size, sha in opened:
                _native_check(api, path, handle, identity, size, sha, block_bytes)
            _audit_exact_private_tree(stage, root_id, files, directories)
            _audit_hf_bytes(stage, first, max_total_bytes)


def run_with_private_selected_hf_stage(source_directory, *, stage_parent,
                                       expected_tree_sha256, child,
                                       expected_model_key=None,
                                       max_total_bytes=MAX_HF_SELECTED_BYTES,
                                       max_selected_files=MAX_HF_SELECTED_FILES,
                                       block_bytes=MAX_IO_BLOCK_BYTES):
    if not callable(child):
        raise HfHeldStageError('an injected callable HF child is required')
    with hold_private_selected_hf_stage(
            source_directory, stage_parent=stage_parent,
            expected_tree_sha256=expected_tree_sha256,
            expected_model_key=expected_model_key,
            max_total_bytes=max_total_bytes,
            max_selected_files=max_selected_files,
            block_bytes=block_bytes) as ticket:
        return child(ticket)
