# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from . import onnx_closure, source_freeze

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_ATTRIBUTE_DIRECTORY = 0x00000010
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
FILE_BEGIN = 0
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
MAX_STAGED_BYTES = 64 * 1024 * 1024
MAX_READ_BLOCK = 1024 * 1024


class SourceLockError(ValueError):
    pass


class SourceLockUnavailable(SourceLockError):
    pass


class StagedSourceInvalid(SourceLockError):
    pass


@dataclass(frozen=True)
class LockedSource:
    compiler_input: str
    source_tree_sha256: str
    staged_file_sha256: str
    bytes: int
    status: str = 'WIN32_READ_SHARE_ONLY_CANDIDATE_NOT_RELEASE_GRADE'
    release_grade_immutable: bool = False


@dataclass(frozen=True)
class LockedFileSet:
    compiler_input: str
    source_tree_sha256: str
    staged_model_sha256: str
    bytes: int
    file_count: int
    status: str = 'WIN32_SELECTED_FILE_SET_CANDIDATE_NOT_RELEASE_GRADE'
    release_grade_immutable: bool = False


class _HandleIdentity(NamedTuple):
    volume_serial: int
    file_id: int
    file_bytes: int
    last_write_100ns: int
    change_time_100ns: int
    attributes: int


def _is_windows():
    return os.name == 'nt' and sys.platform.startswith('win')


def _trusted_tasklist_path():
    if not _is_windows():
        raise SourceLockUnavailable("Win32 required for trusted process inventory")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_system_dir = kernel.GetSystemDirectoryW
    get_system_dir.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    get_system_dir.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    count = get_system_dir(buffer, len(buffer))
    if not 0 < count < len(buffer):
        raise SourceLockUnavailable("cannot resolve trusted Win32 system directory")
    candidate = Path(buffer.value) / "tasklist.exe"
    source_freeze._safe_ancestors(candidate)
    if not stat.S_ISREG(source_freeze._lstat(candidate).st_mode):
        raise SourceLockUnavailable("trusted Windows tasklist executable missing")
    return candidate


def _approved_sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in '0123456789abcdefABCDEF' for ch in value)):
        raise StagedSourceInvalid(
            'an exact previously approved 64-digit source-tree SHA-256 is required')
    return value.lower()


def _regular_file(path):
    try:
        source_freeze._safe_ancestors(path)
        info = source_freeze._lstat(path)
    except (OSError, source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'unsafe linked, redirected or absent snapshot path: %s' % error
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise StagedSourceInvalid('staged RAW source must be one regular file')
    if not 0 < info.st_size <= MAX_STAGED_BYTES:
        raise StagedSourceInvalid('single-file RAW model exceeds 64 MiB limit')
    if path.suffix.lower() not in ('.onnx', '.tflite'):
        raise StagedSourceInvalid(
            'only local single-file ONNX/TFLite snapshots are supported')
    return info


def _ready_marker(snapshot):
    marker = snapshot / source_freeze.READY_MARKER
    try:
        info = source_freeze._lstat(marker)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 8 * 1024 * 1024:
            raise StagedSourceInvalid('snapshot ready marker is empty/oversized')
        body = marker.read_bytes()
        if len(body) != info.st_size:
            raise StagedSourceInvalid('snapshot marker changed while reading')
        record = json.loads(body.decode('utf-8'))
        if not isinstance(record, dict):
            raise StagedSourceInvalid(
                'snapshot ready marker must be a JSON object')
    except (OSError, UnicodeError, ValueError, TypeError,
            source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'snapshot ready marker is unavailable or malformed: %s' % error
        ) from error
    return record, hashlib.sha256(body).hexdigest()


def _audit(snapshot, expected_tree_sha256):
    try:
        accepted = source_freeze.audit_frozen_source(snapshot)
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            IndexError, UnicodeError,
            source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'source-freeze snapshot byte/marker check refused: %s' % error
        ) from error
    if (accepted.get('status') != 'COPIED_RECHECKED_NOT_IMMUTABLE'
            or accepted.get('source_tree_sha256') != expected_tree_sha256
            or accepted.get('snapshot_tree_sha256') != expected_tree_sha256):
        raise StagedSourceInvalid(
            'snapshot differs from independently approved Preview source-tree SHA')
    path = Path(accepted['compiler_input'])
    if path.parent != snapshot:
        raise StagedSourceInvalid(
            'source-freeze snapshot does not hold exactly one direct RAW file')
    info = _regular_file(path)
    manifest, marker_digest = _ready_marker(snapshot)
    entries = manifest.get('entries')
    if (manifest.get('source_kind') != 'file' or
            manifest.get('snapshot_input') != path.name or
            not isinstance(entries, list) or len(entries) != 1 or
            not isinstance(entries[0], dict) or
            entries[0].get('kind') != 'file' or
            entries[0].get('path') != path.name or
            entries[0].get('bytes') != info.st_size or
            manifest.get('source_tree_sha256') != expected_tree_sha256):
        raise StagedSourceInvalid('incorrect single-file staged ready marker')
    return path, _approved_sha(entries[0].get('sha256')), marker_digest


class _Kernel32:
    def __init__(self):
        if not _is_windows():
            raise SourceLockUnavailable(
                'Win32 read/share-only locking requires native Windows; '
                'no POSIX/advisory/unlocked fallback is allowed')
        from ctypes import wintypes

        class FileInfo(ctypes.Structure):
            _fields_ = [
                ('dwFileAttributes', wintypes.DWORD),
                ('ftCreationTime', wintypes.FILETIME),
                ('ftLastAccessTime', wintypes.FILETIME),
                ('ftLastWriteTime', wintypes.FILETIME),
                ('dwVolumeSerialNumber', wintypes.DWORD),
                ('nFileSizeHigh', wintypes.DWORD),
                ('nFileSizeLow', wintypes.DWORD),
                ('nNumberOfLinks', wintypes.DWORD),
                ('nFileIndexHigh', wintypes.DWORD),
                ('nFileIndexLow', wintypes.DWORD),
            ]

        class BasicInfo(ctypes.Structure):
            _fields_ = [
                ('CreationTime', ctypes.c_longlong),
                ('LastAccessTime', ctypes.c_longlong),
                ('LastWriteTime', ctypes.c_longlong),
                ('ChangeTime', ctypes.c_longlong),
                ('FileAttributes', wintypes.DWORD),
            ]

        self.FileInfo = FileInfo
        self.BasicInfo = BasicInfo
        self.wt = wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(FileInfo)]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel.SetFilePointerEx.argtypes = [
            wintypes.HANDLE, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
        kernel.SetFilePointerEx.restype = wintypes.BOOL
        kernel.ReadFile.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        kernel.ReadFile.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        kernel.GetDriveTypeW.restype = wintypes.UINT
        kernel.GetVolumeInformationW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR, wintypes.DWORD,
        ]
        kernel.GetVolumeInformationW.restype = wintypes.BOOL
        self.kernel = kernel

    def assert_local_ntfs(self, path):
        drive = Path(path).drive
        if len(drive) != 2 or drive[1] != ':':
            raise SourceLockUnavailable(
                'UNC/network/nonlocal staged paths are outside native Win32 '
                'share-read candidate scope')
        volume_root = drive + '\\'

        if self.kernel.GetDriveTypeW(volume_root) != 3:
            raise SourceLockUnavailable(
                'source-lock staging requires a local fixed NTFS volume')
        filesystem = ctypes.create_unicode_buffer(64)
        if not self.kernel.GetVolumeInformationW(
                volume_root, None, 0, None, None, None,
                filesystem, len(filesystem)):
            raise SourceLockUnavailable(
                'GetVolumeInformationW failed winerror=%d; '
                'filesystem safety was not proven' % ctypes.get_last_error())
        if filesystem.value.upper() != 'NTFS':
            raise SourceLockUnavailable(
                'unverified filesystem %r: only local fixed NTFS is supported'
                % filesystem.value)

    def open(self, path):
        handle = self.kernel.CreateFileW(
            str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, None)
        if handle == INVALID_HANDLE_VALUE:
            raise SourceLockUnavailable(
                'CreateFileW(GENERIC_READ,FILE_SHARE_READ) denied: '
                'winerror=%d; preexisting writer/unsupported permissions; '
                'never fall back to an unlocked compiler'
                % ctypes.get_last_error())
        return handle

    def identity(self, handle):
        info = self.FileInfo()
        if not self.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise StagedSourceInvalid(
                'GetFileInformationByHandle failed winerror=%d' %
                ctypes.get_last_error())
        basic = self.BasicInfo()
        if not self.kernel.GetFileInformationByHandleEx(
                handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)):
            raise StagedSourceInvalid(
                'GetFileInformationByHandleEx(FileBasicInfo) failed winerror=%d'
                % ctypes.get_last_error())
        flags = info.dwFileAttributes | basic.FileAttributes
        if flags & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT):
            raise StagedSourceInvalid(
                'open handle describes a directory or reparse target')
        return _HandleIdentity(
            info.dwVolumeSerialNumber,
            (info.nFileIndexHigh << 32) | info.nFileIndexLow,
            (info.nFileSizeHigh << 32) | info.nFileSizeLow,
            basic.LastWriteTime, basic.ChangeTime, flags)

    def hash_bytes(self, handle, *, block_bytes):
        if not self.kernel.SetFilePointerEx(handle, 0, None, FILE_BEGIN):
            raise StagedSourceInvalid(
                'Win32 seek failed winerror=%d' % ctypes.get_last_error())
        buf = ctypes.create_string_buffer(block_bytes)
        count = self.wt.DWORD()
        result = hashlib.sha256()
        total = 0
        while True:
            if not self.kernel.ReadFile(
                    handle, ctypes.byref(buf), block_bytes,
                    ctypes.byref(count), None):
                raise StagedSourceInvalid(
                    'ReadFile failed winerror=%d' % ctypes.get_last_error())
            if not count.value:
                break
            total += count.value
            if total > MAX_STAGED_BYTES:
                raise StagedSourceInvalid('locked file exceeded 64 MiB')
            result.update(buf.raw[:count.value])
        return result.hexdigest(), total

    def close(self, handle):
        if not self.kernel.CloseHandle(handle):
            raise SourceLockError(
                'CloseHandle failed winerror=%d' % ctypes.get_last_error())


def _path_matches_handle(path, identity):
    current = _regular_file(path)
    if (current.st_ino != identity.file_id or
            current.st_size != identity.file_bytes):
        raise StagedSourceInvalid(
            'staged path inode/size differs from locked native file identity')


@contextmanager
def hold_staged_single_file(snapshot_directory, *, expected_tree_sha256,
                            block_bytes=MAX_READ_BLOCK):
    if not _is_windows():
        raise SourceLockUnavailable(
            'native Windows is required; no unlocked or WSL fallback')
    approved = _approved_sha(expected_tree_sha256)
    if type(block_bytes) is not int or not 1 <= block_bytes <= MAX_READ_BLOCK:
        raise SourceLockError('bounded block_bytes must be 1..1048576')
    try:
        root = source_freeze._absolute_path(
            snapshot_directory, 'staged source root')
        source_freeze._safe_ancestors(root)
        if not stat.S_ISDIR(source_freeze._lstat(root).st_mode):
            raise StagedSourceInvalid('snapshot root must be a real directory')
    except (OSError, source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'unsafe/missing snapshot directory: %s' % error) from error
    path, sha, marker_hash = _audit(root, approved)
    api = _Kernel32()
    api.assert_local_ntfs(path)
    handle = api.open(path)
    try:
        before = api.identity(handle)
        _path_matches_handle(path, before)
        if before.file_bytes > MAX_STAGED_BYTES:
            raise StagedSourceInvalid('oversized staged file')
        digest, size = api.hash_bytes(handle, block_bytes=block_bytes)
        if digest != sha or size != before.file_bytes:
            raise StagedSourceInvalid(
                'locked Win32 handle contents do not match staged source SHA')
        again_path, again_sha, again_marker = _audit(root, approved)
        if (again_path != path or again_sha != sha or
                again_marker != marker_hash or api.identity(handle) != before):
            raise StagedSourceInvalid(
                'snapshot identity/marker changed while acquiring Win32 lock')
        ticket = LockedSource(str(path), approved, sha, size)
        try:
            yield ticket
        except BaseException:
            raise
        else:
            if api.identity(handle) != before:
                raise StagedSourceInvalid(
                    'Win32 file identity/ChangeTime changed during compiler')
            after_hash, after_size = api.hash_bytes(
                handle, block_bytes=block_bytes)
            if after_hash != sha or after_size != size:
                raise StagedSourceInvalid(
                    'staged file changed during compiler (possible earlier '
                    'writable memory mapping)')
            after_path, after_sha, after_marker = _audit(root, approved)
            if (after_path != path or after_sha != sha or
                    after_marker != marker_hash):
                raise StagedSourceInvalid(
                    'snapshot path/marker/contents changed during compilation')
            _path_matches_handle(path, before)
    finally:
        api.close(handle)


def _selected_snapshot(root, approved):
    try:
        audited = onnx_closure.audit_selected_closure(
            root, expected_tree_sha256=approved)
        marker, marker_sha = _ready_marker(root)
        if (marker.get('closure_schema') != onnx_closure.SCHEMA or
                marker.get('closure_version') != onnx_closure.SCHEMA_VERSION or
                marker.get('source_kind') != 'directory' or
                marker.get('model_rel') != audited['model_rel'] or
                marker.get('source_tree_sha256') != approved or
                marker.get('snapshot_tree_sha256') != approved):
            raise StagedSourceInvalid(
                'selected ONNX ready marker differs from approved Preview')
        entries = marker.get('entries')
        if not isinstance(entries, list):
            raise StagedSourceInvalid('selected ONNX ready marker lacks entries')
        members = []
        total = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise StagedSourceInvalid('selected ONNX stage has malformed member')
            if entry.get('kind') != 'file':
                if entry.get('kind') != 'directory':
                    raise StagedSourceInvalid('unexpected staged member kind')
                continue
            relative = onnx_closure._relative(
                entry.get('path'), 'selected staged member')
            size = entry.get('bytes')
            if (type(size) is not int or not 0 < size <= MAX_STAGED_BYTES):
                raise StagedSourceInvalid('selected member has invalid file size')
            total += size
            if total > MAX_STAGED_BYTES:
                raise StagedSourceInvalid('selected stage exceeds 64 MiB aggregate')
            digest = _approved_sha(entry.get('sha256'))
            members.append((relative, root.joinpath(*relative.split('/')),
                            size, digest))
        members.sort(key=lambda row: row[0])
        if (not members or len(members) > onnx_closure.MAX_FILES or
                len(members) != audited['file_count'] or
                total != audited['total_bytes'] or
                total != marker.get('total_bytes')):
            raise StagedSourceInvalid(
                'selected file-set count/size differs from approved stage')
        if str(root.joinpath(*audited['model_rel'].split('/'))) != (
                audited['compiler_input']):
            raise StagedSourceInvalid('selected stage compiler input differs')
        if audited['model_rel'] not in {row[0] for row in members}:
            raise StagedSourceInvalid('selected ONNX file absent from stage')
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'selected ONNX snapshot/marker differs from approved Preview SHA: %s'
            % error) from error
    return audited, tuple(members), marker_sha


def _selected_path_matches_handle(path, identity, expected_size):
    try:
        source_freeze._safe_ancestors(path)
        info = source_freeze._lstat(path)
    except (OSError, source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'unsafe/replaced selected staged path: %s' % error) from error
    if (not stat.S_ISREG(info.st_mode) or
            info.st_ino != identity.file_id or
            info.st_size != identity.file_bytes or
            info.st_size != expected_size):
        raise StagedSourceInvalid(
            'selected staged file path/size differs from Win32 file identity')


@contextmanager
def hold_staged_file_set(snapshot_directory, *, expected_tree_sha256,
                         block_bytes=MAX_READ_BLOCK):
    if not _is_windows():
        raise SourceLockUnavailable(
            'selected file-set locking requires native Windows; '
            'no advisory, WSL or unlocked fallback')
    approved = _approved_sha(expected_tree_sha256)
    if type(block_bytes) is not int or not 1 <= block_bytes <= MAX_READ_BLOCK:
        raise SourceLockError('bounded block_bytes must be 1..1048576')
    try:
        root = source_freeze._absolute_path(
            snapshot_directory, 'selected staged source root')
        source_freeze._safe_ancestors(root)
        if not stat.S_ISDIR(source_freeze._lstat(root).st_mode):
            raise StagedSourceInvalid('selected staged root must be a directory')
    except (OSError, source_freeze.SourceFreezeError) as error:
        raise StagedSourceInvalid(
            'unsafe/missing selected ONNX staged root: %s' % error) from error

    before, files, marker_sha = _selected_snapshot(root, approved)
    api = _Kernel32()
    with ExitStack() as handles:
        opened = []
        for relative, path, expected_size, expected_sha in files:
            api.assert_local_ntfs(path)


            handle = api.open(path)
            handles.callback(api.close, handle)
            identity = api.identity(handle)
            _selected_path_matches_handle(path, identity, expected_size)
            actual_sha, actual_size = api.hash_bytes(
                handle, block_bytes=block_bytes)
            if actual_sha != expected_sha or actual_size != expected_size:
                raise StagedSourceInvalid(
                    'locked selected ONNX/sidecar SHA-256 or byte count differs '
                    'from ready marker: ' + relative)
            opened.append((path, handle, identity, expected_sha, expected_size))


        acquired, entries, acquired_marker = _selected_snapshot(root, approved)
        if (acquired != before or entries != files or
                acquired_marker != marker_sha):
            raise StagedSourceInvalid(
                'selected ONNX stage or marker changed while acquiring handles')
        for path, handle, identity, _, expected_size in opened:
            if api.identity(handle) != identity:
                raise StagedSourceInvalid(
                    'selected Win32 file identity changed during acquisition')
            _selected_path_matches_handle(path, identity, expected_size)

        model_relative = before['model_rel']
        model_hash = next(
            sha for rel, _, _, sha in files if rel == model_relative)
        ticket = LockedFileSet(
            compiler_input=before['compiler_input'],
            source_tree_sha256=approved,
            staged_model_sha256=model_hash,
            bytes=before['total_bytes'],
            file_count=len(files),
        )
        try:
            yield ticket
        except BaseException:
            raise
        else:
            for path, handle, identity, expected_sha, expected_size in opened:
                if api.identity(handle) != identity:
                    raise StagedSourceInvalid(
                        'selected Win32 FileID/ChangeTime changed during compiler')
                actual_sha, actual_size = api.hash_bytes(
                    handle, block_bytes=block_bytes)
                if actual_sha != expected_sha or actual_size != expected_size:
                    raise StagedSourceInvalid(
                        'selected staged ONNX/sidecar bytes changed during compiler')
                _selected_path_matches_handle(path, identity, expected_size)
            after, after_entries, after_marker = _selected_snapshot(root, approved)
            if (after != before or after_entries != files or
                    after_marker != marker_sha):
                raise StagedSourceInvalid(
                    'selected ONNX closure/marker changed during compile/verify')
