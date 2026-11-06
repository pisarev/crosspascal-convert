# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Experimental staging with Win32-held handles for selected Paddle inputs.

Only explicitly selected inference graph/weights/metadata plus a caller-chosen
recognition dictionary are copied. WSL can be given the yielded private path,
but a real DrvFs/paddle2onnx compatibility run is still REQUIRED before this
is allowed to claim a supported compiler-window integration. No unlocked
WSL/POSIX fallback, no default public wizard switch, and no numerical oracle.

SECURITY: The private tree is ALWAYS RETAINED after child access or error.
FileID proves identity but never permits recursive deletion: the child might
have added or changed owner bytes. Original models/dictionaries remain intact.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile

from . import hf_held_stage, paddle_closure, source_freeze, source_lock


MAX_SELECTED_BYTES = paddle_closure.MAX_TOTAL_BYTES
MAX_SELECTED_FILES = paddle_closure.MAX_SELECTED_FILES
MAX_IO_BLOCK_BYTES = paddle_closure.MAX_BLOCK_BYTES
_PREFIX = '.crosspascal-paddle-held-'


class PaddleHeldStageError(ValueError):
    """Selected Paddle source cannot be protected by the bounded native pilot."""


class PaddleHeldStageUnavailable(PaddleHeldStageError):
    """Local Win32 NTFS/space/native handle guarantee is unavailable."""


class PaddleHeldStageChanged(PaddleHeldStageError):
    """Selected original or private staged bytes/identities changed."""


@dataclass(frozen=True)
class HeldPaddleStage:
    """Use compiler_input only while context keeps every selected file held."""

    staged_directory: str
    compiler_input: str
    staged_dictionary: str | None
    source_tree_sha256: str
    role: str
    graph_file: str
    file_count: int
    total_bytes: int
    status: str = 'WIN32_PADDLE_SELECTED_HELD_CANDIDATE_NOT_RELEASE_GRADE'
    verification_status: str = 'NOT_CHECKED'
    release_grade_immutable: bool = False
    retained_after_context: bool = True


def _approved_report(directory, *, role, dictionary, expected_tree_sha256):
    try:
        return paddle_closure.audit_selected_paddle_source(
            directory, role=role, dictionary=dictionary,
            expected_tree_sha256=expected_tree_sha256)
    except (OSError, ValueError, TypeError, paddle_closure.PaddleClosureError) as error:
        raise PaddleHeldStageChanged(
            'original selected Paddle source/dictionary differs from approval: %s'
            % error) from error


def _inventory(report):
    rows = report['entries']
    if (len(rows) == 0 or len(rows) > MAX_SELECTED_FILES or
            len(rows) != report['file_count'] or
            report['total_bytes'] > MAX_SELECTED_BYTES):
        raise PaddleHeldStageError('selected Paddle file/count/byte bound exceeded')
    files = []
    total = 0
    for row in rows:
        rel = hf_held_stage.hf_closure._relative(row['path'])
        size, sha = row['bytes'], hf_held_stage._approved_sha(row['sha256'])
        if type(size) is not int or size <= 0:
            raise PaddleHeldStageError('invalid selected Paddle file size')
        files.append((rel, size, sha))
        total += size
    if total != report['total_bytes']:
        raise PaddleHeldStageError('Paddle selected aggregate byte count changed')
    expected_dirs = ('dictionary',) if report['role'] == 'rec' else ()
    if hf_held_stage._directories(files) != expected_dirs:
        raise PaddleHeldStageError('Paddle staged file dependencies are unclosed')
    return tuple(files), expected_dirs


def _source_path(source, report, relative):
    selected_dictionary = report['dictionary']
    if selected_dictionary and relative == selected_dictionary['path']:
        return Path(selected_dictionary['reference'])
    if '/' in relative:
        raise PaddleHeldStageError('unapproved nested Paddle source file')
    return source / relative


def _copy_one(origin, destination, *, size, digest, block_bytes):
    info = paddle_closure._checked_file(origin, limit=size)
    if info.st_size != size:
        raise PaddleHeldStageChanged('source Paddle asset size changed before copy')
    source_freeze._safe_ancestors(destination, leaf_may_be_missing=True)


    flags = (os.O_RDONLY | getattr(os, 'O_BINARY', 0) |
             getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    fd = os.open(origin, flags)
    received = 0
    sha = hashlib.sha256()
    with os.fdopen(fd, 'rb') as src:
        if (paddle_closure._descriptor_identity(os.fstat(src.fileno())) !=
                paddle_closure._descriptor_identity(info)):
            raise PaddleHeldStageChanged('source identity changed before Paddle copy')
        with destination.open('xb') as dest:
            while True:
                piece = src.read(block_bytes)
                if not piece:
                    break
                received += len(piece)
                if received > size:
                    raise PaddleHeldStageChanged('Paddle source grew during copy')
                sha.update(piece)
                dest.write(piece)
            dest.flush()
            os.fsync(dest.fileno())
        if (paddle_closure._descriptor_identity(os.fstat(src.fileno())) !=
                paddle_closure._descriptor_identity(info)):
            raise PaddleHeldStageChanged('Paddle source metadata changed during copy')
    if received != size or sha.hexdigest() != digest:
        raise PaddleHeldStageChanged('selected Paddle original SHA differs from approval')
    if paddle_closure._identity(paddle_closure._checked_file(origin, limit=size)) != (
            paddle_closure._identity(info)):
        raise PaddleHeldStageChanged('selected Paddle path changed during copy')


def _audit_private(stage, root_id, files, directories, original):
    try:
        hf_held_stage._audit_exact_private_tree(stage, root_id, files, directories)
    except hf_held_stage.HfHeldStageError as error:
        raise PaddleHeldStageChanged(
            'private Paddle selected tree is not intact: %s' % error) from error
    staged_dict = (stage / 'dictionary' /
                   Path(original['dictionary']['reference']).name
                   if original['dictionary'] else None)

    try:
        actual = paddle_closure.scan_selected_paddle_source(
            stage, role=original['role'], dictionary=staged_dict)
    except (OSError, ValueError, paddle_closure.PaddleClosureError) as error:
        raise PaddleHeldStageChanged(
            'private Paddle copy cannot be revalidated: %s' % error) from error
    if (actual['entries'] != original['entries'] or
            actual['graph_file'] != original['graph_file'] or
            actual['file_count'] != original['file_count'] or
            actual['total_bytes'] != original['total_bytes'] or
            (actual['dictionary'] or {}).get('sha256') !=
            (original['dictionary'] or {}).get('sha256')):
        raise PaddleHeldStageChanged('private Paddle bytes/graph/dictionary SHA changed')
    try:
        hf_held_stage._assert_owned_root(stage, root_id)
    except hf_held_stage.HfHeldStageError as error:
        raise PaddleHeldStageChanged(
            'private Paddle root changed during audit: %s' % error) from error
    return staged_dict


@contextmanager
def hold_private_selected_paddle_stage(source_directory, *, stage_parent,
                                       role, dictionary=None,
                                       expected_tree_sha256,
                                       block_bytes=MAX_IO_BLOCK_BYTES):
    """Hold all chosen staged Paddle inputs for an injected converter callback.

    Original bytes are twice rescanned; selected staged files are opened with
    CreateFileW(GENERIC_READ, FILE_SHARE_READ), read back and held until child
    exit. The source AND staged tree are rechecked on successful exit.
    No stage directory is recursively cleaned on either success or failure.
    """
    if not source_lock._is_windows():
        raise PaddleHeldStageUnavailable(
            'native Windows local NTFS required, no unlocked WSL/POSIX fallback')
    if type(block_bytes) is not int or not 1 <= block_bytes <= MAX_IO_BLOCK_BYTES:
        raise PaddleHeldStageError('Paddle stage I/O block must be 1..1048576')
    api = source_lock._Kernel32()
    try:
        source = hf_held_stage._directory(
            source_directory, 'Paddle inference source', api)
        parent = hf_held_stage._directory(
            stage_parent, 'Paddle private stage parent', api)
    except hf_held_stage.HfHeldStageError as error:
        raise PaddleHeldStageUnavailable(str(error)) from error
    if source == parent or source in parent.parents:
        raise PaddleHeldStageError('Paddle private stage parent inside original source')
    if dictionary is not None:
        dict_path = Path(dictionary)
        if dict_path.parent == parent or dict_path.parent in parent.parents:
            raise PaddleHeldStageError(
                'Paddle stage parent inside original dictionary directory')
    original = _approved_report(
        source, role=role, dictionary=dictionary,
        expected_tree_sha256=expected_tree_sha256)
    files, directories = _inventory(original)
    try:
        needed = hf_held_stage._required_stage_space(parent, files, directories)
    except hf_held_stage.HfHeldStageError as error:
        raise PaddleHeldStageUnavailable(str(error)) from error
    if shutil.disk_usage(parent).free < needed:
        raise PaddleHeldStageUnavailable(
            'insufficient free NTFS space for cluster-allocated retained Paddle stage')

    stage = Path(tempfile.mkdtemp(prefix=_PREFIX, dir=parent))
    root_id = hf_held_stage._object_id(source_freeze._lstat(stage))
    for directory in directories:
        hf_held_stage._assert_owned_root(stage, root_id)
        folder = stage.joinpath(*directory.split('/'))
        source_freeze._safe_ancestors(folder, leaf_may_be_missing=True)
        folder.mkdir(exist_ok=False)
    for relative, size, sha in files:
        hf_held_stage._assert_owned_root(stage, root_id)
        _copy_one(_source_path(source, original, relative),
                  stage.joinpath(*relative.split('/')),
                  size=size, digest=sha, block_bytes=block_bytes)
    if _approved_report(
            source, role=role, dictionary=dictionary,
            expected_tree_sha256=expected_tree_sha256) != original:
        raise PaddleHeldStageChanged('selected Paddle source changed while staging')
    staged_dictionary = _audit_private(
        stage, root_id, files, directories, original)
    with ExitStack() as stack:
        opened = []
        for relative, size, sha in files:
            path = stage.joinpath(*relative.split('/'))
            hf_held_stage._assert_owned_root(stage, root_id)
            api.assert_local_ntfs(path)
            handle = api.open(path)
            stack.callback(api.close, handle)
            identity = api.identity(handle)
            if identity.file_bytes != size:
                raise PaddleHeldStageChanged(
                    'private Paddle Win32 handle size differs: ' + relative)
            hf_held_stage._native_check(
                api, path, handle, identity, size, sha, block_bytes)
            opened.append((path, handle, identity, size, sha))
        _audit_private(stage, root_id, files, directories, original)
        ticket = HeldPaddleStage(
            staged_directory=str(stage), compiler_input=str(stage),
            staged_dictionary=str(staged_dictionary) if staged_dictionary else None,
            source_tree_sha256=original['tree_sha256'],
            role=role, graph_file=original['graph_file'],
            file_count=original['file_count'], total_bytes=original['total_bytes'])
        yield ticket
        for path, handle, identity, size, sha in opened:
            hf_held_stage._native_check(
                api, path, handle, identity, size, sha, block_bytes)
        _audit_private(stage, root_id, files, directories, original)
        if _approved_report(
                source, role=role, dictionary=dictionary,
                expected_tree_sha256=expected_tree_sha256) != original:
            raise PaddleHeldStageChanged('selected Paddle original changed during child')


def run_with_private_selected_paddle_stage(source_directory, *, stage_parent,
                                           role, dictionary,
                                           expected_tree_sha256, child):
    """Synchronous child callback with native Win32 held handles; never VERIFIED."""
    if not callable(child):
        raise PaddleHeldStageError('explicit callable child required')
    with hold_private_selected_paddle_stage(
            source_directory, stage_parent=stage_parent, role=role,
            dictionary=dictionary,
            expected_tree_sha256=expected_tree_sha256) as ticket:
        return child(ticket)
