# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Standalone, bounded-RAM local model source snapshot CANDIDATE.

The current master.py Preview/Build fingerprint hashes all original source
bytes, but does not prevent a transient edit and restoration WHILE a compiler
reads that original. This opt-in helper copies one local ONNX/TFLite file or a
whole local directory tree into an exclusive, independently hashed directory.
The path handed to a compiler is the COPIED path, never the mutable original.

This helper is deliberately NOT called from master/pipeline. Its completed
marker means only COPIED_AND_RECHECKED, NOT immutable or VERIFIED. The staged
copy can itself be modified after validation; format-specific dependencies
outside the copied directory are not closed. Transient writes during capture,
filesystems without atomic snapshot semantics, and cross-process TOCTOU
remain open. Do not promote the existing strict master XFAIL into a PASS.

No deletion, overwrite, reuse or cleanup of any PREEXISTING user file occurs.
An interrupted copy may leave its OWN new partial directory without a ready
marker, preventing that path from being treated as a finished snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import unicodedata
from pathlib import Path


SCHEMA = 'crosspascal.local-source-freeze'
VERSION = 1
READY_MARKER = 'source-freeze-ready.json'
DEFAULT_BLOCK_BYTES = 1024 * 1024
DEFAULT_MAX_ENTRIES = 50_000
DEFAULT_MAX_BYTES = 16 * 1024**3
DEFAULT_MIN_DISK_HEADROOM = 1024 * 1024
MAX_RELATIVE_PATH_BYTES = 2048
MAX_INVENTORY_METADATA_BYTES = 6 * 1024 * 1024
_DOS_DEVICE = re.compile(r'(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z',
                         re.IGNORECASE)
_HEX = re.compile(r'[0-9a-fA-F]{64}\Z')


class SourceFreezeError(ValueError):
    """This source or destination cannot be captured under the given contract."""


class SourceChanged(SourceFreezeError):
    """Original contents or directory members changed during snapshot creation."""


class SnapshotNotReady(SourceFreezeError):
    """This snapshot is absent, incomplete or was modified since capture."""


def _fail(message):
    raise SourceFreezeError(message)


def _absolute_path(path, label):
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        _fail('%s path must be a nonempty local filesystem path' % label)
    if raw.lower().startswith(('http://', 'https://')):
        _fail('%s must already be a local source (not URL)' % label)
    if '..' in raw.replace('\\', '/').split('/'):
        _fail('%s lexical path traversal (..) is forbidden' % label)
    return Path(os.path.abspath(raw))


def _safe_name(name):
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or len(name) > 255 or name.endswith(('.', ' '))
            or any(ord(ch) < 32 for ch in name)
            or any(ch in name for ch in '<>:"/\\|?*')
            or _DOS_DEVICE.fullmatch(name)):
        _fail('unsafe source relative filename: %r' % name)
    return name


def _sibling_names(entries):
    seen = set()
    for name in entries:
        _safe_name(name)
        key = unicodedata.normalize('NFC', name).casefold()
        if key in seen:
            _fail('case/Unicode-normalization filename collision: %r' % name)
        seen.add(key)


def _lstat(path):
    """Refuse symlinks and NTFS junction/reparse points, never follow them."""
    info = path.lstat()
    attrs = getattr(info, 'st_file_attributes', 0)
    junction = getattr(os.path, 'isjunction', None)
    if (stat.S_ISLNK(info.st_mode)
            or (attrs & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))
            or (junction and junction(str(path)))):
        _fail('symlink/junction/reparse point is forbidden: %s' % path)
    return info


def _safe_ancestors(path, *, leaf_may_be_missing=False):
    current = path
    missing = False
    while True:
        try:
            info = _lstat(current)
        except FileNotFoundError:
            if not (leaf_may_be_missing and current == path and not missing):
                _fail('missing filesystem ancestor: %s' % current)
            missing = True
        else:
            if current != path and not stat.S_ISDIR(info.st_mode):
                _fail('filesystem ancestor is not a directory: %s' % current)
        if current.parent == current:
            break
        current = current.parent


def _identity(info):
    
    return (info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_dev, info.st_ino)


def _descriptor_identity(info):
    
    
    
    return (info.st_mode, info.st_size, info.st_mtime_ns,
            info.st_dev, info.st_ino)


def _limits(max_entries, max_bytes, block_bytes, min_disk_headroom):
    for name, value in (('max_entries', max_entries),
                        ('max_bytes', max_bytes),
                        ('block_bytes', block_bytes),
                        ('min_disk_headroom', min_disk_headroom)):
        if type(value) is not int or value < (0 if name == 'min_disk_headroom' else 1):
            _fail('%s must be an explicit positive integer' % name)
    if block_bytes > DEFAULT_BLOCK_BYTES:
        _fail('block_bytes exceeds bounded-RAM 1 MiB maximum')
    if max_entries > DEFAULT_MAX_ENTRIES or max_bytes > DEFAULT_MAX_BYTES:
        _fail('snapshot scan exceeds the configured safe entry/byte ceiling')


def _hash_file(path, expected, block_bytes):
    """Hash exact bytes on a file descriptor and verify its path identity."""
    sha = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        if _descriptor_identity(os.fstat(stream.fileno())) != _descriptor_identity(expected):
            raise SourceChanged('source changed between inventory and open: %s'
                                % path)
        while True:
            chunk = stream.read(block_bytes)
            if not chunk:
                break
            count += len(chunk)
            sha.update(chunk)
        if _descriptor_identity(os.fstat(stream.fileno())) != _descriptor_identity(expected):
            raise SourceChanged('source changed while hashing: %s' % path)
    if _identity(_lstat(path)) != _identity(expected) or count != expected.st_size:
        raise SourceChanged('source changed during source inventory: %s' % path)
    return sha.hexdigest()


def _scan_source(source, *, max_entries, max_bytes, block_bytes):
    """Complete regular-file tree inventory, with deterministic relative names."""
    _safe_ancestors(source)
    root = _lstat(source)
    if stat.S_ISREG(root.st_mode):
        if source.suffix.lower() not in ('.onnx', '.tflite'):
            _fail('single-file snapshot supports only local ONNX/TFLite')
        _safe_name(source.name)
        source_kind = 'file'
        paths = [(source, source.name, 'file')]
    elif stat.S_ISDIR(root.st_mode):
        source_kind = 'directory'
        paths = []

        def walk(folder, prefix, depth):
            _safe_ancestors(folder)
            _lstat(folder)
            children = sorted(folder.iterdir(), key=lambda p: p.name)
            _sibling_names([item.name for item in children])
            for item in children:
                relative = prefix + item.name
                if (depth >= 64 or
                        len(relative.encode('utf-8', 'surrogatepass'))
                        > MAX_RELATIVE_PATH_BYTES):
                    _fail('relative source model path exceeds safe depth/length bound')
                info = _lstat(item)
                if stat.S_ISDIR(info.st_mode):
                    paths.append((item, relative, 'directory'))
                    if len(paths) > max_entries:
                        _fail('source directory exceeds entry count limit')
                    walk(item, relative + '/', depth + 1)
                elif stat.S_ISREG(info.st_mode):
                    paths.append((item, relative, 'file'))
                    if len(paths) > max_entries:
                        _fail('source directory exceeds entry count limit')
                else:
                    _fail('special source file is forbidden: %s' % item)

        walk(source, '', 0)
    else:
        _fail('source must be regular ONNX/TFLite or a regular directory')
    if not paths or not any(item[2] == 'file' for item in paths):
        _fail('model source directory has no regular files')
    if source_kind == 'directory' and any(
            item[1] == READY_MARKER for item in paths):
        _fail('source root uses reserved snapshot ready-marker filename')
    if len(paths) > max_entries:
        _fail('source directory exceeds entry count limit')
    total = 0
    inventory_bytes = 0
    inventory = []
    for path, relative, kind in sorted(paths, key=lambda p: p[1]):
        
        
        
        path_bytes = len(relative.encode('utf-8', 'surrogatepass'))
        if len(relative.split('/')) > 64 or path_bytes > MAX_RELATIVE_PATH_BYTES:
            _fail('relative source model path exceeds safe depth/length bound')
        inventory_bytes += path_bytes + 180
        if inventory_bytes > MAX_INVENTORY_METADATA_BYTES:
            _fail('source inventory metadata exceeds bounded-RAM manifest size')
        before = _lstat(path)
        if kind == 'directory':
            if not stat.S_ISDIR(before.st_mode):
                raise SourceChanged('source directory changed: %s' % path)
            size, digest = 0, None
        else:
            if not stat.S_ISREG(before.st_mode):
                raise SourceChanged('source file type changed: %s' % path)
            total += before.st_size
            if total > max_bytes:
                _fail('source exceeds snapshot byte limit (%d)' % max_bytes)
            size = before.st_size
            digest = _hash_file(path, before, block_bytes)
        inventory.append({'path': relative.replace('\\', '/'),
                          'kind': kind, 'bytes': size, 'sha256': digest})
    if _identity(_lstat(source)) != _identity(root):
        raise SourceChanged('root source changed during recursive inventory')
    if total == 0:
        _fail('model source contains no nonempty regular file bytes')
    return {'source_kind': source_kind, 'entries': inventory,
            'total_bytes': total}


def _tree_sha(inventory):
    encoded = json.dumps(inventory['entries'], sort_keys=True,
                         ensure_ascii=False, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_content(source, *, max_entries=DEFAULT_MAX_ENTRIES,
                        max_bytes=DEFAULT_MAX_BYTES,
                        block_bytes=DEFAULT_BLOCK_BYTES):
    """Independent Preview-time content pin for a future caller integration.

    This does not replace master._source_fingerprint; that value also hashes
    provenance/NTFS metadata and is a separate, independently checked gate.
    """
    _limits(max_entries, max_bytes, block_bytes, 0)
    origin = _absolute_path(source, 'source')
    listing = _scan_source(origin, max_entries=max_entries, max_bytes=max_bytes,
                           block_bytes=block_bytes)
    return {'tree_sha256': _tree_sha(listing),
            'source_kind': listing['source_kind'],
            'total_bytes': listing['total_bytes'],
            'entries': len(listing['entries'])}


def _free_bytes(path):
    return shutil.disk_usage(path).free


def _copy_stream(source, target, expected_sha, expected_bytes, block_bytes):
    """Exclusive copied file, checked against the first full source SHA."""
    _safe_ancestors(source)
    before = _lstat(source)
    copied = hashlib.sha256()
    count = 0
    with source.open('rb') as origin, target.open('xb') as destination:
        if _descriptor_identity(os.fstat(origin.fileno())) != _descriptor_identity(before):
            raise SourceChanged('source path replaced before copying: %s'
                                % source)
        while True:
            chunk = origin.read(block_bytes)
            if not chunk:
                break
            copied.update(chunk)
            count += len(chunk)
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
        if _descriptor_identity(os.fstat(origin.fileno())) != _descriptor_identity(before):
            raise SourceChanged('source changed while copying: %s' % source)
    if _identity(_lstat(source)) != _identity(before):
        raise SourceChanged('source path replaced during copying: %s' % source)
    if count != expected_bytes or copied.hexdigest() != expected_sha:
        raise SourceChanged('copied source bytes differ from initial SHA: %s'
                            % source)
    actual = _hash_file(target, _lstat(target), block_bytes)
    if actual != expected_sha:
        raise SourceChanged('staged bytes differ from original source SHA: %s'
                            % target)


def _publish_marker(destination, content):
    """Only AFTER verification, exclusive atomic marker link; no old overwrite."""
    path = destination / READY_MARKER
    payload = (json.dumps(content, sort_keys=True, indent=2,
                          ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if len(payload) > 8 * 1024 * 1024:
        _fail('snapshot ready marker exceeds bounded metadata size')
    scratch = destination / ('.source-freeze-' + secrets.token_hex(12) + '.part')
    with scratch.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if scratch.read_bytes() != payload:
        raise SnapshotNotReady('ready-marker scratch bytes disagree after fsync')
    
    os.link(scratch, path)
    
    scratch.unlink()
    return path


def _compare_sources(before, after, *, prefix):
    if before != after:
        raise SourceChanged(prefix + ': source content/relative paths changed')


def _assert_owned_stage(target, identity):
    """Verify the exclusive new staging root has not become a junction/link."""
    _safe_ancestors(target)
    now = _lstat(target)
    if not stat.S_ISDIR(now.st_mode) or (
            now.st_dev, now.st_ino, now.st_mode) != identity:
        raise SourceChanged('exclusive snapshot destination was substituted')


def freeze_source(source, destination, *, expected_tree_sha256=None,
                  expected_source_kind=None,
                  max_entries=DEFAULT_MAX_ENTRIES,
                  max_bytes=DEFAULT_MAX_BYTES,
                  block_bytes=DEFAULT_BLOCK_BYTES,
                  min_disk_headroom=DEFAULT_MIN_DISK_HEADROOM,
                  on_created=None):
    """Create a new no-clobber directory; return its copied compiler input.

    expected_tree_sha256 and expected_source_kind are OPTIONAL preview-time
    content-tree/kind pins, never substitutes for a Preview token or master's
    metadata-sensitive _source_fingerprint. An external caller must keep both
    approved facts bound to its request; this helper is not auto-wired.
    """
    _limits(max_entries, max_bytes, block_bytes, min_disk_headroom)
    origin = _absolute_path(source, 'source')
    target = _absolute_path(destination, 'destination')
    if origin == target:
        _fail('source and snapshot directory cannot be identical')
    _safe_ancestors(origin)
    _safe_ancestors(target, leaf_may_be_missing=True)
    if os.path.lexists(target):
        _fail('destination already exists; no user files can be overwritten')
    if not target.parent.is_dir():
        _fail('destination parent is not a regular directory')
    info = _lstat(origin)
    if stat.S_ISDIR(info.st_mode) and (
            origin in target.parents or target in origin.parents):
        _fail('snapshot directory cannot contain or be contained by source tree')
    first = _scan_source(origin, max_entries=max_entries, max_bytes=max_bytes,
                         block_bytes=block_bytes)
    if expected_source_kind is not None and (
            expected_source_kind != first['source_kind']):
        raise SourceChanged('source changed from approved file/directory kind')
    tree_hash = _tree_sha(first)
    if expected_tree_sha256 is not None:
        if (not isinstance(expected_tree_sha256, str)
                or not _HEX.fullmatch(expected_tree_sha256)):
            _fail('expected_tree_sha256 must be a 64-character hex digest')
        if expected_tree_sha256.lower() != tree_hash:
            raise SourceChanged('initial source tree SHA differs from approved '
                                'content hash')
    required_free = first['total_bytes'] + min_disk_headroom + 4096
    if _free_bytes(target.parent) < required_free:
        _fail('insufficient disk space for complete source copy and headroom')

    
    
    
    target.mkdir(exist_ok=False)
    target_info = _lstat(target)
    stage_identity = (target_info.st_dev, target_info.st_ino,
                      target_info.st_mode)
    if on_created is not None:
        on_created((target_info.st_dev, target_info.st_ino))
    for entry in first['entries']:
        _assert_owned_stage(target, stage_identity)
        relative = entry['path']
        pieces = relative.split('/')
        copied = target.joinpath(*pieces)
        original_file = (origin if first['source_kind'] == 'file'
                         else origin.joinpath(*pieces))
        _safe_ancestors(copied.parent)
        if entry['kind'] == 'directory':
            copied.mkdir(exist_ok=False)
        else:
            _copy_stream(original_file, copied, entry['sha256'],
                         entry['bytes'], block_bytes)
    
    
    second = _scan_source(origin, max_entries=max_entries, max_bytes=max_bytes,
                          block_bytes=block_bytes)
    _compare_sources(first, second, prefix='post-copy recheck')
    staged = _scan_source(target, max_entries=max_entries, max_bytes=max_bytes,
                          block_bytes=block_bytes)
    
    if staged['entries'] != first['entries']:
        raise SourceChanged('private snapshot files are not byte-exact source '
                            'copies (or were changed during validation)')
    _assert_owned_stage(target, stage_identity)
    identity_after = _lstat(target)
    
    
    if (identity_after.st_dev, identity_after.st_ino,
            identity_after.st_mode) != (
            target_info.st_dev, target_info.st_ino, target_info.st_mode):
        raise SourceChanged('snapshot root changed while staging')
    content = {
        'schema': SCHEMA,
        'schema_version': VERSION,
        'status': 'COPIED_RECHECKED_NOT_IMMUTABLE',
        'source_kind': first['source_kind'],
        'source_path': str(origin),
        'snapshot_input': (origin.name if first['source_kind'] == 'file'
                           else '.'),
        'source_tree_sha256': tree_hash,
        'snapshot_tree_sha256': _tree_sha(staged),
        'total_bytes': first['total_bytes'],
        'entries': first['entries'],
        'limitations': [
            'Master Preview/Build does not yet use this source-freeze candidate.',
            'Staging copies and original bytes can change during/after the '
            'last SHA scan; this is not an OS-level immutable snapshot/lock.',
            'External tensors, absolute paths, downloads, plugin-specific '
            'resources and dependencies outside this tree are not frozen.',
            'Ready marker records only a byte-exact scanned copy at one time, '
            'not native build correctness, portability or VERIFIED status.',
        ],
    }
    _assert_owned_stage(target, stage_identity)
    _publish_marker(target, content)
    _assert_owned_stage(target, stage_identity)
    return {'snapshot_directory': str(target),
            'compiler_input': str(target / origin.name) if
            first['source_kind'] == 'file' else str(target),
            'ready_marker': str(target / READY_MARKER),
            'tree_sha256': tree_hash,
            'status': content['status']}


def audit_frozen_source(destination, *, block_bytes=DEFAULT_BLOCK_BYTES):
    """Read-only per-byte revalidation just BEFORE an isolated compiler call.

    It cannot lock the stage through the compiler: a further writer can still
    mutate staged bytes after this audit, which remains a named TOCTOU gap.
    """
    if type(block_bytes) is not int or not 0 < block_bytes <= DEFAULT_BLOCK_BYTES:
        _fail('block_bytes must be within 1..1MiB')
    target = _absolute_path(destination, 'snapshot')
    try:
        _safe_ancestors(target)
        _lstat(target)
        marker = target / READY_MARKER
        marker_info = _lstat(marker)
        if not stat.S_ISREG(marker_info.st_mode):
            raise SnapshotNotReady('snapshot ready marker is not a file')
        if marker_info.st_size > 8 * 1024 * 1024:
            raise SnapshotNotReady('snapshot ready marker exceeds bounded size')
        with marker.open('r', encoding='utf-8') as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError, SourceFreezeError) as exc:
        raise SnapshotNotReady('snapshot has no valid ready marker: %s' % exc) from exc
    if (manifest.get('schema') != SCHEMA or
            manifest.get('schema_version') != VERSION or
            manifest.get('status') != 'COPIED_RECHECKED_NOT_IMMUTABLE'):
        raise SnapshotNotReady('snapshot manifest is not a completed candidate')
    expected = manifest.get('entries')
    if not isinstance(expected, list):
        raise SnapshotNotReady('snapshot manifest has no entries')
    if manifest.get('source_kind') not in ('file', 'directory'):
        raise SnapshotNotReady('snapshot kind is invalid')
    
    
    scanned = _scan_snapshot_without_marker(target, block_bytes)
    if (scanned['entries'] != expected or
            _tree_sha(scanned) != manifest.get('snapshot_tree_sha256') or
            _tree_sha(scanned) != manifest.get('source_tree_sha256')):
        raise SnapshotNotReady('snapshot content/files changed since publication')
    if manifest['source_kind'] == 'file':
        name = _safe_name(manifest.get('snapshot_input'))
        if len(expected) != 1 or expected[0]['path'] != name or (
                expected[0]['kind'] != 'file'):
            raise SnapshotNotReady('single-model snapshot marker filename '
                                   'does not match the frozen file inventory')
        path = target / name
    else:
        if manifest.get('snapshot_input') != '.':
            raise SnapshotNotReady('directory snapshot input path is invalid')
        path = target
    if not path.exists():
        raise SnapshotNotReady('snapshot compiler input is absent')
    return {'compiler_input': str(path),
            'source_tree_sha256': manifest['source_tree_sha256'],
            'snapshot_tree_sha256': _tree_sha(scanned),
            'status': 'COPIED_RECHECKED_NOT_IMMUTABLE'}


def _scan_snapshot_without_marker(target, block_bytes):
    
    root_names = [p.name for p in target.iterdir()]
    if READY_MARKER not in root_names:
        raise SnapshotNotReady('ready marker vanished')
    _sibling_names(root_names)
    
    paths = []

    def walk(folder, prefix, depth=0):
        children = sorted(folder.iterdir(), key=lambda p: p.name)
        _sibling_names([item.name for item in children])
        for item in children:
            if folder == target and item.name == READY_MARKER:
                continue
            relative = prefix + item.name
            if (depth >= 64 or
                    len(relative.encode('utf-8', 'surrogatepass'))
                    > MAX_RELATIVE_PATH_BYTES):
                raise SnapshotNotReady('unexpected deeply nested snapshot path')
            _safe_name(item.name)
            info = _lstat(item)
            if stat.S_ISDIR(info.st_mode):
                paths.append({'path': relative, 'kind': 'directory',
                              'bytes': 0, 'sha256': None})
                walk(item, relative + '/', depth + 1)
            elif stat.S_ISREG(info.st_mode):
                paths.append({'path': relative, 'kind': 'file',
                              'bytes': info.st_size,
                              'sha256': _hash_file(item, info, block_bytes)})
            else:
                raise SnapshotNotReady('special file appeared in snapshot')
            if len(paths) > DEFAULT_MAX_ENTRIES:
                raise SnapshotNotReady('snapshot contains too many entries')
    walk(target, '')
    if sum(item['bytes'] for item in paths) > DEFAULT_MAX_BYTES:
        raise SnapshotNotReady('snapshot exceeds bounded byte budget')
    return {'entries': sorted(paths, key=lambda p: p['path'])}
