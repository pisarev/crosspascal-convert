# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Inventory and copy one ONNX model and its referenced external data files.

The inventory covers the graph and the regular files named by its tensors.
The snapshot copies those files into a separate directory and checks them
against the approved digest. The caller must hold or recheck the staged files
when it needs a stable input for compilation.
"""

from __future__ import annotations

import json
import os
import re
import stat
import unicodedata
from pathlib import Path

from . import source_freeze

SCHEMA = 'crosspascal.onnx-selected-closure'
SCHEMA_VERSION = 1
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_FILES = 32
MAX_EXTERNAL_REFERENCES = 128
MAX_GRAPH_DEPTH = 32
MAX_GRAPH_ITEMS = 20000
MAX_RELATIVE_DEPTH = 16
BLOCK_BYTES = source_freeze.DEFAULT_BLOCK_BYTES
STATUS = 'COPIED_RECHECKED_NOT_IMMUTABLE'
_SHA = re.compile(r'[0-9a-fA-F]{64}\Z')
_DECIMAL = re.compile(r'[0-9]+\Z')
_URL_ESCAPE = re.compile(r'%[0-9a-fA-F]{2}')


_DTYPE_BYTES = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8,
    9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 14: 8, 15: 16,
    16: 2, 17: 1, 18: 1, 19: 1, 20: 1,
}
_EXTERNAL_KEYS = frozenset(('location', 'offset', 'length'))
_INLINE_FIELDS = (
    'raw_data', 'float_data', 'int32_data', 'string_data',
    'int64_data', 'double_data', 'uint64_data',
)


class OnnxClosureError(ValueError):
    """A selected source, external reference or private snapshot is unsafe."""


class OnnxClosureChanged(OnnxClosureError):
    """The approved selected files differ from the current actual bytes."""


class OnnxClosureNotReady(OnnxClosureError):
    """A private stage has an invalid marker, member or byte fingerprint."""


def _limits(max_total_bytes, max_files):
    if (type(max_total_bytes) is not int or
            not 1 <= max_total_bytes <= MAX_TOTAL_BYTES):
        raise OnnxClosureError('selected closure byte limit must be 1..64 MiB')
    if type(max_files) is not int or not 1 <= max_files <= MAX_FILES:
        raise OnnxClosureError('selected closure file limit must be 1..32')


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise OnnxClosureError('an exact approved 64-digit SHA-256 is required')
    return value.lower()


def _absolute(path, label):
    try:
        return source_freeze._absolute_path(path, label)
    except (OSError, TypeError, ValueError, source_freeze.SourceFreezeError) as exc:
        raise OnnxClosureError('%s is not a safe local path: %s' % (label, exc)) from exc


def _relative(raw, label='ONNX external_data location'):
    """Accept only lexical, unambiguous relative POSIX-style path components."""
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise OnnxClosureError('%s must be a nonempty relative string' % label)
    if ('\\' in raw or ':' in raw or raw.startswith('/') or
            _URL_ESCAPE.search(raw) or
            unicodedata.normalize('NFC', raw) != raw or
            any(unicodedata.category(char).startswith('C') for char in raw)):
        raise OnnxClosureError('%s has an absolute/escaped/ambiguous path' % label)
    parts = raw.split('/')
    if (len(parts) > MAX_RELATIVE_DEPTH or
            len(raw.encode('utf-8', 'surrogatepass')) >
            source_freeze.MAX_RELATIVE_PATH_BYTES):
        raise OnnxClosureError('%s exceeds relative-path bounds' % label)
    try:
        for component in parts:
            source_freeze._safe_name(component)
    except source_freeze.SourceFreezeError as exc:
        raise OnnxClosureError('%s is unsafe: %s' % (label, exc)) from exc
    if any(component in ('.', '..') for component in parts):
        raise OnnxClosureError('%s cannot contain . or ..' % label)
    if parts[0] == source_freeze.READY_MARKER:
        raise OnnxClosureError('%s collides with private ready-marker name' % label)
    return raw


def _regular(root, relative, *, require_nonempty=True):
    path = root.joinpath(*relative.split('/'))
    try:
        source_freeze._safe_ancestors(path)
        info = source_freeze._lstat(path)
    except (OSError, source_freeze.SourceFreezeError) as exc:
        raise OnnxClosureError('missing/unsafe selected file %s: %s'
                               % (relative, exc)) from exc
    if not stat.S_ISREG(info.st_mode) or (require_nonempty and info.st_size <= 0):
        raise OnnxClosureError('selected file must be nonempty regular file: '
                               + relative)
    if info.st_size > MAX_TOTAL_BYTES:
        raise OnnxClosureError('selected file exceeds the 64 MiB hard limit: '
                               + relative)
    return path, info


def _hash(path, info):
    try:
        return source_freeze._hash_file(path, info, BLOCK_BYTES)
    except (OSError, source_freeze.SourceFreezeError) as exc:
        raise OnnxClosureChanged('selected source changed while hashing %s: %s'
                                 % (path, exc)) from exc


def _number(raw, field):
    if (not isinstance(raw, str) or len(raw) > 20 or
            not _DECIMAL.fullmatch(raw)):
        raise OnnxClosureError('external tensor %s must be a decimal integer'
                               % field)
    value = int(raw)
    if value > MAX_TOTAL_BYTES:
        raise OnnxClosureError('external tensor %s exceeds 64 MiB' % field)
    return value


def _expected_bytes(tensor):
    dtype = int(tensor.data_type)
    item_bytes = _DTYPE_BYTES.get(dtype)
    if item_bytes is None:
        raise OnnxClosureError('external tensor has unsupported dtype %s'
                               % dtype)
    total = item_bytes
    dims = []
    for dim in tensor.dims:
        value = int(dim)
        if value <= 0 or total > MAX_TOTAL_BYTES // value:
            raise OnnxClosureError('external tensor has empty/oversized dimensions')
        total *= value
        dims.append(value)
    if total > MAX_TOTAL_BYTES:
        raise OnnxClosureError('external tensor exceeds 64 MiB')
    return dtype, dims, total


def _external_tensors(model):
    """Discover external weights in graphs, nested attributes and sparse data."""
    import onnx

    if model.training_info or model.functions:
        raise OnnxClosureError('ONNX training/function graphs are out of scope')
    refs = []
    visited_items = 0

    def tensor_ref(tensor):
        values = list(tensor.external_data)
        is_external = int(tensor.data_location) == onnx.TensorProto.EXTERNAL
        if not values and not is_external:
            return
        if not is_external or not values:
            raise OnnxClosureError('external tensor metadata/data_location disagree')
        if any(getattr(tensor, field) for field in _INLINE_FIELDS):
            raise OnnxClosureError('external tensor also contains inline tensor data')
        if tensor.HasField('segment'):
            raise OnnxClosureError('segmented external tensor is unsupported')
        meta = {}
        for item in values:
            if item.key not in _EXTERNAL_KEYS or item.key in meta:
                raise OnnxClosureError('unknown/duplicate external tensor metadata key')
            meta[item.key] = item.value
        location = _relative(meta.get('location'),
                             'external tensor location')
        offset = _number(meta['offset'], 'offset') if 'offset' in meta else 0
        length = (_number(meta['length'], 'length')
                  if 'length' in meta else None)
        dtype, dims, expected = _expected_bytes(tensor)
        if length is not None and length != expected:
            raise OnnxClosureError('external tensor length differs from dtype/shape')
        refs.append({
            'tensor': tensor.name,
            'location': location,
            'offset': offset,
            'length': length,
            'expected_bytes': expected,
            'dtype': dtype,
            'dims': dims,
        })
        if len(refs) > MAX_EXTERNAL_REFERENCES:
            raise OnnxClosureError('too many external tensor references')

    def sparse_ref(tensor):
        tensor_ref(tensor.values)
        tensor_ref(tensor.indices)

    def graph_ref(graph, depth=0):
        nonlocal visited_items
        if depth > MAX_GRAPH_DEPTH:
            raise OnnxClosureError('ONNX nested graph depth exceeds strict limit')
        visited_items += (len(graph.initializer) + len(graph.sparse_initializer)
                          + len(graph.node))
        if visited_items > MAX_GRAPH_ITEMS:
            raise OnnxClosureError('ONNX graph item count exceeds strict limit')
        for tensor in graph.initializer:
            tensor_ref(tensor)
        for tensor in graph.sparse_initializer:
            sparse_ref(tensor)
        for node in graph.node:
            if node.domain not in ('', 'ai.onnx', 'ai.onnx.ml'):
                raise OnnxClosureError('custom ONNX operator domain is out of scope')
            for attr in node.attribute:
                if attr.HasField('t'):
                    tensor_ref(attr.t)
                for tensor in attr.tensors:
                    tensor_ref(tensor)
                if attr.HasField('sparse_tensor'):
                    sparse_ref(attr.sparse_tensor)
                for tensor in attr.sparse_tensors:
                    sparse_ref(tensor)
                if attr.HasField('g'):
                    graph_ref(attr.g, depth + 1)
                for nested in attr.graphs:
                    graph_ref(nested, depth + 1)

    graph_ref(model.graph)
    return sorted(refs, key=lambda r: (
        r['location'], r['offset'], r['tensor'], r['dtype'],
        tuple(r['dims']), -1 if r['length'] is None else r['length']))


def _entries_with_directories(root, relative_files, *, max_total_bytes, max_files):
    if len(relative_files) > max_files:
        raise OnnxClosureError('too many selected ONNX/sidecar files')
    members = {}
    for relative in relative_files:
        pieces = relative.split('/')
        for i in range(1, len(pieces)):
            path = '/'.join(pieces[:i])
            previous = members.setdefault(path, 'directory')
            if previous != 'directory':
                raise OnnxClosureError('file/directory path collision: ' + path)
        if members.setdefault(relative, 'file') != 'file':
            raise OnnxClosureError('file/directory path collision: ' + relative)
    folded = {}
    for path in members:
        alias = unicodedata.normalize('NFC', path).casefold()
        if alias in folded and folded[alias] != path:
            raise OnnxClosureError('case/Unicode path collision: %s / %s'
                                   % (path, folded[alias]))
        folded[alias] = path
    if len(members) > MAX_FILES + MAX_FILES * MAX_RELATIVE_DEPTH:
        raise OnnxClosureError('selected closure has excessive directory entries')
    total = 0
    entries = []
    for rel in sorted(members):
        kind = members[rel]
        if kind == 'directory':
            directory = root.joinpath(*rel.split('/'))
            try:
                source_freeze._safe_ancestors(directory)
                info = source_freeze._lstat(directory)
            except (OSError, source_freeze.SourceFreezeError) as exc:
                raise OnnxClosureError('unsafe selected directory %s: %s'
                                       % (rel, exc)) from exc
            if not stat.S_ISDIR(info.st_mode):
                raise OnnxClosureError('selected dependency parent is not a directory')
            entries.append({'path': rel, 'kind': kind,
                            'bytes': 0, 'sha256': None})
            continue
        file_path, info = _regular(root, rel)
        total += info.st_size
        if total > max_total_bytes:
            raise OnnxClosureError('selected ONNX plus sidecars exceeds byte limit')
        entries.append({'path': rel, 'kind': kind,
                        'bytes': info.st_size, 'sha256': _hash(file_path, info)})
    return entries, total


def _forbidden_file(root, relative, forbidden_roots):
    file_path = os.path.normcase(os.path.abspath(
        os.fspath(root.joinpath(*relative.split('/')))))
    for forbidden in forbidden_roots:
        directory = os.path.normcase(os.path.abspath(os.fspath(
            _absolute(forbidden, 'reserved output directory'))))
        if file_path == directory or file_path.startswith(directory + os.sep):
            raise OnnxClosureError('ONNX dependency points into reserved output')


def scan_selected_closure(model_path, *, max_total_bytes=MAX_TOTAL_BYTES,
                          max_files=MAX_FILES, forbidden_roots=()):
    """Fingerprint only a local ONNX and referenced sidecar files, never siblings.

    The resulting SHA covers relative member names/types/sizes/whole bytes.
    It is a deterministic COPY input, not a lock on original source files.
    """
    _limits(max_total_bytes, max_files)
    model_file = _absolute(model_path, 'ONNX model')
    if model_file.suffix.lower() != '.onnx':
        raise OnnxClosureError('selective external data requires one .onnx file')
    model_rel = _relative(model_file.name, 'ONNX filename')
    root = model_file.parent
    initial_file, initial = _regular(root, model_rel)
    if initial.st_size > MAX_MODEL_BYTES:
        raise OnnxClosureError('ONNX protobuf exceeds bounded 16 MiB parse limit')
    original_sha = _hash(initial_file, initial)
    try:
        import onnx
    except ImportError as exc:
        raise OnnxClosureError('onnx parser is required for dependency closure') from exc
    try:
        model = onnx.load(str(model_file), load_external_data=False)
    except Exception as exc:
        raise OnnxClosureError('cannot parse selected ONNX without external data: %s'
                               % exc) from exc
    refs = _external_tensors(model)
    if any(ref['location'].casefold() == model_rel.casefold() for ref in refs):
        raise OnnxClosureError('ONNX cannot name itself as an external sidecar')
    names = {model_rel}
    names.update(row['location'] for row in refs)
    if len(names) > max_files:
        raise OnnxClosureError('selected ONNX has too many external sidecar files')
    for name in names:
        _forbidden_file(root, name, forbidden_roots)
    entries, total = _entries_with_directories(
        root, names, max_total_bytes=max_total_bytes, max_files=max_files)
    by_name = {row['path']: row for row in entries if row['kind'] == 'file'}
    if by_name[model_rel]['sha256'] != original_sha:
        raise OnnxClosureChanged('ONNX model changed while parsing external locations')
    for ref in refs:
        size = by_name[ref['location']]['bytes']
        offset = ref['offset']
        amount = (ref['length'] if ref['length'] is not None
                  else size - offset)
        if (offset >= size or amount != ref['expected_bytes'] or
                amount <= 0 or amount > size - offset):
            raise OnnxClosureError(
                'external tensor range/shape exceeds or disagrees with sidecar')
    return {
        'schema': SCHEMA,
        'schema_version': SCHEMA_VERSION,
        'source_model': str(model_file),
        'source_root': str(root),
        'model_rel': model_rel,
        'source_kind': 'selected_onnx_files',
        'entries': entries,
        'references': refs,
        'dependencies': sorted(names - {model_rel}),
        'file_count': len(names),
        'total_bytes': total,
        'tree_sha256': source_freeze._tree_sha({'entries': entries}),
        'status': 'SELECTED_FILES_SCANNED_NOT_LOCKED',
        'release_grade_immutable': False,
    }


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OnnxClosureNotReady('private marker has a duplicate JSON key')
        result[key] = value
    return result


def audit_selected_closure(snapshot_directory, *, expected_tree_sha256):
    """Independently reparse staged ONNX and verify ALL private stage members."""
    approved = _sha(expected_tree_sha256)
    stage = _absolute(snapshot_directory, 'selected snapshot')
    marker_path = stage / source_freeze.READY_MARKER
    try:
        source_freeze._safe_ancestors(marker_path)
        info = source_freeze._lstat(marker_path)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 128 * 1024:
            raise OnnxClosureNotReady('private ready marker is empty/oversized')
        marker = json.loads(marker_path.read_bytes().decode('utf-8'),
                            object_pairs_hook=_unique_pairs)
        if (not isinstance(marker, dict) or marker.get('closure_schema') != SCHEMA
                or marker.get('closure_version') != SCHEMA_VERSION
                or marker.get('model_rel') is None
                or marker.get('source_tree_sha256') != approved
                or marker.get('snapshot_tree_sha256') != approved):
            raise OnnxClosureNotReady('private marker does not match approved closure')
        model_rel = _relative(marker['model_rel'], 'private selected model')
        audited = source_freeze.audit_frozen_source(stage)
        if (audited['compiler_input'] != str(stage) or
                audited['source_tree_sha256'] != approved):
            raise OnnxClosureNotReady('private stage tree/marker differs from approval')
        fresh = scan_selected_closure(stage / model_rel)
        if (fresh['tree_sha256'] != approved or
                fresh['entries'] != marker.get('entries') or
                fresh['dependencies'] != marker.get('dependencies') or
                fresh['references'] != marker.get('references')):
            raise OnnxClosureNotReady('private staged ONNX/sidecars differ from marker')
    except OnnxClosureNotReady:
        raise
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            source_freeze.SourceFreezeError) as exc:
        raise OnnxClosureNotReady('invalid private ONNX closure: %s' % exc) from exc
    return {
        'snapshot_directory': str(stage),
        'compiler_input': str(stage / model_rel),
        'model_rel': model_rel,
        'tree_sha256': approved,
        'dependencies': fresh['dependencies'],
        'file_count': fresh['file_count'],
        'total_bytes': fresh['total_bytes'],
        'status': STATUS,
        'release_grade_immutable': False,
    }


def freeze_selected_closure(model_path, destination, *,
                            expected_tree_sha256,
                            max_total_bytes=MAX_TOTAL_BYTES,
                            max_files=MAX_FILES, forbidden_roots=(),
                            on_created=None):
    """Copy ONLY approved graph/sidecars into an exclusive private directory.

    Does not lock either source or staged paths through compilation. On failure
    after exclusive mkdir an OWNED incomplete stage is left without a valid
    ready marker; no pre-existing path is removed or overwritten.
    """
    _limits(max_total_bytes, max_files)
    approved = _sha(expected_tree_sha256)
    first = scan_selected_closure(
        model_path, max_total_bytes=max_total_bytes,
        max_files=max_files, forbidden_roots=forbidden_roots)
    if first['tree_sha256'] != approved:
        raise OnnxClosureChanged('selected source differs from Preview SHA-256')
    target = _absolute(destination, 'private snapshot destination')
    source_model = Path(first['source_model'])
    if target == source_model or target == source_model.parent:
        raise OnnxClosureError('private stage cannot overwrite source/model directory')
    try:
        source_freeze._safe_ancestors(target, leaf_may_be_missing=True)
        if os.path.lexists(target):
            raise OnnxClosureError('private stage already exists: no overwrite allowed')
        if source_freeze._free_bytes(target.parent) < (
                first['total_bytes'] + source_freeze.DEFAULT_MIN_DISK_HEADROOM + 4096):
            raise OnnxClosureError('insufficient disk space for private ONNX closure')
    except (OSError, source_freeze.SourceFreezeError) as exc:
        raise OnnxClosureError('unsafe private snapshot destination: %s' % exc) from exc
    target.mkdir(exist_ok=False)
    info = source_freeze._lstat(target)
    identity = (info.st_dev, info.st_ino, info.st_mode)
    if on_created is not None:
        on_created((info.st_dev, info.st_ino))
    source_root = Path(first['source_root'])
    for entry in first['entries']:
        source_freeze._assert_owned_stage(target, identity)
        rel = entry['path']
        copied = target.joinpath(*rel.split('/'))
        source_freeze._safe_ancestors(copied.parent)
        if entry['kind'] == 'directory':
            copied.mkdir(exist_ok=False)
        else:
            selected = source_root.joinpath(*rel.split('/'))
            source_freeze._copy_stream(
                selected, copied, entry['sha256'], entry['bytes'], BLOCK_BYTES)
    second = scan_selected_closure(
        model_path, max_total_bytes=max_total_bytes,
        max_files=max_files, forbidden_roots=forbidden_roots)
    if (second['entries'] != first['entries'] or
            second['references'] != first['references'] or
            second['tree_sha256'] != approved):
        raise OnnxClosureChanged('selected source changed during private capture')
    staged = scan_selected_closure(
        target / first['model_rel'], max_total_bytes=max_total_bytes,
        max_files=max_files)
    if (staged['entries'] != first['entries'] or
            staged['references'] != first['references']):
        raise OnnxClosureChanged('staged model or dependencies differ from source')


    all_stage = source_freeze._scan_source(
        target, max_entries=MAX_FILES * MAX_RELATIVE_DEPTH + MAX_FILES,
        max_bytes=max_total_bytes, block_bytes=BLOCK_BYTES)
    if all_stage['entries'] != first['entries']:
        raise OnnxClosureNotReady('private stage contains an unlisted member')
    marker = {
        'schema': source_freeze.SCHEMA,
        'schema_version': source_freeze.VERSION,
        'closure_schema': SCHEMA,
        'closure_version': SCHEMA_VERSION,
        'status': STATUS,
        'source_kind': 'directory',
        'source_path': first['source_model'],
        'snapshot_input': '.',
        'model_rel': first['model_rel'],
        'source_tree_sha256': approved,
        'snapshot_tree_sha256': source_freeze._tree_sha(all_stage),
        'total_bytes': first['total_bytes'],
        'entries': first['entries'],
        'dependencies': first['dependencies'],
        'references': first['references'],
        'limitations': [
            'Selected source COPY only; no Win32 lock is held by this helper.',
            'Unlisted owner-directory files are not frozen or read.',
            'No model numeric validation, portability or VERIFIED claim.',
        ],
    }
    source_freeze._assert_owned_stage(target, identity)
    source_freeze._publish_marker(target, marker)
    source_freeze._assert_owned_stage(target, identity)
    return audit_selected_closure(target, expected_tree_sha256=approved)
