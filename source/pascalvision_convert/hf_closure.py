# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Inventory selected local Hugging Face files for Qwen and Gemma.

The scanner reads the model configuration, tokenizer metadata and either a
single safetensors file or the shards listed in the safetensors index. It
returns a digest of the files read. Provenance and runtime validation
require separate checks.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import unicodedata

from . import llm_family_profile, source_freeze


SCHEMA = 'crosspascal.hf-selected-source-closure'
SCHEMA_VERSION = 1
STATUS = 'SCANNED_NOT_LOCKED'
MAX_TOTAL_BYTES = 16 * 1024**3
MAX_SHARDS = 4096
MAX_TENSORS = 100000
MAX_TOKENIZER_MERGES = 1000000
MAX_SELECTED_FILES = MAX_SHARDS + 16
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_SHARD_DEPTH = 16
MAX_SHARD_PATH_CHARS = 1024
MAX_PATH_BYTES = 2048
MAX_TENSOR_ELEMENTS = 2_000_000_000
MAX_TENSOR_DIMS = 16
MAX_NAME_CHARS = 1024
HASH_BLOCK_BYTES = 1024 * 1024
_DTYPE_BYTES = {'F16': 2, 'BF16': 2, 'F32': 4}
_ESCAPED_OCTET = re.compile(r'%[0-9a-fA-F]{2}')
_SHA = re.compile(r'[0-9a-fA-F]{64}\Z')
_DOS_DEVICES = frozenset(
    ('con', 'prn', 'aux', 'nul', 'com¹', 'com²', 'com³',
     'lpt¹', 'lpt²', 'lpt³') +
    tuple('com%d' % n for n in range(1, 10)) +
    tuple('lpt%d' % n for n in range(1, 10)))
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)

_MANDATORY = frozenset(('config.json', 'tokenizer.json'))
_OPTIONAL_JSON = frozenset((
    'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
    'generation_config.json', 'preprocessor_config.json',
    'processor_config.json', 'feature_extractor_config.json', 'vocab.json'))
_OPTIONAL_OTHER = frozenset((
    'chat_template.jinja', 'tokenizer.model', 'merges.txt'))
_LOCAL_REFERENCE_FIELDS = frozenset((
    'tokenizer_file', 'vocab_file', 'merges_file', 'sp_model_file',
    'chat_template_file'))


class HfClosureError(ValueError):
    """Selected HF dependency cannot be represented safely."""


class HfClosureChanged(HfClosureError):
    """Selected bytes no longer equal the separately approved SHA tree."""


def _deny(reason):
    raise HfClosureError('HF source rejected: ' + reason)


def _sha(value):
    if type(value) is not str or not _SHA.fullmatch(value):
        _deny('expected a 64-digit approved SHA-256')
    return value.lower()


def _identity(info):
    return (info.st_mode, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_dev, info.st_ino)


def _descriptor_identity(info):

    return (info.st_mode, info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)


def _root_identity(info):

    return info.st_mode, info.st_dev, info.st_ino


def _root(path):
    try:
        raw = os.fspath(path)
        if type(raw) is not str or raw.startswith(('\\\\', '//')):
            _deny('HF source must be a local directory, not UNC/device path')
        base = source_freeze._absolute_path(path, 'HF source')
        source_freeze._safe_ancestors(base)
        info = source_freeze._lstat(base)
    except (OSError, TypeError, ValueError, source_freeze.SourceFreezeError) as exc:
        raise HfClosureError('unsafe/unavailable HF root: %s' % exc) from exc
    if not stat.S_ISDIR(info.st_mode):
        _deny('HF source must be a local regular directory')
    return base, info


def _relative(value):
    """Lexically unambiguous POSIX-like relative path; nested shards allowed."""
    if (type(value) is not str or not value or
            unicodedata.normalize('NFC', value) != value or
            '\\' in value or ':' in value or value.startswith('/') or
            _ESCAPED_OCTET.search(value) or
            len(value) > MAX_SHARD_PATH_CHARS or
            len(value.encode('utf-8', 'surrogatepass')) > MAX_PATH_BYTES):
        _deny('absolute, encoded or ambiguous HF relative path %r' % value)
    parts = value.split('/')
    if len(parts) > MAX_SHARD_DEPTH:
        _deny('HF relative path depth exceeds %d' % MAX_SHARD_DEPTH)
    for part in parts:
        if (not part or part in ('.', '..') or len(part) > 255 or
                part.endswith(('.', ' ')) or
                part.split('.', 1)[0].casefold() in _DOS_DEVICES or
                any(unicodedata.category(ch).startswith('C') or
                    ch in '<>:"\\|?*' for ch in part)):
            _deny('unsafe HF filename component in %r' % value)
    return value


def _register_path(aliases, relative):
    """Check case/NFC aliases at EVERY nested ancestor, not just file leaves."""
    parts = _relative(relative).split('/')
    for i in range(1, len(parts) + 1):
        prefix = '/'.join(parts[:i])
        folded = prefix.casefold()
        prior = aliases.setdefault(folded, prefix)
        if prior != prefix:
            _deny('case/Unicode alias between %r and %r' % (prior, prefix))


def _regular(base, relative, *, max_bytes=None):
    _relative(relative)
    path = base.joinpath(*relative.split('/'))
    try:
        if os.path.commonpath((str(base), str(path))) != str(base):
            _deny('selected HF file escapes source root')
        source_freeze._safe_ancestors(path)
        info = source_freeze._lstat(path)
    except (OSError, ValueError, source_freeze.SourceFreezeError) as exc:
        raise HfClosureError('missing/unsafe HF member %s: %s' %
                             (relative, exc)) from exc
    if (not stat.S_ISREG(info.st_mode) or
            bool(getattr(info, 'st_file_attributes', 0) & _REPARSE) or
            info.st_size <= 0 or
            (max_bytes is not None and info.st_size > max_bytes)):
        _deny('HF member is linked, not regular, empty or oversized: ' + relative)
    return path, info


@contextmanager
def _opened_regular(base, relative, *, max_bytes=None):
    path, info = _regular(base, relative, max_bytes=max_bytes)


    flags = (os.O_RDONLY | getattr(os, 'O_BINARY', 0) |
             getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0))
    fd = None
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as handle:
            fd = None
            if _descriptor_identity(os.fstat(handle.fileno())) != _descriptor_identity(info):
                _deny('HF member replaced between lstat and open: ' + relative)
            yield handle, info
            if _descriptor_identity(os.fstat(handle.fileno())) != _descriptor_identity(info):
                _deny('HF member changed while reading: ' + relative)
    except OSError as exc:
        raise HfClosureError('HF member open/read failed %s: %s' %
                             (relative, exc)) from exc
    finally:
        if fd is not None:
            os.close(fd)
    try:
        source_freeze._safe_ancestors(path)
        after = source_freeze._lstat(path)
    except (OSError, source_freeze.SourceFreezeError) as exc:
        raise HfClosureError('HF member path changed %s: %s' %
                             (relative, exc)) from exc
    if _identity(after) != _identity(info):
        _deny('HF member metadata/path changed during read: ' + relative)


def _read_bounded(base, relative, limit):
    with _opened_regular(base, relative, max_bytes=limit) as (handle, info):
        raw = handle.read(limit + 1)
        if len(raw) != info.st_size:
            _deny('HF metadata changed/truncated while reading: ' + relative)
    return raw


def _unique_pairs(max_keys):
    def unique(pairs):
        if len(pairs) > max_keys:
            raise ValueError('JSON object exceeds field bound')
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key %r' % key)
            result[key] = value
        return result
    return unique


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('nonfinite JSON number')
    return parsed


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant %s' % value)


def _decode_json(raw, label, *, max_keys=MAX_TENSORS + 2):
    try:
        return json.loads(raw.decode('utf-8'),
                          object_pairs_hook=_unique_pairs(max_keys),
                          parse_float=_finite_float,
                          parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise HfClosureError('invalid HF JSON %s: %s' % (label, exc)) from exc


def _json_file(base, relative, limit, *, max_keys=MAX_TENSORS + 2):
    raw = _read_bounded(base, relative, limit)
    return _decode_json(raw, relative, max_keys=max_keys), hashlib.sha256(raw).hexdigest()


def _check_reference_metadata(data, label, present_names):
    if not isinstance(data, dict):
        _deny('HF metadata must be an object: ' + label)
    for field in ('auto_map', 'custom_pipelines'):
        if data.get(field):
            _deny('%s requires unsupported remote/custom code in %s' %
                  (field, label))
    if data.get('trust_remote_code'):
        _deny('trust_remote_code unsupported in ' + label)
    for field in _LOCAL_REFERENCE_FIELDS:
        value = data.get(field)
        if value is None:
            continue
        if type(value) is not str or value not in present_names:
            _deny('unclosed/unsupported %s reference in %s: %r' %
                  (field, label, value))


def _valid_tensor_name(name):
    return (type(name) is str and bool(name) and len(name) <= MAX_NAME_CHARS and
            unicodedata.normalize('NFC', name) == name and
            not any(unicodedata.category(char).startswith('C') for char in name))


def _safetensors(base, relative):
    """Hash one exact opened descriptor while verifying its bounded tensor header."""
    with _opened_regular(base, relative, max_bytes=MAX_TOTAL_BYTES) as (handle, info):
        prefix = handle.read(8)
        if len(prefix) != 8:
            _deny('truncated safetensors prefix: ' + relative)
        header_size = struct.unpack('<Q', prefix)[0]
        if (not 2 <= header_size <= MAX_HEADER_BYTES or
                header_size > info.st_size - 8):
            _deny('invalid safetensors header length: ' + relative)
        header_raw = handle.read(header_size)
        if len(header_raw) != header_size:
            _deny('truncated safetensors header: ' + relative)
        header = _decode_json(header_raw, relative)
        if not isinstance(header, dict):
            _deny('safetensors header must be an object: ' + relative)
        meta = header.get('__metadata__')
        if '__metadata__' in header and (
                type(meta) is not dict or
                any(type(k) is not str or type(v) is not str
                    for k, v in meta.items())):
            _deny('invalid safetensors __metadata__: ' + relative)
        names = set(header) - {'__metadata__'}
        if not 0 < len(names) <= MAX_TENSORS:
            _deny('safetensors tensor count is invalid: ' + relative)
        data_bytes = info.st_size - 8 - header_size
        spans = []
        for name in names:
            if not _valid_tensor_name(name):
                _deny('invalid safetensors tensor name in ' + relative)
            entry = header[name]
            if (type(entry) is not dict or
                    set(entry) != {'dtype', 'shape', 'data_offsets'}):
                _deny('invalid safetensors tensor fields in ' + relative)
            item_bytes = _DTYPE_BYTES.get(entry['dtype']) if type(entry['dtype']) is str else None
            dims = entry['shape']
            offsets = entry['data_offsets']
            if (item_bytes is None or type(dims) is not list or
                    len(dims) > MAX_TENSOR_DIMS or type(offsets) is not list or
                    len(offsets) != 2 or any(type(v) is not int for v in offsets)):
                _deny('invalid safetensors dtype/shape/offset types in ' + relative)
            elements = 1
            for dim in dims:
                if (type(dim) is not int or dim <= 0 or
                        dim > MAX_TENSOR_ELEMENTS // elements):
                    _deny('invalid/oversized safetensors shape in ' + relative)
                elements *= dim
            begin, end = offsets
            if (begin < 0 or end <= begin or end > data_bytes or
                    end - begin != elements * item_bytes):
                _deny('safetensors tensor byte range differs from shape in ' + relative)
            spans.append((begin, end))
        cursor = 0
        for begin, end in sorted(spans):
            if begin != cursor:
                _deny('safetensors tensor ranges overlap/have gaps in ' + relative)
            cursor = end
        if cursor != data_bytes:
            _deny('safetensors has trailing or unassigned data in ' + relative)

        digest = hashlib.sha256(prefix + header_raw)
        total = 8 + header_size
        while True:
            piece = handle.read(HASH_BLOCK_BYTES)
            if not piece:
                break
            digest.update(piece)
            total += len(piece)
        if total != info.st_size:
            _deny('truncated safetensors file during SHA read: ' + relative)
    return {'path': relative, 'bytes': total, 'sha256': digest.hexdigest()}, names, data_bytes


def _index_map(index):
    if (type(index) is not dict or
            set(index) - {'metadata', 'weight_map'}):
        _deny('invalid safetensors index fields')
    metadata = index.get('metadata', {})
    mapping = index.get('weight_map')
    if (type(metadata) is not dict or type(mapping) is not dict or
            not 0 < len(mapping) <= MAX_TENSORS):
        _deny('invalid safetensors index metadata/weight_map')
    if 'total_size' in metadata and (
            type(metadata['total_size']) is not int or
            not 0 <= metadata['total_size'] <= MAX_TOTAL_BYTES):
        _deny('invalid safetensors index total_size')
    shards = {}
    for tensor, name in mapping.items():
        if not _valid_tensor_name(tensor):
            _deny('invalid safetensors index tensor name')
        _relative(name)
        if not name.lower().endswith('.safetensors'):
            _deny('safetensors shard must end with .safetensors')
        shards.setdefault(name, set()).add(tensor)
    if not 0 < len(shards) <= MAX_SHARDS:
        _deny('safetensors shard count exceeds bound')
    return metadata, shards


def _metadata_selection(base):
    """Find exact, supported local metadata paths without traversing the tree."""
    present = set(_MANDATORY)
    for name in _OPTIONAL_JSON | _OPTIONAL_OTHER:
        if os.path.lexists(base / name):
            present.add(name)
    has_single = os.path.lexists(base / 'model.safetensors')
    has_index = os.path.lexists(base / 'model.safetensors.index.json')
    if has_single == has_index:
        _deny('require exactly one single safetensors or indexed shard layout')
    present.add('model.safetensors' if has_single else 'model.safetensors.index.json')
    return present, has_index


def scan_selected_hf_source(directory, *, expected_model_key=None,
                            max_total_bytes=MAX_TOTAL_BYTES):
    """Inventory selected files and compute a digest of their relative paths and bytes.

    The digest is independent of the parent directory and unrelated files.
    Callers that need a stable input must also hold or recheck the files.
    """
    if (type(max_total_bytes) is not int or
            not 1 <= max_total_bytes <= MAX_TOTAL_BYTES):
        _deny('HF selected source quota must be 1..16GiB')
    base, root_before = _root(directory)
    selected, has_index = _metadata_selection(base)


    aliases = {}
    for name in sorted(selected):
        _register_path(aliases, name)
    config, config_hash = _json_file(base, 'config.json', MAX_CONFIG_BYTES)
    try:
        from . import qwen_universal
        model_key = llm_family_profile.identify_model_key(
            config, generic=(expected_model_key == qwen_universal.KEY))
    except ValueError as exc:
        raise HfClosureError('unsupported HF model config: %s' % exc) from exc
    if model_key not in ('qwen2.5-0.5b', 'qwen2.5-1.5b',
                         'gemma-2-2b', qwen_universal.KEY):
        _deny('unsupported HF model structural profile')
    if expected_model_key is not None and expected_model_key != model_key:
        _deny('HF source model profile differs from selected model')


    tokenizer, token_hash = _json_file(
        base, 'tokenizer.json', MAX_TOKENIZER_BYTES, max_keys=512000)
    if (type(tokenizer) is not dict or
            type(tokenizer.get('model')) is not dict or
            tokenizer['model'].get('type') != 'BPE' or
            type(tokenizer['model'].get('vocab')) is not dict or
            not 0 < len(tokenizer['model']['vocab']) <= 512000 or
            type(tokenizer['model'].get('merges')) is not list or
            len(tokenizer['model']['merges']) > MAX_TOKENIZER_MERGES or
            (model_key == 'gemma-2-2b' and
             tokenizer['model'].get('byte_fallback') is not True)):
        _deny('unsupported HF tokenizer.json schema for model family')

    parsed_hashes = {'config.json': config_hash, 'tokenizer.json': token_hash}
    for name in sorted(selected & _OPTIONAL_JSON):
        cap = MAX_TOKENIZER_BYTES if name == 'vocab.json' else MAX_METADATA_BYTES
        doc, digest = _json_file(
            base, name, cap, max_keys=512000 if name == 'vocab.json' else MAX_TENSORS + 2)
        parsed_hashes[name] = digest
        if name in ('tokenizer_config.json', 'preprocessor_config.json',
                    'processor_config.json', 'feature_extractor_config.json'):
            _check_reference_metadata(doc, name, selected)
    _check_reference_metadata(config, 'config.json', selected)
    _check_reference_metadata(tokenizer, 'tokenizer.json', selected)
    if model_key == qwen_universal.KEY:
        # Do not infer compatibility from a model label or config alone:
        # require the exact accepted tokenizer and the complete unfused
        # Safetensors header layout before a potentially costly hash/Build.
        qwen_universal.validate_config(config)
        qwen_universal.verify_tokenizer(base)
        qwen_universal.inspect_local_headers(base, qwen_universal.validate_config(config))

    index_meta, shards = {}, {}
    if has_index:
        raw = _read_bounded(base, 'model.safetensors.index.json', MAX_INDEX_BYTES)
        index_meta, shards = _index_map(
            _decode_json(raw, 'model.safetensors.index.json'))
        parsed_hashes['model.safetensors.index.json'] = hashlib.sha256(raw).hexdigest()
        for relative in sorted(shards):
            _register_path(aliases, relative)
            selected.add(relative)
    else:
        shards = {'model.safetensors': None}

    if len(selected) > MAX_SELECTED_FILES:
        _deny('selected HF file count exceeds bound')

    preflight_total = 0
    for relative in sorted(selected):
        cap = (MAX_TOTAL_BYTES if relative in shards else
               MAX_TOKENIZER_BYTES if relative in ('tokenizer.json', 'vocab.json') else
               MAX_INDEX_BYTES if relative == 'model.safetensors.index.json' else
               MAX_CONFIG_BYTES if relative == 'config.json' else
               MAX_METADATA_BYTES)
        _, info = _regular(base, relative, max_bytes=cap)
        preflight_total += info.st_size
        if preflight_total > max_total_bytes:
            _deny('selected HF source exceeds total byte quota')

    rows = []
    tensor_count = 0
    payload_bytes = 0
    for relative in sorted(shards):
        row, names, data_bytes = _safetensors(base, relative)
        if shards[relative] is not None and names != shards[relative]:
            _deny('index weight_map does not match shard header ' + relative)
        tensor_count += len(names)
        if tensor_count > MAX_TENSORS:
            _deny('selected safetensors tensors exceed bound')
        payload_bytes += data_bytes
        rows.append(row)
    if ('total_size' in index_meta and
            index_meta['total_size'] != payload_bytes):
        _deny('safetensors index total_size differs from actual tensor bytes')

    row_paths = {row['path'] for row in rows}
    total_bytes = sum(row['bytes'] for row in rows)
    for relative in sorted(selected - row_paths):
        cap = (MAX_TOKENIZER_BYTES if relative in ('tokenizer.json', 'vocab.json') else
               MAX_INDEX_BYTES if relative == 'model.safetensors.index.json' else
               MAX_CONFIG_BYTES if relative == 'config.json' else
               MAX_METADATA_BYTES)
        path, info = _regular(base, relative, max_bytes=cap)
        try:
            digest = source_freeze._hash_file(path, info, HASH_BLOCK_BYTES)
        except (OSError, source_freeze.SourceFreezeError) as exc:
            raise HfClosureChanged('selected HF file changed: %s: %s' %
                                   (relative, exc)) from exc
        if relative in parsed_hashes and digest != parsed_hashes[relative]:
            raise HfClosureChanged('selected HF JSON changed after parsing: ' + relative)
        total_bytes += info.st_size
        rows.append({'path': relative, 'bytes': info.st_size, 'sha256': digest})
    if total_bytes > max_total_bytes:
        _deny('selected HF source exceeds total byte quota')
    try:
        current_root = source_freeze._lstat(base)
    except (OSError, source_freeze.SourceFreezeError) as exc:
        raise HfClosureChanged('HF source root changed: %s' % exc) from exc
    if _root_identity(current_root) != _root_identity(root_before):
        raise HfClosureChanged('HF source root identity changed during scan')

    rows.sort(key=lambda entry: entry['path'])
    if sum(len(row['path'].encode('utf-8')) + 180 for row in rows) > 6 * 1024**2:
        _deny('selected HF manifest exceeds bounded metadata quota')
    canonical = {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
        'model_key': model_key,
        'layout': 'sharded' if has_index else 'single',
        'entries': rows,
    }
    digest = hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
        'source_root': str(base), 'model_key': model_key,
        'layout': canonical['layout'], 'entries': rows,
        'file_count': len(rows), 'tensor_count': tensor_count,
        'total_bytes': total_bytes, 'tree_sha256': digest,
        'status': STATUS, 'verification_status': 'NOT_CHECKED',
        'release_grade_immutable': False,
    }


def audit_selected_hf_source(directory, *, expected_tree_sha256,
                             expected_model_key=None):
    """Read-only rescan of live source against separately approved exact bytes."""
    approved = _sha(expected_tree_sha256)
    current = scan_selected_hf_source(
        directory, expected_model_key=expected_model_key)
    if current['tree_sha256'] != approved:
        raise HfClosureChanged('HF selected source differs from approved tree SHA')
    return current
