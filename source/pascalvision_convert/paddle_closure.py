# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Read-only, bounded selected Paddle inference resource inventory.

Exactly one local inference.json (PIR) OR inference.pdmodel (opaque legacy
graph), plus inference.pdiparams, is mandatory. Only known, present optional
inference.pdiparams.info, config.yml/.yaml and inference.yml/.yaml are selected.
Recognition (role='rec') additionally REQUIRES a caller-selected, absolute
dictionary reference; neither a guessed filename nor an inference.yml path
can select it implicitly. Detection (role='det') has no dictionary.

This stdlib-only pilot hashes these *selected* files in <=1MiB chunks and
includes the exact dictionary provenance in a deterministic versioned tree
SHA. It inspects supported JSON/YAML reference names conservatively and
refuses unclosed/ambiguous resources. It CANNOT parse all Paddle operators,
Paddle legacy binary graphs, opaque .pdiparams.info or general YAML semantics;
an unrecognised, independently referenced external asset is NOT proven closed.
No file is copied, locked, opened for writing, modified or removed here.

SCANNED_NOT_LOCKED is NOT a Preview source pin, pinned model revision,
compiler-held Win32 stage, WSL exporter compatibility proof, immutable byte
snapshot, target execution, trained-model oracle, human OCR ground truth, CTC
dictionary/class count validation, VERIFIED receipt, or release approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import unicodedata


SCHEMA = 'crosspascal.paddle-selected-source-closure'
SCHEMA_VERSION = 1
STATUS = 'SCANNED_NOT_LOCKED'
MAX_TOTAL_BYTES = 16 * 1024**3
MAX_SELECTED_FILES = 8
MAX_GRAPH_BYTES = 64 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2
MAX_PARAMS_BYTES = MAX_TOTAL_BYTES
MAX_INFO_BYTES = 4 * 1024**2
MAX_YAML_BYTES = 2 * 1024**2
MAX_DICTIONARY_BYTES = 8 * 1024**2
MAX_DICTIONARY_TOKENS = 200_000
MAX_ROOT_ENTRIES = 4096
MAX_JSON_NODES = 200_000
MAX_JSON_DEPTH = 40
MAX_BLOCK_BYTES = 1024 * 1024

_GRAPHS = ('inference.json', 'inference.pdmodel')
_WEIGHTS = 'inference.pdiparams'
_OPTIONAL_INFO = 'inference.pdiparams.info'
_CONFIGS = ('config.yml', 'config.yaml')
_INFER_YAMLS = ('inference.yml', 'inference.yaml')
_KNOWN = frozenset((*_GRAPHS, _WEIGHTS, _OPTIONAL_INFO,
                    *_CONFIGS, *_INFER_YAMLS))
_ESCAPED_OCTET = re.compile(r'%[0-9a-fA-F]{2}')
_SHA = re.compile(r'[0-9a-fA-F]{64}\Z')
_DOS_DEVICES = frozenset(
    ('con', 'prn', 'aux', 'nul', 'com¹', 'com²', 'com³',
     'lpt¹', 'lpt²', 'lpt³') +
    tuple('com%d' % n for n in range(1, 10)) +
    tuple('lpt%d' % n for n in range(1, 10)))
_YAML_KEY_VALUE = re.compile(r'^\s*([A-Za-z_][A-Za-z_0-9.\-]{0,127})\s*:\s*(.*?)\s*$')
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


class PaddleClosureError(ValueError):
    """Unsafe/unknown Paddle resources cannot be represented by this scanner."""


class PaddleClosureChanged(PaddleClosureError):
    """Selected Paddle source or dictionary bytes differ from an approved SHA."""


def _deny(reason):
    raise PaddleClosureError('Paddle selected source refused: ' + reason)


def _changed(reason):
    raise PaddleClosureChanged('Paddle selected source changed: ' + reason)


def _sha(value):
    if type(value) is not str or not _SHA.fullmatch(value):
        _deny('approved SHA-256 must be exactly 64 hexadecimal characters')
    return value.lower()


def _limits(max_total_bytes, block_bytes):
    if (type(max_total_bytes) is not int or
            not 1 <= max_total_bytes <= MAX_TOTAL_BYTES):
        _deny('Paddle selected total-byte quota must be 1..16GiB')
    if type(block_bytes) is not int or not 1 <= block_bytes <= MAX_BLOCK_BYTES:
        _deny('Paddle block_bytes limit must be 1..1048576')


def _safe_name(name):
    if (type(name) is not str or not name or name in ('.', '..') or
            len(name) > 255 or
            unicodedata.normalize('NFC', name) != name or
            name.endswith(('.', ' ')) or _ESCAPED_OCTET.search(name) or
            ':' in name or any(ch in '<>:"/\\|?*' for ch in name) or
            any(unicodedata.category(ch).startswith('C') for ch in name) or
            name.split('.', 1)[0].casefold() in _DOS_DEVICES or
            re.search(r'~[0-9]+(?:\.|$)', name)):
        _deny('unsafe, ambiguous or non-NFC filename component: %r' % name)
    return name


def _raw_absolute(path, label):
    try:
        raw = os.fspath(path)
    except TypeError as error:
        raise PaddleClosureError('%s requires a canonical absolute local path'
                                 % label) from error
    if (type(raw) is not str or not raw or '\x00' in raw or
            raw.startswith(('\\\\', '//')) or not os.path.isabs(raw) or
            unicodedata.normalize('NFC', raw) != raw or
            _ESCAPED_OCTET.search(raw) or
            any(unicodedata.category(ch).startswith('C') for ch in raw)):
        _deny('%s must be an NFC absolute local path, not UNC/device/alias'
              % label)

    drive, suffix = os.path.splitdrive(raw)
    if os.name == 'nt' and (
            len(drive) != 2 or drive[1] != ':' or not suffix.startswith(
                ('\\', '/'))):
        _deny('%s must use a fully qualified local Windows drive path'
              % label)
    components = suffix.replace('\\', '/').split('/')
    if (not components or any(item == '' for item in components[1:]) or
            any(item in ('.', '..') for item in components)):
        _deny('%s has empty/dot/traversal or duplicate path separators' % label)
    for name in components:
        if name:
            _safe_name(name)
    return Path(raw)


def _lstat_unlinked(path):
    try:
        result = path.lstat()
    except OSError as error:
        raise PaddleClosureError(
            'unavailable selected Paddle file/path %s: %s' % (path, error)
        ) from error
    junction = getattr(os.path, 'isjunction', None)
    if (stat.S_ISLNK(result.st_mode) or
            getattr(result, 'st_file_attributes', 0) & _REPARSE or
            junction and junction(str(path))):
        _deny('symlink/junction/reparse selected path refused: ' + str(path))
    return result


def _exact_spelling(path):
    parent = path.parent
    if parent == path:
        return
    match = []
    try:
        with os.scandir(parent) as iterator:
            for count, candidate in enumerate(iterator, 1):
                if count > MAX_ROOT_ENTRIES:
                    _deny('directory exceeds bounded alias-check entry limit')
                if candidate.name.casefold() == path.name.casefold():
                    match.append(candidate.name)
    except OSError as error:
        raise PaddleClosureError(
            'cannot check exact selected path spelling: %s' % error) from error
    if match != [path.name]:
        _deny('case/Unicode alias or missing exact filename: ' + str(path))


def _safe_path(path, *, is_file):
    for item in reversed((path, *path.parents)):
        current = _lstat_unlinked(item)
        if item != path or not is_file:
            if not stat.S_ISDIR(current.st_mode):
                _deny('non-directory Paddle path ancestor: ' + str(item))
        if item.parent != item:
            _exact_spelling(item)
    return _lstat_unlinked(path)


def _checked_file(path, *, limit):
    info = _safe_path(path, is_file=True)
    if not stat.S_ISREG(info.st_mode):
        _deny('selected Paddle resource is not a regular file: ' + str(path))
    if info.st_nlink != 1:
        _deny('selected Paddle resource has hardlink aliases: ' + str(path))
    if not 0 < info.st_size <= limit:
        _deny('selected Paddle file empty or exceeds byte limit: ' + str(path))
    return info


def _identity(info):

    return (info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_dev, info.st_ino, info.st_nlink)


def _descriptor_identity(info):

    return (info.st_mode, info.st_size, info.st_mtime_ns,
            info.st_dev, info.st_ino, info.st_nlink)


def _root_identity(info):

    return info.st_mode, info.st_dev, info.st_ino


def _hash_file(path, *, info=None, max_bytes=MAX_TOTAL_BYTES,
               block_bytes=MAX_BLOCK_BYTES):
    """Full-stream SHA of one selected regular inode; never buffer the tensor."""
    first = info if info is not None else _checked_file(path, limit=max_bytes)
    if first.st_size > max_bytes:
        _deny('Paddle selected source quota exceeded before streaming SHA')
    flags = (os.O_RDONLY | getattr(os, 'O_BINARY', 0) |
             getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0))
    fd = None
    digest = hashlib.sha256()
    size = 0
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            if (_descriptor_identity(os.fstat(stream.fileno())) !=
                    _descriptor_identity(first)):
                _changed('file replaced before SHA read: ' + str(path))
            while True:
                piece = stream.read(block_bytes)
                if not piece:
                    break
                size += len(piece)
                if size > first.st_size or size > max_bytes:
                    _changed('file grew while streaming SHA: ' + str(path))
                digest.update(piece)
            if (_descriptor_identity(os.fstat(stream.fileno())) !=
                    _descriptor_identity(first)):
                _changed('descriptor metadata changed during SHA: ' + str(path))
    except OSError as error:
        raise PaddleClosureChanged(
            'selected file changed/unreadable during SHA: %s: %s' %
            (path, error)) from error
    finally:
        if fd is not None:
            os.close(fd)
    if size != first.st_size:
        _changed('selected file truncated during SHA: ' + str(path))
    if _identity(_checked_file(path, limit=max_bytes)) != _identity(first):
        _changed('selected file path/metadata changed during SHA: ' + str(path))
    return digest.hexdigest(), size


def _read_metadata(path, *, size_limit, approved_sha, block_bytes):
    info = _checked_file(path, limit=size_limit)
    try:
        with path.open('rb') as stream:
            if (_descriptor_identity(os.fstat(stream.fileno())) !=
                    _descriptor_identity(info)):
                _changed('metadata replaced before parse: ' + str(path))
            raw = stream.read(size_limit + 1)
            if len(raw) != info.st_size:
                _changed('metadata truncated/oversized before parse: ' + str(path))
            if (_descriptor_identity(os.fstat(stream.fileno())) !=
                    _descriptor_identity(info)):
                _changed('metadata replaced while parsing: ' + str(path))
    except OSError as error:
        raise PaddleClosureChanged(
            'metadata became inaccessible: %s: %s' % (path, error)) from error
    if hashlib.sha256(raw).hexdigest() != approved_sha:
        _changed('selected metadata buffer differs from approved SHA: ' + str(path))
    actual, size = _hash_file(
        path, info=info, max_bytes=size_limit, block_bytes=block_bytes)
    if actual != approved_sha or size != info.st_size:
        _changed('selected metadata bytes differ after parsing: ' + str(path))
    return raw


def _unique_json_pairs(pairs):
    if len(pairs) > MAX_JSON_NODES:
        _deny('PIR JSON object has too many keys')
    result = {}
    for name, value in pairs:
        if name in result:
            _deny('PIR JSON duplicate key: %r' % name)
        result[name] = value
    return result


def _json_float(value):
    result = float(value)
    if not math.isfinite(result):
        _deny('PIR JSON non-finite number')
    return result


def _reject_json_constant(value):
    _deny('PIR JSON non-finite constant: ' + str(value))


def _resource_key(name):
    low = name.casefold()
    return any(fragment in low for fragment in (
        'path', 'filename', 'file_name', 'external', 'dictionary',
        'dict_', 'vocab', 'weight', 'checkpoint', 'pretrained', 'resource'))


def _json_scalar_safe(value):
    if type(value) is str and (
            len(value) > MAX_JSON_BYTES or
            unicodedata.normalize('NFC', value) != value or
            any(unicodedata.category(char).startswith('C') for char in value)):
        _deny('unsafe/non-NFC Paddle PIR scalar text')


def _reference_allowed(field, value, *, graph_file, role, dictionary_name,
                       dictionary_local, selected):
    """A path-bearing metadata key MUST refer to a known selected resource."""
    if value is None or value == '':
        return
    if type(value) is not str:
        _deny('Paddle unclosed/ambiguous resource reference in ' + field)
    lower = field.casefold()
    if ('dict' in lower or 'vocab' in lower or
            'character' in lower and 'file' in lower):
        if role == 'rec' and value == dictionary_name:
            if dictionary_local:
                return
            _deny('Paddle dictionary reference is relative but caller-selected '
                  'dictionary is EXTERNAL to source root: not proven identical '
                  'at converter runtime')
        _deny('Paddle dictionary reference is not the caller-selected dictionary')
    if lower in ('params_filename', 'params_file', 'params_path'):
        if value == _WEIGHTS:
            return
    elif lower in ('model_filename', 'model_file', 'graph_filename'):
        if value == graph_file:
            return
    elif lower in ('info_filename', 'params_info_filename'):
        if value == _OPTIONAL_INFO and _OPTIONAL_INFO in selected:
            return
    _deny('Paddle graph/YAML resource reference outside exact selected files: '
          '%s=%r' % (field, value))


def _inspect_json_graph(raw, *, graph_file, role, dictionary_name,
                        dictionary_local, selected):
    if len(raw) > MAX_JSON_BYTES:
        _deny('Paddle PIR JSON graph exceeds bounded JSON parser limit')
    try:
        document = json.loads(
            raw.decode('utf-8'), object_pairs_hook=_unique_json_pairs,
            parse_float=_json_float, parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError, TypeError, OverflowError,
            RecursionError) as error:
        raise PaddleClosureError('invalid Paddle PIR JSON graph: %s' %
                                 error) from error
    if type(document) is not dict:
        _deny('Paddle inference.json graph requires a PIR object')

    base_code = document.get('base_code')
    if document.get('magic') == 'pir' and base_code is None:
        pass
    elif (type(base_code) is dict and
            set(base_code) == {'magic', 'trainable', 'version'} and
            base_code.get('magic') == 'pir' and
            type(base_code.get('trainable')) is bool and
            type(base_code.get('version')) is int and
            base_code['version'] == 1 and
            document.get('magic') is None):
        if type(document.get('program')) is not dict:
            _deny('Paddle namespaced PIR graph has no bounded program object')
    else:
        _deny('Paddle inference.json graph requires exact PIR magic=pir')
    stack = [(document, 0)]
    visited = 0
    while stack:
        item, depth = stack.pop()
        visited += 1
        if visited > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _deny('Paddle PIR graph depth/element bound exceeded')
        if type(item) is dict:
            for key, value in item.items():
                if (type(key) is not str or len(key) > 256 or
                        unicodedata.normalize('NFC', key) != key or
                        any(unicodedata.category(char).startswith('C')
                            for char in key)):
                    _deny('PIR JSON has unsafe key')
                if key.casefold() in ('model_type', 'model_role'):
                    if type(value) is str:
                        if value.casefold() not in ('det', 'rec'):
                            _deny('unsupported declared Paddle PIR model role')
                        if value.casefold() != role:
                            _deny('Paddle model role differs from selected det/rec')
                if _resource_key(key):
                    _reference_allowed(
                        key, value, graph_file=graph_file, role=role,
                        dictionary_name=dictionary_name,
                        dictionary_local=dictionary_local, selected=selected)
                if type(value) in (dict, list):
                    stack.append((value, depth + 1))
                else:
                    _json_scalar_safe(value)
        elif type(item) is list:
            if len(item) > MAX_JSON_NODES:
                _deny('Paddle PIR graph list exceeds bounded size')
            for value in item:
                if type(value) in (dict, list):
                    stack.append((value, depth + 1))
                else:
                    _json_scalar_safe(value)


def _inspect_server_export_yaml(source, *, role, dictionary_tokens,
                                graph_file, dictionary_name,
                                dictionary_local, selected):
    """Strict subset used by genuine PP-OCRv5 Server Det/Rec inference.yml.

    Only a single local TensorRT shape anchor+matching alias is permitted;
    arbitrary YAML tags/aliases still fail closed. Rec's large embedded
    character_dict must have exactly the *same ordered Unicode characters*
    as the separately chosen SHA-pinned external Pascal CTC dictionary.
    No broad YAML interpreter, implicit dictionary or network dependency.
    """
    lines = source.splitlines()
    model_name = '  model_name: PP-OCRv5_server_' + role
    if (lines[:2] != ['Global:', model_name] or
            lines.count('Hpi:') != 1 or
            lines.count('PreProcess:') != 1 or
            lines.count('PostProcess:') != 1 or
            lines.count(model_name) != 1):
        _deny('PP-OCRv5 Server YAML model identity/sections differ from selected role')
    post = lines.index('PostProcess:')
    process_name = ('  name: CTCLabelDecode' if role == 'rec'
                    else '  name: DBPostProcess')
    if lines[post + 1:post + 2] != [process_name]:
        _deny('PP-OCRv5 Server YAML postprocess differs from selected role')
    if role == 'rec':
        if dictionary_tokens is None or lines.count('  character_dict:') != 1:
            _deny('Rec YAML embedded character_dict requires the selected dictionary')
        position = lines.index('  character_dict:')
        if position <= post + 1:
            _deny('Rec YAML character_dict is outside approved postprocess')
        embedded = lines[position + 1:]
        if len(embedded) != len(dictionary_tokens) or not embedded:
            _deny('Rec YAML embedded character count differs from selected dictionary')
        for index, line in enumerate(embedded):
            if not line.startswith('  - '):
                _deny('Rec YAML has unexpected content after embedded character_dict')
            scalar = line[4:]
            if scalar.startswith("'"):
                if (len(scalar) < 2 or not scalar.endswith("'") or
                        "'" + scalar[1:-1].replace("''", "'").replace("'", "''")
                        + "'" != scalar):
                    _deny('Rec YAML has noncanonical quoted dictionary character')
                scalar = scalar[1:-1].replace("''", "'")
            elif not scalar or scalar.startswith((' ', '\t')):

                _deny('Rec YAML has ambiguous ASCII-spaced character scalar')
            if scalar != dictionary_tokens[index]:
                _deny('Rec YAML embedded character differs from selected dictionary')
        lines = lines[:position]
    elif '  character_dict:' in lines:
        _deny('Det YAML cannot carry recognition character dictionary')

    anchors = []
    aliases = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        anchor = re.fullmatch(r'      trt_dynamic_shapes: &(id[0-9]{3})', line)
        if anchor:
            anchors.append(anchor.group(1))
            continue
        alias = re.fullmatch(r'      dynamic_shapes: \*(id[0-9]{3})', line)
        if alias:
            aliases.append(alias.group(1))
            continue
        if (any(char in stripped for char in ('!', '&', '*', '\t',
                                             '{', '}', '[', ']')) or
                stripped.startswith(('<<', '---', '...'))):
            _deny('PP-OCRv5 Server YAML has unapproved tag, anchor or flow syntax')
        match = _YAML_KEY_VALUE.match(stripped)
        if match:
            key, value = match.group(1), match.group(2).strip()
            if _resource_key(key):
                if not value:
                    _deny('PP-OCRv5 YAML has unclosed resource reference')
                _reference_allowed(
                    key, value, graph_file=graph_file, role=role,
                    dictionary_name=dictionary_name,
                    dictionary_local=dictionary_local, selected=selected)
        elif _resource_key(stripped):
            _deny('PP-OCRv5 YAML has unparseable resource reference')
    if len(anchors) != 1 or aliases != anchors:
        _deny('PP-OCRv5 Server YAML shape anchor/alias differs or is unclosed')


def _inspect_yaml(raw, *, graph_file, role, dictionary_name,
                  dictionary_local, selected, dictionary_tokens=None):
    """Conservative reference check, NOT a general-purpose YAML parser."""
    try:
        source = raw.decode('utf-8-sig')
    except UnicodeError as error:
        raise PaddleClosureError('invalid Paddle inference YAML encoding') from error
    if '\x00' in source or len(source) > MAX_YAML_BYTES:
        _deny('invalid/oversized Paddle inference YAML')
    if source.count('\n') > 50_000:
        _deny('Paddle YAML line-count bound exceeded')

    if ('  model_name: PP-OCRv5_server_det' in source or
            '  model_name: PP-OCRv5_server_rec' in source):
        return _inspect_server_export_yaml(
            source, role=role, dictionary_tokens=dictionary_tokens,
            graph_file=graph_file, dictionary_name=dictionary_name,
            dictionary_local=dictionary_local, selected=selected)
    for line in source.splitlines():
        if line.lstrip().startswith('#'):
            continue

        if '#' in line:
            marker = re.search(r'\s+#', line)
            if marker is None:
                _deny('ambiguous Paddle YAML hash in scalar/reference')
            line = line[:marker.start()]
        stripped = line.strip()
        if not stripped:
            continue
        if (any(char in stripped for char in ('!', '&', '*', '\t')) or
                any(char in stripped for char in ('{', '}', '[', ']')) or
                stripped.startswith(('<<', '---', '...'))):
            _deny('unsafe YAML anchors/tags/flow collection or ambiguous syntax')
        match = _YAML_KEY_VALUE.match(stripped)
        if match is None:


            if _resource_key(stripped):
                _deny('unparseable Paddle YAML resource reference')
            continue
        key, value = match.group(1), match.group(2).strip()
        if value.startswith(('"', "'")) and value.endswith(('"', "'")):
            value = value[1:-1]
        if key.casefold() in ('model_type', 'model_role'):
            if value and value.casefold() not in ('det', 'rec'):
                _deny('Paddle YAML declares unsupported model role')
            if value.casefold() in ('det', 'rec') and value.casefold() != role:
                _deny('Paddle YAML declares wrong model role det/rec')
        if _resource_key(key):
            if not value:
                _deny('Paddle YAML has unclosed multiline resource reference')
            _reference_allowed(
                key, value, graph_file=graph_file, role=role,
                dictionary_name=dictionary_name,
                dictionary_local=dictionary_local, selected=selected)


def _root_inventory(root):
    names = set()
    aliases = {}
    try:
        with os.scandir(root) as iterator:
            for count, entry in enumerate(iterator, 1):
                if count > MAX_ROOT_ENTRIES:
                    _deny('Paddle source root exceeds bounded entry-count limit')
                name = entry.name
                folded = unicodedata.normalize('NFC', name).casefold()
                other = aliases.setdefault(folded, name)
                if other != name:
                    _deny('case/NFC alias between root members %r and %r' %
                          (other, name))
                if folded in {item.casefold() for item in _KNOWN}:
                    if name not in _KNOWN:
                        _deny('case/Unicode alias of selected Paddle filename: '
                              + name)
                if (name.casefold().endswith(('.pdmodel', '.pdiparams')) and
                        name not in _GRAPHS + (_WEIGHTS,)):
                    _deny('unknown Paddle graph/parameter resource alias: ' + name)
                names.add(name)
    except OSError as error:
        raise PaddleClosureError(
            'cannot enumerate selected Paddle source names: %s' % error) from error
    graph = [name for name in _GRAPHS if name in names]
    if len(graph) != 1:
        _deny('require exactly one Paddle graph inference.json OR inference.pdmodel')
    if _WEIGHTS not in names:
        _deny('Paddle inference.pdiparams is required')
    if len(names.intersection(_CONFIGS)) > 1:
        _deny('ambiguous config.yml/config.yaml resource alias')
    if len(names.intersection(_INFER_YAMLS)) > 1:
        _deny('ambiguous inference.yml/inference.yaml resource alias')
    selected = {graph[0], _WEIGHTS}
    selected.update(names.intersection(
        (_OPTIONAL_INFO, *_CONFIGS, *_INFER_YAMLS)))
    if len(selected) > MAX_SELECTED_FILES:
        _deny('Paddle selected-file count exceeds quota')
    for name in selected:
        _safe_name(name)
    return graph[0], tuple(sorted(selected))


def _dictionary_text(raw):
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError as error:
        raise PaddleClosureError('Paddle dictionary is not UTF-8') from error
    tokens = [part for part in text.splitlines() if part != '']
    if not 0 < len(tokens) <= MAX_DICTIONARY_TOKENS:
        _deny('Paddle dictionary has no tokens or exceeds token bound')
    if any(any(ord(char) < 32 for char in token) for token in tokens):
        _deny('Paddle dictionary contains forbidden control characters')
    return tokens


def scan_selected_paddle_source(directory, *, role, dictionary=None,
                                expected_dictionary_sha256=None,
                                max_total_bytes=MAX_TOTAL_BYTES,
                                block_bytes=MAX_BLOCK_BYTES):
    """Deterministic strict allowlist of local Paddle assets, WITHOUT locking.

    'role' is required det/rec; for rec 'dictionary' must be a full canonical
    local file path explicitly chosen by the caller. For det it must be None.
    A previous expected_dictionary_sha256 is optional for first inventory and
    checked when supplied; the required full-tree pin belongs to audit().
    """
    if type(role) is not str or role not in ('det', 'rec'):
        _deny('Paddle role must explicitly be det or rec')
    if role == 'det' and dictionary is not None:
        _deny('Paddle det cannot silently accept a recognition dictionary')
    if role == 'rec' and dictionary is None:
        _deny('Paddle rec requires a caller-selected dictionary path')
    _limits(max_total_bytes, block_bytes)
    root = _raw_absolute(directory, 'Paddle inference source root')
    root_info = _safe_path(root, is_file=False)
    graph_file, filenames = _root_inventory(root)
    selected = {name: root / name for name in filenames}

    dictionary_path = None
    dictionary_name = None
    dictionary_relative = None
    if role == 'rec':
        dictionary_path = _raw_absolute(dictionary, 'recognition dictionary path')
        dictionary_name = _safe_name(dictionary_path.name)
        dictionary_relative = 'dictionary/' + dictionary_name
        if dictionary_path in selected.values():
            _deny('recognition dictionary aliases selected Paddle graph/params')
    dictionary_local = (
        dictionary_path is not None and dictionary_path.parent == root)

    sizes = {}
    total = 0
    for name, path in sorted(selected.items()):
        cap = (MAX_GRAPH_BYTES if name in _GRAPHS else
               MAX_PARAMS_BYTES if name == _WEIGHTS else
               MAX_INFO_BYTES if name == _OPTIONAL_INFO else MAX_YAML_BYTES)
        sizes[name] = _checked_file(path, limit=cap)
        total += sizes[name].st_size
        if total > max_total_bytes:
            _deny('Paddle selected aggregate bytes exceed quota')
    if dictionary_path is not None:
        try:
            sizes[dictionary_relative] = _checked_file(
                dictionary_path, limit=MAX_DICTIONARY_BYTES)
        except PaddleClosureError as error:

            raise PaddleClosureError(
                'recognition dictionary path refused: %s' % error) from error
        total += sizes[dictionary_relative].st_size
    if total > max_total_bytes:
        _deny('Paddle model plus selected dictionary exceeds aggregate quota')

    rows = []
    selected_paths = {**selected}
    if dictionary_path is not None:
        selected_paths[dictionary_relative] = dictionary_path
    for relative, path in sorted(selected_paths.items()):
        cap = (MAX_GRAPH_BYTES if relative in _GRAPHS else
               MAX_PARAMS_BYTES if relative == _WEIGHTS else
               MAX_DICTIONARY_BYTES if relative == dictionary_relative else
               MAX_INFO_BYTES if relative == _OPTIONAL_INFO else MAX_YAML_BYTES)
        digest, size = _hash_file(
            path, info=sizes[relative], max_bytes=cap, block_bytes=block_bytes)
        rows.append({'path': relative, 'bytes': size, 'sha256': digest})
    hashed = {entry['path']: entry for entry in rows}

    dictionary_tokens = None
    if dictionary_path is not None:


        raw_dictionary = _read_metadata(
            dictionary_path, size_limit=MAX_DICTIONARY_BYTES,
            approved_sha=hashed[dictionary_relative]['sha256'],
            block_bytes=block_bytes)
        dictionary_tokens = _dictionary_text(raw_dictionary)
        if expected_dictionary_sha256 is not None:
            if hashed[dictionary_relative]['sha256'] != _sha(
                    expected_dictionary_sha256):
                _changed('caller-approved dictionary SHA differs')
    elif expected_dictionary_sha256 is not None:
        _deny('dictionary SHA only valid for Paddle rec')

    if graph_file == 'inference.json':
        raw = _read_metadata(
            selected[graph_file], size_limit=MAX_JSON_BYTES,
            approved_sha=hashed[graph_file]['sha256'],
            block_bytes=block_bytes)
        _inspect_json_graph(
            raw, graph_file=graph_file, role=role,
            dictionary_name=dictionary_name,
            dictionary_local=dictionary_local, selected=filenames)
    for name in filenames:
        if name in (*_CONFIGS, *_INFER_YAMLS):
            raw = _read_metadata(
                selected[name], size_limit=MAX_YAML_BYTES,
                approved_sha=hashed[name]['sha256'], block_bytes=block_bytes)
            _inspect_yaml(
                raw, graph_file=graph_file, role=role,
                dictionary_name=dictionary_name,
                dictionary_local=dictionary_local, selected=filenames,
                dictionary_tokens=dictionary_tokens)

    for relative, path in sorted(selected_paths.items()):
        size = sizes[relative].st_size
        again_digest, again_size = _hash_file(
            path, max_bytes=max_total_bytes, block_bytes=block_bytes)
        if again_size != size or again_digest != hashed[relative]['sha256']:
            _changed('selected file changed during Paddle multi-file scan: ' +
                     relative)
    if _root_identity(_safe_path(root, is_file=False)) != (
            _root_identity(root_info)):
        _changed('Paddle source root inode changed while scanning')
    rows.sort(key=lambda row: row['path'])
    dictionary_record = (None if dictionary_path is None else {
        'reference': str(dictionary_path),
        'path': dictionary_relative,
        'bytes': hashed[dictionary_relative]['bytes'],
        'sha256': hashed[dictionary_relative]['sha256']})
    canonical = {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
        'role': role, 'graph_file': graph_file,
        'entries': rows,
        'dictionary': dictionary_record,
    }
    tree_sha = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
        'source_root': str(root), 'role': role, 'graph_file': graph_file,
        'entries': rows, 'file_count': len(rows), 'total_bytes': total,
        'dictionary': dictionary_record, 'tree_sha256': tree_sha,
        'status': STATUS, 'verification_status': 'NOT_CHECKED',
        'release_grade_immutable': False,
    }


def audit_selected_paddle_source(directory, *, role, expected_tree_sha256,
                                 dictionary=None):
    """Read-only rescan; refuse a source different from an earlier approved tree.

    A matching digest does not hold bytes stable between this audit and a
    converter opening them. Private stage/held handles and WSL exact-path
    replay, complete dependency analysis, model numerics and CTC verification
    remain separate mandatory product tasks.
    """
    approved = _sha(expected_tree_sha256)
    current = scan_selected_paddle_source(
        directory, role=role, dictionary=dictionary)
    if current['tree_sha256'] != approved:
        _changed('selected Paddle graph/params/dictionary tree SHA differs')
    return current
