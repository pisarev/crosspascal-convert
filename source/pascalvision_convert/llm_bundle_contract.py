# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Fail-closed, stdlib-only inspection of native LLM compilation outputs.

This is NOT an LLM numerical oracle, a source/weight SHA snapshot, or an
atomic directory commit. It checks that the *actual on-disk* native compiler
outputs describe the approved BuildPlan and the Pascal consumer's real ABI.
The stricter consumer mode additionally requires tokenizer/runtime/licences.
Neither mode can issue VERIFIED from metadata.
"""

import hashlib
import json
import os
from pathlib import Path
import stat

from .build_plan import validate_recorded
from .manifest import emit_manifest_text


_MAX_JSON = 8 * 1024 * 1024
_MAX_INDEX = 8 * 1024 * 1024
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_FAMILIES = {
    'qwen2.5-0.5b': ('qwen2', 'qwen2', 'byte_bpe',
                      ('tokenizer-tokens.bin', 'tokenizer-tokens.index',
                       'tokenizer-merges.bin', 'tokenizer-specials.txt',
                       'tokenizer.index')),
    'gemma-2-2b': ('gemma-2-2b', 'gemma2', 'sentencepiece',
                    ('tokenizer-tokens.bin', 'tokenizer-tokens.index',
                     'tokenizer-merges.bin', 'tokenizer-chars.bin',
                     'tokenizer-bytes.bin', 'tokenizer-specials.txt',
                     'tokenizer.index')),
}
_RUNTIME = ('tvm_runtime.dll', 'tvm_runtime_extra.dll', 'tvm_ffi.dll')


def _normal_root(directory):
    root = Path(directory).absolute()
    if os.path.normcase(os.path.realpath(root)) != os.path.normcase(str(root)):
        raise ValueError('native LLM output directory redirects through a link')
    if not root.is_dir():
        raise ValueError('native LLM output directory is unavailable')
    if root.is_symlink() or getattr(root, 'is_junction', lambda: False)():
        raise ValueError('native LLM output directory is a link/junction')
    return root


def _regular(root, name, *, max_bytes=None):
    if ('/' in name or '\\' in name or ':' in name or
            name in ('', '.', '..') or name.endswith(('.', ' '))):
        raise ValueError('unsafe native LLM bundle member %r' % name)
    path = root / name
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError('native LLM output missing %s: %s' % (name, exc)) from exc
    if (not stat.S_ISREG(info.st_mode) or
            bool(getattr(info, 'st_file_attributes', 0) & _REPARSE)):
        raise ValueError('native LLM output %s is linked or not a regular file' % name)
    if info.st_size <= 0 or (max_bytes is not None and info.st_size > max_bytes):
        raise ValueError('native LLM output %s is empty or too large' % name)
    return path, info.st_size


def _read_bounded(root, name, limit, *, approved_sha256=None):
    path, size = _regular(root, name, max_bytes=limit)
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(payload) != size or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('native LLM output %s changed while reading' % name)
    if (approved_sha256 is not None and
            hashlib.sha256(payload).hexdigest() != approved_sha256):
        raise ValueError('native LLM output %s changed since initial ZIP inventory'
                         % name)
    return payload


def _approved_sha(approved_sha256s, name):
    """Require a complete trusted original ZIP pin for every semantic reread.

    Standalone builder routes intentionally pass None: they do not possess an
    external selected ZIP authority and keep their existing API contract.
    """
    if approved_sha256s is None:
        return None
    if type(approved_sha256s) is not dict:
        raise ValueError('native LLM approved ZIP SHA mapping is invalid')
    sha = approved_sha256s.get(name)
    if (type(sha) is not str or len(sha) != 64 or
            any(ch not in '0123456789abcdef' for ch in sha)):
        raise ValueError('native LLM missing/malformed original ZIP SHA for %s'
                         % name)
    return sha


def _pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key %r' % key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON value %r' % value)


def _json(root, name, *, approved_sha256=None):
    try:
        return json.loads(_read_bounded(
            root, name, _MAX_JSON, approved_sha256=approved_sha256).decode('utf-8'),
                          object_pairs_hook=_pairs_unique,
                          parse_constant=_reject_constant)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('native LLM %s JSON invalid: %s' % (name, exc)) from exc


def _hash_file(root, name, *, approved_size=None, approved_sha256=None):
    try:
        path, expected = _regular(root, name)
    except ValueError as exc:
        raise ValueError('native LLM library is missing/unsafe: %s' % exc) from exc
    if approved_size is not None:


        if type(approved_size) is not int or approved_size <= 0:
            raise ValueError('native LLM library approved size is invalid')
        if expected != approved_size:
            raise ValueError('native LLM library changed since initial ZIP inventory')
    digest = hashlib.sha256()
    nbytes = 0
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected:
            raise ValueError('native LLM library changed before SHA validation')


        remaining = expected
        while remaining:
            part = stream.read(min(1024 * 1024, remaining))
            if not part or len(part) > remaining:
                raise ValueError('native LLM library changed during bounded SHA validation')
            digest.update(part)
            nbytes += len(part)
            remaining -= len(part)
        if stream.read(1):
            raise ValueError(
                'native LLM library grew beyond admitted byte size during SHA validation')
        after = os.fstat(stream.fileno())
    if (nbytes != expected or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('native LLM library changed during SHA validation')
    measured_sha = digest.hexdigest()
    if approved_sha256 is not None and measured_sha != approved_sha256:


        raise ValueError('native LLM library changed since initial ZIP inventory')
    return measured_sha


def _match(actual, expected, what):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('native LLM %s differs from BuildPlan/consumer: %r vs %r'
                         % (what, actual, expected))


def _check_parameters(root, model, dtype, device, *,
                      approved_params_size=None, approved_sha256s=None):
    params = _json(root, 'params.json',
                   approved_sha256=_approved_sha(approved_sha256s, 'params.json'))
    if not isinstance(params, dict):
        raise ValueError('native LLM params.json must contain an object')
    _match(params.get('dtype'), dtype, 'params dtype')
    tensors = params.get('tensors')
    order = model.get('param_order')
    if (not isinstance(tensors, list) or not tensors or len(tensors) > 100000 or
            not isinstance(order, list) or len(order) != len(tensors)):
        raise ValueError('native LLM param_order differs from params.json tensors')
    bin_path, disk_bytes = _regular(root, 'params.bin')
    if approved_params_size is not None:


        if type(approved_params_size) is not int or approved_params_size <= 0:
            raise ValueError('native LLM params.bin approved size is invalid')
        if disk_bytes != approved_params_size:
            raise ValueError('native LLM params.bin changed since initial ZIP inventory')
    total_bytes = params.get('total_bytes')
    if type(total_bytes) is not int or total_bytes != disk_bytes:
        raise ValueError('native LLM params.bin length differs from params.json')
    if model.get('llm_profile_id') is not None or 'params_sha256' in model:
        expected_sha = model.get('params_sha256')
        if (type(expected_sha) is not str or len(expected_sha) != 64 or
                any(ch not in '0123456789abcdef' for ch in expected_sha)):
            raise ValueError('native LLM weight digest is missing or invalid')
        _match(model.get('params_bytes'), total_bytes, 'weight byte count')
        _match(params.get('sha256'), expected_sha, 'serialized weight digest')
        if approved_sha256s is not None:
            _match(_approved_sha(approved_sha256s, 'params.bin'), expected_sha,
                   'inventory weight digest')
    raw_index = _read_bounded(
        root, 'params.index', _MAX_INDEX,
        approved_sha256=_approved_sha(approved_sha256s, 'params.index'))
    try:
        lines = raw_index.decode('utf-8').splitlines()
    except UnicodeError as exc:
        raise ValueError('native LLM params.index encoding invalid') from exc
    if (len(lines) != len(tensors) + 2 or not lines[0].startswith('#') or
            lines[1].split() != [str(len(tensors)), dtype, device]):
        raise ValueError('native LLM params.index header/count/dtype/device mismatch')
    offset = 0
    seen = set()
    bytes_per_scalar = {'float32': 4, 'float16': 2}[dtype]
    for pos, item in enumerate(tensors):
        if not isinstance(item, dict):
            raise ValueError('native LLM params.json tensor is not an object')
        name = item.get('name')
        shape = item.get('shape')
        if (not isinstance(name, str) or not name or name in seen or
                order[pos] != name or item.get('dtype') != dtype or
                not isinstance(shape, list) or not shape or
                any(type(dim) is not int or dim <= 0 for dim in shape)):
            raise ValueError('native LLM param_order/shape/dtype mismatch')
        seen.add(name)
        expected_bytes = bytes_per_scalar
        for dim in shape:
            expected_bytes *= dim
            if expected_bytes > disk_bytes:
                raise ValueError('native LLM params.bin tensor exceeds file length')
        if (type(item.get('offset')) is not int or item['offset'] != offset or
                type(item.get('nbytes')) is not int or
                item['nbytes'] != expected_bytes):
            raise ValueError('native LLM params.json offset/nbytes mismatch')
        expected_row = ([name, str(offset), str(expected_bytes),
                         str(len(shape))] + [str(dim) for dim in shape])
        if lines[pos + 2].split() != expected_row:
            raise ValueError('native LLM params.index tensor shape/offset mismatch')
        offset += expected_bytes
    if offset != total_bytes:
        raise ValueError('native LLM params.bin total_bytes/count mismatch')


def validate_llm_output(directory, plan, model_key, *, require_consumer=False,
                        approved_library_size=None, approved_params_size=None,
                        approved_member_sha256s=None):
    """Check structural native artifacts; optionally insist on portable closure.

    Native/adapter can use require_consumer=False to check compiled outputs;
    receipt/public download MUST request True. The returned status never claims
    independent numerical verification, including on a successful compilation.
    """
    validate_recorded(plan)
    if plan['route'] != 'family_full_build':
        raise ValueError('native LLM outputs require a family BuildPlan')
    if model_key not in _FAMILIES:
        raise ValueError('native LLM model_key is not a supported exact profile')
    stem, adapter, tokenizer, tokenizer_files = _FAMILIES[model_key]
    root = _normal_root(directory)
    deploy = plan['target_profile']['deployment']
    host = plan['target_profile']['build_host']
    device = deploy['device']
    dtype = plan['options']['dtype']
    sm = deploy['sm'] or ''
    if deploy['os'] != 'windows' or deploy['arch'] not in ('x86_64', 'arm64'):
        raise ValueError('native LLM output contract currently requires Windows Pascal ABI')
    library = '%s-%s.dll' % (stem, device)
    model = _json(
        root, 'model.json',
        approved_sha256=_approved_sha(approved_member_sha256s, 'model.json'))
    if not isinstance(model, dict):
        raise ValueError('native LLM model.json must contain an object')
    for key, expected in (
        ('model_key', model_key),
        ('library', library),
        ('target', device),
        ('arch', sm),
        ('dtype', dtype),
        ('build_plan_sha256', plan['plan_sha256']),
        ('build_host_cpu_arch', host['arch']),
        ('tokenizer', tokenizer),
    ):
        _match(model.get(key), expected, ('BuildPlan ' if key == 'build_plan_sha256'
                                         else '') + key)
    selected_llm = plan['options'].get('llm')
    if selected_llm is not None:
        from . import model_sources
        official = model_sources.catalog_model(model_key)
        for key, expected in (
                ('model_key', selected_llm['model_key']),
                ('context_window_size', selected_llm['context_window_size']),
                ('prefill_chunk_size', selected_llm['prefill_chunk_size']),
                ('max_batch_size', selected_llm['max_batch_size']),
                ('llm_profile_id', selected_llm['profile_id']),
                ('source_repo', official['repository']),
                ('source_revision', official['revision']),
                ('source_tree_sha256', official['tree_sha256'])):
            _match(model.get(key), expected, 'selected LLM profile ' + key)
    lib_sha = _hash_file(
        root, library, approved_size=approved_library_size,
        approved_sha256=_approved_sha(approved_member_sha256s, library))
    passport = _json(
        root, 'manifest.json',
        approved_sha256=_approved_sha(approved_member_sha256s, 'manifest.json'))
    if not isinstance(passport, dict):
        raise ValueError('native LLM manifest.json must be an object')
    if 'license' in passport:


        raise ValueError('native LLM model license is not attested by child')
    artifact = passport.get('artifact')
    if not isinstance(artifact, dict):
        raise ValueError('native LLM manifest artifact missing')
    if passport.get('task_kind') != 'text_generation':
        raise ValueError('native LLM manifest task differs from text_generation')
    _match(passport.get('adapter'), adapter, 'manifest adapter')
    for field, expected in (
        ('file', library), ('os', deploy['os']), ('arch', deploy['arch']),
        ('device', device), ('dtype', dtype), ('format', 'dll'),
    ):
        _match(artifact.get(field), expected, 'manifest artifact ' + field)
    _match(artifact.get('device_arch'), sm or None, 'manifest artifact GPU arch')
    _match(artifact.get('hash'), lib_sha, 'manifest artifact hash')
    if passport.get('tokenizer') != {'style': tokenizer}:
        raise ValueError('native LLM manifest tokenizer mismatches model.json')
    generation = passport.get('text')
    if not isinstance(generation, dict):
        raise ValueError('native LLM manifest text section is missing')
    bos = model.get('bos_token_id')
    eos = model.get('eos_token_id')
    if isinstance(bos, list):
        bos = bos[0] if bos else None
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    for field, expected in (
        ('context_window', model.get('context_window_size')),
        ('bos', bos), ('eos', eos),
    ):
        _match(generation.get(field), expected, 'manifest text ' + field)
    try:
        actual_text = _read_bounded(
            root, 'manifest.txt', _MAX_JSON,
            approved_sha256=_approved_sha(approved_member_sha256s,
                                          'manifest.txt'))
        permitted = (emit_manifest_text(passport, newline='\n').encode('utf-8'),
                     emit_manifest_text(passport, newline='\r\n').encode('utf-8'))
    except (TypeError, ValueError, KeyError, RecursionError) as exc:
        raise ValueError('native LLM manifest text/JSON invalid: %s' % exc) from exc
    if actual_text not in permitted:
        raise ValueError('native LLM manifest JSON/text disagreement')
    _check_parameters(root, model, dtype, device,
                      approved_params_size=approved_params_size,
                      approved_sha256s=approved_member_sha256s)
    if require_consumer:
        for filename in tokenizer_files:
            try:
                _regular(root, filename)
            except ValueError as exc:
                raise ValueError('native LLM tokenizer closure incomplete: %s' % exc) from exc
        for filename in _RUNTIME + (('tvm_runtime_cuda.dll',) if device == 'cuda' else ()):
            try:
                _regular(root, filename)
            except ValueError as exc:
                raise ValueError('native LLM runtime closure incomplete: %s' % exc) from exc
        licenses = list(root.glob('LICENSE-tvm-*'))
        if not licenses:
            raise ValueError('native LLM licence closure incomplete: no TVM licence')
        for license_file in licenses:
            _regular(root, license_file.name)
        if plan['options'].get('llm') is not None:
            from .model_sources import catalog_model
            expected_license = catalog_model(model_key)['license_file']
            legal_name = 'LICENSE-Qwen2.5-0.5B-Instruct.txt'
            _hash_file(root, legal_name, approved_size=expected_license['bytes'],
                       approved_sha256=expected_license['sha256'])
            _regular(root, 'MODEL-SOURCE.json', max_bytes=_MAX_JSON)
            notices = list(root.glob('NOTICE-tvm-*'))
            if not notices:
                raise ValueError('native LLM licence closure incomplete: no TVM FFI notice')
            for notice in notices:
                _regular(root, notice.name)
    return {
        'model_key': model_key, 'artifact': library,
        'artifact_sha256': lib_sha, 'verification_status': 'NOT_CHECKED',
        'consumer_closure_checked': bool(require_consumer),
    }
