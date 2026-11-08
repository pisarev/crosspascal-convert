# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Convert Hugging Face weights to the model's native parameter layout.

BF16 values are widened exactly to FP32 by shifting their uint16 bits into the
high half of a uint32 value. Q/K/V and gate/up projections are fused to match
the exported model. Tensors are read and serialized one at a time, retaining
only the current fusion inputs and output. Pascal reads the flat binary file
with the accompanying JSON offsets and plain-text parameter index.
"""
import hashlib
import json
import math
import os
import stat
import struct
import sys
import unicodedata

import numpy as np


_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_TENSORS = 100000
_MAX_SHARDS = 4096
_MAX_TENSOR_ELEMENTS = 2000000000
_MAX_DIMS = 16
_MAX_NAME_CHARS = 1024
_MAX_SHARD_PATH_CHARS = 1024
_DTYPE_BYTES = {'F16': 2, 'BF16': 2, 'F32': 4}
_DOS_DEVICES = frozenset(['CON', 'PRN', 'AUX', 'NUL'] +
                         ['COM%d' % i for i in range(1, 10)] +
                         ['LPT%d' % i for i in range(1, 10)])


def _unique_json_object(pairs):
    if len(pairs) > _MAX_TENSORS + 2:
        raise ValueError('REFUSED: too many JSON fields')
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('REFUSED: duplicate JSON key %r' % key)
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError('REFUSED: invalid JSON number %s' % value)


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        _reject_nonfinite(value)
    return number


def _parse_json(raw, description):
    try:
        return json.loads(raw.decode('utf-8'),
                          object_pairs_hook=_unique_json_object,
                          parse_constant=_reject_nonfinite,
                          parse_float=_finite_json_float)
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ValueError('REFUSED: invalid %s: %s' % (description, exc)) from exc


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size,
            getattr(info, 'st_mtime_ns', int(info.st_mtime * 1000000000)))


def _open_binary_regular(path):


    flags = (os.O_RDONLY | getattr(os, 'O_BINARY', 0) |
             getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0) |
             getattr(os, 'O_NONBLOCK', 0))
    descriptor = os.open(path, flags)
    try:
        return os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise


def _checked_entry(path, *, directory=False, min_size=1, max_size=None):
    """lstat every path component, including NTFS reparse points above the leaf."""
    current = os.path.abspath(os.fspath(path))
    leaf = True
    while True:
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise ValueError('REFUSED: missing or inaccessible path %s' % current) from exc
        if (stat.S_ISLNK(info.st_mode) or
                (getattr(info, 'st_file_attributes', 0) & 0x400)):
            raise ValueError('REFUSED: link, junction, or reparse point in path %s' % current)
        if leaf:
            if directory:
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError('REFUSED: expected directory %s' % current)
            elif (not stat.S_ISREG(info.st_mode) or info.st_size < min_size or
                  (max_size is not None and info.st_size > max_size)):
                raise ValueError('REFUSED: invalid file size or type %s' % current)
            result = info
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError('REFUSED: parent is not a directory %s' % current)
        parent = os.path.dirname(current)
        if parent == current:
            return result
        current = parent
        leaf = False


def _safe_shard_name(name):
    """Relative POSIX-style paths are allowed, including nested shard folders."""
    if (type(name) is not str or not name or len(name) > _MAX_SHARD_PATH_CHARS or
            unicodedata.normalize('NFC', name) != name or
            name.startswith('/') or '\\' in name or ':' in name or
            not name.lower().endswith('.safetensors')):
        raise ValueError('REFUSED: invalid shard path %r' % name)
    parts = name.split('/')
    for component in parts:
        if (not component or component in ('.', '..') or
                len(component) > 255 or component.endswith(('.', ' ')) or
                component.split('.')[0].upper() in _DOS_DEVICES or
                any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF or
                    char in '<>"|?*' for char in component)):
            raise ValueError('REFUSED: unsafe shard path component %r' % name)
    return parts


def _valid_tensor_name(name):
    return (type(name) is str and bool(name) and len(name) <= _MAX_NAME_CHARS and
            unicodedata.normalize('NFC', name) == name and
            not any(ord(char) < 32 or char in '\u2028\u2029' or
                    0xD800 <= ord(char) <= 0xDFFF for char in name))


class SafeTensors(object):
    """Read .safetensors without a separate safetensors dependency.

    The first eight bytes give the JSON header length. Tensor offsets are
    relative to the contiguous data region immediately after that header.
    """

    def __init__(self, path, guard=None):
        self.path = os.path.abspath(os.fspath(path))
        self._guard = guard or (lambda: _checked_entry(self.path, min_size=9))
        self._identity = None
        with self._checked_open() as f:
            length = f.read(8)
            if len(length) != 8:
                raise ValueError('REFUSED: truncated safetensors header length')
            head_len = struct.unpack('<Q', length)[0]
            file_size = os.fstat(f.fileno()).st_size
            if not (2 <= head_len <= _MAX_HEADER_BYTES and
                    head_len <= file_size - 8):
                raise ValueError('REFUSED: invalid safetensors header length')
            raw_header = f.read(head_len)
            if len(raw_header) != head_len:
                raise ValueError('REFUSED: truncated safetensors header')
            self.header = _parse_json(raw_header, 'safetensors header')
            self.data_start = 8 + head_len
            self._validate_header(file_size - self.data_start)
            self._identity = _file_identity(os.fstat(f.fileno()))
            if _file_identity(self._guard()) != self._identity:
                raise ValueError('REFUSED: safetensors file changed during parsing')

    def _checked_open(self):
        before = self._guard()
        f = _open_binary_regular(self.path)
        try:
            opened = os.fstat(f.fileno())
            after = self._guard()
            if (_file_identity(before) != _file_identity(opened) or
                    _file_identity(after) != _file_identity(opened) or
                    (self._identity is not None and
                     self._identity != _file_identity(opened))):
                raise ValueError('REFUSED: safetensors file was replaced during open')
        except BaseException:
            f.close()
            raise
        return f

    def _validate_header(self, data_bytes):
        if type(self.header) is not dict:
            raise ValueError('REFUSED: safetensors header is not an object')
        metadata = self.header.get('__metadata__')
        if '__metadata__' in self.header and (
                type(metadata) is not dict or
                any(type(k) is not str or type(v) is not str
                    for k, v in metadata.items())):
            raise ValueError('REFUSED: invalid safetensors __metadata__')
        self.names = sorted(k for k in self.header if k != '__metadata__')
        if not (0 < len(self.names) <= _MAX_TENSORS):
            raise ValueError('REFUSED: invalid safetensors tensor count')
        spans = []
        for name in self.names:
            if not _valid_tensor_name(name):
                raise ValueError('REFUSED: invalid tensor name %r' % name)
            entry = self.header[name]
            if (type(entry) is not dict or
                    set(entry) != {'dtype', 'shape', 'data_offsets'}):
                raise ValueError('REFUSED: invalid fields for tensor %s' % name)
            kind = entry['dtype']
            if type(kind) is not str or kind not in _DTYPE_BYTES:
                raise ValueError('REFUSED: unknown dtype %r for %s' % (kind, name))
            shape = entry['shape']
            if type(shape) is not list or len(shape) > _MAX_DIMS:
                raise ValueError('REFUSED: invalid shape for %s' % name)
            elements = 1
            for dimension in shape:
                if (type(dimension) is not int or dimension <= 0 or
                        dimension > _MAX_TENSOR_ELEMENTS // elements):
                    raise ValueError('REFUSED: invalid or excessive dimensions for %s' % name)
                elements *= dimension
            offsets = entry['data_offsets']
            if (type(offsets) is not list or len(offsets) != 2 or
                    any(type(offset) is not int for offset in offsets)):
                raise ValueError('REFUSED: invalid offsets for %s' % name)
            begin, end = offsets
            if (begin < 0 or end <= begin or end > data_bytes or
                    end - begin != elements * _DTYPE_BYTES[kind]):
                raise ValueError('REFUSED: offsets and byte count disagree for %s' % name)
            spans.append((begin, end))
        cursor = 0
        for begin, end in sorted(spans):
            if begin != cursor:
                raise ValueError('REFUSED: overlapping or noncontiguous safetensors spans')
            cursor = end
        if cursor != data_bytes:
            raise ValueError('REFUSED: unexpected trailing safetensors bytes')

    def dtype_of(self, name):
        return self.header[name]['dtype']

    def shape_of(self, name):
        return tuple(self.header[name]['shape'])

    def read(self, name):
        """Return the tensor as float32 regardless of its source precision."""
        entry = self.header[name]
        begin, end = entry['data_offsets']
        shape = tuple(entry['shape'])
        kind = entry['dtype']
        with self._checked_open() as f:
            f.seek(self.data_start + begin)
            raw = f.read(end - begin)
            if (len(raw) != end - begin or
                    _file_identity(os.fstat(f.fileno())) != self._identity or
                    _file_identity(self._guard()) != self._identity):
                raise ValueError('REFUSED: safetensors data changed or was truncated')
        if kind == 'BF16':
            half = np.frombuffer(raw, dtype='<u2')
            wide = np.zeros(half.shape, dtype='<u4')
            wide[:] = half
            wide <<= 16
            out = wide.view('<f4')
        elif kind == 'F32':
            out = np.frombuffer(raw, dtype='<f4')
        elif kind == 'F16':
            out = np.frombuffer(raw, dtype='<f2').astype('<f4')
        else:
            raise ValueError('REFUSED: unknown weight type %s for %s'
                             % (kind, name))
        if out.size != int(np.prod(shape)):
            raise ValueError('REFUSED: %s contains %d values but its shape requires %d'
                             % (name, out.size, int(np.prod(shape))))
        return out.reshape(shape)


class ShardedSafeTensors(object):
    """Read indexed .safetensors shards through the same interface as one file.

    model.safetensors.index.json maps each tensor name to its shard. The
    conversion plan can use this reader without retaining all shard data.
    """

    def __init__(self, hf_dir):
        self.dir = os.path.abspath(os.fspath(hf_dir))
        root_stat = _checked_entry(self.dir, directory=True)
        self._root_id = (root_stat.st_dev, root_stat.st_ino)
        index_path = os.path.join(self.dir, 'model.safetensors.index.json')
        before = _checked_entry(index_path, max_size=_MAX_INDEX_BYTES)
        with _open_binary_regular(index_path) as handle:
            opened = os.fstat(handle.fileno())
            if _file_identity(opened) != _file_identity(before):
                raise ValueError('REFUSED: weight index was replaced during open')
            raw = handle.read(_MAX_INDEX_BYTES + 1)
            if (len(raw) != opened.st_size or
                    _file_identity(os.fstat(handle.fileno())) !=
                    _file_identity(opened)):
                raise ValueError('REFUSED: weight index changed during reading')
        if _file_identity(_checked_entry(index_path, max_size=_MAX_INDEX_BYTES)) != _file_identity(before):
            raise ValueError('REFUSED: weight index was replaced during parsing')
        idx = _parse_json(raw, 'shard index')
        if type(idx) is not dict or set(idx) - {'metadata', 'weight_map'}:
            raise ValueError('REFUSED: invalid shard index structure')
        if 'metadata' in idx and type(idx['metadata']) is not dict:
            raise ValueError('REFUSED: index metadata is not an object')
        size_claim = idx.get('metadata', {}).get('total_size')
        if ('total_size' in idx.get('metadata', {}) and
                (type(size_claim) is not int or size_claim < 0)):
            raise ValueError('REFUSED: metadata.total_size is not an integer byte count')
        weight_map = idx.get('weight_map')
        if type(weight_map) is not dict or not (0 < len(weight_map) <= _MAX_TENSORS):
            raise ValueError('REFUSED: empty or excessive weight_map')
        seen_paths = {}
        expected_shard_names = {}
        for name, filename in weight_map.items():
            if not _valid_tensor_name(name):
                raise ValueError('REFUSED: invalid weight name %r' % name)
            _safe_shard_name(filename)
            alias = unicodedata.normalize('NFC', filename.casefold())
            if alias in seen_paths and seen_paths[alias] != filename:
                raise ValueError('REFUSED: case or NFC alias in shard paths')
            seen_paths[alias] = filename
            expected_shard_names.setdefault(filename, set()).add(name)
        if len(seen_paths) > _MAX_SHARDS:
            raise ValueError('REFUSED: too many shards')
        self._shard_identities = {}
        for filename in seen_paths.values():
            self._shard_identities[filename] = _file_identity(self._guard_shard(filename))
        self.weight_map = weight_map
        self._expected_shard_names = expected_shard_names
        self.names = sorted(weight_map)
        self._shards = {}

    def _guard_shard(self, filename):
        root = _checked_entry(self.dir, directory=True)
        if (root.st_dev, root.st_ino) != self._root_id:
            raise ValueError('REFUSED: HF model directory changed')
        candidate = os.path.abspath(os.path.join(self.dir, *_safe_shard_name(filename)))
        try:
            if (os.path.commonpath((self.dir, candidate)) != self.dir or
                    os.path.commonpath((os.path.realpath(self.dir),
                                        os.path.realpath(candidate))) !=
                    os.path.realpath(self.dir)):
                raise ValueError('REFUSED: shard escapes the HF directory')
        except ValueError as exc:
            raise ValueError('REFUSED: shard path is outside the HF model') from exc
        inspected = _checked_entry(candidate, min_size=9)
        original = self._shard_identities.get(filename)
        if original is not None and _file_identity(inspected) != original:
            raise ValueError('REFUSED: shard changed after index reading: %s' % filename)
        return inspected

    def _shard(self, name):
        fn = self.weight_map[name]
        self._guard_shard(fn)
        if fn not in self._shards:
            candidate = os.path.join(self.dir, *_safe_shard_name(fn))
            shard = SafeTensors(candidate, guard=lambda fn=fn: self._guard_shard(fn))
            if set(shard.names) != self._expected_shard_names[fn]:
                raise ValueError('REFUSED: weight_map differs from shard %s' % fn)
            self._shards[fn] = shard
        return self._shards[fn]

    def dtype_of(self, name):
        return self._shard(name).dtype_of(name)

    def shape_of(self, name):
        return self._shard(name).shape_of(name)

    def read(self, name):
        return self._shard(name).read(name)


def open_source(hf_path):
    """Open one .safetensors file or a directory with indexed shards."""
    if os.path.isdir(hf_path):
        if os.path.lexists(os.path.join(hf_path, 'model.safetensors.index.json')):
            return ShardedSafeTensors(hf_path)
        single = os.path.join(hf_path, 'model.safetensors')
        if os.path.lexists(single):
            return SafeTensors(single)
        raise ValueError(
            'REFUSED: %s contains neither a shard index nor model.safetensors' % hf_path)
    return SafeTensors(hf_path)


def _fuse_one_at_a_time(source, names, axis=0):
    """Concatenate selected trained weights without retaining all inputs.

    The source reader materializes float32 arrays. Keep only the destination
    and current input live, which matters for full Qwen/Gemma checkpoint builds.
    """
    first = source.read(names[0])
    shape = list(first.shape)
    widths = [shape[axis]]


    for name in names[1:]:
        other_shape = source.shape_of(name)
        if (len(other_shape) != len(shape) or any(
                other_shape[i] != shape[i] for i in range(len(shape)) if i != axis)):
            raise ValueError('REFUSED: incompatible shapes for fused HF weights')
        widths.append(other_shape[axis])
    shape[axis] = sum(widths)
    out = np.empty(shape, dtype=first.dtype)
    cursor = 0
    for index, name in enumerate(names):
        part = first if index == 0 else source.read(name)
        if part.shape[axis] != widths[index] or part.dtype != out.dtype:
            raise ValueError('REFUSED: HF weight shape or dtype changed during fusion')
        segment = [slice(None)] * out.ndim
        segment[axis] = slice(cursor, cursor + widths[index])
        out[tuple(segment)] = part
        cursor += widths[index]
        if index == 0:
            first = None
        del part
    return out


def plan(source, num_layers):
    """Map native parameter names to lazy Hugging Face weight readers.

    Each zero-argument callable produces only the array currently needed.
    """
    take = source.read
    joined = {}

    def fuse(names, axis=0):
        return lambda: _fuse_one_at_a_time(source, names, axis=axis)

    joined['model.embed_tokens.weight'] = lambda: take('model.embed_tokens.weight')
    joined['model.norm.weight'] = lambda: take('model.norm.weight')
                                                                           
                                                                           
    if 'lm_head.weight' in source.names:
        joined['lm_head.weight'] = lambda: take('lm_head.weight')

    for i in range(num_layers):
        at = 'model.layers.%d.' % i
        joined[at + 'self_attn.c_attn.weight'] = fuse([
            at + 'self_attn.q_proj.weight',
            at + 'self_attn.k_proj.weight',
            at + 'self_attn.v_proj.weight'])
        joined[at + 'self_attn.c_attn.bias'] = fuse([
            at + 'self_attn.q_proj.bias',
            at + 'self_attn.k_proj.bias',
            at + 'self_attn.v_proj.bias'])
        joined[at + 'self_attn.o_proj.weight'] = (
            lambda n=at + 'self_attn.o_proj.weight': take(n))
        joined[at + 'mlp.gate_up_proj.weight'] = fuse([
            at + 'mlp.gate_proj.weight',
            at + 'mlp.up_proj.weight'])
        joined[at + 'mlp.down_proj.weight'] = (
            lambda n=at + 'mlp.down_proj.weight': take(n))
        joined[at + 'input_layernorm.weight'] = (
            lambda n=at + 'input_layernorm.weight': take(n))
        joined[at + 'post_attention_layernorm.weight'] = (
            lambda n=at + 'post_attention_layernorm.weight': take(n))

    return joined


def plan_gemma2(source, num_layers):
    """Build the Gemma2 weight plan.

    Q/K/V have no bias. Each layer has four normalization weights, and each
    Gemma RMSNorm weight gains one in float32 to match ordinary RMSNorm.
    """
    take = source.read

    def fuse(names, axis=0):
        return lambda: _fuse_one_at_a_time(source, names, axis=axis)

    def norm(n):

        return lambda name=n: take(name) + np.float32(1.0)

    joined = {}
    joined['model.embed_tokens.weight'] = lambda: take('model.embed_tokens.weight')
    joined['model.norm.weight'] = norm('model.norm.weight')

    for i in range(num_layers):
        at = 'model.layers.%d.' % i
        joined[at + 'self_attn.c_attn.weight'] = fuse([
            at + 'self_attn.q_proj.weight',
            at + 'self_attn.k_proj.weight',
            at + 'self_attn.v_proj.weight'])
        joined[at + 'self_attn.o_proj.weight'] = (
            lambda n=at + 'self_attn.o_proj.weight': take(n))
        joined[at + 'mlp.gate_up_proj.weight'] = fuse([
            at + 'mlp.gate_proj.weight',
            at + 'mlp.up_proj.weight'])
        joined[at + 'mlp.down_proj.weight'] = (
            lambda n=at + 'mlp.down_proj.weight': take(n))
        joined[at + 'input_layernorm.weight'] = norm(at + 'input_layernorm.weight')
        joined[at + 'post_attention_layernorm.weight'] = norm(
            at + 'post_attention_layernorm.weight')
        joined[at + 'pre_feedforward_layernorm.weight'] = norm(
            at + 'pre_feedforward_layernorm.weight')
        joined[at + 'post_feedforward_layernorm.weight'] = norm(
            at + 'post_feedforward_layernorm.weight')

    return joined


def convert(hf_path, out_dir, order, expected_shapes, num_layers, dtype='float32',
            target='cpu', plan_fn=None):
    """Serialize trained weights in the order declared by the exported module.

    The provided names and shapes must match the conversion plan exactly.
    """
    source = open_source(hf_path)
    recipe = (plan_fn or plan)(source, num_layers)

    missing = [n for n in order if n not in recipe]
    if missing:
        raise ValueError('REFUSED: module requests weights absent from the recipe: %s'
                         % ', '.join(missing[:8]))
    extra = [n for n in recipe if n not in order]
    if extra:
        raise ValueError('REFUSED: recipe prepares weights not requested by the module: %s'
                         % ', '.join(extra[:8]))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    bin_path = os.path.join(out_dir, 'params.bin')
    index = []
    offset = 0
    digest = hashlib.sha256()

    with open(bin_path, 'wb') as sink:
        for pos, name in enumerate(order):
            arr = np.ascontiguousarray(recipe[name]().astype(dtype))
            want = tuple(expected_shapes[name])
            if arr.shape != want:
                raise ValueError(
                    'REFUSED: weight %s has shape %s but the module expects %s'
                    % (name, arr.shape, want))
            raw = arr.tobytes()
            sink.write(raw)
            digest.update(raw)
            index.append({'name': name, 'dtype': dtype,
                          'shape': list(arr.shape),
                          'offset': offset, 'nbytes': len(raw)})
            offset += len(raw)
            del arr, raw
            if (pos + 1) % 25 == 0 or pos + 1 == len(order):
                sys.stdout.write('  weights: %d of %d, %.0f MiB\n'
                                 % (pos + 1, len(order), offset / 1048576.0))
                sys.stdout.flush()

    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump({'dtype': dtype, 'total_bytes': offset, 'tensors': index,
                   'sha256': digest.hexdigest()},
                  f, indent=1)

    with open(os.path.join(out_dir, 'params.index'), 'w') as f:
        f.write('# count dtype target, then: name offset nbytes ndim dims...\n')
        f.write('%d %s %s\n' % (len(index), dtype, target))
        for e in index:
            f.write('%s %d %d %d %s\n'
                    % (e['name'], e['offset'], e['nbytes'], len(e['shape']),
                       ' '.join(str(d) for d in e['shape'])))
    return offset, len(index)
