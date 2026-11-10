# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import unicodedata
from urllib.parse import urlsplit

from . import detect

SCHEMA = 'crosspascal.onnx-source-pack.v1'
SOURCE_FILE = 'SOURCE.json'
BUNDLE_FOLDER = 'model-source'
MAX_METADATA = 128 * 1024
MAX_MODEL = 512 * 1024 * 1024
MAX_FILES = 64
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
_CHUNK = 1024 * 1024
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_ID = re.compile(r'[a-z0-9][a-z0-9._-]{0,79}\Z')
_LICENSE = re.compile(r'[A-Za-z0-9][A-Za-z0-9.+-]{0,127}\Z')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {
    base + suffix for base in ('COM', 'LPT')
    for suffix in ('1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '\u00b9', '\u00b2', '\u00b3')}
_GRAPH_CACHE = {}


def normalize_selection(value):
    value = str(value or '').strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if '"' in value:
        raise ValueError('The model path contains an unexpected double quote')
    return value


def _text(value, label, maximum=512):
    if (not isinstance(value, str) or not value or len(value) > maximum
            or any(ord(ch) < 32 for ch in value)):
        raise ValueError(label + ' must be bounded single-line text')
    return value


def _relative(value):
    value = _text(value, 'source-pack path', 240)
    if (value.startswith('/') or '\\' in value or
            unicodedata.normalize('NFC', value) != value):
        raise ValueError('source-pack path must be canonical and relative')
    for part in value.split('/'):
        if (not part or part in ('.', '..') or part.endswith(('.', ' '))
                or any(ch in part for ch in '<>:"|?*')
                or part.split('.')[0].upper() in _RESERVED):
            raise ValueError('unsafe source-pack path: ' + value)
    return value


def _pin(row, size_limit, path_key):
    if not isinstance(row, dict):
        raise ValueError('source-pack file record must be an object')
    name = _relative(row.get(path_key))
    size = row.get('bytes')
    sha = row.get('sha256')
    if type(size) is not int or not 0 < size <= size_limit:
        raise ValueError('source-pack file size is missing or exceeds its limit')
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ValueError('source-pack file needs a lowercase SHA-256')
    return {'path': name, 'bytes': size, 'sha256': sha}


def _regular(path, directory=False):
    info = os.lstat(path)
    if (stat.S_ISLNK(info.st_mode) or
            getattr(info, 'st_file_attributes', 0) & _REPARSE or
            not (stat.S_ISDIR if directory else stat.S_ISREG)(info.st_mode)
            or (not directory and info.st_nlink != 1)):
        raise ValueError('source-pack path is linked or is not a regular file/directory')
    return info


def _path_state(info):
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _common_state(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _parents(path):
    node = Path(os.path.abspath(path)).parent
    while True:
        _regular(node, directory=True)
        if node.parent == node:
            return
        node = node.parent


def _read(path, maximum, expected=None, keep=False):
    _parents(path)
    before = _regular(path)
    if not 0 < before.st_size <= maximum:
        raise ValueError('source-pack file is empty or exceeds its byte limit')
    if expected is not None and before.st_size != expected['bytes']:
        raise ValueError('source-pack file size changed: ' + str(path))
    digest = hashlib.sha256()
    blocks = [] if keep else None
    count = 0
    with open(path, 'rb') as handle:
        opened = os.fstat(handle.fileno())
        if _common_state(opened) != _common_state(before):
            raise ValueError('source-pack file was replaced at open')
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            count += len(block)
            if count > before.st_size:
                raise ValueError('source-pack file grew while reading')
            digest.update(block)
            if keep:
                blocks.append(block)
        if _path_state(os.fstat(handle.fileno())) != _path_state(opened):
            raise ValueError('source-pack open file changed while reading')
    if count != before.st_size or _path_state(_regular(path)) != _path_state(before):
        raise ValueError('source-pack path changed while reading')
    sha = digest.hexdigest()
    if expected is not None and sha != expected['sha256']:
        raise ValueError('source-pack SHA-256 differs: ' + str(path))
    return {'bytes': count, 'sha256': sha, 'identity': list(_path_state(before))}, (
        b''.join(blocks) if keep else None)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate SOURCE.json key: ' + key)
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError('non-finite SOURCE.json number: ' + value)


def validate_document(data, model_name=None):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError('unsupported adjacent SOURCE.json schema')
    if not isinstance(data.get('profile_id'), str) or not _ID.fullmatch(data['profile_id']):
        raise ValueError('source-pack profile_id is invalid')
    _text(data.get('display_name'), 'display_name', 160)
    if not isinstance(data.get('license'), str) or not _LICENSE.fullmatch(data['license']):
        raise ValueError('source-pack license must be one SPDX identifier')
    model = _pin(data.get('model'), MAX_MODEL, 'file')
    if ('/' in model['path'] or not model['path'].lower().endswith('.onnx')
            or data['model'].get('self_contained') is not True
            or (model_name is not None and model['path'] != model_name)):
        raise ValueError('SOURCE.json must select this self-contained ONNX basename')
    rows = data.get('files')
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_FILES:
        raise ValueError('source-pack ancillary file count is invalid')
    files = [_pin(row, MAX_FILE, 'path') for row in rows]
    if sum(row['bytes'] for row in files) > MAX_TOTAL:
        raise ValueError('source-pack ancillary bytes exceed the limit')
    paths = {}
    for row in [{'path': SOURCE_FILE}, model] + files:
        parts = row['path'].split('/')
        for index in range(1, len(parts) + 1):
            name = '/'.join(parts[:index])
            kind = 'file' if index == len(parts) else 'directory'
            key = name.casefold()
            old = paths.get(key)
            if old is not None and (old != (name, kind) or kind == 'file'):
                raise ValueError('source-pack duplicate, alias or file/directory collision')
            paths[key] = (name, kind)
    names = {row['path'] for row in files}
    for row in rows:
        _text(row.get('role'), 'ancillary role', 80)
    if _relative(data.get('license_file')) not in names:
        raise ValueError('source-pack license_file must be a declared ancillary file')
    upstream = data.get('upstream')
    if not isinstance(upstream, dict):
        raise ValueError('source-pack upstream record is required')
    upstream_pin = _pin(upstream, 8 * 1024 ** 3, 'file')
    for key in ('publisher', 'repository', 'revision'):
        _text(upstream.get(key), 'upstream.' + key)
    url = urlsplit(_text(upstream.get('url'), 'upstream.url', 2048))
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError('upstream.url must be a public HTTPS URL without credentials')
    if 'export' not in data:
        raise ValueError('source-pack export must be null or an explicit record')
    exported = data['export']
    if exported is None:
        if any(upstream_pin[key] != model[key] for key in ('bytes', 'sha256')):
            raise ValueError('direct upstream ONNX identity differs from model identity')
    elif isinstance(exported, dict):
        for key in ('recipe_file', 'environment_file'):
            if _relative(exported.get(key)) not in names:
                raise ValueError('export.' + key + ' must be a declared ancillary file')
    else:
        raise ValueError('source-pack export must be null or an object')
    conversion = data.get('conversion')
    if not isinstance(conversion, dict):
        raise ValueError('source-pack conversion record is required')
    task = conversion.get('task')
    adapter = conversion.get('adapter')
    if (task not in ('classification', 'object_detection')
            or adapter not in ('yolo11', 'rtdetr')
            or (task == 'classification' and adapter != 'yolo11')):
        raise ValueError('unsupported source-pack Pascal vision task/adapter')
    name = _text(conversion.get('input_name'), 'input_name', 160)
    if any(ch.isspace() or ch in ':,;' for ch in name):
        raise ValueError('input_name cannot contain shape separators or whitespace')
    shape = conversion.get('input_shape')
    if (not isinstance(shape, list) or len(shape) != 4
            or any(type(dim) is not int or not 0 < dim <= 4096 for dim in shape)
            or shape[:2] != [1, 3]
            or conversion.get('dtype') != 'float32'
            or conversion.get('layout') != 'NCHW'):
        raise ValueError('source pack requires one batch-1 RGB float32 NCHW input')
    if conversion.get('classes') != (1000 if task == 'classification' else 80):
        raise ValueError('source-pack class count does not match its vision contract')
    mean, std = conversion.get('mean'), conversion.get('std')
    if (not isinstance(mean, list) or not isinstance(std, list)
            or len(mean) != len(std) or len(mean) not in (0, 3)
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for x in mean + std) or any(x <= 0 for x in std)):
        raise ValueError('source-pack normalization must be two empty or RGB arrays')
    for field in ('score_threshold', 'nms_threshold'):
        value = conversion.get(field)
        if value is not None and (type(value) not in (int, float)
                                 or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('source-pack threshold must be null or a finite value in [0,1]')
        required = task == 'object_detection' and (
            field == 'score_threshold' or adapter == 'yolo11')
        if (value is None) == required:
            raise ValueError('source-pack threshold presence differs from its adapter')
    return model, files


def _inspect_onnx(payload, conversion):
    import onnx
    from .onnx_closure import _external_tensors

    model = onnx.load_model_from_string(payload)
    if _external_tensors(model):
        raise ValueError('source packs require ONNX with all tensor data inside the file')
    initializer_names = {row.name for row in model.graph.initializer}
    inputs = [row for row in model.graph.input if row.name not in initializer_names]
    if len(inputs) != 1 or inputs[0].name != conversion['input_name']:
        raise ValueError('ONNX graph input names differ from SOURCE.json')
    tensor = inputs[0].type.tensor_type
    dims = tensor.shape.dim
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(dims) != 4:
        raise ValueError('ONNX graph must expose one float32 rank-four input')
    for index, (dim, bound) in enumerate(zip(dims, conversion['input_shape'])):
        if dim.HasField('dim_value'):
            if dim.dim_value != bound:
                raise ValueError('ONNX fixed input dimensions differ from SOURCE.json')
        elif index != 0:
            raise ValueError('only the ONNX batch dimension may need a static binding')


def discover(model_path):
    model_path = normalize_selection(model_path)
    if not model_path.lower().endswith('.onnx') or not os.path.isabs(model_path):
        return None
    model_path = os.path.abspath(model_path)
    root = os.path.dirname(model_path)
    metadata_path = os.path.join(root, SOURCE_FILE)
    if not os.path.lexists(metadata_path):
        return None
    source_pin, payload = _read(metadata_path, MAX_METADATA, keep=True)
    data = json.loads(payload.decode('utf-8'), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)
    model, files = validate_document(data, os.path.basename(model_path))
    graph_key = (model['sha256'], json.dumps(data['conversion'], sort_keys=True))
    needs_graph = graph_key not in _GRAPH_CACHE
    if needs_graph:
        detect._admit_onnx_preparse(model_path)
    actual_model, model_bytes = _read(model_path, MAX_MODEL, model, keep=needs_graph)
    if needs_graph:
        _inspect_onnx(model_bytes, data['conversion'])
        if len(_GRAPH_CACHE) >= 16:
            _GRAPH_CACHE.clear()
        _GRAPH_CACHE[graph_key] = True
    measured = []
    for item in files:
        observed, _ = _read(os.path.join(root, *item['path'].split('/')),
                            MAX_FILE, item)
        measured.append(dict(item, identity=observed['identity']))
    after, _ = _read(metadata_path, MAX_METADATA, source_pin)
    if after != source_pin:
        raise ValueError('SOURCE.json changed while its selected files were checked')
    return {'metadata': data, 'source': source_pin, 'model': actual_model,
            'files': measured, 'root': root}


def selection(pack):
    return {'sha256': pack['source']['sha256'],
            'profile_id': pack['metadata']['profile_id']}


def display(pack):
    data = pack['metadata']
    return dict(selection(pack), display_name=data['display_name'],
                license=data['license'], upstream=data['upstream'],
                conversion=data['conversion'],
                note='Source declaration checked against selected bytes; not legal clearance')


def require_request(pack, request):
    if request.get('onnx_source_pack') != selection(pack):
        raise ValueError('Source pack changed or is not recognized; press Recognise again')
    if (request.get('full_build') or request.get('target') != 'cpu'
            or request.get('target_os') != 'windows'
            or request.get('target_cpu_arch') != 'x86_64'
            or request.get('tune') is not False or request.get('cpu_codegen')
            or request.get('dtype') not in ('', 'float32', None)
            or request.get('freeze_local_raw_source')
            or request.get('atomic_complete_bundle')):
        raise ValueError('This source pack requires its untuned Windows x86_64 CPU profile')
    c = pack['metadata']['conversion']
    shape = c['input_name'] + ':' + ','.join(str(x) for x in c['input_shape'])
    for key, expected in (('task', c['task']), ('adapter', c['adapter']),
                          ('layout', c['layout']), ('shape', shape)):
        if request.get(key) != expected:
            raise ValueError('Build ' + key + ' differs from the source-pack profile')
    for key in ('mean', 'std'):
        raw = request.get(key)
        try:
            value = [] if not raw else [float(x.strip()) for x in raw.split(',')]
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError('Invalid source-pack normalization request') from exc
        if value != c[key]:
            raise ValueError('Build ' + key + ' differs from the source-pack profile')
    for key in ('score_threshold', 'nms_threshold'):
        raw = request.get(key)
        value = None if raw in (None, '') else float(raw)
        if value != c.get(key):
            raise ValueError('Build threshold differs from the source-pack profile')
    if request.get('labels') or request.get('keypoint_count'):
        raise ValueError('Additional labels/keypoints are not part of this source-pack profile')


def selected(request):
    pack = discover(request.get('path')) if not request.get('full_build') else None
    if pack is not None:
        require_request(pack, request)
    elif request.get('onnx_source_pack'):
        raise ValueError('The recognized SOURCE.json is missing; press Recognise again')
    return pack


def validate_binding(value):
    if (not isinstance(value, dict) or value.get('schema') != SCHEMA
            or value.get('path') != BUNDLE_FOLDER + '/' + SOURCE_FILE
            or not isinstance(value.get('sha256'), str)
            or not _SHA.fullmatch(value['sha256'])
            or not isinstance(value.get('model_sha256'), str)
            or not _SHA.fullmatch(value['model_sha256'])
            or not isinstance(value.get('profile_id'), str)
            or not _ID.fullmatch(value['profile_id'])
            or not isinstance(value.get('license'), str)
            or not _LICENSE.fullmatch(value['license'])
            or not isinstance(value.get('upstream'), dict)):
        raise ValueError('Invalid manifest source-pack binding')
    return value


def binding(pack):
    data = pack['metadata']
    return {'schema': SCHEMA, 'path': BUNDLE_FOLDER + '/' + SOURCE_FILE,
            'sha256': pack['source']['sha256'],
            'profile_id': data['profile_id'], 'license': data['license'],
            'model_sha256': data['model']['sha256'],
            'upstream': data['upstream']}


def install(pack, model_path, output_directory):
    if discover(model_path) != pack:
        raise ValueError('Source pack changed since Preview')
    destination = os.path.join(output_directory, BUNDLE_FOLDER)
    os.mkdir(destination)
    entries = [{'path': SOURCE_FILE, **pack['source']}] + pack['files']
    for entry in entries:
        relative = entry['path']
        source_path = os.path.join(pack['root'], *relative.split('/'))
        _, payload = _read(source_path, MAX_METADATA if relative == SOURCE_FILE
                           else MAX_FILE, entry, keep=True)
        target = os.path.join(destination, *relative.split('/'))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _parents(target)
        with open(target, 'xb') as handle:
            if handle.write(payload) != len(payload):
                raise OSError('short write while preserving source-pack metadata')
    if discover(model_path) != pack:
        raise ValueError('Source pack changed while preserving its metadata')
    return binding(pack)


def validate_collected(entries, pack, passport):
    expected = {BUNDLE_FOLDER + '/' + SOURCE_FILE: pack['source']}
    expected.update({BUNDLE_FOLDER + '/' + row['path']: row for row in pack['files']})
    observed = {row.name: row for row in entries
                if row.name.startswith(BUNDLE_FOLDER + '/')}
    if set(observed) != set(expected):
        raise ValueError('Download source-pack files differ from the approved selection')
    for name, pin in expected.items():
        row = observed[name]
        if row.size != pin['bytes'] or row.sha256 != pin['sha256']:
            raise ValueError('Download source-pack content changed: ' + name)
    if (not isinstance(passport, dict)
            or passport.get('license') != pack['metadata']['license']
            or (passport.get('provenance') or {}).get('source_pack') != binding(pack)
            or (passport.get('provenance') or {}).get('source_hash') !=
            pack['metadata']['model']['sha256']):
        raise ValueError('Download manifest does not bind the approved source pack')

