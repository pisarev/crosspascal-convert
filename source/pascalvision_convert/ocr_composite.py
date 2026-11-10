# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from . import bundle_atomic
from .manifest import emit_manifest_text
from .targets import validate_recorded_profile


PLAN_VERSION = 2
PLAN_NAME = 'composite-plan.json'
_DET_BGR_MEAN = (0.406, 0.456, 0.485)
_DET_BGR_STD = (0.225, 0.224, 0.229)
_PINNED_PROCESS_ENV = frozenset((
    'PPOCR_KEEP_BGR', 'PPOCR_NORM',
    'PPOCR_CTC_BLANK_ID', 'PPOCR_CTC_TEMPERATURE',
    'PPOCR_CTC_LOGIT_TEMP', 'PPOCR_CTC_BLANK_MARGIN_ON',
    'PPOCR_CTC_BLANK_MARGIN', 'PADDLE_OCR_INFER_YML',
))
_HEX64 = re.compile(r'[0-9a-fA-F]{64}\Z')
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_DICTIONARY_BYTES = 8 * 1024 * 1024
_MAX_ASSET_BYTES = bundle_atomic.MAX_NATIVE_FAMILY_BYTES
_MAX_ASSET_AGGREGATE_BYTES = bundle_atomic.MAX_NATIVE_FAMILY_BYTES
_MAX_SELECTED_MEMBERS = bundle_atomic.MAX_MEMBERS


_MAX_DIRECTORY_ENTRIES = max(4096, bundle_atomic.MAX_MEMBERS * 8)
_MAX_DICT_TOKENS = 65536
_REQUIRED_RUNTIME = ('tvm_runtime', 'tvm_runtime_extra', 'tvm_ffi')
_CUDA_RUNTIME = 'tvm_runtime_cuda'
_VENDOR_DLL = re.compile(
    r'^(?:lib)?(?:cudart|cudnn|cublaslt|cublas|cufft|curand|cusolver|'
    r'cusparse|cupti|nvrtc|nvjitlink|nvptxcompiler|nccl|nvcuda|cuda)'
    r'[\w.-]*\.(?:dll|dylib|so(?:\.\d+)*)$', re.I)
_TASKS = {
    'det': frozenset(('object_detection', 'text_detection')),
    'rec': frozenset(('text_recognition', 'ocr_recognition',
                      'optical_character_recognition', 'classification')),
}


class CompositeContractError(ValueError):
    pass


def _reject(reason):
    raise CompositeContractError(reason)


def _dict(value, label):
    if not isinstance(value, dict):
        _reject('%s must be a JSON object' % label)
    return value


def _required(value, key, label):
    if key not in value or value[key] is None:
        _reject('%s is missing required field %s' % (label, key))
    return value[key]


def _sha256_valid(value, label):
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _reject('%s requires an actual SHA-256 hexadecimal digest' % label)
    return value.lower()


def _not_link(path):
    current = path
    missing_leaf = False
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if current != path or missing_leaf:
                _reject('missing parent folder: %s' % current)
            missing_leaf = True
        else:
            if (stat.S_ISLNK(metadata.st_mode)
                    or (getattr(metadata, 'st_file_attributes', 0)
                        & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))):
                _reject('symbolic link or Windows reparse point is forbidden: %s'
                        % current)
        if current.parent == current:
            break
        current = current.parent


def _declared_path(value, label):
    raw = os.fspath(value)
    if not isinstance(raw, str) or '\x00' in raw:
        _reject('%s path must be a nonempty string' % label)
    if '..' in raw.replace('\\', '/').split('/'):
        _reject('%s path traversal (..) is forbidden' % label)
    return Path(os.path.abspath(raw))


def _existing_directory(directory):
    path = _declared_path(directory, 'source directory')
    _not_link(path)
    if not path.is_dir():
        _reject('bundle source directory does not exist: %s' % path)
    return path


def _safe_name(name):
    if (not isinstance(name, str) or not name or len(name) > 255
            or name in ('.', '..') or name.endswith(('.', ' '))
            or any(c in name for c in '/\\:\x00')
            or any(ord(c) < 32 for c in name)
            or re.fullmatch(r'(?:con|prn|aux|nul)(?:\..*)?', name, re.I)
            or re.fullmatch(r'(?:com|lpt)[1-9](?:\..*)?', name, re.I)):
        _reject('unsafe asset filename / path traversal: %r' % name)
    return name


def _source_file(directory, name):
    name = _safe_name(name)
    file = directory / name
    _not_link(file)
    try:
        mode = file.lstat().st_mode
    except FileNotFoundError:
        _reject('required bundle asset is missing: %s' % file)
    if not stat.S_ISREG(mode):
        _reject('bundle asset is not a regular file: %s' % file)
    return file


def _digest(file, *, max_bytes=_MAX_ASSET_BYTES):
    sha = hashlib.sha256()
    length = 0
    before = file.stat()
    if before.st_size > max_bytes:
        _reject('asset exceeds %d-byte limit before SHA: %s'
                % (max_bytes, file))
    with file.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _reject('source asset FileID changed before SHA: %s' % file)
        if opened.st_size > max_bytes:
            _reject('asset exceeds %d-byte limit before SHA: %s'
                    % (max_bytes, file))
        while True:
            block = stream.read(min(1 << 20, max_bytes - length + 1))
            if not block:
                break
            length += len(block)
            if length > max_bytes:
                _reject('asset grew beyond byte limit during SHA: %s' % file)
            sha.update(block)
    if not length:
        _reject('bundle asset is empty: %s' % file)
    return {'sha256': sha.hexdigest(), 'bytes': length}


def _read_bytes(file, *, limit):
    with file.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size > limit:
            _reject('metadata/dictionary file exceeds %d bytes: %s'
                    % (limit, file))
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        _reject('metadata/dictionary file grew beyond %d bytes: %s'
                % (limit, file))
    return payload


def _duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _reject('duplicate JSON key %r is ambiguous' % key)
        result[key] = value
    return result


def _read_json(file):
    try:
        return json.loads(_read_bytes(file, limit=_MAX_METADATA_BYTES)
                          .decode('utf-8-sig'),
                          object_pairs_hook=_duplicate_json_keys,
                          parse_constant=lambda value: _reject(
                              'non-finite JSON constant %s' % value))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _reject('invalid UTF-8 JSON at %s: %s' % (file, exc))


def _asset(directory, filename, destination, *, expected_sha=None,
           max_bytes=_MAX_ASSET_BYTES):
    path = _source_file(directory, filename)
    if path.stat().st_size > max_bytes:
        _reject('asset exceeds %d-byte limit before SHA: %s'
                % (max_bytes, path))
    record = _digest(path, max_bytes=max_bytes)
    if expected_sha and record['sha256'] != _sha256_valid(
            expected_sha, 'manifest artifact hash'):
        _reject('SHA-256 mismatch for %s: manifest expects %s but file is %s'
                % (path, expected_sha, record['sha256']))
    return {'source': path, 'path': destination,
            'sha256': record['sha256'], 'bytes': record['bytes']}


def _unique_asset_names(assets):
    names = [asset['path'].casefold() for asset in assets]
    if len(set(names)) != len(names):
        _reject('duplicate/case-colliding asset paths would overwrite a file')


def _report_says_verified(file):
    try:
        lines = _read_bytes(file, limit=_MAX_METADATA_BYTES).decode(
            'utf-8-sig').splitlines()
    except UnicodeDecodeError:
        _reject('report.txt must be UTF-8: %s' % file)
    required = {
        'VERIFICATION: VERIFIED_LOCAL',
        'RESULT: VERIFIED',
        '[done] Comparison with the reference',
        '[done] Writing the bundle manifest',
    }
    if not required.issubset({line.strip() for line in lines}):
        _reject('report.txt does not prove VERIFIED_LOCAL and completed '
                'reference + individual manifest gates: %s' % file)
    if any(re.match(r'^\s*\[(?:refused|not checked)\]', line, re.I)
           for line in lines):
        _reject('report.txt includes FAILED/NOT_CHECKED step: %s' % file)
    if any(re.match(r'^\s*(?:VERIFICATION|RESULT)\s*:', line)
           and line.strip() not in required for line in lines):
        _reject('report.txt includes a conflicting verification state: %s'
                % file)


def _input_sidecar(file, io):
    try:
        text = _read_bytes(file, limit=_MAX_METADATA_BYTES).decode('utf-8-sig')
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            index, name, dimensions = line.split('\t')
            dims = [int(dim) for dim in dimensions.split(',')]
            rows.append((int(index), name, dims))
    except (ValueError, UnicodeError) as exc:
        _reject('invalid .inputs sidecar %s: %s' % (file, exc))
    if rows != [(0, io['name'], io['shape'])]:
        _reject('individual .inputs sidecar disagrees with manifest io or '
                'declares extra/dynamic inputs: %s' % file)


def _target_record(manifest, recipe, role):
    artifact = _dict(_required(manifest, 'artifact', role), 'artifact')
    os_name = _required(artifact, 'os', 'artifact')
    arch = _required(artifact, 'arch', 'artifact')
    device = _required(artifact, 'device', 'artifact')
    if os_name not in ('windows', 'linux', 'darwin'):
        _reject('%s has unsupported target OS %r' % (role, os_name))
    if arch not in ('x86_64', 'arm64'):
        _reject('%s has unsupported CPU architecture %r' % (role, arch))
    if device not in ('cpu', 'cuda'):
        _reject('%s has unsupported device %r' % (role, device))
    if recipe.get('target') != device:
        _reject('%s recipe target disagrees with manifest device' % role)
    sm = artifact.get('device_arch')
    profile = manifest.get('target_profile')
    recorded_recipe_profile = recipe.get('target_profile')
    if (profile is None) != (recorded_recipe_profile is None):
        _reject('%s recipe/manifest target profiles disagree' % role)
    if profile is not None:
        if profile != recorded_recipe_profile:
            _reject('%s recipe/manifest target profiles have different fields'
                    % role)
        try:
            validate_recorded_profile(
                profile, target=device, req_sm=recipe.get('sm'))
        except (TypeError, ValueError) as exc:
            _reject('%s invalid recorded target profile: %s' % (role, exc))
        dep = profile['deployment']
        if profile['reachability']['level'] != 'LOCAL':
            _reject('%s cannot claim VERIFIED_LOCAL with a recorded '
                    'non-local target reachability level' % role)
        if (dep['os'], dep['arch'], dep['device']) != (os_name, arch, device):
            _reject('%s profile target differs from manifest artifact' % role)
        if dep.get('sm') != sm:
            _reject('%s GPU SM differs between profile and artifact' % role)
        driver = dep.get('driver_api')
    else:
        driver = None
    if device == 'cuda':
        if (profile is None or sm is None or sm != recipe.get('sm')
                or isinstance(driver, bool) or not isinstance(driver, int)
                or driver < 1000):
            _reject('%s CUDA target needs explicit matching GPU SM and '
                    'driver_api in VERIFIED_LOCAL manifest + recipe profile'
                    % role)
    elif sm is not None or recipe.get('sm') is not None or driver is not None:
        _reject('%s CPU bundle unexpectedly declares GPU SM or CUDA driver'
                % role)
    return {'os': os_name, 'arch': arch, 'device': device,
            'sm': sm if device == 'cuda' else None,
            'driver_api': driver if device == 'cuda' else None}


def _bounded_dirnames(directory, role):
    names = []
    with os.scandir(directory) as stream:
        for entry in stream:
            if len(names) >= _MAX_DIRECTORY_ENTRIES:
                _reject('%s folder exceeds bounded directory entry quota '
                        'before source SHA' % role)
            names.append(entry.name)
    return names


def _runtime_assets(directory, role, os_name, device):
    suffix = {'windows': '.dll', 'linux': '.so',
              'darwin': '.dylib'}[os_name]
    logical_components = _REQUIRED_RUNTIME + (
        (_CUDA_RUNTIME,) if device == 'cuda' else ())
    all_names = _bounded_dirnames(directory, role)
    listed = {name.casefold(): name for name in all_names}
    if len(listed) != len(all_names):
        _reject('%s folder contains case-colliding runtime/asset filenames'
                % role)

    licences = sorted(name for name in all_names
                      if name.startswith('LICENSE-tvm-'))
    if not licences:
        _reject('%s has no copied TVM wheel licence files' % role)
    if len(licences) + len(logical_components) + 6 + 4 > _MAX_SELECTED_MEMBERS:
        _reject('%s selected runtime/licence member count exceeds quota '
                'before SHA' % role)

    runtime = {}
    assets = []
    for logical in logical_components:
        expected = [(logical + suffix), ('lib' + logical + suffix)]
        found = [listed[e.casefold()] for e in expected
                 if e.casefold() in listed]
        if len(found) != 1:
            _reject('%s needs one unambiguous %s runtime library (found %s)'
                    % (role, logical, found))
        asset = _asset(directory, found[0], '%s/%s' % (role, found[0]))
        assets.append(asset)
        runtime[logical] = {'file': found[0], 'sha256': asset['sha256']}
    for name in all_names:
        lower = name.casefold()
        if _VENDOR_DLL.fullmatch(name):
            _reject('%s has unmanaged CUDA toolkit dependency %s; no driver '
                    'compatibility has been established for that library'
                    % (role, name))
        if (lower.endswith(('.dll', '.dylib', '.so'))
                and (re.search(r'(?:^|lib)tvm_(?:runtime|ffi)', lower))
                and name not in {record['file'] for record in runtime.values()}):
            _reject('%s has extra ambiguous TVM runtime %s' % (role, name))
    for name in licences:
        assets.append(_asset(directory, name, '%s/%s' % (role, name)))
    return assets, runtime


def _read_individual_bundle(path, role):
    root = _existing_directory(path)
    manifest_path = _source_file(root, 'manifest.json')
    recipe_path = _source_file(root, 'recipe.json')
    text_path = _source_file(root, 'manifest.txt')
    report_path = _source_file(root, 'report.txt')
    manifest = _dict(_read_json(manifest_path), 'manifest.json')
    recipe = _dict(_read_json(recipe_path), 'recipe.json')
    for kind, record in (('manifest', manifest), ('recipe', recipe)):
        if (record.get('verification_status') is not None
                and record['verification_status'] != 'VERIFIED_LOCAL'):
            _reject('%s %s explicitly says FAILED/NOT_CHECKED or another '
                    'non-VERIFIED_LOCAL status' % (role, kind))
    if manifest.get('format_version') != 1:
        _reject('%s manifest has unsupported format_version' % role)
    if (isinstance(recipe.get('recipe_version'), bool)
            or recipe.get('recipe_version') not in (1, 2)):
        _reject('%s recipe has unsupported recipe_version' % role)
    if (recipe.get('target_profile') is None) != (
            recipe.get('recipe_version') == 1):
        _reject('%s recipe_version conflicts with target_profile presence'
                % role)
    _report_says_verified(report_path)
    try:
        expected_text = emit_manifest_text(manifest)
        saved_text = _read_bytes(
            text_path, limit=_MAX_METADATA_BYTES).decode('utf-8-sig')
    except (TypeError, ValueError, UnicodeError) as exc:
        _reject('%s manifest.txt cannot be reconciled with JSON: %s'
                % (role, exc))
    if saved_text.replace('\r\n', '\n') != expected_text:
        _reject('%s original manifest.txt disagrees with manifest.json' % role)

    kind = _required(manifest, 'task_kind', 'manifest')
    adapter = _required(manifest, 'adapter', 'manifest')
    if (kind not in _TASKS[role]
            or not isinstance(adapter, str)
            or not ('paddle' in adapter.lower() or 'ppocr' in adapter.lower())
            or (role == 'rec' and kind == 'classification'
                and 'rec' not in adapter.lower())):
        _reject('%s has no PaddleOCR %s adapter/task identity: %s/%s'
                % (role, role, adapter, kind))
    if (isinstance(manifest.get('adapter_api_version'), bool)
            or not isinstance(manifest.get('adapter_api_version'), int)
            or manifest['adapter_api_version'] <= 0):
        _reject('%s adapter_api_version is invalid' % role)
    recipe_contract = _dict(
        _required(recipe, 'contract', 'recipe'), 'recipe.contract')
    if (recipe_contract.get('task') != kind
            or recipe_contract.get('adapter') != adapter
            or recipe_contract.get('adapter_api_version', 1)
            != manifest['adapter_api_version']):
        _reject('%s recipe contract contradicts verified manifest task/adapter'
                % role)
    artifact = _dict(_required(manifest, 'artifact', role), 'artifact')
    if (artifact.get('format') != 'so' or artifact.get('entry') != 'main'
            or artifact.get('dtype') != 'float32'):
        _reject('%s artifact has unsupported format/entry/dtype' % role)
    suffix = {'windows': '.dll', 'linux': '.so',
              'darwin': '.dylib'}.get(artifact.get('os'))
    model_name = _safe_name(_required(artifact, 'file', 'artifact'))
    if not suffix or not model_name.lower().endswith(suffix):
        _reject('%s artifact filename extension disagrees with its OS' % role)
    model = _asset(root, model_name, '%s/%s' % (role, model_name),
                   expected_sha=_required(artifact, 'hash', 'artifact'))

    io = _dict(_required(manifest, 'io', role), 'manifest.io')
    shape = _required(io, 'shape', 'io')
    if (io.get('dtype') != 'float32' or io.get('layout') != 'NCHW'
            or not isinstance(io.get('name'), str) or not io['name']
            or not isinstance(shape, list) or len(shape) != 4
            or any(isinstance(d, bool) or not isinstance(d, int) or d <= 0
                   for d in shape) or shape[0:2] != [1, 3]):
        _reject('%s OCR currently requires exactly one static [1,3,H,W] '
                'NCHW float32 input in the individual manifest' % role)
    if recipe_contract.get('layout', 'NCHW') != io['layout']:
        _reject('%s recipe layout differs from manifest io' % role)
    hint = recipe.get('shape')
    if hint:
        if not isinstance(hint, str) or ':' not in hint:
            _reject('%s recipe shape hint is malformed' % role)
        input_name, _, raw_shape = hint.rpartition(':')
        try:
            hinted = [int(d) for d in raw_shape.split(',')]
        except ValueError:
            _reject('%s recipe shape hint contains non-integers' % role)
        if (input_name != io['name'] or hinted != shape):
            _reject('%s recipe input shape/name differs from manifest io'
                    % role)
    sidecar_name = model_name + '.inputs'
    sidecar = _asset(root, sidecar_name, '%s/%s' % (role, sidecar_name),
                     max_bytes=_MAX_METADATA_BYTES)
    _input_sidecar(sidecar['source'], io)

    source = _dict(_required(manifest, 'provenance', role),
                   'manifest.provenance')
    original_name = _required(source, 'source', 'manifest.provenance')
    original_sha = _sha256_valid(
        _required(source, 'source_hash', 'manifest.provenance'),
        'original model SHA-256')
    built_at = _required(source, 'built_at', 'manifest.provenance')
    if (not isinstance(original_name, str) or not original_name
            or not isinstance(built_at, str) or not built_at
            or (original_name, original_sha, built_at)
            != (recipe.get('source'), recipe.get('source_hash'),
                recipe.get('built_at'))):
        _reject('%s original source provenance differs across manifest/recipe'
                % role)
    toolchain = _dict(_required(manifest, 'toolchain', role),
                      'manifest.toolchain')
    if (toolchain != recipe.get('toolchain')
            or not all(isinstance(toolchain.get(key), str)
                       and toolchain[key] not in ('', 'unknown')
                       for key in ('tvm', 'onnx'))):
        _reject('%s toolchain is unknown or differs between manifest and recipe'
                % role)
    if manifest.get('license') != recipe.get('license'):
        _reject('%s source licence differs between recipe and manifest' % role)
    target = _target_record(manifest, recipe, role)
    runtime_assets, runtime = _runtime_assets(
        root, role, target['os'], target['device'])

    asset_names = ('manifest.json', 'manifest.txt',
                   'recipe.json', 'report.txt')
    assets = [model, sidecar] + [
        _asset(root, name, '%s/%s' % (role, name),
               max_bytes=_MAX_METADATA_BYTES)
        for name in asset_names] + runtime_assets
    _unique_asset_names(assets)
    return {
        'root': root, 'role': role, 'manifest': manifest, 'recipe': recipe,
        'target': target, 'runtime': runtime, 'assets': assets,
        'source_provenance': source, 'toolchain': toolchain,
    }


def _exact_numeric(value, expected, label):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value != expected):
        _reject('%s must be %s as implemented by native TVM.PaddleOcr.pas'
                % (label, expected))


def _exact_channels(value, expected, label):
    if not isinstance(value, list) or len(value) != len(expected):
        _reject('%s must list exactly three ordered BGR channel coefficients'
                % label)
    for index, (got, want) in enumerate(zip(value, expected)):
        _exact_numeric(got, want, '%s[%d]' % (label, index))


def _validate_preprocessing(value, sources):
    value = _dict(value, 'preprocessing')
    if set(value) != {'det', 'rec', 'native_process_env'}:
        _reject('preprocessing needs det, rec and explicit native_process_env; '
                'the legacy DET/REC settings are not native-equivalent')
    stages = {}
    common = {
        'source_color', 'model_color', 'normalization',
        'resize', 'interpolation', 'layout', 'dtype',
    }
    for role in ('det', 'rec'):
        row = _dict(value[role], 'preprocessing.' + role)
        extra = ({'pixel_scale', 'mean', 'std'} if role == 'det' else
                 {'pixel_scale', 'stride', 'width_rounding', 'padding_side',
                  'padding_domain', 'padding_raw_value',
                  'padding_tensor_value'})
        if set(row) != common | extra:
            _reject('preprocessing.%s must explicitly supply exactly %s'
                    % (role, sorted(common | extra)))


        expected_model_color = 'BGR' if role == 'det' else 'RGB'
        if (row['source_color'] != 'BGR'
                or row['model_color'] != expected_model_color):
            _reject('preprocessing.%s requires BGR source and %s model '
                    'channel order in native Pascal' %
                    (role, expected_model_color))
        if row['pixel_scale'] != 'divide_by_255':
            _reject('preprocessing.%s requires explicit divide_by_255 '
                    'before normalization' % role)
        expected_norm = ('channelwise_mean_std_after_divide_by_255'
                         if role == 'det' else
                         'minus_one_to_one_after_divide_by_255')
        if row['normalization'] != expected_norm:
            _reject('preprocessing.%s normalization must be %s, not a '
                    'generic or legacy mode' % (role, expected_norm))
        if row['interpolation'] != 'bilinear':
            _reject('preprocessing.%s must use native INTER_LINEAR/bilinear'
                    % role)
        expected_resize = ('stretch_to_model_input' if role == 'det'
                           else 'height_aspect_stride_right_zero_pad')
        if row['resize'] != expected_resize:
            _reject('preprocessing.%s resize must explicitly name %s'
                    % (role, expected_resize))
        manifest_io = sources[role]['manifest']['io']
        if (row['layout'] != manifest_io['layout']
                or row['dtype'] != manifest_io['dtype']):
            _reject('preprocessing.%s tensor layout/dtype conflicts with '
                    'the verified manifest' % role)
        if role == 'det':
            _exact_channels(row['mean'], _DET_BGR_MEAN, 'det mean_BGR')
            _exact_channels(row['std'], _DET_BGR_STD, 'det std_BGR')
            formula = '(float32(raw_BGR)/255 - mean_BGR) / std_BGR'
        else:
            if row['stride'] != 8 or type(row['stride']) is not int:
                _reject('preprocessing.rec stride must be exactly 8')
            if row['width_rounding'] != 'ceil_clamp_floor_stride_min1':
                _reject('preprocessing.rec width rounding must match Pascal: '
                        'ceil, clamp to model width, floor to stride, min 1')
            if (row['padding_side'] != 'right'
                    or row['padding_domain']
                    != 'raw_bgr_before_color_conversion_and_normalization'):
                _reject('preprocessing.rec padding must be on the right in '
                        'the raw BGR image before normalization')
            _exact_numeric(row['padding_raw_value'], 0,
                           'preprocessing.rec padding_raw_value')
            _exact_numeric(row['padding_tensor_value'], -1,
                           'preprocessing.rec padding_tensor_value')
            formula = 'float32(BGR_to_RGB(raw_BGR_zero_padded))/255*2 - 1'
        stages[role] = dict(
            row, input_name=manifest_io['name'],
            input_shape=list(manifest_io['shape']),
            normalization_formula=formula,
        )
    return stages


def _validate_native_process_env(declared, ctc_record, dictionary_asset):
    declared = _dict(declared, 'preprocessing.native_process_env')
    if set(declared) != _PINNED_PROCESS_ENV:
        _reject('native_process_env must explicitly pin %s (and no aliases)'
                % sorted(_PINNED_PROCESS_ENV))
    expected = {
        'PPOCR_KEEP_BGR': '0',
        'PPOCR_NORM': '2',
        'PPOCR_CTC_BLANK_ID': str(ctc_record['blank_index']),
        'PPOCR_CTC_TEMPERATURE': '1',
        'PPOCR_CTC_LOGIT_TEMP': str(ctc_record['logit_temperature']),
        'PPOCR_CTC_BLANK_MARGIN_ON': '0',
        'PPOCR_CTC_BLANK_MARGIN': None,
        'PADDLE_OCR_INFER_YML': None,
    }
    for name, required_value in expected.items():
        if declared[name] != required_value or (
                required_value is not None and type(declared[name]) is not str):
            _reject('native_process_env.%s must be exactly %r to match '
                    'Pascal normalization/CTC; environment aliases, defaults '
                    'and unstated overrides are forbidden'
                    % (name, required_value))
    return {
        'required_set': {name: value for name, value in expected.items()
                         if value is not None},
        'required_unset': [name for name, value in expected.items()
                           if value is None],
        'resolved_asset_binding': {
            'PADDLE_OCR_DICT': {
                'bundle_relative_path': dictionary_asset['path'],
                'sha256': dictionary_asset['sha256'],
                'resolution': 'absolute_path_on_target_before_native_start',
            },
        },
        'enforcement': 'REQUIRED_BEFORE_NATIVE_PROCESS_START',
        'enforced_and_measured': False,
        'ambient_environment_not_trusted': True,
    }


def _dictionary_tokens(file):
    try:
        raw = _read_bytes(file, limit=_MAX_DICTIONARY_BYTES)
        lines = raw.decode('utf-8-sig').splitlines()
    except UnicodeError as exc:
        _reject('dictionary is not valid UTF-8: %s' % exc)


    tokens = [line for line in lines if line != '']
    if not tokens or len(tokens) > _MAX_DICT_TOKENS:
        _reject('dictionary is empty or exceeds the class-count cap')
    if any('\x00' in token or '\t' in token
           or any(ord(c) < 32 for c in token) for token in tokens):
        _reject('dictionary contains unsupported control characters')
    return tokens


def _validate_ctc(value, tokens):
    value = _dict(value, 'ctc')
    required = {
        'decoder', 'expected_classes', 'blank_index', 'blank_slot',
        'use_space_char', 'logits_layout', 'collapse_repeats', 'skip_blank',
        'logit_temperature', 'blank_margin_on',
    }
    if set(value) != required:
        _reject('ctc must explicitly include exactly %s' % sorted(required))
    classes = value['expected_classes']
    if (isinstance(classes, bool) or not isinstance(classes, int)
            or not 2 <= classes <= _MAX_DICT_TOKENS + 1):
        _reject('CTC expected_classes must be an explicit finite class count')
    if value['decoder'] != 'greedy':
        _reject('only explicit greedy CTC decoding is represented')
    if value['blank_slot'] not in ('included', 'excluded'):
        _reject('CTC blank_slot must say included or excluded in dictionary')
    if not isinstance(value['use_space_char'], bool):
        _reject('CTC use_space_char must be an explicit boolean')
    if value['use_space_char'] is not True:
        _reject('native TPaddleOcr.Recognize hardcodes '
                'PpocrLoadDict(Dict, True): use_space_char=False would '
                'misrepresent the actual native dictionary')
    if (type(value['blank_index']) is not int
            or value['blank_index'] not in (0, classes - 1)):
        _reject('CTC blank_index must explicitly choose class 0 or last')
    if (value['logits_layout'] not in ('NTC', 'NCT', 'TC', 'CT')
            or value['collapse_repeats'] is not True
            or value['skip_blank'] is not True):
        _reject('CTC logits layout, repeated token collapse and blank handling '
                'must be explicit and supported')
    temperature = value['logit_temperature']
    if (isinstance(temperature, bool)
            or not isinstance(temperature, (float, int))
            or not math.isfinite(temperature)
            or temperature <= 0 or temperature > 100):
        _reject('CTC logit_temperature must be an explicit positive number')
    if value['blank_margin_on'] is not False:
        _reject('CTC blank margin rescue is unsupported by the draft contract')
    raw_count = len(tokens)
    adjusted = raw_count + int(value['use_space_char'])
    required_count = adjusted + (value['blank_slot'] == 'excluded')
    if required_count != classes:
        _reject('dictionary/CTC classes mismatch: %d raw tokens + %d appended '
                'space + %d separate blank class = %d, but expected_classes=%d'
                % (raw_count, int(value['use_space_char']),
                   int(value['blank_slot'] == 'excluded'), required_count,
                   classes))
    return dict(
        value,
        raw_dictionary_tokens=raw_count,
        explicit_space_tokens=sum(token == ' ' for token in tokens),
        appended_space_tokens=int(value['use_space_char']),
        effective_dictionary_tokens=adjusted,
        total_classes_by_dictionary=required_count,
        expected_classes_evidence='caller declaration (native output unmeasured)',
        class_count_runtime_gate_required=True,
    )


def _preflight_selected_byte_quota(det_bundle, rec_bundle, dictionary):
    selected = []
    roots = []
    for role, supplied in (('det', det_bundle), ('rec', rec_bundle)):
        root = _existing_directory(supplied)
        roots.append(root)
        manifest = _dict(_read_json(_source_file(root, 'manifest.json')),
                         'preflight %s manifest' % role)
        artifact = _dict(_required(manifest, 'artifact', role),
                         'preflight artifact')
        module = _safe_name(_required(artifact, 'file', 'preflight artifact'))
        suffix = {'windows': '.dll', 'linux': '.so',
                  'darwin': '.dylib'}.get(artifact.get('os'))
        device = artifact.get('device')
        if suffix is None:
            _reject('preflight unsupported target OS %r' % artifact.get('os'))
        if device not in ('cpu', 'cuda'):
            _reject('preflight unsupported device %r' % device)
        if not module.lower().endswith(suffix):
            _reject('preflight artifact suffix disagrees with target OS')
        items = ['manifest.json', 'manifest.txt', 'recipe.json',
                 'report.txt', module, module + '.inputs']
        all_names = _bounded_dirnames(root, role)
        listed = {name.casefold(): name for name in all_names}
        if len(listed) != len(all_names):
            _reject('%s preflight found case-colliding asset filenames' % role)
        logical = _REQUIRED_RUNTIME + (
            (_CUDA_RUNTIME,) if device == 'cuda' else ())
        for library in logical:
            possible = (library + suffix, 'lib' + library + suffix)
            found = [listed[name.casefold()] for name in possible
                     if name.casefold() in listed]
            if len(found) != 1:
                _reject('%s preflight needs one unambiguous runtime %s'
                        % (role, library))
            items.append(found[0])
        licences = sorted(name for name in all_names
                          if name.startswith('LICENSE-tvm-'))
        if not licences:
            _reject('%s preflight lacks TVM wheel licence' % role)
        items.extend(licences)
        if len({name.casefold() for name in items}) != len(items):
            _reject('%s preflight selected duplicate/colliding members' % role)
        if len(selected) + len(items) + 1 + 4 > _MAX_SELECTED_MEMBERS:
            _reject('OCR selected member count quota exceeded before any '
                    'compiled model SHA')
        for name in items:
            selected.append((_source_file(root, _safe_name(name)),
                             _MAX_METADATA_BYTES if name in (
                                 'manifest.json', 'manifest.txt', 'recipe.json',
                                 'report.txt', module + '.inputs') else
                             _MAX_ASSET_BYTES))
    if roots[0] == roots[1]:
        _reject('Det and Rec cannot use the same source bundle')
    dictionary_path = _declared_path(dictionary, 'dictionary')
    _not_link(dictionary_path)
    selected.append((_source_file(dictionary_path.parent,
                                  _safe_name(dictionary_path.name)),
                     _MAX_DICTIONARY_BYTES))
    if len(selected) + 4 > _MAX_SELECTED_MEMBERS:
        _reject('OCR selected member count quota exceeded before source SHA')
    total = 0
    for member, per_file_limit in selected:
        actual = member.stat().st_size
        if not 0 < actual <= per_file_limit:
            _reject('OCR selected asset has invalid/oversized bytes before SHA: '
                    '%s' % member)
        total += actual
        if total > _MAX_ASSET_AGGREGATE_BYTES:
            _reject('OCR selected asset aggregate byte quota exceeded BEFORE '
                    'SHA (%d > %d)' %
                    (total, _MAX_ASSET_AGGREGATE_BYTES))
    return total


def plan_composite(det_bundle, rec_bundle, dictionary, *,
                   preprocessing, ctc):
    _preflight_selected_byte_quota(det_bundle, rec_bundle, dictionary)
    det = _read_individual_bundle(det_bundle, 'det')
    rec = _read_individual_bundle(rec_bundle, 'rec')
    if det['root'] == rec['root']:
        _reject('Det and Rec cannot use the same verified source bundle')
    sources = {'det': det, 'rec': rec}
    if det['target'] != rec['target']:
        _reject('Det/Rec deployment OS, CPU architecture, device, GPU SM or '
                'CUDA driver API differ: %s vs %s'
                % (det['target'], rec['target']))
    if det['toolchain'] != rec['toolchain']:
        _reject('Det/Rec toolchain versions differ; sharing runtime ABI is '
                'not established')
    if det['runtime'] != rec['runtime']:
        _reject('Det/Rec TVM runtime libraries differ in names or SHA-256; '
                'one process cannot safely load both versions')
    dictionary_path = _declared_path(dictionary, 'dictionary')
    _not_link(dictionary_path)
    if not dictionary_path.is_file() or not stat.S_ISREG(
            dictionary_path.lstat().st_mode):
        _reject('dictionary must be an existing regular non-symlink file')
    dict_name = _safe_name(dictionary_path.name)
    dict_asset = _asset(dictionary_path.parent, dict_name,
                        'dictionary/' + dict_name,
                        max_bytes=_MAX_DICTIONARY_BYTES)
    tokens = _dictionary_tokens(dictionary_path)
    settings = _validate_preprocessing(preprocessing, sources)
    ctc_record = _validate_ctc(ctc, tokens)
    native_env = _validate_native_process_env(
        preprocessing['native_process_env'], ctc_record, dict_asset)

    assets = det['assets'] + rec['assets'] + [dict_asset]
    _unique_asset_names(assets)
    if len(assets) + 4 > _MAX_SELECTED_MEMBERS:
        _reject('OCR selected asset count changed/exceeds member quota '
                'during SHA validation')
    selected_total = sum(asset['bytes'] for asset in assets)
    if selected_total > _MAX_ASSET_AGGREGATE_BYTES:
        _reject('OCR selected source aggregate quota changed during hashing')
    public_assets = [
        {'role': ('dictionary' if asset['path'].startswith('dictionary/')
                  else asset['path'].split('/', 1)[0]),
         'path': asset['path'], 'bytes': asset['bytes'],
         'sha256': asset['sha256']}
        for asset in sorted(assets, key=lambda x: x['path'])]
    plan = {
        'schema': 'crosspascal.paddleocr.composite-plan',
        'schema_version': PLAN_VERSION,
        'status': 'DRAFT',
        'composite_verification': 'NOT_CHECKED',
        'facade_manifest_issued': False,
        'native_composite_oracle': 'NOT_RUN',
        'independent_ocr_oracle': {
            'status': 'NOT_CHECKED',
            'checker': 'python -m pascalvision_convert.ocr_composite_oracle '
                       '--bundle <draft_dir> --evidence <capture_dir>',
            'evidence_schema': 'crosspascal.paddleocr.independent-evidence',
            'evidence_schema_version': 1,
            'source_image_format': 'lossless PNG8 RGB non-interlaced',
            'tensor_transport': 'flat C-contiguous float32 little-endian '
                                '(float32le) with SHA-256 and exact shape',
            'matching_policy': 'all scalar input/output comparisons, unique '
                               '1:1 Det candidate boxes/scores to independent '
                               'GT boxes; CTC frames/characters and exact text',
            'cli_exit_policy': 'exit 2 even for matching supplied captures; '
                               'exit 1 for contradictory evidence; never exit '
                               '0 without independently attested native runner',
            'required': [
                'Original, hash-matched Det and Rec ONNX files plus ORT '
                'CPUExecutionProvider capture receipt',
                'Raw float32le ORT and native Det preprocessed input and '
                'raw output tensors, all scalars (SHA-256)',
                'Independent ORT/native Det bbox + confidence candidates '
                'with unique one-to-one assignment to human GT boxes',
                'Per-matched-crop original ORT and native preprocessed inputs '
                'and raw REC logits, native transcript and independent human text',
                'Measured native process environment, actual dictionary '
                'absolute path/hash and colored/padded preprocessing samples',
                'Lossless RGB PNG source with image SHA-256 and separate '
                'human-annotated text+pixel boxes bound to the same image',
            ],
            'capture_independence_attested': False,
            'verified_pascal_facade_issued': False,
        },
        'limitations': [
            'Det and Rec individually declare VERIFIED_LOCAL in their '
            'preserved local reports; their composition has not been run.',
            'Dictionary class count and preprocessing choices need independent '
            'native CTC/image execution on the intended target.',
            'Target profile is a recorded deployment requirement, not '
            'evidence of portability to an untested foreign machine.',
            'The native process environment (PPOCR_KEEP_BGR, PPOCR_NORM, '
            'CTC overrides and dictionary path) is pinned but has NOT been '
            'applied or verified by this DRAFT assembler.',
        ],
        'target': dict(det['target']),
        'preprocessing': settings,
        'native_process_environment': native_env,
        'ctc': ctc_record,
        'dictionary': {
            'file': dict_asset['path'], 'sha256': dict_asset['sha256'],
            'bytes': dict_asset['bytes'], 'encoding': 'UTF-8',
            'empty_lines_ignored': True,
            'raw_tokens': len(tokens),
        },
        'components': {
            role: {
                'verification_status': 'VERIFIED_LOCAL',
                'verification_evidence': 'individual report.txt, manifest '
                                         'and SHA-256 consistency only',
                'manifest_path': role + '/manifest.json',
                'pascal_manifest_path': role + '/manifest.txt',
                'recipe_path': role + '/recipe.json',
                'report_path': role + '/report.txt',
                'artifact_path': role + '/' +
                                 src['manifest']['artifact']['file'],
                'artifact_sha256': src['manifest']['artifact']['hash'].lower(),
                'task_kind': src['manifest']['task_kind'],
                'adapter': src['manifest']['adapter'],
                'toolchain': src['toolchain'],
                'original_provenance': src['source_provenance'],
                'source_license': src['manifest'].get('license'),
            }
            for role, src in sources.items()
        },
        'runtime': {
            'verified_same_binary_sha256': True,
            'components': {name: dict(item)
                           for name, item in det['runtime'].items()},
            'external_cuda_dependencies_verified': False,
        },
        'assets': public_assets,
    }
    return plan, assets


@contextmanager
def _held_draft_destination(target):
    if os.name != 'nt':
        target.mkdir(exist_ok=False)
        for role in ('det', 'rec', 'dictionary'):
            (target / role).mkdir(exist_ok=False)
        yield
        return
    from .omdet_public_candidate import (
        OmDetPublicCandidateError, _hold_plain_ntfs_directory,
    )
    parent = target.parent


    try:
        canonical = bundle_atomic._local_absolute(str(target))
        if os.path.normcase(canonical) != os.path.normcase(str(target)):
            _reject('OCR destination must use a canonical local Windows path')
        bundle_atomic._guard_existing_ancestors(str(parent))
        bundle_atomic._require_ntfs(str(parent))
    except bundle_atomic.UnsafeAtomicBundle as exc:
        raise CompositeContractError('OCR destination requires local NTFS '
                                     'filesystem admission: %s' % exc) from exc
    drive, _ = os.path.splitdrive(canonical)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_drive_type = kernel.GetDriveTypeW
    get_drive_type.argtypes = [ctypes.c_wchar_p]
    get_drive_type.restype = ctypes.c_uint32
    if get_drive_type(drive + '\\') != 3:
        _reject('OCR destination requires a fixed local NTFS drive, not a '
                'mapped, remote, removable or unsupported filesystem')
    if (os.path.normcase(os.path.abspath(str(parent))) !=
            os.path.normcase(os.path.realpath(str(parent)))):
        _reject('destination parent has a redirected or NTFS short-name alias')
    try:
        with ExitStack() as holders:
            for ancestor in reversed((parent, *parent.parents)):
                holders.enter_context(_hold_plain_ntfs_directory(ancestor))
            if os.path.lexists(target):
                _reject('destination became occupied before held stage creation')
            target.mkdir(exist_ok=False)
            holders.enter_context(_hold_plain_ntfs_directory(target))
            for role in ('det', 'rec', 'dictionary'):
                child = target / role
                child.mkdir(exist_ok=False)
                holders.enter_context(_hold_plain_ntfs_directory(child))
            yield
    except OmDetPublicCandidateError as exc:
        raise CompositeContractError('held NTFS OCR stage refused: %s' % exc) from exc


def assemble_composite(det_bundle, rec_bundle, dictionary, destination, *,
                       preprocessing, ctc):
    plan, assets = plan_composite(det_bundle, rec_bundle, dictionary,
                                  preprocessing=preprocessing, ctc=ctc)
    target = _declared_path(destination, 'destination')
    _not_link(target)
    if os.path.lexists(target):
        _reject('destination already exists; choose a fresh directory to avoid '
                'any filename overwrite: %s' % target)
    if not target.parent.is_dir():
        _reject('destination parent directory does not exist: %s'
                % target.parent)
    roots = [_declared_path(root, 'source')
             for root in (det_bundle, rec_bundle)]
    if os.name == 'nt':
        for label, selected in (('det source', roots[0]),
                                ('rec source', roots[1]),
                                ('destination parent', target.parent)):
            if (os.path.normcase(os.path.abspath(str(selected))) !=
                    os.path.normcase(os.path.realpath(str(selected)))):
                _reject('%s has NTFS short-name or redirected alias' % label)
    for root in roots:
        if target == root or target in root.parents or root in target.parents:
            _reject('destination must not contain or be contained by a source '
                    'bundle directory')
    dictionary_path = _declared_path(dictionary, 'dictionary')
    if target == dictionary_path.parent:
        _reject('destination cannot overwrite the dictionary directory')
    with _held_draft_destination(target):
        copied_total = 0
        for asset in assets:
            origin = asset['source']
            _not_link(origin)
            dest = target.joinpath(*asset['path'].split('/'))
            digest = hashlib.sha256()
            length = 0
            with origin.open('rb') as source, dest.open('xb') as written:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    if (length + len(chunk) > asset['bytes']
                            or copied_total + len(chunk) >
                            _MAX_ASSET_AGGREGATE_BYTES):
                        _reject('OCR asset grew beyond admitted individual or '
                                'aggregate quota while copying; no descriptor')
                    digest.update(chunk)
                    length += len(chunk)
                    copied_total += len(chunk)
                    written.write(chunk)
            if (digest.hexdigest() != asset['sha256']
                    or length != asset['bytes']):
                _reject('asset changed while staging; partial output has NO '
                        'composite descriptor: %s' % origin)
        descriptor = target / PLAN_NAME
        with descriptor.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    return plan


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description='Stage a PaddleOCR Det+Rec+dictionary DRAFT; '
                    'never produces a verified Pascal facade manifest')
    parser.add_argument('--det', required=True, help='individually verified Det')
    parser.add_argument('--rec', required=True, help='individually verified Rec')
    parser.add_argument('--dictionary', required=True, help='UTF-8 token file')
    parser.add_argument('--settings', required=True,
                        help='JSON object containing preprocessing and ctc')
    parser.add_argument('--into', required=True, help='NEW destination directory')
    args = parser.parse_args(argv)
    try:
        settings_path = _declared_path(args.settings, 'settings')
        _not_link(settings_path)
        if not settings_path.is_file():
            _reject('settings JSON must exist')
        config = _dict(_read_json(settings_path), 'settings')
        if set(config) != {'preprocessing', 'ctc'}:
            _reject('settings must contain only preprocessing and ctc')
        result = assemble_composite(
            args.det, args.rec, args.dictionary, args.into,
            preprocessing=config['preprocessing'], ctc=config['ctc'])
    except (CompositeContractError, OSError) as error:
        parser.exit(2, 'OCR composite NOT CREATED: %s\n' % error)
    print('DRAFT: %s (composite_verification=%s)'
          % (os.path.join(args.into, PLAN_NAME),
             result['composite_verification']))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
