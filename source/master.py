# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Serve the model conversion wizard on this computer."""

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

if os.name == 'nt':


    import ctypes
    from ctypes import wintypes

    class _WinFileBasicInfo(ctypes.Structure):
        _fields_ = [('CreationTime', ctypes.c_longlong),
                    ('LastAccessTime', ctypes.c_longlong),
                    ('LastWriteTime', ctypes.c_longlong),
                    ('ChangeTime', ctypes.c_longlong),
                    ('FileAttributes', wintypes.DWORD)]

    _win_kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _win_kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _win_kernel.CreateFileW.restype = wintypes.HANDLE
    _win_kernel.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    _win_kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _win_kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _win_kernel.CloseHandle.restype = wintypes.BOOL


def _native_change_time(path):
    """Read Windows ChangeTime (100 ns ticks); POSIX st_ctime is sufficient."""
    if os.name != 'nt':
        return None


    handle = _win_kernel.CreateFileW(path, 0x80, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = _WinFileBasicInfo()
        if not _win_kernel.GetFileInformationByHandleEx(
                handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.ChangeTime
    finally:
        _win_kernel.CloseHandle(handle)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pascalvision_convert import detect, environment, pipeline, samples, targets  # noqa: E402
from pascalvision_convert import app_paths, llm_build_profile, consumer_acceptance, model_sources  # noqa: E402
from pascalvision_convert import onnx_source_pack  # noqa: E402
from pascalvision_convert.contract import (  # noqa: E402
    build_contract_input, parse_float_csv, validate_vision_adapter)
from pascalvision_convert import bundle_download, bundle_publication, source_freeze, source_lock, build_plan, onnx_closure, retained_ledger, omdet_bundle_profile, omdet_public_candidate  # noqa: E402
from pascalvision_convert.recipe_file import write_family_receipt, read_recipe  # noqa: E402
import import_adapters  # noqa: E402


PAGE = os.path.join(HERE, 'pascalvision_convert', 'master_page.html')
ADDRESS = '127.0.0.1'
DEFAULT_PORT = 8765


JOBS = {}
LOCK = threading.Lock()
MODEL_DOWNLOADS = {}
ACTIVE_REQUESTS = 0
SHUTTING_DOWN = False
SESSION_TOKEN = secrets.token_urlsafe(32)


ATOMIC_PENDING_OUTPUTS = set()
OCR_DRAFT_PREVIEW = None
OCR_DRAFT_JOBS = {}


OMDET_CANDIDATE_PREVIEW = None
OMDET_CANDIDATE_SERIAL = 0


ACTIVE_PREVIEW = None
PREVIEW_SERIAL = 0
PREVIEW_LIFETIME_SECONDS = 600


def _model_inputs(path, kind):
    """Input names and shapes, so nobody has to type them out by hand."""
    if kind == detect.TFLITE:
        import tflite
        from pascalvision_convert.convert import _tflite_input_names
        with open(path, 'rb') as handle:
            model = tflite.Model.GetRootAsModel(handle.read(), 0)
        rows = _tflite_input_names(model, None)
    else:
        import onnx
        from pascalvision_convert.convert import _onnx_input_names


        rows = _onnx_input_names(
            onnx.load(path, load_external_data=False), None)
    inputs = []
    for name, dims in rows:
        inputs.append({
            'name': name,
            'shape': ','.join(str(d) for d in dims),
            'free': any(int(d) < 0 for d in dims),
        })
    return inputs


def _environment_report(python_executable=None):
    """Check the local tools and the selected language model compiler environment."""
    report = environment.collect()
    adapter = import_adapters.LLMImportAdapter()
    selected_python = llm_build_profile.console_python(python_executable or adapter.PYTHON)
    interpreter = os.path.isfile(selected_python)
    builder = os.path.isfile(adapter.BUILDER)

    step = report.step('Full native build (LLM), a toolchain of its own')
    step.say('interpreter %s' % selected_python)
    step.say('builder     %s' % adapter.BUILDER)
    version = ''
    if interpreter:


        try:
            asked = subprocess.run(
                [selected_python, '-B', '-c', 'import tvm,sys; sys.stdout.write(tvm.__version__)'],
                capture_output=True, text=True, encoding='utf-8',
                errors='replace', timeout=300)
            if asked.returncode == 0:
                version = (asked.stdout or '').strip().splitlines()[-1:] or ['']
                version = version[0]
        except Exception:
            version = ''
    step.say('TVM there   %s' % (version or 'not reported'))


    clang, _on_path = environment.find_clang()
    step.say('linker      %s' % (clang or 'not found'))
    verdict = environment.full_build_verdict(interpreter, builder, bool(version),
                                             bool(clang))
    if verdict['ready']:
        step.done(verdict['reason'])
    else:
        step.refuse(verdict['reason'])
    step.numbers['tvm'] = version

    report.summary['full_build_ready'] = verdict['ready']
    report.summary['full_build_reason'] = verdict['reason']
    report.summary['llm_python'] = selected_python
    report.summary['output_root'] = str(app_paths.data_path('bundles'))
    report.summary['native_consumer_kit_source'] = str(app_paths.consumer_kit())
    return report


def _output_directory(model_path, require_new=False, parent=None):
    """Select an unused bundle directory; never overwrite an earlier result.

    The first build retains the established `<source>-tvm` spelling. A repeat
    automatically moves to `<source>-tvm-2`, then -3, and so on. The legacy
    direct builder may reuse an empty directory. HTTP Preview uses require_new
    because Build must claim the *exact* displayed path via atomic mkdir.
    """
    base = os.path.splitext(os.path.basename(model_path))[0]
    initial = os.path.join(
        parent or os.path.dirname(os.path.abspath(model_path)), base + '-tvm')
    for index in range(1, 10001):
        proposed = initial if index == 1 else initial + '-' + str(index)
        if not os.path.lexists(proposed):
            return proposed
        if not require_new and os.path.isdir(proposed) and not os.listdir(proposed):
            return proposed
    raise RuntimeError(
        '10000 output directories are already occupied for %s; '
        'choose another model location' % model_path)


def _output_parent_identity(path):
    """Resolve the physical output parent, including any symlink/junction."""
    return os.path.normcase(os.path.realpath(os.path.dirname(path)))


def _reserved_output_identity(path):
    """Require our reserved folder, not a substituted link or junction."""
    junction = getattr(os.path, 'isjunction', None)
    if os.path.islink(path) or (junction and junction(path)):
        raise ValueError('Reserved output was replaced by a link or junction')
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('Reserved output is no longer a directory')
    return (os.path.normcase(os.path.realpath(path)), info.st_dev,
            info.st_ino, info.st_ctime_ns)


def _source_fingerprint(path):
    """Hash the complete source tree and its metadata to detect changes between scans."""
    if not path or path.lower().startswith(('http://', 'https://')):
        raise ValueError('Recognise/download the model to a local file before Preview')
    root = os.path.abspath(path)
    digest = hashlib.sha256()
    entries = 0

    def add_item(filename, relative):
        nonlocal entries
        entries += 1
        if entries > 50000:
            raise ValueError('source directory has over 50000 entries; cannot pin Preview')
        link = os.lstat(filename)
        source = os.stat(filename)
        native_change = _native_change_time(filename)
        details = (relative, os.path.realpath(filename),
                   source.st_mode, source.st_size, source.st_mtime_ns,
                   source.st_ctime_ns, source.st_dev, source.st_ino,
                   link.st_mode, link.st_size, link.st_mtime_ns,
                   link.st_ctime_ns, native_change)
        digest.update(repr(details).encode('utf-8', 'surrogatepass'))
        if stat.S_ISREG(source.st_mode):
            with open(filename, 'rb') as handle:
                remaining = source.st_size
                while remaining:
                    block = handle.read(min(1048576, remaining))
                    if not block:
                        raise OSError('source changed while hashing: ' + filename)
                    digest.update(block)
                    remaining -= len(block)
        observed = os.stat(filename)
        changed = _native_change_time(filename)


        if (observed.st_mode, observed.st_size, observed.st_mtime_ns,
                observed.st_ctime_ns, observed.st_dev, observed.st_ino) != (
                source.st_mode, source.st_size, source.st_mtime_ns,
                source.st_ctime_ns, source.st_dev, source.st_ino) or (
                    changed != native_change):
            raise OSError('source changed while inspecting: ' + filename)

    def walk_error(error):
        raise error

    add_item(root, '.')
    if os.path.isdir(root):
        for folder, directories, files in os.walk(
                root, followlinks=False, onerror=walk_error):
            for name in sorted(directories + files):
                current = os.path.join(folder, name)
                is_junction = getattr(os.path, 'isjunction', None)
                if (os.path.isdir(current) and
                        (os.path.islink(current) or
                         (is_junction and is_junction(current)))):
                    raise ValueError('directory model contains an untracked linked '
                                     'directory: ' + current)
                add_item(current, os.path.relpath(current, root))
    return {'signature': digest.hexdigest(),
            'extent': ('full small-file SHA-256' if os.path.isfile(root)
                       and os.path.getsize(root) <= 8 * 1048576
                       else 'full source-file SHA-256 (not immutable)' if
                       os.path.isfile(root) else
                       'full recursive file SHA-256 (not immutable)')}


_FROZEN_RAW_GRAPH_MAX_BYTES = 64 * 1048576


def _frozen_raw_closure(path):
    """Prove a bounded RAW graph has no known external weight dependencies.

    This deliberately excludes family exporters, custom-op file loaders and
    models carrying other resources. It is a *static candidate* proof, not an
    OS lock, nor an exhaustive assertion about runtime access by every op.
    """
    candidate = os.path.abspath(path)
    pinned = source_freeze.fingerprint_content(
        candidate, max_entries=1, max_bytes=_FROZEN_RAW_GRAPH_MAX_BYTES)
    if pinned['source_kind'] != 'file':
        raise ValueError('Source freeze opt-in supports only one RAW model file')
    if pinned['total_bytes'] > _FROZEN_RAW_GRAPH_MAX_BYTES:
        raise ValueError('Source freeze graph exceeds bounded 64 MiB scope')
    extension = os.path.splitext(candidate)[1].lower()
    if extension not in ('.onnx', '.tflite'):
        raise ValueError('Source freeze opt-in supports only .onnx or .tflite')
    recognised = detect.recognise(candidate)
    expected_kind = detect.ONNX if extension == '.onnx' else detect.TFLITE
    if (recognised.kind != expected_kind or
            not recognised.convertible or recognised.family):
        raise ValueError('Source freeze requires a genuinely recognised RAW %s '
                         'model with no family exporter' % extension)
    if expected_kind == detect.ONNX:
        try:
            import onnx
            model = onnx.load(candidate, load_external_data=False)
        except Exception as error:
            raise ValueError('Cannot inspect complete ONNX dependency closure: '
                             '%s' % error) from error
        if model.training_info or model.functions:
            raise ValueError('ONNX training/function subgraphs are outside '
                             'the supported self-contained RAW scope')

        def check_tensor(tensor):
            if tensor.external_data or (
                    tensor.data_location == onnx.TensorProto.EXTERNAL):
                raise ValueError('ONNX has external tensor weights; freeze '
                                 'would omit a model dependency')

        def check_sparse(tensor):
            check_tensor(tensor.values)
            check_tensor(tensor.indices)

        def check_graph(graph):
            for weight in graph.initializer:
                check_tensor(weight)
            for weight in graph.sparse_initializer:
                check_sparse(weight)
            for node in graph.node:
                if node.domain not in ('', 'ai.onnx', 'ai.onnx.ml'):
                    raise ValueError('ONNX custom operator domain could load '
                                     'external dependencies: ' + node.domain)
                for attribute in node.attribute:
                    if attribute.HasField('t'):
                        check_tensor(attribute.t)
                    for weight in attribute.tensors:
                        check_tensor(weight)
                    if attribute.HasField('sparse_tensor'):
                        check_sparse(attribute.sparse_tensor)
                    for weight in attribute.sparse_tensors:
                        check_sparse(weight)
                    if attribute.HasField('g'):
                        check_graph(attribute.g)
                    for subgraph in attribute.graphs:
                        check_graph(subgraph)

        check_graph(model.graph)
    else:


        try:
            import tflite
        except ImportError as error:
            raise ValueError('TFLite source closure NOT CHECKED: FlatBuffer '
                             'schema package is unavailable') from error
        with open(candidate, 'rb') as model_file:
            data = model_file.read()
        try:
            graph = tflite.Model.GetRootAsModel(data, 0)
            if graph.SubgraphsLength() < 1 or graph.BuffersLength() < 1:
                raise ValueError('TFLite has no executable subgraph/buffers')
            if graph.MetadataLength():
                raise ValueError('TFLite contains associated metadata with '
                                 'unclosed possible external dependencies')
            builtin_ops = getattr(tflite, 'BuiltinOperator', None)
            custom_id = getattr(builtin_ops, 'CUSTOM', None)
            if custom_id is None:
                builtin_ops = getattr(builtin_ops, 'BuiltinOperator', None)
                custom_id = getattr(builtin_ops, 'CUSTOM', None)
            if custom_id is None:
                raise ValueError('TFLite BuiltinOperator.CUSTOM cannot be '
                                 'identified; source closure unsupported')
            for index in range(graph.OperatorCodesLength()):
                code = graph.OperatorCodes(index)
                if code.CustomCode() or code.BuiltinCode() == custom_id:
                    raise ValueError('TFLite custom operator dependencies '
                                     'cannot be proven self-contained')
            for index in range(graph.BuffersLength()):
                buffer = graph.Buffers(index)
                offset, count = buffer.Offset(), buffer.Size()
                if offset or count:
                    if offset < 0 or count < 0 or offset + count > len(data):
                        raise ValueError('TFLite external buffer lies outside '
                                         'the source file')
        except (AttributeError, IndexError, TypeError, OverflowError) as error:
            raise ValueError('TFLite dependency closure is ambiguous or '
                             'unsupported by the available schema') from error
    after = source_freeze.fingerprint_content(
        candidate, max_entries=1, max_bytes=_FROZEN_RAW_GRAPH_MAX_BYTES)
    if after != pinned:
        raise ValueError('RAW model source changed while inspecting dependency '
                         'closure; issue a fresh Preview')
    return {'source_kind': 'file',
            'tree_sha256': pinned['tree_sha256'],
            'total_bytes': pinned['total_bytes'],
            'format': expected_kind,
            'status': 'STATIC_SELF_CONTAINED_CANDIDATE_NOT_LOCKED'}


def _raw_source_freeze_candidate(path, *, forbidden_roots=()):
    """Validate the selected raw model and its external files for compiler-held staging."""
    candidate = os.path.abspath(path)
    if (candidate.lower().endswith('.onnx') and os.path.isfile(candidate)
            and os.path.getsize(candidate) <= onnx_closure.MAX_MODEL_BYTES):
        selected = onnx_closure.scan_selected_closure(
            candidate, forbidden_roots=forbidden_roots)
        if selected['dependencies']:
            if not callable(getattr(source_lock, 'hold_staged_file_set', None)):
                raise ValueError('External ONNX closure requires a native Win32 '
                                 'compiler-held multi-file lock, not copy-only')
            recognised = detect.recognise(candidate)
            if (not recognised.convertible or recognised.kind != detect.ONNX or
                    getattr(recognised, 'family', '')):
                raise ValueError('External ONNX freeze requires genuinely '
                                 'recognised unclassified RAW ONNX')
            return {
                'source_kind': selected['source_kind'],
                'format': detect.ONNX,
                'status': selected['status'],
                'closure_schema': selected['schema'],
                'source_closure_version': selected['schema_version'],
                'source_model': selected['source_model'],
                'source_root': selected['source_root'],
                'model_rel': selected['model_rel'],
                'file_count': selected['file_count'],
                'total_bytes': selected['total_bytes'],
                'tree_sha256': selected['tree_sha256'],
                'dependencies': selected['dependencies'],
                'entries': selected['entries'],
                'references': selected['references'],
                'release_grade_immutable': False,
            }
    return _frozen_raw_closure(path)


def _raw_freeze_warning(frozen):
    if frozen['source_kind'] == 'selected_onnx_files':
        return ('Opt-in ONNX external_data source freeze selects ONLY the ONNX '
                'and its explicitly referenced sidecars. Build must hold native '
                'read/share-only handles on ALL staged files through compiler '
                'and Verify; this is NOT release-grade immutable source proof.')
    return ('Opt-in RAW source snapshot is planned only after Build '
            'reserves its destination. It copies approved local bytes '
            'for the compiler but is NOT an immutable OS lock: staged '
            'files and unclosed external resources remain TOCTOU risks.')


def _frozen_raw_preview(data, *, reserved_output=None):
    """Return the only supported opt-in configuration, or an explicit refusal."""
    requested = data.get('freeze_local_raw_source', False)
    if requested is False:
        return None
    if requested is not True:
        raise ValueError('freeze_local_raw_source must be a boolean true/false')
    if (data.get('full_build') or data.get('task') or data.get('adapter')):
        raise ValueError('Source freeze opt-in supports unclassified RAW builds '
                         'only; no Pascal adapter, PaddleOCR, or family builder')
    path = str(data.get('path') or '').strip()
    if not path or path.lower().startswith(('http://', 'https://')):
        raise ValueError('Source freeze opt-in requires a local RAW file, '
                         'not a URL or downloaded family asset')


    if os.name != 'nt':
        raise ValueError('RAW compiler-held source lock requires native Win32')
    source_lock._Kernel32().assert_local_ntfs(path)
    forbidden = (reserved_output,) if reserved_output is not None else ()
    return _raw_source_freeze_candidate(path, forbidden_roots=forbidden)


def _raw_source_protection_offer(path, recognised):
    """Return an honest server-measured UI default, not a promise of a lock.

    The actual held Win32 handle is acquired later in _start_job after the
    one-time Preview token and a new full source/dependency verification. Never
    infer suitability from just a .onnx extension or the browser OS string.
    """
    result = {
        'eligible': False,
        'recommended': False,
        'status': 'NOT_APPLICABLE',
        'release_grade_immutable': False,
        'reason': 'Protected source staging applies only to bounded, fully '
                  'closed unclassified RAW ONNX/TFLite on native Windows NTFS',
    }
    if os.name != 'nt':
        result['reason'] = ('Win32 staged source locking needs native Windows; '
                            'no unlocked POSIX/WSL fallback')
        return result
    if (not recognised.convertible or getattr(recognised, 'family', '') or
            recognised.kind not in (detect.ONNX, detect.TFLITE) or
            not os.path.isfile(path)):
        return result
    try:
        source_lock._Kernel32().assert_local_ntfs(path)
        reserved = _output_directory(path, require_new=True)
        candidate = _raw_source_freeze_candidate(
            path, forbidden_roots=(reserved,))
        if candidate['status'] not in (
                'STATIC_SELF_CONTAINED_CANDIDATE_NOT_LOCKED',
                'SELECTED_FILES_SCANNED_NOT_LOCKED'):
            raise ValueError('native RAW protection candidate state is unknown')
    except (OSError, ValueError, source_freeze.SourceFreezeError,
            onnx_closure.OnnxClosureError) as trouble:
        result['status'] = 'BLOCKED_UNSUPPORTED_SOURCE'
        result['reason'] = str(trouble)
        return result
    result.update({
        'eligible': True,
        'recommended': True,
        'status': candidate['status'],
        'reason': 'Win32 local NTFS + complete bounded RAW model dependency bytes qualify '
                  'for a locked private stage at Build; this is still NOT '
                  'release-grade and needs a new Preview and runtime lock',
    })
    return result


def _request_signature(data):
    """Exact JSON request/plan comparison, without key-order dependence."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _omdet_candidate_directory_id(path):
    """FileID/creation identity only; unrelated siblings may be pre-staged."""
    info = omdet_public_candidate.bundle_atomic._guard_stat(
        str(path), directory=True)
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode),
            info.st_ctime_ns)


def _omdet_candidate_file_id(path, info):
    """A selected file's pathname, NTFS identity, native ChangeTime and length."""
    return (info.st_dev, info.st_ino, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, _native_change_time(str(path)))


def _omdet_candidate_selection(data):
    """Check the candidate files and the independent pins required for packaging."""
    if type(data) is not dict or set(data) != {
            'candidate_dir', 'expected_pins_file', 'into'}:
        raise ValueError('OmDet candidate requires exact candidate_dir, '
                         'expected_pins_file and into JSON paths')
    for name in ('candidate_dir', 'expected_pins_file', 'into'):
        value = data[name]
        if (type(value) is not str or not 0 < len(value) <= 4096 or
                value != value.strip() or not os.path.isabs(value)):
            raise ValueError('OmDet %s must be one canonical absolute local path'
                             % name)


    root, final = omdet_public_candidate._paths(
        data['candidate_dir'], data['into'])
    if os.path.lexists(final):
        raise FileExistsError('OmDet destination is already occupied')
    atomic = omdet_public_candidate.bundle_atomic
    pin_file = Path(atomic._local_absolute(data['expected_pins_file']))
    atomic._guard_existing_ancestors(str(pin_file.parent))
    if (os.path.normcase(os.path.abspath(str(pin_file.parent))) !=
            os.path.normcase(os.path.realpath(str(pin_file.parent)))):
        raise ValueError('OmDet independent pinset parent uses a redirected alias')


    pin_name = os.path.normcase(str(pin_file))
    for protected in (root, final):
        checked = os.path.normcase(str(protected))
        try:
            shared = os.path.commonpath((pin_name, checked))
        except ValueError:
            shared = ''
        if shared == checked:
            raise ValueError('OmDet independently provided pinset cannot be '
                             'inside candidate or output directory')

    pin_bytes = omdet_public_candidate._small_bytes(
        pin_file.parent, pin_file.name, omdet_public_candidate._JSON_LIMIT)
    pins = omdet_public_candidate._strict_object(
        pin_bytes, 'independently provided OmDet expected_pins')
    profile_bytes = omdet_public_candidate._small_bytes(
        root, omdet_public_candidate.PROFILE_NAME,
        omdet_public_candidate._JSON_LIMIT)
    profile = omdet_public_candidate._strict_object(
        profile_bytes, omdet_public_candidate.PROFILE_NAME)


    omdet_public_candidate._preflight_candidate_byte_quota(
        root, profile, profile_bytes=len(profile_bytes))
    validated = omdet_bundle_profile.validate_candidate_profile(
        profile, root, expected_pins=pins)
    target = omdet_public_candidate._validate_fixed_profile(root, validated)
    omdet_public_candidate._reject_external_onnx(
        root, omdet_public_candidate._pin(validated['source_onnx'], 'source_onnx'))

    records = [
        ('source_onnx', validated['source_onnx']),
        ('export_contract', validated['export_contract']),
        ('feature_manifest', validated['feature_manifest']),
        ('processor', validated['hf_configs']['processor']),
        ('model', validated['hf_configs']['model']),
        ('compiled_module', validated['compiled_module']),
        ('compiled_inputs', validated['compiled_inputs']),
    ]
    records.extend(
        (name, validated['trained_features'][name])
        for name in omdet_public_candidate.FEATURE_FILES)
    records.extend(
        ('runtime_' + name, record)
        for name, record in sorted(validated['runtime']['libraries'].items()))
    selected, identities, aliases = [], [], set()
    total_bytes = 0
    for role, source_record in records:


        pin = omdet_public_candidate._pin(source_record, role)
        relative = pin['file']
        folded = relative.casefold()
        if folded in aliases:
            raise ValueError('OmDet selected source roles share a file/alias')
        aliases.add(folded)
        source_path, info = omdet_public_candidate._member(root, relative)
        if info.st_size != pin['bytes']:
            raise ValueError('OmDet selected source size changed: ' + relative)
        selected.append({'role': role, 'file': relative,
                         'bytes': pin['bytes'], 'sha256': pin['sha256']})
        identities.append((relative, _omdet_candidate_file_id(source_path, info)))
        total_bytes += pin['bytes']
        if total_bytes > atomic.MAX_NATIVE_FAMILY_BYTES:
            raise ValueError('OmDet selected source exceeds native-family budget')


    selected.sort(key=lambda item: (item['file'], item['role']))
    tree_descriptor = {
        'schema': omdet_bundle_profile.SCHEMA,
        'candidate_profile_sha256': hashlib.sha256(profile_bytes).hexdigest(),
        'selected': selected,
    }
    tree_sha = hashlib.sha256(
        _request_signature(tree_descriptor).encode('utf-8')).hexdigest()
    profile_path, profile_info = omdet_public_candidate._member(
        root, omdet_public_candidate.PROFILE_NAME)
    pin_path, pin_info = omdet_public_candidate._member(pin_file.parent, pin_file.name)
    evidence = {
        'source_tree_sha256': tree_sha,
        'candidate_profile_sha256': tree_descriptor['candidate_profile_sha256'],
        'expected_pins_sha256': hashlib.sha256(pin_bytes).hexdigest(),
        'source_identity': _omdet_candidate_directory_id(root),
        'output_parent_identity': _omdet_candidate_directory_id(final.parent),
        'profile_identity': _omdet_candidate_file_id(profile_path, profile_info),
        'pinset_identity': _omdet_candidate_file_id(pin_path, pin_info),
        'selected_identities': tuple(sorted(identities)),
    }
    return {'root': str(root), 'final': str(final), 'pins': pins,
            'target': target['device'], 'evidence': evidence,
            'selected_file_count': len(selected), 'selected_total_bytes': total_bytes}


def _omdet_candidate_preview(data):
    """Issue one short-lived, independent OmDet authorization from real SHA."""
    global OMDET_CANDIDATE_PREVIEW, OMDET_CANDIDATE_SERIAL
    with LOCK:
        OMDET_CANDIDATE_SERIAL += 1
        serial = OMDET_CANDIDATE_SERIAL
        OMDET_CANDIDATE_PREVIEW = None
    try:
        selected = _omdet_candidate_selection(data)
        signature = _request_signature(data)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return {'trouble': 'OmDet candidate Preview refused: %s' % exc}
    with LOCK:
        if serial != OMDET_CANDIDATE_SERIAL:
            return {'trouble': 'A newer OmDet Preview superseded this request'}
        token = secrets.token_urlsafe(32)
        OMDET_CANDIDATE_PREVIEW = {
            'token': token, 'request': signature,
            'evidence': selected['evidence'], 'final': selected['final'],
            'serial': serial,
            'expires': time.monotonic() + PREVIEW_LIFETIME_SECONDS,
        }
    return {
        'ready_for_build': True, 'preview_token': token,
        'output_directory': selected['final'],
        'target': selected['target'],
        'selected_file_count': selected['selected_file_count'],
        'selected_total_bytes': selected['selected_total_bytes'],
        'source_tree_sha256': selected['evidence']['source_tree_sha256'],
        'candidate_profile_sha256': selected['evidence']['candidate_profile_sha256'],
        'expected_pins_sha256': selected['evidence']['expected_pins_sha256'],
        'status': 'NOT_CHECKED', 'pascal_facade_uat': 'NOT_RUN',
        'runtime_provenance': 'NOT_ATTESTED', 'source_v1_facade': 'DISABLED',
        'download_ready': False,
        'warning': ('Selected-source SHA checks and private NTFS publication '
                    'are not native OmDet numerical UAT, authenticated runtime '
                    'provenance or verified Pascal Facade acceptance. '
                    'No HTTP ZIP READY is issued for this v2 candidate.'),
    }


def _omdet_candidate_build(data):
    """Consume Preview once, recheck all source bytes, invoke ONLY v2 publish."""
    global OMDET_CANDIDATE_PREVIEW
    if type(data) is not dict:
        return {'trouble': 'OmDet candidate Build requires a JSON object'}
    options = dict(data)
    token = options.pop('preview_token', None)
    try:
        signature = _request_signature(options)
    except (ValueError, TypeError) as exc:
        return {'trouble': 'Invalid OmDet candidate Build request: %s' % exc}
    with LOCK:
        approved = OMDET_CANDIDATE_PREVIEW
        if (not approved or type(token) is not str or
                not secrets.compare_digest(token, approved['token']) or
                time.monotonic() >= approved['expires']):
            return {'trouble': 'OmDet Preview token is missing, expired or consumed'}


        OMDET_CANDIDATE_PREVIEW = None
        if signature != approved['request']:
            return {'trouble': 'OmDet Build paths differ from the approved Preview'}

    try:
        fresh = _omdet_candidate_selection(options)
        if (fresh['evidence'] != approved['evidence'] or
                fresh['final'] != approved['final']):
            raise ValueError('selected OmDet source/pinset/parent identity or '
                             'full SHA changed since Preview')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return {'trouble': 'OmDet candidate Build revalidation refused: %s' % exc}
    reservation = os.path.normcase(os.path.abspath(fresh['final']))
    with LOCK:
        if (OMDET_CANDIDATE_SERIAL != approved['serial'] or
                time.monotonic() >= approved['expires']):
            return {'trouble': 'OmDet Preview superseded or expired before Build'}
        if os.path.lexists(fresh['final']) or reservation in ATOMIC_PENDING_OUTPUTS:
            return {'trouble': 'OmDet final destination already occupied or reserved'}
        ATOMIC_PENDING_OUTPUTS.add(reservation)
    try:


        publication = omdet_public_candidate.publish(
            options['candidate_dir'], options['into'],
            expected_pins=fresh['pins'])
        if (type(publication) is not dict or
                publication.get('schema') != omdet_public_candidate.SCHEMA or
                publication.get('status') != 'NOT_CHECKED' or
                publication.get('pascal_facade_uat') != 'NOT_RUN' or
                publication.get('runtime_provenance') != 'NOT_ATTESTED' or
                publication.get('source_v1_facade') != 'DISABLED' or
                publication.get('publication_status') !=
                omdet_public_candidate.bundle_atomic.STATUS or
                os.path.normcase(os.path.abspath(str(
                    publication.get('destination', '')))) != reservation):
            raise ValueError('candidate publisher did not return the expected '
                             'unverified, no-overwrite namespace receipt')
        return {
            'status': 'NOT_CHECKED', 'pascal_facade_uat': 'NOT_RUN',
            'runtime_provenance': 'NOT_ATTESTED', 'source_v1_facade': 'DISABLED',
            'download_ready': False,
            'output_directory': fresh['final'],
            'publication': publication,
            'warning': ('Candidate directory published; native model/Facade '
                        'UAT NOT_RUN. Runtime NOT_ATTESTED. '
                        'No HTTP ZIP READY for unverified v2 closure.'),
        }
    except Exception as exc:


        return {'trouble': 'OmDet candidate Build refused; retained data '
                           'untouched: %s' % exc,
                'status': 'NOT_CHECKED', 'download_ready': False}
    finally:
        with LOCK:
            ATOMIC_PENDING_OUTPUTS.discard(reservation)


def _ocr_draft_selection(data):
    """Strictly selected, independent Det+Rec DRAFT inputs and typed settings."""
    from pascalvision_convert import ocr_composite
    if not isinstance(data, dict) or set(data) != {
            'det', 'rec', 'dictionary', 'settings'}:
        raise ValueError('OCR DRAFT requires det, rec, dictionary and settings')
    for key in ('det', 'rec', 'dictionary'):
        if (type(data[key]) is not str or not os.path.isabs(data[key])
                or data[key] != data[key].strip()):
            raise ValueError('OCR DRAFT %s must be an absolute path' % key)
    settings = data['settings']
    if (type(settings) is not dict or set(settings) !=
            {'preprocessing', 'ctc'} or
            any(type(settings[key]) is not dict for key in settings)):
        raise ValueError('OCR DRAFT settings require typed preprocessing and ctc objects')
    plan, assets = ocr_composite.plan_composite(
        data['det'], data['rec'], data['dictionary'],
        preprocessing=settings['preprocessing'], ctc=settings['ctc'])
    return plan, tuple((a['path'], a['sha256'], a['bytes']) for a in assets)


def _ocr_draft_preview(data):
    global OCR_DRAFT_PREVIEW
    with LOCK:
        OCR_DRAFT_PREVIEW = None
    try:
        plan, assets = _ocr_draft_selection(data)
        from pascalvision_convert import bundle_atomic
        parent = os.path.dirname(os.path.abspath(data['det']))
        output = os.path.join(parent, 'ocr-composite-draft-' + secrets.token_hex(12))
        bundle_atomic._local_absolute(output)
        bundle_atomic._require_ntfs(parent)
        if os.path.lexists(output):
            raise ValueError('OCR DRAFT destination occupied')
        signature = _request_signature(data)
        token = secrets.token_urlsafe(32)
        with LOCK:
            OCR_DRAFT_PREVIEW = {
                'token': token, 'signature': signature,
                'plan': _request_signature(plan), 'assets': assets,
                'output': output, 'parent': _output_parent_identity(output),
                'expires': time.monotonic() + PREVIEW_LIFETIME_SECONDS}
        return {'preview_token': token, 'output_directory': output,
                'asset_count': len(assets), 'plan': plan,
                'status': 'DRAFT', 'verification_status': 'NOT_CHECKED'}
    except (OSError, ValueError, TypeError) as exc:
        return {'trouble': str(exc)}


def _ocr_draft_build(data):
    global OCR_DRAFT_PREVIEW
    if not isinstance(data, dict):
        return {'trouble': 'OCR DRAFT request must be an object'}
    request = dict(data)
    token = request.pop('preview_token', None)
    with LOCK:
        approved = OCR_DRAFT_PREVIEW
        OCR_DRAFT_PREVIEW = None
    if not (approved and type(token) is str and
            secrets.compare_digest(token, approved['token']) and
            time.monotonic() < approved['expires']):
        return {'trouble': 'OCR DRAFT Preview token is missing, expired or consumed'}
    try:
        if _request_signature(request) != approved['signature']:
            raise ValueError('OCR DRAFT inputs differ from Preview')
        plan, assets = _ocr_draft_selection(request)
        if (_request_signature(plan) != approved['plan'] or
                assets != approved['assets']):
            raise ValueError('OCR DRAFT components or dictionary changed after Preview')
        destination = approved['output']
        if (_output_parent_identity(destination) != approved['parent']
                or os.path.lexists(destination)):
            raise ValueError('OCR DRAFT final name/parent changed')
        from pascalvision_convert import ocr_composite, bundle_atomic, bundle_publication
        stage = os.path.join(os.path.dirname(destination),
                             '.crosspascal-stage-' + secrets.token_hex(16))
        bundle_atomic._local_absolute(stage)
        if not bundle_atomic.STAGE_NAME.fullmatch(os.path.basename(stage)):
            raise ValueError('OCR DRAFT private stage name invalid')
        bundle_publication._admit_private_atomic_stage(
            os.path.dirname(destination), capacity_profile='native_family')
        with LOCK:
            if destination in ATOMIC_PENDING_OUTPUTS:
                raise ValueError('OCR DRAFT destination already reserved')
            ATOMIC_PENDING_OUTPUTS.add(destination)
        try:
            ocr_composite.assemble_composite(
                request['det'], request['rec'], request['dictionary'], stage,
                preprocessing=request['settings']['preprocessing'],
                ctc=request['settings']['ctc'])
            checked_plan, checked_assets = _ocr_draft_selection(request)
            if (_request_signature(checked_plan) != approved['plan'] or
                    checked_assets != approved['assets']):
                raise ValueError('OCR DRAFT source changed during assembly')
            inventory = bundle_atomic.scan_bundle_tree(
                stage, max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
            approved_files = {name: (sha, size) for name, sha, size
                              in approved['assets']}
            staged_files = {m.path: (m.sha256, m.size) for m in inventory.members
                            if m.kind == 'file'}
            expected_files = set(approved_files) | {'composite-plan.json'}
            if (set(staged_files) != expected_files or
                    any(staged_files.get(name) != evidence
                        for name, evidence in approved_files.items()) or
                    {m.path for m in inventory.members if m.kind == 'directory'}
                    != {'det', 'rec', 'dictionary'}):
                raise ValueError('OCR DRAFT contains unapproved or changed staged members')
            with open(os.path.join(stage, 'composite-plan.json'),
                      encoding='utf-8') as descriptor:
                recorded_plan = json.load(descriptor)
            if _request_signature(recorded_plan) != approved['plan']:
                raise ValueError('OCR DRAFT descriptor differs from approved Preview')
            if not any(m.path == 'composite-plan.json' and m.kind == 'file'
                       for m in inventory.members):
                raise ValueError('complete OCR DRAFT descriptor missing')

            if any(m.path in ('manifest.json', 'manifest.txt')
                   for m in inventory.members):
                raise ValueError('OCR DRAFT must not issue a root verified manifest')
            receipt = bundle_atomic.commit_staged_bundle(
                stage, destination, inventory,
                required_files=('composite-plan.json',),
                max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
            with LOCK:
                job = secrets.token_hex(12)
                OCR_DRAFT_JOBS[job] = {
                    'directory': receipt.destination, 'inventory': inventory,
                    'verification_status': 'NOT_CHECKED'}
            return {'draft_job': job, 'status': 'DRAFT',
                    'verification_status': 'NOT_CHECKED',
                    'output_directory': receipt.destination}
        finally:
            with LOCK:
                ATOMIC_PENDING_OUTPUTS.discard(destination)
    except (OSError, ValueError, RuntimeError) as exc:
        return {'trouble': 'OCR DRAFT not published: %s' % exc}


def _ocr_draft_evidence(data):
    """Advisory captured-evidence review; never certifies native execution."""
    if (type(data) is not dict or set(data) !=
            {'draft_job', 'evidence_directory'}):
        return {'trouble': 'OCR evidence requires draft_job and evidence_directory'}
    job_id, evidence = data['draft_job'], data['evidence_directory']
    if (type(job_id) is not str or not re.fullmatch(r'[0-9a-f]{24}', job_id)
            or type(evidence) is not str or not os.path.isabs(evidence)
            or evidence != evidence.strip()):
        return {'trouble': 'OCR evidence job or absolute directory invalid'}
    with LOCK:
        chosen = OCR_DRAFT_JOBS.get(job_id)
    if chosen is None:
        return {'trouble': 'Unknown committed OCR DRAFT job'}
    from pascalvision_convert import bundle_atomic, ocr_composite_oracle
    try:
        root = chosen['directory']
        inventory = chosen['inventory']
        if (not isinstance(inventory, bundle_atomic.Inventory) or
                bundle_atomic.scan_bundle_tree(
                    root, max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES) != inventory):
            raise ValueError('committed OCR DRAFT Inventory changed before evidence review')
        advisory = ocr_composite_oracle.audit_composite_evidence(root, evidence)
        if (type(advisory) is not dict or advisory.get('status') not in
                ('FAILED', 'NOT_CHECKED', 'NUMERIC_MATCH_SUPPLIED_EVIDENCE') or
                advisory.get('composite_verification') != 'NOT_CHECKED' or
                advisory.get('facade_manifest_issued') is not False):
            raise ValueError('OCR oracle returned unsupported or elevated claims')
        if bundle_atomic.scan_bundle_tree(
                root, max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES) != inventory:
            raise ValueError('committed OCR DRAFT Inventory changed during evidence review')
        return {'status': 'DRAFT', 'verification_status': 'NOT_CHECKED',
                'composite_verification': 'NOT_CHECKED',
                'facade_manifest_issued': False, 'advisory': advisory}
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return {'trouble': 'OCR evidence review refused: %s' % exc}


def _selected_paddle_preview(data):
    """Compute a server-approved exact Paddle Det/Rec tree for Preview.

    An external dictionary belongs to this digest. Scanning is not locking.
    """
    path = str(data.get('path') or '').strip()
    role = data.get('paddle_role', '')
    dictionary = data.get('paddle_dictionary', '')
    held = data.get('paddle_native_hold', False)
    if (type(role) is not str or type(dictionary) is not str or
            type(held) is not bool):
        raise ValueError('Paddle role/dictionary must be strings; native hold boolean')
    requested = bool(role or dictionary or held)
    if not os.path.isdir(path):
        if requested:
            raise ValueError('Paddle selected source requires a local Paddle directory')
        return None
    recognised = detect.recognise(path)
    adapter = (import_adapters.find_adapter(path, recognised)
               if not recognised.convertible else None)
    paddle = (adapter is not None and adapter.name == 'paddleocr'
              and adapter.kind == 'export')
    if not paddle:
        if requested:
            raise ValueError('Paddle source options cannot select another family')
        return None
    if (data.get('full_build') or data.get('freeze_local_raw_source') or
            data.get('atomic_complete_bundle')):
        raise ValueError('Paddle selected source is not a RAW/LLM atomic build')
    if role not in ('det', 'rec'):
        raise ValueError('Paddle wizard Preview requires explicit role det/rec')
    if dictionary != dictionary.strip():
        raise ValueError('Paddle dictionary path has surrounding whitespace')
    if role == 'rec' and not dictionary:
        raise ValueError('Paddle rec requires explicit dictionary path')
    if role == 'det' and dictionary:
        raise ValueError('Paddle det cannot accept a dictionary')
    from pascalvision_convert import paddle_closure
    return paddle_closure.scan_selected_paddle_source(
        path, role=role, dictionary=dictionary if role == 'rec' else None)


def _paddle_preview_warning(paddle, held_requested=False):
    return ('Paddle %s graph, weights%s: selected SHA-256 bytes bound to this '
            'Preview. Pre/post WSL exporter scans are NOT a compiler-held lock '
            'or source revision attestation. Numerical verification NOT_CHECKED.'
            % (paddle['role'], '/dictionary' if paddle['dictionary'] else '')
               ) + (
            ' Experimental Win32-held private source requested for Build; '
            'not yet allocated or proved compatible with actual WSL/DrvFs. '
            'Any unsupported/error condition refuses without original fallback. '
            'Private stages are retained and consume disk space.'
            if held_requested else '')


def _consumer_preview_display(selected):
    return {'source': selected['source'],
            'tree_sha256': selected['inventory'].tree_sha256,
            'file_count': len(selected['inventory'].members),
            'verification_status': 'NOT_CHECKED'}


def _selected_hf_preview(data):
    """Record the selected language model weights and tokenizer for build validation."""
    if not data.get('full_build'):
        return None
    path = str(data.get('path') or '').strip()
    key = import_adapters.LLMImportAdapter()._key_for(path)
    if key not in ('qwen2.5-0.5b', 'gemma-2-2b'):
        raise ValueError('Full native HF Preview requires a supported exact '
                         'Qwen2.5/Gemma2 model profile')
    from pascalvision_convert import hf_closure, hf_held_stage
    selected = hf_closure.scan_selected_hf_source(path,
                                                  expected_model_key=key)
    if data.get('llm_profile'):
        model_sources.match_source(selected, key)
    if selected['file_count'] > hf_held_stage.MAX_HF_SELECTED_FILES:
        raise ValueError('selected HF source exceeds the current 512-file '
                         'Win32 held private-stage capability')
    return selected


def _hf_preview_display(selected):
    try:
        official_source = model_sources.match_source(selected, selected['model_key'])
    except ValueError:
        official_source = None
    return {
        'source_root': selected['source_root'],
        'model_key': selected['model_key'],
        'tree_sha256': selected['tree_sha256'],
        'file_count': selected['file_count'],
        'total_bytes': selected['total_bytes'],
        'status': 'SCANNED_NOT_LOCKED',
        'verification_status': 'NOT_CHECKED',
        'release_grade_immutable': False,
        'official_source': official_source,
    }


def _hf_preview_warning():
    return (
        'Preview selected the exact Qwen/Gemma safetensors/tokenizer SHA '
        'tree. At Build, a separate Win32 private-source stage must hold '
        'these same bytes through child execution. Source revision and '
        'native inference remain NOT_ATTESTED / NOT_CHECKED.'
    )


def _selected_family_consumer(data, report):
    opt = data.get('native_family_atomic', False)
    if type(opt) is not bool:
        raise ValueError('native_family_atomic must be boolean')
    source = data.get('native_consumer_kit_source') or ''
    if not opt:
        if data.get('full_build'):
            raise ValueError('A complete LLM bundle requires its tokenizer and runtime kit')
        if source:
            raise ValueError('native consumer kit requires explicit family atomic opt-in')
        return None
    if (not data.get('full_build') or data.get('atomic_complete_bundle') or
            data.get('freeze_local_raw_source') or not isinstance(source, str) or
            not os.path.isabs(source)):
        raise ValueError('native family atomic requires a full-build model and an absolute selected consumer root')
    path = str(data.get('path') or '').strip()
    key = import_adapters.LLMImportAdapter()._key_for(path)
    if key is None:
        raise ValueError('native family atomic requires a recognised native LLM')
    inventory = bundle_publication.inspect_selected_native_consumer_kit(
        source, model_key=key, device=report['target']['device'])
    if data.get('llm_profile'):
        members = {item.path: item for item in inventory.members}
        license_file = model_sources.catalog_model(key)['license_file']
        legal = members.get('LICENSE-Qwen2.5-0.5B-Instruct.txt')
        if (legal is None or legal.size != license_file['bytes'] or
                legal.sha256 != license_file['sha256'] or
                'MODEL-SOURCE.json' not in members or
                not any(name.startswith('NOTICE-tvm-') for name in members)):
            raise ValueError('The supplied Qwen kit must include its pinned model license, source record, and TVM FFI notice')
    return {'source': os.path.abspath(source), 'key': key,
            'device': report['target']['device'], 'inventory': inventory}


def _issue_preview(data):
    """Issue one short-lived authorization for the most recent HTTP Preview."""
    global ACTIVE_PREVIEW, PREVIEW_SERIAL
    with LOCK:
        PREVIEW_SERIAL += 1
        serial = PREVIEW_SERIAL
        ACTIVE_PREVIEW = None
    if not isinstance(data, dict):
        return {'trouble': 'Preview request must be a JSON object'}
    path = str(data.get('path') or '').strip()
    try:
        before = _source_fingerprint(path)
    except (OSError, ValueError, RuntimeError) as exc:
        return {'trouble': str(exc)}
    try:
        report = _preview_build_plan(data)
        if report.get('trouble') or not report.get('ready_for_build'):
            return report
        pack = onnx_source_pack.selected(data)
        if pack is not None:
            report['onnx_source_pack'] = onnx_source_pack.display(pack)
            report['planned_files'].append('model-source/ - declared source metadata, licences and export records')
        family_consumer = _selected_family_consumer(data, report)
        if family_consumer is not None:
            report['native_family_consumer'] = _consumer_preview_display(family_consumer)
        hf = _selected_hf_preview(data)
        if hf is not None:
            report['hf_selected_source'] = _hf_preview_display(hf)
            report['warnings'].append(_hf_preview_warning())
        paddle = _selected_paddle_preview(data)
        if paddle is not None:
            report['paddle_selected_source'] = paddle
            report['warnings'].append(_paddle_preview_warning(
                paddle, data.get('paddle_native_hold') is True))
        frozen = _frozen_raw_preview(
            data, reserved_output=report['output_directory'])
        if frozen is not None:
            report['source_freeze'] = frozen
            report['warnings'].append(_raw_freeze_warning(frozen))
        after = _source_fingerprint(path)
        if after != before:
            return {'trouble': 'Model source changed during Preview; review it again'}
        if paddle is not None and _selected_paddle_preview(data) != paddle:
            return {'trouble': 'Selected Paddle source/dictionary changed during Preview'}
        if hf is not None and _selected_hf_preview(data) != hf:
            return {'trouble': 'Selected HF weights/tokenizer changed during Preview'}
        if pack is not None and onnx_source_pack.selected(data) != pack:
            return {'trouble': 'Source-pack metadata or files changed during Preview'}
        signature = _request_signature(data)
        plan_signature = _request_signature(report)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return {'trouble': 'Cannot pin Preview: %s' % exc}
    with LOCK:
        if serial != PREVIEW_SERIAL:
            return {'trouble': 'A newer Preview superseded this request'}
        token = secrets.token_urlsafe(32)
        ACTIVE_PREVIEW = {
            'token': token, 'request': signature, 'plan': plan_signature,
            'source': after, 'output': report['output_directory'],
            'freeze': frozen, 'paddle': paddle, 'hf': hf,
            'onnx_source_pack': pack,
            'family_consumer': family_consumer,
            'output_parent': _output_parent_identity(report['output_directory']),
            'expires': time.monotonic() + PREVIEW_LIFETIME_SECONDS,
        }
    report['preview_token'] = token
    if after['extent'] != 'full small-file SHA-256':
        report['warnings'].append(
            'All model-source bytes were SHA-256 measured in a streaming scan. '
            'This detects changes between scans but is NOT an immutable model '
            'snapshot or a source lock through compilation. Large source trees '
            'may take time to rescan.')
    return report


def _build_from_preview(data):
    """Revalidate the exact displayed plan, then atomically claim its output."""
    global ACTIVE_PREVIEW
    if not isinstance(data, dict):
        return {'trouble': 'Build request must be a JSON object'}
    options = dict(data)
    wants_atomic = options.get('atomic_complete_bundle', False) is True
    wants_family_atomic = options.get('native_family_atomic', False) is True
    token = options.pop('preview_token', None)
    try:
        request = _request_signature(options)
    except (ValueError, TypeError) as exc:
        return {'trouble': 'Invalid Build options: %s' % exc}
    with LOCK:
        approved = ACTIVE_PREVIEW
        if not (isinstance(token, str) and approved and
                secrets.compare_digest(token, approved['token']) and
                time.monotonic() < approved['expires']):
            return {'trouble': 'Missing, expired or stale Preview token; review again'}


        ACTIVE_PREVIEW = None
        preview_serial = PREVIEW_SERIAL
        if approved['request'] != request:
            return {'trouble': 'Build options differ from reviewed Preview; review again'}

    try:
        before = _source_fingerprint(str(options.get('path') or '').strip())
        if before != approved['source']:
            return {'trouble': 'Model source changed since Preview; review again'}
        fresh_plan = _preview_build_plan(options)
        if fresh_plan.get('trouble'):
            return {'trouble': fresh_plan['trouble']}
        fresh_pack = onnx_source_pack.selected(options)
        if fresh_pack != approved.get('onnx_source_pack'):
            return {'trouble': 'Source-pack metadata or files changed since Preview'}
        if fresh_pack is not None:
            fresh_plan['onnx_source_pack'] = onnx_source_pack.display(fresh_pack)
            fresh_plan['planned_files'].append('model-source/ - declared source metadata, licences and export records')
        fresh_consumer = _selected_family_consumer(options, fresh_plan)
        if fresh_consumer != approved.get('family_consumer'):
            return {'trouble': 'Selected native consumer kit changed since Preview'}
        if fresh_consumer is not None:
            fresh_plan['native_family_consumer'] = _consumer_preview_display(fresh_consumer)
        fresh_hf = _selected_hf_preview(options)
        if fresh_hf != approved.get('hf'):
            return {'trouble': 'Selected HF weights/tokenizer differ from '
                    'reviewed Preview; review again'}
        if fresh_hf is not None:
            fresh_plan['hf_selected_source'] = _hf_preview_display(fresh_hf)
            fresh_plan['warnings'].append(_hf_preview_warning())
        fresh_paddle = _selected_paddle_preview(options)
        if fresh_paddle is not None:
            fresh_plan['paddle_selected_source'] = fresh_paddle
            fresh_plan['warnings'].append(_paddle_preview_warning(
                fresh_paddle, options.get('paddle_native_hold') is True))
        if fresh_paddle != approved.get('paddle'):
            return {'trouble': 'Selected Paddle source/dictionary differs from '
                    'reviewed Preview; review again'}
        fresh_freeze = _frozen_raw_preview(
            options, reserved_output=fresh_plan['output_directory'])
        if fresh_freeze is not None:
            fresh_plan['source_freeze'] = fresh_freeze
            fresh_plan['warnings'].append(_raw_freeze_warning(fresh_freeze))
        if fresh_freeze != approved.get('freeze'):
            return {'trouble': 'Source freeze content/closure differs from '
                    'reviewed Preview; review again'}
        if _request_signature(fresh_plan) != approved['plan']:
            return {'trouble': 'Target or output plan changed since Preview; review again'}
        after = _source_fingerprint(str(options.get('path') or '').strip())
        if after != before:
            return {'trouble': 'Model source changed during Build validation; review again'}
        if (fresh_paddle is not None and
                _selected_paddle_preview(options) != fresh_paddle):
            return {'trouble': 'Selected Paddle source changed during Build '
                    'validation; review again'}
        if (fresh_hf is not None and
                _selected_hf_preview(options) != fresh_hf):
            return {'trouble': 'Selected HF source changed during Build '
                    'validation; review again'}
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return {'trouble': 'Cannot revalidate Preview: %s' % exc}

    with LOCK:
        if PREVIEW_SERIAL != preview_serial or time.monotonic() >= approved['expires']:
            return {'trouble': 'A newer Preview superseded this Build; review again'}
        if _output_parent_identity(approved['output']) != approved['output_parent']:
            return {'trouble': 'Preview output parent changed or was redirected; '
                    'review again'}
        if wants_atomic or wants_family_atomic:
            pending_name = os.path.normcase(os.path.abspath(approved['output']))
            if os.path.lexists(approved['output']) or pending_name in ATOMIC_PENDING_OUTPUTS:
                return {'trouble': 'Atomic final output already occupied/reserved; '
                        'review again for a fresh destination'}
            ATOMIC_PENDING_OUTPUTS.add(pending_name)
            output_identity = ('PENDING_ATOMIC', approved['output_parent'])
        else:
            try:


                os.mkdir(approved['output'])
            except FileExistsError:
                return {'trouble': 'Preview output directory is already occupied; '
                        'review again for a new destination'}
            except OSError as exc:
                return {'trouble': 'Cannot reserve Preview output directory: %s' % exc}
            try:
                output_identity = _reserved_output_identity(approved['output'])
                if _output_parent_identity(approved['output']) != approved['output_parent']:
                    raise ValueError('output parent was redirected during reservation')
            except (ValueError, OSError) as exc:
                return {'trouble': 'Reserved output changed: %s' % exc}

    compiled = dict(options)
    compiled['arch'] = (fresh_plan['target']['sm'] or ''
                        if fresh_plan['target']['device'] == 'cuda' else '')
    try:
        started = {'output_directory': approved['output'],
                   'source_fingerprint': approved['source'],
                   'output_identity': output_identity,
                   'approved_build_plan': fresh_plan['build_plan'],
                   'atomic_complete_bundle': wants_atomic,
                   'approved_family_consumer': fresh_consumer,
                   'native_family_atomic': wants_family_atomic}
        if fresh_freeze is not None:
            started['approved_raw_freeze'] = fresh_freeze
        if fresh_paddle is not None:
            started['approved_paddle_closure'] = fresh_paddle
        if fresh_hf is not None:
            started['approved_hf_closure'] = fresh_hf
        if fresh_pack is not None:
            started['approved_onnx_source_pack'] = fresh_pack
        number = _start_job(compiled, **started)
    except Exception as exc:
        if wants_atomic or wants_family_atomic:
            with LOCK:
                ATOMIC_PENDING_OUTPUTS.discard(
                    os.path.normcase(os.path.abspath(approved['output'])))
        return {'trouble': 'Reserved output could not start Build: %s' % exc}
    return {'job': number}


UPLOAD_CHUNK = 1048576
_ILLEGAL_NAME = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def _safe_upload_name(raw):
    """A basename safe to write inside uploads/, or a fixed fallback.

    The browser hands over the picked file's name but never its path, and even the
    name is treated as hostile. Only the last component is kept - any directory
    part, forward or backward slash, is dropped - characters a path could turn
    against us are replaced, and a name that reduces to nothing or to dots alone is
    refused for a fallback. So the write can never climb out of uploads/.
    """
    name = str(raw or '').replace('\\', '/').split('/')[-1]
    name = _ILLEGAL_NAME.sub('_', name).strip().strip('.').strip()

    dos_base = name.split('.', 1)[0].upper()
    if (dos_base in ('CON', 'PRN', 'AUX', 'NUL') or
            re.fullmatch(r'(?:COM|LPT)[1-9]', dos_base)):
        name = 'upload-' + name
    return name or 'model.bin'


def _form_float(value):
    """A float out of a form field, or None when it was left blank.

    The page sends every field as text, so a number arrives as a string; a blank
    one arrives as the empty string, which must become None rather than a zero.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return float(value)


def _form_int(value):
    """An int out of a form field, or None when it was left blank. See _form_float."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return int(value)


def _contract_from_request(data):
    """Parse the model contract shared by the wizard and command-line builder."""
    return build_contract_input(
        data.get('task') or '', data.get('adapter') or '',
        adapter_api_version=_form_int(data.get('adapter_api_version')) or 1,
        layout=data.get('layout') or 'NCHW',
        score_threshold=_form_float(data.get('score_threshold')),
        nms_threshold=_form_float(data.get('nms_threshold')),
        mean=parse_float_csv(data.get('mean')),
        std=parse_float_csv(data.get('std')),
        keypoint_count=_form_int(data.get('keypoint_count')),
        labels=data.get('labels') or None)


def _llm_plan_arguments(data):
    if not data.get('full_build'):
        if data.get('llm_profile') or data.get('llm_python'):
            raise ValueError('LLM profile and Python selection require a native LLM')
        return {}
    model_key = import_adapters.LLMImportAdapter()._key_for(data.get('path') or '')
    return {
        'llm_options': llm_build_profile.select(
            data.get('llm_profile'), model_key=model_key),
        'compiler_python': llm_build_profile.python_snapshot(
            data.get('llm_python') or import_adapters.LLMImportAdapter.PYTHON),
    }


def _preview_output_directory(data, model_path):
    selected = data.get('output_directory')
    if selected:
        if type(selected) is not str or not os.path.isabs(selected):
            raise ValueError('Output directory must be an absolute local path')
        result = os.path.abspath(selected)
        if os.path.lexists(result):
            raise ValueError('Output directory already exists; choose a new directory')
        if not os.path.isdir(os.path.dirname(result)):
            raise ValueError('Output parent directory does not exist')
    else:
        result = _output_directory(
            model_path, require_new=True, parent=str(app_paths.data_path('bundles')))
    if data.get('full_build'):
        source = os.path.normcase(os.path.realpath(model_path))
        destination = os.path.normcase(os.path.realpath(result))
        try:
            shared = os.path.commonpath((source, destination))
        except ValueError:
            shared = None
        if shared in (source, destination):
            raise ValueError('Choose an output directory outside the original LLM source directory')
    return result


def _preview_build_plan(data):
    """Describe the ACTUAL selected target before building anything.

    This is a preview, not a successful build or a device verification. It uses
    the same target classifier as the core, and refuses options the local
    pipeline cannot honour. Never report a guessed CUDA compatibility result.
    """
    path = str(data.get('path') or '').strip()
    atomic_request = data.get('atomic_complete_bundle', False)
    if type(atomic_request) is not bool:
        return {'trouble': 'atomic_complete_bundle must be a boolean'}
    if atomic_request and (data.get('freeze_local_raw_source') is not True or
                           data.get('full_build')):
        return {'trouble': 'atomic complete-directory candidate requires an '
                'explicit Win32 protected RAW model, not a family build'}
    if atomic_request and data.get('tune'):
        return {'trouble': 'atomic bounded RAW pilot does not package tuning-db; '
                'disable tuning or use the visibly non-atomic legacy route'}
    if atomic_request and not os.path.isabs(path):
        return {'trouble': 'atomic complete-directory candidate requires an '
                'absolute local RAW source path (no relative source alias)'}
    family_atomic = data.get('native_family_atomic', False)
    if type(family_atomic) is not bool:
        return {'trouble': 'native_family_atomic must be boolean'}
    if data.get('full_build') and not family_atomic:
        return {'trouble': 'Select the supplied tokenizer and runtime kit before Preview'}
    if family_atomic and (atomic_request or not data.get('full_build') or
                          not os.path.isabs(path)):
        return {'trouble': 'native family atomic requires absolute local LLM source and separate opt-in'}
    device = str(data.get('target') or 'cpu').strip().lower()
    requested_sm = str(data.get('arch') or '').strip()
    if not path:
        return {'trouble': 'choose and recognise a model before continuing'}
    if device not in ('cpu', 'cuda'):
        return {'trouble': 'unsupported device %r: choose cpu or cuda' % device}
    if device == 'cpu' and requested_sm:
        return {'trouble': 'GPU architecture has no effect on CPU builds; clear it'}
    if device == 'cpu' and data.get('target_driver_api'):
        return {'trouble': 'CUDA driver API has no effect on CPU builds; clear it'}


    contract_scope = None
    contract = None
    if not data.get('full_build') and (data.get('task') or data.get('adapter')):
        try:
            contract = _contract_from_request(data)
            if contract and contract.get('task'):
                contract_scope = validate_vision_adapter(
                    contract['task'], contract.get('adapter'))
            elif data.get('adapter'):
                raise ValueError('adapter was selected without a task; choose '
                                 'a supported task or clear the adapter')
        except (ValueError, TypeError) as trouble:
            return {'trouble': str(trouble)}


    try:
        sm = targets.normalize_cuda_sm(requested_sm) if requested_sm else None
    except ValueError as exc:
        return {'trouble': str(exc)}
    host = targets.host()
    target_os = str(data.get('target_os') or host['os']).strip().lower()
    target_arch = str(data.get('target_cpu_arch') or host['arch']).strip().lower()
    target_driver_api = data.get('target_driver_api') if device == 'cuda' else None
    if target_driver_api == '':
        target_driver_api = None
    try:
        trials = int(data.get('trials_per_task') or 0) or None
        plan = build_plan.resolve(
            host, target=device, req_sm=sm,
            req_os=target_os, req_arch=target_arch,
            req_driver_api=target_driver_api,
            route='family_full_build' if data.get('full_build') else 'raw',
            source=path, shape=data.get('shape') or None,
            dtype=(data.get('dtype') or 'float32') if data.get('full_build') else 'float32',
            tune=bool(data.get('tune')), trials_per_task=trials,
            contract=contract, cpu_codegen=data.get('cpu_codegen'),
            **_llm_plan_arguments(data))
    except (ValueError, TypeError) as exc:
        return {'trouble': str(exc)}
    profile = plan['target_profile']
    level = profile['reachability']['level']
    reason = profile['reachability']['reason']
    selected = profile['deployment']
    if level in (targets.RUNNER_NEEDED, targets.UNSUPPORTED_V1):
        return {'trouble': reason}
    if device == 'cuda' and not selected['sm']:
        return {'trouble': 'no CUDA card was detected: enter the target compute capability'}
    if not path.lower().startswith(('http://', 'https://')) and not os.path.exists(path):
        return {'trouble': 'the selected source model does not exist: %s' % path}

    if data.get('cpu_codegen') is not None:


        if not os.path.isfile(path):
            return {'trouble': 'selected CPU codegen requires a local ONNX file '
                    'for tuned compilation, not a URL or model directory'}
        try:
            cpu_source = detect.recognise(path)
        except (OSError, ValueError, TypeError) as exc:
            return {'trouble': 'cannot identify selected CPU ONNX source: %s'
                    % exc}
        if cpu_source.kind != detect.ONNX:
            return {'trouble': 'selected CPU codegen is supported only for '
                    'ONNX tuned CPU compilation; no TFLite or exporter fallback'}

    full_build = bool(data.get('full_build'))
    if full_build and path.lower().startswith(('http://', 'https://')):
        return {'trouble': 'full_build requires a local recognised HF model '
                'directory; download and pin the source before Preview'}
    if full_build or os.path.isdir(path):


        try:
            recognised = detect.recognise(path)
            adapter = (import_adapters.find_adapter(path, recognised)
                       if not recognised.convertible else None)
            actual_full = bool(adapter is not None and
                               getattr(adapter, 'kind', 'export') == 'full_build')
        except (OSError, ValueError, TypeError) as exc:
            return {'trouble': 'cannot identify source family for Preview: %s'
                    % exc}
        if full_build != actual_full:
            return {'trouble': 'full_build selection differs from the actual '
                    'registered model-family route; recognise the source again'}

    if path.lower().startswith(('http://', 'https://')):


        eventual_path = str(app_paths.data_path('downloads') / 'model.bin')
    else:
        eventual_path = path
    path_name = os.path.splitext(os.path.basename(eventual_path.rstrip('/\\')))[0]
    try:
        output_dir = _preview_output_directory(data, eventual_path)
    except (OSError, ValueError) as exc:
        return {'trouble': str(exc)}


    module = (None if full_build else
              'model.' + device + pipeline.module_extension())
    files = (['family builder bundle (model.json, native library, weight index/data)',
              'family manifest.txt / manifest.json (runtime contract, NOT numerical VERIFIED)',
              'build-plan.json', 'recipe.json', 'report.txt',
              'import-provenance.json'] if full_build else
             [module, module + '.inputs', 'recipe.json', 'report.txt',
              'matching TVM runtime libraries and redistribution licences'])


    conditional_files = (
        ['manifest.txt', 'manifest.json']
    if not full_build and data.get('task') else [])
    checks = ('The family builder has NO independent numerical oracle; '
              'result will remain NOT_CHECKED on this and other machines'
              if full_build else
              'The target GPU is not on this build host: compilation only; '
              'numeric verification on the destination is still required'
              if level == targets.COMPILE_ONLY else
              'A local numeric verification is planned, NOT yet completed')
    warnings = []
    if atomic_request:


        files += [
            'source/ - exact selected RAW model and approved sidecars (full SHA-256)',
            'source-closure.json only for selected ONNX external_data '
            '(issued AFTER Win32 source-lock post-audit)',
        ]
        warnings.append(
            'EXPERIMENTAL opt-in RAW atomic bundle: at most 512 members and '
            '512 MiB on a single Windows NTFS volume. No-clobber whole-directory '
            'namespace commit is NOT power-loss durable, NOT release-grade '
            'immutable and NOT native end-target execution proof. Compiler, '
            'source-copy, SHA, runtime/recipe/report and post-lock audits must '
            'all succeed before HTTP READY; no tuning database is packaged.')
    if family_atomic:
        files += ['caller-selected tokenizer/runtime/licences (approved full SHA-256)',
                  'complete native family kit, atomically renamed after validation']
        warnings.append('The selected tokenizer and runtime files are checked before and after the build. Open and run the included Delphi or Lazarus example to check generated text. Incomplete builds remain separate from the final output directory.')
    if contract_scope == 'INDIVIDUAL_RAW_STAGE_ONLY':
        warnings.append(
            'PaddleOCR Det or Rec is an independently verifiable RAW stage, '
            'not a ready Pascal Det+Rec+dictionary composite. A separate '
            'native CTC/image oracle is required for the combined passport.')
    build_cuda = None
    runtime_inventory = None
    if not full_build:
        from pascalvision_convert import runtime
        try:
            runtime_options = {'include_ffi_licences': True} if data.get('onnx_source_pack') else {}
            runtime_inventory = runtime.plan_runtime_files(device, **runtime_options)
        except (ImportError, OSError, ValueError) as exc:
            return {'trouble': 'cannot inspect runtime libraries before Build: %s'
                    % exc}
        if not runtime_inventory['ready']:
            return {
                'trouble': ('the output runtime would be incomplete: ' +
                            ', '.join(runtime_inventory['missing'])),
            }
    if device == 'cuda':
        nvcc = environment.find_cuda() if not full_build else None
        if not full_build and not nvcc:
            return {'trouble': 'CUDA compilation is unavailable on this computer: '
                    'no NVIDIA toolkit compiler was found. Choose CPU or install '
                    'a compatible toolkit.'}
        if not full_build and level == targets.LOCAL:
            from pascalvision_convert import cuda_preflight
            cuda = cuda_preflight.check_cuda_toolkit(
                nvcc, environment._ask_program)
            build_cuda = {
                'compiler': nvcc,
                'driver_api': cuda.get('driver_api'),
                'toolkit_version': cuda.get('toolkit_version'),
                'runtime_version': cuda.get('runtime_version'),
                'status': cuda.get('status'),
                'reason': cuda.get('reason'),
            }
            if cuda['status'] == 'incompatible':
                return {'trouble': 'CUDA build would use an incompatible toolkit: ' +
                        cuda['reason'] + '. Choose CPU or another compatible toolkit.'}
            if cuda['status'] != 'compatible':
                warnings.append('Local CUDA runtime preflight is not confirmed: ' +
                                cuda['reason'])
        else:
            warnings.append('Target runtime compatibility must be tested by its native builder/runner')
            if nvcc:
                warnings.append('The CUDA compiler found on this computer: ' + nvcc)
        destination_driver = selected.get('driver_api')
        if destination_driver is not None and not full_build:
            from pascalvision_convert import cuda_preflight
            toolkit = cuda_preflight.installed_toolkit_version(
                nvcc, environment._ask_program)
            state, statement = cuda_preflight.compare_driver_toolkit(
                destination_driver, toolkit)
            if state == 'incompatible':
                return {
                    'trouble': ('selected deployment CUDA driver is incompatible '
                                'with the installed build toolkit: ' + statement),
                }
            if state == 'unknown':
                warnings.append(
                    'Target driver/toolkit compatibility is not established: ' +
                    statement)
            else:
                warnings.append(
                    'Target CUDA Driver API check: ' + statement +
                    '; this is not yet an end-device model test')
        warnings.append('The binary architecture must be checked after TVM code generation')
    return {
        'target': selected,
        'target_profile': profile,
        'build_plan': plan,
        'reachability': level,
        'reason': reason,
        'verification': checks,
        'output_directory': output_dir,
        'planned_files': files,
        'conditional_files': conditional_files,
        'build_cuda': build_cuda,
        'runtime_inventory': runtime_inventory,
        'warnings': warnings,
        'ready_for_build': True,
        'model': path_name,
    }


def _start_job(data, *, output_directory=None, source_fingerprint=None,
               output_identity=None, approved_raw_freeze=None,
               approved_build_plan=None, atomic_complete_bundle=False,
               approved_paddle_closure=None, native_family_atomic=False,
               approved_family_consumer=None, approved_hf_closure=None,
               approved_onnx_source_pack=None):
    """Run a build on its own thread and return the job number."""
    if (approved_onnx_source_pack != onnx_source_pack.selected(data) or
            (approved_onnx_source_pack is not None and
             (source_fingerprint is None or approved_build_plan is None))):
        raise ValueError('ONNX source pack lacks its exact Preview approval')
    wants_freeze = data.get('freeze_local_raw_source', False)
    if wants_freeze is not False and wants_freeze is not True:
        raise ValueError('freeze_local_raw_source must be boolean')
    if type(atomic_complete_bundle) is not bool:
        raise ValueError('atomic_complete_bundle must be boolean')
    if type(native_family_atomic) is not bool:
        raise ValueError('native_family_atomic must be boolean')
    if native_family_atomic != (data.get('native_family_atomic', False) is True):
        raise ValueError('native family opt-in differs from worker authorization')
    if not native_family_atomic and data.get('native_consumer_kit_source'):
        raise ValueError('native consumer source supplied without approved opt-in')
    if atomic_complete_bundle and (not wants_freeze or
            not isinstance(output_directory, str) or
            output_identity != ('PENDING_ATOMIC',
                                _output_parent_identity(output_directory))):
        raise ValueError('atomic complete-directory build requires a pinned '
                         'protected RAW Preview and an uncreated final name')
    if native_family_atomic:
        if (atomic_complete_bundle or wants_freeze or not data.get('full_build') or
                not isinstance(approved_family_consumer, dict) or
                output_identity != ('PENDING_ATOMIC',
                                    _output_parent_identity(output_directory)) or
                source_fingerprint is None or approved_build_plan is None or
                approved_family_consumer != _selected_family_consumer(
                    data, {'target': {'device': data.get('target', 'cpu')}})):
            raise ValueError('native family consumer differs from Preview-approved Inventory')
    elif approved_family_consumer is not None:
        raise ValueError('unrequested native family Inventory')
    if approved_hf_closure is not None:
        if (not isinstance(approved_hf_closure, dict) or
                source_fingerprint is None or not data.get('full_build') or
                approved_build_plan is None or
                approved_hf_closure.get('status') != 'SCANNED_NOT_LOCKED' or
                approved_hf_closure.get('verification_status') != 'NOT_CHECKED' or
                approved_hf_closure.get('release_grade_immutable') is not False):
            raise ValueError('native HF selected Preview approval is malformed '
                             'or lacks its shared BuildPlan/source authorization')
        from pascalvision_convert import hf_closure
        observed = hf_closure.scan_selected_hf_source(
            str(data.get('path') or '').strip(),
            expected_model_key=approved_hf_closure.get('model_key'))
        if observed != approved_hf_closure:
            raise ValueError('native HF source differs from Preview-approved '
                             'selected file closure BEFORE worker allocation')
    if wants_freeze:
        source_kind = (approved_raw_freeze.get('source_kind')
                       if isinstance(approved_raw_freeze, dict) else None)
        approved_status = {
            'file': 'STATIC_SELF_CONTAINED_CANDIDATE_NOT_LOCKED',
            'selected_onnx_files': 'SELECTED_FILES_SCANNED_NOT_LOCKED',
        }.get(source_kind)
        if (source_fingerprint is None or output_identity is None
                or not output_directory
                or not isinstance(approved_raw_freeze, dict)
                or approved_status is None
                or approved_raw_freeze.get('status') != approved_status):
            raise ValueError('RAW source freezing requires an exact approved '
                             'Preview, output identity and full source pin')
    elif approved_raw_freeze is not None:
        raise ValueError('Unexpected source freeze pin without an opt-in')
    if approved_paddle_closure is None:
        if (data.get('paddle_role') or data.get('paddle_dictionary') or
                data.get('paddle_native_hold')):
            raise ValueError('Paddle selected source requires an approved '
                             'Preview tree; a direct worker cannot forge it')
    else:
        if (not isinstance(approved_paddle_closure, dict) or
                source_fingerprint is None or not output_directory or
                output_identity is None or wants_freeze or
                atomic_complete_bundle or
                _selected_paddle_preview(data) != approved_paddle_closure):
            raise ValueError('Paddle selected source differs from the exact '
                             'Preview-approved graph, weights or dictionary')
    with LOCK:
        number = str(len(JOBS) + 1)
        if SHUTTING_DOWN:
            raise RuntimeError('The converter is closing; no new build was started')
        JOBS[number] = {'finished': False, 'sound': False,
                        'download_ready': False, 'steps': [], 'summary': {}}

    def work():
        path = data.get('path', '').strip()
        requested_source = path
        frozen_snapshot = None
        approved_native_io = None
        approved_bundle_sources = None
        private_atomic_stage = None
        private_atomic_stage_id = None
        private_source_snapshot_path = None
        private_source_snapshot_id = None
        retained_ledger_run = None
        retained_ledger_job_nonce = None
        retained_ledger_confirmed_receipts = {}
        retained_ledger_confirmed_intents = {}
        retained_ledger_publication_observation = None
        retained_ledger_precommit_observation = None
        deferred_lease_download_ready = False
        committed_inventory = None

        def ledger_observation():
            confirmed = dict(retained_ledger_confirmed_receipts)
            state = ('TWO_DPAPI_RECEIPTS_WRITTEN_NOT_APPLICATION_AUTH'
                     if set(confirmed) == {'output', 'snapshot'} else
                     'PARTIAL_DPAPI_RECEIPTS_NOT_RECOVERED'
                     if confirmed else 'NO_CONFIRMED_DPAPI_RECEIPTS')
            result = ({
                'evidence_run': retained_ledger_run,
                'status': state,
                'receipts': confirmed,
                'intents': dict(retained_ledger_confirmed_intents),
                'ready_recovered': False,
                'cleanup_allowed': False,
                'power_loss_durable': False,
            } if retained_ledger_run is not None else None)
            if result is not None and retained_ledger_publication_observation is not None:
                result['publication_observation'] = retained_ledger_publication_observation
            if result is not None and retained_ledger_precommit_observation is not None:
                result['precommit_observation'] = retained_ledger_precommit_observation
            return result

        def remember_output_intent(stage_path):
            if retained_ledger_run is None:
                raise RuntimeError('unapproved output INTENT outside opt-in journal')
            candidate = bundle_publication.bundle_atomic._local_absolute(stage_path)
            if 'output' in retained_ledger_confirmed_intents:
                raise RuntimeError('duplicate output stage INTENT')
            retained_ledger_confirmed_intents['output'] = (
                retained_ledger.record_selected_stage_intent(
                    evidence_directory=retained_ledger_run,
                    output_final=output_directory, selected_stage=candidate,
                    kind='output', job_nonce=retained_ledger_job_nonce))

        def remember_created_output(created_path, identity):
            nonlocal private_atomic_stage_id, private_atomic_stage
            if (private_atomic_stage_id is not None or
                    private_atomic_stage is not None or
                    not isinstance(identity, tuple) or len(identity) != 2 or
                    any(type(value) is not int for value in identity)):
                raise RuntimeError('private atomic stage creation identity invalid')
            raw = bundle_publication.bundle_atomic
            candidate = raw._local_absolute(created_path)
            selected_parent = os.path.normcase(os.path.dirname(
                os.path.abspath(output_directory)))
            if (os.path.normcase(os.path.dirname(candidate)) != selected_parent or
                    not raw.STAGE_NAME.fullmatch(os.path.basename(candidate))):
                raise RuntimeError('private atomic stage creation path invalid')


            private_atomic_stage = candidate
            private_atomic_stage_id = identity
            if retained_ledger_run is not None:
                if 'output' not in retained_ledger_confirmed_intents:
                    raise RuntimeError('created output has no signed prior INTENT')


                retained_ledger_confirmed_receipts['output'] = (
                    retained_ledger.record_selected_retained_stage(
                    evidence_directory=retained_ledger_run,
                    output_final=output_directory, selected_stage=candidate,
                    kind='output', job_nonce=retained_ledger_job_nonce,
                    created_file_id=identity, require_intent=True))

        def remember_created_snapshot(identity):
            nonlocal private_source_snapshot_id
            if (private_source_snapshot_id is not None or
                    not isinstance(identity, tuple) or len(identity) != 2 or
                    any(type(value) is not int for value in identity)):
                raise RuntimeError('private source snapshot creation identity invalid')
            private_source_snapshot_id = identity
            if retained_ledger_run is not None:
                if 'snapshot' not in retained_ledger_confirmed_intents:
                    raise RuntimeError('created snapshot has no signed prior INTENT')


                retained_ledger_confirmed_receipts['snapshot'] = (
                    retained_ledger.record_selected_retained_stage(
                    evidence_directory=retained_ledger_run,
                    output_final=output_directory,
                    selected_stage=private_source_snapshot_path,
                    kind='snapshot', job_nonce=retained_ledger_job_nonce,
                    created_file_id=identity, require_intent=True))

        def remember_validated_precommit(stage, final, inventory):
            nonlocal retained_ledger_precommit_observation
            if retained_ledger_run is None or retained_ledger_precommit_observation is not None:
                raise RuntimeError('duplicate or unauthorized PRECOMMIT observation')
            if (os.path.normcase(os.path.abspath(stage)) !=
                    os.path.normcase(os.path.abspath(private_atomic_stage)) or
                    os.path.normcase(os.path.abspath(final)) !=
                    os.path.normcase(os.path.abspath(output_directory))):
                raise RuntimeError('PRECOMMIT callback differs from selected output')
            retained_ledger_precommit_observation = (
                retained_ledger.record_selected_precommit_inventory_observation(
                    evidence_directory=retained_ledger_run,
                    output_final=output_directory, selected_stage=stage,
                    job_nonce=retained_ledger_job_nonce,
                    approved_inventory=inventory))

        selected_file_set = bool(
            wants_freeze and approved_raw_freeze['source_kind']
            == 'selected_onnx_files')


        stage_locks = ExitStack()


        lease_stack = ExitStack()
        stage_ticket = None
        exported_onnx_ticket = None
        paddle_rec_dictionary_receipt = None
        def recheck_paddle(stage):
            if approved_paddle_closure is None:
                return
            from pascalvision_convert import paddle_closure
            observed = paddle_closure.audit_selected_paddle_source(
                requested_source, role=approved_paddle_closure['role'],
                dictionary=(approved_paddle_closure['dictionary'] or {}).get(
                    'reference'),
                expected_tree_sha256=approved_paddle_closure['tree_sha256'])
            if observed != approved_paddle_closure:
                raise RuntimeError('Paddle selected bytes/metadata changed '
                                   '%s; cannot authorize this Build' % stage)

        def recheck_hf(stage):
            if approved_hf_closure is None:
                return
            from pascalvision_convert import hf_closure
            observed = hf_closure.scan_selected_hf_source(
                requested_source,
                expected_model_key=approved_hf_closure['model_key'])
            if observed != approved_hf_closure:
                raise RuntimeError('Preview-approved HF selected bytes changed '
                                   '%s; must not dispatch the native builder'
                                   % stage)

        def recheck_model(stage):
            if (approved_onnx_source_pack is not None and
                    onnx_source_pack.selected(data) != approved_onnx_source_pack):
                raise RuntimeError('ONNX source pack changed ' + stage)
            if (source_fingerprint is not None and
                    _source_fingerprint(requested_source) != source_fingerprint):
                raise RuntimeError('Model source changed %s; Build output must '
                                   'not be published as verified' % stage)
            if (selected_file_set and
                    _raw_source_freeze_candidate(
                        requested_source, forbidden_roots=(output_directory,))
                    != approved_raw_freeze):
                raise RuntimeError('ONNX external tensor dependency closure '
                                   'changed %s; request a new Preview' % stage)

        def recheck_staged(stage):
            if frozen_snapshot is not None:
                if selected_file_set:
                    checked = onnx_closure.audit_selected_closure(
                        frozen_snapshot['snapshot_directory'],
                        expected_tree_sha256=approved_raw_freeze['tree_sha256'])
                    if (checked['compiler_input']
                            != frozen_snapshot['compiler_input'] or
                            checked['tree_sha256']
                            != approved_raw_freeze['tree_sha256'] or
                            checked['model_rel'] != approved_raw_freeze['model_rel'] or
                            checked['dependencies']
                            != approved_raw_freeze['dependencies']):
                        raise RuntimeError('Frozen ONNX/sidecar set changed %s'
                                           % stage)
                else:
                    checked = source_freeze.audit_frozen_source(
                        frozen_snapshot['snapshot_directory'])
                    if (checked['compiler_input']
                            != frozen_snapshot['compiler_input'] or
                            checked['snapshot_tree_sha256']
                            != approved_raw_freeze['tree_sha256']):
                        raise RuntimeError('Frozen compiler model bytes changed %s'
                                           % stage)
        try:
            if atomic_complete_bundle:


                configured = os.environ.get(
                    'PASCALVISION_I03_RETAINED_EVIDENCE_ROOT')
                if configured:
                    retained_ledger_run = retained_ledger.create_unique_evidence_run(
                        configured)
                    retained_ledger_job_nonce = secrets.token_hex(16)
                    lease_stack.enter_context(
                        retained_ledger.hold_active_stage_lease(
                            retained_ledger_run, retained_ledger_job_nonce,
                            output_directory))
            if source_fingerprint is not None:
                recheck_model('after Preview')
                if atomic_complete_bundle or native_family_atomic:
                    if (_output_parent_identity(output_directory) != output_identity[1]
                            or os.path.lexists(output_directory)):
                        raise RuntimeError('Atomic final output was created or parent '
                                           'redirected after Preview; no overwrite')
                else:
                    if (output_identity is not None and
                            _reserved_output_identity(output_directory) != output_identity):
                        raise RuntimeError('Reserved output directory identity changed '
                                           'after Preview; nothing was overwritten')
                    if not (os.path.isdir(output_directory) and
                            not os.listdir(output_directory)):
                        raise RuntimeError('Reserved output directory was changed '
                                           'before compilation; nothing was overwritten')


            contract = _contract_from_request(data)
            if path.lower().startswith(('http://', 'https://')):
                downloads = str(app_paths.data_path('downloads'))
                os.makedirs(downloads, exist_ok=True)
                path = pipeline.download(path, downloads)


            published_path = (output_directory if output_directory is not None
                              else _output_directory(path))
            build_directory = (bundle_publication.create_private_stage(
                published_path, on_created=remember_created_output,
                on_intent=(remember_output_intent
                           if retained_ledger_run is not None else None),
                **({'capacity_profile': 'native_family'}
                   if native_family_atomic else {}))
                     if atomic_complete_bundle or native_family_atomic else published_path)
            if atomic_complete_bundle or native_family_atomic:
                if private_atomic_stage != build_directory:
                    raise RuntimeError('private atomic stage callback/path mismatch')


                identity = bundle_publication.bundle_atomic._guard_stat(
                    build_directory, directory=True)
                if (identity.st_dev, identity.st_ino) != private_atomic_stage_id:
                    raise RuntimeError('private atomic stage changed after creation')


            import_provenance = None
            if wants_freeze:


                current_freeze = _frozen_raw_preview(
                    data, reserved_output=output_directory)
                if current_freeze != approved_raw_freeze:
                    raise RuntimeError('RAW model bytes/dependencies differ '
                                       'from exact Preview-approved source')
                if atomic_complete_bundle:
                    if (_output_parent_identity(output_directory) != output_identity[1]
                            or os.path.lexists(output_directory)):
                        raise RuntimeError('Atomic final output appeared before '
                                           'RAW model source freeze')
                elif _reserved_output_identity(output_directory) != output_identity:
                    raise RuntimeError('Reserved output identity changed before '
                                       'RAW model source freeze')
                private_dir = (os.path.abspath(output_directory) +
                               '.source-freeze-' + secrets.token_hex(12))
                if atomic_complete_bundle:


                    private_source_snapshot_path = private_dir
                    if retained_ledger_run is not None:


                        retained_ledger_confirmed_intents['snapshot'] = (
                            retained_ledger.record_selected_stage_intent(
                                evidence_directory=retained_ledger_run,
                                output_final=output_directory,
                                selected_stage=private_dir, kind='snapshot',
                                job_nonce=retained_ledger_job_nonce))
                if selected_file_set:


                    frozen_snapshot = onnx_closure.freeze_selected_closure(
                        requested_source, private_dir,
                        expected_tree_sha256=approved_raw_freeze['tree_sha256'],
                        forbidden_roots=(output_directory,),
                        on_created=(remember_created_snapshot
                                    if atomic_complete_bundle else None))
                else:
                    frozen_snapshot = source_freeze.freeze_source(
                        requested_source, private_dir,
                        expected_tree_sha256=approved_raw_freeze['tree_sha256'],
                        expected_source_kind='file', max_entries=1,
                        max_bytes=_FROZEN_RAW_GRAPH_MAX_BYTES,
                        on_created=(remember_created_snapshot
                                    if atomic_complete_bundle else None))
                if atomic_complete_bundle:


                    observed = bundle_publication.bundle_atomic._guard_stat(
                        private_dir, directory=True)
                    if ((observed.st_dev, observed.st_ino)
                            != private_source_snapshot_id):
                        raise RuntimeError('private source snapshot changed after creation')
                path = frozen_snapshot['compiler_input']
                holder = (source_lock.hold_staged_file_set
                          if selected_file_set else
                          source_lock.hold_staged_single_file)
                stage_ticket = stage_locks.enter_context(holder(
                    frozen_snapshot['snapshot_directory'],
                    expected_tree_sha256=approved_raw_freeze['tree_sha256']))
                if (os.path.normcase(os.path.abspath(stage_ticket.compiler_input))
                        != os.path.normcase(os.path.abspath(path))
                        or stage_ticket.source_tree_sha256
                        != approved_raw_freeze['tree_sha256']
                        or stage_ticket.bytes != approved_raw_freeze['total_bytes']
                        or stage_ticket.release_grade_immutable is not False):
                    raise RuntimeError('Locked Win32 staged source identity/bytes '
                                       'differ from Preview-approved source')
                recheck_staged('immediately after staging')
                if (not selected_file_set and
                        _frozen_raw_closure(path) != approved_raw_freeze):
                    raise RuntimeError('Staged RAW source closure differs '
                                       'from Preview-approved original')
                recheck_model('while making a private RAW source copy')
                _record_step(
                    number, 'Local RAW source freeze (opt-in)', 'done', 'copied',
                    ['full SHA-256 source/staged copy validated',
                     'compiler will consume an exact selected private model '
                     'and its allowed sidecars where applicable',
                     'original source provenance preserved; native Win32 '
                     'share-read-only handles held through compiler and Verify',
                     'candidate only, NOT release-grade immutable'])
            recognised = detect.recognise(path)


            recheck_model('during model recognition')
            recheck_staged('during model recognition')
            if wants_freeze and (
                    not recognised.convertible or
                    recognised.kind != approved_raw_freeze['format']):
                raise RuntimeError('Staged source was not recognised as the '
                                   'original reviewed RAW format')
            if atomic_complete_bundle:


                approved_native_io = bundle_publication.expected_native_inputs(
                    path, recognised.kind, data.get('shape') or None)
            adapter = (import_adapters.find_adapter(path, recognised)
                       if not recognised.convertible else None)
            if approved_paddle_closure is not None and (
                    adapter is None or adapter.name != 'paddleocr' or
                    getattr(adapter, 'kind', 'export') != 'export'):
                raise RuntimeError('Preview-approved Paddle source no longer '
                                   'uses the registered Paddle exporter')
            is_full_family = (adapter is not None and
                              getattr(adapter, 'kind', 'export') == 'full_build')
            if approved_hf_closure is not None and (
                    not is_full_family or adapter.name != 'llm' or
                    adapter._key_for(path) != approved_hf_closure['model_key']):
                raise RuntimeError('Preview-approved Qwen/Gemma source changed '
                                   'family registration before native compiler')
            if atomic_complete_bundle and adapter is not None:
                raise RuntimeError('atomic protected RAW kit cannot call a family '
                                   'builder/exporter before format-aware closure')
            if ('full_build' in data and
                    bool(data.get('full_build')) != is_full_family):
                raise RuntimeError('selected full_build mode differs from actual '
                                   'recognised model family; review Preview again')
            if adapter is not None and getattr(adapter, 'kind', 'export') == 'full_build':


                label = 'Full build: %s' % adapter.name
                recheck_hf('immediately before native full-family build')


                device = data.get('target', 'cpu')
                arch = data.get('arch') or ''
                precision = data.get('dtype') or 'float32'
                plan = build_plan.resolve(
                    targets.host(), target=device, req_sm=arch or None,
                    req_os=data.get('target_os') or None,
                    req_arch=data.get('target_cpu_arch') or None,
                    req_driver_api=data.get('target_driver_api') or None,
                    route='family_full_build', source=requested_source,
                    dtype=precision, shape=data.get('shape') or None,
                    tune=bool(data.get('tune')),
                    trials_per_task=(int(data.get('trials_per_task') or 0)
                                     or None),
                    **_llm_plan_arguments(data))
                if (approved_build_plan is not None and
                        approved_build_plan != plan):
                    raise RuntimeError('LLM target/BuildPlan changed since Preview; '
                                       'no unapproved target may be compiled')
                arch = plan['target_profile']['deployment']['sm'] or ''
                if native_family_atomic:
                    if adapter.name != 'llm':
                        raise RuntimeError('native family atomic supports LLM only')
                    selected = approved_family_consumer
                    if selected['key'] != adapter._key_for(path) or selected['device'] != device:
                        raise RuntimeError('native family source/profile changed')
                    bundle_publication.stage_selected_native_consumer_kit(
                        selected['source'], build_directory, selected['inventory'],
                        model_key=selected['key'], device=device)
                    from pascalvision_convert import native_consumer_hold


                    stage_locks.enter_context(
                        native_consumer_hold.hold_private_native_consumer_stage(
                            build_directory, selected['inventory'],
                            model_key=selected['key'], device=device))
                _record_step(number, label, 'running', 'building',
                             ['building the native bundle with the %s builder - '
                              'this can take a few minutes' % adapter.name,
                              'device %s%s, weights %s'
                              % (device, (', arch ' + arch) if arch else '',
                                 precision)])
                adapter_options = {}
                if native_family_atomic:
                    adapter_options.update({
                        'consumer_kit_source': approved_family_consumer['source'],
                        'approved_consumer_inventory': approved_family_consumer['inventory'],
                    })
                if approved_hf_closure is not None:
                    adapter_options['approved_hf_source_tree_sha256'] = (
                        approved_hf_closure['tree_sha256'])
                built = adapter.build(path, build_directory, target=device, arch=arch,
                                      dtype=precision, build_plan=plan,
                                      **adapter_options)
                recheck_model('during full-family model build')
                recheck_hf('after native full-family build')
                if approved_hf_closure is not None:


                    proven = built.get('provenance')
                    if type(proven) is not dict:
                        raise RuntimeError('Preview-approved HF builder returned '
                                           'no selected-source provenance')
                    selected_fields = (
                        'schema', 'schema_version', 'model_key', 'layout',
                        'entries', 'file_count', 'tensor_count', 'total_bytes',
                        'tree_sha256', 'status', 'verification_status',
                        'release_grade_immutable')
                    expected_selected = {
                        field: approved_hf_closure[field]
                        for field in selected_fields}
                    held = proven.get('compiler_held_source')
                    if (proven.get('adapter') != 'llm' or
                            proven.get('model_key') !=
                            approved_hf_closure['model_key'] or
                            built.get('key') !=
                            approved_hf_closure['model_key'] or
                            built.get('target') != device or
                            built.get('arch') != arch or
                            built.get('dtype') != precision or
                            proven.get('target') != device or
                            proven.get('arch') != (arch or None) or
                            proven.get('dtype') != precision or
                            proven.get('build_plan_sha256') !=
                            plan['plan_sha256'] or
                            os.path.normcase(os.path.abspath(str(
                                proven.get('source') or ''))) !=
                            os.path.normcase(os.path.abspath(requested_source)) or
                            proven.get('hf_source_closure') != expected_selected or
                            type(held) is not dict or
                            held.get('status') !=
                            'WIN32_HF_SELECTED_PRIVATE_STAGE_CANDIDATE_NOT_RELEASE_GRADE' or
                            held.get('tree_sha256') !=
                            approved_hf_closure['tree_sha256'] or
                            held.get('file_count') !=
                            approved_hf_closure['file_count'] or
                            held.get('total_bytes') !=
                            approved_hf_closure['total_bytes'] or
                            held.get('verification_status') != 'NOT_CHECKED' or
                            held.get('revision_status') != 'NOT_ATTESTED' or
                            held.get('release_grade_immutable') is not False):
                        raise RuntimeError('Preview-approved HF builder missing '
                                           'matching private Win32 selected-stage '
                                           'post-audit receipt; refusing READY')
                for option, expected in (('target', device), ('arch', arch),
                                         ('dtype', precision)):
                    if (option in built and built[option] != expected):
                        raise RuntimeError('native family builder returned %s=%r, '
                                           'approved BuildPlan requires %r' % (
                                               option, built[option], expected))


                spoken = [ln for ln in (built.get('log_tail') or '').splitlines()
                          if ln.strip()]
                tail = spoken[-8:]
                if len(spoken) > len(tail):
                    tail = ['last %d lines of %d from the builder:'
                            % (len(tail), len(spoken))] + tail
                _record_step(number, label, 'done', 'built', tail or ['built'])


                write_family_receipt(
                    plan, build_directory, model_key=built['key'],
                    provenance=built['provenance'])
                if native_family_atomic:


                    stage_locks.close()
                    recheck_model('before native family publication')
                    committed_inventory, committed_receipt = (
                        bundle_publication.commit_native_family_stage(
                            build_directory, published_path, plan, model_key=built['key'],
                            approved_consumer_inventory=approved_family_consumer['inventory']))
                    build_directory = committed_receipt.destination
                    deferred_lease_download_ready = True
                with LOCK:
                    JOBS[number].update({
                        'finished': not native_family_atomic, 'sound': True,
                        'download_ready': not native_family_atomic,
                        'bundle_root': os.path.abspath(build_directory),
                        'steps': JOBS[number]['steps'],
                        'summary': {'directory': build_directory if native_family_atomic else built['bundle_dir'],
                                    'verification_status': 'NOT_CHECKED',
                                    'acceptance': consumer_acceptance.after_native_build(plan),
                                    **({'atomic_bundle': {
                                        'status': committed_receipt.status,
                                        'tree_sha256': committed_receipt.tree_sha256,
                                        'release_grade_immutable': False,
                                        'power_loss_durable': False}}
                                       if native_family_atomic else {}),
                                    'verification_note': (
                                        'Built. Open and run the included Delphi or Lazarus '
                                        'example to verify. The wizard has not '
                                        'run the application.')},
                    })
                    if native_family_atomic:
                        JOBS[number]['_atomic_inventory'] = committed_inventory
                return
            if adapter is not None:


                recheck_paddle('immediately before Paddle exporter')
                label = 'Import: %s to ONNX' % adapter.name
                _record_step(number, label, 'running', 'exporting',
                             ['exporting %s to a canonical ONNX'
                              % os.path.basename(path)])
                if approved_paddle_closure is None:
                    exported = adapter.export(path, build_directory)
                elif data.get('paddle_native_hold') is True:
                    exported = adapter.export_held(
                        path, build_directory, stage_parent=os.path.dirname(build_directory),
                        role=approved_paddle_closure['role'],
                        dictionary=(approved_paddle_closure['dictionary'] or {}).get(
                            'reference'),
                        approved_tree_sha256=approved_paddle_closure['tree_sha256'])
                else:
                    exported = adapter.export(
                        path, build_directory, role=approved_paddle_closure['role'],
                        dictionary=(approved_paddle_closure['dictionary'] or {}).get(
                            'reference'),
                        approved_tree_sha256=approved_paddle_closure['tree_sha256'])
                    if (exported.get('provenance') or {}).get(
                            'selected_source_closure') != approved_paddle_closure:
                        raise RuntimeError('Paddle exporter failed to return '
                                           'the exact Preview-approved source receipt')
                if (approved_paddle_closure is not None and
                        (exported.get('provenance') or {}).get(
                            'selected_source_closure') != approved_paddle_closure):
                    raise RuntimeError('Paddle exporter selected-source receipt '
                                       'does not match Preview approval')
                recheck_paddle('after Paddle exporter')
                recheck_model('during framework ONNX export')
                path = exported['onnx']
                import_provenance = exported['provenance']
                if (approved_paddle_closure is not None and
                        data.get('paddle_native_hold') is True):


                    exporter_hold = import_provenance.get('compiler_held_source')
                    if (type(exporter_hold) is not dict or
                            exporter_hold.get('status') !=
                            'WIN32_PADDLE_SELECTED_HELD_CANDIDATE_NOT_RELEASE_GRADE' or
                            exporter_hold.get('source_tree_sha256') !=
                            approved_paddle_closure['tree_sha256'] or
                            exporter_hold.get('release_grade_immutable') is not False or
                            exporter_hold.get('verification_status') != 'NOT_CHECKED' or
                            exporter_hold.get('retained_after_context') is not True):
                        raise RuntimeError('experimental Paddle held-source '
                                           'exporter receipt missing or not approved')
                _record_step(number, label, 'done', 'exported',
                             ['saved ' + os.path.basename(path),
                              'onnx sha256 '
                              + import_provenance.get('onnx_sha256', '')[:16],
                              'toolchain '
                              + json.dumps(import_provenance.get('toolchain', {}))])
            tune_dir = None
            if data.get('tune'):
                tune_dir = os.path.join(build_directory, 'tuning-db')
            recheck_model('before TVM compilation')
            recheck_staged('before TVM compilation')
            paddle_recipe_kwargs = (
                {'approved_paddle_source': approved_paddle_closure}
                if approved_paddle_closure is not None else {})
            cpu_codegen_kwargs = (
                {'cpu_codegen': data['cpu_codegen']}
                if data.get('cpu_codegen') is not None else {})
            if (approved_paddle_closure is not None and
                    data.get('paddle_native_hold') is True):
                from pascalvision_convert import exported_onnx_hold
                exported_source = path
                exported_onnx_ticket = stage_locks.enter_context(
                    exported_onnx_hold.hold_private_exported_onnx(
                        exported_source,
                        stage_parent=os.path.dirname(os.path.abspath(build_directory)),
                        expected_sha256=import_provenance['onnx_sha256'],
                        expected_output_directory=build_directory))
                if (os.path.normcase(os.path.abspath(
                        exported_onnx_ticket.source_exported_onnx)) !=
                        os.path.normcase(os.path.abspath(exported_source)) or
                        exported_onnx_ticket.original_export_sha256 !=
                        import_provenance['onnx_sha256'] or
                        exported_onnx_ticket.staged_model_sha256 !=
                        import_provenance['onnx_sha256'] or
                        exported_onnx_ticket.status !=
                        'WIN32_EXPORTED_ONNX_COMPILER_HELD_NOT_RELEASE_GRADE' or
                        exported_onnx_ticket.verification_status != 'NOT_CHECKED' or
                        exported_onnx_ticket.release_grade_immutable is not False or
                        exported_onnx_ticket.retained_after_context is not True):
                    raise RuntimeError('Paddle exported ONNX native-held compiler '
                                       'ticket differs from approved exporter SHA')

                path = exported_onnx_ticket.compiler_input


                from pascalvision_convert import paddle_consumer_dictionary
                paddle_rec_dictionary_receipt = stage_locks.enter_context(
                    paddle_consumer_dictionary.hold_selected_rec_dictionary(
                        approved_paddle_closure, build_directory))
            report = pipeline.run(
                path, build_directory,
                target=data.get('target', 'cpu'),
                req_sm=data.get('arch') or None,
                req_os=data.get('target_os') or None,
                req_arch=data.get('target_cpu_arch') or None,
                req_driver_api=data.get('target_driver_api') or None,
                approved_build_plan=approved_build_plan,
                approved_source_closure=(approved_raw_freeze
                                         if selected_file_set else None),
                shape=data.get('shape') or None,
                tune_work_dir=tune_dir,
                trials_per_task=int(data.get('trials_per_task') or 0) or None,
                contract=contract,
                source=requested_source,
                license=(approved_onnx_source_pack['metadata']['license']
                         if approved_onnx_source_pack is not None else None),
                approved_onnx_source_pack=approved_onnx_source_pack,
                on_step=lambda step: _record(number, step),
                **paddle_recipe_kwargs, **cpu_codegen_kwargs)
            recheck_model('during TVM compilation/verification')
            recheck_paddle('during TVM compilation/verification')
            recheck_staged('during TVM compilation/verification')
            if exported_onnx_ticket is not None:


                stage_locks.close()


                verify_observed = (
                    report.sound is True and any(
                        getattr(step, 'name', None) == 'Comparison with the reference'
                        and getattr(step, 'state', None) in ('done', 'unchecked')
                        for step in getattr(report, 'steps', ())))
                import_provenance['compiler_window_held_exported_onnx'] = {
                    'status': 'POST_LOCK_AUDIT_PASSED_NOT_RELEASE_GRADE',
                    'exporter_sha256': exported_onnx_ticket.original_export_sha256,
                    'staged_model_sha256': exported_onnx_ticket.staged_model_sha256,
                    'bytes': exported_onnx_ticket.bytes,
                    'held_during_pipeline_run': True,
                    'pipeline_sound': report.sound is True,
                    'held_through_pipeline_verify': verify_observed,
                    'verification_status': 'NOT_CHECKED',
                    'release_grade_immutable': False,
                    'retained_after_context': True,
                }
                if paddle_rec_dictionary_receipt is not None:
                    if (paddle_rec_dictionary_receipt['status'] !=
                            'POST_WIN32_HELD_SELECTED_REC_DICTIONARY_NOT_MODEL_UAT' or
                            paddle_rec_dictionary_receipt['source_tree_sha256'] !=
                            approved_paddle_closure['tree_sha256'] or
                            paddle_rec_dictionary_receipt['sha256'] !=
                            approved_paddle_closure['dictionary']['sha256'] or
                            paddle_rec_dictionary_receipt['native_pascal_ctc'] !=
                            'NOT_RUN'):
                        raise RuntimeError('selected Rec dictionary was not '
                                           'held through TVM compiler/Verify')
                    import_provenance['compiler_window_held_rec_dictionary'] = (
                        paddle_rec_dictionary_receipt)
            if approved_paddle_closure is not None:
                report.summary['paddle_selected_source'] = {
                    'source_root': approved_paddle_closure['source_root'],
                    'tree_sha256': approved_paddle_closure['tree_sha256'],
                    'role': approved_paddle_closure['role'],
                    'dictionary': approved_paddle_closure['dictionary'],
                    'status': 'SCANNED_NOT_LOCKED',
                    'verification_status': 'NOT_CHECKED',
                    'release_grade_immutable': False,
                }
                if data.get('paddle_native_hold') is True:
                    held_receipt = import_provenance.get('compiler_held_source')
                    if (not isinstance(held_receipt, dict) or
                            not held_receipt.get('status', '').startswith(
                                'WIN32_PADDLE_SELECTED_HELD') or
                            held_receipt.get('source_tree_sha256') !=
                            approved_paddle_closure['tree_sha256'] or
                            held_receipt.get('release_grade_immutable') is not False):
                        raise RuntimeError('experimental Paddle held-source '
                                           'receipt missing or not approved')
                    report.summary['paddle_selected_source'][
                        'native_held_export'] = held_receipt
                    if (exported_onnx_ticket is None or
                            import_provenance.get(
                                'compiler_window_held_exported_onnx', {}).get(
                                    'held_during_pipeline_run') is not True):
                        raise RuntimeError('Paddle exported ONNX was not held '
                                           'through the selected TVM pipeline call')
                    report.summary['paddle_selected_source'][
                        'compiler_window_held_exported_onnx'] = import_provenance[
                            'compiler_window_held_exported_onnx']
                    if paddle_rec_dictionary_receipt is not None:
                        report.summary['paddle_selected_source'][
                            'compiler_window_held_rec_dictionary'] = (
                                paddle_rec_dictionary_receipt)
            if (atomic_complete_bundle and report.sound and
                    report.summary.get('verification_status') in
                    ('VERIFIED_LOCAL', 'NOT_CHECKED')):
                if selected_file_set:
                    model_entries = approved_raw_freeze['entries']
                else:
                    model_entries = [{
                        'path': os.path.basename(requested_source),
                        'kind': 'file',
                        'bytes': stage_ticket.bytes,
                        'sha256': stage_ticket.staged_file_sha256,
                    }]


                approved_bundle_sources = bundle_publication.stage_approved_sources(
                    build_directory, frozen_snapshot['snapshot_directory'], model_entries)
            if frozen_snapshot is not None:


                stage_locks.close()
                report.summary['source_freeze'] = {
                    'status': stage_ticket.status,
                    'tree_sha256': approved_raw_freeze['tree_sha256'],
                    'source_path': requested_source,
                    'lock_through_compilation': True,
                    'release_grade_immutable': False,
                    'external_dependency_closure': (
                        'ONNX_SELECTED_EXTERNAL_DATA_V1' if selected_file_set
                        else 'NARROW_RAW_GRAPH_ONLY'),
                }
                if selected_file_set:


                    report.summary['source_freeze']['selected_closure'] = {
                        'schema': approved_raw_freeze['closure_schema'],
                        'version': approved_raw_freeze['source_closure_version'],
                        'source_model': approved_raw_freeze['source_model'],
                        'model_rel': approved_raw_freeze['model_rel'],
                        'tree_sha256': approved_raw_freeze['tree_sha256'],
                        'dependencies': approved_raw_freeze['dependencies'],
                        'entries': approved_raw_freeze['entries'],
                        'references': approved_raw_freeze['references'],
                        'total_bytes': approved_raw_freeze['total_bytes'],
                    }
                    if report.sound:


                        saved_recipe = read_recipe(
                            os.path.join(build_directory, 'recipe.json'))
                        if (saved_recipe.get('source_closure')
                                != approved_raw_freeze or
                                saved_recipe.get('source') != requested_source):
                            raise RuntimeError('Pipeline recipe lost exact '
                                               'Preview-approved external dependencies')
                        sealed = {
                            'schema': 'crosspascal.onnx-selected-closure-receipt',
                            'schema_version': 1,
                            'source_closure': approved_raw_freeze,
                            'source_path': requested_source,
                            'stage_lock_status': stage_ticket.status,
                            'status': 'POST_LOCK_AUDIT_PASSED_NOT_RELEASE_GRADE',
                            'held_through_pipeline_verify': True,
                            'release_grade_immutable': False,
                        }
                        seal = report.step('Recording selected ONNX dependency closure')
                        with open(os.path.join(build_directory, 'source-closure.json'),
                                  'x', encoding='utf-8', newline='\n') as handle:
                            json.dump(sealed, handle, ensure_ascii=False,
                                      indent=2, sort_keys=True)
                            handle.write('\n')
                            handle.flush()
                            os.fsync(handle.fileno())
                        seal.say('approved ONNX+sidecars source tree sha256 %s'
                                 % approved_raw_freeze['tree_sha256'])
                        seal.say('native compiler-held file-set post-audit PASS; '
                                 'still NOT a release-grade immutable model')
                        seal.done()
                        report.summary['source_closure_receipt'] = {
                            'status': sealed['status'],
                            'tree_sha256': approved_raw_freeze['tree_sha256'],
                        }
            if import_provenance is not None:
                with open(os.path.join(build_directory, 'import-provenance.json'), 'w',
                          encoding='utf-8', newline='\n') as handle:
                    json.dump(import_provenance, handle,
                              ensure_ascii=False, indent=2)
            if os.path.isdir(build_directory):
                with open(os.path.join(build_directory, 'report.txt'), 'w',
                          encoding='utf-8', newline='\n') as handle:
                    handle.write(report.as_text() + '\n')
            recheck_paddle('before Build publication/READY')
            committed_inventory = None
            if atomic_complete_bundle:
                if (not report.sound or report.summary.get('verification_status')
                        not in ('VERIFIED_LOCAL', 'NOT_CHECKED')):
                    raise bundle_publication.UnsafePublication(
                        'compiler/independent Verify did not accept this bundle; '
                        'private partial stage stays unpublished')
                selected_receipt = (report.summary.get('source_closure_receipt')
                                    or {}).get('tree_sha256')
                selected_details = (report.summary.get('source_freeze')
                                    or {}).get('selected_closure')
                if selected_file_set and (
                        not isinstance(selected_receipt, str) or
                        not isinstance(selected_details, dict)):
                    raise bundle_publication.UnsafePublication(
                        'selected ONNX external-data post-lock receipt is missing')
                expected_model_sha256 = (
                    stage_ticket.staged_model_sha256 if selected_file_set else
                    stage_ticket.staged_file_sha256)
                committed_inventory, receipt = bundle_publication.commit_verified_stage(
                    build_directory, published_path, report, data.get('target', 'cpu'),
                    requested_source, expected_model_sha256=expected_model_sha256,
                    expected_input_manifest=approved_native_io,
                    expected_source_entries=approved_bundle_sources,
                    closure_sha=selected_receipt,
                    closure=selected_details,
                    **({'on_validated_precommit': remember_validated_precommit}
                       if retained_ledger_run is not None else {}))
                if retained_ledger_run is not None:


                    retained_ledger_publication_observation = (
                        retained_ledger.record_selected_publication_observation(
                            evidence_directory=retained_ledger_run,
                            output_final=published_path,
                            original_stage=private_atomic_stage,
                            job_nonce=retained_ledger_job_nonce,
                            expected_private_stage_file_id=private_atomic_stage_id,
                            committed_tree_sha256=committed_inventory.tree_sha256))
                report.summary['module'] = os.path.join(
                    receipt.destination, os.path.basename(report.summary['module']))
                report.summary['directory'] = receipt.destination
                report.summary['atomic_bundle'] = {
                    'status': receipt.status,
                    'tree_sha256': receipt.tree_sha256,
                    'member_count': receipt.member_count,
                    'release_grade_immutable': False,
                    'power_loss_durable': False,
                }
                build_directory = receipt.destination
            if retained_ledger_run is not None:


                report.summary['retained_dpapi_ledger'] = ledger_observation()
            candidate_download_ready = bool(
                report.sound and report.summary.get('verification_status') in
                ('VERIFIED_LOCAL', 'NOT_CHECKED') and os.path.isdir(build_directory))


            deferred_lease_download_ready = bool(
                candidate_download_ready and retained_ledger_run is not None)
            with LOCK:


                steps = list(JOBS[number]['steps'])
                recorded = {entry['name'] for entry in steps}
                for step in report.steps:
                    if step.name not in recorded:
                        steps.append(step.as_dict())
                JOBS[number].update({


                    'finished': not deferred_lease_download_ready,
                    'sound': report.sound,


                    'download_ready': bool(
                        candidate_download_ready and
                        retained_ledger_run is None),
                    'bundle_root': os.path.abspath(build_directory),
                    'steps': steps,
                    'summary': report.summary,
                })
                if approved_onnx_source_pack is not None:
                    JOBS[number]['_onnx_source_pack'] = approved_onnx_source_pack
                if committed_inventory is not None:


                    JOBS[number]['_atomic_inventory'] = committed_inventory
        except Exception as trouble:
            try:
                if exported_onnx_ticket is not None:


                    stage_locks.__exit__(type(trouble), trouble,
                                         trouble.__traceback__)
                else:
                    stage_locks.close()
            except Exception as lock_problem:
                trouble = RuntimeError('%s; Win32 stage release/audit also failed: %s'
                                       % (trouble, lock_problem))


            retained = None
            retained_unproven = None
            if (atomic_complete_bundle or native_family_atomic) and private_atomic_stage:
                try:
                    raw = bundle_publication.bundle_atomic
                    candidate = raw._local_absolute(private_atomic_stage)
                    parent = os.path.normcase(os.path.dirname(
                        os.path.abspath(output_directory)))
                    if (os.path.normcase(os.path.dirname(candidate)) == parent
                            and raw.STAGE_NAME.fullmatch(os.path.basename(candidate))
                            and os.path.lexists(candidate)):
                        current = raw._guard_stat(candidate, directory=True)
                        if (private_atomic_stage_id is not None and
                                (current.st_dev, current.st_ino) ==
                                private_atomic_stage_id):
                            retained = {
                                'path': candidate,
                                'status': 'PRESENT_FOR_OWNER_INSPECTION_NOT_VERIFIED',
                                'cleanup_performed': False,
                            }
                        else:
                            retained_unproven = {
                                'candidate_path': candidate,
                                'status': 'RETENTION_PATH_IDENTITY_UNPROVEN',
                                'cleanup_performed': False,
                            }
                except (OSError, ValueError, TypeError):
                    retained = None


            retained_source = None
            retained_source_unproven = None
            if atomic_complete_bundle and private_source_snapshot_path:
                try:
                    candidate = os.path.abspath(private_source_snapshot_path)
                    final = os.path.abspath(output_directory)
                    prefix = os.path.basename(final) + '.source-freeze-'
                    leaf = os.path.basename(candidate)
                    if (os.path.normcase(os.path.dirname(candidate)) ==
                            os.path.normcase(os.path.dirname(final)) and
                            leaf.startswith(prefix) and
                            re.fullmatch(r'[0-9a-f]{24}', leaf[len(prefix):]) and
                            os.path.lexists(candidate)):
                        source_freeze._safe_ancestors(
                            source_freeze.Path(candidate))
                        current = bundle_publication.bundle_atomic._guard_stat(
                            candidate, directory=True)
                        if (private_source_snapshot_id is not None and
                                (current.st_dev, current.st_ino) ==
                                private_source_snapshot_id):
                            retained_source = {
                                'path': candidate,
                                'status': 'PRESENT_FOR_OWNER_INSPECTION_NOT_VERIFIED',
                                'cleanup_performed': False,
                            }
                        else:
                            retained_source_unproven = {
                                'candidate_path': candidate,
                                'status': 'SOURCE_SNAPSHOT_PATH_IDENTITY_UNPROVEN',
                                'cleanup_performed': False,
                            }
                except (OSError, ValueError, TypeError):
                    retained_source = None
            with LOCK:
                JOBS[number]['finished'] = True
                JOBS[number]['sound'] = False
                if retained_ledger_run is not None:
                    JOBS[number]['summary'][
                        'retained_dpapi_ledger'] = ledger_observation()
                if retained is not None:
                    JOBS[number]['summary']['retained_private_stage'] = retained
                if retained_unproven is not None:
                    JOBS[number]['summary']['retained_private_stage_warning'] = retained_unproven
                if retained_source is not None:
                    JOBS[number]['summary']['retained_source_snapshot'] = retained_source
                if retained_source_unproven is not None:
                    JOBS[number]['summary'][
                        'retained_source_snapshot_warning'] = retained_source_unproven
                JOBS[number]['steps'].append({
                    'name': 'Run interrupted',
                    'state': 'refused',
                    'state_name': 'refused',
                    'lines': ([str(trouble)] + (
                        ['Retained private stage for owner inspection: ' +
                         retained['path'] + '; no automatic cleanup performed']
                        if retained else []) + (
                        ['Private stage candidate identity is UNPROVEN: ' +
                         retained_unproven['candidate_path'] +
                         '; do not attribute or cleanup']
                        if retained_unproven else []) + (
                        ['Retained RAW source snapshot for owner inspection: ' +
                         retained_source['path'] +
                         '; no automatic cleanup performed']
                        if retained_source else []) + (
                        ['Source snapshot candidate identity is UNPROVEN: ' +
                         retained_source_unproven['candidate_path'] +
                         '; do not attribute or cleanup']
                        if retained_source_unproven else [])),
                    'numbers': {},
                })
        finally:


            lease_finalized_successfully = False
            try:
                try:
                    stage_locks.close()
                finally:
                    lease_stack.close()
                lease_finalized_successfully = True
            except Exception as lease_problem:


                with LOCK:
                    JOBS[number]['sound'] = False
                    JOBS[number]['download_ready'] = False
                    JOBS[number]['finished'] = True
                    JOBS[number]['steps'].append({
                        'name': 'Worker native resource finalization refused',
                        'state': 'refused',
                        'state_name': 'refused',
                        'lines': [str(lease_problem) + '; no cleanup performed'],
                        'numbers': {},
                    })
            finally:


                with LOCK:
                    if (atomic_complete_bundle or native_family_atomic) and output_directory is not None:
                        ATOMIC_PENDING_OUTPUTS.discard(
                            os.path.normcase(os.path.abspath(output_directory)))
                    if (lease_finalized_successfully and
                            deferred_lease_download_ready and
                            JOBS[number]['sound'] and
                            committed_inventory is not None):
                        JOBS[number]['finished'] = True
                        JOBS[number]['download_ready'] = True
                    elif deferred_lease_download_ready:


                        JOBS[number]['finished'] = True
                        JOBS[number]['sound'] = False
                        JOBS[number]['download_ready'] = False

    try:
        threading.Thread(target=work, daemon=True).start()
    except Exception as trouble:


        with LOCK:
            JOBS[number].update({
                'finished': True,
                'sound': False,
                'download_ready': False,
            })
            JOBS[number]['steps'].append({
                'name': 'Worker startup refused',
                'state': 'refused', 'state_name': 'refused',
                'lines': [str(trouble)], 'numbers': {},
            })
        raise
    return number


def _record(number, step):
    """Append a step as it lands, so the page shows progress and not just a result."""
    with LOCK:
        job = JOBS.get(number)
        if job is None:
            return
        job['steps'] = [s for s in job['steps'] if s['name'] != step.name]
        job['steps'].append(step.as_dict())
    diagnostic = getattr(step, '_build_failure_diagnostic', None)
    if diagnostic is None:
        return

    try:
        header = {'job': number, 'step': step.name,
                  **{key: value for key, value in diagnostic.items()
                     if key != 'traceback'}}
        message = ('\nCrossPascal build failure\n'
                   + json.dumps(header, ensure_ascii=False, sort_keys=True) + '\n'
                   + diagnostic.get('traceback', '') + '\n')
        if sys.stderr.write(message) != len(message):
            raise OSError('Incomplete diagnostic log write')
        sys.stderr.flush()
    except Exception as log_error:

        with LOCK:
            if JOBS.get(number) is job:
                job['diagnostic_log_error'] = {
                    'step': step.name,
                    'exception_type': '%s.%s' % (type(log_error).__module__,
                                                type(log_error).__qualname__),
                }


def _record_step(number, name, state, state_name, lines):
    """Append a synthetic step - the import (framework -> ONNX) step - so the page
    shows it like any core step. Same replace-by-name rule as _record."""
    with LOCK:
        job = JOBS.get(number)
        if job is None:
            return
        job['steps'] = [s for s in job['steps'] if s['name'] != name]
        job['steps'].append({'name': name, 'state': state,
                             'state_name': state_name, 'lines': lines,
                             'numbers': {}})


class _JSONPostRejected(ValueError):
    """Client-side HTTP refusal before a JSON route can run any build action."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Responder(BaseHTTPRequestHandler):
    """Serve the converter page and its loopback API."""

    server_version = 'PascalVisionMaster'
    JSON_POST_MAX_BYTES = 1024 * 1024

    def log_message(self, pattern, *args):
        pass

    def _send(self, body, content_type='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        raw_length = self.headers.get('Content-Length')
        if raw_length is None:
            return {}
        if (type(raw_length) is not str or not raw_length or
                not raw_length.isascii() or not raw_length.isdecimal()):
            raise _JSONPostRejected(400, 'Invalid JSON Content-Length')


        if len(raw_length) > 16:
            raise _JSONPostRejected(413, 'JSON request exceeds byte limit')
        length = int(raw_length)
        if length > self.JSON_POST_MAX_BYTES:
            raise _JSONPostRejected(413, 'JSON request exceeds byte limit')
        if not length:
            return {}

        connection = getattr(self, 'connection', None)
        previous_timeout = None
        if connection is not None:
            previous_timeout = connection.gettimeout()
            connection.settimeout(30.0)
        try:
            try:
                payload = self.rfile.read(length)
            except (TimeoutError, OSError) as exc:
                raise _JSONPostRejected(408, 'JSON body read timed out or failed') from exc
        finally:
            if connection is not None:
                connection.settimeout(previous_timeout)
        if len(payload) != length:
            raise _JSONPostRejected(400, 'Incomplete JSON request body')

        def unique_pairs(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise _JSONPostRejected(400, 'Duplicate JSON request key')
                result[name] = value
            return result

        def reject_nonfinite(value):
            raise _JSONPostRejected(400, 'Nonfinite JSON request token')

        try:
            return json.loads(payload.decode('utf-8'),
                              object_pairs_hook=unique_pairs,
                              parse_constant=reject_nonfinite)
        except _JSONPostRejected:
            raise
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise _JSONPostRejected(400, 'Malformed JSON request body') from exc

    def do_GET(self):
        global ACTIVE_REQUESTS
        expected_host = '127.0.0.1:%d' % self.server.server_port
        if self.headers.get_all('Host') != [expected_host]:
            self.send_error(403, 'Use the local converter address')
            return
        with LOCK:
            closing = SHUTTING_DOWN
            if not closing:
                ACTIVE_REQUESTS += 1
        if closing:
            self.send_error(503, 'Converter is closing')
            return
        try:
            self._handle_get()
        finally:
            with LOCK:
                ACTIVE_REQUESTS -= 1

    def _handle_get(self):
        route = urlparse(self.path)
        if route.path == '/api/identity':
            self._send(json.dumps({
                'app': 'CrossPascal Convert', 'pid': os.getpid(),
                'port': self.server.server_port,
                'instance': self.server.instance_id}))
            return
        if route.path == '/api/model-download-progress':
            job = (parse_qs(route.query).get('job') or [''])[0]
            with LOCK:
                result = dict(MODEL_DOWNLOADS.get(job) or {'trouble': 'Unknown download'})
            self._send(json.dumps(result))
            return
        if route.path == '/api/ocr-draft-download':
            job_id = (parse_qs(route.query).get('job') or [''])[0]
            with LOCK:
                selected = OCR_DRAFT_JOBS.get(job_id)
            if selected is None:
                self.send_error(404, 'Unknown OCR DRAFT')
                return
            spool = None
            try:
                entries = bundle_download.collect_bundle_files(
                    selected['directory'],
                    expected_atomic_inventory=selected['inventory'])
                spool = tempfile.TemporaryFile(mode='w+b')
                bundle_download.stream_bundle(
                    spool, entries, job=job_id,
                    verification_status='NOT_CHECKED',
                    expected_atomic_inventory=selected['inventory'])
                length = spool.tell()
                spool.seek(0)
            except Exception:
                if spool is not None:
                    spool.close()
                with LOCK:
                    OCR_DRAFT_JOBS.pop(job_id, None)
                self.send_error(409, 'OCR DRAFT ZIP integrity refused')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition',
                             'attachment; filename="ocr-composite-DRAFT-NOT_CHECKED.zip"')
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                shutil.copyfileobj(spool, self.wfile, 1024 * 1024)
            finally:
                spool.close()
            return
        if route.path == '/api/download':
            self._download_job(route)
            return
        if route.path in ('/', '/index.html'):
            with open(PAGE, 'rb') as handle:
                html = handle.read().replace(
                    b'__CROSSPASCAL_SESSION__', SESSION_TOKEN.encode('ascii'))
                self._send(html, 'text/html; charset=utf-8')
            return
        if route.path == '/api/environment':
            self._send(json.dumps(
                _environment_report().as_dict(),
                ensure_ascii=False))
            return
        if route.path == '/api/progress':
            number = (parse_qs(route.query).get('job') or [''])[0]
            with LOCK:
                state = dict(JOBS.get(number, {'finished': True, 'sound': False,
                                               'steps': [], 'summary': {}}))


            state.pop('_atomic_inventory', None)
            self._send(json.dumps(state, ensure_ascii=False))
            return
        self.send_error(404)

    def _download_job(self, route):
        """Serve only a finished, successful job-owned artifact tree.

        The request supplies a job ID, never a filesystem path. Validate all
        members and finish the ZIP before sending HTTP 200, including legacy
        jobs without an atomic inventory. Late integrity failures return 409
        without a partial 200. Atomic jobs additionally verify every byte
        against their committed SHA inventory; legacy jobs gain no immutable
        snapshot or new verification claim. An unverified compile stays
        NOT_CHECKED within the downloaded ZIP.
        """
        values = parse_qs(route.query).get('job') or []
        job_id = values[0] if len(values) == 1 else ''
        if not job_id.isascii() or not job_id.isdecimal():
            self.send_error(404, 'Unknown build job')
            return
        with LOCK:
            state = dict(JOBS.get(job_id) or {})
        if not (state.get('finished') and state.get('sound')
                and state.get('download_ready')):
            self.send_error(404, 'No completed downloadable bundle for job')
            return
        verification = (state.get('summary') or {}).get('verification_status')
        if verification not in ('VERIFIED_LOCAL', 'NOT_CHECKED'):
            self.send_error(409, 'Build is not exportable')
            return
        atomic_info = (state.get('summary') or {}).get('atomic_bundle')
        atomic_inventory = state.get('_atomic_inventory')
        if ((atomic_info is None) != (atomic_inventory is None) or
                (atomic_info is not None and (
                    not isinstance(atomic_info, dict) or
                    not isinstance(atomic_inventory,
                                   bundle_publication.bundle_atomic.Inventory) or
                    atomic_info.get('tree_sha256') != atomic_inventory.tree_sha256 or
                    atomic_info.get('status') != bundle_publication.bundle_atomic.STATUS or
                    not os.path.isabs(state.get('bundle_root', ''))))):
            self.send_error(409, 'No trusted committed inventory for job')
            return
        sealed_zip = None
        try:
            approved_closure = ((state.get('summary') or {}).get(
                'source_closure_receipt') or {}).get('tree_sha256')
            approved_details = ((state.get('summary') or {}).get(
                'source_freeze') or {}).get('selected_closure')
            selected_paddle = ((state.get('summary') or {}).get(
                'paddle_selected_source') or {})
            rec_dictionary = (selected_paddle.get(
                'compiler_window_held_rec_dictionary')
                if isinstance(selected_paddle, dict) else None)
            files = bundle_download.collect_bundle_files(
                state.get('bundle_root'),
                expected_source_closure_sha256=approved_closure,
                expected_source_closure=approved_details,
                expected_atomic_inventory=atomic_inventory,
                expected_onnx_source_pack=state.get('_onnx_source_pack'),
                **({'expected_paddle_rec_dictionary': rec_dictionary,
                    'expected_paddle_source_root': selected_paddle.get('source_root'),
                    'expected_paddle_build_plan': (state.get('summary') or {}).get(
                        'build_plan')}
                   if rec_dictionary is not None else {}))


            sealed_zip = tempfile.TemporaryFile(mode='w+b')
            bundle_download.stream_bundle(
                sealed_zip, files, job=job_id,
                verification_status=verification,
                expected_atomic_inventory=atomic_inventory)
            length = sealed_zip.tell()
            sealed_zip.seek(0)
        except bundle_download.UnsafeBundle as exc:
            if sealed_zip is not None:
                try:
                    sealed_zip.close()
                except OSError:
                    pass
            if (atomic_inventory is not None or
                    isinstance(exc, bundle_download.UnsafeBundleIntegrity)):
                with LOCK:
                    recorded = JOBS.get(job_id)
                    if (recorded is not None and
                            recorded.get('bundle_root') == state.get('bundle_root') and
                            recorded.get('_atomic_inventory') is atomic_inventory):


                        recorded['download_ready'] = False
                        recorded['summary']['bundle_export_status'] = (
                            'REFUSED_COMMITTED_INVENTORY_OR_ZIP_INTEGRITY'
                            if atomic_inventory is not None else
                            'REFUSED_LEGACY_ZIP_INTEGRITY')
                        recorded['summary']['bundle_export_error_type'] = (
                            type(exc).__name__)
            self.send_error(409, 'Unsafe or incomplete bundle')
            return
        except Exception:


            if sealed_zip is not None:
                try:
                    sealed_zip.close()
                except OSError:
                    pass
            self.send_error(503, 'Temporary ZIP export unavailable; retry')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/zip')
        self.send_header('Content-Disposition',
                         'attachment; filename="pascalvision-job-%s.zip"' % job_id)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(length))
        self.end_headers()
        try:
            shutil.copyfileobj(sealed_zip, self.wfile, 1024 * 1024)
        except (BrokenPipeError, ConnectionResetError):


            pass
        finally:
            sealed_zip.close()

    def do_POST(self):
        global ACTIVE_REQUESTS, SHUTTING_DOWN
        expected_host = '127.0.0.1:%d' % self.server.server_port
        if (self.headers.get('Host') != expected_host or
                self.headers.get('Origin') != 'http://' + expected_host or
                not secrets.compare_digest(
                    self.headers.get('X-CrossPascal-Session', ''), SESSION_TOKEN)):
            self.send_error(403, 'Use the local converter page for this action')
            return
        route = urlparse(self.path)
        if route.path == '/api/quit':
            with LOCK:
                busy = (ACTIVE_REQUESTS != 0 or
                        any(not job.get('finished') for job in JOBS.values()) or
                        any(not job.get('finished') for job in MODEL_DOWNLOADS.values()))
                if not busy:
                    SHUTTING_DOWN = True
            if busy:
                self._send(json.dumps({'trouble':
                    'An operation is still running. Wait for it to finish before closing the converter.'}))
                return
            self._send(json.dumps({'status': 'CLOSING'}))
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        with LOCK:
            closing = SHUTTING_DOWN
            if not closing:
                ACTIVE_REQUESTS += 1
        if closing:
            self.send_error(503, 'Converter is closing')
            return
        try:
            self._handle_post()
        finally:
            with LOCK:
                ACTIVE_REQUESTS -= 1

    def _handle_post(self):
        route = urlparse(self.path)


        json_routes = (
            '/api/omdet-candidate-preview', '/api/omdet-candidate-build',
            '/api/ocr-draft-evidence', '/api/ocr-draft-preview',
            '/api/ocr-draft-build', '/api/recognise', '/api/preview',
            '/api/build', '/api/samples', '/api/toolchain', '/api/model-download')
        if route.path in json_routes:
            try:
                body = self._read_body()
            except _JSONPostRejected as exc:
                self.send_error(exc.status, str(exc))
                return
        if route.path == '/api/model-download':
            if type(body) is not dict or body.get('model_key') != 'qwen2.5-0.5b':
                self._send(json.dumps({'trouble': 'Choose a model from the pinned catalog'}))
                return
            with LOCK:
                existing = next((key for key, value in MODEL_DOWNLOADS.items()
                                 if not value.get('finished')), None)
                if existing is None:
                    number = secrets.token_hex(12)
                    MODEL_DOWNLOADS[number] = {'finished': False, 'status': 'DOWNLOADING'}
            if existing is not None:
                self._send(json.dumps({'job': existing}))
                return
            def work():
                def progress(value):
                    with LOCK:
                        MODEL_DOWNLOADS[number].update(value)
                try:
                    result = model_sources.download('qwen2.5-0.5b', progress=progress)
                except Exception as exc:
                    result = {'status': 'ERROR', 'trouble': str(exc)}
                with LOCK:
                    MODEL_DOWNLOADS[number].update(result)
                    MODEL_DOWNLOADS[number]['finished'] = True
            try:
                threading.Thread(target=work, daemon=True).start()
            except Exception as exc:
                with LOCK:
                    MODEL_DOWNLOADS[number].update(
                        {'finished': True, 'status': 'ERROR', 'trouble': str(exc)})
                self._send(json.dumps({'trouble': str(exc)}))
                return
            self._send(json.dumps({'job': number}))
            return
        if route.path == '/api/toolchain':
            expected_host = '127.0.0.1:%d' % self.server.server_port
            if (self.headers.get('Host') != expected_host or
                    self.headers.get('Origin') != 'http://' + expected_host or
                    self.headers.get_content_type() != 'application/json'):
                self.send_error(403, 'Use the local wizard to select a toolchain')
                return
            try:
                selected = llm_build_profile.python_snapshot(body.get('python'))
                self._send(json.dumps(
                    _environment_report(selected['path']).as_dict(),
                    ensure_ascii=False))
            except (OSError, ValueError, TypeError) as exc:
                self._send(json.dumps({'trouble': str(exc)}))
            return
        if route.path == '/api/omdet-candidate-preview':
            self._send(json.dumps(_omdet_candidate_preview(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/omdet-candidate-build':
            self._send(json.dumps(_omdet_candidate_build(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/ocr-draft-evidence':
            self._send(json.dumps(_ocr_draft_evidence(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/ocr-draft-preview':
            self._send(json.dumps(_ocr_draft_preview(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/ocr-draft-build':
            self._send(json.dumps(_ocr_draft_build(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/recognise':
            self._send(json.dumps(self._recognise(body), ensure_ascii=False))
            return
        if route.path == '/api/preview':
            self._send(json.dumps(_issue_preview(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/build':
            self._send(json.dumps(_build_from_preview(body),
                                  ensure_ascii=False))
            return
        if route.path == '/api/samples':
            self._send(json.dumps(samples.check_installation().as_dict(),
                                  ensure_ascii=False))
            return
        if route.path == '/api/upload':
            self._send(json.dumps(self._upload(route), ensure_ascii=False))
            return
        self.send_error(404)

    def _upload(self, route):
        """Save a posted file into uploads/ without replacing another source.

        The page cannot pass the picked file's path - the browser withholds it -
        so it posts the bytes here instead. The name rides in the query and is
        made safe (see _safe_upload_name); the body is the raw file, read by
        Content-Length a megabyte at a time straight to disk, so a large model is
        never held whole in memory. Re-uploading the same name must not silently
        change the bytes of an earlier recognised model: a numbered filename and
        exclusive create protect the old source even when two uploads race.

        Only fully received, independently reread bytes acquire a normal model
        filename through an exclusive hardlink. A short request leaves its own
        uniquely named .part evidence, NOT a normal recognisable model. Existing
        model filenames are never overwritten, even in an upload race.
        """
        try:
            raw_name = (parse_qs(route.query).get('name') or [''])[0]
            name = _safe_upload_name(raw_name)
            remaining = int(self.headers.get('Content-Length') or 0)
            if remaining <= 0:
                return {'trouble': 'empty upload: no file content received'}
            uploads = str(app_paths.data_path('uploads'))
            os.makedirs(uploads, exist_ok=True)
            if (os.path.islink(uploads) or
                    getattr(os.path, 'isjunction', lambda _: False)(uploads)):
                return {'trouble': 'upload folder redirects through a link/junction'}


            available = shutil.disk_usage(uploads).free
            reserve = max(256 * 1048576, available // 50)
            if remaining > max(0, available - reserve):
                return {'trouble': 'upload exceeds disk space remaining after '
                        'safety reserve; no user file was created'}
            stem, extension = os.path.splitext(name)
            total = remaining
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                    mode='wb', prefix='.pascalvision-upload-', suffix='.part',
                    dir=uploads, delete=False) as staged:
                temporary = staged.name
                while remaining > 0:
                    block = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not block:
                        break
                    staged.write(block)
                    digest.update(block)
                    remaining -= len(block)
            if remaining:
                return {'trouble': 'incomplete upload: received %d of %d bytes; '
                        'no model was accepted, temporary .part evidence retained'
                        % (total - remaining, total)}


            reread = hashlib.sha256()
            count = 0
            with open(temporary, 'rb') as handle:
                for block in iter(lambda: handle.read(UPLOAD_CHUNK), b''):
                    reread.update(block)
                    count += len(block)
            if count != total or reread.digest() != digest.digest():
                return {'trouble': 'uploaded temporary bytes changed during '
                        'SHA-256 verification; no model was published'}
            for index in range(1, 10001):
                unique = (name if index == 1 else
                          '%s-%d%s' % (stem, index, extension))
                target = os.path.join(uploads, unique)


                if os.path.lexists(target):
                    continue
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    continue


                os.unlink(temporary)
                return {'path': os.path.abspath(target), 'name': unique,
                        'size': total, 'sha256': reread.hexdigest()}
            return {'trouble': 'too many uploads named %s; cannot select a '
                    'new path without replacing an existing source' % name}
        except Exception as trouble:
            return {'trouble': str(trouble)}

    def _recognise(self, data):
        try:
            path = onnx_source_pack.normalize_selection(data.get('path'))
        except ValueError as trouble:
            return {'trouble': str(trouble)}
        if not path:
            return {'trouble': 'no path given'}
        try:
            if path.lower().startswith(('http://', 'https://')):
                downloads = str(app_paths.data_path('downloads'))
                os.makedirs(downloads, exist_ok=True)


                per_request = tempfile.mkdtemp(prefix='model-', dir=downloads)
                path = pipeline.download(path, per_request)
            if not os.path.exists(path):
                return {'trouble': 'no such path: %s' % path}
            pack = onnx_source_pack.discover(path)
            recognised = (detect.Recognised(
                detect.ONNX, 'ONNX graph with a verified source-pack input binding')
                if pack is not None else detect.recognise(path))
            answer = recognised.as_dict()
            answer['path'] = path
            if pack is not None:
                answer['onnx_source_pack'] = onnx_source_pack.display(pack)
                answer['raw_source_protection'] = {
                    'eligible': False, 'recommended': False,
                    'status': 'NOT_APPLICABLE', 'release_grade_immutable': False,
                    'reason': 'This source pack uses its reviewed vision adapter; '
                              'unclassified RAW source staging is not selected',
                }
            else:
                answer['raw_source_protection'] = _raw_source_protection_offer(
                    path, recognised)


            a_file = os.path.isfile(path)
            answer['size'] = ('%.1f' % (os.path.getsize(path) / 1048576.0)) if a_file else '-'
            answer['first_bytes'] = detect.first_bytes(path) if a_file else 'a directory'
            if pack is not None:


                conversion = pack['metadata']['conversion']
                answer['inputs'] = [{
                    'name': conversion['input_name'],
                    'shape': ','.join(str(x) for x in conversion['input_shape']),
                    'free': False,
                }]
            else:
                answer['inputs'] = (_model_inputs(path, recognised.kind)
                                    if recognised.convertible else [])


            if not recognised.convertible:
                adapter = import_adapters.find_adapter(path, recognised)
                if adapter is not None:
                    answer['exportable'] = True
                    answer['export_adapter'] = adapter.name


                    answer['export_kind'] = getattr(adapter, 'kind', 'export')
                    if answer['export_kind'] == 'full_build':
                        answer['model_key'] = adapter._key_for(path)
            return answer
        except detect.OnnxAdmissionError as trouble:
            return {'trouble': 'ONNX inspection deferred by the capacity guard: '
                    + str(trouble), 'status': 'DEFERRED', 'path': path}
        except MemoryError:
            return {'trouble': 'Model inspection ran out of memory; validity '
                    'is not established. Free memory before trying Recognise again.',
                    'status': 'DEFERRED', 'path': path}
        except Exception as trouble:
            return {'trouble': str(trouble)}


def serve(port=DEFAULT_PORT, open_browser=True):
    """Start the responder and return it. If the port is taken, take the next."""
    app_paths.prepare()
    for attempt in range(port, port + 20):
        try:
            responder = ThreadingHTTPServer((ADDRESS, attempt), Responder)
        except OSError:
            continue
        url = 'http://%s:%d/' % (ADDRESS, attempt)
        responder.instance_id = secrets.token_hex(16)
        print('Model conversion wizard: %s' % url)
        print('Stop with Ctrl+C.')
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        return responder
    raise RuntimeError('no free port in %d..%d' % (port, port + 20))


if __name__ == '__main__':
    server = serve()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('')
        print('Wizard stopped.')
