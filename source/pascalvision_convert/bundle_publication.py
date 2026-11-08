# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Publish a bounded, complete RAW kit on NTFS after validation.

The worker compiles into a private sibling directory and calls this module only
AFTER compiler/independent Verify and every selected-source Win32 post-lock audit.
There is no persisted READY marker: an interrupted process cannot resurrect an
unreviewed directory. A successful receipt proves a namespace commit, NOT power
loss durability, release-grade immutable bytes or end-device acceptance.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from . import bundle_atomic, bundle_download, hf_held_stage, runtime, source_freeze
from .recipe_file import read_recipe


class UnsafePublication(ValueError):
    """The caller must not expose a ZIP or report download_ready after refusal."""

MAX_RETAINED_ATOMIC_STAGES_PER_PARENT = 16
MAX_ATOMIC_PARENT_SCAN_ENTRIES = 16384
MIN_ATOMIC_SOURCE_RESERVE = bundle_atomic.MAX_BYTES + 256 * 1024 * 1024
_STAGE_ADMISSION_LOCK = threading.Lock()
_RETAINED_SOURCE_SNAPSHOT = re.compile(r'.+\.source-freeze-[0-9a-f]{24}\Z')


@contextmanager
def _cross_process_atomic_admission(parent):
    """Win32 named mutex: serialize per-parent scan+mkdir across processes.

    No lockfile is created/reclaimed, and a crashed owner leaves no disk file
    requiring unsafe deletion. An abandoned mutex is usable ONLY after the
    admission scan executes again. Future compiler growth is not reserved.
    """
    if os.name != 'nt':
        raise UnsafePublication('cross-process atomic stage admission requires Win32')
    import ctypes
    from ctypes import wintypes
    import hashlib

    info = bundle_atomic._guard_stat(parent, directory=True)
    key = '%s|%d|%d' % (
        os.path.normcase(os.path.realpath(parent)), info.st_dev, info.st_ino)

    name = ('Global\\CrossPascal-I03-AtomicStage-' +
            hashlib.sha256(key.encode('utf-8')).hexdigest())
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL,
                                 wintypes.LPCWSTR)
    api.CreateMutexW.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    api.ReleaseMutex.restype = wintypes.BOOL
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateMutexW(None, False, name)
    if not handle:
        raise UnsafePublication('could not acquire Win32 atomic stage admission mutex')
    acquired = False
    try:
        outcome = api.WaitForSingleObject(handle, 10000)
        if outcome not in (0, 0x80):
            raise UnsafePublication(
                'Win32 atomic stage admission mutex busy/unavailable; no stage created')
        acquired = True
        yield
    finally:
        try:
            if acquired and not api.ReleaseMutex(handle):
                raise UnsafePublication(
                    'Win32 atomic stage admission mutex release failed')
        finally:


            api.CloseHandle(handle)


def _admit_private_atomic_stage(parent, *, capacity_profile='raw'):
    """Reserve BOTH output and future source-snapshot slots for one RAW Build.

    Every admitted atomic RAW Build creates a private output stage and may
    later retain a separate source-freeze sibling. Count an existing output
    stage as TWO occupied slots, including its not-yet-created snapshot; a
    visible snapshot consumes one. Double counting when both already exist
    is deliberately conservative and never confers deletion rights.
    """
    if capacity_profile not in ('raw', 'native_family'):
        raise UnsafePublication('unsupported atomic stage capacity profile')
    occupied_slots = 0
    try:
        with os.scandir(parent) as children:
            for count, entry in enumerate(children, 1):
                if count > MAX_ATOMIC_PARENT_SCAN_ENTRIES:
                    raise UnsafePublication(
                        'atomic stage parent has too many entries for safe admission')
                if bundle_atomic.STAGE_NAME.fullmatch(entry.name):
                    occupied_slots += 2
                elif _RETAINED_SOURCE_SNAPSHOT.fullmatch(entry.name):
                    occupied_slots += 1


    except OSError as exc:
        raise UnsafePublication('cannot inventory retained private atomic stages') from exc
    if occupied_slots + 2 > MAX_RETAINED_ATOMIC_STAGES_PER_PARENT:
        raise UnsafePublication(
            'retained private atomic stage quota lacks two reserved slots '
            'for output and source snapshot; owner inspection needed; '
            'no automatic cleanup or deletion authorized')

    if capacity_profile == 'native_family':
        from . import hf_closure
        bundle_ceiling = bundle_atomic.MAX_NATIVE_FAMILY_BYTES
        source_reserve = hf_closure.MAX_TOTAL_BYTES + MIN_ATOMIC_SOURCE_RESERVE
    else:
        bundle_ceiling = bundle_atomic.MAX_BYTES
        source_reserve = MIN_ATOMIC_SOURCE_RESERVE
    largest_bundle = [
        ('bundle-capacity', bundle_ceiling, ''),
        *[('member-allocation', 1, '')
          for _ in range(bundle_atomic.MAX_MEMBERS - 1)],
    ]
    try:
        minimum = (hf_held_stage._required_stage_space(
            Path(parent), largest_bundle, ()) + source_reserve)
        free = shutil.disk_usage(parent).free
    except (OSError, ValueError) as exc:
        raise UnsafePublication('cannot measure NTFS space for private '
                                'atomic stage admission') from exc
    if free < minimum:
        raise UnsafePublication(
            'insufficient NTFS allocation-aware free disk reserve for '
            'new retained atomic source/output stage; existing data untouched')


def _validate_windows_pe_dll(path, *, target_arch):
    """Bounded PE32+ structural/CPU-machine preflight, never an execution oracle.

    This is a necessary gate before publishing an unchecked Windows DLL:
    source-integrity hashes and a .dll basename alone cannot prove the compiler
    produced a Windows binary for the approved destination architecture.
    """
    machines = {'x86_64': 0x8664, 'arm64': 0xAA64}
    expected = machines.get(target_arch)
    if expected is None:
        raise UnsafePublication('Windows native DLL target architecture is unsupported')
    try:
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if before.st_size < 512 or before.st_size > bundle_atomic.MAX_BYTES:
                raise UnsafePublication('Windows native DLL PE file size invalid')
            dos = stream.read(64)
            if len(dos) != 64 or dos[:2] != b'MZ':
                raise UnsafePublication('Windows native DLL lacks MZ DOS header')
            pe_offset = struct.unpack_from('<I', dos, 0x3C)[0]
            if pe_offset < 64 or pe_offset > 1024 * 1024:
                raise UnsafePublication('Windows native DLL PE header offset invalid')
            if pe_offset + 24 + 112 > before.st_size:
                raise UnsafePublication('Windows native DLL PE header truncated')
            stream.seek(pe_offset)
            coff = stream.read(24)
            if len(coff) != 24 or coff[:4] != b'PE\x00\x00':
                raise UnsafePublication('Windows native DLL lacks PE/COFF signature')
            machine, sections = struct.unpack_from('<HH', coff, 4)
            optional_size, characteristics = struct.unpack_from('<HH', coff, 20)
            if machine != expected:
                raise UnsafePublication('Windows native DLL PE machine differs '
                                        'from approved BuildPlan target ISA')
            if not 1 <= sections <= 96 or not 112 <= optional_size <= 4096:
                raise UnsafePublication('Windows native DLL sections/optional header invalid')
            if not (characteristics & 0x2000) or not (characteristics & 0x0002):
                raise UnsafePublication('Windows PE executable lacks DLL characteristics')
            if (pe_offset + 24 + optional_size + sections * 40 >
                    before.st_size):
                raise UnsafePublication('Windows native DLL section table truncated')
            optional = stream.read(112)
            if (len(optional) != 112 or
                    struct.unpack_from('<H', optional, 0)[0] != 0x20B):
                raise UnsafePublication('Windows native DLL requires PE32+ optional header')
            section_alignment = struct.unpack_from('<I', optional, 32)[0]
            image_size = struct.unpack_from('<I', optional, 56)[0]
            if section_alignment == 0 or image_size < section_alignment:
                raise UnsafePublication('Windows native DLL PE image sizing invalid')
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise UnsafePublication('Windows native DLL changed while PE was checked')
    except (OSError, struct.error) as exc:
        raise UnsafePublication('Windows native DLL PE inspection failed') from exc


def _unique_json_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise UnsafePublication('duplicate key in staged recipe: ' + key)
        record[key] = value
    return record


def _finite_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise UnsafePublication('nonfinite number in staged recipe')
    return value


def _strict_recipe(path):
    """Review bounded unambiguous JSON *before* the legacy recipe validator."""
    with open(path, 'rb') as source:
        raw = source.read(1024 * 1024 + 1)
    if not 0 < len(raw) <= 1024 * 1024 or raw.startswith(b'\xef\xbb\xbf'):
        raise UnsafePublication('staged recipe is empty, oversized or has a BOM')
    parsed = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_json_object,
                        parse_float=_finite_json_float,
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            UnsafePublication('nonfinite JSON constant in recipe')))


    validated = read_recipe(path)
    if not isinstance(parsed, dict) or parsed != validated:
        raise UnsafePublication('staged recipe changed while being checked')
    return parsed


def _validate_native_inputs(path):
    """Bounded RAW multi-input TSV check; do not imply a Pascal adapter exists."""
    with open(path, 'rb') as source:
        raw = source.read(65537)
    if not 0 < len(raw) <= 65536 or not raw.endswith(b'\n') or b'\r' in raw:
        raise UnsafePublication('native .inputs absent, oversized or not canonical LF')
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeError as exc:
        raise UnsafePublication('native .inputs is not valid UTF-8') from exc
    if not 1 <= len(lines) <= 128:
        raise UnsafePublication('native .inputs needs 1..128 named inputs')
    names = set()
    for index, line in enumerate(lines):
        parts = line.split('\t')
        if (len(parts) != 3 or parts[0] != str(index) or
                not parts[1] or any(ord(ch) < 32 for ch in parts[1]) or
                parts[1] in names or
                not re.fullmatch(r'-?[0-9]+(?:,-?[0-9]+){0,15}', parts[2])):
            raise UnsafePublication('native .inputs index/name/shape TSV is malformed')
        dims = [int(value) for value in parts[2].split(',')]
        if any(value == 0 or value < -1 or value > 2147483647 for value in dims):
            raise UnsafePublication('native .inputs dimensions are out of range')
        names.add(parts[1])
    return raw


def expected_native_inputs(model_path, model_format, shape=None):
    """Pin the converter's native IO ABI to actual *selected compiler input*.

    This is evaluated while the Preview-approved private model file is held by
    Win32 source_lock, before the compiler runs. The output .inputs sidecar is
    never trusted to describe itself. It is a structural IO check, NOT native
    loader execution or independent numerical evidence.
    """
    from . import convert, detect, pipeline

    if not isinstance(model_path, str) or not os.path.isfile(model_path):
        raise UnsafePublication('trusted compiler source missing for native IO pin')
    input_name, dims = pipeline.parse_shape(shape)
    overrides = {input_name: dims} if input_name and dims else None
    if model_format == detect.ONNX:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        rows = convert._onnx_input_names(model, overrides)
    elif model_format == detect.TFLITE:
        import tflite

        with open(model_path, 'rb') as source:
            payload = source.read(64 * 1024 * 1024 + 1)
        if not 8 <= len(payload) <= 64 * 1024 * 1024:
            raise UnsafePublication('bounded native TFLite source size violated')
        model = tflite.Model.GetRootAsModel(payload, 0)
        rows = convert._tflite_input_names(model, overrides)
    else:
        raise UnsafePublication('native IO pin supports approved RAW ONNX/TFLite only')

    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise UnsafePublication('selected RAW source has no bounded native inputs')
    try:
        document = ''.join(
            '%d\t%s\t%s\n' % (index, name, ','.join(str(d) for d in dimensions))
            for index, (name, dimensions) in enumerate(rows))
        payload = document.encode('utf-8')
    except (TypeError, UnicodeError) as exc:
        raise UnsafePublication('source graph has invalid native IO metadata') from exc
    if not 0 < len(payload) <= 65536:
        raise UnsafePublication('source graph native IO metadata exceeds bound')


    for line in document.splitlines():
        columns = line.split('\t')
        if (len(columns) != 3 or not columns[1] or
                any(ord(ch) < 32 for ch in columns[1]) or
                not re.fullmatch(r'-?[0-9]+(?:,-?[0-9]+){0,15}', columns[2]) or
                any(int(number) == 0 or int(number) < -1
                    or int(number) > 2147483647
                    for number in columns[2].split(','))):
            raise UnsafePublication('source graph native IO names/shapes unsafe')
    return payload


def _checked_source_entries(entries):
    """Validate exact approved RAW source file/sidecar paths and full SHAs."""
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise UnsafePublication('approved RAW source file-set has invalid entry count')
    selected = []
    seen = set()
    file_count = 0
    total = 0
    for entry in entries:
        if (not isinstance(entry, dict) or
                set(entry) != {'path', 'kind', 'bytes', 'sha256'}):
            raise UnsafePublication('malformed selected RAW source member')
        rel = bundle_atomic.validate_member_name(entry['path'])
        alias = rel.casefold()
        if alias in seen:
            raise UnsafePublication('duplicate/case-aliased RAW source member')
        seen.add(alias)
        kind = entry['kind']
        if kind == 'directory':
            if entry['bytes'] != 0 or entry['sha256'] is not None:
                raise UnsafePublication('invalid selected source directory metadata')
        elif kind == 'file':
            if (type(entry['bytes']) is not int or
                    not 0 < entry['bytes'] <= 64 * 1024 * 1024 or
                    not isinstance(entry['sha256'], str) or
                    not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
                raise UnsafePublication('selected RAW source file length/SHA invalid')
            file_count += 1
            total += entry['bytes']
        else:
            raise UnsafePublication('unexpected selected RAW source member kind')
        selected.append(entry)
    if not 1 <= file_count <= 32 or total > 64 * 1024 * 1024:
        raise UnsafePublication('selected source file/byte quotas exceeded')
    directories = {entry['path'] for entry in selected
                   if entry['kind'] == 'directory'}
    required_dirs = {
        '/'.join(entry['path'].split('/')[:depth])
        for entry in selected for depth in range(1, len(entry['path'].split('/')))
    }
    if directories != required_dirs:
        raise UnsafePublication('selected source missing/extra parent directory')
    return sorted(selected, key=lambda row: row['path'])


def stage_approved_sources(output_stage, locked_source_stage, source_entries):
    """Copy every approved RAW model/sidecar under bundle/source while LOCKED.

    Caller holds ALL native Win32 selected source file handles through this copy
    and later post-lock audit. The copy has its own byte-exact SHA and fsync;
    neither copy nor SHA means power-loss durable or release-grade immutable.
    An exception leaves ONLY a private unpublished stage; no cleanup of owner
    files, no fallback to references in the original model directory.
    """
    entries = _checked_source_entries(source_entries)
    destination = Path(bundle_atomic._local_absolute(output_stage))
    origin = source_freeze._absolute_path(locked_source_stage, 'selected source stage')
    source_freeze._safe_ancestors(origin)
    if not origin.is_dir():
        raise UnsafePublication('selected source stage must be a regular directory')
    bundle_atomic._guard_existing_ancestors(str(destination))
    target = destination / 'source'
    if os.path.lexists(target):
        raise UnsafePublication('private output has pre-existing source directory')
    target.mkdir(exist_ok=False)
    for entry in entries:
        path = entry['path']
        source = origin.joinpath(*path.split('/'))
        copied = target.joinpath(*path.split('/'))
        source_freeze._safe_ancestors(source)
        if entry['kind'] == 'directory':
            bundle_atomic._guard_stat(str(source), directory=True)
            copied.mkdir(exist_ok=False)
        else:
            source_freeze._copy_stream(
                source, copied, entry['sha256'], entry['bytes'], 1024 * 1024)


    inventory = bundle_atomic.scan_bundle_tree(str(target))
    files = {member.path: member for member in inventory.members
             if member.kind == 'file'}
    directories = {member.path for member in inventory.members
                   if member.kind == 'directory'}
    if (set(files) != {entry['path'] for entry in entries
                      if entry['kind'] == 'file'} or
            directories != {entry['path'] for entry in entries
                            if entry['kind'] == 'directory'} or
            any(files[entry['path']].sha256 != entry['sha256'] or
                files[entry['path']].size != entry['bytes'] for entry in entries
                if entry['kind'] == 'file')):
        raise UnsafePublication('copied selected source tree differs from approved bytes')
    return entries


def create_private_stage(destination, *, on_created=None, on_intent=None,
                         capacity_profile='raw'):
    """Exclusively create a same-parent NTFS stage; never claim the final name."""
    final = bundle_atomic._local_absolute(destination)
    parent = os.path.dirname(final)
    bundle_atomic._guard_existing_ancestors(parent)
    bundle_atomic._require_ntfs(parent)
    if os.path.lexists(final):
        raise UnsafePublication('selected final output is occupied before staging')
    import secrets
    with _STAGE_ADMISSION_LOCK:
        with _cross_process_atomic_admission(parent):
            if os.path.lexists(final):
                raise UnsafePublication('final output occupied during stage admission')
            admitted_parent = bundle_atomic._guard_stat(parent, directory=True)
            _admit_private_atomic_stage(parent,
                                        capacity_profile=capacity_profile)
            stage = os.path.join(
                parent, '.crosspascal-stage-' + secrets.token_hex(16))

            if on_intent is not None:
                on_intent(stage)

            try:
                bundle_atomic._guard_existing_ancestors(parent)
                current_parent = bundle_atomic._guard_stat(parent, directory=True)
            except (OSError, ValueError) as exc:
                raise UnsafePublication(
                    'atomic stage parent became unsafe after INTENT; no mkdir') from exc
            if ((current_parent.st_dev, current_parent.st_ino) !=
                    (admitted_parent.st_dev, admitted_parent.st_ino)):
                raise UnsafePublication(
                    'atomic stage parent identity changed after INTENT; no mkdir')
            try:
                os.mkdir(stage)
            except OSError as exc:
                raise UnsafePublication(
                    'private stage could not be exclusively created') from exc

            created = bundle_atomic._guard_stat(stage, directory=True)
            if on_created is not None:
                on_created(stage, (created.st_dev, created.st_ino))
    return stage


def _validate_kit(stage, report, target, approved_source, expected_model_sha256,
                  expected_input_manifest, expected_source_entries,
                  closure_sha, closure):
    """Bind exact compiled bytes, wheel runtime, licence, plan and source receipt."""
    if target not in ('cpu', 'cuda') or not report.sound:
        raise UnsafePublication('failed or invalid compiler report cannot commit')
    if report.summary.get('verification_status') not in ('NOT_CHECKED', 'VERIFIED_LOCAL'):
        raise UnsafePublication('no exportable independent verification result')
    from .report import DONE, UNCHECKED
    comparisons = [item for item in report.steps
                   if item.name == 'Comparison with the reference']
    verification = report.summary['verification_status']
    if (len(comparisons) != 1 or
            (verification == 'VERIFIED_LOCAL' and comparisons[0].state != DONE) or
            (verification == 'NOT_CHECKED' and comparisons[0].state != UNCHECKED)):
        raise UnsafePublication('independent Verify step and claimed status disagree')

    module = report.summary.get('module')
    stage = bundle_atomic._local_absolute(stage)
    if (not isinstance(module, str) or not os.path.isabs(module) or
            os.path.normcase(os.path.abspath(os.path.dirname(module))) !=
            os.path.normcase(stage)):
        raise UnsafePublication('compiler module is not inside the selected private stage')
    module_name = os.path.basename(module)
    if module_name != 'model.' + target + '.dll':
        raise UnsafePublication('unexpected Windows RAW native module name/target')

    approved = bundle_atomic.scan_bundle_tree(stage)
    actual = {m.path: m for m in approved.members if m.kind == 'file'}

    try:
        entries = bundle_download.collect_bundle_files(
            stage, expected_source_closure_sha256=closure_sha,
            expected_source_closure=closure)
        recipe = _strict_recipe(os.path.join(stage, 'recipe.json'))
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafePublication('staged receipt/recipe/ZIP closure refused') from exc
    if (type(recipe.get('recipe_version')) is not int or
            recipe['recipe_version'] != 2 or
            recipe.get('source') != approved_source or
            recipe.get('build_plan') != report.summary.get('build_plan') or
            recipe.get('target') != target or recipe.get('name') != 'model'):
        raise UnsafePublication('saved recipe differs from exact compiler BuildPlan')
    _validate_windows_pe_dll(
        os.path.join(stage, module_name),
        target_arch=recipe['build_plan']['target_profile']['deployment']['arch'])

    try:
        wheel = runtime.plan_runtime_files(target, target_dir=stage)
    except (OSError, ValueError) as exc:
        raise UnsafePublication('runtime wheel source inventory unavailable') from exc
    if (not isinstance(wheel, dict) or not wheel.get('ready') or
            not wheel.get('libraries') or not wheel.get('licence_files')):
        raise UnsafePublication('runtime or redistribution licences are incomplete')

    if set(actual) != {entry.name for entry in entries}:
        raise UnsafePublication('directory inventory differs from inspected ZIP members')

    manifest_names = {'manifest.json', 'manifest.txt'}
    has_contract = (isinstance(recipe.get('contract'), dict) and
                    bool(recipe['contract'].get('task')))
    if has_contract and not manifest_names.issubset(actual):
        raise UnsafePublication('declared Pascal adapter has no complete verified passport')
    if not has_contract and manifest_names.intersection(actual):
        raise UnsafePublication('unclassified RAW build cannot mint a Pascal passport')

    required = {module_name, module_name + '.inputs', 'recipe.json', 'report.txt'}
    if has_contract:

        if verification != 'VERIFIED_LOCAL':
            raise UnsafePublication('Pascal manifest needs independently '
                                    'VERIFIED_LOCAL native model acceptance')
        try:
            passport = bundle_download._validate_pascal_passport_pair(entries)
        except (OSError, ValueError, TypeError) as exc:
            raise UnsafePublication('Pascal manifest pair is invalid') from exc
        contract = recipe['contract']
        deployment = recipe['build_plan']['target_profile']['deployment']
        options = recipe['build_plan']['options']
        artifact = passport.get('artifact')
        io = passport.get('io')
        if (not isinstance(artifact, dict) or not isinstance(io, dict) or
                passport.get('task_kind') != contract.get('task') or
                passport.get('adapter') != contract.get('adapter') or
                type(passport.get('adapter_api_version')) is not int or
                passport['adapter_api_version'] != contract.get('adapter_api_version') or
                io.get('layout') != contract.get('layout') or
                artifact.get('file') != module_name or
                artifact.get('device') != target or
                artifact.get('os') != deployment.get('os') or
                artifact.get('arch') != deployment.get('arch') or
                artifact.get('dtype') != options.get('dtype') or
                io.get('dtype') != options.get('dtype') or
                artifact.get('hash') != actual[module_name].sha256):
            raise UnsafePublication('Pascal manifest task/adapter/ABI differs '
                                    'from approved recipe and BuildPlan')

        try:
            from .contract import build_vision_contract
            expected_pascal = build_vision_contract(
                contract['task'], contract['adapter'], contract['layout'],
                module_name, io['name'], io['shape'],
                adapter_api_version=contract['adapter_api_version'],
                os_=deployment['os'], arch=deployment['arch'],
                device=target, dtype=options['dtype'],
                score_threshold=contract.get('score_threshold'),
                nms_threshold=contract.get('nms_threshold'),
                mean=contract.get('mean'), std=contract.get('std'),
                keypoint_count=contract.get('keypoint_count'))
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsafePublication('Pascal manifest approved task contract '
                                    'cannot be reconstructed') from exc
        if (any(passport.get(key) != expected_pascal.get(key)
                for key in ('defaults', 'vision', 'pose')) or
                artifact.get('entry') != expected_pascal['artifact']['entry'] or
                artifact.get('format') != expected_pascal['artifact']['format']):
            raise UnsafePublication('Pascal manifest task defaults or native '
                                    'ABI differs from approved recipe')

        provenance = passport.get('provenance')
        if (not isinstance(provenance, dict) or
                provenance.get('source') != recipe['source'] or
                provenance.get('source_hash') != recipe.get('source_hash') or
                provenance.get('built_at') != recipe.get('built_at') or
                passport.get('toolchain') != recipe.get('toolchain') or
                passport.get('license') != recipe.get('license')):
            raise UnsafePublication('Pascal manifest provenance/toolchain/'
                                    'licence differs from approved recipe')

        try:
            native_io = _validate_native_inputs(
                os.path.join(stage, module_name + '.inputs')).decode('utf-8')
            lines = native_io.splitlines()
            if len(lines) != 1:
                raise ValueError('multiple inputs need a family adapter')
            index, name, shape = lines[0].split('\t')
            if (index != '0' or io.get('name') != name or
                    io.get('shape') != [int(dim) for dim in shape.split(',')]):
                raise ValueError('Pascal input differs from native model')
        except (ValueError, TypeError, KeyError) as exc:
            raise UnsafePublication('Pascal manifest input ABI differs from '
                                    'approved compiled .inputs') from exc
        required.update(manifest_names)
    mandatory_runtime = {'tvm_runtime.dll', 'tvm_runtime_extra.dll', 'tvm_ffi.dll'}
    if target == 'cuda':
        mandatory_runtime.add('tvm_runtime_cuda.dll')
    returned_libraries = {item.get('file') for item in wheel['libraries']
                          if isinstance(item, dict)}
    if not mandatory_runtime.issubset(returned_libraries):
        raise UnsafePublication('runtime wheel inventory omits mandatory TVM libraries')
    if not any(item.get('file', '').startswith('LICENSE-tvm-')
               for item in wheel['licence_files'] if isinstance(item, dict)):
        raise UnsafePublication('TVM wheel has no attributable licence notice')
    for component in wheel['libraries'] + wheel['licence_files']:
        if (not isinstance(component, dict) or
                not isinstance(component.get('file'), str) or
                not isinstance(component.get('sha256'), str) or
                type(component.get('bytes')) is not int):
            raise UnsafePublication('runtime wheel inventory contains invalid member')
        name = bundle_atomic.validate_member_name(component['file'])
        required.add(name)
        member = actual.get(name)
        if (member is None or member.size != component['bytes'] or
                member.sha256 != component['sha256']):
            raise UnsafePublication('bundled runtime/licence differs from wheel SHA: ' + name)
    if (not required.issubset(actual) or
            any(actual[name].size <= 0 for name in required)):
        raise UnsafePublication('missing/empty native module, IO, runtime or receipt')
    root_dlls = {name for name in actual if '/' not in name and name.lower().endswith('.dll')}
    allowed_dlls = {module_name} | returned_libraries
    if root_dlls != allowed_dlls:
        raise UnsafePublication('unexpected/uninventoried native DLL in staged bundle')
    if actual['report.txt'].size > 1024 * 1024:
        raise UnsafePublication('compiler report exceeds bounded receipt metadata')
    with open(os.path.join(stage, 'report.txt'), 'rb') as source:
        saved_report = source.read(1024 * 1024 + 1)
    if saved_report != (report.as_text() + '\n').encode('utf-8'):
        raise UnsafePublication('staged report.txt does not match job-owned truth')
    actual_input_manifest = _validate_native_inputs(
        os.path.join(stage, module_name + '.inputs'))
    if (not isinstance(expected_input_manifest, bytes) or
            actual_input_manifest != expected_input_manifest):
        raise UnsafePublication('native .inputs disagrees with Preview-approved '
                                'compiler source graph IO contract')

    source_hash = recipe.get('source_hash')
    if (not isinstance(source_hash, str) or
            not re.fullmatch(r'[0-9a-f]{64}', source_hash) or
            not isinstance(expected_model_sha256, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_model_sha256) or
            source_hash != expected_model_sha256):
        raise UnsafePublication('compiler input SHA differs from Win32 post-lock '
                                'approved source, or recipe has no valid SHA')
    if closure_sha is not None:
        required.add('source-closure.json')
        if 'source-closure.json' not in actual:
            raise UnsafePublication('selected ONNX source-lock receipt missing')
        if not isinstance(closure, dict):
            raise UnsafePublication('selected source tree lacks trusted job details')
        model = next((item for item in closure.get('entries', ())
                      if item.get('path') == closure.get('model_rel')
                      and item.get('kind') == 'file'), None)
        if model is None or model.get('sha256') != source_hash:
            raise UnsafePublication('compiler input SHA differs from selected closure')
        if expected_source_entries != closure.get('entries'):
            raise UnsafePublication('included source/sidecars differ from locked closure')
    elif 'source-closure.json' in actual or recipe.get('source_closure') is not None:
        raise UnsafePublication('unapproved selected-source receipt in single-file RAW kit')

    source_entries = _checked_source_entries(expected_source_entries)
    source_files = [entry for entry in source_entries if entry['kind'] == 'file']
    model_matches = [entry for entry in source_files
                     if entry['path'] == os.path.basename(approved_source)]
    if (len(model_matches) != 1 or
            model_matches[0]['sha256'] != expected_model_sha256 or
            (closure_sha is None and len(source_files) != 1)):
        raise UnsafePublication('RAW bundled model differs from locked model source')
    for entry in source_files:
        name = 'source/' + entry['path']
        required.add(name)
        member = actual.get(name)
        if (member is None or member.size != entry['bytes'] or
                member.sha256 != entry['sha256']):
            raise UnsafePublication('source/sidecar bytes differ from held compiler input')

    directories = {member.path for member in approved.members
                   if member.kind == 'directory'}
    expected_dirs = {'source'} | {'source/' + entry['path']
                                  for entry in source_entries
                                  if entry['kind'] == 'directory'}
    if set(actual) != required or directories != expected_dirs:
        raise UnsafePublication('staged RAW kit has unapproved files/directories')
    return approved, tuple(sorted(required))


def commit_verified_stage(stage, destination, report, target, approved_source,
                          *, expected_model_sha256, expected_input_manifest,
                          expected_source_entries,
                          closure_sha=None, closure=None,
                          on_validated_precommit=None):
    """Commit the complete stage after all validations, with no final replacement.

    This function never removes a previous final path. The caller must have
    reserved this output *in memory*, not by creating a directory on disk.
    """
    if os.path.lexists(destination):
        raise UnsafePublication('final output already occupied; no replacement')
    inventory, required = _validate_kit(
        stage, report, target, approved_source, expected_model_sha256,
        expected_input_manifest, expected_source_entries,
        closure_sha, closure)
    if on_validated_precommit is not None:

        on_validated_precommit(stage, destination, inventory)
    receipt = bundle_atomic.commit_staged_bundle(
        stage, destination, inventory, required_files=required)
    return inventory, receipt


def inspect_selected_native_consumer_kit(directory, *, model_key, device,
                                         expected_tree_sha256=None):
    """Bound a caller-selected tokenizer/runtime/licence input tree exactly.

    A full SHA snapshot is a build input declaration, NOT an executable/native
    provenance check or redistribution-rights attestation. Nothing is copied,
    published or deleted by this read-only scanner.
    """
    from . import llm_bundle_contract
    profiles = llm_bundle_contract._FAMILIES
    if model_key not in profiles or device not in ('cpu', 'cuda'):
        raise UnsafePublication('unsupported selected native consumer kit profile')
    if not isinstance(directory, str) or not os.path.isabs(directory):
        raise UnsafePublication('selected native consumer kit requires absolute root')
    inventory = bundle_atomic.scan_bundle_tree(directory)
    files = {member.path: member for member in inventory.members
             if member.kind == 'file'}
    directories = [member.path for member in inventory.members
                   if member.kind == 'directory']
    expected = set(profiles[model_key][3]) | set(llm_bundle_contract._RUNTIME)
    if device == 'cuda':
        expected.add('tvm_runtime_cuda.dll')
    licenses = {name for name in files if name.startswith('LICENSE-tvm-')}
    notices = {name for name in files if name.startswith('NOTICE-tvm-')}
    qwen_licenses = {
        'qwen2.5-0.5b': 'LICENSE-Qwen2.5-0.5B-Instruct.txt',
        'qwen2.5-1.5b': 'LICENSE-Qwen2.5-1.5B-Instruct.txt',
                                                                            
                                                                        
        'qwen2-compatible-local': 'LICENSE-Qwen2.5-0.5B-Instruct.txt',
    }
    model_files = ({qwen_licenses[model_key], 'MODEL-SOURCE.json'}
                   if model_key in qwen_licenses else set())
    selected_model_files = set(files) & model_files
    if (directories or not licenses or
            any(not name.startswith('LICENSE-tvm-') for name in licenses) or
            set(files) != expected | licenses | notices | selected_model_files or
            any(member.size <= 0 for member in files.values())):
        raise UnsafePublication('unapproved or incomplete selected native consumer kit')
    if model_key == 'qwen2-compatible-local':
        from .model_sources import catalog_model
                                                                      
                                                                             
                                                                             
        official_kit = catalog_model('qwen2.5-0.5b')
        pin = official_kit['license_file']
        name = qwen_licenses[model_key]
        item = files.get(name)
        if (item is None or item.size != pin['bytes'] or
                item.sha256 != pin['sha256'] or
                'MODEL-SOURCE.json' not in files or not notices):
            raise UnsafePublication('generic Qwen requires an intact selected '
                                    'Qwen tokenizer/runtime kit, not an unlicensed relabel')
    if model_key == 'qwen2.5-1.5b':
        
        
        
        from .model_sources import catalog_model
        from pathlib import Path
        official = catalog_model(model_key)
        name = qwen_licenses[model_key]
        legal = files.get(name)
        license_pin = official['license_file']
        if (legal is None or legal.size != license_pin['bytes'] or
                legal.sha256 != license_pin['sha256'] or
                'MODEL-SOURCE.json' not in files):
            raise UnsafePublication('Qwen 1.5B consumer kit lacks pinned license/source marker')
        marker = llm_bundle_contract._json(
            Path(directory), 'MODEL-SOURCE.json',
            approved_sha256=files['MODEL-SOURCE.json'].sha256)
        if (not isinstance(marker, dict) or
                marker.get('schema') != 'crosspascal-qwen-model-source-v1' or
                marker.get('repository') != official['repository'] or
                marker.get('revision') != official['revision'] or
                marker.get('source_tree_sha256') != official['tree_sha256'] or
                not isinstance(marker.get('license'), dict) or
                marker['license'].get('file') != name or
                any(marker['license'].get(k) != license_pin[k]
                    for k in ('bytes', 'sha256', 'url')) or
                not isinstance(marker.get('files'), list) or
                not all(isinstance(row, dict) and type(row.get('name')) is str
                        for row in marker['files']) or
                {row.get('name'): (row.get('bytes'), row.get('sha256'),
                                   row.get('url'))
                 for row in marker['files']} !=
                {row['path']: (row['bytes'], row['sha256'], row['url'])
                 for row in official['files']} or
                len(marker['files']) != len(official['files'])):
            raise UnsafePublication('Qwen 1.5B kit MODEL-SOURCE differs from the pinned official revision')
    if (expected_tree_sha256 is not None and
            inventory.tree_sha256 != expected_tree_sha256):
        raise UnsafePublication('selected native consumer kit SHA changed')
    return inventory


def stage_selected_native_consumer_kit(source_directory, private_stage,
                                       approved_inventory, *, model_key, device):
    """Exclusively copy a selected flat consumer kit into an EMPTY private stage.

    The original stays untouched; partial errors retain the stage for inspection.
    Full source/target SHA checks are point-in-time, not held Win32 custody.
    """
    if not isinstance(approved_inventory, bundle_atomic.Inventory):
        raise UnsafePublication('selected native consumer kit lacks approved Inventory')
    origin = bundle_atomic._local_absolute(source_directory)
    target = bundle_atomic._local_absolute(private_stage)
    if not bundle_atomic.STAGE_NAME.fullmatch(os.path.basename(target)):
        raise UnsafePublication('consumer kit destination is not a private stage')
    real_origin = os.path.normcase(os.path.realpath(origin))
    real_target = os.path.normcase(os.path.realpath(target))
    try:
        common_root = os.path.commonpath((real_origin, real_target))
    except ValueError:
        common_root = None
    if common_root in (real_origin, real_target):
        raise UnsafePublication('consumer kit source and stage overlap')
    before = inspect_selected_native_consumer_kit(
        origin, model_key=model_key, device=device,
        expected_tree_sha256=approved_inventory.tree_sha256)
    if before != approved_inventory:
        raise UnsafePublication('selected native consumer kit Inventory changed')
    bundle_atomic._guard_existing_ancestors(target)
    initial = bundle_atomic._guard_stat(target, directory=True)
    with os.scandir(target) as contents:
        if next(contents, None) is not None:
            raise UnsafePublication('private consumer stage is not empty')
    for member in approved_inventory.members:
        if member.kind != 'file':
            raise UnsafePublication('nested consumer kit member is not approved')

        bundle_atomic._guard_existing_ancestors(target)
        current = bundle_atomic._guard_stat(target, directory=True)
        if (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino):
            raise UnsafePublication('private consumer stage identity changed before copy')
        source_freeze._safe_ancestors(Path(target) / member.path,
                                      leaf_may_be_missing=True)
        source_freeze._copy_stream(
            Path(origin) / member.path, Path(target) / member.path,
            member.sha256, member.size, 1024 * 1024)
    if (bundle_atomic._guard_stat(target, directory=True).st_dev,
            bundle_atomic._guard_stat(target, directory=True).st_ino) != (
            initial.st_dev, initial.st_ino):
        raise UnsafePublication('private consumer stage identity changed')
    if (inspect_selected_native_consumer_kit(
            origin, model_key=model_key, device=device) != approved_inventory or
            inspect_selected_native_consumer_kit(
                target, model_key=model_key, device=device) != approved_inventory):
        raise UnsafePublication('selected consumer copy differs from approved SHA')
    return approved_inventory


_QWEN_KIT_LOCK = threading.Lock()


def select_installed_qwen15_kit(source, cache_directory):
    """Use the installed runtime and shared Qwen tokenizer with 1.5B provenance."""
    try:
        inventory = inspect_selected_native_consumer_kit(
            source, model_key='qwen2.5-1.5b', device='cuda')
        return source, inventory
    except UnsafePublication:
        base = inspect_selected_native_consumer_kit(
            source, model_key='qwen2.5-0.5b', device='cuda')

    from .model_sources import catalog_model
    official = catalog_model('qwen2.5-1.5b')
    smaller = catalog_model('qwen2.5-0.5b')
    by_name = {member.path: member for member in base.members}
    old_name = 'LICENSE-Qwen2.5-0.5B-Instruct.txt'
    legal_name = 'LICENSE-Qwen2.5-1.5B-Instruct.txt'
    license_pin = official['license_file']
    license_file = by_name.get(old_name)
    tokenizer_inputs = {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}

    def tokenizer_identity(model):
        return {row['path']: (row['bytes'], row['sha256']) for row in model['files']
                if row['path'] in tokenizer_inputs}

    if (len(tokenizer_identity(official)) != len(tokenizer_inputs) or
            tokenizer_identity(official) != tokenizer_identity(smaller) or
            license_file is None or license_file.size != license_pin['bytes'] or
            license_file.sha256 != license_pin['sha256'] or
            'MODEL-SOURCE.json' not in by_name):
        raise UnsafePublication('Installed Qwen kit cannot supply the pinned 1.5B source files')

    parent = Path(cache_directory)
    bundle_atomic._guard_existing_ancestors(parent)
    target = parent / f'qwen15-kit-{base.tree_sha256}-{official["tree_sha256"][:12]}'
    with _QWEN_KIT_LOCK:
        if not os.path.lexists(target):
            stage = parent / ('.crosspascal-stage-' + os.urandom(16).hex())
            stage.mkdir(mode=0o700)
            stage_selected_native_consumer_kit(
                source, str(stage), base, model_key='qwen2.5-0.5b', device='cuda')
            (stage / old_name).rename(stage / legal_name)
            marker = {
                'schema': 'crosspascal-qwen-model-source-v1',
                'repository': official['repository'],
                'revision': official['revision'],
                'source_tree_sha256': official['tree_sha256'],
                'files': [{'name': row['path'], 'bytes': row['bytes'],
                           'sha256': row['sha256'], 'url': row['url']}
                          for row in official['files']],
                'license': {'file': legal_name, **{
                    key: license_pin[key] for key in ('bytes', 'sha256', 'url')}},
                'tokenizer_exports': {
                    'format': 'CrossPascal byte BPE',
                    'provenance': 'Shared pinned tokenizer inputs. Build validates the exports.',
                    'model_weights_included': False,
                },
            }
            (stage / 'MODEL-SOURCE.json').write_text(
                json.dumps(marker, indent=2) + '\n', encoding='utf-8')
            staged = inspect_selected_native_consumer_kit(
                str(stage), model_key='qwen2.5-1.5b', device='cuda')
            bundle_atomic.commit_staged_bundle(
                stage, target, staged, required_files=(legal_name, 'MODEL-SOURCE.json'))
        selected = inspect_selected_native_consumer_kit(
            str(target), model_key='qwen2.5-1.5b', device='cuda')
        common = {member.path: member for member in selected.members
                  if member.path not in (legal_name, 'MODEL-SOURCE.json')}
        if common != {name: member for name, member in by_name.items()
                      if name not in (old_name, 'MODEL-SOURCE.json')}:
            raise UnsafePublication('Prepared Qwen 1.5B kit differs from the installed runtime')
    return str(target), selected


def commit_native_family_stage(stage, destination, plan, *, model_key,
                               approved_consumer_inventory=None):
    """Validate the native family kit and atomically expose its entire tree."""
    if os.path.lexists(destination):
        raise UnsafePublication('native family final output already occupied')
    from .llm_bundle_contract import validate_llm_output
    from .qwen_universal import KEY

    validate_llm_output(stage, plan, model_key, require_consumer=True)
    if model_key == KEY:
        if not isinstance(approved_consumer_inventory, bundle_atomic.Inventory):
            raise UnsafePublication('local Qwen bundle requires an approved tokenizer kit')
        marker = Path(stage) / 'MODEL-SOURCE.json'
        kit_marker = marker.with_name('TOKENIZER-KIT-SOURCE.json')
        if os.path.lexists(kit_marker):
            raise UnsafePublication('local Qwen tokenizer kit marker already exists')
        marker.rename(kit_marker)
    entries = bundle_download.collect_bundle_files(stage)
    inventory = bundle_atomic.scan_bundle_tree(
        stage, max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
    if approved_consumer_inventory is not None:
        if not isinstance(approved_consumer_inventory, bundle_atomic.Inventory):
            raise UnsafePublication('native consumer commit requires approved Inventory')
        approved_files = {member.path: member for member in
                          approved_consumer_inventory.members if member.kind == 'file'}
        if model_key == KEY:
            source = approved_files.pop('MODEL-SOURCE.json', None)
            if source is None:
                raise UnsafePublication('local Qwen tokenizer kit lacks its source record')
            approved_files['TOKENIZER-KIT-SOURCE.json'] = replace(
                source, path='TOKENIZER-KIT-SOURCE.json')
        actual_files = {member.path: member for member in inventory.members
                        if member.kind == 'file'}
        from .llm_bundle_contract import _FAMILIES
        device = plan['target_profile']['deployment']['device']
        library = '%s-%s.dll' % (_FAMILIES[model_key][0], device)

        native_outputs = {library, 'model.json', 'params.bin', 'params.json',
                          'params.index', 'manifest.json', 'manifest.txt'}
        receipts = {'recipe.json', 'build-plan.json',
                    'import-provenance.json', 'report.txt'}
        required_files = set(approved_files) | native_outputs | receipts

        optional_import_lib = '%s-%s.lib' % (_FAMILIES[model_key][0], device)
        allowed_files = required_files | {optional_import_lib}
        if (len(approved_files) != len(approved_consumer_inventory.members) or
                len(actual_files) != len(inventory.members) or
                not required_files.issubset(actual_files) or
                not set(actual_files).issubset(allowed_files) or
                any(actual_files.get(name) != member
                    for name, member in approved_files.items())):
            raise UnsafePublication('native family stage has unapproved or changed files')
    files = {member.path for member in inventory.members if member.kind == 'file'}
    if files != {entry.name for entry in entries}:
        raise UnsafePublication('native family inventory and ZIP closure differ')
    required = ('model.json', 'params.bin', 'params.json', 'params.index',
                'manifest.json', 'manifest.txt', 'recipe.json',
                'build-plan.json', 'import-provenance.json', 'report.txt')
    receipt = bundle_atomic.commit_staged_bundle(
        stage, destination, inventory, required_files=required,
        max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
    if approved_consumer_inventory is not None:


        final_inventory = bundle_atomic.scan_bundle_tree(
            destination, max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
        final_files = {member.path: member for member in final_inventory.members
                       if member.kind == 'file'}
        if (final_inventory != inventory or
                any(final_files.get(name) != member
                    for name, member in approved_files.items())):
            raise UnsafePublication('published native consumer inventory changed')
    return inventory, receipt
