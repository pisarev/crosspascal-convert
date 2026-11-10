# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import hashlib
import json
import os
import re
import stat
import unicodedata
import zipfile

from .manifest import emit_manifest_text


INFO_FILE = '_pascalvision_download_info.json'
READ_CHUNK = 1024 * 1024

_MAX_LEGACY_MEMBERS = 4096
_MAX_LEGACY_BYTES = 64 * 1024 ** 3
_MAX_LLM_BUILDPLAN_BYTES = 1024 * 1024
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_PASSPORT_PAIR = frozenset(('manifest.json', 'manifest.txt'))
_MAX_PASSPORT_BYTES = 1024 * 1024
_HEX_SHA256 = re.compile(r'[a-fA-F0-9]{64}\Z')
_SOURCE_CLOSURE_RECEIPT = 'source-closure.json'
_WINDOWS_RESERVED_DEVICES = {'CON', 'PRN', 'AUX', 'NUL'} | {
    '%s%s' % (base, suffix)
    for base in ('COM', 'LPT')
    for suffix in ('1', '2', '3', '4', '5', '6', '7', '8', '9',
                   '\u00b9', '\u00b2', '\u00b3')
}


class UnsafeBundle(ValueError):
    pass


class UnsafeBundleIntegrity(UnsafeBundle):
    pass


def _reserved_windows_component(name):
    return name.split('.', 1)[0].upper() in _WINDOWS_RESERVED_DEVICES


class _CollectedMember:
    __slots__ = ('name', 'path', 'size', 'identity', 'sha256')

    def __init__(self, name, path, info, sha256):
        self.name = name
        self.path = path
        self.size = info.st_size
        self.identity = _identity(info)
        self.sha256 = sha256

    def __iter__(self):
        yield self.name
        yield self.path
        yield self.size


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _regular_path(path, *, directory, structural_admission=False):
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise UnsafeBundle('bundle path is inaccessible: %s' % exc) from exc
    refuse = UnsafeBundleIntegrity if structural_admission else UnsafeBundle
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & _REPARSE:
        raise refuse('bundle path contains a symlink or reparse point')
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode):
        raise refuse('bundle path is not a regular %s'
                     % ('directory' if directory else 'file'))
    if not directory and info.st_nlink != 1:
        raise refuse('bundle member is a hardlink or has unsafe link count')
    return info


def _refuse_walk_error(error):
    raise UnsafeBundle(
        'cannot fully enumerate bundle directory: %s' % error) from error


def _bounded_zip_members(root, *, allow_download_info):
    pending = [(root, '')]
    selected = []
    member_count = 0
    total_bytes = 0
    while pending:
        folder, prefix = pending.pop()
        _regular_path(folder, directory=True, structural_admission=True)
        listing = []
        try:
            with os.scandir(folder) as entries:
                for item in entries:
                    if member_count + len(listing) + 1 > _MAX_LEGACY_MEMBERS:
                        raise UnsafeBundleIntegrity('legacy ZIP exceeds member quota')
                    listing.append(item)
        except OSError as exc:
            raise UnsafeBundle(
                'cannot fully enumerate bundle directory: %s' % exc) from exc
        alias = set()
        directories = []
        for item in sorted(listing, key=lambda x: x.name):
            name = item.name
            relative = prefix + name
            if (not name or name in ('.', '..')
                    or unicodedata.normalize('NFC', relative) != relative
                    or any(ch in relative for ch in (':', '\\', '\x00'))
                    or any(ch in name for ch in '<>"|?*')
                    or _reserved_windows_component(name)
                    or name.endswith(('.', ' '))
                    or any(ord(ch) < 32 for ch in name)):
                raise UnsafeBundleIntegrity(
                    'legacy ZIP has noncanonical Unicode or '
                    'unsafe Windows archive member name')
            folded = unicodedata.normalize('NFC', name).casefold()
            if folded in alias:
                raise UnsafeBundleIntegrity(
                    'legacy ZIP has case/Unicode aliased members')
            alias.add(folded)
            full = os.path.join(folder, name)
            info = _regular_path(full, directory=item.is_dir(follow_symlinks=False),
                                 structural_admission=True)
            member_count += 1
            if stat.S_ISDIR(info.st_mode):
                if name.startswith('.pascalvision-passport-'):
                    raise UnsafeBundleIntegrity(
                        'bundle contains unpublished passport staging')
                directories.append((full, relative + '/'))
                continue
            if not stat.S_ISREG(info.st_mode):
                raise UnsafeBundleIntegrity('bundle contains a special file')
            if (name.endswith('.part') and name.startswith(
                    ('.tvm-runtime-', '.pascalvision-upload-'))):
                raise UnsafeBundleIntegrity(
                    'bundle contains an unpublished partial file')
            if name.casefold() == INFO_FILE.casefold() and not (
                    prefix == '' and name == INFO_FILE and allow_download_info):
                raise UnsafeBundleIntegrity(
                    'bundle collides with reserved export metadata')
            if info.st_size == 0 and relative.lower().endswith(
                    ('.dll', '.so', '.dylib', '.exe', '.cubin', '.fatbin',
                     '.ptx', '.onnx', '.safetensors')):
                raise UnsafeBundleIntegrity(
                    'bundle contains an empty executable/model binary: %s'
                    % relative)
            if info.st_size < 0 or total_bytes + info.st_size > _MAX_LEGACY_BYTES:
                raise UnsafeBundleIntegrity('legacy ZIP exceeds aggregate byte quota')
            total_bytes += info.st_size
            selected.append((relative, full, info))

        pending.extend(reversed(directories))
    return selected


def _hash_inventory_member(path, info):
    initial = _identity(info)
    measured = hashlib.sha256()
    seen = 0
    if _identity(_regular_path(path, directory=False)) != initial:
        raise UnsafeBundle('bundle file identity changed before inventory SHA')
    try:
        with open(path, 'rb') as handle:
            if _identity(os.fstat(handle.fileno())) != initial:
                raise UnsafeBundle('bundle file identity changed at inventory SHA open')
            while True:
                block = handle.read(READ_CHUNK)
                if not block:
                    break
                seen += len(block)
                if seen > info.st_size:
                    raise UnsafeBundle('bundle file grew during inventory SHA')
                measured.update(block)
            if seen != info.st_size or _identity(os.fstat(handle.fileno())) != initial:
                raise UnsafeBundle('bundle file identity changed during inventory SHA')
    except OSError as exc:
        raise UnsafeBundle('bundle file could not be hashed at inventory') from exc
    if _identity(_regular_path(path, directory=False)) != initial:
        raise UnsafeBundle('bundle file identity changed after inventory SHA')
    return measured.hexdigest()


def _bounded_pascal_file(entry):
    _, path, size = entry
    if not 0 < size <= _MAX_PASSPORT_BYTES:
        raise UnsafeBundleIntegrity('Pascal manifest/passport exceeds safe byte limit')
    identity = getattr(entry, 'identity', None)
    if identity is not None and _identity(
            _regular_path(path, directory=False)) != identity:
        raise UnsafeBundle('Pascal manifest changed after bundle inventory')
    try:
        with open(path, 'rb') as handle:
            before = os.fstat(handle.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_size != size or
                    (identity is not None and _identity(before) != identity)):
                raise UnsafeBundle('Pascal manifest changed before validation')
            payload = handle.read(_MAX_PASSPORT_BYTES + 1)
            after = os.fstat(handle.fileno())
            if len(payload) != size or _identity(before) != _identity(after):
                raise UnsafeBundle('Pascal manifest changed during validation')
    except OSError as exc:
        raise UnsafeBundle('Pascal manifest cannot be read safely') from exc
    if identity is not None and _identity(
            _regular_path(path, directory=False)) != identity:
        raise UnsafeBundle('Pascal manifest path changed during validation')
    pinned_sha = getattr(entry, 'sha256', None)
    if pinned_sha is not None and hashlib.sha256(payload).hexdigest() != pinned_sha:
        raise UnsafeBundle('Pascal manifest content identity changed '
                           'since bundle inventory')
    return payload


def _hash_pascal_artifact(entry):
    _, path, expected_bytes = entry
    identity = getattr(entry, 'identity', None)
    if identity is not None and _identity(
            _regular_path(path, directory=False)) != identity:
        raise UnsafeBundle('Pascal artifact identity changed before SHA check')
    digest = hashlib.sha256()
    count = 0
    try:
        with open(path, 'rb') as stream:
            first = os.fstat(stream.fileno())
            if (not stat.S_ISREG(first.st_mode) or
                    first.st_size != expected_bytes or
                    (identity is not None and _identity(first) != identity)):
                raise UnsafeBundle('Pascal artifact changed before SHA check')
            remaining = expected_bytes
            while remaining:
                part = stream.read(min(READ_CHUNK, remaining))
                if not part or len(part) > remaining:
                    raise UnsafeBundle('Pascal artifact identity changed '
                                       'during bounded SHA read')
                digest.update(part)
                count += len(part)
                remaining -= len(part)
            if stream.read(1):
                raise UnsafeBundle('Pascal artifact grew beyond inventoried '
                                   'byte size during SHA check')
            if count != expected_bytes or _identity(
                    os.fstat(stream.fileno())) != _identity(first):
                raise UnsafeBundle('Pascal artifact changed during SHA check')
    except OSError as exc:
        raise UnsafeBundle('Pascal artifact unavailable during SHA check') from exc
    if identity is not None and _identity(
            _regular_path(path, directory=False)) != identity:
        raise UnsafeBundle('Pascal artifact path changed during SHA check')
    pinned_sha = getattr(entry, 'sha256', None)
    if pinned_sha is not None and digest.hexdigest() != pinned_sha:
        raise UnsafeBundle('Pascal artifact content identity changed '
                           'since bundle inventory')
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise UnsafeBundle('Pascal manifest JSON has duplicate key: %s' % name)
        result[name] = value
    return result


def _reject_json_constant(value):
    raise UnsafeBundle('Pascal manifest JSON contains nonfinite value: %s' % value)


def _validate_pascal_passport_pair(entries):
    pair = {entry.name: entry for entry in entries if entry.name in _PASSPORT_PAIR}
    if set(pair) != _PASSPORT_PAIR:
        raise UnsafeBundleIntegrity('Pascal manifest pair is incomplete')
    raw_json = _bounded_pascal_file(pair['manifest.json'])
    raw_text = _bounded_pascal_file(pair['manifest.txt'])
    try:
        contract = json.loads(raw_json.decode('utf-8'),
                              object_pairs_hook=_unique_object,
                              parse_constant=_reject_json_constant)
        if not isinstance(contract, dict):
            raise ValueError('manifest must contain a JSON object')
        scalar_fields = [contract.get('task_kind'), contract.get('adapter'),
                         contract.get('license')]
        for section in ('artifact', 'io', 'output', 'provenance', 'toolchain',
                        'tokenizer'):
            row = contract.get(section)
            if isinstance(row, dict):
                scalar_fields.extend(value for value in row.values()
                                     if isinstance(value, str))
        if any(isinstance(value, str) and any(c in value for c in '\r\n\x00')
               for value in scalar_fields):
            raise ValueError('Pascal line delimiter inside manifest scalar')
        artifact = contract.get('artifact')
        filename = artifact.get('file') if isinstance(artifact, dict) else None
        if (not isinstance(filename, str) or not filename or
                any(ch in filename for ch in ('/', '\\', ':', '\x00')) or
                filename.endswith(('.', ' '))):
            raise ValueError('Pascal artifact file is not a safe basename')
        if artifact.get('hash') is not None and (
                not isinstance(artifact['hash'], str) or
                not _HEX_SHA256.fullmatch(artifact['hash'])):
            raise ValueError('Pascal artifact hash has invalid SHA-256 syntax')
        lf = emit_manifest_text(contract, newline='\n').encode('utf-8')
        crlf = emit_manifest_text(contract, newline='\r\n').encode('utf-8')
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError,
            RecursionError) as exc:
        raise UnsafeBundleIntegrity('Pascal manifest JSON/text is malformed') from exc
    if raw_text not in (lf, crlf):
        raise UnsafeBundleIntegrity('Pascal manifest JSON/text passport disagreement')
    declared_hash = artifact.get('hash')
    if declared_hash is not None:
        by_name = {entry.name: entry for entry in entries}
        if filename not in by_name:
            raise UnsafeBundleIntegrity('Pascal manifest artifact is missing from bundle')
        if _hash_pascal_artifact(by_name[filename]) != declared_hash.lower():
            raise UnsafeBundleIntegrity('Pascal manifest artifact SHA-256 mismatch')
    return contract


def _require_vision_bundle_closure(entries, contract):
    if contract.get('task_kind') == 'text_generation':
        _require_native_llm_bundle_closure(entries)
        return
    if not isinstance(contract.get('io'), dict):
        raise UnsafeBundleIntegrity('Pascal vision input contract is missing from passport')
    names = {entry.name: entry for entry in entries if '/' not in entry.name}
    artifact = contract['artifact']
    filename = artifact['file']
    required = [filename, filename + '.inputs', 'recipe.json', 'report.txt']
    if artifact.get('os') == 'windows':
        if artifact.get('device') not in ('cpu', 'cuda'):
            raise UnsafeBundleIntegrity('Pascal vision bundle has unsupported device')
        if not filename.lower().endswith('.dll'):
            raise UnsafeBundleIntegrity('Windows Pascal vision model has no DLL artifact')
        required += ['tvm_runtime.dll', 'tvm_runtime_extra.dll', 'tvm_ffi.dll']
        if artifact['device'] == 'cuda':
            required.append('tvm_runtime_cuda.dll')
    for name in required:
        entry = names.get(name)
        if entry is None or entry.size <= 0:
            raise UnsafeBundleIntegrity(
                'incomplete Pascal vision bundle: missing/empty required file %s' % name)
    if not any(name.startswith('LICENSE-tvm-') and entry.size > 0
               for name, entry in names.items()):
        raise UnsafeBundleIntegrity('incomplete Pascal vision bundle: missing TVM licence')


def _native_llm_closure_transient_read_failure(error, approved_sizes=None):
    seen = set()
    current = error
    for _ in range(16):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, OSError):
            return True
        if type(current) is ValueError:
            message = str(current)
            if (message.startswith('native BuildPlan changed while reading')
                    or message == 'native BuildPlan content changed since initial ZIP inventory'
                    or message == 'native BuildPlan must be a regular file <= 1MiB'
                    or message.startswith('native LLM library changed during SHA validation')
                    or message.startswith('native LLM library changed before SHA validation')
                    or message.startswith('native LLM library changed during bounded SHA validation')
                    or message.startswith('native LLM library grew beyond admitted byte size during SHA validation')
                    or message.startswith('native LLM library changed since initial ZIP inventory')
                    or message.startswith('native LLM params.bin changed since initial ZIP inventory')
                    or (message.startswith('native LLM output ') and
                        message.endswith(' changed while reading'))
                    or (message.startswith('native LLM output ') and
                        message.endswith(' changed since initial ZIP inventory'))
                    or (message.startswith('native LLM output ') and
                        message.endswith(' is linked or not a regular file'))
                    or message in (
                        'native LLM output directory is a link/junction',
                        'native LLM output directory redirects through a link',
                        'native LLM output directory is unavailable',
                        'linked native BuildPlan transfer file is untrusted',

                        'native LLM licence closure incomplete: no TVM licence',
                    )):
                return True

            suffix = ' is empty or too large'
            prefix = 'native LLM output '
            if message.startswith(prefix) and message.endswith(suffix):
                name = message[len(prefix):-len(suffix)]
                if (approved_sizes is not None and
                        type(approved_sizes.get(name)) is int and
                        approved_sizes[name] > 0):
                    return True
        current = current.__cause__
    return False


def _require_native_llm_bundle_closure(entries):
    from .build_plan import read_native_plan
    from .llm_bundle_contract import (_FAMILIES, _RUNTIME, _MAX_JSON,
                                      _MAX_INDEX, validate_llm_output)
    from .recipe_file import validate_family_license_receipt
    by_name = {entry.name: entry for entry in entries}
    required = ('model.json', 'params.json', 'params.index', 'params.bin',
                'recipe.json', 'build-plan.json', 'report.txt',
                'import-provenance.json')
    for name in required:
        if name not in by_name or by_name[name].size <= 0:
            raise UnsafeBundleIntegrity(
                'incomplete native LLM bundle: missing/empty %s' % name)
    try:
        receipt = json.loads(
            _bounded_pascal_file(by_name['recipe.json']).decode('utf-8'),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant)
        if not isinstance(receipt, dict):
            raise ValueError('family receipt must be an object')
        family = receipt.get('family_builder')
        if (type(receipt.get('recipe_version')) is not int or
                receipt['recipe_version'] not in (2, 3) or
                not isinstance(family, dict) or family.get('adapter') != 'llm' or
                not isinstance(family.get('model_key'), str)):
            raise ValueError('native LLM recipe lacks exact family identity')
        validate_family_license_receipt(receipt)

        if by_name['build-plan.json'].size > _MAX_LLM_BUILDPLAN_BYTES:
            raise UnsafeBundleIntegrity(
                'native LLM BuildPlan exceeds initial selected 1 MiB quota')
        plan = read_native_plan(
            by_name['build-plan.json'].path,
            expected_sha256=by_name['build-plan.json'].sha256)
        if (receipt.get('build_plan') != plan or
                receipt.get('source') != plan.get('source') or
                receipt.get('target_profile') != plan.get('target_profile') or
                family.get('dtype') != plan['options']['dtype']):
            raise ValueError('native LLM recipe and BuildPlan disagree')

        model_key = family['model_key']
        if model_key not in _FAMILIES:
            raise ValueError('native LLM recipe selected unknown model family')
        device = plan['target_profile']['deployment']['device']
        stem, _, _, tokenizer_files = _FAMILIES[model_key]
        expected_consumer = {
            '%s-%s.dll' % (stem, device), *tokenizer_files, *_RUNTIME,
        }
        if device == 'cuda':
            expected_consumer.add('tvm_runtime_cuda.dll')
        for selected_name in sorted(expected_consumer):
            member = by_name.get(selected_name)
            if member is None or member.size <= 0:
                raise UnsafeBundleIntegrity(
                    'native LLM selected consumer missing/empty at initial ZIP '
                    'inventory: %s' % selected_name)
        if not any(name.startswith('LICENSE-tvm-') and member.size > 0
                   for name, member in by_name.items() if '/' not in name):
            raise UnsafeBundleIntegrity(
                'native LLM selected consumer has no TVM licence at inventory')

        admitted_small_files = {
            'model.json': _MAX_JSON, 'params.json': _MAX_JSON,
            'manifest.json': _MAX_JSON, 'manifest.txt': _MAX_JSON,
            'params.index': _MAX_INDEX,
        }
        for name, limit in admitted_small_files.items():
            if by_name[name].size > limit:
                raise UnsafeBundleIntegrity(
                    'native LLM selected metadata exceeds initial byte quota: '
                    + name)
        report = _bounded_pascal_file(by_name['report.txt'])
        if b'NOT_CHECKED' not in report:
            raise ValueError('native LLM report lacks honest NOT_CHECKED state')
        root = os.path.dirname(by_name['manifest.json'].path)
        validate_llm_output(root, plan, family['model_key'],
                            require_consumer=True,
                            approved_library_size=by_name[
                                '%s-%s.dll' % (stem, device)].size,
                            approved_params_size=by_name['params.bin'].size,
                            approved_member_sha256s={
                                name: member.sha256
                                for name, member in by_name.items()})
    except UnsafeBundle:
        raise
    except OSError as exc:
        raise UnsafeBundle('native LLM ZIP closure temporarily inaccessible: %s'
                           % exc) from exc
    except (ValueError, TypeError, KeyError, UnicodeError,
            RecursionError) as exc:
        if _native_llm_closure_transient_read_failure(
                exc, {name: member.size for name, member in by_name.items()}):
            raise UnsafeBundle(
                'native LLM ZIP closure changed or was unavailable while reading: %s'
                % exc) from exc
        raise UnsafeBundleIntegrity(
            'incomplete/incorrect native LLM ZIP closure: %s' % exc) from exc


def _validate_selected_source_receipt(entries, expected_tree_sha256=None,
                                      expected_closure=None):
    from .recipe_file import validate_selected_source_closure

    names = {entry.name: entry for entry in entries if '/' not in entry.name}
    recipe_entry = names.get('recipe.json')
    seal_entry = names.get(_SOURCE_CLOSURE_RECEIPT)
    selected = (seal_entry is not None or
                expected_tree_sha256 is not None or expected_closure is not None)
    recipe = None
    if recipe_entry is not None and 0 < recipe_entry.size <= _MAX_PASSPORT_BYTES:
        try:
            recipe_bytes = _bounded_pascal_file(recipe_entry)
        except UnsafeBundle:
            if selected:
                raise
            recipe_bytes = None
        if recipe_bytes is not None:
            try:
                recipe = json.loads(recipe_bytes.decode('utf-8'),
                                object_pairs_hook=_unique_object,
                                parse_constant=_reject_json_constant)
            except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
                if selected:
                    raise UnsafeBundleIntegrity(
                        'selected source closure recipe JSON is malformed') from exc


    elif selected:
        raise UnsafeBundleIntegrity(
            'selected source closure recipe missing or oversized')
    declared = recipe.get('source_closure') if isinstance(recipe, dict) else None
    if (declared is None and seal_entry is None and
            expected_tree_sha256 is None and expected_closure is None):
        return
    if declared is None or seal_entry is None:
        raise UnsafeBundleIntegrity(
            'selected ONNX source closure has no paired recipe and post-lock receipt')
    try:
        closure = validate_selected_source_closure(declared)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise UnsafeBundleIntegrity(
            'invalid selected source closure recipe schema: %s' % exc) from exc
    if (type(recipe.get('recipe_version')) is not int or
            recipe['recipe_version'] != 2 or
            recipe.get('source') != closure['source_model']):
        raise UnsafeBundleIntegrity(
            'selected ONNX closure recipe version/original source differs '
            'from approved source model')


    seal_bytes = _bounded_pascal_file(seal_entry)
    try:
        seal = json.loads(seal_bytes.decode('utf-8'),
                          object_pairs_hook=_unique_object,
                          parse_constant=_reject_json_constant)
        if (not isinstance(seal, dict) or set(seal) != {
                'schema', 'schema_version', 'source_closure', 'source_path',
                'stage_lock_status', 'status', 'held_through_pipeline_verify',
                'release_grade_immutable'} or
                seal['schema'] != 'crosspascal.onnx-selected-closure-receipt' or
                type(seal['schema_version']) is not int or
                seal['schema_version'] != 1 or
                seal['source_closure'] != closure or
                seal['source_path'] != recipe.get('source') or
                seal['source_path'] != closure['source_model'] or
                seal['stage_lock_status'] !=
                'WIN32_SELECTED_FILE_SET_CANDIDATE_NOT_RELEASE_GRADE' or
                seal['status'] != 'POST_LOCK_AUDIT_PASSED_NOT_RELEASE_GRADE' or
                seal['held_through_pipeline_verify'] is not True or
                seal['release_grade_immutable'] is not False or
                (expected_tree_sha256 is not None and
                 closure['tree_sha256'] != expected_tree_sha256)):
            raise UnsafeBundleIntegrity(
                'selected source closure receipt mismatches '
                'job-approved tree/recipe/post-lock state')
        if any(name in names for name in _PASSPORT_PAIR):
            raise UnsafeBundleIntegrity(
                'selected unclassified RAW external_data cannot '
                'claim a Pascal verified manifest passport')
        if expected_closure is not None:
            expected_fields = {
                'schema': closure['closure_schema'],
                'version': closure['source_closure_version'],

                'source_model': closure['source_model'],
                'model_rel': closure['model_rel'],
                'tree_sha256': closure['tree_sha256'],
                'dependencies': closure['dependencies'],
                'entries': closure['entries'],
                'references': closure['references'],
                'total_bytes': closure['total_bytes'],
            }
            if expected_fields != expected_closure:
                raise UnsafeBundleIntegrity(
                    'selected source closure differs from '
                    'in-memory post-audited job record')
    except UnsafeBundleIntegrity:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError,
            RecursionError) as exc:
        raise UnsafeBundleIntegrity(
            'invalid selected source closure receipt JSON/schema: %s' % exc) from exc


def _validate_selected_paddle_rec_dictionary(
        entries, expected, *, expected_source_root=None,
        expected_build_plan=None):
    if expected is None:
        return
    if (type(expected_source_root) is not str or
            not os.path.isabs(expected_source_root) or
            type(expected_build_plan) is not dict or
            expected_build_plan.get('route') != 'raw' or
            type(expected_build_plan.get('target_profile')) is not dict or
            type(expected_build_plan['target_profile'].get('deployment')) is not dict):
        raise UnsafeBundleIntegrity(
            'selected Rec ZIP lacks independent in-memory owner/BuildPlan approval')
    if (type(expected) is not dict or
            expected.get('schema') !=
            'crosspascal.paddle-selected-rec-consumer-dictionary.v1' or
            expected.get('status') !=
            'POST_WIN32_HELD_SELECTED_REC_DICTIONARY_NOT_MODEL_UAT' or
            expected.get('held_through_pipeline_run') is not True or
            expected.get('verification_status') != 'NOT_CHECKED' or
            expected.get('native_pascal_ctc') != 'NOT_RUN' or
            expected.get('release_grade_immutable') is not False or
            type(expected.get('bytes')) is not int or
            not 0 < expected['bytes'] <= 8 * 1024 * 1024 or
            type(expected.get('sha256')) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', expected['sha256']) or
            type(expected.get('source_tree_sha256')) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', expected['source_tree_sha256'])):
        raise UnsafeBundleIntegrity(
            'selected Paddle Rec consumer job receipt lacks post-lock approval')
    path = expected.get('relative_path')
    if (type(path) is not str or not path.startswith('dictionary/') or
            path.count('/') != 1 or not path[len('dictionary/'):] or
            path.endswith(('.', ' ')) or '\\' in path):
        raise UnsafeBundleIntegrity('invalid selected Rec dictionary ZIP member')
    selected = {entry.name: entry for entry in entries}
    member = selected.get(path)
    provenance = selected.get('import-provenance.json')
    recipe_entry = selected.get('recipe.json')
    if (member is None or provenance is None or
            recipe_entry is None or
            member.size != expected['bytes'] or
            member.sha256 != expected['sha256']):
        raise UnsafeBundleIntegrity(
            'actual selected Paddle Rec dictionary ZIP bytes differ from job approval')
    if (not 0 < provenance.size <= _MAX_PASSPORT_BYTES or
            not 0 < recipe_entry.size <= _MAX_PASSPORT_BYTES):
        raise UnsafeBundleIntegrity('selected Paddle recipe/provenance missing or oversized')
    payload = _bounded_pascal_file(provenance)
    recipe_bytes = _bounded_pascal_file(recipe_entry)
    try:
        from .recipe_file import validate_paddle_selected_source
        report = json.loads(payload.decode('utf-8'),
                            object_pairs_hook=_unique_object,
                            parse_constant=_reject_json_constant)
        recipe = json.loads(recipe_bytes.decode('utf-8'),
                            object_pairs_hook=_unique_object,
                            parse_constant=_reject_json_constant)
        if (type(recipe) is not dict or
                type(recipe.get('recipe_version')) is not int or
                recipe['recipe_version'] != 2 or
                type(recipe.get('build_plan')) is not dict or
                recipe['build_plan'].get('route') != 'raw' or
                type(recipe.get('source')) is not str or
                recipe['source'] != expected_source_root or
                recipe['build_plan'] != expected_build_plan or
                recipe.get('target_profile') !=
                    expected_build_plan['target_profile'] or
                recipe.get('target') !=
                    expected_build_plan['target_profile']['deployment'].get('device')):
            raise UnsafeBundleIntegrity('selected Paddle Rec recipe has no raw BuildPlan')
        approved = validate_paddle_selected_source(
            recipe.get('paddle_selected_source'), source=recipe['source'])
        original = report['selected_source_closure']
        declared = report['compiler_window_held_rec_dictionary']

        checked = validate_paddle_selected_source(
            original, source=recipe['source'])
        dictionary = original['dictionary']
        if (type(report) is not dict or type(original) is not dict or
                type(dictionary) is not dict or approved != checked or
                declared != expected or
                original.get('role') != 'rec' or
                approved.get('source_root') != expected_source_root or
                checked.get('source_root') != expected_source_root or
                original.get('tree_sha256') != expected['source_tree_sha256'] or
                dictionary.get('path') != path or
                dictionary.get('bytes') != member.size or
                dictionary.get('sha256') != member.sha256):
            raise UnsafeBundleIntegrity(
                'selected Paddle Rec dictionary provenance disagrees with job')
    except UnsafeBundleIntegrity:
        raise
    except (ValueError, KeyError, TypeError, UnicodeError, RecursionError) as error:
        raise UnsafeBundleIntegrity(
            'selected Paddle Rec consumer import provenance is invalid: %s' % error
        ) from error


def collect_bundle_files(root, *, allow_download_info=False,
                         expected_source_closure_sha256=None,
                         expected_source_closure=None,
                         expected_atomic_inventory=None,
                         expected_paddle_rec_dictionary=None,
                         expected_paddle_source_root=None,
                         expected_paddle_build_plan=None,
                         expected_onnx_source_pack=None):
    if not isinstance(root, (str, os.PathLike)) or not os.path.isabs(root):
        raise UnsafeBundle('bundle directory must be an absolute server path')
    root = os.path.abspath(root)
    if expected_atomic_inventory is not None:
        from . import bundle_atomic
        if not isinstance(expected_atomic_inventory, bundle_atomic.Inventory):
            raise UnsafeBundle('committed bundle has no trusted inventory object')
        try:
            max_bytes = (bundle_atomic.MAX_NATIVE_FAMILY_BYTES
                         if expected_atomic_inventory.total_bytes >
                         bundle_atomic.MAX_BYTES else bundle_atomic.MAX_BYTES)
            measured = bundle_atomic.scan_bundle_tree(root, max_bytes=max_bytes)
        except (OSError, ValueError) as exc:
            raise UnsafeBundle('committed bundle is inaccessible or unsafe') from exc
        if measured != expected_atomic_inventory:
            raise UnsafeBundle('committed bundle inventory SHA/members changed')
    if os.path.normcase(os.path.realpath(root)) != os.path.normcase(root):
        if expected_atomic_inventory is None:
            raise UnsafeBundleIntegrity('bundle directory resolves through a link')
        raise UnsafeBundle('committed bundle directory resolves through a link')
    ancestor = root
    while True:
        _regular_path(ancestor, directory=True,
                      structural_admission=(expected_atomic_inventory is None))
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            break
        ancestor = parent

    trusted_member_hashes = (None if expected_atomic_inventory is None else
                             {member.path: member.sha256
                              for member in expected_atomic_inventory.members
                              if member.kind == 'file'})
    entries = []
    for archive_name, full, info in _bounded_zip_members(
            root, allow_download_info=allow_download_info):
        if trusted_member_hashes is None:
            content_sha = _hash_inventory_member(full, info)
        else:
            content_sha = trusted_member_hashes.get(archive_name)
            if content_sha is None:
                raise UnsafeBundle(
                    'bundle file is absent from trusted atomic inventory')
        entries.append(_CollectedMember(archive_name, full, info,
                                        content_sha))

    present = {name for name, _, _ in entries if name in _PASSPORT_PAIR}
    _validate_selected_source_receipt(
        entries, expected_tree_sha256=expected_source_closure_sha256,
        expected_closure=expected_source_closure)
    _validate_selected_paddle_rec_dictionary(
        entries, expected_paddle_rec_dictionary,
        expected_source_root=expected_paddle_source_root,
        expected_build_plan=expected_paddle_build_plan)

    if not present and any(entry.name == 'build-plan.json' for entry in entries):
        raise UnsafeBundleIntegrity(
            'native family BuildPlan has no manifest/passport pair')
    if present and present != _PASSPORT_PAIR:
        raise UnsafeBundleIntegrity('bundle has an incomplete Pascal manifest pair')
    if present and not allow_download_info:
        passport = _validate_pascal_passport_pair(entries)
        _require_vision_bundle_closure(entries, passport)
    if expected_onnx_source_pack is not None:
        try:
            from .onnx_source_pack import validate_collected
            passport = _validate_pascal_passport_pair(entries)
            validate_collected(entries, expected_onnx_source_pack, passport)
        except (ValueError, TypeError, KeyError) as exc:
            raise UnsafeBundleIntegrity('Source-pack download binding failed: ' + str(exc)) from exc
    if not entries or not any(size > 0 for _, _, size in entries):
        raise UnsafeBundleIntegrity(
            'bundle directory contains no nonempty regular files')
    if expected_atomic_inventory is not None:
        expected_files = {member.path for member in expected_atomic_inventory.members
                          if member.kind == 'file'}
        if {entry.name for entry in entries} != expected_files:
            raise UnsafeBundle('ZIP members differ from committed file inventory')
    return entries


def stream_bundle(destination, entries, *, job, verification_status,
                  expected_atomic_inventory=None):
    if verification_status not in ('VERIFIED_LOCAL', 'NOT_CHECKED'):
        raise UnsafeBundle('this job did not finish in an exportable state')
    expected_files = None
    if expected_atomic_inventory is not None:
        from . import bundle_atomic
        if not isinstance(expected_atomic_inventory, bundle_atomic.Inventory):
            raise UnsafeBundle('ZIP has no trusted atomic file inventory')
        expected_files = {member.path: member for member in
                          expected_atomic_inventory.members if member.kind == 'file'}
        if {entry.name for entry in entries} != set(expected_files):
            raise UnsafeBundle('ZIP input names differ from committed inventory')
    hashes = {}
    with zipfile.ZipFile(destination, mode='w', compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        for entry in entries:
            archive_name, full, expected_size = entry

            scanned = getattr(entry, 'identity', None)
            pinned_sha = getattr(entry, 'sha256', None)
            if (type(scanned) is not tuple or len(scanned) != 4
                    or type(pinned_sha) is not str
                    or not _HEX_SHA256.fullmatch(pinned_sha)):
                raise UnsafeBundle('ZIP source lacks pinned identity/SHA '
                                   'from trusted file inventory')
            if _identity(_regular_path(full, directory=False)) != scanned:
                raise UnsafeBundle('bundle file identity changed after inventory')
            digest = hashlib.sha256()
            actual_size = 0
            with open(full, 'rb') as source:
                info = os.fstat(source.fileno())
                if (not stat.S_ISREG(info.st_mode) or
                        info.st_size != expected_size or
                        (scanned is not None and _identity(info) != scanned)):
                    raise UnsafeBundle('bundle file changed before streaming')
                with archive.open(archive_name, 'w', force_zip64=True) as target:
                    remaining = expected_size
                    while remaining:
                        block = source.read(min(READ_CHUNK, remaining))
                        if not block or len(block) > remaining:
                            raise UnsafeBundle(
                                'bundle file identity changed during bounded ZIP read')
                        target.write(block)
                        digest.update(block)
                        actual_size += len(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise UnsafeBundle('bundle file grew beyond '
                                           'inventoried ZIP byte size')
                if scanned is not None and _identity(os.fstat(
                        source.fileno())) != scanned:
                    raise UnsafeBundle('bundle file mutated while streaming')
            if actual_size != expected_size:
                raise UnsafeBundle('bundle file changed while streaming')
            if expected_files is not None:
                expected = expected_files[archive_name]
                if (actual_size != expected.size or
                        digest.hexdigest() != expected.sha256):
                    raise UnsafeBundle('streamed bytes differ from committed SHA: '
                                       + archive_name)
            if digest.hexdigest() != pinned_sha:
                raise UnsafeBundle('bundle file identity changed (content SHA-256) '
                                   'after inventory: ' + archive_name)
            hashes[archive_name] = {'bytes': actual_size,
                                    'sha256': digest.hexdigest()}
        description = {
            'schema': 1,
            'job': str(job),
            'verification_status': verification_status,
            'independent_verification_by_downloader': False,
            'files': hashes,
        }
        archive.writestr(INFO_FILE, json.dumps(description, ensure_ascii=False,
                                               indent=2, sort_keys=True) + '\n')
