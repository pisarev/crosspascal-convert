# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Put a matching runtime next to the compiled module.

WHY. Whoever pressed the button installed nothing - that is the whole point. So
they have nothing to run the module with either: TVM's runtime is not on their
machine. Handing over the module alone is handing over half the job.

WHY THIS RUNTIME AND NOT ANY COMPATIBLE ONE. A module built by one version and
run under another crashes at address zero. That is not a worry, it is a
measurement: on 31.08.2026 rtdetr did exactly that - same model, same code, the
only difference being that the run happened outside the layout holding the right
libraries. The one way to promise agreement is to take the libraries from the
SAME wheel that did the building.

AND THAT MATCH IS MEASURED, NOT ASSUMED. The working layout of crosspascal
(model-zoo/run026) was compared with the apache-tvm 0.26.0 wheel by md5:

    tvm_runtime        0b51491e0d1bea07ef950a1b03d72aa9   identical
    tvm_ffi            e4984aee5fdc677f8ee22fed8d61587b   identical
    tvm_runtime_extra  300ccbc36211c8827e8a216a139a7685   identical

So the live checks run on precisely the libraries this module copies.

THE COMPILER IS LEFT OUT ON PURPOSE. tvm_compiler is 169 MB and is not needed to
run anything. Hence a LIST OF NAMES rather than a wildcard: a wildcard would drag
it along, and every model would cost the reader an extra hundred megabytes.

A MISSING LIBRARY IS A REFUSAL, NOT A WARNING. An incomplete set copied quietly
produces an artifact that will not start, and the person who finds that out is
the user, not us.
"""

import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

from .report import Step

EXTENSIONS = {
    'win32': '.dll',
    'darwin': '.dylib',
}
DEFAULT_EXTENSION = '.so'

REQUIRED = ('tvm_runtime', 'tvm_runtime_extra')
REQUIRED_FFI = ('tvm_ffi',)
FOR_GPU = ('tvm_runtime_cuda',)
_RUNTIME_COMPONENTS = REQUIRED + REQUIRED_FFI + FOR_GPU




_CUDA_DEPENDENCY = re.compile(
    r'^(?:lib)?(?:cudart|cudnn|cublaslt|cublas|cufft|curand|'
    r'cusolver|cusparse|cupti|nvrtc|nvjitlink|nvptxcompiler|'
    r'nccl|nvcuda|cuda)[\w.-]*\.(?:dll|dylib|so(?:\.\d+)*)$', re.I)


def _file_info(path):
    """One file's byte count and SHA-256, read without loading it into RAM."""
    digest = hashlib.sha256()
    count = 0
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            count += len(block)
            digest.update(block)
    return {'bytes': count, 'sha256': digest.hexdigest()}


def _source_record(path, filename):
    return dict(file=filename, source=path, **_file_info(path))


def _matching_libraries(directory, name, suffix):
    """Find both known filename spellings; aliases must be byte-identical."""
    if not directory:
        return [], None
    matches = []
    for filename in (name + suffix, 'lib' + name + suffix):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            matches.append(_source_record(candidate, filename))
    if len(matches) > 1 and any(
            row['sha256'] != matches[0]['sha256'] for row in matches[1:]):
        return [], ('ambiguous runtime library %s: its prefixed and unprefixed '
                    'copies have different hashes' % name)
    return matches, None


def _wheel_licences(tvm_dir):
    """Flatten wheel licence filenames only when doing so is unambiguous.

    The established bundle format is LICENSE-tvm-<basename>. A nested licence
    sharing a basename with different contents cannot be represented by that
    format. Refusing avoids silently dropping a legally required notice.
    """
    if not tvm_dir:
        return [], ['Apache-2.0 licence from installed TVM wheel']
    site = os.path.dirname(os.path.dirname(tvm_dir))
    items = {}
    missing = []
    try:
        distribution_entries = sorted(os.listdir(site))
    except OSError as exc:
        return [], ['cannot inspect TVM wheel licences: %s' % exc]
    for distribution in distribution_entries:
        if not distribution.startswith('apache_tvm-'):
            continue
        licence_dir = os.path.join(site, distribution, 'licenses')
        if not os.path.isdir(licence_dir):
            continue
        for root, folders, files in os.walk(licence_dir):
            folders.sort()
            for name in sorted(files):
                source = os.path.join(root, name)
                destination = 'LICENSE-tvm-' + name
                try:
                    record = _source_record(source, destination)
                except OSError as exc:
                    missing.append('unreadable TVM licence %s: %s' % (name, exc))
                    continue
                if record['bytes'] == 0:
                    missing.append('empty TVM licence %s: redistribution terms '
                                   'cannot be verified' % destination)
                    continue
                previous = items.get(destination)
                if previous and previous['sha256'] != record['sha256']:
                    missing.append('ambiguous TVM licence %s: multiple wheel '
                                   'files would overwrite the same bundle name'
                                   % destination)
                elif previous is None:
                    items[destination] = record
    if not items:
        missing.append('Apache-2.0 licence from installed TVM wheel')
    return list(items.values()), missing


def _public_record(record):
    """Omit the build machine's internal source path from an inventory preview."""
    return {field: record[field] for field in ('file', 'bytes', 'sha256')}


def _destination_status(destination, expected):
    """Return reason if a destination is unsafe, otherwise None.

    lexists catches broken symlinks that exists() would let us follow into a
    different location; exclusive create is also used at the actual write.
    """
    if not os.path.lexists(destination):
        return None
    if os.path.islink(destination):
        return 'runtime destination is a symbolic link: %s' % destination
    if not os.path.isfile(destination):
        return 'runtime destination is not a file: %s' % destination
    try:
        actual = _file_info(destination)
    except OSError as exc:
        return 'cannot inspect existing runtime %s: %s' % (destination, exc)
    if (actual['bytes'] != expected['bytes'] or
            actual['sha256'] != expected['sha256']):
        item = os.path.basename(destination)
        kind = ('TVM licence' if item.startswith('LICENSE-tvm-')
                else 'runtime')
        return ('existing %s differs from the current wheel: %s. Choose '
                'another output directory to preserve the previous bundle.'
                % (kind, item))
    return None


def _copy_exclusive(source, destination, expected_hash):
    """Copy fully attested wheel bytes, then publish them atomically no-replace.

    Opening the FINAL filename with ``xb`` prevents clobbering but exposes a
    partially copied DLL/licence while the copy is still running. A disk-full
    or interrupted process could leave that trusted filename truncated. Stage
    under our own unique sibling name, verify SHA of both copied and reread
    bytes, then publish one hardlink (same filesystem, exclusive name).
    Failure to hardlink is a refusal; never fall back to an overwriting move.
    """
    parent = Path(destination).absolute().parent
    if not parent.is_dir():
        raise OSError('runtime target folder is missing: %s' % parent)
    for ancestor in (parent, *parent.parents):
        if ancestor.is_symlink() or getattr(ancestor, 'is_junction', lambda: False)():
            raise OSError('runtime target has a linked/junction parent: %s'
                          % ancestor)
    if os.path.lexists(destination):
        raise FileExistsError('runtime destination already exists: %s' % destination)
    scratch = None
    try:
        with open(source, 'rb') as origin, tempfile.NamedTemporaryFile(
                mode='wb', prefix='.tvm-runtime-', suffix='.part',
                dir=parent, delete=False) as staged:
            scratch = Path(staged.name)
            digest = hashlib.sha256()
            length = 0
            for chunk in iter(lambda: origin.read(1048576), b''):
                staged.write(chunk)
                digest.update(chunk)
                length += len(chunk)
        if length < 1 or digest.hexdigest() != expected_hash:
            raise OSError('the selected wheel bytes changed while copying: %s'
                          % os.path.basename(source))
        result = _file_info(scratch)
        if result['bytes'] != length or result['sha256'] != expected_hash:
            raise OSError('staged TVM runtime bytes failed readback integrity: %s'
                          % os.path.basename(source))
        os.link(scratch, destination)
    finally:
        
        
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def library_extension():
    return EXTENSIONS.get(sys.platform, DEFAULT_EXTENSION)


def package_lib_dir(package_name):
    """The lib directory of an installed package, or None.

    The package is asked by import rather than searched for on disk: what
    matters is the one that will actually be used, not a namesake sitting in a
    neighbouring environment.
    """
    try:
        package = __import__(package_name)
    except ImportError:
        return None
    root = os.path.dirname(os.path.abspath(package.__file__))
    lib = os.path.join(root, 'lib')
    return lib if os.path.isdir(lib) else None


def _inspect_destination(target_dir, libraries, licences, tvm_dir, ffi_dir):
    """Report every preexisting conflict before writing the first bundle byte."""
    blocked = []
    unmanaged = []
    reused = []
    required = {row['file'].lower(): row for row in libraries + licences}
    for row in libraries + licences:
        destination = os.path.join(target_dir, row['file'])
        reason = _destination_status(destination, row)
        if reason:
            blocked.append(reason)
        elif os.path.lexists(destination):
            reused.append(row['file'])

    
    
    
    
    try:
        present = sorted(os.listdir(target_dir))
    except OSError as exc:
        return ['cannot inspect destination: %s' % exc], [], reused
    suffix = library_extension()
    for filename in present:
        if filename.startswith('.tvm-runtime-'):
            blocked.append('incomplete previously interrupted TVM runtime '
                           'stage in output: %s; select a clean destination '
                           'without deleting previous evidence' % filename)
            continue
        if filename.lower() in required:
            continue
        if filename.startswith('LICENSE-tvm-'):
            blocked.append('untracked old TVM licence in output: %s' % filename)
            continue
        logical = None
        for name in _RUNTIME_COMPONENTS:
            if filename.lower() in ((name + suffix).lower(),
                                    ('lib' + name + suffix).lower()):
                logical = name
                break
        if logical:
            source_dir = ffi_dir if logical in REQUIRED_FFI else tvm_dir
            candidates, conflict = _matching_libraries(
                source_dir, logical, suffix)
            if conflict or not candidates:
                blocked.append('untracked runtime %s has no unambiguous matching '
                               'wheel library' % filename)
                continue
            reason = _destination_status(
                os.path.join(target_dir, filename), candidates[0])
            if reason:
                blocked.append(reason)
            else:
                reused.append(filename)
        elif _CUDA_DEPENDENCY.fullmatch(filename):
            unmanaged.append(filename)
            blocked.append(
                'unmanaged CUDA runtime dependency in output: %s; choose a fresh '
                'directory because this toolkit/driver match was not verified'
                % filename)
    return blocked, unmanaged, reused


def _plan_runtime_files(target='cpu', target_dir=None):
    """Return the public inventory and private source records for exact copying."""
    device = str(target).strip().lower()
    if device not in ('cpu', 'cuda'):
        raise ValueError('runtime target must be cpu or cuda')
    tvm_dir = package_lib_dir('tvm')
    ffi_dir = package_lib_dir('tvm_ffi')
    components = [(tvm_dir, name) for name in REQUIRED]
    components += [(ffi_dir, name) for name in REQUIRED_FFI]
    if device == 'cuda':
        components += [(tvm_dir, name) for name in FOR_GPU]
    suffix = library_extension()
    libraries = []
    missing = []
    for directory, name in components:
        if not directory:
            missing.append(name + suffix + ' (package not found)')
            continue
        try:
            matches, conflict = _matching_libraries(directory, name, suffix)
        except OSError as exc:
            missing.append('cannot read wheel runtime %s: %s' % (name, exc))
            continue
        if conflict:
            missing.append(conflict)
        elif not matches:
            missing.append(name + suffix)
        else:
            libraries.append(matches[0])
    licences, licence_errors = _wheel_licences(tvm_dir)
    missing.extend(licence_errors)

    blocked, unmanaged, reused = [], [], []
    if target_dir is not None:
        if not os.path.isdir(target_dir):
            blocked.append('no such output directory: %s' % target_dir)
        else:
            blocked, unmanaged, reused = _inspect_destination(
                target_dir, libraries, licences, tvm_dir, ffi_dir)
    inventory = {
        'ready': not (missing or blocked),
        'target': device,
        'libraries': [_public_record(row) for row in libraries],
        'licences': [row['file'] for row in licences],
        'licence_files': [_public_record(row) for row in licences],
        'missing': missing,
        'blocked_existing': blocked,
        'unmanaged_cuda_files': unmanaged,
        'existing_verified': reused,
        'bytes': sum(row['bytes'] for row in libraries),
        'external_cuda_dependencies': (
            'not fully inventoried; CUDA Toolkit/runtime, driver and the '
            'exported module must be checked on the destination'
            if device == 'cuda' else None),
        'portable_verified': False,
    }
    return inventory, libraries, licences


def plan_runtime_files(target='cpu', target_dir=None):
    """Read-only SHA-256 inventory of the selected wheel and optional output.

    The target_dir check also reports *unmanaged* libraries left by older
    bundles. A ready inventory verifies source bytes and copy suitability, NOT
    that the model is correct or portable to a foreign GPU/operating system.
    """
    inventory, _, _ = _plan_runtime_files(target, target_dir)
    return inventory


def copy_license(target_dir):
    """Copy licence files only if ALL are present and prechecked.

    Kept as a callable for existing integrations; runtime.place uses the same
    exclusive copy mechanism after it has preflighted libraries AND licences.
    """
    if not os.path.isdir(target_dir):
        raise OSError('no such output directory: %s' % target_dir)
    tvm_dir = package_lib_dir('tvm')
    if tvm_dir is None:
        return 0
    records, missing = _wheel_licences(tvm_dir)
    if missing:
        raise OSError('incomplete TVM licences: ' + ', '.join(missing))
    for row in records:
        destination = os.path.join(target_dir, row['file'])
        problem = _destination_status(destination, row)
        if problem:
            raise OSError(problem)
    for row in records:
        destination = os.path.join(target_dir, row['file'])
        if not os.path.lexists(destination):
            _copy_exclusive(row['source'], destination, row['sha256'])
        if _destination_status(destination, row):
            raise OSError('copied TVM licence hash differs from the wheel: ' +
                          row['file'])
    return len(records)


def place(target_dir, target='cpu'):
    """Put an exactly inventoried TVM runtime beside a module, without clobber.

    The preflight checks both libraries and licences before the first copy.
    This step reports matching wheel bytes only; CUDA driver/runtime viability
    and numerical model correctness remain separate verification gates.
    """
    step = Step('Runtime beside the module')
    if not os.path.isdir(target_dir):
        return step.refuse('no such output directory: %s' % target_dir)
    try:
        inventory, libraries, licences = _plan_runtime_files(
            target, target_dir=target_dir)
    except (OSError, ValueError) as exc:
        return step.refuse('cannot inspect runtime wheel: %s' % exc)
    if not inventory['ready']:
        reasons = inventory['missing'] + inventory['blocked_existing']
        return step.refuse('runtime inventory refused: ' + ', '.join(reasons))

    for entry in libraries + licences:
        filename = entry['file']
        destination = os.path.join(target_dir, filename)
        
        
        if not os.path.lexists(destination):
            try:
                _copy_exclusive(entry['source'], destination, entry['sha256'])
            except OSError as exc:
                return step.refuse('cannot copy runtime library or licence %s: %s'
                                   % (filename, exc))
        problem = _destination_status(destination, entry)
        if problem:
            return step.refuse('runtime checksum check failed: ' + problem)

    for entry in libraries:
        step.say('%-24s %6.1f MB sha256=%s'
                 % (entry['file'], entry['bytes'] / 1048576.0,
                    entry['sha256']))
    step.say('licences verified: %d' % len(licences))
    step.numbers['bytes'] = inventory['bytes']
    step.numbers['library_sha256'] = {
        row['file']: row['sha256'] for row in libraries}
    step.numbers['licence_sha256'] = {
        row['file']: row['sha256'] for row in licences}
    step.numbers['portable_verified'] = False
    if inventory['target'] == 'cuda':
        step.say('NOT CHECKED: the destination CUDA toolkit/runtime, driver '
                 'and compiled module still require independent verification')
    else:
        step.say('NOT CHECKED: portability to another OS/CPU/runtime has not '
                 'been independently verified')
    return step.done('%.1f MB of matching TVM runtime and licences; '
                     'destination compatibility NOT CHECKED'
                     % (inventory['bytes'] / 1048576.0))
