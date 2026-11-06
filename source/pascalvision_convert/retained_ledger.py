# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Selected Win32 retained-stage DPAPI receipt: read-only, NOT recovered READY.

This bounded building block writes only to an EXPLICIT already-created evidence
directory under the configured application evidence root. It does not
integrate with the wizard or automatically discover jobs.

DPAPI CurrentUser protects a receipt against accidental edits and other users,
NOT against a malicious process with the SAME Windows logon identity. FileIDs
are point-in-time observations; no atomic NTFS mkdir/handle or power-loss
transaction, persistent active-space lease, ownership recovery, immutable stage,
verification, READY, export, reuse, or deletion permission is established.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import hashlib
import json
import os
import re
import secrets
import stat
import threading
from pathlib import Path

from . import bundle_atomic
from . import app_paths


SCHEMA = 'crosspascal.retained-stage-dpapi-selected'
INTENT_SCHEMA = 'crosspascal.retained-stage-intent-dpapi-selected'
PUBLISHED_SCHEMA = 'crosspascal.retained-stage-namespace-observation'
PRECOMMIT_SCHEMA = 'crosspascal.retained-stage-precommit-inventory'
VERSION = 1
SUFFIX = '.retained.dpapi'
INTENT_SUFFIX = '.intent.dpapi'
PUBLISHED_SUFFIX = '.published.dpapi'
PRECOMMIT_SUFFIX = '.precommit.dpapi'
MAX_PLAINTEXT = 8192
MAX_PROTECTED = 32768
OWNER_EVIDENCE_ROOT = str(app_paths.data_path('evidence'))
_TOKEN = re.compile(r'[0-9a-f]{32}\Z')
_RECEIPT_LEAF = re.compile(r'([0-9a-f]{32})-(output|snapshot)\.retained\.dpapi\Z')
_INTENT_LEAF = re.compile(r'([0-9a-f]{32})-(output|snapshot)\.intent\.dpapi\Z')
_PUBLISHED_LEAF = re.compile(r'([0-9a-f]{32})-output\.published\.dpapi\Z')
_PRECOMMIT_LEAF = re.compile(r'([0-9a-f]{32})-output\.precommit\.dpapi\Z')
_SOURCE = re.compile(r'.+\.source-freeze-[0-9a-f]{24}\Z')
_ENTROPY = b'CrossPascal I-03 selected retained stage receipt v1'
_FLAGS = 0x1
_LOCAL_LEASES = threading.local()
_KEYS = frozenset({
    'schema', 'version', 'kind', 'job_nonce', 'evidence_directory',
    'evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
    'output_final', 'stage', 'parent_dev', 'parent_ino', 'parent_ctime_ns',
    'stage_dev', 'stage_ino', 'stage_ctime_ns',
})
_INTENT_KEYS = frozenset({
    'schema', 'version', 'kind', 'job_nonce', 'evidence_directory',
    'evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
    'output_final', 'stage', 'parent_dev', 'parent_ino', 'parent_ctime_ns',
})
_PUBLISHED_KEYS = frozenset({
    'schema', 'version', 'job_nonce', 'evidence_directory',
    'evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
    'output_final', 'original_stage',
    'parent_dev', 'parent_ino', 'parent_ctime_ns',
    'committed_dev', 'committed_ino', 'committed_ctime_ns',
    'bundle_tree_sha256',
})
_PRECOMMIT_KEYS = frozenset({
    'schema', 'version', 'job_nonce', 'evidence_directory',
    'evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
    'output_final', 'original_stage',
    'parent_dev', 'parent_ino', 'parent_ctime_ns',
    'stage_dev', 'stage_ino', 'stage_ctime_ns',
    'bundle_tree_sha256', 'bundle_member_count', 'bundle_total_bytes',
})


class UnsafeLedger(ValueError):
    """A refusal grants NO ownership, download, READY or cleanup authority."""


class _Blob(ctypes.Structure):
    _fields_ = [
        ('cbData', wintypes.DWORD),
        ('pbData', ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob(raw):
    if not raw:
        raise UnsafeLedger('DPAPI record cannot be empty')
    backing = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
    return _Blob(len(raw), ctypes.cast(backing, ctypes.POINTER(ctypes.c_ubyte))), backing


def _crypt(raw, *, protect):
    """Encrypt/decrypt with profile-scoped Windows DPAPI; free kernel memory."""
    if os.name != 'nt':
        raise UnsafeLedger('selected DPAPI retained ledger requires native Win32')
    limit = MAX_PLAINTEXT if protect else MAX_PROTECTED
    if not 0 < len(raw) <= limit:
        raise UnsafeLedger('invalid or oversized DPAPI retained record')
    incoming, incoming_keepalive = _blob(raw)
    entropy, entropy_keepalive = _blob(_ENTROPY)
    outgoing = _Blob()
    crypt32 = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel32.LocalFree.restype = ctypes.c_void_p
    if protect:
        call = crypt32.CryptProtectData
        call.argtypes = (ctypes.POINTER(_Blob), wintypes.LPCWSTR,
                         ctypes.POINTER(_Blob), ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(_Blob))
        call.restype = wintypes.BOOL
        arguments = (ctypes.byref(incoming), 'CrossPascal retained stage v1',
                     ctypes.byref(entropy), None, None, _FLAGS,
                     ctypes.byref(outgoing))
    else:
        call = crypt32.CryptUnprotectData
        call.argtypes = (ctypes.POINTER(_Blob), ctypes.c_void_p,
                         ctypes.POINTER(_Blob), ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(_Blob))
        call.restype = wintypes.BOOL
        arguments = (ctypes.byref(incoming), None, ctypes.byref(entropy),
                     None, None, _FLAGS, ctypes.byref(outgoing))
    try:
        if not call(*arguments):
            raise UnsafeLedger('DPAPI CurrentUser record cannot authenticate/decrypt')
        if (not outgoing.pbData or outgoing.cbData == 0 or
                outgoing.cbData > (MAX_PROTECTED if protect else MAX_PLAINTEXT)):
            raise UnsafeLedger('DPAPI output malformed or exceeds bounded receipt size')
        return ctypes.string_at(outgoing.pbData, outgoing.cbData)
    finally:

        _ = (incoming_keepalive, entropy_keepalive)
        if outgoing.pbData:
            kernel32.LocalFree(ctypes.cast(outgoing.pbData, ctypes.c_void_p))


def _absolute_directory(path, label):
    try:
        canonical = bundle_atomic._local_absolute(path)
        bundle_atomic._guard_existing_ancestors(canonical)
        bundle_atomic._require_ntfs(canonical)
        bundle_atomic._guard_stat(canonical, directory=True)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('unsafe local NTFS ' + label) from exc
    return canonical


def _approved_evidence_directory(path):
    """Require an existing run directory below the application evidence root."""
    evidence = _absolute_directory(path, 'evidence run')
    root = _absolute_directory(OWNER_EVIDENCE_ROOT, 'application evidence root')
    try:
        within = os.path.commonpath((evidence, root))
        relative = os.path.relpath(evidence, root)
    except ValueError as exc:
        raise UnsafeLedger('selected evidence directory is outside the application evidence root') from exc
    if (os.path.normcase(within) != os.path.normcase(root) or
            relative in ('.', '..') or relative.startswith('..' + os.sep)):
        raise UnsafeLedger('selected evidence directory must be inside the application evidence root')


    first_child = relative.split(os.sep)[0]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{9,119}', first_child):
        raise UnsafeLedger('evidence run identifier is malformed')
    return evidence


def create_unique_evidence_run(evidence_parent):
    """Create one new never-reused evidence subrun, without touching output.

    The explicit parent must already be below the application evidence root.
    An interrupted empty run stays on disk for inspection.
    """
    parent = _approved_evidence_directory(evidence_parent)
    path = os.path.join(parent, 'i03-stage-ledger-' + secrets.token_hex(16))
    try:
        os.mkdir(path)
    except OSError as exc:
        raise UnsafeLedger('new evidence run could not be exclusively created') from exc
    return _approved_evidence_directory(path)


def _final_and_parent(output_final):
    try:
        final = bundle_atomic._local_absolute(output_final)
        parent = os.path.dirname(final)
        bundle_atomic._guard_existing_ancestors(parent)
        bundle_atomic._require_ntfs(parent)
        parent_info = bundle_atomic._guard_stat(parent, directory=True)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('unsafe approved final output parent') from exc
    return final, parent, parent_info


def _check_selected_name(final, selected, kind):
    if type(kind) is not str or kind not in ('output', 'snapshot'):
        raise UnsafeLedger('unexpected selected retained stage kind')
    if (os.path.normcase(os.path.dirname(selected)) !=
            os.path.normcase(os.path.dirname(final))):
        raise UnsafeLedger('retained stage is outside approved final parent')
    leaf = os.path.basename(selected)
    if kind == 'output':
        valid = bool(bundle_atomic.STAGE_NAME.fullmatch(leaf))
    else:
        valid = bool(_SOURCE.fullmatch(leaf)) and leaf.startswith(
            os.path.basename(final) + '.source-freeze-')
    if not valid:
        raise UnsafeLedger('selected stage/snapshot name does not match approved final')


def _checked_identity(created_file_id):
    if (not isinstance(created_file_id, tuple) or len(created_file_id) != 2 or
            any(type(value) is not int or value < 0 for value in created_file_id)):
        raise UnsafeLedger('created FileID identity must be a native dev/inode pair')
    return created_file_id


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise UnsafeLedger('duplicate DPAPI selected ledger key')
        result[key] = value
    return result


def _observed_stage(path):
    """Never follow linked/reparse descendants or acquire child cleanup rights."""
    try:
        bundle_atomic._guard_existing_ancestors(os.path.dirname(path))
        return bundle_atomic._guard_stat(path, directory=True)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('stage path or FileID is unsafe/unavailable') from exc


def _confirmed_path_absent(path):
    """Only ENOENT confirms absence; ACL/reparse/I/O errors mean UNKNOWN.

    os.path.lexists silently maps many OSError cases to False on Windows.
    That is unsuitable for forensic classifications after a possible crash.
    This does not pin the parent or provide an atomic namespace observation.
    """
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise UnsafeLedger('ambiguous inaccessible NTFS namespace candidate') from exc
    return False


def _intent_record_path(evidence, kind, job_nonce):
    if type(job_nonce) is not str or not _TOKEN.fullmatch(job_nonce):
        raise UnsafeLedger('INTENT job nonce must be exact lowercase 128-bit hex')
    if type(kind) is not str or kind not in ('output', 'snapshot'):
        raise UnsafeLedger('INTENT stage kind unsupported')
    return os.path.join(evidence, job_nonce + '-' + kind + INTENT_SUFFIX)


def record_selected_stage_intent(*, evidence_directory, output_final,
                                 selected_stage, kind, job_nonce):
    """Write an exclusive DPAPI INTENT before creating the selected stage.

    A recorded intent authenticates only this CurrentUser DPAPI blob's bounded
    content, not CrossPascal application identity or the future stage object.
    Missing/failed BOUND must NEVER confer READY, recovery or cleanup rights.
    """
    evidence = _approved_evidence_directory(evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(output_final)
    try:
        stage = bundle_atomic._local_absolute(selected_stage)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('unsafe INTENT selected stage pathname') from exc
    _check_selected_name(final, stage, kind)
    if not _confirmed_path_absent(stage):
        raise UnsafeLedger('INTENT stage path already exists before mkdir')
    receipt = _intent_record_path(evidence, kind, job_nonce)
    record = {
        'schema': INTENT_SCHEMA, 'version': VERSION, 'kind': kind,
        'job_nonce': job_nonce, 'evidence_directory': evidence,
        'evidence_dev': evidence_info.st_dev,
        'evidence_ino': evidence_info.st_ino,
        'evidence_ctime_ns': evidence_info.st_ctime_ns,
        'output_final': final, 'stage': stage,
        'parent_dev': parent_info.st_dev, 'parent_ino': parent_info.st_ino,
        'parent_ctime_ns': parent_info.st_ctime_ns,
    }
    document = json.dumps(record, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    protected = _crypt(document, protect=True)
    try:
        with open(receipt, 'xb') as stream:
            stream.write(protected)
            stream.flush()
            os.fsync(stream.fileno())
        saved = bundle_atomic._guard_stat(receipt, directory=False)
        if saved.st_nlink != 1 or saved.st_size != len(protected):
            raise UnsafeLedger('new INTENT record unsafe file identity/size')
        after = bundle_atomic._guard_stat(evidence, directory=True)
        if (after.st_dev, after.st_ino, after.st_ctime_ns) != (
                evidence_info.st_dev, evidence_info.st_ino,
                evidence_info.st_ctime_ns):
            raise UnsafeLedger('INTENT evidence directory changed during write')
    except FileExistsError as exc:
        raise UnsafeLedger('selected DPAPI INTENT already exists; no overwrite') from exc
    except OSError as exc:
        raise UnsafeLedger('selected DPAPI INTENT could not be flushed') from exc
    return receipt


def _read_selected_stage_intent(evidence, final, kind, nonce):
    """Validate an exact DPAPI INTENT without requiring stage existence."""
    receipt = _intent_record_path(evidence, kind, nonce)
    if _confirmed_path_absent(receipt):
        return None
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    _chosen_final, _parent, parent_info = _final_and_parent(final)
    try:
        first = bundle_atomic._guard_stat(receipt, directory=False)
        if first.st_nlink != 1 or not 0 < first.st_size <= MAX_PROTECTED:
            raise UnsafeLedger('INTENT record has unsafe link/size')
        with open(receipt, 'rb') as stream:
            current = os.fstat(stream.fileno())
            if (current.st_dev, current.st_ino, current.st_size) != (
                    first.st_dev, first.st_ino, first.st_size):
                raise UnsafeLedger('INTENT pathname/FileID changed before read')
            protected = stream.read(MAX_PROTECTED + 1)
            if len(protected) != first.st_size:
                raise UnsafeLedger('INTENT record changed during read')
            final_stat = os.fstat(stream.fileno())
            if (final_stat.st_dev, final_stat.st_ino, final_stat.st_size,
                    final_stat.st_mtime_ns) != (
                    first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns):
                raise UnsafeLedger('INTENT record changed during protected read')
        last = bundle_atomic._guard_stat(receipt, directory=False)
        if (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns) != (
                first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns):
            raise UnsafeLedger('INTENT record pathname changed after read')
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('untrusted selected INTENT record') from exc
    try:
        raw = _crypt(protected, protect=False)
        record = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_json,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                UnsafeLedger('INTENT invalid JSON constant')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnsafeLedger('INTENT DPAPI decrypted content invalid') from exc
    if (not isinstance(record, dict) or set(record) != _INTENT_KEYS or
            record.get('schema') != INTENT_SCHEMA or
            type(record.get('version')) is not int or record['version'] != VERSION or
            record.get('kind') != kind or record.get('job_nonce') != nonce or
            record.get('evidence_directory') != evidence or
            record.get('output_final') != final or
            any(type(record.get(key)) is not int or record[key] < 0
                for key in ('evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
                            'parent_dev', 'parent_ino', 'parent_ctime_ns')) or
            (record['evidence_dev'], record['evidence_ino'],
             record['evidence_ctime_ns']) !=
            (evidence_info.st_dev, evidence_info.st_ino,
             evidence_info.st_ctime_ns) or
            (record['parent_dev'], record['parent_ino'],
             record['parent_ctime_ns']) !=
            (parent_info.st_dev, parent_info.st_ino,
             parent_info.st_ctime_ns)):
        raise UnsafeLedger('INTENT DPAPI parent/evidence/job binding invalid')
    try:
        selected = bundle_atomic._local_absolute(record['stage'])
        _check_selected_name(final, selected, kind)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('INTENT selected stage name unsafe') from exc
    return record


def inspect_selected_stage_events(*, evidence_directory, output_final,
                                  kind, job_nonce):
    """Read-only 0/INTENT/BOUND observations; NEVER recover READY or cleanup."""
    evidence = _approved_evidence_directory(evidence_directory)
    final, _parent, _info = _final_and_parent(output_final)
    intent = _read_selected_stage_intent(evidence, final, kind, job_nonce)
    bound = os.path.join(evidence, job_nonce + '-' + kind + SUFFIX)
    if intent is None:
        if not _confirmed_path_absent(bound):
            raise UnsafeLedger('BOUND exists without selected signed INTENT')
        status = 'NO_INTENT_OBSERVED_NOT_RECOVERED'
        selected = None
    else:
        selected = intent['stage']
        if not _confirmed_path_absent(bound):
            observation = inspect_selected_retained_stage(
                bound, expected_evidence_directory=evidence,
                expected_output_final=final,
                _include_recorded_identity=(kind == 'output'))
            if (observation['kind'] != kind or
                    observation['job_nonce'] != job_nonce or
                    observation['stage'] != selected):
                raise UnsafeLedger('INTENT and BOUND disagree on exact stage/job')
            status = ('INTENT_AND_FILEID_BOUND_NOT_RECOVERED'
                      if observation['status'] ==
                      'DPAPI_CURRENT_USER_FILEID_MATCH_NOT_RECOVERED'
                      else 'INTENT_AND_BOUND_IDENTITY_UNPROVEN_NOT_RECOVERED')
            if (kind == 'output' and
                    observation['status'] !=
                    'DPAPI_CURRENT_USER_FILEID_MATCH_NOT_RECOVERED' and
                    _confirmed_path_absent(selected) and
                    _confirmed_path_absent(_published_path(evidence, job_nonce))):


                try:
                    final_identity = _observed_stage(final)
                    if ((final_identity.st_dev, final_identity.st_ino,
                         final_identity.st_ctime_ns) ==
                            observation['_recorded_stage_identity'] and
                            _confirmed_path_absent(selected) and
                            _confirmed_path_absent(
                                _published_path(evidence, job_nonce))):
                        status = (
                            'POSSIBLE_RENAME_FILEID_MATCH_'
                            'UNJOURNALLED_NOT_RECOVERED')
                except UnsafeLedger:
                    pass
        elif _confirmed_path_absent(selected):
            status = 'INTENT_ONLY_STAGE_ABSENT_NOT_RECOVERED'
        else:
            try:
                _observed_stage(selected)
            except UnsafeLedger:
                status = 'INTENT_ONLY_UNSAFE_CANDIDATE_NOT_RECOVERED'
            else:
                status = 'INTENT_ONLY_UNBOUND_STAGE_PRESENT_NOT_RECOVERED'
    return {
        'schema': INTENT_SCHEMA, 'version': VERSION,
        'kind': kind, 'job_nonce': job_nonce, 'stage': selected,
        'status': status, 'ready': False, 'cleanup_allowed': False,
        'associated_job_proven': False, 'power_loss_durable': False,
        'same_sid_application_authenticated': False,
    }


def _published_path(evidence, nonce):
    if type(nonce) is not str or not _TOKEN.fullmatch(nonce):
        raise UnsafeLedger('PUBLISHED observation needs exact 128-bit job nonce')
    return os.path.join(evidence, nonce + '-output' + PUBLISHED_SUFFIX)


def _checked_published_prerequisites(evidence, final, stage, nonce):
    """Verify prior CurrentUser receipts, never revive a published ZIP."""
    intent = _read_selected_stage_intent(evidence, final, 'output', nonce)
    if intent is None or intent['stage'] != stage:
        raise UnsafeLedger('PUBLISHED observation has no matching prior INTENT')
    bound = os.path.join(evidence, nonce + '-output' + SUFFIX)
    try:
        prior = inspect_selected_retained_stage(
            bound, expected_evidence_directory=evidence,
            expected_output_final=final, _include_recorded_identity=True)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('PUBLISHED observation has no valid prior FileID BOUND') from exc
    if (prior['kind'] != 'output' or prior['job_nonce'] != nonce
            or prior['stage'] != stage):
        raise UnsafeLedger('PUBLISHED prior BOUND differs from exact job INTENT')
    return prior['_recorded_stage_identity']


def _precommit_path(evidence, nonce):
    if type(nonce) is not str or not _TOKEN.fullmatch(nonce):
        raise UnsafeLedger('PRECOMMIT inventory requires exact 128-bit job nonce')
    return os.path.join(evidence, nonce + '-output' + PRECOMMIT_SUFFIX)


def record_selected_precommit_inventory_observation(
        *, evidence_directory, output_final, selected_stage, job_nonce,
        approved_inventory):
    """Observe a complete approved private kit BEFORE a no-clobber rename.

    The CurrentUser DPAPI record is not proof of program authorship, publication,
    power-loss durability, verification, recovered READY or cleanup permission.
    """
    evidence = _approved_evidence_directory(evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(output_final)
    try:
        stage = bundle_atomic._local_absolute(selected_stage)
        _check_selected_name(final, stage, 'output')
    except (ValueError, TypeError, OSError) as exc:
        raise UnsafeLedger('PRECOMMIT selected output stage path invalid') from exc
    if not isinstance(approved_inventory, bundle_atomic.Inventory):
        raise UnsafeLedger('PRECOMMIT approved inventory type invalid')
    if (approved_inventory.schema != bundle_atomic.SCHEMA or
            approved_inventory.version != bundle_atomic.SCHEMA_VERSION):
        raise UnsafeLedger('PRECOMMIT inventory version invalid')
    if not _confirmed_path_absent(final):
        raise UnsafeLedger('PRECOMMIT destination already exists before rename')
    prior_identity = _checked_published_prerequisites(
        evidence, final, stage, job_nonce)
    current = _observed_stage(stage)
    if (current.st_dev, current.st_ino, current.st_ctime_ns) != prior_identity:
        raise UnsafeLedger('PRECOMMIT stage FileID differs from signed BOUND')
    try:
        inventory = bundle_atomic.scan_bundle_tree(stage)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('PRECOMMIT cannot scan complete selected kit') from exc
    if inventory != approved_inventory:
        raise UnsafeLedger('PRECOMMIT private kit differs from validated Inventory')
    record = {
        'schema': PRECOMMIT_SCHEMA, 'version': VERSION,
        'job_nonce': job_nonce, 'evidence_directory': evidence,
        'evidence_dev': evidence_info.st_dev, 'evidence_ino': evidence_info.st_ino,
        'evidence_ctime_ns': evidence_info.st_ctime_ns,
        'output_final': final, 'original_stage': stage,
        'parent_dev': parent_info.st_dev, 'parent_ino': parent_info.st_ino,
        'parent_ctime_ns': parent_info.st_ctime_ns,
        'stage_dev': current.st_dev, 'stage_ino': current.st_ino,
        'stage_ctime_ns': current.st_ctime_ns,
        'bundle_tree_sha256': inventory.tree_sha256,
        'bundle_member_count': len(inventory.members),
        'bundle_total_bytes': inventory.total_bytes,
    }
    plaintext = json.dumps(record, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode('utf-8')
    protected = _crypt(plaintext, protect=True)
    path = _precommit_path(evidence, job_nonce)
    try:
        with open(path, 'xb') as stream:
            stream.write(protected)
            stream.flush()
            os.fsync(stream.fileno())
        saved = bundle_atomic._guard_stat(path, directory=False)
        if saved.st_nlink != 1 or saved.st_size != len(protected):
            raise UnsafeLedger('PRECOMMIT record unsafe hardlink/size')
        after = _observed_stage(stage)
        checked = bundle_atomic.scan_bundle_tree(stage)
        evidence_after = bundle_atomic._guard_stat(evidence, directory=True)
        if ((after.st_dev, after.st_ino, after.st_ctime_ns) !=
                prior_identity or checked != inventory or
                (evidence_after.st_dev, evidence_after.st_ino,
                 evidence_after.st_ctime_ns) !=
                (evidence_info.st_dev, evidence_info.st_ino,
                 evidence_info.st_ctime_ns)):
            raise UnsafeLedger('PRECOMMIT stage/evidence changed during record write')
    except FileExistsError as exc:
        raise UnsafeLedger('duplicate PRECOMMIT observation; no overwrite') from exc
    except OSError as exc:
        raise UnsafeLedger('PRECOMMIT observation could not be flushed') from exc
    return path


def inspect_selected_precommit_inventory_observation(
        receipt, *, expected_evidence_directory, expected_output_final):
    """Read one signed precommit kit SHA; no post-crash publication recovery."""
    evidence = _approved_evidence_directory(expected_evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(expected_output_final)
    try:
        path = bundle_atomic._local_absolute(receipt)
        if os.path.normcase(os.path.dirname(path)) != os.path.normcase(evidence):
            raise UnsafeLedger('PRECOMMIT record outside the selected evidence run')
        match = _PRECOMMIT_LEAF.fullmatch(os.path.basename(path))
        if match is None:
            raise UnsafeLedger('PRECOMMIT filename invalid')
        stat_before = bundle_atomic._guard_stat(path, directory=False)
        if stat_before.st_nlink != 1 or not 0 < stat_before.st_size <= MAX_PROTECTED:
            raise UnsafeLedger('PRECOMMIT receipt size/hardlink unsafe')
        with open(path, 'rb') as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                    stat_before.st_dev, stat_before.st_ino, stat_before.st_size):
                raise UnsafeLedger('PRECOMMIT receipt pathname FileID changed')
            protected = stream.read(MAX_PROTECTED + 1)
            if len(protected) != stat_before.st_size:
                raise UnsafeLedger('PRECOMMIT receipt changed while reading')
            after_read = os.fstat(stream.fileno())
            if (after_read.st_dev, after_read.st_ino, after_read.st_size,
                    after_read.st_mtime_ns) != (
                    stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
                    stat_before.st_mtime_ns):
                raise UnsafeLedger('PRECOMMIT protected data changed during read')
        reopened = bundle_atomic._guard_stat(path, directory=False)
        if (reopened.st_dev, reopened.st_ino, reopened.st_size,
                reopened.st_mtime_ns) != (
                stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
                stat_before.st_mtime_ns):
            raise UnsafeLedger('PRECOMMIT record replaced during read')
    except (OSError, TypeError, ValueError) as exc:
        raise UnsafeLedger('PRECOMMIT selected record unsafe/inaccessible') from exc
    try:
        record = json.loads(_crypt(protected, protect=False).decode('utf-8'),
                            object_pairs_hook=_unique_json,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                UnsafeLedger('PRECOMMIT JSON constant invalid')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnsafeLedger('PRECOMMIT DPAPI decrypted record invalid') from exc
    numeric = ('evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
               'parent_dev', 'parent_ino', 'parent_ctime_ns',
               'stage_dev', 'stage_ino', 'stage_ctime_ns',
               'bundle_member_count', 'bundle_total_bytes')
    if (not isinstance(record, dict) or set(record) != _PRECOMMIT_KEYS or
            record.get('schema') != PRECOMMIT_SCHEMA or
            type(record.get('version')) is not int or record['version'] != VERSION or
            record.get('job_nonce') != match.group(1) or
            record.get('evidence_directory') != evidence or
            record.get('output_final') != final or
            type(record.get('bundle_tree_sha256')) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', record['bundle_tree_sha256']) or
            any(type(record.get(key)) is not int or record[key] < 0 for key in numeric) or
            (record['evidence_dev'], record['evidence_ino'],
             record['evidence_ctime_ns']) !=
            (evidence_info.st_dev, evidence_info.st_ino, evidence_info.st_ctime_ns)):
        raise UnsafeLedger('PRECOMMIT DPAPI schema/evidence/job binding invalid')
    try:
        stage = bundle_atomic._local_absolute(record['original_stage'])
        _check_selected_name(final, stage, 'output')
        prior = _checked_published_prerequisites(evidence, final, stage,
                                                 record['job_nonce'])
    except (TypeError, ValueError, OSError) as exc:
        raise UnsafeLedger('PRECOMMIT signed INTENT/BOUND chain invalid') from exc
    if prior != (record['stage_dev'], record['stage_ino'],
                 record['stage_ctime_ns']):
        raise UnsafeLedger('PRECOMMIT signed old-stage FileID mismatch')
    status = 'PRECOMMIT_NAMESPACE_UNPROVEN_NOT_RECOVERED'
    if (parent_info.st_dev, parent_info.st_ino, parent_info.st_ctime_ns) == (
            record['parent_dev'], record['parent_ino'], record['parent_ctime_ns']):
        if not _confirmed_path_absent(stage):
            try:
                current = _observed_stage(stage)
                if (current.st_dev, current.st_ino, current.st_ctime_ns) == prior:
                    status = 'PRECOMMIT_PRIVATE_STAGE_PRESENT_NOT_RECOVERED'
            except UnsafeLedger:
                pass
        else:
            try:
                current_final = _observed_stage(final)
                if ((current_final.st_dev, current_final.st_ino,
                     current_final.st_ctime_ns) == prior and
                        _confirmed_path_absent(stage)):
                    try:
                        inventory = bundle_atomic.scan_bundle_tree(final)


                        after_final = _observed_stage(final)
                        if ((after_final.st_dev, after_final.st_ino,
                             after_final.st_ctime_ns) == prior and
                                _confirmed_path_absent(stage)):
                            status = (
                                'POSSIBLE_RENAME_PRECOMMIT_TREE_MATCH_NOT_RECOVERED'
                                if (inventory.tree_sha256 == record['bundle_tree_sha256'] and
                                    len(inventory.members) == record['bundle_member_count'] and
                                    inventory.total_bytes == record['bundle_total_bytes'])
                                else 'POSSIBLE_RENAME_PRECOMMIT_BYTES_CHANGED_NOT_RECOVERED')
                    except (OSError, ValueError, TypeError):


                        pass
            except UnsafeLedger:
                pass
    return {
        'schema': PRECOMMIT_SCHEMA, 'version': VERSION,
        'job_nonce': record['job_nonce'], 'stage': stage, 'final': final,
        'status': status, 'bundle_tree_sha256': record['bundle_tree_sha256'],
        'ready': False, 'cleanup_allowed': False,
        'associated_job_proven': False, 'power_loss_durable': False,
        'same_sid_application_authenticated': False,
    }


def record_selected_publication_observation(
        *, evidence_directory, output_final, original_stage,
        job_nonce, expected_private_stage_file_id, committed_tree_sha256):
    """Append only a post-rename identity observation in the selected evidence run.

    Caller must have independently completed compiler/Verify and exact bundle
    validation. This is NOT a durable commit marker or restored READY. A write
    refusal after namespace exposure must leave that public directory intact.
    """
    evidence = _approved_evidence_directory(evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(output_final)
    try:
        stage = bundle_atomic._local_absolute(original_stage)
        _check_selected_name(final, stage, 'output')
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('PUBLISHED source private-stage pathname invalid') from exc
    expected = _checked_identity(expected_private_stage_file_id)
    if (type(committed_tree_sha256) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', committed_tree_sha256)):
        raise UnsafeLedger('PUBLISHED bundle tree SHA-256 invalid')
    prior_stage_identity = _checked_published_prerequisites(
        evidence, final, stage, job_nonce)
    if prior_stage_identity[:2] != expected:
        raise UnsafeLedger('PUBLISHED prior signed BOUND FileID differs from stage')
    if not _confirmed_path_absent(stage):
        raise UnsafeLedger('PUBLISHED private stage still exists; no proven rename')
    current = _observed_stage(final)
    if ((current.st_dev, current.st_ino, current.st_ctime_ns)
            != prior_stage_identity):
        raise UnsafeLedger('PUBLISHED final FileID differs from original stage')
    try:
        observed_inventory = bundle_atomic.scan_bundle_tree(final)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('PUBLISHED final tree cannot be SHA-inventoried') from exc
    if observed_inventory.tree_sha256 != committed_tree_sha256:
        raise UnsafeLedger('PUBLISHED final tree bytes disagree with committed SHA')
    record = {
        'schema': PUBLISHED_SCHEMA, 'version': VERSION,
        'job_nonce': job_nonce,
        'evidence_directory': evidence,
        'evidence_dev': evidence_info.st_dev,
        'evidence_ino': evidence_info.st_ino,
        'evidence_ctime_ns': evidence_info.st_ctime_ns,
        'output_final': final, 'original_stage': stage,
        'parent_dev': parent_info.st_dev, 'parent_ino': parent_info.st_ino,
        'parent_ctime_ns': parent_info.st_ctime_ns,
        'committed_dev': current.st_dev,
        'committed_ino': current.st_ino,
        'committed_ctime_ns': current.st_ctime_ns,
        'bundle_tree_sha256': committed_tree_sha256,
    }
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    protected = _crypt(raw, protect=True)
    path = _published_path(evidence, job_nonce)
    try:
        with open(path, 'xb') as stream:
            stream.write(protected)
            stream.flush()
            os.fsync(stream.fileno())
        saved = bundle_atomic._guard_stat(path, directory=False)
        if saved.st_nlink != 1 or saved.st_size != len(protected):
            raise UnsafeLedger('PUBLISHED protected observation unsafe link/size')
        recheck = _observed_stage(final)
        checked_tree = bundle_atomic.scan_bundle_tree(final)
        evidence_after = bundle_atomic._guard_stat(evidence, directory=True)
        if ((recheck.st_dev, recheck.st_ino, recheck.st_ctime_ns) !=
                (current.st_dev, current.st_ino, current.st_ctime_ns) or
                (evidence_after.st_dev, evidence_after.st_ino,
                 evidence_after.st_ctime_ns) !=
                (evidence_info.st_dev, evidence_info.st_ino,
                 evidence_info.st_ctime_ns) or
                checked_tree != observed_inventory):
            raise UnsafeLedger('PUBLISHED final or evidence directory changed')
    except FileExistsError as exc:
        raise UnsafeLedger('duplicate PUBLISHED observation; no overwrite') from exc
    except OSError as exc:
        raise UnsafeLedger('PUBLISHED observation could not be flushed') from exc
    return path


def inspect_selected_publication_observation(
        receipt, *, expected_evidence_directory, expected_output_final):
    """Inspect post-rename DPAPI bytes; ALWAYS forbid recovered READY."""
    evidence = _approved_evidence_directory(expected_evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(expected_output_final)
    try:
        path = bundle_atomic._local_absolute(receipt)
        if os.path.normcase(os.path.dirname(path)) != os.path.normcase(evidence):
            raise UnsafeLedger('PUBLISHED observation outside the selected evidence run')
        match = _PUBLISHED_LEAF.fullmatch(os.path.basename(path))
        if match is None:
            raise UnsafeLedger('PUBLISHED observation filename invalid')
        original = bundle_atomic._guard_stat(path, directory=False)
        if original.st_nlink != 1 or not 0 < original.st_size <= MAX_PROTECTED:
            raise UnsafeLedger('PUBLISHED record hardlink or size unsafe')
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size) != (
                    original.st_dev, original.st_ino, original.st_size):
                raise UnsafeLedger('PUBLISHED receipt identity changed')
            protected = stream.read(MAX_PROTECTED + 1)
            if len(protected) != original.st_size:
                raise UnsafeLedger('PUBLISHED receipt truncated or grew')
            after = os.fstat(stream.fileno())
            if (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns) != (
                    original.st_dev, original.st_ino, original.st_size,
                    original.st_mtime_ns):
                raise UnsafeLedger('PUBLISHED receipt changed during read')
        final_path_stat = bundle_atomic._guard_stat(path, directory=False)
        if (final_path_stat.st_dev, final_path_stat.st_ino,
                final_path_stat.st_size, final_path_stat.st_mtime_ns) != (
                original.st_dev, original.st_ino, original.st_size,
                original.st_mtime_ns):
            raise UnsafeLedger('PUBLISHED receipt path replaced during read')
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('unsafe PUBLISHED protected record') from exc
    try:
        plaintext = _crypt(protected, protect=False)
        record = json.loads(plaintext.decode('utf-8'),
                            object_pairs_hook=_unique_json,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                UnsafeLedger('invalid PUBLISHED JSON constant')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnsafeLedger('PUBLISHED decrypted DPAPI observation invalid') from exc
    if (not isinstance(record, dict) or set(record) != _PUBLISHED_KEYS or
            record.get('schema') != PUBLISHED_SCHEMA or
            type(record.get('version')) is not int or record['version'] != VERSION or
            record.get('job_nonce') != match.group(1) or
            record.get('evidence_directory') != evidence or
            record.get('output_final') != final or
            type(record.get('bundle_tree_sha256')) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', record['bundle_tree_sha256']) or
            any(type(record.get(key)) is not int or record[key] < 0
                for key in ('evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
                            'parent_dev', 'parent_ino', 'parent_ctime_ns',
                            'committed_dev', 'committed_ino', 'committed_ctime_ns')) or
            (record['evidence_dev'], record['evidence_ino'],
             record['evidence_ctime_ns']) !=
            (evidence_info.st_dev, evidence_info.st_ino,
             evidence_info.st_ctime_ns)):
        raise UnsafeLedger('PUBLISHED evidence/job/nonce schema binding invalid')
    try:
        original_stage = bundle_atomic._local_absolute(record['original_stage'])
        _check_selected_name(final, original_stage, 'output')
        signed_prior_identity = _checked_published_prerequisites(
            evidence, final, original_stage, record['job_nonce'])
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('PUBLISHED INTENT/BOUND event association invalid') from exc
    if signed_prior_identity != (
            record['committed_dev'], record['committed_ino'],
            record['committed_ctime_ns']):
        raise UnsafeLedger('PUBLISHED signed prior BOUND FileID differs from final')
    if (parent_info.st_dev, parent_info.st_ino, parent_info.st_ctime_ns) != (
            record['parent_dev'], record['parent_ino'], record['parent_ctime_ns']):
        status = 'COMMITTED_NAMESPACE_IDENTITY_UNPROVEN_NOT_RECOVERED'
    else:
        try:
            now = _observed_stage(final)
            if (now.st_dev, now.st_ino, now.st_ctime_ns) != (
                    record['committed_dev'], record['committed_ino'],
                    record['committed_ctime_ns']):
                status = 'COMMITTED_NAMESPACE_IDENTITY_UNPROVEN_NOT_RECOVERED'
            else:


                try:
                    measured = bundle_atomic.scan_bundle_tree(final)
                    status = ('COMMITTED_NAMESPACE_FILEID_MATCH_NOT_RECOVERED'
                              if measured.tree_sha256 == record['bundle_tree_sha256']
                              else 'COMMITTED_NAMESPACE_BYTES_CHANGED_NOT_RECOVERED')
                except (OSError, ValueError, TypeError):
                    status = 'COMMITTED_NAMESPACE_BYTES_CHANGED_NOT_RECOVERED'
        except UnsafeLedger:
            status = 'COMMITTED_NAMESPACE_IDENTITY_UNPROVEN_NOT_RECOVERED'
    return {
        'schema': PUBLISHED_SCHEMA, 'version': VERSION,
        'job_nonce': record['job_nonce'], 'final': final,
        'status': status, 'bundle_tree_sha256': record['bundle_tree_sha256'],
        'ready': False, 'cleanup_allowed': False,
        'associated_job_proven': False,
        'same_sid_application_authenticated': False,
        'power_loss_durable': False,
    }


def record_selected_retained_stage(*, evidence_directory, output_final,
                                   selected_stage, kind, job_nonce,
                                   created_file_id, require_intent=False):
    """Write one no-overwrite DPAPI CurrentUser stage observation, no READY.

    `created_file_id` must be supplied by the trusted synchronous creation
    callback. Even so, the observation is not atomic FileID custody: the stage
    might be substituted between mkdir and the creator's first pathname stat.
    """
    evidence = _approved_evidence_directory(evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(output_final)
    try:
        stage = bundle_atomic._local_absolute(selected_stage)
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('invalid selected stage path') from exc
    _check_selected_name(final, stage, kind)
    expected = _checked_identity(created_file_id)
    if type(job_nonce) is not str or not _TOKEN.fullmatch(job_nonce):
        raise UnsafeLedger('job nonce must be exact 128-bit lowercase hex')
    observed = _observed_stage(stage)
    if (observed.st_dev, observed.st_ino) != expected:
        raise UnsafeLedger('stage FileID identity differs from creation callback')
    if require_intent:
        if type(require_intent) is not bool:
            raise UnsafeLedger('require_intent must be boolean')
        paired = _read_selected_stage_intent(evidence, final, kind, job_nonce)
        if paired is None or paired['stage'] != stage:
            raise UnsafeLedger('BOUND requires exact preceding DPAPI INTENT')
    record = {
        'schema': SCHEMA,
        'version': VERSION,
        'kind': kind,
        'job_nonce': job_nonce,
        'evidence_directory': evidence,
        'evidence_dev': evidence_info.st_dev,
        'evidence_ino': evidence_info.st_ino,
        'evidence_ctime_ns': evidence_info.st_ctime_ns,
        'output_final': final,
        'stage': stage,
        'parent_dev': parent_info.st_dev,
        'parent_ino': parent_info.st_ino,
        'parent_ctime_ns': parent_info.st_ctime_ns,
        'stage_dev': observed.st_dev,
        'stage_ino': observed.st_ino,
        'stage_ctime_ns': observed.st_ctime_ns,
    }
    document = json.dumps(record, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    protected = _crypt(document, protect=True)
    receipt = os.path.join(evidence, job_nonce + '-' + kind + SUFFIX)
    try:
        with open(receipt, 'xb') as stream:
            stream.write(protected)
            stream.flush()
            os.fsync(stream.fileno())
        saved = bundle_atomic._guard_stat(receipt, directory=False)
        if saved.st_nlink != 1 or saved.st_size != len(protected):
            raise UnsafeLedger('new protected receipt FileID/link/size unsafe')
        after = _observed_stage(stage)
        if ((after.st_dev, after.st_ino) != expected or
                after.st_ctime_ns != observed.st_ctime_ns):
            raise UnsafeLedger('selected stage identity changed during ledger write')
        evidence_after = bundle_atomic._guard_stat(evidence, directory=True)
        if ((evidence_after.st_dev, evidence_after.st_ino,
             evidence_after.st_ctime_ns) !=
                (evidence_info.st_dev, evidence_info.st_ino,
                 evidence_info.st_ctime_ns)):
            raise UnsafeLedger('evidence directory changed during ledger write')
    except FileExistsError as exc:
        raise UnsafeLedger('duplicate retained receipt exists; never overwrite') from exc
    except OSError as exc:
        raise UnsafeLedger('cannot durably flush new selected DPAPI receipt') from exc

    return receipt


def inspect_selected_retained_stage(receipt, *, expected_evidence_directory,
                                    expected_output_final,
                                    _include_recorded_identity=False):
    """Inspect a known DPAPI receipt after restart; never infer READY or cleanup."""
    evidence = _approved_evidence_directory(expected_evidence_directory)
    evidence_info = bundle_atomic._guard_stat(evidence, directory=True)
    final, _parent, parent_info = _final_and_parent(expected_output_final)
    try:
        chosen = bundle_atomic._local_absolute(receipt)
        if os.path.normcase(os.path.dirname(chosen)) != os.path.normcase(evidence):
            raise UnsafeLedger('receipt outside explicitly selected evidence directory')
        leaf = os.path.basename(chosen)
        match = _RECEIPT_LEAF.fullmatch(leaf)
        if match is None:
            raise UnsafeLedger('unsafe selected DPAPI receipt basename')
        first = bundle_atomic._guard_stat(chosen, directory=False)
        if first.st_nlink != 1 or not 0 < first.st_size <= MAX_PROTECTED:
            raise UnsafeLedger('receipt link/size is unsafe')
        with open(chosen, 'rb') as stream:
            active = os.fstat(stream.fileno())
            if (active.st_dev, active.st_ino, active.st_size) != (
                    first.st_dev, first.st_ino, first.st_size):
                raise UnsafeLedger('receipt FileID differs from guarded pathname')
            ciphertext = stream.read(MAX_PROTECTED + 1)
            if len(ciphertext) != first.st_size:
                raise UnsafeLedger('receipt changed during bounded read')
            after = os.fstat(stream.fileno())
            if (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns) != (
                    first.st_dev, first.st_ino, first.st_size,
                    first.st_mtime_ns):
                raise UnsafeLedger('receipt changed during protected read')
        last = bundle_atomic._guard_stat(chosen, directory=False)
        if (last.st_dev, last.st_ino, last.st_size,
                last.st_mtime_ns) != (
                first.st_dev, first.st_ino, first.st_size,
                first.st_mtime_ns):
            raise UnsafeLedger('receipt pathname replaced during read')
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('unsafe selected evidence record') from exc
    try:
        clear = _crypt(ciphertext, protect=False)
        record = json.loads(clear.decode('utf-8'),
                            object_pairs_hook=_unique_json,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                UnsafeLedger('noncanonical DPAPI record constant')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnsafeLedger('invalid decrypted DPAPI selected record') from exc
    if (not isinstance(record, dict) or set(record) != _KEYS or
            record.get('schema') != SCHEMA or
            type(record.get('version')) is not int or record['version'] != VERSION or
            record.get('evidence_directory') != evidence or
            (record.get('evidence_dev'), record.get('evidence_ino'),
             record.get('evidence_ctime_ns')) !=
            (evidence_info.st_dev, evidence_info.st_ino,
             evidence_info.st_ctime_ns) or
            record.get('output_final') != final or
            record.get('job_nonce') != match.group(1) or
            record.get('kind') != match.group(2) or
            any(type(record.get(key)) is not int or record[key] < 0
                for key in ('evidence_dev', 'evidence_ino', 'evidence_ctime_ns',
                            'parent_dev', 'parent_ino', 'parent_ctime_ns',
                            'stage_dev', 'stage_ino', 'stage_ctime_ns'))):
        raise UnsafeLedger('decrypted DPAPI receipt binding/schema invalid')
    try:
        stage = bundle_atomic._local_absolute(record['stage'])
        _check_selected_name(final, stage, record['kind'])
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeLedger('DPAPI protected stage path is invalid') from exc
    if (parent_info.st_dev, parent_info.st_ino, parent_info.st_ctime_ns) != (
            record['parent_dev'], record['parent_ino'],
            record['parent_ctime_ns']):
        status = 'IDENTITY_MISMATCH_NOT_RECOVERED'
    else:
        try:
            current = _observed_stage(stage)
            status = ('DPAPI_CURRENT_USER_FILEID_MATCH_NOT_RECOVERED'
                      if (current.st_dev, current.st_ino, current.st_ctime_ns) ==
                      (record['stage_dev'], record['stage_ino'],
                       record['stage_ctime_ns'])
                      else 'IDENTITY_MISMATCH_NOT_RECOVERED')
        except UnsafeLedger:
            status = 'IDENTITY_UNPROVEN_NOT_RECOVERED'
    result = {
        'schema': SCHEMA,
        'version': VERSION,
        'status': status,
        'stage': stage,
        'kind': record['kind'],
        'job_nonce': record['job_nonce'],
        'ready': False,
        'cleanup_allowed': False,
        'associated_job_proven': False,
        'authentication_limit': 'DPAPI_CURRENT_USER_NOT_SAME_SID_APPLICATION_AUTH',
        'power_loss_durable': False,
    }
    if _include_recorded_identity is True:


        result['_recorded_stage_identity'] = (
            record['stage_dev'], record['stage_ino'], record['stage_ctime_ns'])
    return result


def _global_lease_name(evidence_directory, job_nonce, output_final):
    """A job-specific Win32 kernel name, not a persisted lock file."""
    evidence = _approved_evidence_directory(evidence_directory)
    final, _parent, _info = _final_and_parent(output_final)
    if type(job_nonce) is not str or not _TOKEN.fullmatch(job_nonce):
        raise UnsafeLedger('Global lease requires an exact 128-bit job nonce')
    message = '\x00'.join((evidence, job_nonce, final)).encode('utf-8')
    name = ('Global\\CrossPascal-I03-RetainedLease-' +
            hashlib.sha256(message).hexdigest())
    return name


def _thread_owned_lease_names():
    owned = getattr(_LOCAL_LEASES, 'names', None)
    if owned is None:
        owned = set()
        _LOCAL_LEASES.names = owned
    return owned


@contextmanager
def hold_active_stage_lease(evidence_directory, job_nonce, output_final):
    """Own a fresh Global mutex for the current worker thread's lifetime.

    A collision is refused rather than joined, even when created by this same
    SID. Exiting releases kernel ownership and ALWAYS attempts CloseHandle.
    Mutex lifetime is observational; it does not authenticate an application,
    prevent another process from using the same SID, or authorize recovery.
    """
    if os.name != 'nt':
        raise UnsafeLedger('Global Win32 active lease requires native Windows')
    name = _global_lease_name(evidence_directory, job_nonce, output_final)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL,
                                    wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    handle = kernel.CreateMutexW(None, True, name)
    if not handle:
        raise UnsafeLedger('Global Win32 active stage lease cannot be created')
    collision = ctypes.get_last_error() == 183
    registered = False
    try:
        if collision:
            raise UnsafeLedger('Global Win32 stage lease already exists')
        _thread_owned_lease_names().add(name)
        registered = True
        yield
    finally:
        try:
            if registered:
                _thread_owned_lease_names().discard(name)
            if not collision and not kernel.ReleaseMutex(handle):
                raise UnsafeLedger('Global Win32 stage lease release failed')
        finally:
            if not kernel.CloseHandle(handle):
                raise UnsafeLedger('Global Win32 stage lease CloseHandle failed')


def inspect_active_stage_lease(receipt, *, expected_evidence_directory,
                               expected_output_final):
    """Observe a named Global kernel mutex, no disk edits and no recovery.

    WAIT_TIMEOUT shows only that the named mutex is held *at this instant*,
    not who holds it. WAIT_OBJECT_0/ABANDONED temporarily takes ownership and
    releases it immediately; inaccessible/malformed kernel state is UNKNOWN.
    This does not claim same-SID application authentication or cleanup rights.
    """
    result = inspect_selected_retained_stage(
        receipt, expected_evidence_directory=expected_evidence_directory,
        expected_output_final=expected_output_final)
    result['same_sid_application_authenticated'] = False
    result['lease_status'] = 'LEASE_STATUS_UNKNOWN'
    name = _global_lease_name(
        expected_evidence_directory, result['job_nonce'],
        expected_output_final)


    if name in _thread_owned_lease_names():
        result['lease_status'] = (
            'GLOBAL_MUTEX_HELD_BY_CALLING_THREAD_OWNER_UNVERIFIED')
        return result
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL,
                                  wintypes.LPCWSTR)
    kernel.OpenMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    handle = kernel.OpenMutexW(0x00100001, False, name)
    if not handle:
        if ctypes.get_last_error() == 2:
            result['lease_status'] = 'NO_ACTIVE_LEASE_OBSERVED'
        return result
    try:
        outcome = kernel.WaitForSingleObject(handle, 0)
        if outcome == 0x102:
            result['lease_status'] = 'GLOBAL_MUTEX_BUSY_OWNER_UNVERIFIED'
        elif outcome in (0, 0x80):
            if kernel.ReleaseMutex(handle):
                result['lease_status'] = (
                    'NO_ACTIVE_LEASE_OBSERVED' if outcome == 0 else
                    'ABANDONED_LEASE_OBSERVED_NOT_RECOVERED')
    finally:
        if not kernel.CloseHandle(handle):
            result['lease_status'] = 'LEASE_STATUS_UNKNOWN'
    return result
