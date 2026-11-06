# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Actual selected Paddle Rec dictionary bytes for a compiled Pascal consumer.

The exporter holds its input dictionary only until WSL returns. A later TVM
compiler bundle must not contain merely its former *absolute source path*:
there would be no selected recognition alphabet to deliver to Pascal/CTC.

This opt-in compiler-window scope copies the exact Preview-approved dictionary
into output/dictionary, holds BOTH owner and output bytes with native Win32
read/share-only handles until the caller's compiler returns, and then audits
the complete selected source and output again. Failed scopes retain all output
bytes. It is resource delivery, NOT Det+Rec+CTC execution, a numerical oracle,
an immutable checkpoint or a release-grade manifest.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path

from . import hf_held_stage, omdet_public_candidate, paddle_closure, source_lock


SCHEMA = 'crosspascal.paddle-selected-rec-consumer-dictionary.v1'
_BLOCK_BYTES = 1024 * 1024


class PaddleConsumerDictionaryError(ValueError):
    """Selected dictionary cannot be shipped to the Pascal consumer safely."""


def _selection(approved):
    if type(approved) is not dict or approved.get('role') not in ('det', 'rec'):
        raise PaddleConsumerDictionaryError('approved selected Paddle role missing')
    if approved['role'] == 'det':
        if approved.get('dictionary') is not None:
            raise PaddleConsumerDictionaryError('Det cannot carry a Rec dictionary')
        return None
    record = approved.get('dictionary')
    if type(record) is not dict or type(record.get('reference')) is not str:
        raise PaddleConsumerDictionaryError('selected Rec dictionary receipt missing')
    source = paddle_closure._raw_absolute(
        record['reference'], 'approved recognition dictionary')
    filename = paddle_closure._safe_name(source.name)
    relative = 'dictionary/' + filename
    if (record.get('path') != relative or
            type(record.get('bytes')) is not int or
            not 0 < record['bytes'] <= paddle_closure.MAX_DICTIONARY_BYTES or
            type(record.get('sha256')) is not str or
            len(record['sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in record['sha256'])):
        raise PaddleConsumerDictionaryError('invalid selected Rec dictionary size/SHA/path')
    rows = approved.get('entries')
    if (type(rows) is not list or
            [row for row in rows if type(row) is dict and row.get('path') == relative]
            != [{'path': relative, 'bytes': record['bytes'],
                 'sha256': record['sha256']}]):
        raise PaddleConsumerDictionaryError('approved Rec dictionary is not in selected tree')
    return source, relative, record['bytes'], record['sha256']


@contextmanager
def hold_selected_rec_dictionary(approved, output_directory):
    """Yield in-compiler advisory receipt while both exact Rec dictionaries hold.

    The same mutable receipt is promoted to POST only *after* Win32/FileID/SHA
    checks pass on context exit. Never publish the pending receipt beforehand.
    """
    selected = _selection(approved)
    if selected is None:
        yield None
        return
    if not source_lock._is_windows():
        raise PaddleConsumerDictionaryError('native Win32 selected dictionary required')
    source, relative, length, digest = selected
    try:
        current = paddle_closure.audit_selected_paddle_source(
            approved['source_root'], role='rec', dictionary=str(source),
            expected_tree_sha256=approved['tree_sha256'])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PaddleConsumerDictionaryError(
            'Preview-approved Paddle dictionary is no longer selected: %s' % error
        ) from error
    if current != approved:
        raise PaddleConsumerDictionaryError('original selected Paddle inventory changed')

    api = source_lock._Kernel32()
    try:
        root = hf_held_stage._directory(
            output_directory, 'Paddle compiler output', api)
        api.assert_local_ntfs(source)
        if source == root or root in source.parents:
            raise PaddleConsumerDictionaryError('selected owner dictionary inside output')
        source_info = paddle_closure._checked_file(
            source, limit=paddle_closure.MAX_DICTIONARY_BYTES)
        if source_info.st_size != length:
            raise PaddleConsumerDictionaryError('selected dictionary length changed')
        folder = root / 'dictionary'
        destination = folder / source.name
        if os.path.lexists(folder) or os.path.lexists(destination):
            raise PaddleConsumerDictionaryError('compiler dictionary destination occupied')

        with ExitStack() as holds:

            opened_directories = set()
            for tree in (root, source.parent):
                for directory in reversed((tree, *tree.parents)):
                    if directory in opened_directories:
                        continue
                    holds.enter_context(
                        omdet_public_candidate._hold_plain_ntfs_directory(directory))
                    opened_directories.add(directory)
            source_handle = api.open(source)
            holds.callback(api.close, source_handle)
            source_identity = api.identity(source_handle)
            if source_identity.file_bytes != length:
                raise PaddleConsumerDictionaryError('held selected dictionary length differs')
            hf_held_stage._native_check(
                api, source, source_handle, source_identity, length, digest,
                _BLOCK_BYTES)

            folder.mkdir(exist_ok=False)
            holds.enter_context(
                omdet_public_candidate._hold_plain_ntfs_directory(folder))
            copied_hash = hashlib.sha256()
            total = 0


            with source.open('rb') as original, destination.open('xb') as target:
                if (paddle_closure._descriptor_identity(os.fstat(original.fileno()))
                        != paddle_closure._descriptor_identity(source_info)):
                    raise PaddleConsumerDictionaryError('owner dictionary changed before copy')
                while True:
                    piece = original.read(_BLOCK_BYTES)
                    if not piece:
                        break
                    total += len(piece)
                    if total > length:
                        raise PaddleConsumerDictionaryError('owner dictionary grew during copy')
                    target.write(piece)
                    copied_hash.update(piece)
                target.flush()
                os.fsync(target.fileno())
            if total != length or copied_hash.hexdigest() != digest:
                raise PaddleConsumerDictionaryError('copied consumer dictionary SHA differs')

            consumer_handle = api.open(destination)
            holds.callback(api.close, consumer_handle)
            consumer_identity = api.identity(consumer_handle)
            hf_held_stage._native_check(
                api, destination, consumer_handle, consumer_identity, length,
                digest, _BLOCK_BYTES)
            receipt = {
                'schema': SCHEMA,
                'source_tree_sha256': approved['tree_sha256'],
                'relative_path': relative,
                'sha256': digest,
                'bytes': length,
                'status': 'WIN32_HELD_SELECTED_REC_DICTIONARY_NOT_MODEL_UAT',
                'held_through_pipeline_run': False,
                'verification_status': 'NOT_CHECKED',
                'native_pascal_ctc': 'NOT_RUN',
                'release_grade_immutable': False,
                'retained_after_context': True,
            }
            def post_audit():


                hf_held_stage._native_check(
                    api, source, source_handle, source_identity, length,
                    digest, _BLOCK_BYTES)
                hf_held_stage._native_check(
                    api, destination, consumer_handle, consumer_identity,
                    length, digest, _BLOCK_BYTES)
                after = paddle_closure.audit_selected_paddle_source(
                    approved['source_root'], role='rec', dictionary=str(source),
                    expected_tree_sha256=approved['tree_sha256'])
                if after != approved:
                    raise PaddleConsumerDictionaryError(
                        'selected Paddle source/dictionary changed through compiler')
            try:
                yield receipt
            except BaseException:
                post_audit()
                raise
            else:
                post_audit()
                receipt['status'] = (
                    'POST_WIN32_HELD_SELECTED_REC_DICTIONARY_NOT_MODEL_UAT')
                receipt['held_through_pipeline_run'] = True
    except (OSError, ValueError, source_lock.SourceLockError,
            hf_held_stage.HfHeldStageError,
            omdet_public_candidate.OmDetPublicCandidateError) as error:
        if isinstance(error, PaddleConsumerDictionaryError):
            raise
        raise PaddleConsumerDictionaryError(
            'Paddle selected Rec dictionary consumer stage refused: %s' % error
        ) from error
