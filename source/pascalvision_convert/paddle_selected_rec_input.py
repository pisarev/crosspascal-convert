# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys

from . import ocr_png_preprocess, ocr_ort_raw_capture, paddle_closure
from . import source_freeze, source_lock


SCHEMA = 'crosspascal.step08.paddle-selected-ppocrv5-server-rec-input.v1'
STATUS = 'SOURCE_PROFILE_PREPROCESSED_ONLY_NOT_MODEL_ORACLE'
SHAPE = (1, 3, 48, 320)
INPUT_NAME = 'x'
INPUT_BYTES = 4 * 1 * 3 * 48 * 320
MAX_REPORT = 64 * 1024
_PINNED_TREE_SHA = (
    'c231e4f7b6b97ca3d74c41683d96c3456fdca30830dfcd1cdd27118ae625c460')
_PINNED_DICT_SHA = (
    '1ea29636956177e400af712d9782e7693f3fb25f98617bed10479d2965a836fd')
_PINNED_SOURCE = {
    'dictionary/ppocrv5_dict.txt': _PINNED_DICT_SHA,
    'inference.json': '8e6e12e5d42531840310977fffb58165bf889fc5061408c5a8afdb6985f47fcb',
    'inference.pdiparams': '63853f062a5f4089befc16f565a68277618e0da5cb45468b49d11079de0ada77',
    'inference.yml': '2c719dba044c4e2228aef8ff92f5f575394d75d24c16de096a33b7cfd902f66d',
}
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_INPUT = 'input.f32le'
_RECEIPT = 'input-source.json'


class SelectedRecInputRefused(ValueError):
    pass


def _require(test, reason):
    if not test:
        raise SelectedRecInputRefused('selected PP-OCRv5 Server Rec input refused: ' + reason)


def _model(source_directory, dictionary, approved_tree_sha256):
    _require(type(approved_tree_sha256) is str and
             _HEX.fullmatch(approved_tree_sha256),
             'explicit lowercase selected Rec source-tree SHA required')
    _require(approved_tree_sha256 == _PINNED_TREE_SHA,
             'input producer is pinned to this real PP-OCRv5 Server Rec revision')
    try:
        chosen = paddle_closure.audit_selected_paddle_source(
            source_directory, role='rec', dictionary=dictionary,
            expected_tree_sha256=approved_tree_sha256)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SelectedRecInputRefused('real pinned model cannot be audited: %s' % exc) from exc
    rows = {row['path']: row['sha256'] for row in chosen['entries']}
    _require(rows == _PINNED_SOURCE and chosen['file_count'] == 4 and
             chosen['dictionary']['sha256'] == _PINNED_DICT_SHA,
             'graph/params/inference.yml/18383-token alphabet differ from real source')
    _require(chosen['verification_status'] == 'NOT_CHECKED' and
             chosen['release_grade_immutable'] is False,
             'point-in-time source selection cannot assert trained verification')
    return chosen


def _encode_source_rec_png(png_bytes):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SelectedRecInputRefused('NumPy and OpenCV are required') from exc
    pixels = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8),
                          cv2.IMREAD_COLOR)
    _require(pixels is not None and pixels.ndim == 3 and
             pixels.shape[2] == 3 and pixels.dtype == np.uint8 and
             0 < pixels.shape[0] * pixels.shape[1] <=
             ocr_png_preprocess.MAX_IMAGE_PIXELS,
             'selected Rec BGR8 decoded source pixels missing/oversized')
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    resized_width = min(SHAPE[3], max(1, (SHAPE[2] * width + height - 1) // height))
    resized = cv2.resize(pixels, (resized_width, SHAPE[2]),
                         interpolation=cv2.INTER_LINEAR)


    normalized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    normalized -= 0.5
    normalized /= 0.5
    padded = np.zeros(SHAPE[1:], dtype=np.float32)
    padded[:, :, :resized_width] = normalized
    _require(np.isfinite(padded).all(), 'nonfinite normalized Rec input values')
    packed = np.ascontiguousarray(padded, dtype=np.dtype('<f4')).tobytes(order='C')
    _require(len(packed) == INPUT_BYTES and INPUT_BYTES <=
             ocr_ort_raw_capture.MAX_INPUT_BYTES,
             'source-profile Rec tensor bytes/shape exceeded bounded quota')
    return packed, resized_width, (width, height)


@contextmanager
def _hold_retained_stage_for_final_receipt(target, original_id):
    def check_original():
        source_freeze._safe_ancestors(target)
        identity = source_freeze._lstat(target)
        _require(stat.S_ISDIR(identity.st_mode) and
                 (identity.st_dev, identity.st_ino) == original_id,
                 'retained selected Rec input stage identity was replaced')

    if os.name != 'nt':
        check_original()
        yield
        check_original()
        return
    from .omdet_public_candidate import (
        OmDetPublicCandidateError, _hold_plain_ntfs_directory,
    )
    try:
        source_lock._Kernel32().assert_local_ntfs(target.parent)
        with ExitStack() as holders:
            for ancestor in reversed((target, *target.parents)):
                holders.enter_context(_hold_plain_ntfs_directory(ancestor))
            check_original()
            yield
            check_original()
    except (OmDetPublicCandidateError, source_lock.SourceLockError) as exc:
        raise SelectedRecInputRefused(
            'original retained Rec output NTFS namespace not held: %s' % exc
        ) from exc


def prepare(source_directory, dictionary, approved_tree_sha256,
            image_png, into):
    try:
        chosen = _model(source_directory, dictionary, approved_tree_sha256)
        image, image_sha, png_bytes, png_shape = ocr_png_preprocess._checked_source(
            image_png)
        target = ocr_ort_raw_capture._absolute(into, 'new original Rec input stage')
        _require(not os.path.lexists(target), 'output stage already exists')
        ocr_ort_raw_capture._safe_path(target, missing_leaf=True)
        _require(stat.S_ISDIR(source_freeze._lstat(target.parent).st_mode),
                 'existing local destination parent required')
        if os.name == 'nt':
            source_lock._Kernel32().assert_local_ntfs(target.parent)
        for owner in (Path(chosen['source_root']),
                      Path(chosen['dictionary']['reference']).parent):
            _require(owner != target and owner not in target.parents and
                     target not in owner.parents,
                     'output must not overlap selected original model/dictionary')
        _require(image != target, 'selected PNG may not be overwritten')
        _require(shutil.disk_usage(target.parent).free >= INPUT_BYTES + MAX_REPORT +
                 1024 * 1024,
                 'insufficient physical disk for retained actual input bytes')

        pending = None
        stage_id = None
        source_record = None
        try:
            with ocr_ort_raw_capture._held_files(
                    [(image, image_sha, len(png_bytes))]):
                _require(ocr_ort_raw_capture._sha_file(
                    image, 'held selected source PNG',
                    limit=ocr_png_preprocess.MAX_IMAGE_BYTES) ==
                    (image_sha, len(png_bytes)),
                    'source PNG changed before selected preprocessing')
                encoded, resized_width, decoded_shape = _encode_source_rec_png(
                    png_bytes)
                _require(decoded_shape == png_shape,
                         'decoded PNG image dimensions changed after IHDR admission')
                input_sha = hashlib.sha256(encoded).hexdigest()
                source_record = {
                    'schema': SCHEMA,
                    'status': STATUS,
                    'selected_source_tree_sha256': chosen['tree_sha256'],
                    'selected_source_status': 'SCANNED_NOT_LOCKED',
                    'dictionary_sha256': chosen['dictionary']['sha256'],
                    'reference_operator': 'DecodeImage_BGR_RecResizeImg_CHW_F32_ZERO_PAD',
                    'producer_independence': 'NUMPY_OPENCV_RECONSTRUCTION_NOT_VENDOR_EXECUTION',
                    'original_onnx_executed': False,
                    'native_pascal': 'NOT_RUN',
                    'human_ground_truth': 'NOT_ATTESTED',
                    'acceptance_status': 'NOT_CHECKED',
                    'verification_status': 'NOT_CHECKED',
                    'release_grade_immutable': False,
                    'facade_manifest_issued': False,
                    'ready_issued': False,
                    'source_png': {
                        'path': str(image), 'sha256': image_sha,
                        'bytes': len(png_bytes),
                        'width': png_shape[0], 'height': png_shape[1],
                    },
                    'input': {'file': _INPUT, 'name': INPUT_NAME,
                              'shape': list(SHAPE), 'dtype': 'float32le',
                              'bytes': INPUT_BYTES, 'sha256': input_sha,
                              'resized_width': resized_width,
                              'channel_order': 'BGR',
                              'normalized_padding': 0.0},
                }
                body = (json.dumps(source_record, sort_keys=True, indent=2,
                                   ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
                _require(len(body) <= MAX_REPORT, 'source-profile receipt exceeds quota')
                with ocr_ort_raw_capture._fresh_held_destination(target):
                    stamped = source_freeze._lstat(target)
                    stage_id = stamped.st_dev, stamped.st_ino
                    with (target / _INPUT).open('xb') as output:
                        output.write(encoded)
                        output.flush()
                        os.fsync(output.fileno())
                    _require(ocr_ort_raw_capture._sha_file(
                        target / _INPUT, 'selected Rec input.f32le',
                        limit=ocr_ort_raw_capture.MAX_INPUT_BYTES,
                        expected_bytes=INPUT_BYTES) == (input_sha, INPUT_BYTES),
                        'physical staged NCHW bytes differ from producer')
                    _require(_model(source_directory, dictionary,
                                    approved_tree_sha256) == chosen,
                             'pinned original graph/params/dictionary changed during preprocessing')
                    _require(ocr_ort_raw_capture._sha_file(
                        image, 'source PNG after selected preprocessing',
                        limit=ocr_png_preprocess.MAX_IMAGE_BYTES) ==
                        (image_sha, len(png_bytes)),
                        'source PNG changed before receipt reservation')


                    pending = (target / _RECEIPT).open('xb')


            with _hold_retained_stage_for_final_receipt(target, stage_id):
                _require(ocr_ort_raw_capture._sha_file(
                    target / _INPUT, 'selected Rec input post-hold SHA',
                    limit=ocr_ort_raw_capture.MAX_INPUT_BYTES,
                    expected_bytes=INPUT_BYTES) == (input_sha, INPUT_BYTES),
                    'staged model input bytes changed after native holder')
                receipt_info = os.fstat(pending.fileno())
                name_info = source_freeze._lstat(target / _RECEIPT)
                _require(stat.S_ISREG(name_info.st_mode) and
                         receipt_info.st_dev == name_info.st_dev and
                         receipt_info.st_ino == name_info.st_ino and
                         receipt_info.st_size == name_info.st_size == 0,
                         'reserved output receipt leaf FileID/size changed')
                with os.scandir(target) as members:
                    seen = []
                    for member in members:
                        seen.append(member.name)
                        _require(len(seen) <= 2,
                                 'extra member appeared in retained Rec stage')
                _require(set(seen) == {_INPUT, _RECEIPT} and len(seen) == 2,
                         'retained Rec input stage has unsupported members')
            pending.write(body)
            pending.flush()
            os.fsync(pending.fileno())
            receipt_owner = os.fstat(pending.fileno())
            pending.close()
            pending = None


            _require(ocr_ort_raw_capture._sha_file(
                target / _RECEIPT, 'post-audit input receipt',
                limit=MAX_REPORT, expected_bytes=len(body)) ==
                (hashlib.sha256(body).hexdigest(), len(body)),
                'post-audit selected Rec input receipt bytes differ')


            final_dir = source_freeze._lstat(target)
            final_receipt = source_freeze._lstat(target / _RECEIPT)
            _require(stat.S_ISDIR(final_dir.st_mode) and
                     (final_dir.st_dev, final_dir.st_ino) == stage_id,
                     'selected Rec final destination directory FileID changed')
            _require(stat.S_ISREG(final_receipt.st_mode) and
                     (final_receipt.st_dev, final_receipt.st_ino) ==
                     (receipt_owner.st_dev, receipt_owner.st_ino) and
                     final_receipt.st_size == receipt_owner.st_size == len(body),
                     'selected Rec final receipt is not the originally reserved leaf')
            _require(ocr_ort_raw_capture._sha_file(
                target / _INPUT, 'post-receipt selected Rec input.f32le',
                limit=ocr_ort_raw_capture.MAX_INPUT_BYTES,
                expected_bytes=INPUT_BYTES) == (input_sha, INPUT_BYTES),
                'selected Rec input bytes changed after receipt publication')
            with os.scandir(target) as entries:
                final_names = []
                for entry in entries:
                    final_names.append(entry.name)
                    _require(len(final_names) <= 2,
                             'selected Rec stage acquired extra post-publication member')
            _require(set(final_names) == {_INPUT, _RECEIPT} and
                     len(final_names) == 2,
                     'selected Rec stage final membership includes foreign markers')
        finally:
            if pending is not None:
                pending.close()
        return source_record
    except SelectedRecInputRefused:
        raise
    except (ocr_png_preprocess.PreprocessRefused,
            ocr_ort_raw_capture.CaptureRefused,
            paddle_closure.PaddleClosureError,
            source_freeze.SourceFreezeError, source_lock.SourceLockError,
            OSError, ValueError, TypeError, KeyError, IndexError,
            RuntimeError, MemoryError) as exc:
        raise SelectedRecInputRefused('real selected Rec input rejected: %s' % exc) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-rec-input',
        description=('Capture source-profile BGR/RecResizeImg float32 input for '
                     'the EXACT pinned PP-OCRv5 Server Rec revision; no model '
                     'execution, Pascal/ORT/numerical oracle, GT or READY.'))
    for flag in ('--source-dir', '--dictionary', '--approved-tree-sha256',
                 '--image', '--into'):
        parser.add_argument(flag, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        result = prepare(args.source_dir, args.dictionary,
                         args.approved_tree_sha256, args.image, args.into)
    except (SelectedRecInputRefused, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
                          'native_pascal': 'NOT_RUN', 'reason': str(exc)},
                         ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 2


__all__ = ['SelectedRecInputRefused', 'prepare', 'main']
