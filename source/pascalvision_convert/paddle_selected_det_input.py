# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys

from . import ocr_png_preprocess, ocr_ort_raw_capture, paddle_closure
from . import paddle_selected_rec_input as rec_stage
from . import source_freeze, source_lock


SCHEMA = 'crosspascal.step08.paddle-selected-ppocrv5-server-det-input.v1'
STATUS = 'SOURCE_PROFILE_PREPROCESSED_ONLY_NOT_MODEL_ORACLE'
SHAPE = (1, 3, 960, 960)
INPUT_NAME = 'x'
INPUT_BYTES = 4 * 1 * 3 * 960 * 960
_MAX_REPORT = 64 * 1024
_MODEL_TREE_SHA = (
    '64fc7a0f0e3671db054719de3b996f3b5b9d0a704545fe48180c1d3b1f018137')
_SOURCE_SHAS = {
    'inference.json': 'af5876933d8806a1b50d895867e0781e135cd92ff37381992828fc8d1b842d28',
    'inference.pdiparams': '183146fe9d9910352f68482f623bcbbb9fa7b9e8fa1463b9ad288cef00524d2d',
    'inference.yml': '28fb721efc3634fc8aa677e474b9602cb815a91cf569ef357a7a553d7b3ce685',
}
_MEAN_BGR = (0.485, 0.456, 0.406)
_STD_BGR = (0.229, 0.224, 0.225)
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_INPUT = 'input.f32le'
_RECEIPT = 'input-source.json'


class SelectedDetInputRefused(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise SelectedDetInputRefused('selected PP-OCRv5 Server Det input refused: ' + message)


def _model(source_dir, approved_sha):
    _require(type(approved_sha) is str and _HEX.fullmatch(approved_sha),
             'lowercase 64-digit caller-approved source tree SHA required')
    _require(approved_sha == _MODEL_TREE_SHA,
             'source SHA must match exact pinned original Det Server revision')
    try:
        chosen = paddle_closure.audit_selected_paddle_source(
            source_dir, role='det', expected_tree_sha256=approved_sha)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SelectedDetInputRefused('pinned original Det source changed: %s' % exc) from exc
    rows = {row['path']: row['sha256'] for row in chosen['entries']}
    _require(chosen['file_count'] == 3 and rows == _SOURCE_SHAS and
             chosen['tree_sha256'] == _MODEL_TREE_SHA,
             'genuine Det graph/weights/inference.yml SHA closure differs')
    _require(chosen['verification_status'] == 'NOT_CHECKED' and
             chosen['release_grade_immutable'] is False,
             'scanned input must not claim trained/immutable source verification')
    return chosen


def _encode_original_det_square(png_bytes):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SelectedDetInputRefused('NumPy and OpenCV are required') from exc
    bgr = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    _require(bgr is not None and bgr.ndim == 3 and bgr.dtype == np.uint8 and
             bgr.shape[2] == 3 and
             0 < bgr.shape[0] * bgr.shape[1] <=
             ocr_png_preprocess.MAX_IMAGE_PIXELS,
             'source lossless PNG did not decode to bounded BGR8 pixels')
    height, width = int(bgr.shape[0]), int(bgr.shape[1])
    _require(width == height,
             'non-square images cannot prove vendor resize_long960 equivalence '
             'to this static 960x960 TVM graph; this diagnostic requires a square image')
    resized = cv2.resize(bgr, (SHAPE[3], SHAPE[2]),
                         interpolation=cv2.INTER_LINEAR)


    normalized = np.empty((SHAPE[2], SHAPE[3], 3), dtype=np.float32)
    scalar_scale = np.float32(1.0 / 255.0)
    for c in range(3):
        normalized[:, :, c] = (
            (resized[:, :, c].astype(np.float32) * scalar_scale -
             np.float32(_MEAN_BGR[c])) / np.float32(_STD_BGR[c]))
    _require(np.isfinite(normalized).all(),
             'source-profile Det preprocessing produced a nonfinite float')
    packed = np.ascontiguousarray(normalized.transpose(2, 0, 1),
                                  dtype=np.dtype('<f4')).tobytes(order='C')
    _require(len(packed) == INPUT_BYTES and
             INPUT_BYTES <= ocr_ort_raw_capture.MAX_INPUT_BYTES,
             'static selected Det tensor exceeded exact input byte quota')
    return packed, (width, height)


def prepare(source_dir, approved_sha, image_png, into):
    try:
        chosen = _model(source_dir, approved_sha)
        image, image_sha, png_bytes, png_dims = ocr_png_preprocess._checked_source(
            image_png)
        target = ocr_ort_raw_capture._absolute(into, 'fresh selected Det input stage')
        _require(not os.path.lexists(target), 'new Det output directory occupied')
        ocr_ort_raw_capture._safe_path(target, missing_leaf=True)
        _require(stat.S_ISDIR(source_freeze._lstat(target.parent).st_mode),
                 'fresh Det stage must have an existing ordinary parent directory')
        if os.name == 'nt':
            source_lock._Kernel32().assert_local_ntfs(target.parent)
        owner = Path(chosen['source_root'])
        _require(target != owner and target not in owner.parents and
                 owner not in target.parents and image != target,
                 'Det input evidence path overlaps pinned original model or PNG')
        _require(shutil.disk_usage(target.parent).free >=
                 INPUT_BYTES + _MAX_REPORT + 1024 * 1024,
                 'insufficient free disk for create-only 11MiB input stage')

        pending = None
        stage_id = None
        try:
            with ocr_ort_raw_capture._held_files(
                    [(image, image_sha, len(png_bytes))]):
                _require(ocr_ort_raw_capture._sha_file(
                    image, 'held original selected Det PNG',
                    limit=ocr_png_preprocess.MAX_IMAGE_BYTES) ==
                    (image_sha, len(png_bytes)),
                    'real Det PNG changed before source-profile preprocessing')
                tensor, dimensions = _encode_original_det_square(png_bytes)
                _require(dimensions == png_dims,
                         'Det PNG IHDR differs from decoded actual image dimensions')
                tensor_sha = hashlib.sha256(tensor).hexdigest()
                receipt = {
                    'schema': SCHEMA,
                    'status': STATUS,
                    'acceptance_status': 'NOT_CHECKED',
                    'verification_status': 'NOT_CHECKED',
                    'model_tree_sha256': chosen['tree_sha256'],
                    'model_source_status': 'SCANNED_NOT_LOCKED',
                    'model_release_grade_immutable': False,
                    'source_profile': {
                        'original_inference_yml_sha256': _SOURCE_SHAS['inference.yml'],
                        'original_decode_mode': 'BGR',
                        'resize_long': 960,
                        'approved_input_geometry': 'SQUARE_IMAGE_ONLY',
                        'resize': 'INTER_LINEAR to 960x960',
                        'normalization_order': 'HWC',
                        'normalization_scale': '1./255.',
                        'mean_BGR': list(_MEAN_BGR),
                        'std_BGR': list(_STD_BGR),
                        'layout': 'NCHW',
                        'independence': 'NUMPY_OPENCV_RECONSTRUCTION_NOT_VENDOR_EXECUTION',
                    },
                    'source_png': {'path': str(image), 'sha256': image_sha,
                                   'bytes': len(png_bytes),
                                   'width': png_dims[0], 'height': png_dims[1]},
                    'input': {'file': _INPUT, 'name': INPUT_NAME,
                              'shape': list(SHAPE), 'dtype': 'float32le',
                              'sha256': tensor_sha, 'bytes': len(tensor)},
                    'original_onnx_executed': False,
                    'native_pascal': 'NOT_RUN',
                    'human_ground_truth': 'NOT_ATTESTED',
                    'facade_manifest_issued': False,
                    'ready_issued': False,
                }
                body = (json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                                   allow_nan=False, indent=2) + '\n').encode('utf-8')
                _require(len(body) <= _MAX_REPORT, 'bounded Det input receipt exceeded')
                with ocr_ort_raw_capture._fresh_held_destination(target):
                    info = source_freeze._lstat(target)
                    stage_id = (info.st_dev, info.st_ino)
                    with (target / _INPUT).open('xb') as dest:
                        dest.write(tensor)
                        dest.flush()
                        os.fsync(dest.fileno())
                    _require(ocr_ort_raw_capture._sha_file(
                        target / _INPUT, 'selected Det actual input.f32le',
                        limit=ocr_ort_raw_capture.MAX_INPUT_BYTES,
                        expected_bytes=len(tensor)) == (tensor_sha, len(tensor)),
                        'selected Det input output differs from calculated data')
                    _require(_model(source_dir, approved_sha) == chosen,
                             'genuine Det graph/weights/YAML changed during conversion')
                    _require(ocr_ort_raw_capture._sha_file(
                        image, 'selected Det PNG before receipt reservation',
                        limit=ocr_png_preprocess.MAX_IMAGE_BYTES) ==
                        (image_sha, len(png_bytes)),
                        'Det PNG source changed before receipt reservation')
                    pending = (target / _RECEIPT).open('xb')


            with rec_stage._hold_retained_stage_for_final_receipt(target, stage_id):
                _require(ocr_ort_raw_capture._sha_file(
                    target / _INPUT, 'post-closure selected Det original input',
                    limit=ocr_ort_raw_capture.MAX_INPUT_BYTES,
                    expected_bytes=len(tensor)) == (tensor_sha, len(tensor)),
                    'original Det input changed after two native holder finalizers')
                fd = os.fstat(pending.fileno())
                leaf = source_freeze._lstat(target / _RECEIPT)
                _require(stat.S_ISREG(leaf.st_mode) and
                         fd.st_dev == leaf.st_dev and fd.st_ino == leaf.st_ino and
                         fd.st_size == leaf.st_size == 0,
                         'reserved Det receipt no longer references our new zero-byte leaf')
                with os.scandir(target) as iterator:
                    members = []
                    for member in iterator:
                        members.append(member.name)
                        _require(len(members) <= 2,
                                 'foreign stage member appeared during Det publication')
                _require(set(members) == {_INPUT, _RECEIPT} and len(members) == 2,
                         'Det original input stage contains unexpected member/READY')


            pending.write(body)
            pending.flush()
            os.fsync(pending.fileno())
            receipt_owner = os.fstat(pending.fileno())
            pending.close()
            pending = None
            _require(ocr_ort_raw_capture._sha_file(
                target / _RECEIPT, 'selected Det input-source.json',
                limit=_MAX_REPORT, expected_bytes=len(body)) ==
                (hashlib.sha256(body).hexdigest(), len(body)),
                'published Det input receipt bytes differ from source record')
            observed_stage = source_freeze._lstat(target)
            observed_receipt = source_freeze._lstat(target / _RECEIPT)
            _require(stat.S_ISDIR(observed_stage.st_mode) and
                     (observed_stage.st_dev, observed_stage.st_ino) == stage_id,
                     'selected Det final stage FileID changed after receipt')
            _require(stat.S_ISREG(observed_receipt.st_mode) and
                     (observed_receipt.st_dev, observed_receipt.st_ino) ==
                     (receipt_owner.st_dev, receipt_owner.st_ino) and
                     observed_receipt.st_size == receipt_owner.st_size == len(body),
                     'selected Det final receipt no longer has reserved FileID')
            _require(ocr_ort_raw_capture._sha_file(
                target / _INPUT, 'selected Det input after receipt',
                limit=ocr_ort_raw_capture.MAX_INPUT_BYTES,
                expected_bytes=len(tensor)) == (tensor_sha, len(tensor)),
                'selected Det input bytes differ after receipt publication')
            with os.scandir(target) as iterator:
                after_names = []
                for member in iterator:
                    after_names.append(member.name)
                    _require(len(after_names) <= 2,
                             'foreign post-publication Det stage member found')
            _require(set(after_names) == {_INPUT, _RECEIPT} and
                     len(after_names) == 2,
                     'selected Det stage final inventory has extra member/READY')
        finally:
            if pending is not None:
                pending.close()
        return receipt
    except SelectedDetInputRefused:
        raise
    except (ocr_png_preprocess.PreprocessRefused,
            ocr_ort_raw_capture.CaptureRefused,
            paddle_closure.PaddleClosureError,
            source_freeze.SourceFreezeError, source_lock.SourceLockError,
            rec_stage.SelectedRecInputRefused,
            OSError, ValueError, TypeError, KeyError, IndexError,
            RuntimeError, MemoryError) as exc:
        raise SelectedDetInputRefused('selected original Det source/input refused: %s' % exc) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-det-input',
        description=('Prepare held and SHA-checked PP-OCRv5 Server Det BGR '
                     'float32 input from a genuine SQUARE PNG for exact '
                     'static [1,3,960,960], no model/ORT/Pascal/GT evidence.'))
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--approved-tree-sha256', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--into', required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        result = prepare(args.source_dir, args.approved_tree_sha256,
                         args.image, args.into)
    except (SelectedDetInputRefused, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
                          'native_pascal': 'NOT_RUN', 'ready_issued': False,
                          'reason': str(exc)}, ensure_ascii=False,
                         allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 2


__all__ = ['SelectedDetInputRefused', 'prepare', 'main']
