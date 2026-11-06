# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Prepare one *real PNG* as a bounded, advisory original-ORT OCR input.

This executes image decoding, OpenCV interpolation and the explicit OCR DRAFT
preprocessing formula. It does NOT run ONNX Runtime, Pascal, Det->crop->Rec,
CTC or a human oracle. A source-image SHA and a generated float32 tensor are
file-integrity observations, NOT proof of producer independence or native UAT.

No existing input or destination is overwritten, moved or cleaned up. Failed
attempts retain their newly created private evidence directory for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

from . import ocr_capture_bridge as bridge
from . import ocr_composite_oracle as oracle
from . import ocr_ort_raw_capture as raw_producer
from . import source_freeze


SCHEMA = 'crosspascal.paddleocr.png-input-preparation'
SCHEMA_VERSION = 1
STATUS = 'PREPROCESSED_INPUT_ONLY'
MAX_IMAGE_BYTES = oracle.MAX_IMAGE_BYTES
MAX_IMAGE_PIXELS = oracle.MAX_IMAGE_PIXELS
MAX_TENSOR_BYTES = raw_producer.MAX_INPUT_BYTES
MAX_REPORT_BYTES = 64 * 1024
_TENSOR = 'input.f32le'
_REPORT = 'input-source.json'


class PreprocessRefused(ValueError):
    """The original RGB8 PNG cannot safely produce this draft tensor."""


def _require(condition, reason):
    if not condition:
        raise PreprocessRefused(reason)


def _checked_source(png):
    """Read one bounded PNG and validate its *actual* bytes, not its suffix."""
    path = raw_producer._absolute(png, 'source RGB8 PNG')
    _require(path.suffix.lower() == '.png', 'source image must be an 8-bit truecolor (RGB) PNG')
    info = raw_producer._regular(path, 'source RGB8 PNG',
                                  limit=MAX_IMAGE_BYTES)
    digest, size = raw_producer._sha_file(
        path, 'source RGB8 PNG', limit=MAX_IMAGE_BYTES)
    _require(size == info.st_size, 'source PNG byte length changed')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        _require(stat.S_ISREG(opened.st_mode) and
                 raw_producer._identity(opened) == raw_producer._identity(info),
                 'source PNG changed FileID while opening')
        raw = stream.read(MAX_IMAGE_BYTES + 1)
        _require(len(raw) == size, 'source PNG changed size during bounded read')
    _require(hashlib.sha256(raw).hexdigest() == digest,
             'source PNG bytes changed between SHA and decoder admission')
    _require(raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw) >= 33,
             'source image must contain a genuine PNG IHDR')
    width, height = struct.unpack_from('>II', raw, 16)
    _require(0 < width <= 16384 and 0 < height <= 16384 and
             width * height <= MAX_IMAGE_PIXELS,
             'source PNG image exceeds bounded pixel quota')
    try:
        info_png = oracle._validated_source_png(
            path, width, height, expected_sha=digest)
    except (oracle.InvalidEvidence, oracle.MissingEvidence) as exc:
        raise PreprocessRefused('invalid unsupported PNG image: %s' % exc) from exc
    _require(info_png['format'] == 'lossless_png_rgb8',
             'OCR preprocessing requires an 8-bit truecolor (RGB) PNG')
    return path, digest, raw, (width, height)


def _preprocess_bgr(png_bytes, shape, stage):
    """Perform OpenCV bilinear resizing with exact DRAFT math and bounded RAM.

    The same Python implementation is NOT an independent witness of Pascal's
    internal floating-point rounding or original Paddle vendor preprocessing.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise PreprocessRefused('NumPy and OpenCV are required to decode a PNG') from exc
    decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    _require(decoded is not None and decoded.ndim == 3 and
             decoded.shape[2] == 3 and decoded.dtype == np.uint8 and
             0 < decoded.shape[0] * decoded.shape[1] <= MAX_IMAGE_PIXELS,
             'OpenCV did not return bounded decoded BGR8 PNG pixels')
    height, width = shape[2], shape[3]
    _require(height * width * 3 <= raw_producer.MAX_INPUT_ELEMENTS,
             'preprocessing destination tensor exceeds float32 byte quota')
    if stage == 'det':


        bgr = cv2.resize(decoded, (width, height),
                         interpolation=cv2.INTER_LINEAR)
        mean, std = (0.406, 0.456, 0.485), (0.225, 0.224, 0.229)
        normalized = np.empty((height, width, 3), dtype=np.float32)
        for channel in range(3):
            normalized[:, :, channel] = (
                (bgr[:, :, channel].astype(np.float64) / 255.0
                 - mean[channel]) / std[channel]).astype(np.float32)
        resized_width = width
    else:

        ratio = decoded.shape[1] / max(1, decoded.shape[0])
        resized_width = min(width, int(math.ceil(height * ratio)))
        resized_width = max(1, resized_width - resized_width % 8)
        resized = cv2.resize(decoded, (resized_width, height),
                             interpolation=cv2.INTER_LINEAR)
        bgr = np.zeros((height, width, 3), dtype=np.uint8)
        bgr[:, :resized_width, :] = resized


        normalized = (bgr[:, :, ::-1].astype(np.float64) / 255.0 * 2.0
                      - 1.0).astype(np.float32)
        if resized_width < width:
            _require(bool(np.all(normalized[:, resized_width:, :] == -1.0)),
                     'REC right padding is not the required tensor value -1')
    _require(bool(np.isfinite(normalized).all()),
             'actual normalized input contains NaN or infinity')
    encoded = np.ascontiguousarray(normalized.transpose(2, 0, 1),
                                   dtype=np.dtype('<f4')).tobytes(order='C')
    _require(0 < len(encoded) <= MAX_TENSOR_BYTES and
             len(encoded) == 4 * shape[0] * shape[1] * height * width,
             'encoded little-endian NCHW tensor has invalid byte count')
    return encoded, resized_width, tuple(map(int, decoded.shape[:2]))


def prepare_png_input(bundle, stage, source_png, destination) -> dict:
    """Create-only one PNG-derived ORT input; no model or GT is executed.

    Output directory and all selected files are exclusively created; the
    Win32 branch holds every destination NTFS ancestor against reparse swaps.
    An advisory receipt is issued only after bounded source and output SHAs.
    """
    _require(type(stage) is str and stage in ('det', 'rec'),
             'stage must be det or rec')
    try:
        bundle_root, plan, plan_sha = oracle._validate_plan(bundle)
        name, shape = bridge._input_contract(plan, bundle_root, stage)
        source, source_sha, png_bytes, dims = _checked_source(source_png)
        target = raw_producer._absolute(destination, 'fresh PNG input destination')
        _require(not os.path.lexists(target),
                 'PNG preprocessing destination exists; fresh --into required')
        raw_producer._safe_path(target, missing_leaf=True)
        _require(stat.S_ISDIR(source_freeze._lstat(target.parent).st_mode),
            'PNG preprocessing destination parent must be an existing directory')


        _require(target != bundle_root and bundle_root not in target.parents and
                 target not in bundle_root.parents and target != source,
                 'PNG destination overlaps selected source/bundle tree')
        with raw_producer._held_files([(source, source_sha, len(png_bytes))]):


            _require(raw_producer._sha_file(
                source, 'held source PNG', limit=MAX_IMAGE_BYTES) ==
                (source_sha, len(png_bytes)), 'source PNG changed before decoding')
            encoded, resized_width, decoded_dims = _preprocess_bgr(
                png_bytes, shape, stage)
            _require(decoded_dims == (dims[1], dims[0]),
                     'OpenCV decoder dimensions disagree with validated PNG IHDR')
            input_sha = hashlib.sha256(encoded).hexdigest()
            report = {
                'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
                'status': STATUS, 'acceptance_status': 'NOT_CHECKED',
                'composite_verification': 'NOT_CHECKED',
                'original_onnx_executed': False,
                'native_pascal': 'NOT_RUN',
                'human_ground_truth': 'NOT_ATTESTED',
                'producer_authenticity': 'NOT_ATTESTED',
                'facade_manifest_issued': False, 'ready_issued': False,
                'stage': stage, 'plan_sha256': plan_sha,
                'source_image_sha256': source_sha,
                'source_image': {'path': str(source), 'sha256': source_sha,
                                 'bytes': len(png_bytes),
                                 'width': dims[0], 'height': dims[1]},
                'input': {'file': _TENSOR, 'name': name, 'shape': list(shape),
                          'dtype': 'float32le', 'bytes': len(encoded),
                          'sha256': input_sha},
                'preprocessing': {'selected_plan_stage': stage,
                                  'bilinear_opencv_input_width': resized_width,
                                  'measured_onnx_or_native_match': False},
                'evidence_limit': 'PNG-derived input only; no actual ORT, Pascal, '
                                  'CTC, original-image/crop human GT or READY.',
            }
            record_bytes = (json.dumps(report, ensure_ascii=False,
                                       sort_keys=True, indent=2,
                                       allow_nan=False) + '\n').encode('utf-8')
            _require(len(record_bytes) <= MAX_REPORT_BYTES,
                     'PNG input receipt exceeds bounded JSON byte quota')

            _require(raw_producer._sha_file(
                source, 'source PNG before capture', limit=MAX_IMAGE_BYTES) ==
                (source_sha, len(png_bytes)), 'source PNG changed during preprocessing')
            with raw_producer._fresh_held_destination(target):
                with (target / _TENSOR).open('xb') as output:
                    output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                observed, count = raw_producer._sha_file(
                    target / _TENSOR, 'prepared PNG input',
                    limit=MAX_TENSOR_BYTES, expected_bytes=len(encoded))
                _require((observed, count) == (input_sha, len(encoded)),
                         'prepared input bytes changed during exclusive creation')
                _require(raw_producer._sha_file(
                    source, 'original PNG', limit=MAX_IMAGE_BYTES) ==
                    (source_sha, len(png_bytes)),
                    'source PNG changed before final create-only receipt')
                with (target / _REPORT).open('xb') as output:
                    output.write(record_bytes)
                    output.flush()
                    os.fsync(output.fileno())
                _require(raw_producer._sha_file(
                    target / _REPORT, 'prepared input receipt',
                    limit=MAX_REPORT_BYTES, expected_bytes=len(record_bytes)) ==
                    (hashlib.sha256(record_bytes).hexdigest(), len(record_bytes)),
                    'prepared input receipt changed during final review')
        return report
    except PreprocessRefused:
        raise
    except (raw_producer.CaptureRefused, bridge.CaptureBridgeRefused,
            oracle.InvalidEvidence, oracle.MissingEvidence, OSError,
            ValueError, TypeError, KeyError, IndexError, MemoryError) as exc:
        raise PreprocessRefused('PNG OCR preprocessing refused: %s' % exc) from exc


def main(argv=None) -> int:
    """Standalone public opt-in PNG preprocessing, never model acceptance."""
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert ocr-png-input',
        description=('Prepare an 8-bit truecolor (RGB) PNG as one OCR Det/Rec '
                     'NCHW float32le tensor using the exact DRAFT preprocessing '
                     'formula. No ONNX Runtime/Pascal/human UAT is performed.'))
    parser.add_argument('--bundle', required=True,
                        help='existing OCR composite DRAFT v2 directory')
    parser.add_argument('--stage', choices=('det', 'rec'), required=True,
                        help='which exact DRAFT preprocessing stage to measure')
    parser.add_argument('--image', required=True,
                        help='existing noninterlaced 8-bit truecolor (RGB) PNG')
    parser.add_argument('--into', required=True,
                        help='fresh, previously unoccupied local output directory')
    options = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        result = prepare_png_input(options.bundle, options.stage,
                                   options.image, options.into)
    except (PreprocessRefused, OSError, ValueError) as exc:
        print(json.dumps({
            'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
            'native_pascal': 'NOT_RUN',
            'human_ground_truth': 'NOT_ATTESTED',
            'original_onnx_executed': False,
            'facade_manifest_issued': False, 'ready_issued': False,
            'reason': str(exc)}, ensure_ascii=False, allow_nan=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))

    return 2


__all__ = ['PreprocessRefused', 'prepare_png_input', 'main']


if __name__ == '__main__':
    raise SystemExit(main())
