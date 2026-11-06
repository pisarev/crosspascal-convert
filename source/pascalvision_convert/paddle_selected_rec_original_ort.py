# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""REAL selected PP-OCRv5 Server Rec source-profile input -> actual ORT CPU.

The pinned selected Rec PIR/params/YAML/18383-token dictionary tree is NOT
the extra model.onnx itself. Require separate caller SHA approval of that
local original ONNX export and rehash before/after running the existing held,
bounded genuine ONNX Runtime child. No fake/model-mock success route exists.
The input is a retained SHA/NTFS-checked PNG->BGR float32 RecResizeImg receipt.

The full parent-document Det ROI, original Paddle/ONNX export equivalence,
native Pascal, CTC, human GT and release remain NOT_ATTESTED or NOT_RUN.
Positive real ORT produces advisory exit2, never verified/READY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from . import ocr_ort_raw_capture as raw
from . import paddle_selected_rec_input as rec
from . import source_freeze
from . import app_paths


STATUS = 'SELECTED_REAL_REC_ORIGINAL_ORT_RAW_ONLY_NOT_NATIVE_UAT'
_PINNED_TREE_SHA = rec._PINNED_TREE_SHA
_PINNED_DICT_SHA = rec._PINNED_DICT_SHA


_EXTRA_ONNX_SHA = 'd9dc333c9c7b042c6dffb8e33d72b6f65c9c1d463d0a3c2f78174fea55e94752'
_EXTRA_ONNX_BYTES = 84503027
_EVIDENCE_ROOT = app_paths.data_path('evidence')
_HEX = re.compile(r'[0-9a-f]{64}\Z')


class SelectedRecOracleRefused(ValueError):
    """Exact original PP-OCRv5 Rec ORT source/PNG/byte binding refused."""


def _require(condition, message):
    if not condition:
        raise SelectedRecOracleRefused(
            'selected original Rec ORT refused: ' + message)


def _read_pinned(path, label, *, limit, expected_size=None):
    try:
        return raw._sha_file(path, label, limit=limit,
                             expected_bytes=expected_size)
    except (raw.CaptureRefused, OSError, ValueError) as exc:
        raise SelectedRecOracleRefused('%s: %s' % (label, exc)) from exc


def _snapshot(stage):
    """Admit actual two-file selected Rec profile and original PNG bytes."""
    raw._safe_path(stage)
    original = source_freeze._lstat(stage)
    _require(stat.S_ISDIR(original.st_mode), 'Rec input stage is not a directory')
    names = []
    with os.scandir(stage) as iterator:
        for entry in iterator:
            names.append(entry.name)
            _require(len(names) <= 2, 'Rec stage contains a foreign member/READY')
    _require(len(names) == 2
             and set(names) == {'input.f32le', 'input-source.json'},
             'Rec input stage must have exactly its two source-profile files')
    receipt_path = stage / 'input-source.json'
    receipt_sha, length = _read_pinned(
        receipt_path, 'selected Rec source-profile receipt',
        limit=rec.MAX_REPORT)
    with receipt_path.open('rb') as source:
        body = source.read(rec.MAX_REPORT + 1)
    _require(len(body) == length and
             hashlib.sha256(body).hexdigest() == receipt_sha,
             'Rec selected input-source receipt changed during second read')

    receipt = raw._strict_json(body)
    _require(type(receipt) is dict
             and receipt.get('schema') == rec.SCHEMA
             and receipt.get('status') == rec.STATUS
             and receipt.get('selected_source_tree_sha256') == _PINNED_TREE_SHA
             and receipt.get('selected_source_status') == 'SCANNED_NOT_LOCKED'
             and receipt.get('dictionary_sha256') == _PINNED_DICT_SHA
             and receipt.get('reference_operator')
             == 'DecodeImage_BGR_RecResizeImg_CHW_F32_ZERO_PAD'
             and receipt.get('producer_independence')
             == 'NUMPY_OPENCV_RECONSTRUCTION_NOT_VENDOR_EXECUTION'
             and receipt.get('original_onnx_executed') is False
             and receipt.get('native_pascal') == 'NOT_RUN'
             and receipt.get('acceptance_status') == 'NOT_CHECKED'
             and receipt.get('verification_status') == 'NOT_CHECKED'
             and receipt.get('release_grade_immutable') is False
             and receipt.get('facade_manifest_issued') is False
             and receipt.get('ready_issued') is False,
             'selected original Rec profile provenance/statuses are contradictory')
    inp = receipt.get('input')
    _require(type(inp) is dict
             and inp.get('file') == 'input.f32le'
             and inp.get('name') == 'x'
             and inp.get('dtype') == 'float32le'
             and inp.get('shape') == list(rec.SHAPE)
             and type(inp.get('bytes')) is int
             and inp['bytes'] == rec.INPUT_BYTES
             and type(inp.get('sha256')) is str
             and _HEX.fullmatch(inp['sha256'])
             and type(inp.get('resized_width')) is int
             and 1 <= inp['resized_width'] <= 320
             and inp.get('channel_order') == 'BGR'
             and type(inp.get('normalized_padding')) is float
             and inp['normalized_padding'] == 0.0,
             'Rec input tensor does not match [1,3,48,320] float32 BGR zero-pad')
    input_sha, input_bytes = _read_pinned(
        stage / 'input.f32le', 'selected Rec float32 input bytes',
        limit=raw.MAX_INPUT_BYTES, expected_size=rec.INPUT_BYTES)
    _require((input_sha, input_bytes) == (inp['sha256'], rec.INPUT_BYTES),
             'selected Rec input bytes changed after producer receipt')
    png = receipt.get('source_png')
    _require(type(png) is dict and type(png.get('path')) is str
             and type(png.get('sha256')) is str
             and _HEX.fullmatch(png['sha256'])
             and type(png.get('bytes')) is int
             and 0 < png['bytes'] <= 32 * 1024 * 1024
             and type(png.get('width')) is int
             and type(png.get('height')) is int
             and png['width'] > 0 and png['height'] > 0
             and png['width'] * png['height'] <= 8_000_000,
             'selected Rec original crop PNG SHA/dimensions invalid')
    expected_width = min(320, max(
        1, (48 * png['width'] + png['height'] - 1) // png['height']))
    _require(expected_width == inp['resized_width'],
             'selected RecResizeImg ceil width inconsistent with source PNG')
    image = raw._absolute(png['path'], 'selected original Rec source PNG')
    _require(image.suffix.lower() == '.png',
             'selected source image must remain its original PNG')
    _require(_read_pinned(image, 'selected original Rec crop PNG',
                          limit=32 * 1024 * 1024,
                          expected_size=png['bytes'])
             == (png['sha256'], png['bytes']),
             'actual selected Rec crop image bytes differ from producer receipt')
    checked = source_freeze._lstat(stage)
    _require((checked.st_dev, checked.st_ino) ==
             (original.st_dev, original.st_ino),
             'selected Rec input stage FileID was replaced')
    return {
        'receipt_sha256': receipt_sha,
        'input_sha256': input_sha,
        'png_sha256': png['sha256'],
        'dictionary_sha256': receipt['dictionary_sha256'],
        'resized_width': expected_width,
    }


def run(source_dir, dictionary, approved_tree_sha256,
        approved_extra_onnx_sha256, input_stage, into):
    """Actual CPU ORT execution on selected Rec source-profile bytes, advisory."""
    _require(type(approved_tree_sha256) is str
             and approved_tree_sha256 == _PINNED_TREE_SHA
             and type(approved_extra_onnx_sha256) is str
             and approved_extra_onnx_sha256 == _EXTRA_ONNX_SHA,
             'explicit separate original Rec PIR tree and ONNX SHA pins required')

    raw._require_heavy_memory()
    source = rec._model(source_dir, dictionary, approved_tree_sha256)
    _require(source['dictionary']['sha256'] == _PINNED_DICT_SHA,
             'selected original Rec dictionary changed')
    owner = Path(source['source_root'])
    onnx = owner / 'model.onnx'
    onnx_pin = _read_pinned(
        onnx, 'extra original PP-OCRv5 Server Rec ONNX',
        limit=raw.MAX_MODEL_BYTES, expected_size=_EXTRA_ONNX_BYTES)
    _require(onnx_pin == (_EXTRA_ONNX_SHA, _EXTRA_ONNX_BYTES),
             'separately pinned original Rec model.onnx is not the approved file')
    stage = raw._absolute(input_stage, 'selected Rec input-source stage')
    _require(owner != stage and owner not in stage.parents
             and stage not in owner.parents,
             'source/input path overlap would mutate original')
    snapshot = _snapshot(stage)
    target = raw._absolute(into, 'fresh Rec original ORT output stage')
    evidence = raw._absolute(str(_EVIDENCE_ROOT), 'converter evidence directory')
    _require(target != evidence and evidence in target.parents
             and target not in owner.parents
             and target not in stage.parents
             and stage not in target.parents
             and not os.path.lexists(target),
             'Rec ORT --into must name a new subdirectory of the converter evidence directory')
    raw._safe_path(target, missing_leaf=True)
    _require(stat.S_ISDIR(source_freeze._lstat(target.parent).st_mode),
             'selected Rec ORT output parent must be a plain existing directory')
    observed = raw._parent_run(
        str(onnx), str(stage / 'input.f32le'),
        '1,3,48,320', str(target), 'x')
    _require(type(observed) is dict
             and observed.get('acceptance_status') == 'NOT_CHECKED'
             and observed.get('original_onnx_sha256') == _EXTRA_ONNX_SHA
             and observed.get('input_sha256') == snapshot['input_sha256']
             and type(observed.get('capture_sha256')) is str
             and _HEX.fullmatch(observed['capture_sha256'])
             and type(observed.get('outputs')) is list
             and len(observed['outputs']) > 0,
             'real original Rec ORT returned incomplete or unrelated source')
    _require(rec._model(source_dir, dictionary, approved_tree_sha256) == source,
             'Paddle PIR/params/dictionary/YAML changed during trained ORT child')
    _require(_read_pinned(
        onnx, 'extra original Rec model.onnx after CPU ORT',
        limit=raw.MAX_MODEL_BYTES, expected_size=_EXTRA_ONNX_BYTES) == onnx_pin,
        'extra original Rec ONNX changed during CPU ORT child')
    _require(_snapshot(stage) == snapshot,
             'selected original Rec input/crop receipt changed during ORT child')
    return {
        'status': STATUS, 'acceptance_status': 'NOT_CHECKED',
        'original_paddle_inference_executed': False,
        'original_onnx_cpu_executed': True,
        'original_onnx_relation_to_paddle_weights':
        'NOT_INDEPENDENTLY_ATTESTED',
        'selected_source_tree_sha256': approved_tree_sha256,
        'selected_dictionary_sha256': snapshot['dictionary_sha256'],
        'original_extra_onnx_sha256': _EXTRA_ONNX_SHA,
        'selected_crop_png_sha256': snapshot['png_sha256'],
        'actual_input_sha256': snapshot['input_sha256'],
        'selected_input_receipt_sha256': snapshot['receipt_sha256'],
        'capture_file': observed.get('capture_file'),
        'capture_sha256': observed['capture_sha256'],
        'outputs': observed['outputs'],
        'native_pascal': 'NOT_RUN', 'det_document_roi': 'NOT_RUN',
        'ctc': 'NOT_RUN', 'human_ground_truth': 'NOT_ATTESTED',
        'ready_issued': False, 'facade_manifest_issued': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-rec-original-ort',
        description='Run actual separately pinned original Rec CPU ONNX Runtime '
                    'on exact selected BGR/RecResizeImg PNG input; advisory ONLY.')
    for arg in ('--source-dir', '--dictionary', '--approved-tree-sha256',
                '--approved-original-onnx-sha256', '--input-stage', '--into'):
        parser.add_argument(arg, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        result = run(args.source_dir, args.dictionary,
                     args.approved_tree_sha256,
                     args.approved_original_onnx_sha256,
                     args.input_stage, args.into)
    except (SelectedRecOracleRefused, raw.CaptureRefused, OSError,
            ValueError, TypeError, KeyError, IndexError, UnicodeError,
            rec.SelectedRecInputRefused,
            source_freeze.SourceFreezeError) as exc:
        print(json.dumps({'status': 'REFUSED',
                          'acceptance_status': 'NOT_CHECKED',
                          'native_pascal': 'NOT_RUN',
                          'human_ground_truth': 'NOT_ATTESTED',
                          'ready_issued': False, 'reason': str(exc)},
                         ensure_ascii=False, allow_nan=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
