# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Selected REAL PP-OCRv5 Server Det input -> separately pinned original ORT.

This executes the ORIGINAL-ONNX consumer of the actual selected Det input
receipt, not another reimplementation of image preprocessing. The pinned
inference.json/pdiparams/yml SHA closure does NOT include separately exported
model.onnx: explicitly pin its independent SHA and byte length. The relation
of that exported ONNX to the Paddle PIR weights remains unverified.

Invokes the existing bounded, held, actual CPU ONNX Runtime child ONLY after
RAM admission. The original source and selected input are checked before and
after. No Delphi/Pascal, DBPostProcess, Rec/CTC, human GT or VERIFIED/READY.
Even successful real ORT is advisory exit 2. All writes remain create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from . import ocr_ort_raw_capture as raw
from . import paddle_selected_det_input as det
from . import source_freeze
from . import app_paths


STATUS = 'SELECTED_REAL_DET_ORIGINAL_ORT_RAW_ONLY_NOT_NATIVE_UAT'
_TREE_SHA = det._MODEL_TREE_SHA


_EXTRA_ONNX_SHA = '10803475a591f7dc623e24670fb5752ec94d39a1f8cf069aac1b6f0ce19cfc85'
_EXTRA_ONNX_BYTES = 88116791
_EVIDENCE_ROOT = app_paths.data_path('evidence')
_RECEIPT_MAX_BYTES = 64 * 1024
_SHA = re.compile(r'[0-9a-f]{64}\Z')


class SelectedDetOracleRefused(ValueError):
    """No original Det source-to-ORT identity was established."""


def _require(condition, message):
    if not condition:
        raise SelectedDetOracleRefused(
            'selected original Det ORT refused: ' + message)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SelectedDetOracleRefused('duplicate receipt JSON key: ' + key)
        result[key] = value
    return result


def _reject_nonfinite(name):
    raise SelectedDetOracleRefused('nonfinite JSON receipt number: ' + name)


def _sha_regular(path, label, *, limit, expected_size=None):
    try:
        return raw._sha_file(path, label, limit=limit,
                             expected_bytes=expected_size)
    except (raw.CaptureRefused, OSError, ValueError) as exc:
        raise SelectedDetOracleRefused('%s: %s' % (label, exc)) from exc


def _stage_snapshot(stage):
    """Read exact two-file source-profile stage; no READY or foreign members."""
    raw._safe_path(stage)
    identity = source_freeze._lstat(stage)
    _require(stat.S_ISDIR(identity.st_mode),
             'source-profile input stage is not an ordinary directory')
    names = []
    with os.scandir(stage) as iterator:
        for member in iterator:
            names.append(member.name)
            _require(len(names) <= 2, 'foreign files in selected input stage')
    _require(set(names) == {'input.f32le', 'input-source.json'}
             and len(names) == 2,
             'selected stage must contain only input.f32le and input-source.json')
    receipt_path = stage / 'input-source.json'
    receipt_sha, size = _sha_regular(
        receipt_path, 'selected original input receipt',
        limit=_RECEIPT_MAX_BYTES)
    with receipt_path.open('rb') as stream:
        body = stream.read(_RECEIPT_MAX_BYTES + 1)
    _require(len(body) == size and hashlib.sha256(body).hexdigest() == receipt_sha,
             'selected input receipt changed during JSON read')
    try:
        receipt = json.loads(body.decode('utf-8'),
                             object_pairs_hook=_no_duplicate_keys,
                             parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise SelectedDetOracleRefused('invalid UTF-8 input-source receipt: %s'
                                        % exc) from exc
    _require(type(receipt) is dict
             and receipt.get('schema') == det.SCHEMA
             and receipt.get('status') == det.STATUS
             and receipt.get('acceptance_status') == 'NOT_CHECKED'
             and receipt.get('verification_status') == 'NOT_CHECKED'
             and receipt.get('model_tree_sha256') == _TREE_SHA
             and receipt.get('model_source_status') == 'SCANNED_NOT_LOCKED'
             and receipt.get('original_onnx_executed') is False
             and receipt.get('ready_issued') is False,
             'wrong selected Det input source provenance or statuses')
    profile = receipt.get('source_profile')
    _require(type(profile) is dict
             and profile.get('original_inference_yml_sha256')
             == det._SOURCE_SHAS['inference.yml']
             and profile.get('approved_input_geometry') == 'SQUARE_IMAGE_ONLY'
             and profile.get('independence')
             == 'NUMPY_OPENCV_RECONSTRUCTION_NOT_VENDOR_EXECUTION',
             'input not prepared with selected BGR square source profile')
    tensor = receipt.get('input')
    _require(type(tensor) is dict and tensor.get('file') == 'input.f32le'
             and tensor.get('name') == 'x'
             and tensor.get('dtype') == 'float32le'
             and tensor.get('shape') == list(det.SHAPE)
             and type(tensor.get('bytes')) is int
             and tensor['bytes'] == det.INPUT_BYTES
             and type(tensor.get('sha256')) is str
             and _SHA.fullmatch(tensor['sha256']),
             'selected Det tensor shape/bytes/name/SHA disagrees with receipt')
    input_sha, input_size = _sha_regular(
        stage / 'input.f32le', 'actual selected Det float32 input',
        limit=raw.MAX_INPUT_BYTES, expected_size=det.INPUT_BYTES)
    _require((input_sha, input_size) ==
             (tensor['sha256'], det.INPUT_BYTES),
             'selected Det tensor differs from input-source receipt')
    png = receipt.get('source_png')
    _require(type(png) is dict and type(png.get('path')) is str
             and type(png.get('sha256')) is str
             and _SHA.fullmatch(png['sha256'])
             and type(png.get('bytes')) is int and 0 < png['bytes'] <= 32 * 1024 * 1024
             and type(png.get('width')) is int and type(png.get('height')) is int
             and 0 < png['width'] == png['height']
             and png['width'] * png['height'] <= 8_000_000,
             'selected original PNG path, SHA or square dimensions absent')
    image = raw._absolute(png['path'], 'real selected Det PNG')
    _require(image.suffix.lower() == '.png', 'original input must be a PNG')
    _require(_sha_regular(image, 'real selected Det PNG', limit=32 * 1024 * 1024,
                          expected_size=png['bytes']) ==
             (png['sha256'], png['bytes']),
             'selected Det original PNG no longer matches source receipt')
    after = source_freeze._lstat(stage)
    _require((after.st_dev, after.st_ino) == (identity.st_dev, identity.st_ino),
             'selected Det stage FileID changed during receipt inspection')
    return {'receipt_sha256': receipt_sha, 'tensor_sha256': input_sha,
            'image_sha256': png['sha256']}


def run(source_dir, tree_sha, extra_onnx_sha, input_stage, into):
    """Run independently SHA-pinned original Det ONNX on selected input bytes."""
    _require(type(tree_sha) is str and tree_sha == _TREE_SHA
             and type(extra_onnx_sha) is str
             and extra_onnx_sha == _EXTRA_ONNX_SHA,
             'explicit exact PIR tree and extra ONNX SHA approvals required')


    raw._require_heavy_memory()
    selected = det._model(source_dir, tree_sha)
    root = Path(selected['source_root'])
    onnx = root / 'model.onnx'
    observed = _sha_regular(
        onnx, 'separately pinned extra original Det model.onnx',
        limit=raw.MAX_MODEL_BYTES, expected_size=_EXTRA_ONNX_BYTES)
    _require(observed == (_EXTRA_ONNX_SHA, _EXTRA_ONNX_BYTES),
             'extra original ONNX export changed independently of PIR tree')
    stage = raw._absolute(input_stage, 'selected original Det input stage')
    _require(stage != root and root not in stage.parents and stage not in root.parents,
             'input stage overlaps original source root')
    snapshot = _stage_snapshot(stage)
    target = raw._absolute(into, 'new selected original Det ORT evidence')
    evidence = raw._absolute(str(_EVIDENCE_ROOT), 'converter evidence directory')
    _require(target != evidence and evidence in target.parents
             and target not in root.parents
             and target not in stage.parents
             and stage not in target.parents,
             'ORT output must be a new directory beneath the converter evidence directory')
    _require(not os.path.lexists(target), 'ORT evidence folder already exists')
    raw._safe_path(target, missing_leaf=True)
    _require(stat.S_ISDIR(source_freeze._lstat(target.parent).st_mode),
             'ORT evidence parent must be an existing ordinary directory')


    output = raw._parent_run(str(onnx), str(stage / 'input.f32le'),
                             '1,3,960,960', str(target), 'x')
    _require(type(output) is dict
             and output.get('acceptance_status') == 'NOT_CHECKED'
             and output.get('original_onnx_sha256') == _EXTRA_ONNX_SHA
             and output.get('input_sha256') == snapshot['tensor_sha256']
             and type(output.get('capture_sha256')) is str
             and _SHA.fullmatch(output['capture_sha256'])
             and type(output.get('outputs')) is list and len(output['outputs']) > 0,
             'raw ORT child returned incomplete/wrong-source advisory receipt')
    _require(det._model(source_dir, tree_sha) == selected,
             'original Paddle PIR/params/YAML changed during ORT execution')
    _require(_sha_regular(
        onnx, 'extra original Det ONNX after ORT',
        limit=raw.MAX_MODEL_BYTES, expected_size=_EXTRA_ONNX_BYTES) == observed,
        'extra ONNX export changed after ORT child')
    _require(_stage_snapshot(stage) == snapshot,
             'selected original Det PNG/input stage changed after ORT child')
    return {'status': STATUS, 'acceptance_status': 'NOT_CHECKED',
            'original_paddle_inference_executed': False,
            'original_onnx_cpu_executed': True,
            'original_onnx_relation_to_paddle_weights':
            'NOT_INDEPENDENTLY_ATTESTED',
            'model_tree_sha256': tree_sha,
            'original_extra_onnx_sha256': _EXTRA_ONNX_SHA,
            'actual_input_sha256': snapshot['tensor_sha256'],
            'selected_input_receipt_sha256': snapshot['receipt_sha256'],
            'capture_sha256': output['capture_sha256'],
            'capture_file': output.get('capture_file'),
            'outputs': output['outputs'],
            'native_pascal': 'NOT_RUN', 'det_db_postprocess': 'NOT_RUN',
            'rec_ctc': 'NOT_RUN', 'human_ground_truth': 'NOT_ATTESTED',
            'ready_issued': False, 'facade_manifest_issued': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-det-original-ort',
        description='Run pinned extra original Det ONNX on exact selected Det '
                    'input receipt (actual CPU ORT advisory; NOT Pascal UAT).')
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--approved-tree-sha256', required=True)
    parser.add_argument('--approved-original-onnx-sha256', required=True)
    parser.add_argument('--input-stage', required=True)
    parser.add_argument('--into', required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        result = run(args.source_dir, args.approved_tree_sha256,
                     args.approved_original_onnx_sha256,
                     args.input_stage, args.into)
    except (SelectedDetOracleRefused, raw.CaptureRefused, OSError, ValueError,
            TypeError, KeyError, ImportError,
            source_freeze.SourceFreezeError) as exc:
        print(json.dumps({'status': 'REFUSED',
                          'acceptance_status': 'NOT_CHECKED',
                          'native_pascal': 'NOT_RUN',
                          'human_ground_truth': 'NOT_ATTESTED',
                          'ready_issued': False, 'reason': str(exc)},
                         ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))

    return 2


if __name__ == '__main__':
    raise SystemExit(main())
