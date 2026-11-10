# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from . import ocr_det_pairwise as legacy_pair
from . import ocr_native_det_capture_bridge as native
from . import ocr_ort_raw_capture as raw
from . import ocr_capture_bridge as ort_bridge
from . import ocr_composite_oracle as oracle
from . import paddle_selected_det_original_ort as selected_ort
from . import paddle_selected_det_input as det


class SelectedDetPairwiseRefused(ValueError):
    pass


_EXPECTED = dict(native._EXPECTED_STATUS,
                 schema='crosspascal.paddleocr.native-selected-det-raw-capture',
                 status='NATIVE_SELECTED_DET_RAW_CAPTURE_ONLY_NOT_ORIGINAL_ORACLE')
_PREPROCESSING = {
    'source_color': 'BGR',
    'resize': 'INTER_LINEAR_square_960_only_vendor_resize_long_960',
    'channels': 'BGR',
    'layout': 'NCHW',
    'formula': '(float32(raw_BGR)/255-mean_BGR)/std_BGR',
    'mean_BGR': [0.485, 0.456, 0.406],
    'std_BGR': [0.229, 0.224, 0.225],
    'selected_vendor_inference_yml_sha256': det._SOURCE_SHAS['inference.yml'],
    'square_source_only': True,
    'original_vendor_execution': False,
}
_EXPECTED_RUNTIME = legacy_pair._RUNTIME_NAMES | {'crosscv.dll'}


def _require(condition, message):
    if not condition:
        raise SelectedDetPairwiseRefused(
            'selected original/Pascal Det pair refused: ' + message)


def _selected_native_receipt(root):
    raw._check_private_root(root)
    names = []
    with os.scandir(root) as iterator:
        for entry in iterator:
            names.append(entry.name)
            _require(len(names) <= 3, 'selected native stage has a foreign member')
    _require(len(names) == 3
             and set(names) == native._CAPTURE_FILES
             and len({name.casefold() for name in names}) == 3,
             'selected native stage must contain exactly three capture files')
    receipt_path = root / 'det-capture.json'
    sha, size = raw._sha_file(
        receipt_path, 'selected native capture receipt', limit=native._JSON_LIMIT)
    with receipt_path.open('rb') as stream:
        body = stream.read(native._JSON_LIMIT + 1)
    _require(len(body) == size and hashlib.sha256(body).hexdigest() == sha,
             'selected native capture receipt changed during second read')
    row = raw._strict_json(body)
    _require(type(row) is dict and set(row) == native._CAPTURE_KEYS,
             'selected native receipt has missing/foreign fields')
    for key, value in _EXPECTED.items():
        _require(row.get(key) == value and type(row.get(key)) is type(value),
                 'selected native status/provenance field differs: ' + key)
    profile = row.get('preprocessing')
    _require(type(profile) is dict and profile == _PREPROCESSING
             and set(profile) == set(_PREPROCESSING),
             'native capture was not the exact selected BGR/HWC vendor profile')
    expected = row.get('runtime_selected_files_observed_sha256')
    _require(type(expected) is dict and set(expected) == _EXPECTED_RUNTIME,
             'current selected native producer must independently report 4 held CPU DLLs')
    _require(type(row.get('evidence_limit')) is str
             and 0 < len(row['evidence_limit']) <= 4096,
             'selected receipt omitted its admission limitations')
    return row, sha, size


def _selected_recipe(model, input_name):
    path = model.parent / 'recipe.json'
    receipt, sha, size = legacy_pair._read_selected_json(
        path, 'native selected compiler recipe', legacy_pair._RECIPE_MAX_BYTES)
    selected_shape = input_name + ':1,3,960,960'
    source_hash = receipt.get('source_hash')
    _require(receipt.get('target') == 'cpu'
             and receipt.get('shape') == selected_shape
             and type(source_hash) is str
             and legacy_pair.oracle._sha(source_hash, 'compiled source hash') == source_hash,
             'selected native compiler recipe lacks actual CPU/source SHA/input contract')


    return path, sha, size, source_hash


def _inspect_selected_det_pairwise(
        native_capture, ort_capture, selected_input_stage, *,
        source_dir, approved_tree_sha256, native_model, original_onnx,
        expected_native_sha256, expected_original_sha256, output_name):
    _require(type(approved_tree_sha256) is str
             and approved_tree_sha256 == selected_ort._TREE_SHA
             and type(expected_original_sha256) is str
             and expected_original_sha256 == selected_ort._EXTRA_ONNX_SHA,
             'selected PIR tree and extra original ONNX SHA need explicit exact approvals')
    _require(type(output_name) is str and 0 < len(output_name) <= 256
             and not any(ord(char) < 32 for char in output_name),
             'exact bounded original ORT probability output name is mandatory')
    source = det._model(source_dir, approved_tree_sha256)
    source_root = Path(source['source_root'])
    pinned_original = source_root / 'model.onnx'
    original_path = raw._absolute(original_onnx, 'original selected Det ONNX')
    _require(original_path == pinned_original,
             'original ORT path is not separately pinned original model-zoo Det export')
    original_sha, original_size = raw._sha_file(
        original_path, 'separate extra Det ONNX original',
        limit=raw.MAX_MODEL_BYTES,
        expected_bytes=selected_ort._EXTRA_ONNX_BYTES)
    _require(original_sha == selected_ort._EXTRA_ONNX_SHA,
             'extra selected original ONNX was replaced')

    stage = raw._absolute(selected_input_stage, 'held selected Det PNG input stage')
    expected_input = selected_ort._stage_snapshot(stage)
    native_root = raw._absolute(native_capture, 'selected Pascal Det capture')
    native_row, native_sha, native_size = _selected_native_receipt(native_root)
    dll, dll_sha = legacy_pair._selected_native_model(
        native_model, native_row, expected_native_sha256)
    name, shape, sidecar, sidecar_sha, sidecar_size = (
        legacy_pair._selected_sidecar(dll, native_row))
    _require(name == 'x' and shape == list(det.SHAPE),
             'selected native x input differs from [1,3,960,960]')
    recipe, recipe_sha, recipe_bytes, compiled_onnx_source_sha = (
        _selected_recipe(dll, name))
    runtime = legacy_pair._runtime_kit(native_row)
    _require(len(runtime) == 4,
             'selected native CrossCV runtime did not bind four actual files')
    png_sha = native._source_image(native_row)
    _require(png_sha == expected_input['image_sha256'],
             'native Pascal and original source-profile PNG bytes differ')
    native_input = native._tensor(native_root, native_row['input'],
                                  role='input', input_shape=shape)
    _require(native_input['sha256'] == expected_input['tensor_sha256'],
             'native Pascal used different NCHW float32 bytes than original input producer')
    native_output = native._tensor(native_root, native_row['raw_output'],
                                   role='output')

    ort_root = raw._absolute(ort_capture, 'original Det ORT raw capture')
    ort_receipt, ort_sha = raw._verify_capture(
        ort_root, original_sha, expected_input['tensor_sha256'], list(det.SHAPE))
    ort_bridge._receipt_grammar(ort_receipt)
    ort_input = ort_receipt.get('input')
    _require(type(ort_input) is dict
             and ort_input.get('name') == 'x'
             and ort_input.get('shape') == list(det.SHAPE)
             and ort_input.get('sha256') == native_input['sha256'],
             'original ORT and Pascal inputs have unequal names/shapes/SHA')
    rows = [row for row in ort_receipt['outputs']
            if row['name'] == output_name]
    _require(len(rows) == 1 and rows[0]['shape'] == native_output['shape']
             and rows[0]['shape'][:2] == [1, 1],
             'ORIGINAL Det probability output name/shape disagrees with Pascal')
    output_match = oracle._compare_tensors(
        legacy_pair._tensor_for_oracle(rows[0], ort_root / rows[0]['file']),
        legacy_pair._tensor_for_oracle(native_output,
                                       native_root / 'det-output.f32le'),
        'selected original ORT vs actual Pascal Det map',
        atol=oracle.OUTPUT_ATOL, rtol=oracle.OUTPUT_RTOL,
        nonvacuous=True)


    for label, path, sha, count, cap in (
        [('selected native DLL', dll, dll_sha,
          native_row['model']['bytes'], native._MODEL_LIMIT),
         ('selected native .inputs', sidecar, sidecar_sha,
          sidecar_size, native._SIDECAR_LIMIT),
         ('selected native recipe', recipe, recipe_sha,
          recipe_bytes, legacy_pair._RECIPE_MAX_BYTES),
         ('original extra ONNX', original_path, original_sha,
          original_size, raw.MAX_MODEL_BYTES)]
        + [('native CPU runtime', path, digest, size, native._MODEL_LIMIT)
           for path, digest, size in runtime]
    ):
        _require(raw._sha_file(path, label + ' final SHA',
                               limit=cap, expected_bytes=count) == (sha, count),
                 label + ' changed during selected pairwise')
    _require(det._model(source_dir, approved_tree_sha256) == source,
             'original Paddle PIR/YAML source revision changed during pairwise')
    _require(selected_ort._stage_snapshot(stage) == expected_input,
             'original PNG/input source-profile stage changed during pairwise')
    _require(native._source_image(native_row) == png_sha
             and native._tensor(native_root, native_row['input'], role='input',
                                input_shape=shape)['sha256']
             == native_input['sha256']
             and native._tensor(native_root, native_row['raw_output'],
                                role='output')['sha256'] == native_output['sha256'],
             'native original image/input/output changed after scalar comparison')
    _require(raw._verify_capture(ort_root, original_sha,
                                 expected_input['tensor_sha256'],
                                 list(det.SHAPE))[1] == ort_sha,
             'original ORT captured output files changed after comparison')
    _require(raw._sha_file(native_root / 'det-capture.json',
                           'native selected final receipt',
                           limit=native._JSON_LIMIT,
                           expected_bytes=native_size)[0] == native_sha,
             'selected native receipt changed after scalar comparison')
    return {
        'status': 'NOT_CHECKED',
        'numeric_status': 'NUMERIC_MATCH_SUPPLIED_SELECTED_DET_ONLY',
        'producer_authenticity': 'NOT_ATTESTED',
        'compiled_onnx_source_relation_to_extra_original':
        'NOT_INDEPENDENTLY_ATTESTED',
        'native_pascal_process_authenticity': 'NOT_ATTESTED',
        'original_onnx_process_authenticity': 'NOT_ATTESTED',
        'native_runtime_provenance': 'NOT_ATTESTED',
        'source_profile_preprocessing': 'SOURCE_RECONSTRUCTION_NOT_VENDOR_EXECUTION',
        'vendor_db_postprocessing': 'NOT_RUN',
        'native_rec_ctc': 'NOT_RUN',
        'human_ground_truth': 'NOT_ATTESTED',
        'facade_manifest_issued': False, 'ready_issued': False,
        'selected_model_tree_sha256': approved_tree_sha256,
        'original_extra_onnx_sha256': original_sha,
        'compiled_source_onnx_sha256': compiled_onnx_source_sha,
        'selected_native_model_sha256': dll_sha,
        'native_capture_sha256': native_sha,
        'original_ort_capture_sha256': ort_sha,
        'selected_png_sha256': png_sha,
        'selected_input_sha256': native_input['sha256'],
        'native_and_ort_output_name': output_name,
        'det_raw_output_metrics': output_match,
        'provenance_limit': (
            'Read-only nonvacuous supplied raw pair only. No independent '
            'original Paddle execution, compiler-trained-weights origin '
            'attestation, native/ORT process provenance, Det DB boxes, '
            'ROI/Rec/CTC, human ground truth or a READY bundle.'),
    }


def inspect_selected_det_pairwise(
        native_capture, ort_capture, selected_input_stage, *,
        source_dir, approved_tree_sha256, native_model, original_onnx,
        expected_native_sha256, expected_original_sha256, output_name):
    try:
        return _inspect_selected_det_pairwise(
            native_capture, ort_capture, selected_input_stage,
            source_dir=source_dir,
            approved_tree_sha256=approved_tree_sha256,
            native_model=native_model, original_onnx=original_onnx,
            expected_native_sha256=expected_native_sha256,
            expected_original_sha256=expected_original_sha256,
            output_name=output_name)
    except SelectedDetPairwiseRefused:
        raise
    except (ValueError, OSError, TypeError, KeyError, IndexError,
            OverflowError, RecursionError, UnicodeError,
            raw.CaptureRefused, native.NativeDetCaptureRefused,
            oracle.InvalidEvidence, oracle.MissingEvidence,
            selected_ort.SelectedDetOracleRefused,
            legacy_pair.DetPairwiseRefused, ort_bridge.CaptureBridgeRefused,
            det.SelectedDetInputRefused) as exc:
        raise SelectedDetPairwiseRefused(
            'selected original-ORT/native-Pascal Det evidence refused: %s' % exc
        ) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-det-pairwise',
        description='Read-only selected Server Det original-ORT versus actual '
                    'native-Pascal raw comparison. Advisory result ONLY.')
    for option in (
        '--native-capture', '--ort-capture', '--selected-input-stage',
        '--source-dir', '--approved-tree-sha256', '--native-model',
        '--original-onnx', '--native-sha256', '--original-sha256',
        '--output-name',
    ):
        parser.add_argument(option, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        result = inspect_selected_det_pairwise(
            args.native_capture, args.ort_capture, args.selected_input_stage,
            source_dir=args.source_dir,
            approved_tree_sha256=args.approved_tree_sha256,
            native_model=args.native_model, original_onnx=args.original_onnx,
            expected_native_sha256=args.native_sha256,
            expected_original_sha256=args.original_sha256,
            output_name=args.output_name)
    except (SelectedDetPairwiseRefused, ValueError, OSError, TypeError,
            KeyError, IndexError, UnicodeError,
            raw.CaptureRefused, native.NativeDetCaptureRefused,
            oracle.InvalidEvidence, oracle.MissingEvidence) as exc:
        print(json.dumps({
            'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
            'native_process_authenticity': 'NOT_ATTESTED',
            'human_ground_truth': 'NOT_ATTESTED', 'ready_issued': False,
            'reason': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
