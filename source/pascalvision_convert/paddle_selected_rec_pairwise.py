# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from . import ocr_capture_bridge as ort_bridge
from . import ocr_composite_oracle as oracle
from . import ocr_det_pairwise as common
from . import ocr_native_det_capture_bridge as native_helper
from . import ocr_ort_raw_capture as raw
from . import paddle_selected_rec_input as rec
from . import paddle_selected_native as selected_native
from . import paddle_selected_rec_original_ort as source
from . import source_lock


class SelectedRecPairwiseRefused(ValueError):
    pass


_MODEL_MAX = 512 * 1024 * 1024
_RECEIPT_MAX = 64 * 1024
_SCORE_MAX = 2_000_000
_EXPECTED_COMPILED_SHA = 'c2c5c9ee48891e24e9a0ba2df593de1bccfd91c3aa0235838f669b314b6b0c03'
_EXPECTED_SELECTED_RECIPE2_SHA = 'ad55badf83416ec6bf5cab054499eaef6068d5ec1d24155d63723fed5a932aa8'
_EXPECTED_COMPILED_SHAS = frozenset((
    _EXPECTED_COMPILED_SHA, _EXPECTED_SELECTED_RECIPE2_SHA,
))
_EXPECTED_KEYS = {
    'schema', 'schema_version', 'status', 'acceptance_status',
    'verification_status', 'composite_verification',
    'native_composite_oracle', 'native_ctc', 'human_ground_truth',
    'crop_document_roi', 'independent_onnx_preprocessing',
    'png_crc_or_pixel_equivalence', 'process_provenance',
    'native_runtime_provenance', 'natural_process_exit_attestation',
    'paddleocr_rec_model_identity', 'facade_manifest_issued',
    'ready_issued', 'actual_pascal_rec_inference', 'device',
    'model', 'source_crop', 'dictionary', 'input_name', 'input',
    'raw_scores', 'raw_output_semantics', 'runtime_observed_path',
    'runtime_selected_files_observed_sha256',
    'runtime_sha_independent_expected_pin', 'preprocessing',
    'evidence_limit',
}
_STATIC = {
    'schema': 'crosspascal.paddleocr.native-selected-rec-raw-capture',
    'schema_version': 1,
    'status': 'NATIVE_SELECTED_REC_RAW_CAPTURE_ONLY_NOT_ORIGINAL_ORACLE',
    'acceptance_status': 'NOT_CHECKED',
    'verification_status': 'NOT_CHECKED',
    'composite_verification': 'NOT_CHECKED',
    'native_composite_oracle': 'NOT_RUN',
    'native_ctc': 'NOT_RUN',
    'human_ground_truth': 'NOT_ATTESTED',
    'crop_document_roi': 'NOT_ATTESTED',
    'independent_onnx_preprocessing': 'NOT_CHECKED',
    'png_crc_or_pixel_equivalence': 'NOT_CHECKED',
    'process_provenance': 'NOT_ATTESTED',
    'native_runtime_provenance': 'NOT_ATTESTED',
    'natural_process_exit_attestation': 'NOT_CHECKED',
    'paddleocr_rec_model_identity': 'CALLER_SELECTED_FIXED_HASH_NOT_VENDOR_ATTESTED',
    'facade_manifest_issued': False,
    'ready_issued': False,
    'actual_pascal_rec_inference': True,
    'device': 'cpu',
    'runtime_sha_independent_expected_pin': False,
    'raw_output_semantics': 'post-softmax CTC probabilities (not logits)',
}
_STATIC_CTC = dict(
    _STATIC,
    schema='crosspascal.paddleocr.native-selected-rec-ctc-capture',
    schema_version=2,
    status='NATIVE_SELECTED_REC_CTC_CAPTURE_NOT_ORIGINAL_ORACLE',
    native_ctc='SAME_RUN_DECODED_NOT_ORACLE',
)
_CTC_RECEIPT_KEYS = _EXPECTED_KEYS | {
    'native_ctc_text', 'native_ctc_score_micros', 'native_ctc_blank_index',
}
_PREPROCESSING = {
    'actual_production_function': 'TVM.PaddleOcr.PpocrPreprocess',
    'source_color': 'OpenCV BGR decoded from caller RGB8 PNG',
    'resize': 'INTER_LINEAR fixed-height ceil aspect ratio NO stride8',
    'right_padding': 'normalized BGR float32 zero-filled AFTER /255*2-1',
    'channels': 'BGR', 'layout': 'NCHW',
    'formula': 'float32(BGR)/255*2-1',
    'selected_vendor_inference_yml_sha256':
        rec._PINNED_SOURCE['inference.yml'],
    'selected_ctc_classes': 18385,
    'original_vendor_execution': False,
}


def _require(condition, text):
    if not condition:
        raise SelectedRecPairwiseRefused('selected Rec ORT/Pascal refused: ' + text)


def _capture(root):
    try:
        raw._check_private_root(root)
    except raw.CaptureRefused as exc:
        raise SelectedRecPairwiseRefused(
            'selected Rec ORT/Pascal refused: invalid private capture root: %s' % exc
        ) from exc
    filenames = []
    with os.scandir(root) as files:
        for file in files:
            filenames.append(file.name)
            _require(len(filenames) <= 3, 'native Rec stage includes foreign bytes')
    _require(len(filenames) == 3
             and set(filenames) == {'rec-input.f32le',
                                    'rec-scores.f32le', 'rec-capture.json'}
             and len({name.casefold() for name in filenames}) == 3,
             'native selected Rec stage has extra/missing members')
    receipt_file = root / 'rec-capture.json'
    sha, size = raw._sha_file(
        receipt_file, 'selected Pascal Rec receipt', limit=_RECEIPT_MAX)
    with receipt_file.open('rb') as data:
        payload = data.read(_RECEIPT_MAX + 1)
    _require(len(payload) == size and hashlib.sha256(payload).hexdigest() == sha,
             'native selected Rec receipt changed while reading JSON')
    row = raw._strict_json(payload)
    selected_ctc = (type(row) is dict and
                    row.get('schema') == _STATIC_CTC['schema'])
    _require(type(row) is dict and
             set(row) == (_CTC_RECEIPT_KEYS if selected_ctc else _EXPECTED_KEYS),
             'native Rec receipt contains missing/unapproved keys')
    for key, expected in (_STATIC_CTC if selected_ctc else _STATIC).items():
        _require(row.get(key) == expected and type(row.get(key)) is type(expected),
                 'native selected Rec receipt status/contract forged: ' + key)
    if selected_ctc:
        _require(type(row['native_ctc_text']) is str and
                 len(row['native_ctc_text']) <= 108 * 16 and
                 type(row['native_ctc_score_micros']) is int and
                 0 <= row['native_ctc_score_micros'] <= 1000000 and
                 type(row['native_ctc_blank_index']) is int and
                 row['native_ctc_blank_index'] == 0,
                 'same-run Pascal selected CTC text/confidence evidence malformed')
    _require(type(row.get('preprocessing')) is dict
             and row['preprocessing'] == _PREPROCESSING
             and set(row['preprocessing']) == set(_PREPROCESSING),
             'native Rec used a legacy RGB/stride8/-1 padding profile')
    _require(type(row.get('evidence_limit')) is str
             and 0 < len(row['evidence_limit']) <= 4096,
             'native Rec omitted its evidence limitations')
    return row, sha, size


def _score_file(root, row, *, native: bool, input_sha=None):
    _require(type(row) is dict and set(row) ==
             {'file', 'dtype', 'layout', 'shape', 'bytes', 'sha256'},
             'Rec float32 score descriptor fields differ')
    shape = row['shape']
    if native:
        expected_name, expected_layout = 'rec-scores.f32le', 'NTC'
    else:
        expected_name, expected_layout = 'rec-input.f32le', 'NCHW'
    _require(row['file'] == expected_name and row['dtype'] == 'float32le'
             and row['layout'] == expected_layout
             and type(shape) is list
             and (len(shape) == 3 if native else len(shape) == 4),
             'native Rec raw tensor file/layout/shape is not selected profile')
    if native:
        _require(shape[0] == 1 and type(shape[1]) is int
                 and 2 <= shape[1] <= 108
                 and type(shape[2]) is int and shape[2] == 18385,
                 'Rec output must be compact float32 [1,T,18385], T=2..108')
        elements = shape[1] * 18385
        _require(elements <= _SCORE_MAX,
                 'post-softmax Rec class array exceeds native 2M quota')
    else:
        _require(shape == list(rec.SHAPE),
                 'native Rec input must be x:[1,3,48,320]')
        elements = rec.INPUT_BYTES // 4
    count = elements * 4
    _require(type(row['bytes']) is int and row['bytes'] == count,
             'selected Rec tensor byte count differs from model shape')
    digest = oracle._sha(row['sha256'], 'native Rec raw tensor SHA')
    path = root / expected_name
    actual_sha, size = raw._sha_file(
        path, 'native Rec raw tensor', limit=max(rec.INPUT_BYTES, _SCORE_MAX * 4),
        expected_bytes=count)
    _require((actual_sha, size) == (digest, count),
             'native selected Rec bytes were substituted')
    if input_sha is not None:
        _require(actual_sha == input_sha,
                 'actual Pascal Rec input differs from pinned original Rec input')
    return {'path': path, 'shape': list(shape), 'count': elements,
            'bytes': count, 'sha256': digest}


def _selected_ctc_alphabet(dictionary_path, dictionary_sha, dictionary_bytes):
    _require(dictionary_sha == rec._PINNED_DICT_SHA
             and type(dictionary_bytes) is int
             and 0 < dictionary_bytes <= 1024 * 1024,
             'selected CTC alphabet requires exact SHA and bounded dictionary')
    with dictionary_path.open('rb') as stream:
        _require(os.fstat(stream.fileno()).st_size == dictionary_bytes,
                 'selected CTC dictionary changed before full read')
        content = stream.read(dictionary_bytes + 1)
    _require(len(content) == dictionary_bytes
             and hashlib.sha256(content).hexdigest() == dictionary_sha,
             'selected CTC dictionary bytes changed during alphabet read')
    try:
        characters = rec.paddle_closure._dictionary_text(content)
    except rec.paddle_closure.PaddleClosureError as exc:
        raise SelectedRecPairwiseRefused(
            'selected CTC dictionary tokens are invalid: %s' % exc) from exc
    _require(len(characters) == 18383,
             'selected CTC original alphabet must contain 18383 tokens')
    return tuple(characters) + (' ',)


def _probability_contract(tensor, *, ctc_alphabet=None):
    _require(tensor['shape'][0] == 1
             and tensor['shape'][2] == 18385
             and 2 <= tensor['shape'][1] <= 108,
             'not a selected Server Rec [1,T,18385] tensor')
    if ctc_alphabet is not None:
        _require(type(ctc_alphabet) in (tuple, list)
                 and len(ctc_alphabet) == 18384
                 and ctc_alphabet[-1] == ' '
                 and all(type(token) is str and len(token) > 0
                         for token in ctc_alphabet),
                 'selected CTC alphabet must have 18384 tokens including appended ASCII space')
    path = tensor['path']
    with path.open('rb') as stream:
        _require(os.fstat(stream.fileno()).st_size == tensor['bytes'],
                 'Rec score bytes changed before probability validation')
        digest = hashlib.sha256()
        winners = []
        winner_probs = []
        for frame in range(tensor['shape'][1]):
            data = stream.read(18385 * 4)
            _require(len(data) == 18385 * 4,
                     'Rec probability row is incomplete')
            digest.update(data)
            total = 0.0
            top_class, top_prob = -1, -1.0
            for col, (value,) in enumerate(struct.iter_unpack('<f', data)):
                _require(math.isfinite(value) and
                         -0.000001 <= value <= 1.00001,
                         'Rec scores are not bounded post-softmax probabilities')
                total += value
                if value > top_prob:
                    top_prob, top_class = value, col
            _require(abs(total - 1.0) <= 0.02,
                     'Rec post-softmax probability row does not sum to one')
            winners.append(top_class)
            winner_probs.append(top_prob)
        _require(stream.read(1) == b'',
                 'Rec scores gained unexpected trailing scalar bytes')
    _require(digest.hexdigest() == tensor['sha256'],
             'Rec score bytes changed during full probability scan')
    result = {
        'frames': len(winners),
        'top_class_sample_first': winners[0],
        'nonblank_argmax_frames': sum(index != 0 for index in winners),
        'blank_id': 0,
        'actual_ctc_decode_executed': False,
    }
    if ctc_alphabet is not None:
        previous, emitted, emitted_scores = -1, [], []
        for index, probability in zip(winners, winner_probs):
            if index != 0 and index != previous:
                emitted.append(ctc_alphabet[index - 1])


                emitted_scores.append(max(0.0, min(1.0, probability)))
            previous = index
        result['source_reference_ctc'] = {
            'status': 'PYTHON_CTC_SOURCE_REFERENCE_ONLY_NOT_NATIVE_OR_VENDOR',
            'text': ''.join(emitted),
            'emitted_tokens': len(emitted),
            'mean_confidence': (sum(emitted_scores) / len(emitted_scores)
                                if emitted_scores else 0.0),
            'blank_index': 0,
            'input_scores_sha256': tensor['sha256'],
        }
    return result


def _native_compiler_recipe(model, input_name, *, expected_native_sha256,
                            chosen):
    receipt, recipe_sha, recipe_bytes = common._read_selected_json(
        model.parent / 'recipe.json', 'selected Rec TVM compiler recipe',
        common._RECIPE_MAX_BYTES)
    source_sha = receipt.get('source_hash')
    _require(receipt.get('target') == 'cpu'
             and receipt.get('shape') == input_name + ':1,3,48,320'
             and type(source_sha) is str
             and oracle._sha(source_sha, 'selected Rec compiler source') == source_sha,
             'native Rec recipe not bound to exact compiler ONNX x:1,3,48,320')
    recipe2_audit = []
    recipe2_runtime = {}
    if expected_native_sha256 == _EXPECTED_SELECTED_RECIPE2_SHA:
        approved, validated, members, compiled_runtime = selected_native._compiled_bundle(
            str(model.parent), 'rec', source_lock._Kernel32())
        _require(validated == receipt and approved == chosen,
                 'new Rec DLL requires the exact original selected recipe2 source tree')
        model_rows = [row for row in members if row['role'] == 'rec_dll']
        _require(len(model_rows) == 1
                 and model_rows[0]['source'] == str(model)
                 and model_rows[0]['sha256'] == expected_native_sha256,
                 'new Rec DLL differs from measured recipe2 compiler output')
        provenance, _, _ = common._read_selected_json(
            model.parent / 'import-provenance.json',
            'selected Rec held compiler provenance', common._RECIPE_MAX_BYTES)
        _require(provenance.get('onnx_sha256') == source_sha,
                 'new Rec recipe source SHA differs from compiler-held ONNX')
        recipe2_audit = [
            ('selected Rec recipe2 ' + row['role'], Path(row['source']),
             row['sha256'], row['bytes'], _MODEL_MAX)
            for row in members if row['role'] not in
            ('rec_dll', 'rec_inputs', 'rec_recipe')
        ]
        recipe2_runtime = {
            name: (row['sha256'], row['bytes'])
            for name, row in compiled_runtime.items()
        }
    return (model.parent / 'recipe.json', recipe_sha, recipe_bytes,
            source_sha, recipe2_audit, recipe2_runtime)


def _inspect(native_capture, ort_capture, selected_input_stage, *,
             source_dir, dictionary, approved_tree_sha256,
             native_model, original_onnx, expected_native_sha256,
             expected_original_sha256, output_name):
    _require(type(approved_tree_sha256) is str
             and approved_tree_sha256 == source._PINNED_TREE_SHA
             and type(expected_original_sha256) is str
             and expected_original_sha256 == source._EXTRA_ONNX_SHA
             and type(expected_native_sha256) is str
             and expected_native_sha256 in _EXPECTED_COMPILED_SHAS,
             'original Rec PIR/extra ONNX and measured native model need exact approved SHAs')
    _require(type(output_name) is str and 0 < len(output_name) <= 256
             and not any(ord(ch) < 32 for ch in output_name),
             'original ORT semantic Rec score tensor name is mandatory')
    chosen = rec._model(source_dir, dictionary, approved_tree_sha256)
    owner = Path(chosen['source_root'])
    _require(chosen['dictionary']['sha256'] == rec._PINNED_DICT_SHA,
             'selected real Rec dictionary changed before comparison')
    dict_path = raw._absolute(dictionary, 'selected Rec full UTF-8 dictionary')
    _require(str(dict_path) == str(chosen['dictionary']['reference']),
             'caller-selected dictionary is not the approved Rec alphabet')
    dict_sha, dict_count = raw._sha_file(
        dict_path, 'selected original Rec Unicode dictionary',
        limit=1024 * 1024)
    _require(dict_sha == rec._PINNED_DICT_SHA,
             'original selected 18383-token dictionary SHA mismatch')
    ctc_alphabet = _selected_ctc_alphabet(dict_path, dict_sha, dict_count)
    model_original = raw._absolute(original_onnx, 'extra original Rec model.onnx')
    _require(model_original == owner / 'model.onnx',
             'original Rec ONNX must be model.onnx in the selected source directory')
    original_sha, original_bytes = raw._sha_file(
        model_original, 'original Rec ONNX export', limit=raw.MAX_MODEL_BYTES,
        expected_bytes=source._EXTRA_ONNX_BYTES)
    _require(original_sha == source._EXTRA_ONNX_SHA,
             'independently pinned extra original Rec ONNX changed')

    stage = raw._absolute(selected_input_stage, 'selected Rec BGR input source-profile')
    snapshot = source._snapshot(stage)
    native_root = raw._absolute(native_capture, 'selected actual Pascal Rec raw capture')
    row, native_receipt_sha, native_receipt_bytes = _capture(native_root)
    native_model_path, native_model_sha = common._selected_native_model(
        native_model, row, expected_native_sha256)
    name, shape, sidecar, sidecar_sha, sidecar_bytes = common._selected_sidecar(
        native_model_path, row)
    _require(name == 'x' and shape == list(rec.SHAPE),
             'selected Rec DLL input contract must be x:[1,3,48,320]')
    recipe_path, recipe_sha, recipe_bytes, compiler_source_sha, recipe2_audit, recipe2_runtime = (
        _native_compiler_recipe(native_model_path, name,
                                expected_native_sha256=expected_native_sha256,
                                chosen=chosen))
    runtime = common._runtime_kit(row)
    _require(len(runtime) == 4, 'selected Rec actual CrossCV must be held')
    if recipe2_runtime:
        observed_runtime = {
            path.name: (digest, count) for path, digest, count in runtime
            if path.name in recipe2_runtime
        }
        _require(observed_runtime == recipe2_runtime,
                 'new selected Rec runtime bytes differ from its compiler BuildPlan kit')

    crop = row.get('source_crop')
    _require(type(crop) is dict
             and set(crop) == {'path', 'sha256', 'bytes', 'width', 'height'},
             'native Rec crop receipt must have exact pinned PNG metadata')

    crop_sha = native_helper._source_image({'image': crop})
    _require(crop_sha == snapshot['png_sha256'],
             'original Rec source-profile and native actual crop PNG bytes differ')
    ndict = row.get('dictionary')
    _require(type(ndict) is dict
             and set(ndict) == {'path', 'sha256', 'bytes',
                                'tokens_including_appended_space',
                                'ctc_blank_index_expected',
                                'ctc_class_count_expected'}
             and ndict['path'] == str(dict_path)
             and ndict['sha256'] == dict_sha
             and type(ndict['bytes']) is int and ndict['bytes'] == dict_count
             and type(ndict['tokens_including_appended_space']) is int
             and ndict['tokens_including_appended_space'] == 18384
             and type(ndict['ctc_blank_index_expected']) is int
             and ndict['ctc_blank_index_expected'] == 0
             and type(ndict['ctc_class_count_expected']) is int
             and ndict['ctc_class_count_expected'] == 18385,
             'native raw Rec dictionary/CTC blank metadata differs from original')
    _require(row.get('input_name') == name,
             'native model and receipt Rec input name differ')
    model_receipt = row['model']
    _require(type(model_receipt) is dict
             and set(model_receipt) ==
             {'path', 'sha256', 'bytes', 'inputs_sha256', 'inputs_bytes',
              'original_source_onnx_independently_attested'}
             and model_receipt['original_source_onnx_independently_attested'] is False,
             'native compiled Rec model receipt falsely asserts original weight origin')

    native_input = _score_file(native_root, row['input'], native=False,
                                input_sha=snapshot['input_sha256'])
    native_scores = _score_file(native_root, row['raw_scores'], native=True)
    native_probabilities = _probability_contract(
        native_scores, ctc_alphabet=ctc_alphabet)
    ort_root = raw._absolute(ort_capture, 'actual original Rec ONNX Runtime capture')
    original_receipt, ort_receipt_sha = raw._verify_capture(
        ort_root, original_sha, snapshot['input_sha256'], list(rec.SHAPE))
    ort_bridge._receipt_grammar(original_receipt)
    _require(original_receipt['input']['name'] == name
             and original_receipt['input']['shape'] == list(rec.SHAPE)
             and original_receipt['input']['sha256'] == native_input['sha256'],
             'original Rec ORT input differs from actual Pascal BGR input')
    matches = [entry for entry in original_receipt['outputs']
               if entry['name'] == output_name]
    _require(len(matches) == 1
             and matches[0]['shape'] == native_scores['shape'],
             'exact original ORT Rec scores name/NTC shape disagree with native')
    ort_score_desc = common._tensor_for_oracle(
        matches[0], ort_root / matches[0]['file'])
    ort_probabilities = _probability_contract(
        ort_score_desc, ctc_alphabet=ctc_alphabet)
    parity = oracle._compare_tensors(
        ort_score_desc, native_scores,
        'selected original Rec post-softmax ORT versus actual Pascal scores',
        atol=oracle.OUTPUT_ATOL, rtol=oracle.OUTPUT_RTOL, nonvacuous=True)
    _require(native_probabilities['source_reference_ctc']['text'] ==
             ort_probabilities['source_reference_ctc']['text'],
             'selected original/native scores decode to different Python CTC text')
    if row['schema'] == _STATIC_CTC['schema']:
        ctc = native_probabilities['source_reference_ctc']
        _require(row['native_ctc_text'] == ctc['text'],
                 'same-run native Pascal CTC differs from independent Python decode')
        _require(abs(row['native_ctc_score_micros'] / 1000000.0 -
                     ctc['mean_confidence']) <= 0.000002,
                 'same-run native Pascal CTC confidence differs from source reference')

    for title, path, sha, count, limit in [
        ('compiled selected Rec model', native_model_path, native_model_sha,
         model_receipt['bytes'], _MODEL_MAX),
        ('Rec .inputs', sidecar, sidecar_sha, sidecar_bytes, 4096),
        ('Rec native recipe', recipe_path, recipe_sha,
         recipe_bytes, common._RECIPE_MAX_BYTES),
        ('original extra Rec ONNX', model_original,
         original_sha, original_bytes, raw.MAX_MODEL_BYTES),
        ('original dictionary', dict_path, dict_sha,
         dict_count, 1024 * 1024),
    ] + [('native runtime', p, digest, count, _MODEL_MAX)
         for p, digest, count in runtime] + recipe2_audit:
        _require(raw._sha_file(path, title + ' final SHA',
                               limit=limit, expected_bytes=count) == (sha, count),
                 title + ' was changed during selected Rec pairwise')
    _require(rec._model(source_dir, dictionary, approved_tree_sha256) == chosen,
             'selected Rec Paddle weights/params/YAML tree changed in comparison')
    _require(source._snapshot(stage) == snapshot,
             'selected Rec original PNG/input stage changed during comparison')
    _require(native_helper._source_image({'image': crop}) == crop_sha
             and _score_file(native_root, row['input'], native=False,
                             input_sha=snapshot['input_sha256'])['sha256']
             == native_input['sha256']
             and _score_file(native_root, row['raw_scores'], native=True)['sha256']
             == native_scores['sha256'],
             'native Rec source crop/input/scores changed after comparison')
    _require(raw._verify_capture(
        ort_root, original_sha, snapshot['input_sha256'], list(rec.SHAPE))[1]
        == ort_receipt_sha,
        'selected Rec original ORT raw capture changed after comparison')
    _require(raw._sha_file(
        native_root / 'rec-capture.json', 'native final Rec receipt',
        limit=_RECEIPT_MAX,
        expected_bytes=native_receipt_bytes)[0] == native_receipt_sha,
        'selected Rec native receipt changed after comparison')
    return {
        'status': 'NOT_CHECKED',
        'numeric_status': 'NUMERIC_MATCH_SUPPLIED_SELECTED_REC_ONLY',
        'input_shape': list(rec.SHAPE),
        'scores_shape': native_scores['shape'],
        'native_probability_contract': native_probabilities,
        'original_ort_probability_contract': ort_probabilities,
        'python_source_reference_ctc_text_agrees': True,
        'postsoftmax_raw_numeric_metrics': parity,
        'source_tree_sha256': approved_tree_sha256,
        'dictionary_sha256': dict_sha,
        'original_extra_onnx_sha256': original_sha,
        'compiled_native_model_sha256': native_model_sha,
        'compiled_source_onnx_sha256': compiler_source_sha,
        'compiled_source_relation_to_extra_original_onnx':
        'NOT_INDEPENDENTLY_ATTESTED',
        'png_sha256': crop_sha, 'input_sha256': native_input['sha256'],
        'native_capture_sha256': native_receipt_sha,
        'original_ort_capture_sha256': ort_receipt_sha,
        'native_process_authenticity': 'NOT_ATTESTED',
        'original_ort_process_authenticity': 'NOT_ATTESTED',
        'actual_ctc_text_generation': (
            'SAME_RUN_SELF_REPORTED_NOT_INDEPENDENTLY_ATTESTED'
            if row['schema'] == _STATIC_CTC['schema'] else 'NOT_RUN'),
        'det_document_roi': 'NOT_RUN',
        'human_ground_truth': 'NOT_ATTESTED',
        'facade_manifest_issued': False, 'ready_issued': False,
    }


def inspect_selected_rec_pairwise(
        native_capture, ort_capture, selected_input_stage, *,
        source_dir, dictionary, approved_tree_sha256,
        native_model, original_onnx, expected_native_sha256,
        expected_original_sha256, output_name):
    try:
        return _inspect(
            native_capture, ort_capture, selected_input_stage,
            source_dir=source_dir, dictionary=dictionary,
            approved_tree_sha256=approved_tree_sha256,
            native_model=native_model, original_onnx=original_onnx,
            expected_native_sha256=expected_native_sha256,
            expected_original_sha256=expected_original_sha256,
            output_name=output_name)
    except SelectedRecPairwiseRefused:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, OverflowError, UnicodeError, RecursionError,
            struct.error, oracle.InvalidEvidence, oracle.MissingEvidence,
            raw.CaptureRefused, native_helper.NativeDetCaptureRefused,
            source.SelectedRecOracleRefused, rec.SelectedRecInputRefused,
            common.DetPairwiseRefused, ort_bridge.CaptureBridgeRefused) as exc:
        raise SelectedRecPairwiseRefused(
            'selected original ORT/native Pascal Rec evidence refused: %s' % exc
        ) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-rec-pairwise',
        description='SHA-bound selected Server Rec BGR input and original ORT/'
                    'Pascal CPU post-softmax numerical comparison, advisory only.')
    for flag in (
        '--native-capture', '--ort-capture', '--selected-input-stage',
        '--source-dir', '--dictionary', '--approved-tree-sha256',
        '--native-model', '--original-onnx', '--native-sha256',
        '--original-sha256', '--output-name',
    ):
        parser.add_argument(flag, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        result = inspect_selected_rec_pairwise(
            args.native_capture, args.ort_capture, args.selected_input_stage,
            source_dir=args.source_dir, dictionary=args.dictionary,
            approved_tree_sha256=args.approved_tree_sha256,
            native_model=args.native_model, original_onnx=args.original_onnx,
            expected_native_sha256=args.native_sha256,
            expected_original_sha256=args.original_sha256,
            output_name=args.output_name)
    except SelectedRecPairwiseRefused as exc:
        print(json.dumps({
            'status': 'REFUSED', 'acceptance_status': 'NOT_CHECKED',
            'native_process_authenticity': 'NOT_ATTESTED',
            'human_ground_truth': 'NOT_ATTESTED',
            'ready_issued': False, 'reason': str(exc)},
            ensure_ascii=False, allow_nan=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
