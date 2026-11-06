# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Read-only, independent NUMERIC evidence gate for a draft OCR composite.

Run only on captured files, never launch ONNX Runtime, GPU, or a native process.
The producer must separately execute the ORIGINAL Det/Rec ONNX models on the
same source image/crops, preserve their raw outputs and preprocessed inputs,
capture the Pascal tensors BEFORE postprocessing, and independently label the
text and boxes. A repeated native stdout transcript is never sufficient.

Tensor transport: flat little-endian float32 (.f32le) in declared C order,
with a SHA-256, shape and dtype for EVERY file. No pickle or automatic layout
guessing. Comparison is streamed with bounded memory and checks every scalar.

The result NEVER claims VERIFIED_LOCAL for the composite and never writes
manifest.json or modifies the source plan. Even complete numerical agreement
is a conditional assessment of the supplied captures; their producers are
external and their independence cannot be attested by this parser alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import bundle_atomic
from . import ocr_composite as composite


EVIDENCE_SCHEMA = 'crosspascal.paddleocr.independent-evidence'
EVIDENCE_VERSION = 1
MAX_TENSOR_ELEMENTS = 8_000_000
MAX_RECOGNITION_ELEMENTS = 2_000_000
MAX_CANDIDATES = 64
MAX_IMAGE_PIXELS = 8_000_000
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024


MAX_SOURCE_BYTES = bundle_atomic.MAX_NATIVE_FAMILY_BYTES
INPUT_ATOL = INPUT_RTOL = 1e-5
OUTPUT_ATOL = OUTPUT_RTOL = 0.003
BOX_ATOL_PX = 0.75
SCORE_ATOL = 0.003


class MissingEvidence(Exception):
    """An essential independent observation is absent; do not judge the model."""


class InvalidEvidence(Exception):
    """All necessary observations were supplied but conflict or are invalid."""


def _needed(mapping, key, where):
    if not isinstance(mapping, dict):
        raise InvalidEvidence('%s must be an object' % where)
    if key not in mapping or mapping[key] is None:
        raise MissingEvidence('%s.%s is required' % (where, key))
    return mapping[key]


def _object(value, where):
    if not isinstance(value, dict):
        raise InvalidEvidence('%s must be an object' % where)
    return value


def _matches(value, expected, where):
    if value != expected or type(value) is not type(expected):
        raise InvalidEvidence('%s disagrees with the measured/recorded '
                              'contract: %r versus %r'
                              % (where, value, expected))


def _sha(value, where):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdefABCDEF' for c in value)):
        raise InvalidEvidence('%s must contain a SHA-256 hexadecimal hash'
                              % where)
    return value.lower()


def _digest(path, *, max_bytes=None, expected_bytes=None, identity=None):
    sha = hashlib.sha256()
    length = 0
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if identity is not None and identity != (opened.st_dev, opened.st_ino):
            raise InvalidEvidence('evidence source FileID changed before SHA')
        if max_bytes is not None and opened.st_size > max_bytes:
            raise InvalidEvidence('evidence source exceeds byte limit before SHA')
        if expected_bytes is not None and opened.st_size != expected_bytes:
            raise InvalidEvidence('evidence source actual bytes differ from declared shape/size')
        limit = min(v for v in (max_bytes, expected_bytes) if v is not None) if (
            max_bytes is not None or expected_bytes is not None) else None
        while True:


            amount = 1 << 20 if limit is None else min(1 << 20, limit - length + 1)
            block = stream.read(amount)
            if not block:
                break
            length += len(block)
            if limit is not None and length > limit:
                raise InvalidEvidence('evidence source grew beyond byte limit during SHA')
            sha.update(block)
    return sha.hexdigest(), length


def _file(root, name, where, expected_sha=None, *, max_bytes=None,
          expected_bytes=None):

    try:
        path = composite._source_file(root, composite._safe_name(name))
    except composite.CompositeContractError as exc:
        raise MissingEvidence('%s: %s' % (where, exc)) from exc
    observed = path.stat()
    if not observed.st_size:
        raise MissingEvidence('%s is an empty evidence file' % where)
    if max_bytes is not None and observed.st_size > max_bytes:
        raise InvalidEvidence('%s exceeds %d-byte limit before SHA'
                              % (where, max_bytes))
    if expected_bytes is not None and observed.st_size != expected_bytes:
        raise InvalidEvidence('%s shape/declared bytes require %d, got %d '
                              '(rejected before SHA)' %
                              (where, expected_bytes, observed.st_size))
    actual, length = _digest(
        path, max_bytes=max_bytes, expected_bytes=expected_bytes,
        identity=(observed.st_dev, observed.st_ino))
    if not length:
        raise MissingEvidence('%s became empty during SHA' % where)
    if expected_bytes is not None and length != expected_bytes:
        raise InvalidEvidence('%s changed byte length during SHA' % where)
    if expected_sha is not None and actual != _sha(expected_sha, where):
        raise InvalidEvidence('%s SHA-256 mismatch: expected %s, found %s'
                              % (where, expected_sha, actual))
    return path, actual, length


def _read_json(root, name, where, sha=None):
    path, digest, _ = _file(root, name, where, sha,
                            max_bytes=MAX_JSON_BYTES)
    try:


        payload = composite._read_bytes(path, limit=MAX_JSON_BYTES)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise InvalidEvidence('%s JSON bytes changed after SHA' % where)
        record = json.loads(
            payload.decode('utf-8-sig'),
            object_pairs_hook=composite._duplicate_json_keys,
            parse_constant=lambda value: composite._reject(
                'non-finite JSON constant %s' % value))
        return record, digest
    except (UnicodeError, json.JSONDecodeError, RecursionError,
            composite.CompositeContractError) as exc:
        raise InvalidEvidence('%s: %s' % (where, exc)) from exc


def _validated_source_png(file, width, height, *, expected_sha=None):
    """Validate the actual lossless RGB image metadata, CRC and pixel stream.

    Captures must include a PNG8 RGB, non-interlaced image. We do not try to
    reproduce OpenCV resizing here (the intermediate tensors provide that
    measurement); unsupported or corrupt image formats remain NOT_CHECKED.
    No image file is ever written.
    """
    if width * height > MAX_IMAGE_PIXELS:
        raise MissingEvidence('source image exceeds bounded pixel capacity')
    with file.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size > MAX_IMAGE_BYTES:
            raise MissingEvidence('source PNG exceeds bounded read capacity')
        payload = stream.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise InvalidEvidence('source PNG grew beyond bounded read capacity')
    if expected_sha is not None and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise InvalidEvidence('source PNG changed bytes after original SHA')
    if not payload.startswith(b'\x89PNG\r\n\x1a\n'):
        raise MissingEvidence('source image must be lossless non-interlaced '
                              '8-bit RGB PNG with a readable IHDR/IDAT; '
                              'arbitrary bytes cannot serve as image evidence')
    offset = 8
    seen_header = False
    seen_idat = False
    seen_end = False
    idat = bytearray()
    while offset + 12 <= len(payload):
        length, = struct.unpack_from('>I', payload, offset)
        kind = payload[offset + 4:offset + 8]
        offset += 8
        if length > MAX_IMAGE_BYTES or offset + length + 4 > len(payload):
            raise InvalidEvidence('source PNG is truncated/oversized')
        data = payload[offset:offset + length]
        crc, = struct.unpack_from('>I', payload, offset + length)
        if zlib.crc32(kind + data) & 0xffffffff != crc:
            raise InvalidEvidence('source PNG chunk CRC mismatch')
        offset += length + 4
        if not seen_header:
            if kind != b'IHDR' or length != 13:
                raise InvalidEvidence('source PNG lacks a first IHDR chunk')
            seen_header = True
            image_width, image_height, depth, color, comp, filt, interlace = (
                struct.unpack('>IIBBBBB', data))
            if (image_width != width or image_height != height
                    or (depth, color, comp, filt, interlace)
                    != (8, 2, 0, 0, 0)):
                raise InvalidEvidence('PNG IHDR dimensions/8-bit RGB layout '
                                      'disagree with captured source image')
        elif kind == b'IDAT':
            seen_idat = True
            idat.extend(data)
            if len(idat) > MAX_IMAGE_BYTES:
                raise InvalidEvidence('PNG IDAT exceeds bounded capacity')
        elif kind == b'IEND':
            if length != 0:
                raise InvalidEvidence('PNG IEND must be empty')
            seen_end = True
            break
        elif kind[0] & 32 == 0:
            raise MissingEvidence('PNG critical chunk %r is unsupported '
                                  'by this strict lossless evidence gate'
                                  % kind)
    if not (seen_header and seen_idat and seen_end) or offset != len(payload):
        raise InvalidEvidence('source PNG missing IDAT/IEND or has trailing data')
    expected = (width * 3 + 1) * height
    try:
        decompress = zlib.decompressobj()
        scanlines = decompress.decompress(bytes(idat), expected + 1)
    except zlib.error as exc:
        raise InvalidEvidence('source PNG IDAT decompression failed: %s'
                              % exc) from exc
    if len(scanlines) != expected or not decompress.eof or (
            decompress.unused_data or decompress.unconsumed_tail):
        raise InvalidEvidence('source PNG pixel stream length is invalid')
    row_bytes = width * 3 + 1
    for row in range(height):
        if scanlines[row * row_bytes] > 4:
            raise InvalidEvidence('source PNG uses unknown row filter')
    return {'width': width, 'height': height, 'format': 'lossless_png_rgb8'}


def _tensor(root, value, where, *, expected_shape=None,
            max_elements=MAX_TENSOR_ELEMENTS):
    value = _object(value, where)
    shape = _needed(value, 'shape', where)
    if (not isinstance(shape, list) or not shape
            or any(type(v) is not int or v < 1 for v in shape)
            or len(shape) > 5):
        raise InvalidEvidence('%s.shape must contain positive dimensions'
                              % where)
    if expected_shape is not None and shape != expected_shape:
        raise InvalidEvidence('%s.shape %s differs from model input %s'
                              % (where, shape, expected_shape))
    _matches(_needed(value, 'dtype', where), 'float32le', where + '.dtype')
    total = 1
    for dimension in shape:
        total *= dimension
        if total > max_elements:
            raise InvalidEvidence('%s exceeds bounded tensor cap (%d)'
                                  % (where, max_elements))
    name = _needed(value, 'file', where)
    path, digest, size = _file(root, name, where,
                               expected_sha=_needed(value, 'sha256', where),
                               expected_bytes=total * 4)
    if size != total * 4:
        raise InvalidEvidence('%s shape requires %d bytes, got %d'
                              % (where, total * 4, size))
    return {'path': path, 'shape': shape, 'count': total,
            'sha256': digest, 'bytes': size}


def _compare_tensors(reference, observed, label, *, atol, rtol,
                     nonvacuous=True):
    if reference['shape'] != observed['shape']:
        raise InvalidEvidence('%s ONNX Runtime/native tensor shapes differ'
                              % label)
    largest, measured = 0.0, 0
    failures, nonzero = 0, 0
    expected = reference['count'] * 4
    if observed['count'] != reference['count']:
        raise InvalidEvidence('%s tensor element counts differ' % label)
    left_sha, right_sha = hashlib.sha256(), hashlib.sha256()
    with reference['path'].open('rb') as left, observed['path'].open('rb') as right:
        if (os.fstat(left.fileno()).st_size != expected
                or os.fstat(right.fileno()).st_size != expected):
            raise InvalidEvidence('%s tensor bytes changed since admission' % label)
        remaining = expected
        while remaining:
            portion = min(32768, remaining)
            a, b = left.read(portion), right.read(portion)
            if not a or len(a) != len(b) or len(a) % 4:
                raise InvalidEvidence('%s has a partial f32 word' % label)
            remaining -= len(a)
            left_sha.update(a)
            right_sha.update(b)
            for (x,), (y,) in zip(struct.iter_unpack('<f', a),
                                 struct.iter_unpack('<f', b)):
                measured += 1
                if not math.isfinite(x) or not math.isfinite(y):
                    raise InvalidEvidence('%s contains NaN/Inf at offset %d'
                                          % (label, measured - 1))
                if x != 0:
                    nonzero += 1
                delta = abs(x - y)
                largest = max(largest, delta)
                if delta > atol + rtol * abs(x):
                    failures += 1
        if left.read(1) or right.read(1):
            raise InvalidEvidence('%s tensor grew during bounded comparison' % label)
    if (left_sha.hexdigest() != reference['sha256']
            or right_sha.hexdigest() != observed['sha256']):
        raise InvalidEvidence('%s tensor SHA-256 changed after admission' % label)
    if measured != reference['count']:
        raise InvalidEvidence('%s comparison truncated' % label)
    if nonvacuous and nonzero == 0:
        raise MissingEvidence('%s original ORT tensor is entirely zero; '
                              'numerical comparison is vacuous' % label)
    if failures:
        raise InvalidEvidence('%s mismatches in %d/%d scalars '
                              '(max abs %.8g; atol=%g rtol=%g)'
                              % (label, failures, measured, largest, atol, rtol))
    return {'elements': measured, 'max_abs_error': largest,
            'mismatched': 0, 'atol': atol, 'rtol': rtol}


def _tensor_at(tensor, y, x, channel):
    shape = tensor['shape']
    if len(shape) != 4 or shape[0:2] != [1, 3]:
        raise InvalidEvidence('preprocessed input must be NCHW [1,3,H,W]')
    if y < 0 or y >= shape[2] or x < 0 or x >= shape[3]:
        raise InvalidEvidence('preprocessing pixel sample lies outside tensor')
    position = (channel * shape[2] * shape[3] + y * shape[3] + x) * 4
    with tensor['path'].open('rb') as stream:
        if os.fstat(stream.fileno()).st_size != tensor['bytes']:
            raise InvalidEvidence('preprocessing tensor changed byte count')
        stream.seek(position)
        data = stream.read(4)
    if len(data) != 4:
        raise InvalidEvidence('preprocessing tensor lacks pixel sample bytes')
    (value,) = struct.unpack('<f', data)
    if not math.isfinite(value):
        raise InvalidEvidence('preprocessing pixel sample is non-finite')
    return value


def _preprocessing_samples(row, stage, ort_tensor, native_tensor, where):
    samples = _needed(row, 'samples', where)
    if not isinstance(samples, list) or len(samples) < (2 if stage == 'det' else 2):
        raise MissingEvidence('%s needs at least two measured pixel samples'
                              % where)
    if len(samples) > 32:
        raise InvalidEvidence('%s has excessive pixel samples' % where)
    colors, padded = set(), 0
    positions = set()
    for index, pixel in enumerate(samples):
        name = '%s.samples[%d]' % (where, index)
        pixel = _object(pixel, name)
        y = _needed(pixel, 'y', name)
        x = _needed(pixel, 'x', name)
        if type(y) is not int or type(x) is not int:
            raise InvalidEvidence('%s pixel coordinates must be integers' % name)
        if (y, x) in positions:
            raise InvalidEvidence('%s duplicates a pixel sample' % name)
        positions.add((y, x))
        raw = _needed(pixel, 'raw_bgr_after_resize', name)
        if (not isinstance(raw, list) or len(raw) != 3
                or any(type(c) is not int or not 0 <= c <= 255 for c in raw)):
            raise InvalidEvidence('%s raw BGR bytes must be [B,G,R] in 0..255'
                                  % name)
        is_pad = _needed(pixel, 'right_padding', name)
        if type(is_pad) is not bool:
            raise InvalidEvidence('%s right_padding must be explicit bool'
                                  % name)
        if stage == 'det' and is_pad:
            raise InvalidEvidence('DET has no right padding')
        if is_pad:
            if raw != [0, 0, 0]:
                raise InvalidEvidence('%s raw right padding must be BGR=0'
                                      % name)
            padded += 1
        else:
            colors.add(tuple(raw))
        norm = [((raw[c] / 255.0 - composite._DET_BGR_MEAN[c])
                 / composite._DET_BGR_STD[c]) for c in range(3)] if stage == 'det' else [
                     (raw[2 - c] / 255.0 * 2 - 1) for c in range(3)]
        for channel, expected in enumerate(norm):
            for tensor, road in ((ort_tensor, 'independent ORT preproc'),
                                 (native_tensor, 'native Pascal preproc')):
                actual = _tensor_at(tensor, y, x, channel)
                if abs(actual - expected) > INPUT_ATOL + INPUT_RTOL * abs(expected):
                    raise InvalidEvidence(
                        '%s channel %d %s violates exact %s math '
                        '(raw_BGR=%r, got %.8g, expected %.8g)'
                        % (name, channel, road, stage, raw, actual, expected))
    if not any(len(set(color)) > 1 for color in colors):
        raise MissingEvidence('%s needs a non-grey calibrated BGR sample to '
                              'distinguish channel order' % where)
    if stage == 'det' and len(colors) < 2:
        raise MissingEvidence('DET needs two distinct non-padding pixel values')
    if stage == 'rec' and padded < 1:
        raise MissingEvidence('REC needs at least one actual right-pad pixel '
                              'to measure tensor value -1')

    for label, tensor in (('ORT', ort_tensor), ('native', native_tensor)):
        _file(tensor['path'].parent, tensor['path'].name,
              '%s %s preprocessing tensor' % (where, label),
              tensor['sha256'], expected_bytes=tensor['bytes'])
    return {'measured_pixels': len(samples), 'color_cases': len(colors),
            'right_padding_pixels': padded}


def _box(raw, label, width, height, *, score_required):
    raw = _object(raw, label)
    name = _needed(raw, 'id', label)
    if (not isinstance(name, str) or not name
            or len(name) > 128 or any(ord(ch) < 32 for ch in name)):
        raise InvalidEvidence('%s must have a short nonempty ID' % label)
    coords = _needed(raw, 'box_xyxy', label)
    if (not isinstance(coords, list) or len(coords) != 4
            or any(type(v) not in (float, int) or not math.isfinite(v)
                   for v in coords)):
        raise InvalidEvidence('%s box must have four finite pixel coordinates'
                              % label)
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise InvalidEvidence('%s box lies outside the source image' % label)
    if score_required:
        score = _needed(raw, 'score', label)
        if type(score) not in (float, int) or not (
                math.isfinite(score) and 0 < score <= 1):
            raise InvalidEvidence('%s score must be finite 0 < p <= 1'
                                  % label)
    else:
        score = None
        text = _needed(raw, 'text', label)
        if (not isinstance(text, str) or not text or len(text) > 4096
                or any(c in text for c in '\r\n\x00')):
            raise MissingEvidence('%s has no valid independent ground truth '
                                  'transcription' % label)
    return {'id': name, 'xyxy': tuple(coords), 'score': score,
            'text': raw.get('text') if not score_required else None}


def _candidates(values, where, width, height, *, score_required):
    if not isinstance(values, list) or not values:
        raise MissingEvidence('%s is empty or missing; no candidate alignment'
                              % where)
    if len(values) > MAX_CANDIDATES:
        raise MissingEvidence('%s needs >%d candidates; increase the bounded '
                              'gate only after resource review'
                              % (where, MAX_CANDIDATES))
    parsed = [
        _box(item, '%s[%d]' % (where, n), width, height,
             score_required=score_required)
        for n, item in enumerate(values)
    ]
    if len({item['id'] for item in parsed}) != len(parsed):
        raise InvalidEvidence('%s duplicates candidate IDs' % where)
    return parsed


def _one_to_one(left, right, compatible, label):
    """Require a mathematically UNIQUE perfect bipartite matching.

    A perfect matching alone can hide ambiguous close boxes. Eliminate every
    edge of the selected assignment in turn and reject if another perfect
    matching still exists. At most MAX_CANDIDATES nodes, bounded.
    """
    if len(left) != len(right):
        raise InvalidEvidence('%s candidates missing/extra: %d vs %d'
                              % (label, len(left), len(right)))
    count = len(left)
    if not count:
        raise MissingEvidence('%s has no nonvacuous detections' % label)

    adjacency = [[j for j, b in enumerate(right) if compatible(a, b)]
                 for a in left]
    if any(not row for row in adjacency):
        raise InvalidEvidence('%s has a candidate with no box/score partner'
                              % label)

    def match(edges, without=None):
        taken = [-1] * count
        assignment = [-1] * count

        def visit(i, seen):
            for j in edges[i]:
                if (i, j) == without or j in seen:
                    continue
                seen.add(j)
                if taken[j] < 0 or visit(taken[j], seen):
                    taken[j] = i
                    assignment[i] = j
                    return True
            return False

        for i in sorted(range(count), key=lambda n: len(edges[n])):
            if not visit(i, set()):
                return None
        return assignment

    answer = match(adjacency)
    if answer is None:
        raise InvalidEvidence('%s lacks complete unique-use 1:1 matching'
                              % label)
    for i, j in enumerate(answer):
        if match(adjacency, without=(i, j)) is not None:
            raise MissingEvidence('%s 1:1 pairing is ambiguous: multiple '
                                  'perfect assignments; capture distinguishable '
                                  'boxes/IDs, do not guess' % label)
    return {left[i]['id']: right[j]['id'] for i, j in enumerate(answer)}


def _close_boxes(a, b, tolerance=BOX_ATOL_PX):
    return all(abs(x - y) <= tolerance for x, y
               in zip(a['xyxy'], b['xyxy']))


def _aligned_candidates(reference, native, truth):
    ref_to_native = _one_to_one(
        reference, native,
        lambda a, b: (_close_boxes(a, b)
                      and abs(a['score'] - b['score']) <= SCORE_ATOL),
        'independent ORT Det -> Pascal Det',
    )
    ref_to_gt = _one_to_one(
        reference, truth,
        lambda a, b: _close_boxes(a, b, tolerance=1.0),
        'independent ORT Det -> human ground-truth annotation',
    )
    return ref_to_native, ref_to_gt


def _recognition_tokens(bundle, plan):
    dict_info = _object(_needed(plan, 'dictionary', 'plan'), 'dictionary')
    name = _needed(dict_info, 'file', 'dictionary')
    parts = name.split('/')
    if len(parts) != 2 or parts[0] != 'dictionary':
        raise InvalidEvidence('dictionary bundle path is unsafe')
    file, digest, size = _file(
        bundle / 'dictionary', parts[1], 'staged OCR dictionary',
        _needed(dict_info, 'sha256', 'dictionary'),
        max_bytes=composite._MAX_DICTIONARY_BYTES)
    if size != dict_info['bytes']:
        raise InvalidEvidence('staged OCR dictionary size differs from plan')
    try:
        tokens = composite._dictionary_tokens(file)
    except composite.CompositeContractError as exc:
        raise InvalidEvidence('invalid staged dictionary: %s' % exc) from exc
    if plan['ctc'].get('use_space_char') is not True:
        raise InvalidEvidence('native CTC requires appended space')
    tokens.append(' ')
    expected = plan['ctc']['expected_classes']
    if len(tokens) + (plan['ctc']['blank_slot'] == 'excluded') != expected:
        raise InvalidEvidence('dictionary and CTC output class count differ')
    return tokens


def _decode_ctc(tensor, ctc, tokens, where):
    """Independent greedy CTC with explicit blank mapping and token frame spans.

    All class logits are read, finite checked, then argmax class IDs collapse
    ONLY consecutively, with blank separating repeated characters. Frames
    return exact alignment intervals so transcribed text stays tied to logits.
    """
    shape = tensor['shape']
    layout = ctc['logits_layout']
    classes = ctc['expected_classes']
    if layout == 'NTC':
        if len(shape) != 3 or shape[0] != 1:
            raise InvalidEvidence('%s must have [1,T,C] logits' % where)
        times, class_axis = shape[1], shape[2]
    elif layout == 'NCT':
        if len(shape) != 3 or shape[0] != 1:
            raise InvalidEvidence('%s must have [1,C,T] logits' % where)
        times, class_axis = shape[2], shape[1]
    elif layout == 'TC':
        if len(shape) != 2:
            raise InvalidEvidence('%s must have [T,C] logits' % where)
        times, class_axis = shape[0], shape[1]
    elif layout == 'CT':
        if len(shape) != 2:
            raise InvalidEvidence('%s must have [C,T] logits' % where)
        times, class_axis = shape[1], shape[0]
    else:
        raise InvalidEvidence('%s CTC layout is unsupported' % where)
    if class_axis != classes or times < 2 or times > 2048:
        raise InvalidEvidence('%s CTC shape has unexpected classes/time '
                              '(%d versus %d)' % (where, class_axis, classes))
    if tensor['count'] > MAX_RECOGNITION_ELEMENTS:
        raise InvalidEvidence('%s oversized REC logits' % where)
    required = tensor['count'] * 4
    with tensor['path'].open('rb') as stream:
        if os.fstat(stream.fileno()).st_size != required:
            raise InvalidEvidence('%s CTC logits changed bytes after SHA' % where)
        encoded = stream.read(required + 1)
    if len(encoded) != required:
        raise InvalidEvidence('%s CTC logits changed bytes during bounded read' % where)
    if hashlib.sha256(encoded).hexdigest() != tensor['sha256']:
        raise InvalidEvidence('%s CTC logits SHA-256 changed after admission' % where)
    floats = struct.unpack('<%df' % tensor['count'], encoded)
    if not all(math.isfinite(x) for x in floats):
        raise InvalidEvidence('%s CTC logits contain NaN/Inf' % where)
    blank = ctc['blank_index']
    if blank not in (0, classes - 1):
        raise InvalidEvidence('%s CTC blank must be explicitly 0 or last'
                              % where)
    frames = []
    text = []
    previous = -1
    for t in range(times):
        values = [floats[t * classes + c]
                  if layout in ('TC', 'NTC') else floats[c * times + t]
                  for c in range(classes)]
        winner = max(range(classes), key=lambda c: values[c])
        if winner == blank:
            previous = blank
            continue
        if winner == previous:
            frames[-1]['end_frame'] = t
            continue
        if ctc['blank_slot'] == 'included':
            index = winner
        elif blank == 0:
            index = winner - 1
        else:
            index = winner
        if not 0 <= index < len(tokens):
            raise InvalidEvidence('%s has CTC class %d without a dictionary '
                                  'token' % (where, winner))
        token = tokens[index]
        if not token:
            raise InvalidEvidence('%s selected an empty dictionary token'
                                  % where)
        text.append(token)
        frames.append({'class_index': winner, 'token': token,
                       'start_frame': t, 'end_frame': t})
        previous = winner
    if not frames:
        raise MissingEvidence('%s greedily decoded only blank classes' % where)
    return {'text': ''.join(text), 'frames': frames, 'timesteps': times}


def _edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prior = list(range(len(b) + 1))
    for i, ch in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, symbol in enumerate(b, 1):
            current[j] = min(current[j-1] + 1, prior[j] + 1,
                             prior[j-1] + (ch != symbol))
        prior = current
    return prior[-1]


def _checked_original_source_files(evidence_root, evidence, plan):
    sources = _needed(evidence, 'original_onnx_files', 'evidence')
    sources = _object(sources, 'original_onnx_files')
    if set(sources) != {'det', 'rec'}:
        raise MissingEvidence('original_onnx_files must name Det and Rec '
                              'ORIGINAL unmodified ONNX models, not TVM binaries')
    details = {}
    for stage in ('det', 'rec'):
        row = _object(sources[stage], 'original_onnx_files.' + stage)
        name = _needed(row, 'file', stage + '.original_onnx')
        provenance = plan['components'][stage]['original_provenance']
        expected = _needed(provenance, 'source_hash', 'plan provenance')
        _, digest, size = _file(evidence_root, name,
                                 'original_onnx_files.' + stage, expected,
                                 max_bytes=MAX_SOURCE_BYTES)
        _matches(_needed(row, 'sha256', stage + '.original_onnx'),
                 digest, 'original_onnx_files.' + stage + '.sha256')
        details[stage] = {'source_sha256': digest, 'bytes': size}
    return details


def _validate_plan(bundle):
    try:
        directory = composite._existing_directory(bundle)
    except composite.CompositeContractError as exc:
        raise MissingEvidence('staged OCR composite bundle absent: %s'
                              % exc) from exc
    plan, digest = _read_json(directory, composite.PLAN_NAME,
                              'composite-plan.json')
    if not isinstance(plan, dict):
        raise InvalidEvidence('composite-plan.json is not an object')
    expected = {
        'schema': 'crosspascal.paddleocr.composite-plan',
        'schema_version': 2, 'status': 'DRAFT',
        'composite_verification': 'NOT_CHECKED',
        'facade_manifest_issued': False,
        'native_composite_oracle': 'NOT_RUN',
    }
    for key, value in expected.items():
        _matches(_needed(plan, key, 'composite plan'), value, 'plan.' + key)
    for name in ('manifest.json', 'manifest.txt'):
        if os.path.lexists(directory / name):
            raise InvalidEvidence('a root %s exists; the composite must '
                                  'remain DRAFT, never a fake facade manifest'
                                  % name)
    assets = _needed(plan, 'assets', 'plan')
    if not isinstance(assets, list) or not assets:
        raise MissingEvidence('composite asset inventory is empty')
    names = set()
    inventory = {}
    admitted = []
    admitted_total = 0
    for i, item in enumerate(assets):
        item = _object(item, 'plan.assets[%d]' % i)
        path = _needed(item, 'path', 'plan asset')
        if not isinstance(path, str):
            raise InvalidEvidence('plan asset path must be a string')
        parts = path.split('/')
        if len(parts) != 2 or parts[0] not in ('det', 'rec', 'dictionary'):
            raise InvalidEvidence('plan has unsafe asset path %r' % path)
        if path.casefold() in names:
            raise InvalidEvidence('duplicate asset path in composite plan')
        names.add(path.casefold())
        size = _needed(item, 'bytes', 'plan asset')
        if type(size) is not int or not 1 <= size <= MAX_SOURCE_BYTES:
            raise InvalidEvidence('plan asset needs bounded positive byte count')
        try:
            member = composite._source_file(directory / parts[0], parts[1])
        except composite.CompositeContractError as exc:
            raise MissingEvidence('plan asset: %s' % exc) from exc
        if member.stat().st_size != size:
            raise InvalidEvidence('SHA-256 mismatch or byte-length change: '
                                  'composite asset size differs from plan '
                                  '(refused before SHA)')
        admitted_total += size
        if admitted_total > MAX_SOURCE_BYTES:
            raise InvalidEvidence('composite selected assets exceed aggregate '
                                  'byte quota before SHA')
        admitted.append((i, item, parts, size))


    for i, item, parts, expected_size in admitted:
        path = item['path']
        _, digest_file, size = _file(
            directory / parts[0], parts[1],
            'plan.assets[%d]' % i, _needed(item, 'sha256', 'plan asset'),
            max_bytes=MAX_SOURCE_BYTES, expected_bytes=expected_size)
        inventory[path] = {'sha256': digest_file, 'bytes': size}
    components = _object(_needed(plan, 'components', 'plan'), 'plan.components')
    if set(components) != {'det', 'rec'}:
        raise InvalidEvidence('plan must name exactly Det and Rec components')
    for stage in ('det', 'rec'):
        source = _object(components[stage], 'plan.components.' + stage)
        path = _needed(source, 'artifact_path', stage)
        fingerprint = _needed(source, 'artifact_sha256', stage)
        if path not in inventory or inventory[path]['sha256'] != fingerprint:
            raise InvalidEvidence('%s binary artifact SHA does not match '
                                  'component identity or asset inventory'
                                  % stage)
        manifest, _ = _read_json(directory / stage, 'manifest.json',
                                 'staged %s manifest' % stage)
        recipe, _ = _read_json(directory / stage, 'recipe.json',
                               'staged %s recipe' % stage)
        if (manifest.get('artifact', {}).get('hash') != fingerprint
                or manifest['artifact']['file'] != path.split('/')[1]
                or manifest.get('provenance') != source.get('original_provenance')
                or recipe.get('source_hash') != (
                    source.get('original_provenance') or {}).get('source_hash')):
            raise InvalidEvidence('%s staged source/manifest/recipe identity '
                                  'disagrees with the original composite plan'
                                  % stage)
    dictionary = _object(_needed(plan, 'dictionary', 'plan'),
                         'plan.dictionary')
    dict_path = _needed(dictionary, 'file', 'plan.dictionary')
    if (dict_path not in inventory
            or inventory[dict_path]['sha256'] != dictionary['sha256']):
        raise InvalidEvidence('staged dictionary SHA differs from tracked '
                              'composite dictionary identity')
    return directory, plan, digest


def _validate_measured_process(evidence, plan, bundle, image_hash,
                               original_source_hashes):
    runners = _object(_needed(evidence, 'runners', 'evidence'), 'runners')
    ort = _object(_needed(runners, 'ort', 'runners'), 'runners.ort')
    native = _object(_needed(runners, 'native', 'runners'), 'runners.native')
    for name, record in (('ort', ort), ('native', native)):
        _matches(_needed(record, 'exit_code', name), 0, name + '.exit_code')
        _matches(_needed(record, 'image_sha256', name), image_hash,
                 name + '.image_sha256')
    _matches(_needed(ort, 'execution_provider', 'ort'),
             'CPUExecutionProvider', 'ort.execution_provider')
    _matches(_needed(ort, 'original_onnx_sha256', 'ort'),
             original_source_hashes, 'ort.original_onnx_sha256')
    _matches(_needed(native, 'target', 'native'), plan['target'],
             'native target OS/CPU/CUDA SM/driver')
    _matches(_needed(native, 'model_sha256', 'native'), {
        stage: plan['components'][stage]['artifact_sha256']
        for stage in ('det', 'rec')}, 'native compiled-model fingerprint')
    _matches(_needed(native, 'before_first_inference', 'native'),
             True, 'native.before_first_inference')

    expected_env = plan['native_process_environment']
    observed = _object(_needed(native, 'observed_environment', 'native'),
                       'native.observed_environment')
    _matches(_needed(observed, 'set', 'native environment'),
             expected_env['required_set'], 'native measured environment')
    _matches(sorted(_needed(observed, 'unset', 'native environment')),
             sorted(expected_env['required_unset']),
             'native measured absent environment variables')

    native_root = _needed(native, 'bundle_root', 'native')
    path_type = (PureWindowsPath if plan['target']['os'] == 'windows'
                 else PurePosixPath)
    if not isinstance(native_root, str) or not native_root:
        raise MissingEvidence('native runner must capture its absolute bundle_root')
    target_root = path_type(native_root)
    if (not target_root.is_absolute()
            or '..' in target_root.parts):
        raise InvalidEvidence('native bundle_root is relative or contains '
                              'path traversal: %r' % native_root)
    rel = plan['dictionary']['file'].split('/')
    if len(rel) != 2 or rel[0] != 'dictionary':
        raise InvalidEvidence('staged OCR dictionary path is unsafe')
    expected_native_dictionary = target_root.joinpath(*rel)
    declared_path = _needed(observed, 'PADDLE_OCR_DICT', 'native environment')
    if not isinstance(declared_path, str):
        raise InvalidEvidence('native PADDLE_OCR_DICT is not a path')
    actual_dictionary = path_type(declared_path)
    if (not actual_dictionary.is_absolute()
            or '..' in actual_dictionary.parts):
        raise InvalidEvidence('native PADDLE_OCR_DICT is not absolute')
    normalize = (lambda p: str(p).casefold()
                 if path_type is PureWindowsPath else str(p))
    if normalize(actual_dictionary) != normalize(expected_native_dictionary):
        raise InvalidEvidence('native PADDLE_OCR_DICT does not point to the '
                              'SHA-tracked staged dictionary')
    _matches(_needed(observed, 'dictionary_sha256', 'native environment'),
             plan['dictionary']['sha256'],
             'native measured dictionary SHA-256')
    calibration = _object(_needed(native, 'preprocessing', 'native'),
                          'native.preprocessing')
    for stage in ('det', 'rec'):
        captured = _needed(calibration, stage, 'native.preprocessing')
        planned = plan['preprocessing'][stage]


        _matches(captured, planned,
                 'native measured %s preprocessing parameters' % stage)


def _validate_ground_truth(evidence_root, evidence, image_sha, width, height):
    name = _object(_needed(evidence, 'ground_truth', 'evidence'),
                   'ground_truth')
    record, digest = _read_json(
        evidence_root,
        _needed(name, 'file', 'ground_truth'),
        'external ground truth',
        sha=_needed(name, 'sha256', 'ground_truth'))
    if not isinstance(record, dict):
        raise InvalidEvidence('ground truth file must be a JSON object')
    _matches(_needed(record, 'schema', 'ground truth'),
             'crosspascal.ocr.human-annotations',
             'ground truth schema')
    _matches(_needed(record, 'image_sha256', 'ground truth'),
             image_sha, 'ground truth source image')
    annotator = record.get('human_annotator')
    if (not isinstance(annotator, str) or not annotator.strip()
            or len(annotator) > 256):
        raise MissingEvidence('ground truth lacks independent human annotator')
    rows = _needed(record, 'annotations', 'ground truth')
    return _candidates(rows, 'human ground truth', width, height,
                       score_required=False), digest


def _evaluate(bundle, evidence_root):
    bundle, plan, plan_hash = _validate_plan(bundle)
    try:
        evidence_root = composite._existing_directory(evidence_root)
    except composite.CompositeContractError as exc:
        raise MissingEvidence('no independently captured OCR evidence folder: '
                              '%s' % exc) from exc
    evidence, evidence_hash = _read_json(
        evidence_root, 'evidence.json', 'independent OCR evidence')
    evidence = _object(evidence, 'evidence.json')
    _matches(_needed(evidence, 'schema', 'evidence'), EVIDENCE_SCHEMA,
             'evidence schema')
    _matches(_needed(evidence, 'schema_version', 'evidence'),
             EVIDENCE_VERSION, 'evidence version')
    _matches(_needed(evidence, 'plan_sha256', 'evidence'),
             plan_hash, 'evidence binding to exact DRAFT plan')
    sources = _checked_original_source_files(evidence_root, evidence, plan)
    original_hashes = {role: record['source_sha256']
                       for role, record in sources.items()}
    source_image = _object(_needed(evidence, 'source_image', 'evidence'),
                           'source_image')
    _, image_hash, _ = _file(
        evidence_root, _needed(source_image, 'file', 'source_image'),
        'real OCR source image', _needed(source_image, 'sha256', 'source_image'),
        max_bytes=MAX_IMAGE_BYTES)
    width = _needed(source_image, 'width', 'source_image')
    height = _needed(source_image, 'height', 'source_image')
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 16384 or not 1 <= height <= 16384):
        raise InvalidEvidence('source image pixel dimensions are invalid')
    image_file = evidence_root / source_image['file']
    image_info = _validated_source_png(image_file, width, height,
                                       expected_sha=image_hash)

    _validate_measured_process(
        evidence, plan, bundle, image_hash, original_hashes)
    truth, gt_hash = _validate_ground_truth(
        evidence_root, evidence, image_hash, width, height)
    detection = _object(_needed(evidence, 'detection', 'evidence'),
                        'detection')
    expected_det_shape = plan['preprocessing']['det']['input_shape']
    ort_input = _tensor(evidence_root, _needed(detection, 'ort_input', 'detection'),
                        'original ORT DET input', expected_shape=expected_det_shape)
    native_input = _tensor(
        evidence_root, _needed(detection, 'native_input', 'detection'),
        'native DET input', expected_shape=expected_det_shape)
    metrics = {
        'det_input': _compare_tensors(
            ort_input, native_input, 'DET preprocessed input',
            atol=INPUT_ATOL, rtol=INPUT_RTOL),
    }
    metrics['det_preprocessing'] = _preprocessing_samples(
        detection, 'det', ort_input, native_input, 'detection')
    ort_output = _tensor(
        evidence_root, _needed(detection, 'ort_output', 'detection'),
        'original ORT DET raw probability/logits')
    native_output = _tensor(
        evidence_root, _needed(detection, 'native_output', 'detection'),
        'native DET raw probability/logits')
    metrics['det_output'] = _compare_tensors(
        ort_output, native_output, 'DET raw ORT-versus-native output',
        atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)

    reference = _candidates(
        _needed(detection, 'ort_candidates', 'detection'),
        'original ORT postprocessed candidates', width, height,
        score_required=True)
    native = _candidates(
        _needed(detection, 'native_candidates', 'detection'),
        'native Pascal postprocessed candidates', width, height,
        score_required=True)
    ref_to_native, ref_to_gt = _aligned_candidates(reference, native, truth)
    gt_lookup = {item['id']: item for item in truth}
    metrics['det_candidates'] = {'matched': len(ref_to_native),
                                 'uniquely_matched': True,
                                 'box_tolerance_px': BOX_ATOL_PX,
                                 'score_atol': SCORE_ATOL}
    recognition = _needed(evidence, 'recognition', 'evidence')
    if not isinstance(recognition, list) or len(recognition) != len(reference):
        raise MissingEvidence('one REC capture (ORT and native logits) is '
                              'required for EACH uniquely matched DET candidate')
    tokens = _recognition_tokens(bundle, plan)
    seen = set()
    alignments = []
    rec_shape = plan['preprocessing']['rec']['input_shape']
    for index, item in enumerate(recognition):
        label = 'recognition[%d]' % index
        item = _object(item, label)
        ref_id = _needed(item, 'ort_candidate_id', label)
        if ref_id not in ref_to_native or ref_id in seen:
            raise InvalidEvidence('%s has duplicate/unknown ORT candidate ID'
                                  % label)
        seen.add(ref_id)
        native_id = ref_to_native[ref_id]
        gt_id = ref_to_gt[ref_id]
        _matches(_needed(item, 'native_candidate_id', label),
                 native_id, label + '.native_candidate_id')
        _matches(_needed(item, 'ground_truth_id', label),
                 gt_id, label + '.ground_truth_id')
        reference_input = _tensor(
            evidence_root, _needed(item, 'ort_input', label),
            label + ' original ORT REC input', expected_shape=rec_shape)
        native_rec_input = _tensor(
            evidence_root, _needed(item, 'native_input', label),
            label + ' native REC input', expected_shape=rec_shape)
        input_diff = _compare_tensors(
            reference_input, native_rec_input, label + ' REC preprocessed input',
            atol=INPUT_ATOL, rtol=INPUT_RTOL)
        sample_metrics = _preprocessing_samples(
            item, 'rec', reference_input, native_rec_input, label)
        ort_logits = _tensor(
            evidence_root, _needed(item, 'ort_logits', label),
            label + ' original ONNXRuntime REC logits',
            max_elements=MAX_RECOGNITION_ELEMENTS)
        native_logits = _tensor(
            evidence_root, _needed(item, 'native_logits', label),
            label + ' native intermediate REC logits',
            max_elements=MAX_RECOGNITION_ELEMENTS)
        logits_diff = _compare_tensors(
            ort_logits, native_logits, label + ' REC class logits',
            atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
        ort_decoded = _decode_ctc(ort_logits, plan['ctc'], tokens,
                                   label + ' ORT CTC')
        native_decoded = _decode_ctc(native_logits, plan['ctc'], tokens,
                                      label + ' native CTC')
        gt_text = gt_lookup[gt_id]['text']
        measured_transcript = _needed(item, 'native_transcript', label)
        cer_numer = _edit_distance(ort_decoded['text'], gt_text)
        if (cer_numer != 0 or native_decoded['text'] != gt_text
                or measured_transcript != gt_text):
            raise InvalidEvidence(
                '%s bbox %s -> %s -> %s CTC transcript disagrees with '
                'human ground truth: ORT=%r, native_logits=%r, '
                'native_stdout=%r, GT=%r, edit_distance=%d'
                % (label, ref_id, native_id, gt_id,
                   ort_decoded['text'], native_decoded['text'],
                   measured_transcript, gt_text, cer_numer))
        alignments.append({
            'ort_candidate_id': ref_id,
            'native_candidate_id': native_id,
            'ground_truth_id': gt_id,
            'transcript': gt_text, 'character_error_rate': 0.0,
            'ort_frame_spans': ort_decoded['frames'],
            'native_frame_spans': native_decoded['frames'],
            'rec_input': input_diff, 'rec_logits': logits_diff,
            'rec_preprocessing': sample_metrics,
        })
    return {
        'status': 'NUMERIC_MATCH_SUPPLIED_EVIDENCE',
        'composite_verification': 'NOT_CHECKED',
        'facade_manifest_issued': False,
        'native_composite_oracle': 'UNATTESTED_CAPTURE_COMPARISON',
        'provenance_limit': (
            'All supplied tensor scalars and ground-truth associations passed; '
            'this read-only checker did not run or attest either ORT/native '
            'producer, process environment, or independent human annotation. '
            'Original root composite-plan.json remains DRAFT/NOT_CHECKED.'),
        'plan_sha256': plan_hash,
        'evidence_sha256': evidence_hash,
        'source_onnx': sources,
        'ground_truth_sha256': gt_hash,
        'source_image': image_info,
        'metrics': dict(metrics, aligned_recognitions=alignments),
    }


def audit_composite_evidence(bundle, evidence_dir):
    """Always fail closed; missing artifacts -> NOT_CHECKED, mismatches -> FAILED.

    Returns a JSON-ready report. No data is written, no root manifest is issued.
    A successful synthetic fixture checks the gate implementation, not a model.
    """
    try:
        return _evaluate(bundle, evidence_dir)
    except (MissingEvidence, FileNotFoundError) as why:
        status, reason = 'NOT_CHECKED', str(why)
    except (InvalidEvidence, composite.CompositeContractError,
            ValueError, TypeError, KeyError) as why:
        status, reason = 'FAILED', str(why)
    return {
        'status': status,
        'reason': reason,
        'composite_verification': 'NOT_CHECKED',
        'facade_manifest_issued': False,
        'native_composite_oracle': 'NOT_RUN' if status == 'NOT_CHECKED'
                                 else 'CAPTURE_COMPARISON_FAILED',
    }


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description='READ ONLY: compare original ORT tensors, native OCR '
                    'intermediates, Det boxes, CTC frame paths and '
                    'human ground truth; NEVER emits a verified manifest.')
    parser.add_argument('--bundle', required=True,
                        help='staged OCR DRAFT directory')
    parser.add_argument('--evidence', required=True,
                        help='capture directory with evidence.json and raw tensors')
    args = parser.parse_args(argv)
    result = audit_composite_evidence(args.bundle, args.evidence)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 1 if result['status'] == 'FAILED' else 2


if __name__ == '__main__':
    raise SystemExit(_main())
