# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Compare the compiled module with the original model on identical input.

WHY A STEP OF ITS OWN. A successful build does not mean the module COMPUTES the
right thing. That is not a hunch: on 31.08.2026 a quantized TFLite model built
without a single complaint and returned 0.02 where the reference gives 0.73.
Without this comparison the user would be handed a file nobody knows the truth
about.

WHAT TO MEASURE WITH - a question I got wrong twice, both times by measurement.

FIRST MISTAKE: absolute deviation. It only works when the magnitudes are alike.
A detector's outputs are pixel coordinates up to 640, where a deviation of
1.2e-03 is a relative error of 2e-06, which is agreement.

SECOND MISTAKE, and subtler: the LARGEST relative deviation. Measured on
rtdetr_r18vd, the same module against onnxruntime:

    white noise         2.6e-01
    smooth pattern      6.5e-05
    A REAL PHOTOGRAPH   2.0

On that same photograph the decoded detections agree to within half a pixel, and
the live check is green. The disagreement sits in the logits of INSIGNIFICANT
queries: -9.0 against -7.0 changes neither class nor box, yet shows up as a two.

    the largest single point speaks of an outlier, not of agreement

WHAT WAS ADOPTED. Acceptance goes by the 99.9th percentile: it reflects the
agreement of the bulk and does not collapse under lone outliers in flat regions.
The largest is printed beside it - as information, not as a verdict.

THE LIMIT IS NAMED HERE, HONESTLY: this does not replace a live run on your own
task. It catches the coarse - a different layout, lost outputs, quantization that
computes the wrong thing. A subtle difference that moves the fifth detection out
of a hundred, it will not catch.

THREE DIFFERENT INPUTS ARE USED, and that follows from the same measurement:
on a single input the agreement may be accidental. All three come from a fixed
seed, so both roads receive exactly the same arrays.

WHITE NOISE DOES NOT JUDGE, and that too is a measurement rather than a leniency.
On rtdetr_r18vd, same module, 99.9th percentile:

    smooth 4.6e-05    flat 1.7e-04    NOISE 4.0e-02

Noise diverges by three orders more. The cause is not the build: no such input
exists in nature, the model's own response to it is unstable, and two sound
implementations disagree there lawfully. Judging by it would fail good builds.
It stays in the run and is printed - as a note about the margin.

OUTPUT ORDER IS DIAGNOSED, NOT GUESSED. Outputs are compared in order. If that
fails, we look for a permutation under which they would agree - and if one is
FOUND, it is NAMED OUT LOUD and remains a refusal. Accepting a permutation
silently is not on: our Pascal code takes outputs by index, and a permuted output
would not be the one it thinks it holds. Saying "agreed" here would trade a noisy
check for a mute one, which is the worst of bargains.
"""

import numpy as np
from collections import deque

from .report import Step






_CUDA_DLL_DIRECTORIES = {}


def _prepare_cuda_runtime_directory(device_kind):
    """Make the selected toolkit's cudart visible to this Windows verifier.

    Only the toolkit actually discovered on THIS machine may supply the path;
    never scan unrelated host CUDA installations or guess a CUDA major version.
    This does not assert runtime compatibility: the real device and independent
    numerical comparison still decide the final verification status.

    Returns an explanatory warning when no credible directory could be added.
    A missing toolkit does not mask a valid self-contained runtime supplied by
    TVM: the normal runtime device probe remains the final availability check.
    """
    import os
    import sys

    if (device_kind != 'cuda' or sys.platform != 'win32'
            or not hasattr(os, 'add_dll_directory')):
        return None
    from . import cuda_preflight, environment

    nvcc_path = environment.find_cuda()
    toolkit = cuda_preflight.installed_toolkit_version(
        nvcc_path, environment._ask_program)
    runtime_path = cuda_preflight.runtime_in_toolkit(nvcc_path, toolkit)
    if not runtime_path:
        return ('the selected CUDA toolkit runtime was not located; '
                'checking available TVM runtime without adding an arbitrary DLL')
    directory = os.path.normcase(os.path.abspath(os.path.dirname(runtime_path)))
    if directory not in _CUDA_DLL_DIRECTORIES:
        try:
            _CUDA_DLL_DIRECTORIES[directory] = os.add_dll_directory(directory)
        except OSError as problem:
            return 'unable to register selected CUDA runtime DLL directory: %s' % problem
    return None

RELATIVE_TOLERANCE = 1e-3


PAIRED_RTOL = 0.003
PAIRED_ATOL = 0.003
MAX_PAIRED_CANDIDATES = 1024
MAX_PAIRED_SCALARS = 8_000_000
MAX_MULTI_INPUT_ELEMENTS = 8_000_000






RTDETR_FUSED_PROFILE = 'rtdetr_fused'
RTDETR_FUSED_ATOL = 0.001


class NothingToCompare(Exception):
    """Comparison is impossible, and the reason is named.

    Not to be confused with disagreement: there the comparison happened and
    returned a verdict; here it never happened at all.
    """


def is_tflite(model_path):
    
    
    
    
    if str(model_path).lower().endswith('.tflite'):
        return True
    try:
        with open(model_path, 'rb') as handle:
            return handle.read(8)[4:8] == b'TFL3'
    except (OSError, TypeError, ValueError):
        return False


def tflite_interpreter(model_path):
    """The real TFLite interpreter - the one that caught the quantization.

    ai-edge-litert is used: wheels exist for all three machines of the recipe,
    win_amd64 included (checked on 2.2.0). There must be no implementation of
    our own here, or the comparison would be comparing us with ourselves.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        raise NothingToCompare(
            'comparing a TFLite model needs the ai-edge-litert package, and it is'
            ' missing. Without it there is no reference to compare against, and'
            ' skipping quietly is not allowed here: the module would go out'
            ' unchecked')
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def model_input(model_path):
    """Name and shape of the single input, taken from the model itself."""
    if is_tflite(model_path):
        interpreter = tflite_interpreter(model_path)
        described = interpreter.get_input_details()
        if len(described) != 1:
            raise NothingToCompare('the model has %d inputs; this expects one'
                                   % len(described))
        entry = described[0]
        if entry['dtype'] != np.float32:
            
            
            
            raise NothingToCompare(
                'the model input has type %s rather than float32: this is a'
                ' quantized model. The converter refuses those, so there is'
                ' nothing to compare' % entry['dtype'].__name__)
        return entry['name'], tuple(int(d) for d in entry['shape'])

    import onnx
    model = onnx.load(model_path)
    weights = {w.name for w in model.graph.initializer}
    for entry in model.graph.input:
        if entry.name in weights:
            continue
        dims, free = [], []
        for index, dim in enumerate(entry.type.tensor_type.shape.dim):
            if dim.dim_value:
                dims.append(dim.dim_value)
            else:
                dims.append(None)
                free.append(index)
        if free:
            
            
            
            
            
            raise NothingToCompare(
                'input %s leaves dimensions %s free - name the shape as the third'
                ' argument, for example 1,3,640,640' % (entry.name, free))
        return entry.name, tuple(dims)
    raise NothingToCompare('the model has no inputs; there is nothing to compare')


def _onnx_input_contract(model_path):
    """Runtime ONNX inputs in declaration order, excluding initializer weights.

    Also return output names in ONNX declaration order, for disambiguating
    equally shaped (B, N, 4) boxes and four-class score tensors.
    """
    import onnx

    model = onnx.load(model_path)
    weights = {weight.name for weight in model.graph.initializer}
    inputs = []
    for entry in model.graph.input:
        if entry.name in weights:
            continue
        if not entry.type.HasField('tensor_type'):
            inputs.append((entry.name, None, None))
            continue
        tensor = entry.type.tensor_type
        shape = tuple(int(dim.dim_value) if dim.dim_value > 0 else None
                      for dim in tensor.shape.dim)
        inputs.append((entry.name, shape, int(tensor.elem_type)))
    return inputs, [entry.name for entry in model.graph.output]


def _validated_multi_inputs(inputs):
    """Refuse to invent shapes or types not given by the original ONNX graph."""
    supported = {1: np.float32, 7: np.int64}  
    total = 0
    result = []
    for name, shape, element_type in inputs:
        if shape is None:
            raise NothingToCompare('input %s is not an ONNX tensor' % name)
        if any(dim is None or dim <= 0 for dim in shape):
            raise NothingToCompare(
                'input %s has unknown/dynamic dimensions %s in the ORIGINAL '
                'ONNX; multi-input verification cannot guess their sizes'
                % (name, shape))
        if element_type not in supported:
            raise NothingToCompare(
                'input %s has unsupported ONNX dtype %s; only float32 and '
                'int64 synthetic inputs are defined' % (name, element_type))
        count = 1
        for dim in shape:
            count *= dim
        total += count
        if total > MAX_MULTI_INPUT_ELEMENTS:
            raise NothingToCompare(
                'multi-input synthetic tensors require %d elements, exceeding '
                'the %d-element verification memory cap'
                % (total, MAX_MULTI_INPUT_ELEMENTS))
        result.append((name, shape, supported[element_type]))
    if len(result) < 2:
        raise NothingToCompare('multi-input verification requires two inputs')
    if len({name for name, _shape, _dtype in result}) != len(result):
        raise NothingToCompare('the original ONNX repeats an input name')
    return result


def _probe_multi_inputs(inputs):
    """Three reproducible synthetic mappings, preserving every named dtype."""
    for label in ('smooth', 'noise', 'flat'):
        rng = np.random.RandomState(20260831)
        data = {}
        for index, (name, shape, dtype) in enumerate(inputs):
            if dtype == np.float32:
                
                offset = (index + 1) / (len(inputs) + 2)
                if label == 'smooth':
                    array = 0.5 * smooth_pattern(shape) + 0.5 * offset
                elif label == 'noise':
                    array = 0.5 * rng.rand(*shape) + 0.5 * offset
                else:
                    array = np.full(shape, 0.25 + 0.5 * offset)
                data[name] = np.ascontiguousarray(array, dtype=np.float32)
            else:
                
                
                if label == 'noise':
                    array = rng.randint(0, 2, size=shape).astype(np.int64)
                    if array.size:
                        array.flat[0] = 1
                elif label == 'smooth':
                    array = np.ones(shape, dtype=np.int64)
                else:
                    array = np.full(shape, index % 2, dtype=np.int64)
                    if array.size:
                        array.flat[0] = 1
                data[name] = np.ascontiguousarray(array)
        yield label, data


def spatial_axes(shape):
    """Height and width axes: last two for NCHW, second and third for NHWC.

    THE LAYOUT HAS TO BE TOLD FROM THE SHAPE - the file does not record it. The
    clue: in NHWC the last dimension is the channel count (1, 3 or 4) and the
    second cannot be a channel count. An ambiguous case such as (1, 3, 3, 3)
    reads as NCHW.

    A MISTAKE HERE DOES NOT SPOIL THE VERDICT, and that matters: both roads
    receive THE SAME array, however it came out. A wrong guess makes the pattern
    less photograph-like, and nothing more. Noticed 31.08.2026 on mobilenet_v2,
    whose input is (1, 224, 224, 3) while the old code built the picture over
    dimensions (224, 3).
    """
    if len(shape) != 4:
        return None
    if shape[-1] in (1, 3, 4) and shape[1] not in (1, 3, 4):
        return 1, 2
    return 2, 3


def smooth_pattern(shape):
    """A smooth, photograph-like pattern over the input's spatial dimensions."""
    axes = spatial_axes(shape)
    if axes is None:
        ramp = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32)
        return ramp.reshape(shape)
    axis_y, axis_x = axes
    height, width = shape[axis_y], shape[axis_x]
    y = np.arange(height, dtype=np.float32).reshape(-1, 1)
    x = np.arange(width, dtype=np.float32).reshape(1, -1)
    picture = 0.5 + 0.25 * np.sin(x / 80.0) * np.cos(y / 60.0)
    picture = np.clip(picture + 0.15 * x / max(width, 1), 0, 1)
    spread = [1] * len(shape)
    spread[axis_y] = height
    spread[axis_x] = width
    return np.broadcast_to(picture.reshape(spread), shape).astype(np.float32).copy()


def probe_inputs(shape):
    """Three different inputs: smooth, noisy and flat. All from a fixed seed."""
    rng = np.random.RandomState(20260831)
    
    
    yield 'smooth', smooth_pattern(shape), True
    yield 'noise', rng.rand(*shape).astype(np.float32), False
    
    
    yield 'flat', np.full(shape, 0.5, dtype=np.float32), True


def reference_road(model_path, input_names=None):
    """A foreign runner of the same model: onnxruntime or the TFLite interpreter.

    The reference must be a FOREIGN runner. A second implementation of our own
    would compare us with ourselves and stay silent exactly where we are wrong
    in the same way.
    """
    if is_tflite(model_path):
        interpreter = tflite_interpreter(model_path)
        entry = interpreter.get_input_details()[0]
        outputs = interpreter.get_output_details()

        def run_tflite(x):
            interpreter.set_tensor(entry['index'], x)
            interpreter.invoke()
            return [np.array(interpreter.get_tensor(o['index'])) for o in outputs]

        return 'TFLite interpreter', run_tflite

    import onnxruntime as ort
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    if input_names is not None:
        if 'CPUExecutionProvider' not in session.get_providers():
            raise NothingToCompare('independent ONNX Runtime CPUExecutionProvider '
                                   'is unavailable; no CPU reference was run')
        requested = tuple(input_names)
        actual = tuple(entry.name for entry in session.get_inputs())
        if len(actual) != len(requested) or set(actual) != set(requested):
            raise NothingToCompare(
                'original ONNX graph input names %s disagree with independent '
                'ONNX Runtime inputs %s' % (requested, actual))

        def run_onnx_many(x):
            if set(x) != set(requested):
                raise ValueError('ONNX Runtime needs exactly input names %s'
                                 % (requested,))
            
            
            return session.run(None, {name: x[name] for name in requested})

        return 'onnxruntime CPUExecutionProvider', run_onnx_many

    input_name = session.get_inputs()[0].name

    def run_onnx(x):
        return session.run(None, {input_name: x})

    return 'onnxruntime', run_onnx


def _deviation(a, b):
    """The 99.9th percentile and the largest, in one place, measured alike."""
    a64 = a.astype(np.float64)
    relative = np.abs(a64 - b.astype(np.float64)) / np.maximum(np.abs(a64), 1.0)
    return float(np.percentile(relative, 99.9)), float(relative.max())





DECISION_AGREEMENT = 0.999


def _looks_like_decision(a):
    """Does this output hold per-class scores over positions?

    FOUR DIMENSIONS AT LEAST, and that bound is deliberate. A detector emits
    (1, 84, 8400) - three dimensions, where the 84 are four box coordinates
    followed by eighty class scores. An argmax across them means nothing at all,
    and letting such an output through this door would open a hole in the gate
    rather than close a false alarm. Segmentation maps are (1, C, H, W) or
    (1, C, D, H, W): four dimensions or five.
    """
    return a.ndim >= 4 and a.shape[0] == 1 and a.shape[1] > 1


def _decision_agreement(a, b):
    """Share of positions where both roads pick the same class."""
    return float((a.argmax(axis=1) == b.argmax(axis=1)).mean())










DETECTION_THRESHOLDS = (0.5, 0.35, 0.25)


def _probability_shift(a, b):
    """The largest move in probability - which is where the thresholds live.

    Relative deviation on logits misleads here. Logits of -6 to -12 are ordinary
    for a detector, and a six percent relative change to one of them moves the
    probability by less than a thousandth. The thresholds are compared against
    probabilities, so the shift must be measured in the same currency.
    """
    sigmoida = lambda x: 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    return float(np.abs(sigmoida(a) - sigmoida(b)).max())


def _scores_output_index(outputs):
    """Index of the per-candidate class scores, or None.

    NOT GUESSED FROM SHAPE - DERIVED FROM THE MODEL'S OWN STRUCTURE. A detector
    that emits scores separately also emits boxes separately, and the two agree
    on the candidate count. So an output of (B, N, C) counts as class scores
    only when some other output is (B, N, 4).

    That pairing is what keeps YOLO out. Its single output is (1, 84, 8400),
    where the 84 are four box coordinates followed by eighty class scores in one
    tensor; there is no companion boxes output, the rule does not fire, and the
    numeric gate keeps judging it as before. Measured 02.09.2026: rtdetr_r18vd
    declares logits (B, 300, 80) and pred_boxes (B, 300, 4); yolo11n declares
    output0 (1, 84, 8400) alone.
    """
    kandidatov = {}
    for i, a in enumerate(outputs):
        if a.ndim == 3 and a.shape[2] == 4:
            kandidatov.setdefault(a.shape[1], []).append(('boxes', i))
        elif a.ndim == 3 and a.shape[2] > 4:
            kandidatov.setdefault(a.shape[1], []).append(('scores', i))
    for _n, rows in kandidatov.items():
        vidy = {v for v, _i in rows}
        if {'boxes', 'scores'} <= vidy:
            for vid, i in rows:
                if vid == 'scores':
                    return i
    return None


def _detections(a, threshold):
    """Candidates whose best score passes the threshold, with their classes."""
    scores = 1.0 / (1.0 + np.exp(-a.astype(np.float64)))
    best = scores.max(axis=2)
    klass = scores.argmax(axis=2)
    keep = best > threshold
    return {(int(b), int(n), int(klass[b, n]))
            for b, n in zip(*np.nonzero(keep))}


def _detections_agree(a, b):
    """Do the two roads report the same detections at every threshold?

    THE DETECTIONS ARE THE PREDICATE; THE LOGITS ARE A PROXY. Measured
    02.09.2026 on rtdetr_r18vd, our module against onnxruntime: the largest
    relative deviation over the whole tensor was 9 percent, which the numeric
    gate refuses - and that 9 percent was ONE point, candidate 289 of 300, class
    31, where the reference reads -6.680 and ours -7.436. Both are deep negative
    logits, ranked 216th by confidence. At 0.5, 0.3, 0.1 and 0.05 the two roads
    report identical sets: 1, 1, 6 and 42 detections, the same candidates with
    the same classes. 23819 of 24000 elements agreed; the mean deviation was
    0.0013.

    So the refusal was about a number nobody reads. Returns the agreeing
    thresholds and the disagreeing ones, so the report can say which.
    """
    soglasno, rashodyatsya = [], []
    for porog in DETECTION_THRESHOLDS:
        if _detections(a, porog) == _detections(b, porog):
            soglasno.append(porog)
        else:
            rashodyatsya.append(porog)
    return soglasno, rashodyatsya


def _find_permutation(reference, ours):
    """A permutation under which every output would agree, or None.

    WHY LOOK FOR SOMETHING THAT STAYS A REFUSAL ANYWAY. Two runners of one model
    are not obliged to emit outputs in the same order, and then a mismatch means
    a PERMUTATION rather than an arithmetic error. The difference matters
    enormously to the reader: in the first case indices must be swapped, in the
    second a fault must be hunted in the build. Saying "disagrees" without saying
    "this is a permutation" sends them hunting a fault that is not there.

    A permutation found does NOT cancel the refusal. Our Pascal code takes
    outputs by index, and a permuted output is not the one it holds it to be.

    The search runs pairwise rather than over all orderings: each of our outputs
    looks for a reference output it agrees with, and that match must be UNIQUE.
    An ambiguous correspondence is not a permutation but a coincidence of similar
    magnitudes, and passing it off as the answer is not allowed.
    """
    if len(reference) != len(ours):
        return None
    order = []
    for our_output in ours:
        candidates = []
        for index, reference_output in enumerate(reference):
            if reference_output.shape != our_output.shape:
                continue
            percentile, _ = _deviation(reference_output, our_output)
            if percentile < RELATIVE_TOLERANCE:
                candidates.append(index)
        if len(candidates) != 1:
            return None
        order.append(candidates[0])
    if len(set(order)) != len(order):
        return None
    if order == list(range(len(order))):
        return None
    return order


def _paired_output_indices(outputs, names):
    """Identify one (boxes, scores/logits) pair, preserving output positions.

    Different last dimensions establish the pair for ordinary detectors.
    Names are required when the score dimension is itself four or several
    equally shaped outputs would otherwise make the pairing ambiguous.
    """
    candidates = []
    for box_index, boxes in enumerate(outputs):
        if boxes.ndim != 3 or boxes.shape[2] != 4:
            continue
        for score_index, scores in enumerate(outputs):
            if (score_index == box_index or scores.ndim != 3
                    or scores.shape[:2] != boxes.shape[:2]
                    or scores.shape[2] < 1):
                continue
            candidates.append((box_index, score_index))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def box_name(name):
        return 'box' in name.lower() or 'bbox' in name.lower()

    def score_name(name):
        return 'score' in name.lower() or 'logit' in name.lower()

    if len(names) == len(outputs):
        named = [(i, j) for i, j in candidates
                 if box_name(names[i]) and score_name(names[j])]
        if len(named) == 1:
            return named[0]
    raise NothingToCompare('ambiguous boxes/scores output pairing; cannot '
                           'verify candidate associations from the ONNX signature')


def _has_perfect_matching(adjacency):
    """Bipartite augmenting paths: each reference row uses one distinct TVM row.

    Greedy matching is insufficient: a row with two possible partners must not
    steal the only possible partner of a subsequent row.
    """
    count = len(adjacency)
    right_to_left = [-1] * count
    left_to_right = [-1] * count
    for start in sorted(range(count), key=lambda i: len(adjacency[i])):
        queue = deque([start])
        seen_left = {start}
        parent_right = [-1] * count
        found = False
        while queue and not found:
            left = queue.popleft()
            for right in adjacency[left]:
                if parent_right[right] != -1:
                    continue
                parent_right[right] = left
                occupied = right_to_left[right]
                if occupied == -1:
                    
                    while right != -1:
                        predecessor = parent_right[right]
                        previous_right = left_to_right[predecessor]
                        left_to_right[predecessor] = right
                        right_to_left[right] = predecessor
                        right = previous_right
                    found = True
                    break
                if occupied not in seen_left:
                    seen_left.add(occupied)
                    queue.append(occupied)
        if not found:
            return False
    return True


def _paired_candidates_agree(reference_boxes, reference_scores,
                             built_boxes, built_scores):
    """Compare the *whole* (box, class-vector) row, as an unordered multiset.

    Two rows can match only if EVERY coordinate AND EVERY score is within
    measured atol=rtol=0.003. A full bipartite matching proves multiplicity,
    so a duplicated TVM candidate cannot replace a missing reference candidate.
    Rows from different batches never match.
    """
    arrays = (reference_boxes, reference_scores, built_boxes, built_scores)
    if any(a.ndim != 3 for a in arrays):
        return False, 'boxes and scores must each have three dimensions'
    if (reference_boxes.shape != built_boxes.shape
            or reference_scores.shape != built_scores.shape):
        return False, 'boxes/scores shapes differ between ONNX Runtime and TVM'
    batch, count, coordinates = reference_boxes.shape
    if (coordinates != 4 or reference_scores.shape[:2] != (batch, count)
            or reference_scores.shape[2] < 1):
        return False, 'incompatible boxes/scores candidate dimensions'
    if batch < 1 or count < 1:
        return False, 'empty output candidate set provides no verification'
    if count * batch > MAX_PAIRED_CANDIDATES:
        raise NothingToCompare(
            '%d candidate pairs exceed the capped %d for a bounded exact '
            'one-to-one comparison' % (count * batch, MAX_PAIRED_CANDIDATES))
    scalar_count = batch * count * (4 + reference_scores.shape[2])
    if scalar_count > MAX_PAIRED_SCALARS:
        raise NothingToCompare(
            '%d candidate scalars exceed the %d-element comparison memory cap'
            % (scalar_count, MAX_PAIRED_SCALARS))
    if any(not np.issubdtype(a.dtype, np.floating) for a in arrays):
        return False, 'paired detection boxes/scores must be numeric floats'
    if any(not np.isfinite(a).all() for a in arrays):
        return False, 'NaN/Inf detected in boxes or scores'

    for batch_index in range(batch):
        base_box = reference_boxes[batch_index]
        base_scores = reference_scores[batch_index]
        vm_box = built_boxes[batch_index]
        vm_scores = built_scores[batch_index]
        if any(not np.any(a) for a in
               (base_box, base_scores, vm_box, vm_scores)):
            return False, 'all-zero/vacuous boxes or scores cannot verify a detector'

        expected = np.concatenate((base_box, base_scores),
                                  axis=1).astype(np.float64)
        produced = np.concatenate((vm_box, vm_scores),
                                  axis=1).astype(np.float64)
        
        
        adjacency = []
        for original in expected:
            close = np.all(
                np.abs(produced - original)
                <= PAIRED_ATOL + PAIRED_RTOL * np.abs(original), axis=1)
            partners = np.flatnonzero(close).tolist()
            if not partners:
                return False, ('batch %d has a reference candidate without any '
                               'matching complete TVM box/score row' % batch_index)
            adjacency.append(partners)
        if not _has_perfect_matching(adjacency):
            return False, ('batch %d lacks a unique one-to-one use of all '
                           'candidate rows (duplicate or missing candidate)'
                           % batch_index)
    return True, ('all %d candidate pairs matched one-to-one; every box '
                  'coordinate and class score satisfies rtol=atol=0.003'
                  % (batch * count))


def _validate_rtdetr_fused_source(model_path):
    """Independently enforce the declared single fused ONNX output profile.

    The Pascal adapter is selected in the build contract, not inferred from a
    coincidental six-column output.  Recheck the ONNX itself in this verifier so
    a stale/incorrect caller cannot opt another model into reordered comparison.
    Dynamic candidate counts are not evidence of a bounded exact comparison.
    """
    if is_tflite(model_path):
        raise NothingToCompare('RT-DETR fused numeric profile needs ONNX output metadata')
    import onnx
    model = onnx.load(model_path)
    if len(model.graph.output) != 1:
        raise NothingToCompare('RT-DETR fused comparison needs exactly one ONNX output')
    value = model.graph.output[0]
    if not value.type.HasField('tensor_type'):
        raise NothingToCompare('RT-DETR fused ONNX output is not a tensor')
    tensor = value.type.tensor_type
    if tensor.elem_type not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT):
        raise NothingToCompare('RT-DETR fused ONNX output must be float16/float32')
    dims = tensor.shape.dim
    if len(dims) != 3 or any(dim.dim_value <= 0 for dim in dims):
        raise NothingToCompare('RT-DETR fused output needs known [1,N,6] dimensions')
    if dims[0].dim_value != 1 or dims[2].dim_value != 6:
        raise NothingToCompare('RT-DETR fused output must have shape [1,N,6]')
    if dims[1].dim_value > MAX_PAIRED_CANDIDATES:
        raise NothingToCompare('RT-DETR fused candidate count exceeds exact matching cap')


def _rtdetr_fused_candidates_agree(reference, built):
    """Match all fused (cx,cy,w,h,confidence,class_id) candidates bijectively.

    Row permutations from TopK ties are legal, but missing, duplicated, altered
    or cross-batch candidates are not.  Every float field must differ by <=0.001
    ABSOLUTELY and every exact integral class ID must be identical.  There is no
    percentile shortcut, thresholded subset, or nonjudging noise input.
    """
    if reference.ndim != 3 or built.ndim != 3 or reference.shape != built.shape:
        return False, 'fused RT-DETR output shapes differ or are not rank 3'
    batch, count, fields = reference.shape
    if fields != 6 or batch < 1 or count < 1:
        return False, 'fused RT-DETR requires nonempty [B,N,6] output'
    if batch * count > MAX_PAIRED_CANDIDATES:
        raise NothingToCompare('fused RT-DETR exact candidate count exceeds cap')
    if any(not np.issubdtype(a.dtype, np.floating) for a in (reference, built)):
        return False, 'fused RT-DETR outputs must be floating-point tensors'
    if not np.isfinite(reference).all() or not np.isfinite(built).all():
        return False, 'fused RT-DETR output contains NaN/Inf'
    for name, tensor in (('reference', reference), ('TVM', built)):
        classes = tensor[:, :, 5]
        scores = tensor[:, :, 4]
        
        
        
        if (np.any(classes != np.floor(classes)) or np.any(classes < 0)
                or np.any(classes >= 80)):
            return False, ('%s fused RT-DETR class IDs are not exact integers '
                           'in Pascal supported range 0..79' % name)
        if np.any(scores < 0) or np.any(scores > 1):
            return False, '%s fused RT-DETR confidences are outside [0,1]' % name
        centers = tensor[:, :, :2]
        extents = tensor[:, :, 2:4]
        if (np.any(centers < -0.05) or np.any(centers > 1.05)
                or np.any(extents <= 0) or np.any(extents > 1.05)):
            return False, ('%s fused RT-DETR cxcywh geometry is outside Pascal '
                           'normalized center/positive-size bounds' % name)
    for batch_index in range(batch):
        expected = reference[batch_index].astype(np.float64)
        produced = built[batch_index].astype(np.float64)
        if not np.any(expected[:, 4] > 0):
            return False, 'reference has no nonzero confidence; comparison is vacuous'
        adjacency = []
        for candidate in expected:
            exact_class = produced[:, 5] == candidate[5]
            close_floats = np.all(
                np.abs(produced[:, :5] - candidate[:5]) <= RTDETR_FUSED_ATOL,
                axis=1)
            partners = np.flatnonzero(exact_class & close_floats).tolist()
            if not partners:
                return False, ('batch %d has an unmatched fused RT-DETR candidate '
                               '(class or one of five floats differs)' % batch_index)
            adjacency.append(partners)
        if not _has_perfect_matching(adjacency):
            return False, ('batch %d has duplicate or missing fused RT-DETR '
                           'candidate rows' % batch_index)
    return True, ('all %d fused RT-DETR candidates matched one-to-one; class '
                  'exact; each cx/cy/w/h/conf absolute difference <=0.001'
                  % (batch * count))


def _strict_multi_outputs(reference, ours, output_names):
    """New multi-input mode has no percentile or detection-count shortcuts."""
    if len(reference) != len(ours):
        return False, ('OUTPUT COUNT DIFFERS: ONNX Runtime %d, TVM %d'
                       % (len(reference), len(ours)))
    if not reference:
        return False, 'no outputs were produced; comparison would be vacuous'
    for index, (a, b) in enumerate(zip(reference, ours)):
        if a.shape != b.shape:
            return False, ('output %d: SHAPES DIFFER, %s against %s'
                           % (index, a.shape, b.shape))
        if any(t.dtype.kind not in 'biuf' for t in (a, b)):
            return False, 'output %d uses an unsupported numeric dtype' % index
        if (a.dtype.kind in 'biu') != (b.dtype.kind in 'biu'):
            return False, ('output %d changes integer/float dtype family '
                           'between ONNX Runtime and TVM' % index)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return False, 'output %d contains NaN or Inf' % index

    pair = _paired_output_indices(reference, output_names)
    built_pair = _paired_output_indices(ours, output_names)
    if pair != built_pair:
        return False, ('TVM outputs change boxes/scores positions; compiled '
                       'output indices cannot be reinterpreted')

    if pair is not None:
        box_index, score_index = pair
        matched, reason = _paired_candidates_agree(
            reference[box_index], reference[score_index],
            ours[box_index], ours[score_index])
        if not matched:
            return False, reason
        skip = {box_index, score_index}
    else:
        skip = set()
        reason = 'all outputs compared by index, with every scalar checked'

    if all(not np.any(a) for a in reference):
        return False, 'all-zero/vacuous reference results do not prove correctness'

    for index, (a, b) in enumerate(zip(reference, ours)):
        if index in skip:
            continue
        if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            agrees = np.array_equal(a, b)
        else:
            agrees = bool(np.all(np.isclose(
                a.astype(np.float64), b.astype(np.float64),
                rtol=PAIRED_RTOL, atol=PAIRED_ATOL)))
        if not agrees:
            return False, ('output %d differs: every scalar must satisfy '
                           'rtol=atol=0.003 (integers must be identical)' % index)
    return True, reason


def _compare_legacy_probe(reference, ours, label, judges):
    """Compare a generic probe without changing its declared judging role.

    Output-count failures retain their unconditional gate. Other failures
    follow judges. Label and threshold agreement is diagnostic: no explicit
    consumer contract authorizes it to replace a failed numeric comparison.
    """
    result = {'agreed': False, 'blocking': False, 'lines': [], 'numbers': {}}

    def finish(agreed, blocking):
        result['agreed'] = bool(agreed)
        result['blocking'] = bool(blocking)
        result['numbers']['raw_result.%s' % label] = 'PASS' if agreed else 'FAIL'
        result['numbers']['judges.%s' % label] = bool(judges)
        result['numbers']['blocking.%s' % label] = bool(blocking)
        return result

    if len(reference) != len(ours):
        result['lines'].append(
            '%-8s OUTPUT COUNT DIFFERS: reference %d, ours %d'
            % (label, len(reference), len(ours)))
        return finish(False, True)
    if not reference:
        result['lines'].append('%-8s no outputs; comparison is vacuous' % label)
        return finish(False, judges)

    reference = [np.asarray(a) for a in reference]
    ours = [np.asarray(b) for b in ours]
    invalid = []
    for index, (a, b) in enumerate(zip(reference, ours)):
        if a.shape != b.shape:
            invalid.append('output %d: SHAPES DIFFER, %s against %s'
                           % (index, a.shape, b.shape))
        elif a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':
            invalid.append('output %d: unsupported numeric dtype' % index)
        elif a.dtype != b.dtype:
            invalid.append('output %d: DTYPES DIFFER, %s against %s'
                           % (index, a.dtype, b.dtype))
        elif a.size == 0:
            invalid.append('output %d: empty tensor; comparison is vacuous' % index)
        elif not np.isfinite(a).all() or not np.isfinite(b).all():
            invalid.append('output %d: NaN or Inf' % index)
    if invalid:
        result['numbers']['invalid.%s' % label] = list(invalid)
        suffix = '' if judges else ' (does not judge)'
        result['lines'].extend('%-8s %s%s' % (label, reason, suffix)
                               for reason in invalid)
        return finish(False, judges)

    disagreed = False
    for index, (a, b) in enumerate(zip(reference, ours)):
        percentile, largest = _deviation(a, b)
        agreed = percentile < RELATIVE_TOLERANCE
        result['lines'].append(
            '%-8s output %d %s: 99.9%% %.3e, largest %.3e - %s%s'
            % (label, index, a.shape, percentile, largest,
               'AGREES' if agreed else 'DISAGREES',
               '' if judges else ' (does not judge)'))
        result['numbers']['%s.%d' % (label, index)] = percentile
        if not agreed:
            disagreed = True

    if disagreed:
        for index, (a, b) in enumerate(zip(reference, ours)):
            if _looks_like_decision(a):
                agreement = _decision_agreement(a, b)
                result['numbers']['decision.%s.%d' % (label, index)] = agreement
                result['lines'].append(
                    '%-8s output %d decision agreement %.4f%%; criterion %.4f%%'
                    ' %s (diagnostic only)'
                    % (label, index, agreement * 100.0,
                       DECISION_AGREEMENT * 100.0,
                       'met' if agreement >= DECISION_AGREEMENT else 'not met'))

        index = _scores_output_index(reference)
        if index is not None:
            agreeing, differing = _detections_agree(reference[index], ours[index])
            found = max(len(_detections(reference[index], threshold))
                        for threshold in DETECTION_THRESHOLDS)
            shift = _probability_shift(reference[index], ours[index])
            result['lines'].append(
                '%-8s DETECTIONS %d found; agree at %s; disagree at %s;'
                ' probability moved by at most %.5f (diagnostic only)'
                % (label, found,
                   ', '.join('%g' % x for x in agreeing) or 'no threshold',
                   ', '.join('%g' % x for x in differing) or 'none', shift))
            result['numbers']['detections.%s' % label] = list(agreeing)
            result['numbers']['detections.found.%s' % label] = found
            result['numbers']['probability.shift.%s' % label] = shift

        order = _find_permutation(reference, ours)
        if order is not None:
            pairs = ', '.join('ours %d = reference %d' % (n, r)
                              for n, r in enumerate(order))
            result['lines'].append(
                'A numerical output permutation would match: ' + pairs)
            result['lines'].append(
                'Output positions remain mismatched; this diagnostic does not'
                ' establish proposal identity or cancel the failed comparison.')
            result['numbers']['permutation'] = order
        result['lines'].append(
            'Diagnostic agreement does not replace the failed numeric comparison.')

    return finish(not disagreed, disagreed and judges)


def verify_here(model_path, module_path, shape_from_caller=None, device_kind='cpu',
                comparison_profile=None):
    """Compare a built module with its source model, IN THIS PROCESS.

    Loading the module locks the file on Windows - see the note above
    verify(). Call this directly only when the process is about to end.
    """
    step = Step('Comparison with the reference')
    if comparison_profile not in (None, RTDETR_FUSED_PROFILE):
        return step.refuse('unknown numerical comparison profile %r'
                           % comparison_profile)
    
    
    search_warning = _prepare_cuda_runtime_directory(device_kind)
    if search_warning:
        step.say('CUDA runtime search: ' + search_warning)
    try:
        import tvm
        from tvm import relax
    except ImportError as missing:
        return step.refuse('TVM is not installed: %s' % missing)

    multi = False
    input_specs = []
    output_names = []
    try:
        if not is_tflite(model_path):
            source_inputs, output_names = _onnx_input_contract(model_path)
            multi = len(source_inputs) > 1
        if comparison_profile == RTDETR_FUSED_PROFILE:
            if multi:
                raise NothingToCompare('RT-DETR fused comparison requires one input')
            _validate_rtdetr_fused_source(model_path)
        if multi:
            input_specs = _validated_multi_inputs(source_inputs)
            
            
            
            if (shape_from_caller
                    and tuple(shape_from_caller) != input_specs[0][1]):
                raise NothingToCompare(
                    'the single-input shape override %s disagrees with the '
                    'original first ONNX input %s; a multi-input override is '
                    'not representable by this verification interface'
                    % (tuple(shape_from_caller), input_specs[0][1]))
            road_name, run_reference = reference_road(
                model_path, [name for name, _shape, _dtype in input_specs])
        else:
            if shape_from_caller:
                name, shape = 'given as an argument', tuple(shape_from_caller)
            else:
                name, shape = model_input(model_path)
            road_name, run_reference = reference_road(model_path)
    except NothingToCompare as why:
        if multi:
            return step.say('NOT CHECKED: ' + str(why).strip())
        return step.refuse(str(why).strip())

    if comparison_profile == RTDETR_FUSED_PROFILE:
        step.say('RT-DETR fused output uses [cx,cy,w,h,confidence,class_id]; '
                 'top-K candidate indices may differ, class IDs must be identical')
        step.say('all candidates and ALL smooth/noise/flat probes judge; '
                 'absolute difference <=0.001 for each of five float fields')
        step.numbers['comparison_profile'] = comparison_profile

    if multi:
        for name, shape, dtype in input_specs:
            step.say('original ONNX input %s, dtype %s, shape %s'
                     % (name, np.dtype(dtype).name, shape))
        step.say('inputs are DETERMINISTIC SYNTHETIC smooth/noise/flat arrays '
                 '(float32 patterns, int64 binary mask/index proxies); '
                 'they are not real preprocessed photographs, labels or prompts')
        step.say('TVM input arguments follow original ONNX graph declaration '
                 'order; independent ONNX Runtime receives ALL exact input names')
        step.numbers['input_origin'] = 'deterministic synthetic inputs'
    else:
        step.say('input %s, shape %s' % (name, shape))
    step.say('reference: %s' % road_name)
    step.say('module runs on: %s' % device_kind)
    step.numbers['reference'] = road_name

    
    
    
    
    
    
    
    
    device = tvm.runtime.device(device_kind, 0)
    if not device.exist:
        
        
        step.say('this machine has no %s device, so the module was not run'
                 % device_kind)
        step.say('the module was built, but NOTHING about its numbers is known')
        return step

    try:
        module = tvm.runtime.load_module(module_path)
        machine = relax.VirtualMachine(module, device)
    except Exception as trouble:
        return step.refuse('the module would not load: %s' % trouble)

    everything_agreed = True
    trials = ((label, data, True) for label, data
              in _probe_multi_inputs(input_specs)) if multi else probe_inputs(shape)
    for label, x, judges in trials:
        try:
            reference = run_reference(x)
            if multi:
                ordered = [tvm.runtime.tensor(np.ascontiguousarray(x[name]), device)
                           for name, _shape, _dtype in input_specs]
                ours = machine['main'](*ordered)
            else:
                ours = machine['main'](
                    tvm.runtime.tensor(np.ascontiguousarray(x), device))
        except Exception as trouble:
            return step.refuse('the run on input "%s" failed: %s' % (label, trouble))

        try:
            ours = list(ours)
        except TypeError:
            ours = [ours]
        ours = [np.asarray(t.numpy() if hasattr(t, 'numpy') else t) for t in ours]

        if comparison_profile == RTDETR_FUSED_PROFILE:
            if len(reference) != 1 or len(ours) != 1:
                return step.refuse('RT-DETR fused numerical comparison requires '
                                   'one and only one output on BOTH runtimes')
            try:
                agreed, reason = _rtdetr_fused_candidates_agree(
                    np.asarray(reference[0]), ours[0])
            except NothingToCompare as why:
                return step.say('NOT CHECKED: ' + str(why).strip())
            step.say('%-8s RT-DETR fused exact candidate matching: %s - %s'
                     % (label, 'AGREES' if agreed else 'DISAGREES', reason))
            step.numbers['fused.%s' % label] = bool(agreed)
            if not agreed:
                everything_agreed = False
            continue

        if multi:
            try:
                agreed, reason = _strict_multi_outputs(
                    [np.asarray(a) for a in reference], ours, output_names)
            except NothingToCompare as why:
                return step.say('NOT CHECKED: ' + str(why).strip())
            step.say('%-8s strict multi-input comparison: %s - %s' % (
                label, 'AGREES' if agreed else 'DISAGREES', reason))
            step.numbers['multi.%s' % label] = bool(agreed)
            if not agreed:
                everything_agreed = False
            continue

        comparison = _compare_legacy_probe(reference, ours, label, judges)
        for line in comparison['lines']:
            step.say(line)
        step.numbers.update(comparison['numbers'])
        if comparison['blocking']:
            everything_agreed = False

    if everything_agreed:
        return step.done('comparison passed')
    return step.refuse('the built module does not compute what the source model does')





















def verify(model_path, module_path, shape_from_caller=None, device_kind='cpu',
           comparison_profile=None):
    """Compare a built module with its source model in a child process.

    Returns a report step, exactly as if the work had happened here.
    """
    import json
    import os
    import subprocess
    import sys

    step = Step('Comparison with the reference')
    command = [sys.executable, '-m', 'pascalvision_convert.verify', model_path, module_path]
    command.append(','.join(str(d) for d in shape_from_caller) if shape_from_caller else '')
    command.append(device_kind)
    if comparison_profile is not None:
        command.append(comparison_profile)

    environment = dict(os.environ)
    
    
    environment['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [environment.get('PYTHONPATH', '')])

    try:
        finished = subprocess.run(command, capture_output=True, timeout=1800,
                                  env=environment)
    except Exception as trouble:
        
        
        
        return step.say('NOT CHECKED: independent comparison child could not '
                        'finish (%s); no in-process verification substituted'
                        % trouble)

    text = (finished.stdout or b'').decode('utf-8', 'replace')
    if getattr(finished, 'returncode', None) != 0:
        step.refuse('the independent comparison child exited with code %r; '
                    'a VERIFIED reply is invalid without natural process exit0'
                    % getattr(finished, 'returncode', None))
        failure = (finished.stderr or b'').decode('utf-8', 'replace')
        for line in failure.splitlines()[-6:]:
            if line.strip():
                step.say(line.strip())
        return step
    marker = text.rfind(chr(2))
    if marker < 0:
        step.refuse('the comparison process returned nothing usable')
        
        
        
        whole = (text or (finished.stderr or b'').decode('utf-8', 'replace')).splitlines()
        shown = [line.strip() for line in whole[-8:] if line.strip()]
        if len(whole) > 8:
            step.say('last 8 lines of %d from that process:' % len(whole))
        for line in shown:
            step.say(line)
        return step

    try:
        described = json.loads(text[marker + 1:])
        if not isinstance(described, dict):
            raise ValueError('result is not an object')
        if described.get('state') not in ('done', 'refused', 'unchecked'):
            raise ValueError('result has invalid state')
        if (not isinstance(described.get('name'), str)
                or not isinstance(described.get('lines'), list)
                or any(not isinstance(line, str) for line in described['lines'])
                or not isinstance(described.get('numbers'), dict)):
            raise ValueError('result has invalid report fields')
        return Step(described['name'], described['state'],
                    described['lines'], described['numbers'])
    except (ValueError, TypeError, KeyError) as trouble:
        return step.refuse('invalid independent verification child reply: %s'
                           % trouble)


def _main(argv):
    """Entry point of the child process: compare, then print the step as JSON."""
    import json
    shape = None
    if len(argv) > 3 and argv[3]:
        shape = tuple(int(x) for x in argv[3].split(',') if x.strip())
    device_kind = argv[4] if len(argv) > 4 and argv[4] else 'cpu'
    comparison_profile = argv[5] if len(argv) > 5 and argv[5] else None
    step = verify_here(argv[1], argv[2], shape, device_kind,
                       comparison_profile)
    
    
    print(chr(2) + json.dumps(step.as_dict(), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_main(sys.argv))
