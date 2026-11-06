# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Build the model contract that :mod:`manifest` emits.

The manifest emitter (:func:`pascalvision_convert.manifest.emit_manifest_text`)
turns a "model contract" into the flat ``manifest.txt`` the Pascal facade reads.
This module builds that contract from the handful of facts a build settles once it
finishes, with one builder per model family the zoo holds:
:func:`build_vision_contract` for a converted VISION model - the task it serves,
the adapter that pre/post-processes it, the artifact it produced and the single
input tensor it takes - and :func:`build_text_contract` for a converted LLM, whose
contract carries text and tokenizer sections in place of that tensor.

WHY A BUILDER RATHER THAN A HAND-WRITTEN DICT. Some parts of a vision contract
are the same for every model, and retyping them per call is where a typo goes
unnoticed: a vision artifact's logical format is ``so`` and its entry point is
``main``; its input tensor is ``float32``. Baking those in here means a caller
supplies only what actually varies, and the fixed parts cannot drift.

WHICH TASK CARRIES WHICH SECTION - read off the shipped bundles, not invented:

    object_detection    a ``default`` line, ONLY when thresholds are given -
                        yolo11 carries scoreThreshold/nmsThreshold, rtdetr none;
    classification      a ``vision`` line, ONLY when mean/std are given - an
                        ImageNet net carries them, a YOLO head does not;
    pose                a ``pose`` line with the keypoint count;
    the rest            instance / semantic / foreground segmentation and
                        oriented detection carry no task-specific line, exactly
                        as their bundles carry none.

That the line's presence is a property of the model, not of the task, is why the
threshold and normalisation facts are optional rather than tied to the kind.

Only the standard library is used; the contract is plain dict / list / str / int
/ float, so it maps straight onto JSON and back through the emitter's companion.
"""


VISION_ARTIFACT_FORMAT = "so"
VISION_ARTIFACT_ENTRY = "main"


IO_DTYPE = "float32"


FORMAT_VERSION = 1

_OBJECT_DETECTION = "object_detection"
_CLASSIFICATION = "classification"
_POSE = "pose"


VISION_TASK_KINDS = frozenset((
    _OBJECT_DETECTION,
    _CLASSIFICATION,


    "text_detection",
    "text_recognition",
    "instance_segmentation",
    "semantic_segmentation",
    "foreground_segmentation",
    _POSE,
    "oriented_detection",
))

_PADDLE_OCR_ROLES = {
    'text_detection': 'paddleocr_det',
    'text_recognition': 'paddleocr_rec',
}


_PASCAL_FACADE_ADAPTERS = {
    'object_detection': frozenset(('yolo11', 'rtdetr')),
    'classification': frozenset(('yolo11',)),
    'instance_segmentation': frozenset(('yolo11',)),
    'semantic_segmentation': frozenset(('tensorflow',)),
    'foreground_segmentation': frozenset(('birefnet',)),
    'oriented_detection': frozenset(('yolo11',)),
    'pose': frozenset(('yolo11',)),
}


def validate_vision_adapter(task_kind, adapter):
    """Refuse fictional Pascal facade promises before expensive compilation.

    PaddleOCR's two independently verifiable raw stages are a distinct scope:
    their individual passports are NOT an end-to-end native composite facade.
    Returning the scope makes that limitation explicit to the pipeline/report.
    """
    task = str(task_kind).strip().lower()
    requested = str(adapter).strip().lower()
    if task in _PADDLE_OCR_ROLES:
        allowed = _PADDLE_OCR_ROLES[task]
        if requested != allowed:
            raise ValueError('%s requires adapter=%s for individual PaddleOCR '
                             'raw-stage contract' % (task, allowed))
        return 'INDIVIDUAL_RAW_STAGE_ONLY'
    allowed = _PASCAL_FACADE_ADAPTERS.get(task)
    if allowed is None:
        raise ValueError('task %r has no supported Pascal facade contract; '
                         'omit task/adapter to compile raw tensors' % task_kind)
    if requested not in allowed:
        raise ValueError('adapter %r has no supported Pascal facade for task %r '
                         '(currently: %s); omit task/adapter to compile raw '
                         'tensors without a verified Pascal manifest'
                         % (adapter, task_kind, ', '.join(sorted(allowed))))
    return 'PASCAL_FACADE'


def requires_paddleocr_output_check(task_kind, adapter):
    """Two separately verifiable PaddleOCR stages, not the composite facade."""
    return str(task_kind).lower() in _PADDLE_OCR_ROLES


def validate_paddleocr_outputs(task_kind, adapter, outputs, *, input_shape=None):
    """Fail-closed structural gate for native PaddleOCR output tensors.

    Det outputs a single [N,1,H,W] map; Rec a [N,T,C] CTC tensor. Symbolic
    ONNX dimensions are allowed only where the real independent numerical
    verifier will later check runtime shapes. This gate alone is NOT a native
    Det->Rec->CTC composite accuracy check.
    """
    role = str(task_kind).lower()
    if role not in _PADDLE_OCR_ROLES:
        return None
    required = _PADDLE_OCR_ROLES[role]
    if str(adapter).lower() != required:
        raise ValueError('%s requires adapter=%s; %r is not a validated '
                         'PaddleOCR native stage' % (role, required, adapter))
    if (not isinstance(input_shape, (tuple, list)) or len(input_shape) != 4
            or any(type(d) is not int or d <= 0 for d in input_shape)
            or tuple(input_shape[:2]) != (1, 3)):
        raise ValueError('PaddleOCR needs explicit [1,3,H,W] positive '
                         'single-image float32 input shape')
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
        raise ValueError('PaddleOCR %s needs exactly one output tensor' % role)
    item = outputs[0]
    shape = item.get('shape')
    if not isinstance(item.get('name'), str) or not item['name']:
        raise ValueError('PaddleOCR output needs a real tensor name')
    if item.get('dtype') != 'float32':
        raise ValueError('PaddleOCR %s needs float32 output, got %r'
                         % (role, item.get('dtype')))
    rank = 4 if role == 'text_detection' else 3
    if not isinstance(shape, (tuple, list)) or len(shape) != rank:
        raise ValueError('PaddleOCR %s output needs rank %d, got %r'
                         % (role, rank, shape))
    for dim in shape:
        if (dim is None or type(dim) is bool
                or (type(dim) is int and dim <= 0)
                or (type(dim) is str and not dim)
                or not isinstance(dim, (int, str))):
            raise ValueError('PaddleOCR output has invalid dimension %r' % dim)
    if type(shape[0]) is int and shape[0] != 1:
        raise ValueError('PaddleOCR output batch differs from selected image')
    if role == 'text_detection':
        if type(shape[1]) is not int or shape[1] != 1:
            raise ValueError('PaddleOCR detection map must have one channel')
        if ((type(shape[2]) is int and shape[2] != input_shape[2])
                or (type(shape[3]) is int and shape[3] != input_shape[3])):
            raise ValueError('PaddleOCR detection map must match input H,W')
        return 'paddleocr_detection_map'
    if type(shape[2]) is not int or not (2 <= shape[2] <= 65536):
        raise ValueError('PaddleOCR CTC class count must be statically known '
                         'in 2..65536, got %r' % shape[2])
    return 'paddleocr_ctc_logits'


def requires_adapter_output_check(task_kind, adapter):
    """The Pascal detector whose outputs have a specific, mandatory signature.

    CrossVision.ML.Facade.BuildDetector selects TRtDetrDetector for this pair.
    TVM.RtDetr.Detect accepts either two legacy logits [B,Q,C] and fractional
    boxes [B,Q,4], or one fused [1,N,6] output decoded as normalised
    [center_x,center_y,width,height,confidence,class_id]. The fused decoder is
    native-tested on the real RT-DETR-L graph; raw ONNX numeric verification
    still has to pass independently before a verified manifest may be issued.
    """
    return (str(task_kind).lower() == _OBJECT_DETECTION
            and str(adapter).lower() == "rtdetr")


def _matching_output_dimension(left, right):
    """Prove two dimensions match, without treating two unknowns as equal."""
    if isinstance(left, int) and not isinstance(left, bool):
        return left > 0 and left == right
    return isinstance(left, str) and bool(left) and left == right


def validate_adapter_outputs(task_kind, adapter, outputs, *, input_batch=None):
    """Check the source output signature before promising a Pascal RT-DETR bundle.

    *outputs* is a sequence of dicts with name, shape and dtype, read from the
    actual ONNX/TFLite graph. A dimension is a positive int, an ONNX symbolic
    dimension name, or None when it cannot be determined. Unrelated adapters
    remain unrestricted: their own Pascal decoders have different contracts.

    Returns 'rtdetr_fused' or 'rtdetr_legacy' for a valid RT-DETR graph; other
    adapters return None. The caller uses the fused mode to select the stricter
    semantic, permutation-aware independent numerical oracle, not to claim
    verification solely from a compatible shape.

    In the legacy case, ONNX symbolic batch/query dimensions are accepted only
    when both outputs name the SAME symbol. The fused Pascal decoder requires
    batch=1, a positive finite query count, six scalar float values per row and
    [cx,cy,w,h,confidence,class_id] semantics. Source graph shape cannot itself
    establish those value semantics; the independent verifier must do that.
    """
    if not requires_adapter_output_check(task_kind, adapter):
        return

    expected = ("Pascal object_detection/rtdetr requires either one fused "
                "rank-3 float output [1,N,6] (cxcywh, confidence, class_id) "
                "or two separate rank-3 float outputs: "
                "logits [1,Q,C] and boxes [1,Q,4]")
    if len(outputs) not in (1, 2):
        raise ValueError(
            "%s; source graph declares %d output(s). Omit the task/adapter "
            "contract to compile unsupported tensors without a Pascal manifest."
            % (expected, len(outputs)))

    for output in outputs:
        name = output.get("name", "(unnamed)")
        shape = output.get("shape")
        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError(
                "%s; output %r has shape %r rather than rank 3"
                % (expected, name, shape))
        if output.get("dtype") not in ("float16", "float32"):
            raise ValueError(
                "%s; output %r has dtype %r, which the Pascal decoder cannot "
                "read as logits/boxes"
                % (expected, name, output.get("dtype")))
        for dimension in shape:
            if (isinstance(dimension, bool) or
                    (isinstance(dimension, int) and dimension < 1) or
                    (isinstance(dimension, str) and not dimension) or
                    not isinstance(dimension, (int, str, type(None)))):
                raise ValueError("%s; output %r has invalid dimension %r"
                                 % (expected, name, dimension))

    if len(outputs) == 1:


        shape = outputs[0]["shape"]
        if type(shape[0]) is not int or shape[0] != 1:
            raise ValueError("%s; fused batch must be statically 1, got %r"
                             % (expected, shape[0]))
        if input_batch is not None and input_batch != 1:
            raise ValueError("%s; selected input batch must be 1, got %r"
                             % (expected, input_batch))
        if (type(shape[1]) is not int or shape[1] < 1 or
                shape[1] > ((2 ** 31 - 1) // 6)):
            raise ValueError(
                "%s; fused candidate count N must be a known positive "
                "Pascal-compatible dimension, got %r"
                % (expected, shape[1]))
        if type(shape[2]) is not int or shape[2] != 6:
            raise ValueError("%s; fused final dimension must be 6, got %r"
                             % (expected, shape[2]))
        return "rtdetr_fused"

    boxes_indices = [index for index, output in enumerate(outputs)
                     if output["shape"][2] == 4]
    if len(boxes_indices) != 1:
        raise ValueError(
            "%s; exactly one output must have a known last dimension of 4 "
            "(boxes), and the other must have a distinct, known class count. "
            "Found %r" % (expected, [row["shape"] for row in outputs]))
    boxes_index = boxes_indices[0]
    boxes = outputs[boxes_index]["shape"]
    logits = outputs[1 - boxes_index]["shape"]
    classes = logits[2]
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 1:
        raise ValueError("%s; logits class dimension must be known and positive"
                         % expected)
    if not _matching_output_dimension(logits[1], boxes[1]):
        raise ValueError(
            "%s; logits/boxes query dimensions differ or cannot be proved equal: "
            "%r vs %r" % (expected, logits[1], boxes[1]))
    if not _matching_output_dimension(logits[0], boxes[0]):
        raise ValueError(
            "%s; logits/boxes batch dimensions differ or cannot be proved equal: "
            "%r vs %r" % (expected, logits[0], boxes[0]))
    if input_batch is not None and input_batch != 1:
        raise ValueError("%s; selected input batch must be 1, got %r"
                         % (expected, input_batch))
    batch = logits[0]
    if batch != 1 and not (isinstance(batch, str) and input_batch == 1):
        raise ValueError(
            "%s; output batch %r is not proven to be 1. Supply an explicit "
            "single-image input shape for a symbolic batch."
            % (expected, logits[0]))
    return "rtdetr_legacy"


def build_vision_contract(task_kind, adapter, layout, artifact_file, io_name,
                          io_shape, *, adapter_api_version=1, os_, arch, device,
                          dtype, score_threshold=None, nms_threshold=None,
                          mean=None, std=None, keypoint_count=None):
    """Assemble the contract for one converted vision model.

    Positional - what every vision model has:
        task_kind      the discriminator, e.g. ``object_detection``
        adapter        the pre/post-processor, e.g. ``yolo11``
        layout         the input layout, ``NCHW`` or ``NHWC``
        artifact_file  the module's file name, e.g. ``yolo11n.cpu.dll``
        io_name        the input tensor's name
        io_shape       the input tensor's shape, a sequence of ints

    Keyword-only - the target the artifact was built for, then the optional
    task-specific facts:
        os_, arch, device, dtype   the artifact's target quad (all required)
        score_threshold, nms_threshold   detection defaults (object_detection)
        mean, std                        per-channel normalisation (classification)
        keypoint_count                   number of keypoints (pose)

    Returns a plain ``dict`` ready for :func:`manifest.emit_manifest_text` and
    :func:`manifest.emit_manifest_json`. A task_kind that is not a vision task is
    refused rather than built into a wrong contract.
    """
    if task_kind not in VISION_TASK_KINDS:
        raise ValueError(
            "%r is not a vision task this builder knows (%s). An LLM contract "
            "carries text/tokenizer sections and is built by a separate path, "
            "not here." % (task_kind, ", ".join(sorted(VISION_TASK_KINDS))))
    validate_vision_adapter(task_kind, adapter)

    contract = {
        "format_version": FORMAT_VERSION,
        "task_kind": task_kind,
        "adapter": adapter,
        "adapter_api_version": adapter_api_version,
        "artifact": {
            "os": os_,
            "arch": arch,
            "device": device,
            "dtype": dtype,
            "file": artifact_file,
            "entry": VISION_ARTIFACT_ENTRY,
            "format": VISION_ARTIFACT_FORMAT,
        },
        "io": {
            "name": io_name,
            "shape": [int(dimension) for dimension in io_shape],
            "layout": layout,
            "dtype": IO_DTYPE,
        },
    }
    contract.update(_task_section(task_kind, score_threshold, nms_threshold,
                                  mean, std, keypoint_count))
    return contract


def _task_section(task_kind, score_threshold, nms_threshold, mean, std,
                  keypoint_count):
    """The task-specific section for *task_kind*, or an empty dict.

    Each branch adds its section only when the facts it needs are present, which
    is what lets one object_detection carry a default line (yolo11) while another
    carries none (rtdetr) from the same code path.
    """
    if task_kind == _OBJECT_DETECTION:
        if score_threshold is not None:
            defaults = {"score_threshold": score_threshold}
            if nms_threshold is not None:
                defaults["nms_threshold"] = nms_threshold
            return {"defaults": defaults}
    elif task_kind == _CLASSIFICATION:
        if mean is not None and std is not None:
            return {"vision": {"mean": list(mean), "std": list(std)}}
    elif task_kind == _POSE:
        if keypoint_count is not None:
            return {"pose": {"keypoint_count": keypoint_count}}
    return {}


TEXT_ARTIFACT_FORMAT = "dll"

_TEXT_GENERATION = "text_generation"


TEXT_TOKENIZER_STYLES = frozenset(("byte_bpe", "sentencepiece"))


TEXT_ADAPTERS = frozenset(("qwen2", "gemma2"))


def build_text_contract(adapter, artifact_file, context_window, bos, eos,
                        tokenizer_style, *, adapter_api_version=1, os_, arch,
                        device, dtype, device_arch=None):
    """Assemble the contract for one converted LLM (task ``text_generation``).

    Positional - what every LLM the zoo builds carries:
        adapter          the runtime adapter, ``qwen2`` or ``gemma2``
        artifact_file    the module's file name, e.g. ``qwen2-cpu.dll``
        context_window   the context length the module was built for
        bos, eos         the begin/end-of-sequence token ids
        tokenizer_style  the tokenizer family, ``byte_bpe`` or ``sentencepiece``

    Keyword-only - the target the artifact was built for:
        os_, arch, device, dtype   the artifact's target quad (all required)
        device_arch                the GPU architecture the module was compiled
                                   for, ``sm_75``; optional, and written only for
                                   a build that has one

    WHY device_arch IS NOT arch. ``arch`` is the PROCESSOR the bundle runs on -
    ``x86_64`` - and it stays that whatever the module was compiled for. A CUDA
    module additionally carries the architecture of the card it was compiled for,
    and the two are different facts about different hardware: a bundle built for
    sm_89 on an x86_64 machine has both. Without the second one a cross-built
    bundle cannot be told from any other: it loads, then fails on a card of the
    wrong generation, with nothing in the bundle to point at the cause.

    An LLM contract differs from a vision one in three ways, each read off the
    shipped bundles rather than invented: the artifact's logical format is ``dll``
    and it names no entry point; there is no ``io`` section, because an LLM's input
    is a token stream rather than a tensor of a fixed shape; and it carries
    ``text`` and ``tokenizer`` sections in place of any vision line.

    Returns a plain ``dict`` ready for :func:`manifest.emit_manifest_text` and
    :func:`manifest.emit_manifest_json`. An adapter or a tokenizer style outside
    the known pairs is refused rather than built into a contract the facade would
    reject.
    """
    if adapter not in TEXT_ADAPTERS:
        raise ValueError(
            "%r is not an LLM adapter this builder knows (%s). The facade's "
            "ReadManifest accepts only qwen2 or gemma2 under text_generation, so a "
            "value outside that pair would build a manifest it rejects - most "
            "likely a typo." % (adapter, ", ".join(sorted(TEXT_ADAPTERS))))
    if tokenizer_style not in TEXT_TOKENIZER_STYLES:
        raise ValueError(
            "%r is not a tokenizer style this builder knows (%s). The LLM bundles "
            "use a byte-level BPE or a SentencePiece model; a value outside that "
            "pair is most likely a typo." % (
                tokenizer_style, ", ".join(sorted(TEXT_TOKENIZER_STYLES))))

    artifact = {
        "os": os_,
        "arch": arch,
        "device": device,
        "dtype": dtype,
        "file": artifact_file,
        "format": TEXT_ARTIFACT_FORMAT,
    }


    if device_arch:
        artifact["device_arch"] = device_arch

    return {
        "format_version": FORMAT_VERSION,
        "task_kind": _TEXT_GENERATION,
        "adapter": adapter,
        "adapter_api_version": adapter_api_version,
        "artifact": artifact,
        "text": {
            "context_window": context_window,
            "bos": bos,
            "eos": eos,
        },
        "tokenizer": {"style": tokenizer_style},
    }


def parse_float_csv(text):
    """Parse "0.485,0.456,0.406" into [0.485, 0.456, 0.406], or None when empty.

    Empty pieces are dropped, so a stray comma adds no spurious value; a blank
    string (or None) yields None, which is what lets an omitted mean/std leave
    the vision line out of the manifest altogether.
    """
    if not text:
        return None
    return [float(piece) for piece in text.split(',') if piece.strip()]


def build_contract_input(task, adapter, *, adapter_api_version=1, layout="NCHW",
                         score_threshold=None, nms_threshold=None, mean=None,
                         std=None, keypoint_count=None, labels=None):
    """Assemble the Model contract dict handed to :func:`pipeline.run`, or None.

    task, adapter    the task the model serves and its pre/post-processor. A task
                     turns the whole thing on; an empty task returns None, and the
                     run then behaves exactly as it did before a contract existed -
                     no manifest is written.
    the rest         the optional facts a manifest may carry, already parsed into
                     their own types: thresholds as floats, mean/std as lists (see
                     :func:`parse_float_csv`), keypoint_count as an int.

    Returns a plain dict carrying the keys :func:`pipeline.run` reads, or None when
    no task is named. A task named without an adapter is refused HERE rather than
    carried into the build: the manifest needs both, and the two faces of the core
    must refuse the same input in the same way.
    """
    task = (task or "").strip()
    if not task:
        return None
    adapter = (adapter or "").strip()
    if not adapter:
        raise ValueError("an adapter is required when a task is given "
                         "(for example yolo11, rtdetr, tensorflow or birefnet)")
    return {
        "task": task,
        "adapter": adapter,
        "adapter_api_version": adapter_api_version,
        "layout": layout or "NCHW",
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "mean": mean,
        "std": std,
        "keypoint_count": keypoint_count,
        "labels": labels or None,
    }
