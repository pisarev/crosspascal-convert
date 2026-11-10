# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


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
    return str(task_kind).lower() in _PADDLE_OCR_ROLES


def validate_paddleocr_outputs(task_kind, adapter, outputs, *, input_shape=None):
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
    return (str(task_kind).lower() == _OBJECT_DETECTION
            and str(adapter).lower() == "rtdetr")


def _matching_output_dimension(left, right):
    if isinstance(left, int) and not isinstance(left, bool):
        return left > 0 and left == right
    return isinstance(left, str) and bool(left) and left == right


def validate_adapter_outputs(task_kind, adapter, outputs, *, input_batch=None):
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
    if not text:
        return None
    return [float(piece) for piece in text.split(',') if piece.strip()]


def build_contract_input(task, adapter, *, adapter_api_version=1, layout="NCHW",
                         score_threshold=None, nms_threshold=None, mean=None,
                         std=None, keypoint_count=None, labels=None):
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
