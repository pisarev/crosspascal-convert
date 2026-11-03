# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Emit a model bundle manifest from a model contract.

The bundle manifest (``manifest.txt``) is the single source of truth a converted
model carries: the converter (the master) writes it, and the Pascal facade
(``CrossVision.ML.Facade`` -> ``TModelBundle``) reads it. This module turns a
"model contract" - a plain description of one model - into exactly the flat text
the facade parses, plus the structured JSON companion meant for tools.

Only the standard library is used, on purpose: the converter's promise is that
nothing has to be installed to run it, and that promise reaches down to here.

The flat format is line oriented: ``key rest``. Structural lines carry
``field=value`` tokens after the key. The facade reads fields by name, so it is
order insensitive, but the manifests already shipped in the model zoo fix a
particular field order; this emitter reproduces it byte for byte.

Contract schema
---------------
A contract is a plain ``dict`` built from ``dict`` / ``list`` / ``str`` / ``int``
/ ``float`` only, so it maps straight onto JSON and back. This one shape can
express every model kind in the zoo today - vision detectors, classifiers,
segmenters, oriented boxes, pose, and the two LLM families::

    {
        "format_version": 1,               # int, optional (defaults to 1)
        "task_kind": "object_detection",   # str, required (the discriminator)
        "adapter": "yolo11",               # str, required
        "adapter_api_version": 1,          # int, required
        "artifact": {                      # dict, required
            "os": "windows",
            "arch": "x86_64",
            "device": "cpu",
            "dtype": "float32",
            "file": "yolo11n.cpu.dll",
            "entry": "main",               # str, optional (LLM artifacts omit it)
            "format": "so",
        },
        "io": {                            # dict, optional (an LLM's input is tokens)
            "name": "images",
            "shape": [1, 3, 640, 640],     # list[int]
            "layout": "NCHW",              # NCHW | NHWC
            "dtype": "float32",
        },
        # at most one task-specific vision section, where the task carries one:
        "defaults": {                      # detection thresholds
            "score_threshold": 0.25,
            "nms_threshold": 0.45,         # optional (RT-DETR has no suppression)
        },
        "vision": {                        # per-channel normalisation (ImageNet nets)
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "pose": {"keypoint_count": 17},
        # text generation carries text + tokenizer instead of io:
        "text": {"context_window": 2048, "bos": 151643, "eos": 151645},
        "tokenizer": {"style": "byte_bpe"},  # byte_bpe | sentencepiece
    }

Optional audit fields make a bundle a verifiable artifact. They are absent from
the contracts the byte-for-byte tests build, so those keep their exact output; the
pipeline adds them on a real build (see :func:`pipeline._emit_manifest` and
``state/MANIFEST-SPEC.md``). Each is emitted only when present::

    "artifact": {..., "hash": "<sha256 hex of the built module>"},  # rides the
                                    # artifact line as an extra token after format
    "provenance": {                 # where the model came from and when
        "source": "yolo11n.onnx",   # the source as recorded (a path or a URL)
        "source_hash": "<sha256 hex>",       # optional: the source file's own hash
        "built_at": "2026-09-06T12:00:00Z",  # optional: ISO 8601 UTC build time
    },
    "toolchain": {                  # what produced the module
        "tvm": "0.26.0", "onnx": "1.22.0",
        "tvm_runtime": "0.26.0",    # optional, only when it advertises a version
    },
    "license": "Apache-2.0",        # optional: the source model's licence

The internal contract keys stay snake_case (source_hash, built_at, tvm_runtime),
as score_threshold and adapter_api_version already are; the flat text renders them
camelCase (sourceHash, builtAt, tvmRuntime), as scoreThreshold already is.

The line ending is a property of the file, not of the contract: the facade reads
CRLF and LF alike (``TStringList.LoadFromFile``), and the manifests in the zoo
happen to use both. ``emit_manifest_text`` therefore takes a ``newline`` - it
defaults to ``"\n"`` (what the converter's own report writer uses) and can be set
to ``"\r\n"`` to reproduce a CRLF manifest exactly.
"""

import json
import re


FORMAT_VERSION = 1

_REQUIRED_FIELDS = ("task_kind", "adapter", "adapter_api_version", "artifact")
_ARTIFACT_FIELDS = ("os", "arch", "device", "dtype", "file", "format")
_IO_FIELDS = ("name", "shape", "layout", "dtype")
_RTDETR_FUSED_FIELDS = ("cx", "cy", "width", "height", "confidence", "class_id")
_RTDETR_FUSED_FORMAT = "rtdetr_fused_cxcywh_v1"
_OMDET_INPUTS = (
    ("pixel_values", (1, 3, 640, 640), "NCHW", "float32"),
    ("class_features", (8, 1, 512), "NCF", "float32"),
    ("task_features", (19, 1, 512), "NCF", "float32"),
    ("task_mask", (1, 19), "NT", "int64"),
)
_OMDET_OUTPUTS = (
    ("boxes", (1, 900, 4), "float32", "normalized_cxcywh"),
    ("scores", (1, 900, 8), "float32", "raw_logits_not_sigmoid"),
)
_OMDET_ASSETS = {
    "profile": "tiny-swin-t-hf-r7",
    "processor_config": "preprocessor_config.json",
    "model_config": "config.json",
    "feature_directory": "features",
}


def emit_manifest_text(contract, newline="\n"):
    """Render *contract* as the flat ``manifest.txt`` the facade reads.

    The result is byte-for-byte identical to a shipped bundle manifest when given
    that file's *newline*. Every line, the last one included, is terminated by
    *newline*.
    """
    data = _validated(contract)

    lines = [
        "formatVersion %s" % _integer(data.get("format_version", FORMAT_VERSION)),
        "task %s" % data["task_kind"],
        "adapter %s" % data["adapter"],
        "adapterApiVersion %s" % _integer(data["adapter_api_version"]),
        _artifact_line(data["artifact"]),
    ]
    if data["adapter"] == "omdet":
        
        
        
        lines.extend(_io_line(item) for item in data["io"])
        lines.extend(_omdet_output_line(item) for item in data["omdet_outputs"])
        lines.append(_omdet_line(data["omdet"]))
    elif data.get("io"):
        lines.append(_io_line(data["io"]))
    
    
    
    if data.get("output"):
        lines.append(_output_line(data["output"]))
    
    
    
    if data.get("defaults"):
        lines.append(_defaults_line(data["defaults"]))
    if data.get("vision"):
        lines.append(_vision_line(data["vision"]))
    if data.get("pose"):
        lines.append(_pose_line(data["pose"]))
    if data.get("text"):
        lines.append(_text_line(data["text"]))
    if data.get("tokenizer"):
        lines.append(_tokenizer_line(data["tokenizer"]))

    
    
    
    
    if data.get("provenance"):
        lines.append(_provenance_line(data["provenance"]))
    if data.get("toolchain"):
        lines.append(_toolchain_line(data["toolchain"]))
    if data.get("license"):
        lines.append(_license_line(data["license"]))

    return "".join(line + newline for line in lines)


def emit_manifest_json(contract, indent=2):
    """Render *contract* as structured JSON (path A of MANIFEST-SPEC).

    This is the companion for tooling; the flat text is what Pascal reads. The
    JSON mirrors the contract faithfully, so ``json.loads(emit_manifest_json(c))``
    returns the same contract.
    """
    return json.dumps(_validated(contract), indent=indent, ensure_ascii=False)




def _artifact_line(artifact):
    parts = [
        "artifact",
        "os=%s" % artifact["os"],
        "arch=%s" % artifact["arch"],
        "device=%s" % artifact["device"],
    ]
    
    
    
    
    
    if artifact.get("device_arch"):
        parts.append("deviceArch=%s" % artifact["device_arch"])
    parts += [
        "dtype=%s" % artifact["dtype"],
        "file=%s" % artifact["file"],
    ]
    
    
    if artifact.get("entry") is not None:
        parts.append("entry=%s" % artifact["entry"])
    parts.append("format=%s" % artifact["format"])
    
    
    
    
    if artifact.get("hash") is not None:
        parts.append("hash=%s" % artifact["hash"])
    return " ".join(parts)


def _io_line(io):
    return " ".join([
        "io",
        "name=%s" % io["name"],
        "shape=%s" % _int_csv(io["shape"]),
        "layout=%s" % io["layout"],
        "dtype=%s" % io["dtype"],
    ])


def _output_line(output):
    """One precisely checked fused output; no guessed dynamic tensor metadata."""
    return " ".join([
        "output",
        "name=%s" % output["name"],
        "shape=%s" % _int_csv(output["shape"]),
        "dtype=%s" % output["dtype"],
        "format=%s" % output["format"],
        "fields=%s" % ",".join(output["fields"]),
    ])


def _omdet_output_line(output):
    return " ".join([
        "omdetOutput",
        "name=%s" % output["name"],
        "shape=%s" % _int_csv(output["shape"]),
        "dtype=%s" % output["dtype"],
        "semantics=%s" % output["semantics"],
    ])


def _omdet_line(values):
    return " ".join([
        "omdet",
        "profile=%s" % values["profile"],
        "processorConfig=%s" % values["processor_config"],
        "modelConfig=%s" % values["model_config"],
        "featureDir=%s" % values["feature_directory"],
    ])


def _defaults_line(defaults):
    parts = ["default", "scoreThreshold=%s" % _number(defaults["score_threshold"])]
    if defaults.get("nms_threshold") is not None:
        parts.append("nmsThreshold=%s" % _number(defaults["nms_threshold"]))
    return " ".join(parts)


def _vision_line(vision):
    return " ".join([
        "vision",
        "mean=%s" % _number_csv(vision["mean"]),
        "std=%s" % _number_csv(vision["std"]),
    ])


def _pose_line(pose):
    return "pose keypointCount=%s" % _integer(pose["keypoint_count"])


def _text_line(text):
    return " ".join([
        "text",
        "contextWindow=%s" % _integer(text["context_window"]),
        "bos=%s" % _integer(text["bos"]),
        "eos=%s" % _integer(text["eos"]),
    ])


def _tokenizer_line(tokenizer):
    return "tokenizer style=%s" % tokenizer["style"]


def _provenance_line(provenance):
    """Where the model came from and when. The source is always written; its own
    hash and the build time follow only when the pipeline could take them (a
    directory model has no file hash; both are best-effort). Values are rendered
    verbatim, as file names already are - a URL, a path, a hex digest and an ISO
    8601 timestamp carry no whitespace to break the token stream."""
    parts = ["provenance", "source=%s" % provenance["source"]]
    if provenance.get("source_hash") is not None:
        parts.append("sourceHash=%s" % provenance["source_hash"])
    if provenance.get("built_at") is not None:
        parts.append("builtAt=%s" % provenance["built_at"])
    return " ".join(parts)


def _toolchain_line(toolchain):
    """What produced the module. tvm and onnx are always written (the pipeline
    records ``unknown`` when a package will not import rather than dropping the
    field); tvm.runtime is written only when it advertises its own version."""
    parts = [
        "toolchain",
        "tvm=%s" % toolchain["tvm"],
        "onnx=%s" % toolchain["onnx"],
    ]
    if toolchain.get("tvm_runtime") is not None:
        parts.append("tvmRuntime=%s" % toolchain["tvm_runtime"])
    return " ".join(parts)


def _license_line(license_value):
    """The source model's licence, written as the bare rest of the line - like
    the labels line - so a value with spaces (a full licence name) survives. The
    facade skips the unknown ``license`` key, so this is informational."""
    return "license %s" % license_value




def _integer(value):
    """An integer field, rendered bare. ``bool`` is refused: it is an ``int``
    subclass, and letting True through as ``1`` would hide a contract mistake."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an int, got %r" % (value,))
    return str(value)


def _number(value):
    """A numeric field: an ``int`` bare, a ``float`` as the shortest decimal that
    round-trips. ``repr`` gives that shortest form, always with a dot separator -
    which is what the facade's invariant-locale parser expects."""
    if isinstance(value, bool):
        raise TypeError("expected a number, got a bool")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise TypeError("expected a number, got %r" % (value,))


def _int_csv(values):
    return ",".join(_integer(v) for v in values)


def _number_csv(values):
    return ",".join(_number(v) for v in values)




def _validated(contract):
    """Check that *contract* carries the fields every manifest needs, then return
    it unchanged. A missing field is a caller error, reported by name rather than
    left to surface as a cryptic ``KeyError`` deep in a line writer."""
    if not isinstance(contract, dict):
        raise TypeError("contract must be a dict, not %s" % type(contract).__name__)
    _require(contract, _REQUIRED_FIELDS, "contract")

    artifact = contract["artifact"]
    if not isinstance(artifact, dict):
        raise TypeError("contract['artifact'] must be a dict")
    _require(artifact, _ARTIFACT_FIELDS, "artifact")

    io = contract.get("io")
    if contract["adapter"] == "omdet":
        _validate_omdet(contract)
    else:
        if ("omdet" in contract or "omdet_outputs" in contract or
                isinstance(io, (list, tuple))):
            raise ValueError("four-input OmDet passport cannot name a single-input adapter")
        if io is not None:
            if not isinstance(io, dict):
                raise TypeError("contract['io'] must be a dict")
            _require(io, _IO_FIELDS, "io")

    output = contract.get("output")
    if output is not None:
        if not isinstance(output, dict):
            raise TypeError("contract['output'] must be a dict")
        _require(output, ("name", "shape", "dtype", "format", "fields"), "output")
        if not isinstance(output["name"], str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_./:-]*", output["name"]):
            raise ValueError("fused RT-DETR output name is not a safe single token")
        shape = output["shape"]
        if (not isinstance(shape, (tuple, list)) or len(shape) != 3
                or any(type(dim) is not int or dim < 1 for dim in shape)
                or shape[0] != 1 or shape[2] != 6):
            raise ValueError("fused RT-DETR output must have known [1,N,6] shape")
        if (output["format"] != _RTDETR_FUSED_FORMAT
                or output["dtype"] not in ("float16", "float32")
                or not isinstance(output["fields"], (list, tuple))
                or tuple(output["fields"]) != _RTDETR_FUSED_FIELDS):
            raise ValueError("unsupported or inconsistent fused RT-DETR output ABI")
        if (contract.get("task_kind") != "object_detection"
                or contract.get("adapter") != "rtdetr"):
            raise ValueError("fused RT-DETR output ABI cannot label another adapter")

    
    
    provenance = contract.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise TypeError("contract['provenance'] must be a dict")
        _require(provenance, ("source",), "provenance")
    toolchain = contract.get("toolchain")
    if toolchain is not None:
        if not isinstance(toolchain, dict):
            raise TypeError("contract['toolchain'] must be a dict")
        _require(toolchain, ("tvm", "onnx"), "toolchain")
    license_value = contract.get("license")
    if license_value is not None and not isinstance(license_value, str):
        raise TypeError("contract['license'] must be a string, not %s"
                        % type(license_value).__name__)
    if license_value is not None and (
            not license_value.strip() or len(license_value) > 1024 or
            any(ord(char) < 32 or ord(char) == 127 or
                char in '\u0085\u2028\u2029'
                for char in license_value)):
        
        
        
        
        raise ValueError("contract['license'] must be one bounded text line")

    return contract


def _validate_omdet(contract):
    """EXACT pinned tiny-v7 Pascal transport; no synthetic one-input fallback.

    A candidate profile still has verification=NOT_CHECKED until the actual
    trained Pascal/ORT acceptance. Schema validation alone does not grant it.
    """
    optional = {"provenance", "toolchain", "license"}
    required = {"format_version", "task_kind", "adapter", "adapter_api_version",
                "artifact", "io", "omdet_outputs", "omdet"}
    if set(contract) - (required | optional) or required - set(contract):
        raise ValueError("OmDet version 2 passport requires exact top-level fields")
    if (type(contract["format_version"]) is not int or
            contract["format_version"] != 2 or
            type(contract["adapter_api_version"]) is not int or
            contract["adapter_api_version"] != 2 or
            contract["task_kind"] != "object_detection"):
        raise ValueError("OmDet requires object_detection manifest/adapter API v2")

    artifact = contract["artifact"]
    device = artifact.get("device")
    required_artifact = set(_ARTIFACT_FIELDS) | {"entry", "hash"}
    exact_artifact = (required_artifact | {"device_arch"}
                      if device == "cuda" else required_artifact)
    if set(artifact) != exact_artifact:
        raise ValueError("OmDet v2 native artifact requires exact typed fields")
    if (artifact.get("os") != "windows" or
            artifact.get("arch") != "x86_64" or
            device not in ("cpu", "cuda") or
            artifact.get("dtype") != "float32" or
            artifact.get("file") != "omdet-turbo-tiny.%s.dll" % device or
            artifact.get("format") != "so" or
            artifact.get("entry") != "main"):
        raise ValueError("unsupported OmDet tiny native module target/ABI")
    if (device == "cuda" and
            (not isinstance(artifact["device_arch"], str) or
             not re.fullmatch(r"sm_[1-9][0-9]{1,2}[a-z]?", artifact["device_arch"]))):
        raise ValueError("OmDet CUDA candidate must pin a concrete SM architecture")
    sha = artifact.get("hash")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("OmDet module must carry pinned lowercase SHA-256")

    inputs = contract["io"]
    if not isinstance(inputs, list) or len(inputs) != len(_OMDET_INPUTS):
        raise ValueError("OmDet requires four ordered io entries")
    for index, (actual, expected) in enumerate(zip(inputs, _OMDET_INPUTS)):
        if (not isinstance(actual, dict) or set(actual) != set(_IO_FIELDS) or
                actual.get("name") != expected[0] or
                actual.get("layout") != expected[2] or
                actual.get("dtype") != expected[3] or
                not isinstance(actual.get("shape"), list) or
                tuple(actual["shape"]) != expected[1] or
                any(type(dim) is not int for dim in actual["shape"])):
            raise ValueError("OmDet native io[%d] differs from pinned graph" % index)

    outputs = contract["omdet_outputs"]
    fields = {"name", "shape", "dtype", "semantics"}
    if not isinstance(outputs, list) or len(outputs) != len(_OMDET_OUTPUTS):
        raise ValueError("OmDet requires two ordered raw outputs")
    for index, (actual, expected) in enumerate(zip(outputs, _OMDET_OUTPUTS)):
        if (not isinstance(actual, dict) or set(actual) != fields or
                actual.get("name") != expected[0] or
                actual.get("dtype") != expected[2] or
                actual.get("semantics") != expected[3] or
                not isinstance(actual.get("shape"), list) or
                tuple(actual["shape"]) != expected[1] or
                any(type(dim) is not int for dim in actual["shape"])):
            raise ValueError("OmDet raw output[%d] semantics/shape invalid" % index)

    assets = contract["omdet"]
    if not isinstance(assets, dict) or assets != _OMDET_ASSETS:
        raise ValueError("OmDet v2 requires exact SHA-pinned HF assets and feature directory")

    provenance = contract.get("provenance")
    if provenance is not None:
        if (not isinstance(provenance, dict) or
                not set(provenance).issubset({"source", "source_hash", "built_at"}) or
                "source" not in provenance):
            raise ValueError("OmDet provenance requires exact supported fields")
        source = provenance["source"]
        if (not isinstance(source, str) or
                not re.fullmatch(r"source/[A-Za-z0-9_][A-Za-z0-9_.-]*\.onnx", source)
                or ".." in source or source.endswith(".")):
            raise ValueError("OmDet original ONNX provenance must be a safe bundled path")
        if "source_hash" in provenance and (
                not isinstance(provenance["source_hash"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", provenance["source_hash"])):
            raise ValueError("OmDet original ONNX provenance SHA-256 invalid")
        if "built_at" in provenance and (
                not isinstance(provenance["built_at"], str) or
                not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                                 provenance["built_at"])):
            raise ValueError("OmDet build timestamp must be unambiguous UTC")

    toolchain = contract.get("toolchain")
    if toolchain is not None:
        if (not isinstance(toolchain, dict) or
                not set(toolchain).issubset({"tvm", "onnx", "tvm_runtime"}) or
                not {"tvm", "onnx"}.issubset(toolchain)):
            raise ValueError("OmDet toolchain must name exact supported fields")
        for key, value in toolchain.items():
            if (not isinstance(value, str) or
                    not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}", value)):
                raise ValueError("OmDet toolchain version is not a safe single token: " + key)


def _require(mapping, keys, what):
    for key in keys:
        if mapping.get(key) is None:
            raise ValueError("%s is missing required field %r" % (what, key))
