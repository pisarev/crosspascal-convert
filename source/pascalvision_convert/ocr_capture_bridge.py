# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Read-only projection of a SUPPLIED original ORT raw capture onto OCR plan v2.

This is a byte-integrity and contract adapter, not an execution witness.
An ORT producer receipt and matching SHA-256 files cannot independently attest
that ORT, Pascal, image preprocessing, or human annotation actually ran.
Nothing here creates evidence.json, runs a model, or issues a Facade/READY marker.

The selected output is identified by its exact declared semantic NAME. The
resulting paths describe a point-in-time inspection of existing files; any
future consumer must check the SHA-256 again when it opens their bytes.
"""

from __future__ import annotations

from pathlib import Path

from . import ocr_composite as composite
from . import ocr_composite_oracle as oracle
from . import ocr_ort_raw_capture as raw_producer


class CaptureBridgeRefused(ValueError):
    """The supplied capture cannot be bound to this OCR DRAFT plan."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureBridgeRefused(message)


def _object(value, label: str) -> dict:
    _require(isinstance(value, dict), "%s must be an object" % label)
    return value


def _exact_keys(value: dict, expected, label: str) -> None:
    _require(set(value) == set(expected),
             "%s has extra/missing keys; unsupported producer schema" % label)


def _input_contract(plan: dict, bundle_root: Path, stage: str):
    preprocessing = _object(plan.get("preprocessing"), "plan.preprocessing")
    stage_preproc = _object(preprocessing.get(stage),
                            "plan.preprocessing." + stage)
    shape = stage_preproc.get("input_shape")
    name = stage_preproc.get("input_name")
    _require(type(name) is str and 0 < len(name) <= 256 and
             not any(ord(char) < 32 for char in name),
             "OCR plan has no bounded exact original input name")
    _require(type(shape) is list and len(shape) == 4 and
             all(type(dim) is int and 1 <= dim <= 4096 for dim in shape) and
             shape[:2] == [1, 3] and
             shape[0] * shape[1] * shape[2] * shape[3] <=
             raw_producer.MAX_INPUT_ELEMENTS,
             "OCR plan has no bounded static [1,3,H,W] input shape")

    manifest, _ = oracle._read_json(
        bundle_root / stage, "manifest.json",
        "staged %s source manifest" % stage)
    io = _object(_object(manifest, "source manifest").get("io"), "source io")
    _require(io.get("name") == name and io.get("shape") == shape and
             io.get("dtype") == "float32" and io.get("layout") == "NCHW",
             "plan input name/shape differs from checked individual manifest")
    return name, list(shape)


def _original_sha(plan: dict, stage: str) -> str:
    components = _object(plan.get("components"), "plan.components")
    component = _object(components.get(stage), "plan.components." + stage)
    provenance = _object(component.get("original_provenance"),
                         "plan original provenance")


    return oracle._sha(provenance.get("source_hash"),
                       "plan.components.%s.original_provenance.source_hash" %
                       stage)


def _receipt_grammar(capture: dict) -> None:
    expected = set(raw_producer._no_attestation()) | {
        "schema", "schema_version", "execution_provider",
        "active_providers", "onnxruntime_version",
        "original_onnx", "input", "outputs", "evidence_limit",
    }
    _exact_keys(capture, expected, "capture.json")

    model = _object(capture.get("original_onnx"), "capture.original_onnx")
    inp = _object(capture.get("input"), "capture.input")
    _exact_keys(model, {"file", "sha256", "bytes", "external_data"},
                "capture.original_onnx")
    _exact_keys(inp, {"file", "name", "dtype", "shape", "bytes", "sha256"},
                "capture.input")
    _require(type(capture.get("onnxruntime_version")) is str and
             0 < len(capture["onnxruntime_version"]) <= 256,
             "capture has no bounded declared ORT version")
    _require(type(capture.get("evidence_limit")) is str and
             0 < len(capture["evidence_limit"]) <= 4096,
             "capture lacks an explicit bounded evidence limitation")

    outputs = capture.get("outputs")
    _require(type(outputs) is list and
             1 <= len(outputs) <= raw_producer.MAX_OUTPUT_TENSORS,
             "capture must contain a bounded list of outputs")
    seen = set()
    for output in outputs:
        row = _object(output, "capture output")
        _exact_keys(row, {"name", "file", "dtype", "shape", "bytes",
                          "sha256"}, "capture output")
        name = row.get("name")
        _require(type(name) is str and 0 < len(name) <= 256 and
                 not any(ord(char) < 32 for char in name),
                 "capture output name is missing/unsafe")
        _require(name not in seen,
                 "ambiguous duplicate ORT output semantic name")
        seen.add(name)


def _checked_tensor_descriptor(root: Path, item: dict) -> dict:
    shape = item["shape"]
    count = 1
    for dim in shape:
        count *= dim
    return {
        "name": item["name"],
        "path": str(root / item["file"]),
        "shape": list(shape),
        "dtype": "float32le",
        "bytes": item["bytes"],
        "count": count,
        "sha256": item["sha256"],
    }


def _check_rec_shape(plan: dict, selected: dict) -> None:
    ctc = _object(plan.get("ctc"), "plan.ctc")
    layout = ctc.get("logits_layout")
    classes = ctc.get("expected_classes")
    shape = selected["shape"]
    _require(type(classes) is int and 2 <= classes <=
             composite._MAX_DICT_TOKENS + 1,
             "planned REC CTC class count is not bounded")
    if layout == "NTC":
        _require(len(shape) == 3 and shape[0] == 1,
                 "REC NTC logits need [1,T,C]")
        times, class_axis = shape[1], shape[2]
    elif layout == "NCT":
        _require(len(shape) == 3 and shape[0] == 1,
                 "REC NCT logits need [1,C,T]")
        times, class_axis = shape[2], shape[1]
    elif layout == "TC":
        _require(len(shape) == 2, "REC TC logits need [T,C]")
        times, class_axis = shape
    elif layout == "CT":
        _require(len(shape) == 2, "REC CT logits need [C,T]")
        class_axis, times = shape
    else:
        raise CaptureBridgeRefused("REC has no supported explicit CTC layout")
    _require(class_axis == classes and 2 <= times <= 2048 and
             selected["bytes"] <= oracle.MAX_RECOGNITION_ELEMENTS * 4,
             "REC logits mismatch CTC classes/time or exceed bounded quota")


def _inspect(bundle, stage, capture_dir, output_name):
    _require(type(stage) is str and stage in ("det", "rec"),
             "stage must be exactly det or rec")
    _require(type(output_name) is str and
             0 < len(output_name) <= 256 and
             not any(ord(char) < 32 for char in output_name),
             "output_name must identify one exact bounded semantic ORT output")

    bundle_root, plan, plan_sha256 = oracle._validate_plan(bundle)
    expected_model_sha = _original_sha(plan, stage)
    expected_input_name, expected_input_shape = _input_contract(
        plan, bundle_root, stage)

    root = raw_producer._absolute(capture_dir, "original ORT capture folder")
    raw_producer._check_private_root(root)

    input_bytes = (4 * expected_input_shape[0] * expected_input_shape[1] *
                   expected_input_shape[2] * expected_input_shape[3])
    actual_input_sha, measured_bytes = raw_producer._sha_file(
        root / raw_producer._INPUT_COPY, "ORT raw input bytes",
        limit=raw_producer.MAX_INPUT_BYTES, expected_bytes=input_bytes)
    _require(measured_bytes == input_bytes, "raw ORT input bytes changed")

    capture, capture_sha256 = raw_producer._verify_capture(
        root, expected_model_sha, actual_input_sha, expected_input_shape)
    _receipt_grammar(capture)
    inp = capture["input"]
    _require(inp["name"] == expected_input_name,
             "ORT input tensor name differs from composite plan")
    _require(inp["shape"] == expected_input_shape,
             "ORT input shape differs from composite plan")

    matches = [item for item in capture["outputs"]
               if item["name"] == output_name]
    _require(len(matches) == 1,
             "ORT semantic output name is unknown or ambiguous")
    selected = matches[0]
    if stage == "det":
        shape = selected["shape"]
        _require(len(shape) == 4 and shape[:2] == [1, 1],
                 "selected Det probability map must be [1,1,H,W]")
    else:
        _check_rec_shape(plan, selected)

    input_desc = _checked_tensor_descriptor(root, inp)
    output_desc = _checked_tensor_descriptor(root, selected)
    return {
        "status": "NOT_CHECKED",
        "source_kind": "SUPPLIED_ORIGINAL_ORT_RAW_CAPTURE",
        "producer_status": raw_producer.STATUS,
        "producer_authenticity": "NOT_ATTESTED",
        "process_provenance": "NOT_ATTESTED",
        "acceptance_status": "NOT_CHECKED",
        "composite_verification": "NOT_CHECKED",
        "native_composite_oracle": "NOT_RUN",
        "human_ground_truth": "NOT_ATTESTED",
        "facade_manifest_issued": False,
        "ready_issued": False,
        "stage": stage,
        "plan_sha256": plan_sha256,
        "capture_sha256": capture_sha256,
        "original_onnx_sha256": expected_model_sha,
        "input": input_desc,
        "output": output_desc,
        "missing_obligations": [
            "Independently witnessed original ORT execution/provenance",
            "Original image/crop, measured preprocessing and image SHA binding",
            "Pascal Det/Rec/CTC captured tensor comparison and process telemetry",
            "Independent detection candidates and human ground-truth annotations",
            "Full per-crop recognition and composite numerical/human acceptance",
        ],
        "provenance_limit": (
            "Selected preexisting ORT files and original model SHA match a "
            "DRAFT plan at inspection time. File hashes do not attest an "
            "independent producer, native Pascal, image/crop identity, "
            "human annotations or acceptance. Rehash paths before later use."),
    }


def inspect_ort_capture(bundle, stage, capture_dir, output_name) -> dict:
    """Inspect a raw ORT capture read-only; refuse conflicts, never issue READY.

    No output tree is created and no model, process, CLI, native runtime, or
    independent annotation is invoked. Even a fully matching capture remains
    NOT_CHECKED; raw producer metadata is inherently self-declared.
    """
    try:
        return _inspect(bundle, stage, capture_dir, output_name)
    except CaptureBridgeRefused:
        raise
    except (oracle.InvalidEvidence, oracle.MissingEvidence,
            composite.CompositeContractError, raw_producer.CaptureRefused,
            OSError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, RecursionError) as exc:
        raise CaptureBridgeRefused("OCR raw capture bridge refused: %s" % exc) from exc


__all__ = ["CaptureBridgeRefused", "inspect_ort_capture"]
