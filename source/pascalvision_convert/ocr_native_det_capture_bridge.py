# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import struct

from . import ocr_capture_bridge as ort_bridge
from . import ocr_composite as composite
from . import ocr_composite_oracle as oracle
from . import ocr_ort_raw_capture as raw


class NativeDetCaptureRefused(ValueError):
    pass


_STATUS = "NATIVE_DET_RAW_CAPTURE_ONLY"
_MODEL_LIMIT = 512 * 1024 * 1024
_JSON_LIMIT = 64 * 1024
_PNG_LIMIT = 32 * 1024 * 1024
_PNG_PIXELS = 8_000_000
_SIDECAR_LIMIT = 4096
_TENSOR_LIMIT = 8_000_000
_RUNTIME_NAMES = {
    "tvm_runtime": "tvm_runtime.dll",
    "tvm_ffi": "tvm_ffi.dll",
    "tvm_runtime_extra": "tvm_runtime_extra.dll",
}
_CAPTURE_FILES = {"det-input.f32le", "det-output.f32le", "det-capture.json"}
_INPUT_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_EXPECTED_STATUS = {
    "schema": "crosspascal.paddleocr.native-det-raw-capture",
    "schema_version": 1,
    "status": _STATUS,
    "acceptance_status": "NOT_CHECKED",
    "verification_status": "NOT_CHECKED",
    "composite_verification": "NOT_CHECKED",
    "native_composite_oracle": "NOT_RUN",
    "native_recognizer": "NOT_RUN",
    "native_ctc": "NOT_RUN",
    "human_ground_truth": "NOT_ATTESTED",
    "independent_onnx_preprocessing": "NOT_CHECKED",
    "png_crc_or_pixel_equivalence": "NOT_CHECKED",
    "native_runtime_provenance": "NOT_ATTESTED",
    "natural_process_exit_attestation": "NOT_CHECKED",
    "paddleocr_det_model_identity": "CALLER_SELECTED_NOT_ATTESTED",
    "facade_manifest_issued": False,
    "ready_issued": False,
    "actual_pascal_det_inference": True,
    "device": "cpu",
    "runtime_sha_independent_expected_pin": False,
}
_CAPTURE_KEYS = set(_EXPECTED_STATUS) | {
    "model", "image", "input_name", "input", "raw_output",
    "runtime_observed_path", "runtime_selected_files_observed_sha256",
    "preprocessing", "evidence_limit",
}
_PREPROCESSING = {
    "source_color": "BGR",
    "resize": "bilinear_stretch_to_declared_shape",
    "channels": "BGR",
    "layout": "NCHW",
    "formula": "(float32(raw_BGR)/255-mean_BGR)/std_BGR",
    "mean_BGR": [0.406, 0.456, 0.485],
    "std_BGR": [0.225, 0.224, 0.229],
}


def _require(condition, why):
    if not condition:
        raise NativeDetCaptureRefused(why)


def _obj(value, where):
    _require(type(value) is dict, where + " must be an object")
    return value


def _keys(value, expected, where):
    _require(set(_obj(value, where)) == set(expected),
             where + " has an unsupported extra/missing schema key")


def _sha(value, where):
    return oracle._sha(value, where)


def _member(root, name, label, *, limit, expected_bytes=None):
    path = root / name
    actual, length = raw._sha_file(
        path, label, limit=limit, expected_bytes=expected_bytes)
    return path, actual, length


def _receipt(root):
    raw._check_private_root(root)


    names = []
    with os.scandir(root) as entries:
        for member in entries:
            names.append(member.name)
            _require(len(names) <= len(_CAPTURE_FILES),
                     "native Det capture exceeds three-member scan quota")
    _require(len(names) == len(_CAPTURE_FILES)
             and set(names) == _CAPTURE_FILES
             and len({name.casefold() for name in names}) == len(names),
             "native Det capture must have exactly its three owned members")
    path, digest, size = _member(
        root, "det-capture.json", "native Det receipt",
        limit=_JSON_LIMIT)
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size == size,
                 "native receipt changed length since admission")
        data = stream.read(_JSON_LIMIT + 1)
    _require(len(data) == size and hashlib.sha256(data).hexdigest() == digest,
             "native Det receipt changed on second read")
    record = _obj(raw._strict_json(data), "native Det receipt")
    _keys(record, _CAPTURE_KEYS, "native Det receipt")
    for name, value in _EXPECTED_STATUS.items():
        _require(record[name] == value and type(record[name]) is type(value),
                 "native receipt has an unsupported status/attestation: " + name)
    limit = record["evidence_limit"]
    _require(type(limit) is str and 0 < len(limit) <= 4096
             and not any(ord(char) < 32 and char not in "\r\n\t"
                         for char in limit),
             "native receipt must state a bounded evidence limitation")
    _require(record["preprocessing"] == _PREPROCESSING
             and type(record["preprocessing"]) is dict
             and set(record["preprocessing"]) == set(_PREPROCESSING),
             "native receipt preprocessing does not describe the exact Pascal Det producer")
    _member(root, "det-capture.json", "native Det receipt",
            limit=_JSON_LIMIT, expected_bytes=size)
    return record, digest, size


def _runtime_contract(plan):
    target = _obj(plan.get("target"), "plan.target")
    _require(target.get("os") == "windows"
             and target.get("arch") == "x86_64"
             and target.get("device") == "cpu"
             and target.get("sm") is None
             and target.get("driver_api") is None,
             "this Pascal Det producer supports only Windows x86-64 CPU")
    runtime = _obj(_obj(plan.get("runtime"), "plan.runtime")
                   .get("components"), "plan.runtime.components")
    _keys(runtime, _RUNTIME_NAMES, "plan.runtime.components")
    assets = _obj(
        {item["path"]: item for item in plan["assets"]},
        "plan asset lookup")
    observed = {}
    for logical, filename in _RUNTIME_NAMES.items():
        row = _obj(runtime[logical], "plan runtime " + logical)
        _keys(row, {"file", "sha256"}, "plan runtime " + logical)
        digest = _sha(row["sha256"], "plan runtime sha256")
        _require(row["file"] == filename,
                 "native Pascal expects exact executable-adjacent CPU runtime filename")
        for stage in ("det", "rec"):
            asset = _obj(assets.get(stage + "/" + filename),
                         stage + " runtime asset")
            _require(_sha(asset.get("sha256"), stage + " runtime sha")
                     == digest,
                     "declared runtime SHA is not bound to BOTH rehashed Det/Rec assets")
        observed[filename] = digest
    return observed


def _preprocess_contract(plan):
    preproc = _obj(_obj(plan.get("preprocessing"), "plan.preprocessing")
                   .get("det"), "plan.preprocessing.det")
    expected = {
        "source_color": "BGR",
        "model_color": "BGR",
        "pixel_scale": "divide_by_255",
        "normalization": "channelwise_mean_std_after_divide_by_255",
        "normalization_formula": "(float32(raw_BGR)/255 - mean_BGR) / std_BGR",
        "resize": "stretch_to_model_input",
        "interpolation": "bilinear",
        "layout": "NCHW",
        "dtype": "float32",
        "mean": _PREPROCESSING["mean_BGR"],
        "std": _PREPROCESSING["std_BGR"],
    }
    for field, value in expected.items():
        _require(preproc.get(field) == value
                 and type(preproc.get(field)) is type(value),
                 "DRAFT Det preprocessing does not match the real Pascal path: " + field)


def _model_and_sidecar(plan, bundle_root, record, input_name, input_shape):
    component = _obj(_obj(plan["components"], "plan.components")
                     .get("det"), "plan.components.det")
    model_rel = component.get("artifact_path")
    _require(type(model_rel) is str and model_rel.startswith("det/")
             and len(model_rel.split("/")) == 2, "missing bounded Det artifact path")
    filename = composite._safe_name(model_rel.split("/")[1])
    _require(filename.casefold().endswith(".cpu.dll"),
             "selected native Det artifact must be a CPU .cpu.dll")
    model_path = bundle_root / "det" / filename
    actual_model_sha = _sha(component["artifact_sha256"], "plan Det module")
    model = _obj(record["model"], "native receipt.model")
    _keys(model, {"path", "sha256", "bytes", "inputs_sha256", "inputs_bytes"},
          "native receipt.model")
    _require(type(model.get("path")) is str and model["path"] == str(model_path),
             "native captured module path differs from selected DRAFT Det artifact")
    _require(_sha(model["sha256"], "native model SHA") == actual_model_sha,
             "native captured Det module SHA differs from DRAFT")
    _require(type(model.get("bytes")) is int
             and 0 < model["bytes"] <= _MODEL_LIMIT,
             "native captured Det module byte count unsupported")
    _, digest, count = _member(
        bundle_root / "det", filename, "selected native model",
        limit=_MODEL_LIMIT, expected_bytes=model["bytes"])
    _require((digest, count) == (actual_model_sha, model["bytes"]),
             "selected native model no longer matches staged Det artifact bytes")
    sidecar_name = filename + ".inputs"
    _require(type(model.get("inputs_bytes")) is int
             and 0 < model["inputs_bytes"] <= _SIDECAR_LIMIT,
             "native .inputs size exceeds Pascal producer bound")
    sidecar_path, sidecar_sha, sidecar_bytes = _member(
        bundle_root / "det", sidecar_name, "selected native .inputs",
        limit=_SIDECAR_LIMIT, expected_bytes=model["inputs_bytes"])
    _require(sidecar_sha == _sha(model["inputs_sha256"], "native sidecar sha"),
             "native model .inputs SHA mismatch")
    plan_asset = {item["path"]: item for item in plan["assets"]}.get(
        "det/" + sidecar_name)
    _require(type(plan_asset) is dict and
             _sha(plan_asset.get("sha256"), "staged .inputs") == sidecar_sha
             and plan_asset.get("bytes") == sidecar_bytes,
             "native .inputs is not the exact hashed staged source sidecar")
    with sidecar_path.open("rb") as source:
        _require(os.fstat(source.fileno()).st_size == sidecar_bytes,
                 "native .inputs size changed after SHA")
        raw_data = source.read(_SIDECAR_LIMIT + 1)
    _require(len(raw_data) == sidecar_bytes
             and hashlib.sha256(raw_data).hexdigest() == sidecar_sha,
             "native .inputs changed on second read")
    canonical = "0\t%s\t%s" % (
        input_name, ",".join(str(x) for x in input_shape))
    _require(raw_data in (canonical.encode("ascii"),
                          (canonical + "\n").encode("ascii"),
                          (canonical + "\r\n").encode("ascii")),
             "sidecar cannot be parsed by exact native Pascal single-input grammar")
    return actual_model_sha


def _source_image(record):
    item = _obj(record["image"], "native receipt.image")
    _keys(item, {"path", "sha256", "bytes", "width", "height"}, "image")
    _require(type(item["path"]) is str and item["path"].lower().endswith(".png"),
             "Pascal source image must be an explicitly selected RGB8 PNG")
    image_path = raw._absolute(item["path"], "native source image")
    _require(str(image_path) == item["path"],
             "native source image path must be canonical")
    width, height, count = item["width"], item["height"], item["bytes"]
    _require(type(width) is int and type(height) is int
             and 1 <= width <= 16384 and 1 <= height <= 16384
             and width * height <= _PNG_PIXELS,
             "native source PNG dimensions exceed Pascal bound")
    _require(type(count) is int and 0 < count <= _PNG_LIMIT,
             "native source PNG file exceeds Pascal byte quota")
    expected_sha = _sha(item["sha256"], "native source image SHA")
    _, actual, _ = _member(image_path.parent, image_path.name,
                          "native original RGB8 image", limit=_PNG_LIMIT,
                          expected_bytes=count)
    _require(expected_sha == actual, "native source image SHA changed")
    oracle._validated_source_png(image_path, width, height,
                                 expected_sha=expected_sha)
    _, post_sha, _ = _member(image_path.parent, image_path.name,
                           "native source image second SHA", limit=_PNG_LIMIT,
                           expected_bytes=count)
    _require(post_sha == expected_sha, "native image changed during PNG admission")
    return expected_sha


def _tensor(root, record, *, role, input_shape=None):
    item = _obj(record, "native " + role)
    _keys(item, {"file", "dtype", "layout", "shape", "bytes", "sha256"},
          "native " + role)
    name = "det-input.f32le" if role == "input" else "det-output.f32le"
    _require(item["file"] == name and item["dtype"] == "float32le"
             and item["layout"] == "NCHW",
             "Pascal Det raw tensor filename/dtype/layout is unsupported")
    shape = item["shape"]
    _require(type(shape) is list and len(shape) == 4
             and all(type(v) is int and 1 <= v <= 4096 for v in shape),
             "native raw tensor shape must be four bounded positive dimensions")
    if role == "input":
        _require(shape == input_shape and shape[:2] == [1, 3],
                 "native preprocessed input shape differs from .inputs/plan")
    else:
        _require(shape[:2] == [1, 1],
                 "native Det probability map must be [1,1,H,W]")
    elements = 1
    for axis in shape:
        elements *= axis
        _require(elements <= _TENSOR_LIMIT,
                 "native raw tensor exceeds Pascal element quota")
    byte_size = elements * 4
    _require(type(item["bytes"]) is int and item["bytes"] == byte_size,
             "native raw tensor byte declaration conflicts with shape")
    expected_sha = _sha(item["sha256"], "native " + role + " sha")
    path, actual_sha, _ = _member(
        root, name, "native " + role + " f32le", limit=_TENSOR_LIMIT * 4,
        expected_bytes=byte_size)
    _require(actual_sha == expected_sha, "native " + role + " tensor SHA changed")
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size == byte_size,
                 "native raw tensor grew during finiteness scan")
        remaining = byte_size
        check_sha = hashlib.sha256()
        while remaining:
            block = stream.read(min(64 * 1024, remaining))
            _require(bool(block) and len(block) % 4 == 0,
                     "truncated/unaligned float32 tensor")
            remaining -= len(block)
            check_sha.update(block)
            for (number,) in struct.iter_unpack("<f", block):
                _require(abs(number) != float("inf") and number == number,
                         "native raw tensor has non-finite f32 values")
        _require(not stream.read(1), "native raw tensor grew during finiteness scan")
    _require(check_sha.hexdigest() == expected_sha,
             "native raw tensor changed after SHA admission")
    _, final_sha, _ = _member(
        root, name, "native " + role + " second SHA",
        limit=_TENSOR_LIMIT * 4, expected_bytes=byte_size)
    _require(final_sha == expected_sha,
             "native raw tensor changed during descriptor construction")
    return {"path": str(path), "dtype": "float32le", "layout": "NCHW",
            "shape": list(shape), "count": elements,
            "bytes": byte_size, "sha256": actual_sha}


def _inspect(bundle, capture_dir, stage):
    _require(type(stage) is str and stage == "det",
             "native Pascal raw capture is strictly Det-only, not Rec")
    bundle_root, plan, plan_sha = oracle._validate_plan(bundle)
    expected_name, expected_shape = ort_bridge._input_contract(
        plan, bundle_root, "det")
    _require(_INPUT_NAME.fullmatch(expected_name) is not None,
             "DRAFT input name cannot be parsed by native Pascal")
    _preprocess_contract(plan)
    runtime_sha = _runtime_contract(plan)
    capture_root = raw._absolute(capture_dir, "native Det capture directory")
    receipt, capture_sha, capture_size = _receipt(capture_root)
    _require(receipt["input_name"] == expected_name,
             "native input semantic name differs from DRAFT Det")
    _require(type(receipt["runtime_observed_path"]) is str and
             len(receipt["runtime_observed_path"]) <= 4096 and
             os.path.isabs(receipt["runtime_observed_path"]) and
             Path(receipt["runtime_observed_path"]).name == "tvm_runtime.dll",
             "native receipt lacks bounded observed TVM runtime path")
    observed_runtime = _obj(
        receipt["runtime_selected_files_observed_sha256"],
        "native self-observed runtime SHA")


    allowed = set(runtime_sha)
    actual = set(observed_runtime)
    _require(actual in (allowed, allowed | {"crosscv.dll"}),
             "unexpected/missing captured runtime or crosscv.dll keys")
    _require({key: observed_runtime[key] for key in runtime_sha} == runtime_sha,
             "native self-observed runtime SHA disagrees with hashed selected CPU kit")
    crosscv = None
    if "crosscv.dll" in observed_runtime:
        crosscv = _sha(observed_runtime["crosscv.dll"],
                       "native self-observed crosscv.dll SHA")
    model_sha = _model_and_sidecar(plan, bundle_root, receipt,
                                   expected_name, expected_shape)
    image_sha = _source_image(receipt)
    inp = _tensor(capture_root, receipt["input"],
                  role="input", input_shape=expected_shape)
    out = _tensor(capture_root, receipt["raw_output"], role="output")
    _, post_sha, _ = _member(capture_root, "det-capture.json",
                            "native receipt final SHA",
                            limit=_JSON_LIMIT, expected_bytes=capture_size)
    _require(post_sha == capture_sha, "native receipt changed during inspection")
    raw._check_private_root(capture_root)
    final_root, _, final_plan_sha = oracle._validate_plan(bundle_root)
    _require(final_root == bundle_root and final_plan_sha == plan_sha,
             "DRAFT composite model assets/plan changed during bridge inspection")
    return {
        "status": "NOT_CHECKED",
        "source_kind": "SUPPLIED_PASCAL_DET_RAW_CAPTURE",
        "producer_status": _STATUS,
        "producer_authenticity": "NOT_ATTESTED",
        "native_process_provenance": "NOT_ATTESTED",
        "native_runtime_provenance": "NOT_ATTESTED",
        "crosscv_self_observed_sha256": crosscv,
        "acceptance_status": "NOT_CHECKED",
        "composite_verification": "NOT_CHECKED",
        "native_composite_oracle": "NOT_RUN",
        "native_recognizer": "NOT_RUN",
        "native_ctc": "NOT_RUN",
        "human_ground_truth": "NOT_ATTESTED",
        "independent_onnx_preprocessing": "NOT_CHECKED",
        "facade_manifest_issued": False,
        "ready_issued": False,
        "stage": "det",
        "plan_sha256": plan_sha,
        "capture_sha256": capture_sha,
        "model_sha256": model_sha,
        "source_image_sha256": image_sha,
        "input": dict(inp, name=expected_name),
        "raw_output": out,
        "missing_obligations": [
            "Independently witnessed native Pascal CPU process/exit/TVM provenance",
            "Original ONNX Runtime Det execution on the identical PNG/preprocessing",
            "Det candidate boxes, Rec crop/raw logits/CTC, dictionary/blank mapping",
            "Independent human-annotated pixel boxes and text ground truth",
            "Measured all-stage OCR numeric and human root acceptance",
        ],
        "provenance_limit": (
            "Only a read-only integrity check of SELF-DECLARED preexisting Pascal "
            "Det receipt and existing bytes against a DRAFT plan. This does NOT "
            "attest actual native execution, PNG-to-tensor equivalence, independent "
            "ONNX inference, Rec/CTC, process exit or human ground truth. Paths "
            "must be rehashed by any future consumer; no manifest/READY issued."),
    }


def inspect_native_det_capture(bundle, capture_dir, stage="det") -> dict:
    try:
        return _inspect(bundle, capture_dir, stage)
    except NativeDetCaptureRefused:
        raise
    except (oracle.InvalidEvidence, oracle.MissingEvidence,
            composite.CompositeContractError, raw.CaptureRefused,
            OSError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, OverflowError, struct.error,
            RecursionError) as exc:
        raise NativeDetCaptureRefused(
            "Pascal Det raw advisory bridge refused: %s" % exc) from exc


__all__ = ["NativeDetCaptureRefused", "inspect_native_det_capture"]
