# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""READ-ONLY paired scalar check of supplied original ORT and Pascal Det tensors.

This deliberately compares only ONE preexisting Det raw probability map.
Matching files cannot independently attest ORT/Pascal process, runtime origin,
image preprocessing or Det, Rec and CTC/human ground truth. No model is launched,
no evidence is written, and neither Facade manifests nor READY can be issued.

Both model SHA-256 pins must be selected outside either producer receipt.
Original model copy + preprocessed input + complete ORT output inventory are
checked by the existing strict original raw producer receipt verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from . import ocr_capture_bridge as ort_bridge
from . import ocr_composite_oracle as oracle
from . import ocr_native_det_capture_bridge as native
from . import ocr_ort_raw_capture as raw


class DetPairwiseRefused(ValueError):
    """Supplied Det-only capture pair is contradictory, missing or unsafe."""


_RECIPE_MAX_BYTES = 2 * 1024 * 1024
_RUNTIME_NAMES = {
    "tvm_runtime.dll", "tvm_ffi.dll", "tvm_runtime_extra.dll",
}
_PINNED_CROSSCV = "crosscv.dll"


def _require(ok, text):
    if not ok:
        raise DetPairwiseRefused(text)


def _object(value, label):
    _require(type(value) is dict, label + " must be an object")
    return value


def _read_selected_json(path, label, limit):
    digest, size = raw._sha_file(path, label, limit=limit)
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size == size,
                 label + " byte length changed before second read")
        data = stream.read(limit + 1)
    _require(len(data) == size and hashlib.sha256(data).hexdigest() == digest,
             label + " changed between admitted byte reads")
    return _object(raw._strict_json(data), label), digest, size


def _selected_native_model(model_arg, receipt, expected_sha):
    root = raw._absolute(model_arg, "selected native Det CPU model")
    _require(root.name.lower().endswith(".cpu.dll"),
             "selected native model must be an exact CPU .cpu.dll")
    model = _object(receipt["model"], "native receipt.model")
    _require(model.get("path") == str(root),
             "native receipt model path differs from explicit selected module")
    expected_sha = oracle._sha(expected_sha, "selected native expected SHA")
    _require(model.get("sha256") == expected_sha,
             "native receipt model SHA disagrees with independent caller pin")
    _require(type(model.get("bytes")) is int
             and 0 < model["bytes"] <= native._MODEL_LIMIT,
             "native selected module has invalid/oversized declared bytes")
    measured_sha, measured_bytes = raw._sha_file(
        root, "selected native Det model", limit=native._MODEL_LIMIT,
        expected_bytes=model["bytes"])
    _require((measured_sha, measured_bytes) == (expected_sha, model["bytes"]),
             "selected native Det model does not match declared and expected SHA")
    return root, expected_sha


def _selected_sidecar(path, receipt):
    model = _object(receipt["model"], "native receipt.model")
    sidecar = Path(str(path) + ".inputs")
    _require(type(model.get("inputs_bytes")) is int
             and 0 < model["inputs_bytes"] <= native._SIDECAR_LIMIT,
             "selected .inputs exceeds native source size cap")
    digest, count = raw._sha_file(
        sidecar, "selected native .inputs",
        limit=native._SIDECAR_LIMIT, expected_bytes=model["inputs_bytes"])
    _require(model.get("inputs_sha256") == digest,
             "native receipt .inputs SHA differs from independently read sidecar")
    name = receipt.get("input_name")
    _require(type(name) is str and native._INPUT_NAME.fullmatch(name) is not None,
             "native Pascal input name lacks exact bounded single-input grammar")
    shape = _object(receipt["input"], "native input receipt").get("shape")
    _require(type(shape) is list and len(shape) == 4
             and shape[:2] == [1, 3]
             and all(type(axis) is int and 1 <= axis <= 4096 for axis in shape)
             and shape[0] * shape[1] * shape[2] * shape[3] <= raw.MAX_INPUT_ELEMENTS,
             "native selected shape must be bounded static [1,3,H,W]")
    with sidecar.open("rb") as stream:
        payload = stream.read(native._SIDECAR_LIMIT + 1)
    _require(len(payload) == count and hashlib.sha256(payload).hexdigest() == digest,
             "native .inputs changed after SHA")
    line = ("0\t" + name + "\t" + ",".join(map(str, shape))).encode("ascii")
    _require(payload in (line, line + b"\n", line + b"\r\n"),
             "native .inputs cannot be parsed by actual Pascal single-row grammar")
    return name, list(shape), sidecar, digest, count


def _selected_original_source(model_arg, expected_sha, name, shape,
                              native_model):
    source = raw._absolute(model_arg, "independently selected original ONNX")
    _require(source.suffix.lower() == ".onnx" and source != native_model,
             "independent original must be a distinct ONNX source")
    expected_sha = oracle._sha(expected_sha, "independently selected original ONNX SHA")
    actual, size = raw._sha_file(
        source, "independently selected original ONNX",
        limit=raw.MAX_MODEL_BYTES)
    _require(actual == expected_sha,
             "original ONNX bytes do not match independent selected SHA")


    recipe_path = native_model.parent / "recipe.json"
    recipe, recipe_sha, recipe_bytes = _read_selected_json(
        recipe_path, "native source recipe", _RECIPE_MAX_BYTES)
    _require(recipe.get("target") == "cpu" and
             recipe.get("source_hash") == expected_sha and
             recipe.get("shape") == name + ":" + ",".join(map(str, shape)),
             "selected TVM recipe original source hash/input/CPU contract differs")
    return source, expected_sha, size, recipe_path, recipe_sha, recipe_bytes


def _runtime_kit(receipt):
    root_str = receipt.get("runtime_observed_path")
    _require(type(root_str) is str
             and 0 < len(root_str) <= 4096,
             "native reported runtime path is missing or excessive")
    root = raw._absolute(root_str, "selected runtime DLL path")
    _require(root.name == "tvm_runtime.dll" and str(root) == root_str,
             "native reported TVM runtime path is not exact/canonical")
    expected = _object(
        receipt.get("runtime_selected_files_observed_sha256"),
        "native selected runtime SHA")
    observed = set(expected)


    _require(observed in (_RUNTIME_NAMES, _RUNTIME_NAMES | {_PINNED_CROSSCV}),
             "native CPU runtime must declare exact three TVM DLLs plus "
             "optional separately pinned crosscv.dll")
    checked = []
    for name in sorted(_RUNTIME_NAMES):
        digest = oracle._sha(expected[name], "native selected runtime SHA")
        path = root.parent / name
        measured, size = raw._sha_file(
            path, "selected runtime " + name,
            limit=native._MODEL_LIMIT)
        _require(measured == digest,
                 "native runtime SHA is not bound to the actual selected DLL")
        checked.append((path, digest, size))
    if _PINNED_CROSSCV in observed:
        crosscv_path = root.parent.parent / _PINNED_CROSSCV
        _require(root.parent.name.lower() == 'tvm',
                 "CrossCV receipt requires a genuine EXE/tvm runtime path")
        crosscv_digest = oracle._sha(
            expected[_PINNED_CROSSCV], "native selected CrossCV SHA")
        actual_digest, count = raw._sha_file(
            crosscv_path, "native executable-adjacent crosscv.dll",
            limit=native._MODEL_LIMIT)
        _require(actual_digest == crosscv_digest,
                 "native CrossCV DLL bytes differ from captured source")
        checked.append((crosscv_path, crosscv_digest, count))
    _require(not os.path.lexists(root.parent / "tvm_runtime_cuda.dll"),
             "CPU Det runtime directory unexpectedly contains a CUDA DLL")
    return checked


def _tensor_for_oracle(row, path):
    shape = row["shape"]
    count = 1
    for axis in shape:
        count *= axis
    return {"path": Path(path), "shape": list(shape),
            "sha256": row["sha256"], "count": count,
            "bytes": row["bytes"]}


def _inspect(native_capture_dir, ort_capture_dir, *,
             native_model, original_onnx,
             expected_native_sha256, expected_original_sha256,
             output_name):
    _require(type(output_name) is str and 0 < len(output_name) <= 256
             and not any(ord(ch) < 32 for ch in output_name),
             "exact semantic ORT output name is mandatory")

    native_root = raw._absolute(native_capture_dir, "Pascal Det capture")
    native_receipt, native_capture_sha, capture_size = native._receipt(native_root)
    selected_model, native_model_sha = _selected_native_model(
        native_model, native_receipt, expected_native_sha256)
    input_name, input_shape, sidecar_path, sidecar_sha, sidecar_bytes = (
        _selected_sidecar(selected_model, native_receipt))
    original, original_sha, original_bytes, recipe_path, recipe_sha, recipe_bytes = (
        _selected_original_source(original_onnx, expected_original_sha256,
                                  input_name, input_shape, selected_model))
    runtime = _runtime_kit(native_receipt)
    image_sha = native._source_image(native_receipt)
    native_input = native._tensor(native_root, native_receipt["input"],
                                  role="input", input_shape=input_shape)
    native_output = native._tensor(native_root, native_receipt["raw_output"],
                                   role="output")

    ort_root = raw._absolute(ort_capture_dir, "original ORT capture")
    ort_receipt, ort_capture_sha = raw._verify_capture(
        ort_root, original_sha, native_input["sha256"], input_shape)
    ort_bridge._receipt_grammar(ort_receipt)
    ort_input = _object(ort_receipt.get("input"), "ORT input")
    _require(ort_input.get("name") == input_name
             and ort_input.get("shape") == input_shape
             and ort_input.get("sha256") == native_input["sha256"],
             "ORT input is not the exact same name/shape/bytes as native Pascal")
    candidates = [row for row in ort_receipt["outputs"]
                  if row["name"] == output_name]
    _require(len(candidates) == 1,
             "ORT selected probability map output name is absent or ambiguous")
    chosen = candidates[0]
    _require(chosen.get("shape") == native_output["shape"]
             and chosen["shape"][:2] == [1, 1],
             "ORT Det probability map shape differs from actual Pascal output")
    result = oracle._compare_tensors(
        _tensor_for_oracle(chosen, ort_root / chosen["file"]),
        _tensor_for_oracle(native_output, native_root / "det-output.f32le"),
        "original ORT versus Pascal Det raw output",
        atol=oracle.OUTPUT_ATOL, rtol=oracle.OUTPUT_RTOL,
        nonvacuous=True)


    for label, path, digest, size, cap in [
        ("native model", selected_model, native_model_sha,
         native_receipt["model"]["bytes"], native._MODEL_LIMIT),
        ("native .inputs", sidecar_path, sidecar_sha,
         sidecar_bytes, native._SIDECAR_LIMIT),
        ("original ONNX", original, original_sha,
         original_bytes, raw.MAX_MODEL_BYTES),
        ("selected source recipe", recipe_path, recipe_sha,
         recipe_bytes, _RECIPE_MAX_BYTES),
    ] + [("runtime", path, digest, count, native._MODEL_LIMIT)
         for path, digest, count in runtime]:
        _require(raw._sha_file(path, label + " final identity",
                              limit=cap, expected_bytes=size) == (digest, size),
                 label + " changed during Det pairwise comparison")
    _require(native._source_image(native_receipt) == image_sha,
             "source image changed during Det scalar comparison")
    _require(native._tensor(native_root, native_receipt["input"],
                            role="input", input_shape=input_shape)["sha256"]
             == native_input["sha256"], "native input changed during comparison")
    _require(native._tensor(native_root, native_receipt["raw_output"],
                            role="output")["sha256"] == native_output["sha256"],
             "native output changed during comparison")
    _require(raw._verify_capture(ort_root, original_sha,
                                 native_input["sha256"], input_shape)[1]
             == ort_capture_sha, "ORT capture changed during comparison")
    _require(raw._sha_file(native_root / "det-capture.json",
                           "native receipt final identity",
                           limit=native._JSON_LIMIT,
                           expected_bytes=capture_size)[0] == native_capture_sha,
             "native receipt changed during comparison")

    return {
        "status": "NOT_CHECKED",
        "numeric_status": "NUMERIC_MATCH_SUPPLIED_DET_ONLY",
        "producer_authenticity": "NOT_ATTESTED",
        "process_provenance": "NOT_ATTESTED",
        "native_runtime_provenance": "NOT_ATTESTED",
        "human_ground_truth": "NOT_ATTESTED",
        "native_composite_oracle": "NOT_RUN",
        "native_recognizer": "NOT_RUN",
        "native_ctc": "NOT_RUN",
        "independent_png_preprocessing": "NOT_CHECKED",
        "facade_manifest_issued": False,
        "ready_issued": False,
        "selected_native_model_sha256": native_model_sha,
        "selected_original_onnx_sha256": original_sha,
        "selected_recipe_sha256": recipe_sha,
        "native_capture_sha256": native_capture_sha,
        "ort_capture_sha256": ort_capture_sha,
        "source_png_sha256": image_sha,
        "native_input_sha256": native_input["sha256"],
        "ort_input_sha256": ort_input["sha256"],
        "output_name": output_name,
        "det_output_metrics": result,
        "missing_obligations": [
            "Independent ORT/native process execution/exit and vendor runtime attestation",
            "Original PNG-to-input preprocessing equivalence (measured pixels)",
            "Det postprocessing candidate boxes and independently labelled human boxes",
            "Per-crop original ORT/native Rec logits, dictionary and CTC acceptance",
            "True Det, Rec and CTC plus independent human text GT and root Facade acceptance",
        ],
        "provenance_limit": (
            "This is a read-only scalar comparison of supplied SHA-checked files "
            "and two caller-selected model pins. It does NOT attest producer "
            "execution, original software author/runtime signature, image "
            "preprocessing, OCR recognition, human accuracy or manifest READY."),
    }


def inspect_det_pair(native_capture_dir, ort_capture_dir, *,
                     native_model, original_onnx,
                     expected_native_sha256, expected_original_sha256,
                     output_name) -> dict:
    """Inspect existing capture bytes only; no process, output or READY writes."""
    try:
        return _inspect(
            native_capture_dir, ort_capture_dir,
            native_model=native_model, original_onnx=original_onnx,
            expected_native_sha256=expected_native_sha256,
            expected_original_sha256=expected_original_sha256,
            output_name=output_name)
    except DetPairwiseRefused:
        raise
    except (raw.CaptureRefused, native.NativeDetCaptureRefused,
            ort_bridge.CaptureBridgeRefused,
            oracle.InvalidEvidence, oracle.MissingEvidence,
            OSError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, RecursionError, OverflowError) as exc:
        raise DetPairwiseRefused(
            "supplied ORT/native Det pairwise comparison refused: %s" % exc) from exc


def main(argv=None) -> int:
    """Public read-only supplied-capture checker; even scalar match exits 2."""
    parser = argparse.ArgumentParser(
        prog="pascalvision-convert ocr-det-pairwise",
        description=("SHA-pin and compare every supplied original ORT/native "
                     "Pascal Det raw scalar, read-only. Never claims process "
                     "attestation, Rec/CTC/human GT, Facade/READY or VERIFIED."))
    parser.add_argument("--native-capture", required=True,
                        help="existing Pascal raw Det-only capture directory")
    parser.add_argument("--ort-capture", required=True,
                        help="existing original ORT raw CPU capture directory")
    parser.add_argument("--native-model", required=True,
                        help="independently selected compiled CPU .cpu.dll")
    parser.add_argument("--original-onnx", required=True,
                        help="independently selected original ONNX file")
    parser.add_argument("--native-sha256", required=True,
                        help="separately approved compiled CPU DLL SHA-256")
    parser.add_argument("--original-sha256", required=True,
                        help="separately approved original ONNX SHA-256")
    parser.add_argument("--output-name", required=True,
                        help="exact ORT output semantic name, never guess an index")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if type(exc.code) is int and exc.code == 0:
            return 0


        print(json.dumps({
            "status": "REFUSED",
            "acceptance_status": "NOT_CHECKED",
            "native_composite_oracle": "NOT_RUN",
            "human_ground_truth": "NOT_ATTESTED",
            "facade_manifest_issued": False,
            "ready_issued": False,
            "reason": "invalid or incomplete ocr-det-pairwise CLI arguments",
        }, ensure_ascii=False))
        return 1
    try:
        report = inspect_det_pair(
            args.native_capture, args.ort_capture,
            native_model=args.native_model,
            original_onnx=args.original_onnx,
            expected_native_sha256=args.native_sha256,
            expected_original_sha256=args.original_sha256,
            output_name=args.output_name)
    except DetPairwiseRefused as exc:
        print(json.dumps({
            "status": "REFUSED",
            "acceptance_status": "NOT_CHECKED",
            "native_composite_oracle": "NOT_RUN",
            "human_ground_truth": "NOT_ATTESTED",
            "facade_manifest_issued": False,
            "ready_issued": False,
            "reason": str(exc),
        }, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 2


__all__ = ["DetPairwiseRefused", "inspect_det_pair", "main"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
