# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


class HR800ReferenceError(ValueError):
    pass


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_INPUT = [1, 3, 800, 800]
_OUTPUT = [1, 1, 800, 800]


def _file(path, expected, role):
    if not isinstance(path, (str, Path)) or not os.path.isabs(os.fspath(path)):
        raise HR800ReferenceError(role + " must be an absolute pinned file")
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise HR800ReferenceError(role + " must have exact lowercase SHA256")
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise HR800ReferenceError(role + " must be a regular non-symlink file")
    with value.open("rb") as stream:
        hashed = hashlib.sha256()
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(part)
        digest = hashed.hexdigest()
    if digest != expected:
        raise HR800ReferenceError(role + " differs from caller-pinned SHA256")
    return value


def _abi(session, role):
    ins, outs = session.get_inputs(), session.get_outputs()
    if (len(ins) != 1 or ins[0].name != "x" or
            ins[0].type != "tensor(float)" or ins[0].shape != _INPUT or
            len(outs) != 1 or outs[0].name != "mask" or outs[0].type not in
            ("tensor(float16)", "tensor(float)", "tensor(double)") or
            outs[0].shape != _OUTPUT):
        raise HR800ReferenceError(role + " violates HR800 input/mask ONNX ABI")
    return ins[0].name, outs[0].name


def build_basic_reference(original_onnx, original_sha256, sample_npy,
                          sample_sha256, output_directory, *, max_abs=1e-6,
                          expected_optimized_sha256=None,
                          expected_optimized_bytes=None):
    if expected_optimized_sha256 is not None and (
            not isinstance(expected_optimized_sha256, str) or
            not _SHA.fullmatch(expected_optimized_sha256)):
        raise HR800ReferenceError("expected optimized SHA must be lowercase SHA256")
    if expected_optimized_bytes is not None and (
            type(expected_optimized_bytes) is not int or expected_optimized_bytes <= 0):
        raise HR800ReferenceError("expected optimized byte size must be positive")
    if not isinstance(max_abs, (int, float)) or not 0 <= max_abs <= 1e-5:
        raise HR800ReferenceError("max_abs must be finite and within [0,1e-5]")
    import math
    if not math.isfinite(max_abs):
        raise HR800ReferenceError("max_abs must be finite")
    model = _file(original_onnx, original_sha256, "original ONNX")
    sample = _file(sample_npy, sample_sha256, "sample NPY")
    if (not isinstance(output_directory, (str, Path)) or
            not os.path.isabs(os.fspath(output_directory))):
        raise HR800ReferenceError("output directory must be absolute and new")
    target = Path(output_directory)
    if target.resolve() in (model.resolve(), sample.resolve()):
        raise HR800ReferenceError("output cannot overlap selected input")

    import numpy as np
    import onnxruntime as ort
    tensor = np.load(sample, allow_pickle=False, mmap_mode="r")
    if tensor.shape != tuple(_INPUT) or tensor.dtype != np.dtype("float32"):
        raise HR800ReferenceError("sample must be exact float32 [1,3,800,800]")
    if not np.isfinite(tensor).all():
        raise HR800ReferenceError("sample contains nonfinite input pixels")


    target.mkdir(parents=False, exist_ok=False)
    optimized = target / "hr800_ort_basic.onnx"
    receipt = target / "hr800_ort_basic_reference.json"
    original_options = ort.SessionOptions()
    original_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    original = ort.InferenceSession(str(model), sess_options=original_options,
                                    providers=["CPUExecutionProvider"])
    original_input, original_output = _abi(original, "original")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(optimized)

    optimized_runtime = ort.InferenceSession(str(model), sess_options=opts,
                                             providers=["CPUExecutionProvider"])
    _abi(optimized_runtime, "optimized runtime")
    if not optimized.is_file() or optimized.stat().st_size == 0:
        raise HR800ReferenceError("ORT did not create optimized ONNX sidefile")

    reload_opts = ort.SessionOptions()
    reload_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reloaded = ort.InferenceSession(str(optimized), sess_options=reload_opts,
                                    providers=["CPUExecutionProvider"])
    optimized_input, optimized_output = _abi(reloaded, "serialized optimized")
    before = original.run([original_output], {original_input: tensor})[0]
    after = reloaded.run([optimized_output], {optimized_input: tensor})[0]
    if (before.shape != tuple(_OUTPUT) or after.shape != tuple(_OUTPUT) or
            not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise HR800ReferenceError("reference output geometry or finiteness differs")
    deviation = float(np.max(np.abs(before.astype(np.float64) -
                                    after.astype(np.float64))))
    if deviation > max_abs:
        raise HR800ReferenceError("optimized HR800 output exceeds sample max_abs")

    _file(model, original_sha256, "original ONNX")
    _file(sample, sample_sha256, "sample NPY")
    optimized_hash = hashlib.sha256()
    with optimized.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            optimized_hash.update(part)
    actual_optimized_sha256 = optimized_hash.hexdigest()
    actual_optimized_bytes = optimized.stat().st_size
    if (expected_optimized_sha256 is not None and
            actual_optimized_sha256 != expected_optimized_sha256):
        raise HR800ReferenceError("optimized ONNX differs from pinned SHA256")
    if (expected_optimized_bytes is not None and
            actual_optimized_bytes != expected_optimized_bytes):
        raise HR800ReferenceError("optimized ONNX differs from pinned byte size")
    evidence = {
        "schema": "crosspascal.hr800.ort-basic-reference.v1",
        "status": "ONE_PINNED_SAMPLE_REFERENCE_ONLY",
        "original_sha256": original_sha256,
        "sample_sha256": sample_sha256,
        "optimized_sha256": actual_optimized_sha256,
        "optimized_bytes": actual_optimized_bytes,
        "optimized_sha256_pin_checked": expected_optimized_sha256 is not None,
        "optimized_bytes_pin_checked": expected_optimized_bytes is not None,
        "input_name": original_input, "output_name": original_output,
        "input_shape": _INPUT, "output_shape": _OUTPUT,
        "optimization": "ORT_ENABLE_BASIC",
        "provider": "CPUExecutionProvider",
        "max_abs_difference": deviation, "max_abs_allowed": float(max_abs),
        "bit_exact": bool(np.array_equal(before, after)),
        "native_uat": "NOT_CHECKED", "whole_domain_parity": "NOT_CHECKED",
    }
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return evidence
