# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile

from . import targets


SCHEMA = "crosspascal.omdet.candidate-bundle.v1"
VERSION = 1
INPUT_ORDER = ("pixel_values", "class_features", "task_features", "task_mask")
OUTPUT_ORDER = ("boxes", "scores")
INPUT_DTYPES = ("float32", "float32", "float32", "int64")
FILE_ROLES = (
    "source_onnx", "export_contract", "feature_manifest",
    "hf_processor_config", "hf_model_config", "compiled_module",
    "compiled_inputs",
)
RUNTIME_BASE = ("tvm_runtime", "tvm_runtime_extra", "tvm_ffi")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_MAX_JSON = 131072
_MAX_INPUT_BYTES = 64 * 1048576
_VERIFICATION = {
    "tier": "NOT_CHECKED",
    "native_target_uat": "NOT_RUN",
    "independent_numeric_oracle": "NOT_RUN",
    "pascal_facade": "DISABLED",
    "hf_preprocessing_oracle": "NOT_RUN",
    "redistribution": "NOT_CHECKED",
    "immutable_source_lock": "NOT_IMPLEMENTED",
}


def _exact_keys(mapping, names, label):
    if not isinstance(mapping, dict):
        raise ValueError("%s must be an object" % label)
    actual, needed = set(mapping), set(names)
    if actual != needed:
        raise ValueError("%s missing=%s unsupported=%s" %
                         (label, sorted(map(str, needed - actual)),
                          sorted(map(str, actual - needed))))


def _root(directory):
    directory = Path(directory).absolute()
    if not directory.is_dir():
        raise ValueError("candidate directory missing")


    for ancestor in (directory, *directory.parents):
        if (ancestor.is_symlink() or
                getattr(ancestor, "is_junction", lambda: False)()):
            raise ValueError("candidate root or parent is symlink/junction")
    return directory


def _local_file(root, filename):
    if (not isinstance(filename, str) or not filename or
            "\\" in filename or ":" in filename):
        raise ValueError("candidate file needs relative safe name")
    parts = PurePosixPath(filename).parts
    if (not parts or filename != "/".join(parts) or
            any(not _COMPONENT.fullmatch(part) or part in (".", "..")
                or part.endswith(".")
                for part in parts)):
        raise ValueError("unsafe candidate relative path: %r" % filename)
    current = root
    for part in parts:
        current = current / part
        if (current.is_symlink() or
                getattr(current, "is_junction", lambda: False)()):
            raise ValueError("candidate path contains symlink/junction: " + filename)
    if not current.is_file():
        raise ValueError("candidate file missing: " + filename)
    return current


def _pinned(root, pin, label, limit=None):
    _exact_keys(pin, ("file", "sha256"), label)
    if not isinstance(pin["sha256"], str) or not _SHA.fullmatch(pin["sha256"]):
        raise ValueError("%s requires lowercase SHA-256" % label)
    path = _local_file(root, pin["file"])
    stat_before = path.stat()
    if stat_before.st_size < 1 or (
            limit is not None and stat_before.st_size > limit):
        raise ValueError("%s empty or exceeds size limit" % label)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            byte_count += len(block)
            if limit is not None and byte_count > limit:
                raise ValueError("%s exceeds size limit" % label)
            digest.update(block)
    stat_after = path.stat()
    if (byte_count != stat_before.st_size or
            stat_after.st_size != stat_before.st_size or
            stat_after.st_mtime_ns != stat_before.st_mtime_ns or
            stat_after.st_ino != stat_before.st_ino):
        raise ValueError("%s source changed during inspection" % label)
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError("%s pinned SHA-256 mismatch" % label)
    return {"file": pin["file"], "bytes": byte_count,
            "sha256": digest.hexdigest()}


def _json(root, pin, label):
    record = _pinned(root, pin, label, _MAX_JSON)

    with _local_file(root, record["file"]).open("rb") as stream:
        payload = stream.read(_MAX_JSON + 1)
    if (len(payload) != record["bytes"] or
            hashlib.sha256(payload).hexdigest() != record["sha256"]):
        raise ValueError("%s JSON bytes changed after SHA-256 inspection" % label)

    def pairs_no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("%s contains duplicate JSON key %s" % (label, key))
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("%s contains nonfinite JSON token %s" % (label, token))))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("%s is not valid strict UTF-8 JSON: %s" %
                         (label, error)) from error
    if not isinstance(value, dict) or not value:
        raise ValueError("%s JSON must be a nonempty object" % label)
    return record, value


def _shape(data, rank, label):
    if (not isinstance(data, list) or len(data) != rank or
            any(type(dim) is not int or dim <= 0 for dim in data)):
        raise ValueError("%s needs rank%d positive integer shape" % (label, rank))
    return list(data)


def _feature_values_safe(path, dtype):
    fmt = "<q" if dtype == "int64le" else "<f"
    unit = 8 if dtype == "int64le" else 4
    has_active_mask = False
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            if len(block) % unit:
                raise ValueError("OmDet feature element alignment mismatch")
            for (value,) in struct.iter_unpack(fmt, block):
                if dtype == "int64le":
                    if value not in (0, 1):
                        raise ValueError("OmDet task_mask must contain 0/1 only")
                    has_active_mask |= bool(value)
                elif not math.isfinite(value):
                    raise ValueError("OmDet class/task feature contains NaN/Inf")
    if dtype == "int64le" and not has_active_mask:
        raise ValueError("OmDet task_mask must select at least one token")


def _inspect_four_inputs(root, contract, features, feature_manifest_file):
    if (contract.get("inputOrder") != list(INPUT_ORDER) or
            contract.get("outputOrder") != list(OUTPUT_ORDER) or
            features.get("schema") != "crosspascal.omdet.static-features.v1" or
            features.get("vm_input_order") != list(INPUT_ORDER)):
        raise ValueError("OmDet four-input exporter/feature schema does not match")
    _exact_keys(contract.get("inputs"), INPUT_ORDER, "export inputs")
    _exact_keys(contract.get("outputs"), OUTPUT_ORDER, "export outputs")
    _exact_keys(features.get("features"), INPUT_ORDER[1:], "feature assets")
    shapes, inputs = {}, []
    for i, name in enumerate(INPUT_ORDER):
        tensor = contract["inputs"][name]
        _exact_keys(tensor, ("shape", "dtype"), "input " + name)
        shape = _shape(tensor["shape"], 4 if i == 0 else 2 if i == 3 else 3,
                       "input " + name)
        if tensor["dtype"] != INPUT_DTYPES[i]:
            raise ValueError("OmDet input dtype mismatch: " + name)
        shapes[name] = shape
        inputs.append({"name": name, "shape": shape, "dtype": tensor["dtype"]})

    pixels, classes, tasks, mask = (shapes[name] for name in INPUT_ORDER)
    if (pixels[:2] != [1, 3] or classes[1] != 1 or tasks[1] != 1 or
            classes[2] != tasks[2] or mask != [1, tasks[0]]):
        raise ValueError("OmDet image/class/task/mask tensor shapes disagree")
    trace_bytes = math.prod(pixels) * 4
    for name in INPUT_ORDER[1:]:
        trace_bytes += math.prod(shapes[name]) * (8 if name == "task_mask" else 4)
    if trace_bytes > _MAX_INPUT_BYTES:
        raise ValueError("OmDet trace tensors exceed bounded 64MiB budget")
    image = features.get("image_input")
    if (not isinstance(image, dict) or image.get("name") != "pixel_values" or
            image.get("shape") != pixels or image.get("dtype") != "float32" or
            image.get("generated_from_fixture") is not False):
        raise ValueError("OmDet requires real nonfixture image profile")

    labels = features.get("class_labels_in_order")
    if (not isinstance(labels, list) or len(labels) != classes[0] or
            any(not isinstance(label, str) or not label.strip()
                for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError("OmDet class labels/count invalid or duplicated")
    outputs = []
    raw = features.get("expected_raw_outputs")
    _exact_keys(raw, OUTPUT_ORDER, "raw outputs")
    for name in OUTPUT_ORDER:
        tensor = contract["outputs"][name]
        _exact_keys(tensor, ("shape", "dtype"), "output " + name)
        shape = _shape(tensor["shape"], 3, "output " + name)
        if tensor["dtype"] != "float32" or shape[0] != 1:
            raise ValueError("OmDet outputs require batch1 float32")
        if name == "boxes":
            if shape[2] != 4:
                raise ValueError("OmDet box output must have four coordinates")
            queries = shape[1]
        elif shape != [1, queries, classes[0]]:
            raise ValueError("OmDet score query/class count mismatch")
        details = raw[name]
        key = "coordinates" if name == "boxes" else "semantics"
        meaning = ("normalized_cxcywh" if name == "boxes" else
                   "raw_logits_not_sigmoid")
        if (not isinstance(details, dict) or details.get("shape") != shape or
                details.get("dtype") != "float32" or details.get(key) != meaning):
            raise ValueError("OmDet pinned raw output semantics disagree: " + name)
        outputs.append({"name": name, "shape": shape, "dtype": "float32",
                        "semantics": meaning})

    assets = {}

    feature_folder = PurePosixPath(feature_manifest_file).parent
    for name in INPUT_ORDER[1:]:
        asset = features["features"][name]
        _exact_keys(asset, ("file", "shape", "dtype", "sha256", "bytes"),
                    "feature " + name)
        dtype = "int64le" if name == "task_mask" else "float32le"
        byte_count = math.prod(shapes[name]) * (8 if name == "task_mask" else 4)
        if (asset["shape"] != shapes[name] or asset["dtype"] != dtype or
                type(asset["bytes"]) is not int or asset["bytes"] != byte_count):
            raise ValueError("OmDet feature shape/dtype/byte mismatch: " + name)
        source_name = asset["file"]
        if (not isinstance(source_name, str) or
                not _COMPONENT.fullmatch(source_name) or
                source_name.endswith(".")):
            raise ValueError("OmDet feature pack requires safe bare filename")
        relative = str(feature_folder / source_name)
        measured = _pinned(root, {"file": relative,
                                   "sha256": asset["sha256"]},
                           "feature " + name, _MAX_INPUT_BYTES)
        if measured["bytes"] != byte_count:
            raise ValueError("OmDet feature bytes differ: " + name)
        _feature_values_safe(_local_file(root, relative), dtype)
        assets[name] = dict(measured, shape=shapes[name], dtype=dtype)
    return inputs, outputs, assets, list(labels)


def _check_native_inputs(root, record, declared):
    if record["bytes"] > 65536:
        raise ValueError("OmDet native .inputs is too large")
    text = _local_file(root, record["file"]).read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) != 4:
        raise ValueError("OmDet requires exactly FOUR native .inputs entries")
    for i, (line, tensor) in enumerate(zip(lines, declared)):
        actual = line.split("\t")
        expected = [str(i), tensor["name"], ",".join(map(str, tensor["shape"]))]
        if actual != expected:
            raise ValueError("OmDet .inputs mismatch at VM index %d" % i)


def _check_target_profile(profile):
    _exact_keys(profile, ("target_profile_version", "build_host",
                          "deployment", "reachability", "requested_tvm_target"),
                "OmDet candidate target_profile")
    if (type(profile["target_profile_version"]) is not int or
            profile["target_profile_version"] != targets.TARGET_PROFILE_VERSION):
        raise ValueError("OmDet candidate target_profile version unsupported")
    _exact_keys(profile["build_host"], ("os", "arch", "gpu_sm"),
                "OmDet target build_host")
    deployment = profile["deployment"]
    _exact_keys(deployment, ("os", "arch", "device", "sm") +
                (("driver_api",) if "driver_api" in deployment else ()),
                "OmDet target deployment")
    _exact_keys(profile["reachability"], ("level", "reason"),
                "OmDet target reachability")
    targets.validate_recorded_profile(profile)


def create_candidate_profile(bundle_root, *, pinned, target_profile, tvm_version):
    root = _root(bundle_root)
    _exact_keys(pinned, (*FILE_ROLES, "runtime_libraries"), "candidate pins")
    if not isinstance(tvm_version, str) or not tvm_version.strip():
        raise ValueError("candidate requires declared compiler TVM version")
    _check_target_profile(target_profile)
    destination = target_profile["deployment"]
    if destination["device"] not in ("cpu", "cuda"):
        raise ValueError("OmDet candidate supports CPU or CUDA only")
    if target_profile["reachability"]["level"] not in (
            targets.LOCAL, targets.COMPILE_ONLY):
        raise ValueError("candidate target requires unsupported native runner")
    if destination["device"] == "cuda" and not destination.get("sm"):
        raise ValueError("OmDet CUDA candidate requires explicit SM")
    if destination["device"] == "cpu" and (
            destination.get("sm") is not None or
            destination.get("driver_api") is not None):
        raise ValueError("CPU candidate must not request CUDA SM/driver")
    extensions = {"windows": ".dll", "linux": ".so", "darwin": ".dylib"}
    if destination["os"] not in extensions:
        raise ValueError("OmDet candidate target OS unknown")
    extension = extensions[destination["os"]]

    records, decoded = {}, {}
    for name in FILE_ROLES:
        if name in ("export_contract", "feature_manifest",
                    "hf_processor_config", "hf_model_config"):
            records[name], decoded[name] = _json(root, pinned[name], name)
        else:
            records[name] = _pinned(root, pinned[name], name)
    if not records["source_onnx"]["file"].endswith(".onnx"):
        raise ValueError("OmDet source must be a pinned ONNX file")
    if not records["compiled_module"]["file"].endswith(
            "." + destination["device"] + extension):
        raise ValueError("OmDet binary filename device/extension differs from "
                         "selected CPU/CUDA target")
    if records["compiled_inputs"]["file"] != (
            records["compiled_module"]["file"] + ".inputs"):
        raise ValueError("OmDet .inputs sidecar must match compiled module path")

    inputs, outputs, assets, labels = _inspect_four_inputs(
        root, decoded["export_contract"], decoded["feature_manifest"],
        records["feature_manifest"]["file"])
    _check_native_inputs(root, records["compiled_inputs"], inputs)

    required_runtime = (*RUNTIME_BASE, *(("tvm_runtime_cuda",)
                                         if destination["device"] == "cuda"
                                         else ()))
    _exact_keys(pinned["runtime_libraries"], required_runtime,
                "candidate TVM runtime inventory")
    libraries = {}
    for name in required_runtime:
        record = _pinned(root, pinned["runtime_libraries"][name],
                         "runtime " + name)
        basename = PurePosixPath(record["file"]).name
        if basename not in (name + extension, "lib" + name + extension):
            raise ValueError("runtime library logical name/extension mismatch: " +
                             name)
        libraries[name] = record

    return {
        "schema": SCHEMA, "schema_version": VERSION,
        "task_kind": "object_detection", "adapter": "omdet",
        "target_profile": json.loads(json.dumps(target_profile)),
        "source_onnx": records["source_onnx"],
        "export_contract": records["export_contract"],
        "feature_manifest": records["feature_manifest"],
        "hf_configs": {
            "processor": records["hf_processor_config"],
            "model": records["hf_model_config"],
        },
        "vm_inputs": inputs, "vm_outputs": outputs,
        "class_labels_in_order": labels, "trained_features": assets,
        "compiled_module": records["compiled_module"],
        "compiled_inputs": records["compiled_inputs"],
        "runtime": {
            "declared_tvm_version": tvm_version,
            "libraries": libraries,
            "binary_runtime_provenance": "NOT_CHECKED",
            "external_cuda_dependencies": "NOT_CHECKED",
        },

        "verification": dict(_VERIFICATION),
    }


def validate_candidate_profile(candidate, bundle_root, *, expected_pins=None):
    if not isinstance(candidate, dict):
        raise ValueError("OmDet candidate must be an object")
    if (candidate.get("schema") != SCHEMA or
            type(candidate.get("schema_version")) is not int or
            candidate["schema_version"] != VERSION):
        raise ValueError("unknown OmDet candidate schema/version")
    if candidate.get("verification") != _VERIFICATION:
        raise ValueError("candidate cannot claim VERIFIED or registered Facade")
    try:
        records = {
            "source_onnx": candidate["source_onnx"],
            "export_contract": candidate["export_contract"],
            "feature_manifest": candidate["feature_manifest"],
            "hf_processor_config": candidate["hf_configs"]["processor"],
            "hf_model_config": candidate["hf_configs"]["model"],
            "compiled_module": candidate["compiled_module"],
            "compiled_inputs": candidate["compiled_inputs"],
            "runtime_libraries": candidate["runtime"]["libraries"],
        }
        measured_pins = {}
        for role, record in records.items():
            if role == "runtime_libraries":
                measured_pins[role] = {
                    name: {"file": row["file"], "sha256": row["sha256"]}
                    for name, row in record.items()}
            else:
                measured_pins[role] = {
                    "file": record["file"], "sha256": record["sha256"]}
        if expected_pins is not None and measured_pins != expected_pins:
            raise ValueError("candidate pins differ from independently pinned source")
        reconstructed = create_candidate_profile(
            bundle_root, pinned=measured_pins,
            target_profile=candidate["target_profile"],
            tvm_version=candidate["runtime"]["declared_tvm_version"])
    except (KeyError, TypeError) as error:
        raise ValueError("OmDet candidate fields missing/malformed: %s" %
                         error) from error
    try:
        original_json = json.dumps(candidate, sort_keys=True, allow_nan=False)
        expected_json = json.dumps(reconstructed, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("OmDet candidate JSON not serializable: %s" %
                         error) from error
    if original_json != expected_json:
        raise ValueError("candidate differs from validated measured schema")
    return reconstructed


def write_candidate_profile(candidate, bundle_root):
    root = _root(bundle_root)
    validated = validate_candidate_profile(candidate, root)
    body = (json.dumps(validated, indent=2, ensure_ascii=False,
                       sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination = root / "omdet-candidate-profile.json"
    if os.path.lexists(destination):
        raise FileExistsError("Candidate filename already exists: %s" % destination)
    with tempfile.NamedTemporaryFile(prefix=".omdet-candidate-", suffix=".part",
                                     dir=root, delete=False) as stream:
        scratch = Path(stream.name)
    try:
        with scratch.open("wb") as stream:
            stream.write(body)
        if (scratch.stat().st_size != len(body) or
                hashlib.sha256(scratch.read_bytes()).digest() !=
                hashlib.sha256(body).digest()):
            raise ValueError("Incomplete candidate JSON scratch write")


        os.link(scratch, destination)
        return destination
    finally:
        scratch.unlink(missing_ok=True)
