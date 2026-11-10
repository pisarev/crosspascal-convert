# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from . import (bundle_atomic, hf_held_stage, omdet_public_candidate,
               source_freeze, source_lock)


SCHEMA = "crosspascal.omdet.compiler-selected-source.v1"
STAGE_PREFIX = ".crosspascal-omdet-compiler-held-"
BLOCK = 1024 * 1024
MAX_ONNX_BYTES = 16 * 1024 ** 3
MAX_SMALL_BYTES = 1024 * 1024
DISK_HEADROOM = 256 * 1024 ** 2
_SHA = re.compile(r"[0-9a-f]{64}\Z")
STAGED_ROLES = (
    ("source_onnx", "omdet-turbo-tiny.onnx", MAX_ONNX_BYTES),
    ("export_contract", "omdet-export-contract.json", MAX_SMALL_BYTES),
    ("feature_manifest", "omdet-feature-assets.json", MAX_SMALL_BYTES),
    ("processor", "preprocessor_config.json", MAX_SMALL_BYTES),
    ("model", "config.json", MAX_SMALL_BYTES),
    ("class_features", "class_features.f32le", MAX_SMALL_BYTES),
    ("task_features", "task_features.f32le", MAX_SMALL_BYTES),
    ("task_mask", "task_mask.i64le", MAX_SMALL_BYTES),
)
OPTIMIZED_TRAINED_SOURCE = (215150131,
    "ac3cfd993880bcd06300f0e5552d136a03bedc82571ff2bbbf6d0ea01956b63b")
OPTIMIZED_ORIGINAL_SOURCE = "56a06084c6170770e547ecad84eb44c4e2cbed5ba5d828634c2857ae57384a77"
OPTIMIZED_REFERENCE_NPZ = "906f0fc89f1627da2017e99440cd99fd36cd6f803b93e2bfb4398b8686383e43"
OPTIMIZED_PRODUCER_REPORT = "99c1a4d9e1c5a7a4d45ce26bc62a4da84673f7c99bb916e6614577317baafb3c"


def inspect_exact_optimized_source(onnx_path, receipt_path, original_pins):
    if (not isinstance(original_pins, dict) or
            set(original_pins) != {row[0] for row in STAGED_ROLES} or
            original_pins["source_onnx"].get("sha256") != OPTIMIZED_ORIGINAL_SOURCE or
            original_pins["source_onnx"].get("bytes") != 215936399):
        raise OmDetCompilerHoldRefused("optimized source needs pinned original eight-role trained graph")
    model = Path(os.path.abspath(os.fspath(onnx_path)))
    receipt = Path(os.path.abspath(os.fspath(receipt_path)))
    if model == receipt:
        raise OmDetCompilerHoldRefused("optimized ONNX and receipt alias")
    size, digest = _bounded_digest(model, MAX_ONNX_BYTES,
                                   OPTIMIZED_TRAINED_SOURCE[0])
    if (size, digest) != OPTIMIZED_TRAINED_SOURCE:
        raise OmDetCompilerHoldRefused("optimized ONNX differs from exact trained SHA")
    length, receipt_sha = _bounded_digest(receipt, MAX_SMALL_BYTES)
    if not 0 < length <= MAX_SMALL_BYTES:
        raise OmDetCompilerHoldRefused("optimized receipt exceeds quota")
    with receipt.open("rb") as stream:
        raw = stream.read(MAX_SMALL_BYTES + 1)
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != receipt_sha:
        raise OmDetCompilerHoldRefused("optimized receipt changed after SHA inspection")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise OmDetCompilerHoldRefused("optimized receipt invalid UTF-8 JSON") from exc
    expected = {
        "source_sha256": OPTIMIZED_ORIGINAL_SOURCE,
        "reference_npz_sha256": OPTIMIZED_REFERENCE_NPZ,
        "producer_report_sha256": OPTIMIZED_PRODUCER_REPORT,
        "optimized_onnx_bytes": OPTIMIZED_TRAINED_SOURCE[0],
        "optimized_onnx_sha256": OPTIMIZED_TRAINED_SOURCE[1],
        "original_onnx_nodes": 6115, "optimized_onnx_nodes": 1636,
    }
    exact_keys = set(expected) | {"schema", "status", "source_original_trained_weights",
        "windows_pascal_native", "human_ground_truth", "outputs"}
    if (not isinstance(value, dict) or set(value) != exact_keys
            or any(value.get(k) != v for k, v in expected.items())
            or value.get("schema") != "crosspascal.omdet.trained-ort-basic-exact-sample.v1"
            or value.get("status") != "ONE_TRAINED_INPUT_FIXTURE_ORIGINAL_ORT_VS_SERIALIZED_ORT_BASIC_NOT_NATIVE"
            or value.get("source_original_trained_weights") != "ORIGINAL_OFFICIAL_NINE_PINNED"
            or value.get("windows_pascal_native") != "NOT_RUN"
            or value.get("human_ground_truth") != "NOT_ATTESTED"
            or not isinstance(value.get("outputs"), dict)
            or set(value["outputs"]) != {"boxes", "scores"}
            or any(value["outputs"].get(name) != {
                "gt_1e3": 0, "max_abs": 0.0, "mean_abs": 0.0, "shape": shape}
                for name, shape in (("boxes", [1, 900, 4]), ("scores", [1, 900, 8])))):
        raise OmDetCompilerHoldRefused("optimized source receipt provenance differs")
    return {"bytes": size, "sha256": digest,
            "receipt_bytes": length, "receipt_sha256": receipt_sha,
            "status": "PRECHECK_ONLY_NOT_WIN32_HELD_NOT_COMPILER_APPROVED"}


class OmDetCompilerHoldRefused(ValueError):
    pass


@dataclass(frozen=True)
class HeldOmDetCompilerSource:
    compiler_input: Path
    stage: Path
    selected_tree_sha256: str
    selected_members: tuple
    optimized_receipt: dict | None = None
    status: str = "WIN32_OMDET_EIGHT_MEMBER_COMPILER_HELD_NOT_RELEASE_GRADE"
    verification_status: str = "NOT_CHECKED"
    native_pascal_run: str = "NOT_RUN"
    release_grade_immutable: bool = False
    stage_retained: bool = True


def _approved_sha(sha):
    if type(sha) is not str or not _SHA.fullmatch(sha):
        raise OmDetCompilerHoldRefused("OmDet selected source requires exact lowercase SHA256")
    return sha


def _original_member(root, relative):
    bundle_atomic.validate_member_name(relative)
    path = root.joinpath(*relative.split("/"))
    source_freeze._safe_ancestors(path)
    info = source_freeze._lstat(path)
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
        raise OmDetCompilerHoldRefused("selected OmDet member is linked/nonregular: " + relative)
    return path, info


def _bounded_digest(path, max_bytes, expected_size=None):
    source_freeze._safe_ancestors(path)
    before = source_freeze._lstat(path)
    if (not stat.S_ISREG(before.st_mode) or
            getattr(before, "st_nlink", 1) != 1 or
            not 0 < before.st_size <= max_bytes or
            (expected_size is not None and before.st_size != expected_size)):
        raise OmDetCompilerHoldRefused("OmDet selected source size/hardlink admission refused")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        first = os.fstat(handle.fileno())
        if source_freeze._descriptor_identity(first) != (
                source_freeze._descriptor_identity(before)):
            raise OmDetCompilerHoldRefused("OmDet selected source changed before SHA read")
        left = first.st_size
        while left:
            body = handle.read(min(BLOCK, left))
            if not body:
                raise OmDetCompilerHoldRefused("OmDet selected source truncated during SHA")
            left -= len(body)
            digest.update(body)
        if handle.read(1):
            raise OmDetCompilerHoldRefused("OmDet selected source grew during SHA")
        if source_freeze._descriptor_identity(os.fstat(handle.fileno())) != (
                source_freeze._descriptor_identity(first)):
            raise OmDetCompilerHoldRefused("OmDet selected source changed during SHA")
    if source_freeze._identity(source_freeze._lstat(path)) != source_freeze._identity(before):
        raise OmDetCompilerHoldRefused("OmDet selected source pathname changed during SHA")
    return first.st_size, digest.hexdigest()


def scan_selected_omdet(root, pins):
    if not isinstance(pins, dict) or set(pins) != {
            item[0] for item in STAGED_ROLES}:
        raise OmDetCompilerHoldRefused("OmDet requires exactly eight selected source roles")
    original = Path(os.path.abspath(os.fspath(root)))
    source_freeze._safe_ancestors(original)
    if not stat.S_ISDIR(source_freeze._lstat(original).st_mode):
        raise OmDetCompilerHoldRefused("OmDet selected owner is not a directory")
    entries, ids, names = [], set(), set()
    total = 0
    for role, staged_name, limit in STAGED_ROLES:
        declared = pins[role]
        if (type(declared) is not dict or
                set(declared) != {"file", "bytes", "sha256"} or
                type(declared["bytes"]) is not int or
                not 0 < declared["bytes"] <= limit):
            raise OmDetCompilerHoldRefused("OmDet source pin role schema invalid: " + role)
        sha = _approved_sha(declared["sha256"])
        path, info = _original_member(original, declared["file"])
        folded = declared["file"].casefold()
        file_id = (info.st_dev, info.st_ino)
        if folded in names or file_id in ids:
            raise OmDetCompilerHoldRefused("OmDet selected resources alias the same source")
        names.add(folded)
        ids.add(file_id)
        size, actual = _bounded_digest(path, limit, declared["bytes"])
        if size != declared["bytes"] or actual != sha:
            raise OmDetCompilerHoldRefused("OmDet selected source SHA differs from approval: " + role)
        entries.append({"role": role, "file": declared["file"],
                        "stage_name": staged_name, "bytes": size, "sha256": actual})
        total += size
        if total > MAX_ONNX_BYTES + 7 * MAX_SMALL_BYTES:
            raise OmDetCompilerHoldRefused("OmDet selected source aggregate quota exceeded")
    encoded = json.dumps(entries, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return {"schema": SCHEMA, "status": "SCANNED_NOT_LOCKED",
            "members": entries, "file_count": len(entries), "total_bytes": total,
            "tree_sha256": hashlib.sha256(encoded).hexdigest(),
            "verification_status": "NOT_CHECKED",
            "trained_model_provenance": "NOT_ATTESTED"}


def _stage_inventory(stage, base_identity, chosen, optimized=None):
    source_freeze._safe_ancestors(stage)
    current = source_freeze._lstat(stage)
    if ((current.st_dev, current.st_ino) != base_identity or
            not stat.S_ISDIR(current.st_mode)):
        raise OmDetCompilerHoldRefused("OmDet compiler stage root FileID changed")
    expected = {row["stage_name"] for row in chosen["members"]}
    if optimized is not None:
        expected.add(optimized["stage_name"])
    names = set()
    with os.scandir(stage) as members:
        for count, member in enumerate(members, 1):
            if count > len(expected):
                raise OmDetCompilerHoldRefused("unapproved OmDet private stage child")
            names.add(member.name)
    if names != expected:
        raise OmDetCompilerHoldRefused("OmDet private stage inventory differs from approval")
    for role, _name, limit in STAGED_ROLES:
        row = next(item for item in chosen["members"] if item["role"] == role)
        amount, sha = _bounded_digest(stage / row["stage_name"], limit, row["bytes"])
        if amount != row["bytes"] or sha != row["sha256"]:
            raise OmDetCompilerHoldRefused("OmDet private compiler bytes changed: " + role)
    if optimized is not None:
        size, sha = _bounded_digest(stage / optimized["stage_name"], MAX_ONNX_BYTES,
                                    optimized["bytes"])
        if (size, sha) != (optimized["bytes"], optimized["sha256"]):
            raise OmDetCompilerHoldRefused("OmDet optimized held compiler source changed")


def _copy_exact(src, dst, row):
    with src.open("rb") as source, dst.open("xb") as target:
        first = os.fstat(source.fileno())
        if (first.st_size != row["bytes"] or
                getattr(first, "st_nlink", 1) != 1):
            raise OmDetCompilerHoldRefused("OmDet original drifted before held copy")
        digest = hashlib.sha256()
        remaining = row["bytes"]
        while remaining:
            block = source.read(min(BLOCK, remaining))
            if not block:
                raise OmDetCompilerHoldRefused("OmDet original shortened during held copy")
            target.write(block)
            digest.update(block)
            remaining -= len(block)
        if source.read(1) or digest.hexdigest() != row["sha256"]:
            raise OmDetCompilerHoldRefused("OmDet selected source drifted during private copy")
        if source_freeze._descriptor_identity(os.fstat(source.fileno())) != (
                source_freeze._descriptor_identity(first)):
            raise OmDetCompilerHoldRefused("OmDet original descriptor changed during copy")
        target.flush()
        os.fsync(target.fileno())


def _require_held_omdet_four_input_graph(path, *, original_symbolic=False):
    from . import detect
    detect._admit_onnx_preparse(str(path))
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise OmDetCompilerHoldRefused(
            "OmDet held ONNX graph is unreadable/unsafe before compiler") from exc
    weights = {tensor.name for tensor in model.graph.initializer}
    observed_inputs = [item for item in model.graph.input if item.name not in weights]
    observed_outputs = list(model.graph.output)

    def typed(item, *, symbolic=False):
        if not item.type.HasField("tensor_type"):
            raise OmDetCompilerHoldRefused("OmDet ONNX input/output is not a tensor")
        tensor = item.type.tensor_type
        if not tensor.HasField("shape"):
            raise OmDetCompilerHoldRefused("OmDet ONNX tensor shape is unspecified")
        dims = []
        for dim in tensor.shape.dim:
            if symbolic and dim.HasField("dim_param") and dim.dim_param:
                dims.append(dim.dim_param)
                continue
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise OmDetCompilerHoldRefused(
                    "OmDet ONNX input/output must have exact static dimensions")
            dims.append(int(dim.dim_value))
        return item.name, int(tensor.elem_type), tuple(dims)

    from .omdet_public_candidate import INPUTS, OUTPUTS
    expected_inputs = [
        (name, onnx.TensorProto.INT64 if dtype == "int64" else onnx.TensorProto.FLOAT,
         tuple(shape))
        for name, shape, _layout, dtype in INPUTS
    ]
    expected_outputs = [
        (name, onnx.TensorProto.FLOAT, tuple(shape))
        for name, shape, _dtype, _semantics in OUTPUTS
    ]
    if original_symbolic:
        symbolic_outputs = [
            ("boxes", onnx.TensorProto.FLOAT,
             ("Gatherboxes_dim_0", "Gatherboxes_dim_1", "Gatherboxes_dim_2")),
            ("scores", onnx.TensorProto.FLOAT,
             ("Gatherscores_dim_0", "Gatherscores_dim_1", 8)),
        ]
        if (_bounded_digest(path, MAX_ONNX_BYTES) !=
                (215936399, OPTIMIZED_ORIGINAL_SOURCE)):
            raise OmDetCompilerHoldRefused("symbolic original output requires exact trained ONNX SHA")
    else:
        symbolic_outputs = expected_outputs
    if ([typed(item) for item in observed_inputs] != expected_inputs or
            [typed(item, symbolic=original_symbolic) for item in observed_outputs]
            != symbolic_outputs):
        raise OmDetCompilerHoldRefused(
            "OmDet held ONNX real four-input/two-output Pascal ABI differs")


def _require_held_omdet_semantic_resources(stage, chosen):
    from . import omdet_bundle_profile
    rows = {row["role"]: row for row in chosen["members"]}

    def json_for(role):
        item = rows[role]
        body = omdet_public_candidate._small_bytes(
            stage, item["stage_name"], MAX_SMALL_BYTES)
        if (len(body) != item["bytes"] or
                hashlib.sha256(body).hexdigest() != item["sha256"]):
            raise OmDetCompilerHoldRefused(
                "held OmDet metadata differs from selected %s SHA" % role)
        return omdet_public_candidate._strict_object(
            body, "held OmDet " + role)

    selected_contract = json_for("export_contract")
    feature_manifest = json_for("feature_manifest")
    inputs, outputs, assets, labels = omdet_bundle_profile._inspect_four_inputs(
        stage, selected_contract, feature_manifest,
        rows["feature_manifest"]["stage_name"])
    expected_inputs = [
        {"name": name, "shape": list(shape), "dtype": dtype}
        for name, shape, _layout, dtype in omdet_public_candidate.INPUTS
    ]
    expected_outputs = [
        {"name": name, "shape": list(shape), "dtype": dtype,
         "semantics": meaning}
        for name, shape, dtype, meaning in omdet_public_candidate.OUTPUTS
    ]
    if (inputs != expected_inputs or outputs != expected_outputs or
            labels != list(omdet_public_candidate.LABELS)):
        raise OmDetCompilerHoldRefused(
            "held OmDet native four-input export/feature ABI changed")
    for role in omdet_public_candidate.FEATURE_FILES:
        chosen_role = rows[role]
        if (assets[role]["file"] != chosen_role["stage_name"] or
                assets[role]["bytes"] != chosen_role["bytes"] or
                assets[role]["sha256"] != chosen_role["sha256"]):
            raise OmDetCompilerHoldRefused(
                "held OmDet %s feature metadata differs from held SHA" % role)


@contextmanager
def hold_selected_omdet(root, pins, *, stage_parent, approved_tree_sha256,
                        optimized_onnx=None, optimization_receipt=None):
    approved = _approved_sha(approved_tree_sha256)
    if not source_lock._is_windows():
        raise OmDetCompilerHoldRefused("Win32 local NTFS required; no unheld fallback")
    chosen = scan_selected_omdet(root, pins)
    if (optimized_onnx is None) != (optimization_receipt is None):
        raise OmDetCompilerHoldRefused("optimized ONNX and independent receipt required together")
    optimized = None
    if optimized_onnx is not None:
        if (pins["export_contract"]["sha256"] !=
                "9982c3e133c81f65c77ca9af058b7c51a38f7aafd6a1d12c7d926f5c5a9678d3" or
                pins["feature_manifest"]["sha256"] !=
                "aef1f159663b02748df0f00732015f0273c98bde8b65ddaf88ad7b9ed160e0f6"):
            raise OmDetCompilerHoldRefused("optimized compiler requires exact trained eight-member contract")
        verified = inspect_exact_optimized_source(optimized_onnx, optimization_receipt, pins)
        optimized = {"role": "optimized_compiler_input", "stage_name": "omdet-trained-ort-basic.onnx",
                     "bytes": verified["bytes"], "sha256": verified["sha256"]}
        expected_receipt = {"bytes": verified["receipt_bytes"],
                            "sha256": verified["receipt_sha256"]}
    if chosen["tree_sha256"] != approved:
        raise OmDetCompilerHoldRefused("OmDet source tree differs from approved Preview")
    original = Path(os.path.abspath(os.fspath(root)))
    parent = Path(os.path.abspath(os.fspath(stage_parent)))
    source_freeze._safe_ancestors(parent)
    if not stat.S_ISDIR(source_freeze._lstat(parent).st_mode):
        raise OmDetCompilerHoldRefused("OmDet private stage parent is missing")


    for ancestor in (parent, *parent.parents):
        if os.path.samefile(ancestor, original):
            raise OmDetCompilerHoldRefused("OmDet private stage is inside owner source tree")
    api = source_lock._Kernel32()
    api.assert_local_ntfs(parent)
    if shutil.disk_usage(parent).free < chosen["total_bytes"] + (optimized["bytes"] if optimized else 0) + DISK_HEADROOM:
        raise OmDetCompilerHoldRefused("insufficient disk headroom for selected OmDet")


    onnx_row = chosen["members"][0]
    omdet_public_candidate._reject_external_onnx(
        original, {"file": onnx_row["file"], "bytes": onnx_row["bytes"],
                   "sha256": onnx_row["sha256"]})
    if optimized is not None:
        optimized_source = Path(os.path.abspath(os.fspath(optimized_onnx)))
        receipt_source = Path(os.path.abspath(os.fspath(optimization_receipt)))
        for candidate in (optimized_source, receipt_source):
            for ancestor in (candidate.parent, *candidate.parents):
                if os.path.samefile(ancestor, original):
                    raise OmDetCompilerHoldRefused("optimized inputs must have independent source owner")
        if os.path.samefile(optimized_source, receipt_source):
            raise OmDetCompilerHoldRefused("optimized source aliases receipt")
        omdet_public_candidate._reject_external_onnx(
            optimized_source.parent, {"file": optimized_source.name,
                "bytes": optimized["bytes"], "sha256": optimized["sha256"]})
    with ExitStack() as handles:
        receipt_held = None
        if optimized is not None:
            receipt_handle = api.open(receipt_source)
            handles.callback(api.close, receipt_handle)
            receipt_native = api.identity(receipt_handle)
            receipt_info = api.FileInfo()
            if (not api.kernel.GetFileInformationByHandle(receipt_handle, ctypes.byref(receipt_info))
                    or receipt_info.nNumberOfLinks != 1):
                raise OmDetCompilerHoldRefused("optimized receipt native hardlink refused")
            receipt_held = (receipt_source, receipt_handle, receipt_native)
        parent_info = source_freeze._lstat(parent)
        parent_handle = api.kernel.CreateFileW(
            str(parent), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if parent_handle == source_lock.INVALID_HANDLE_VALUE:
            raise OmDetCompilerHoldRefused("cannot hold OmDet stage parent before mkdir")
        handles.callback(api.close, parent_handle)
        parent_native = api.FileInfo()

        def check_parent():
            if not api.kernel.GetFileInformationByHandle(
                    parent_handle, ctypes.byref(parent_native)):
                raise OmDetCompilerHoldRefused("cannot inspect held OmDet stage parent")
            observed = source_freeze._lstat(parent)
            if (not (parent_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_DIRECTORY)
                    or (parent_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT)
                    or ((parent_native.nFileIndexHigh << 32) | parent_native.nFileIndexLow)
                    != parent_info.st_ino or
                    (observed.st_dev, observed.st_ino) !=
                    (parent_info.st_dev, parent_info.st_ino)):
                raise OmDetCompilerHoldRefused("OmDet stage parent FileID/reparse changed")

        check_parent()
        stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=parent))
        stage_info = source_freeze._lstat(stage)
        identity = (stage_info.st_dev, stage_info.st_ino)
        root_handle = api.kernel.CreateFileW(
            str(stage), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if root_handle == source_lock.INVALID_HANDLE_VALUE:
            raise OmDetCompilerHoldRefused("cannot hold OmDet private stage root")
        handles.callback(api.close, root_handle)
        native_root = api.FileInfo()

        def check_root():
            check_parent()
            if not api.kernel.GetFileInformationByHandle(
                    root_handle, ctypes.byref(native_root)):
                raise OmDetCompilerHoldRefused("cannot inspect held OmDet private stage")
            observed = source_freeze._lstat(stage)
            if (not (native_root.dwFileAttributes & source_lock.FILE_ATTRIBUTE_DIRECTORY)
                    or (native_root.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT)
                    or native_root.dwVolumeSerialNumber != parent_native.dwVolumeSerialNumber
                    or ((native_root.nFileIndexHigh << 32) | native_root.nFileIndexLow)
                    != stage_info.st_ino or
                    (observed.st_dev, observed.st_ino) != identity):
                raise OmDetCompilerHoldRefused("OmDet held private stage root FileID changed")

        check_root()
        for row in chosen["members"]:
            src = original.joinpath(*row["file"].split("/"))
            _copy_exact(src, stage / row["stage_name"], row)
        if optimized is not None:
            _copy_exact(optimized_source, stage / optimized["stage_name"], optimized)
        _stage_inventory(stage, identity, chosen, optimized)
        opened = []
        for row in chosen["members"] + ([optimized] if optimized else []):
            path = stage / row["stage_name"]
            handle = api.open(path)
            handles.callback(api.close, handle)
            native = api.identity(handle)
            info = api.FileInfo()
            if (not api.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)) or
                    info.nNumberOfLinks != 1):
                raise OmDetCompilerHoldRefused("OmDet staged member has extra hardlink")
            opened.append((path, handle, native, row))

        def audit():
            check_root()
            _stage_inventory(stage, identity, chosen, optimized)
            _require_held_omdet_semantic_resources(stage, chosen)
            for path, handle, native, row in opened:
                hf_held_stage._native_check(
                    api, path, handle, native, row["bytes"], row["sha256"], BLOCK)
            reread = scan_selected_omdet(original, pins)
            if reread["tree_sha256"] != approved:
                raise OmDetCompilerHoldRefused("OmDet original changed during consumer")
            if optimized is not None:
                current = inspect_exact_optimized_source(optimized_source, receipt_source, pins)
                if (current["receipt_bytes"], current["receipt_sha256"]) != (
                        expected_receipt["bytes"], expected_receipt["sha256"]):
                    raise OmDetCompilerHoldRefused("optimized receipt changed during compiler window")
                path, handle, native = receipt_held
                hf_held_stage._native_check(api, path, handle, native,
                    expected_receipt["bytes"], expected_receipt["sha256"], BLOCK)
            check_root()

        stage_model = stage / onnx_row["stage_name"]
        omdet_public_candidate._reject_external_onnx(
            stage, {"file": onnx_row["stage_name"], "bytes": onnx_row["bytes"],
                    "sha256": onnx_row["sha256"]})
        _require_held_omdet_four_input_graph(stage_model,
            original_symbolic=optimized is not None)
        compiler_model = stage_model
        if optimized is not None:
            compiler_model = stage / optimized["stage_name"]
            omdet_public_candidate._reject_external_onnx(
                stage, {"file": optimized["stage_name"], "bytes": optimized["bytes"],
                        "sha256": optimized["sha256"]})
            _require_held_omdet_four_input_graph(compiler_model)
        audit()
        ticket = HeldOmDetCompilerSource(
            compiler_input=compiler_model, stage=stage,
            selected_tree_sha256=approved,
            selected_members=tuple(chosen["members"]),
            optimized_receipt=dict(expected_receipt) if optimized is not None else None)
        try:
            yield ticket
        finally:
            audit()
