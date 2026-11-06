# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Snapshot the ONNX model, configuration, and processor for BiRefNet HR800 SwinL.

The snapshot verifies the three selected files by hash. The configuration
does not establish the origin of the trained weights. Models with ONNX
external data require a separate method that includes and verifies sidecars.

On Windows, the context manager holds private copies on local NTFS while the
caller uses them. It leaves the original files and staged copies in place.
"""


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

from . import hf_held_stage, source_freeze, source_lock


SCHEMA = "crosspascal.birefnet-hr800.selected.v1"
STATUS = "SCANNED_NOT_LOCKED"
MAX_SELECTED_ONNX_BYTES = 16 * 1024 ** 3
MAX_CONFIG_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = MAX_SELECTED_ONNX_BYTES + 2 * MAX_CONFIG_BYTES
BLOCK = 1024 * 1024
DISK_HEADROOM = 256 * 1024 ** 2
STAGE_PREFIX = ".crosspascal-birefnet-hr800-held-"
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class BiRefNetSnapshotError(ValueError):
    """Cannot prove exact, supported selected HR800 source custody."""


@dataclass(frozen=True)
class HeldBiRefNetSnapshot:
    compiler_input: Path
    stage: Path
    tree_sha256: str
    selected_model_sha256: str
    status: str = "WIN32_HR800_SELECTED_HELD_NOT_RELEASE_GRADE"
    verification_status: str = "NOT_CHECKED"
    independent_checkpoint_attestation: str = "NOT_ATTESTED"
    release_grade_immutable: bool = False
    retained_after_context: bool = True


def _sha(value, label):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise BiRefNetSnapshotError(label + " requires exact lowercase SHA256")
    return value


def _members(model, config, processor):
    paths = (
        ("model", Path(os.path.abspath(os.fspath(model))), MAX_SELECTED_ONNX_BYTES),
        ("config", Path(os.path.abspath(os.fspath(config))), MAX_CONFIG_BYTES),
        ("processor", Path(os.path.abspath(os.fspath(processor))), MAX_CONFIG_BYTES),
    )
    if any(p.parent != paths[0][1].parent for _, p, _ in paths):
        raise BiRefNetSnapshotError(
            "selected HR800 ONNX, config and processor must share one safe owner root")
    names = [p.name for _, p, _ in paths]
    if len({name.casefold() for name in names}) != 3:
        raise BiRefNetSnapshotError("ambiguous duplicate HR800 selected resource names")
    for _role, path, _limit in paths:
        source_freeze._safe_name(path.name)
        source_freeze._safe_ancestors(path)
    if paths[0][1].suffix.lower() != ".onnx" or (
            any(p.suffix.lower() != ".json" for _, p, _ in paths[1:])):
        raise BiRefNetSnapshotError("full HR800 requires ONNX+two JSON resources")
    return paths


def _digest_regular(path, limit):
    """Stream exactly one regular, non-reparse, single-hardlink local source."""
    source_freeze._safe_ancestors(path)
    before = source_freeze._lstat(path)
    if (not stat.S_ISREG(before.st_mode) or
            getattr(before, "st_nlink", 1) != 1 or
            not 0 < before.st_size <= limit):
        raise BiRefNetSnapshotError("selected HR800 member linked, empty or oversized")
    digest = hashlib.sha256()
    amount = 0
    with path.open("rb") as input_file:
        opened = os.fstat(input_file.fileno())
        if source_freeze._descriptor_identity(opened) != (
                source_freeze._descriptor_identity(before)):
            raise BiRefNetSnapshotError("selected resource changed before descriptor open")
        while amount < before.st_size:
            data = input_file.read(min(BLOCK, before.st_size - amount))
            if not data:
                raise BiRefNetSnapshotError("selected HR800 source shortened")
            amount += len(data)
            digest.update(data)
        if input_file.read(1):
            raise BiRefNetSnapshotError("selected HR800 source grew during scan")
        if source_freeze._descriptor_identity(os.fstat(input_file.fileno())) != (
                source_freeze._descriptor_identity(opened)):
            raise BiRefNetSnapshotError("selected HR800 source fd changed during scan")
    after = source_freeze._lstat(path)
    if (source_freeze._identity(before) != source_freeze._identity(after) or
            amount != before.st_size):
        raise BiRefNetSnapshotError("selected HR800 source changed during SHA pass")
    return amount, digest.hexdigest()


def _json_record(path, expected_sha, cap):
    size, sha = _digest_regular(path, cap)
    if sha != expected_sha:
        raise BiRefNetSnapshotError("selected HR800 config/processor SHA changed")
    with path.open("rb") as f:
        body = f.read(cap + 1)
    if len(body) != size or hashlib.sha256(body).hexdigest() != sha:
        raise BiRefNetSnapshotError("selected HR800 JSON changed after SHA")

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise BiRefNetSnapshotError("duplicate selected config/processor JSON key")
            out[key] = value
        return out

    try:
        record = json.loads(body.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                BiRefNetSnapshotError("nonfinite config value")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BiRefNetSnapshotError("invalid bounded selected HR800 JSON") from exc
    if not isinstance(record, dict):
        raise BiRefNetSnapshotError("selected HR800 metadata must be an object")
    return record


def scan_selected_hr800(
        model, config, processor, *, expected_model_sha256=None,
        expected_config_sha256=None, expected_processor_sha256=None):
    """Read the three selected files without modifying them.

    The caller's hashes identify the files. Checkpoint provenance requires
    separate evidence.
    """
    paths = _members(model, config, processor)
    explicit = (expected_model_sha256, expected_config_sha256,
                expected_processor_sha256)
    for label, value in zip(("model", "config", "processor"), explicit):
        if value is not None:
            _sha(value, label)
    entries = []
    total = 0
    for (role, path, limit), expected in zip(paths, explicit):
        amount, digest = _digest_regular(path, limit)
        if expected is not None and digest != expected:
            raise BiRefNetSnapshotError("selected HR800 %s SHA differs from approval" % role)
        entries.append({"role": role, "path": path.name,
                        "bytes": amount, "sha256": digest})
        total += amount
        if total > MAX_TOTAL_BYTES:
            raise BiRefNetSnapshotError("selected HR800 aggregate bytes exceed quota")
    config_record = _json_record(paths[1][1], entries[1]["sha256"], MAX_CONFIG_BYTES)
    processor_record = _json_record(
        paths[2][1], entries[2]["sha256"], MAX_CONFIG_BYTES)


    if set(config_record) != {"architecture", "input_shape", "source_revision"}:
        raise BiRefNetSnapshotError(
            "unsupported/unselected HR800 config external resource fields")
    if set(processor_record) != {
            "image_size", "color_mode", "do_resize", "interpolation",
            "imagenet_normalization", "mean_rgb", "std_rgb"}:
        raise BiRefNetSnapshotError(
            "unsupported/unselected HR800 processor external resource fields")
    if (config_record.get("architecture") != "BiRefNet_HR_SwinL" or
            config_record.get("input_shape") != [1, 3, 800, 800]):
        raise BiRefNetSnapshotError(
            "selected config is not explicit full HR800 SwinL architecture")
    if (processor_record.get("do_resize") is not True or
            type(processor_record.get("image_size")) is not int or
            processor_record["image_size"] != 800 or
            processor_record.get("color_mode") != "RGB" or
            processor_record.get("interpolation") != "cubic" or
            processor_record.get("imagenet_normalization") is not True or
            processor_record.get("mean_rgb") != [0.485, 0.456, 0.406] or
            processor_record.get("std_rgb") != [0.229, 0.224, 0.225]):
        raise BiRefNetSnapshotError(
            "selected HR800 processor differs from Pascal cubic/RGB/ImageNet preprocessing")
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return {
        "schema": SCHEMA, "status": STATUS,
        "model_profile": "FULL_HR800_SWINL_CALLER_SELECTED_NOT_ATTESTED",
        "source_revision": config_record.get("source_revision") or "NOT_ATTESTED",
        "verification_status": "NOT_CHECKED",
        "independent_checkpoint_attestation": "NOT_ATTESTED",
        "release_grade_immutable": False,
        "input_shape": [1, 3, 800, 800],
        "file_count": 3, "total_bytes": total,
        "entries": entries, "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _private_tree(stage, base_identity, selected):
    source_freeze._safe_ancestors(stage)
    current = source_freeze._lstat(stage)
    if (not stat.S_ISDIR(current.st_mode) or
            (current.st_dev, current.st_ino) != base_identity):
        raise BiRefNetSnapshotError("private HR800 stage FileID/parent changed")


    seen = set()
    with os.scandir(stage) as entries:
        for count, member in enumerate(entries, 1):
            if count > 3:
                raise BiRefNetSnapshotError("foreign HR800 private stage child")
            seen.add(member.name)
    if seen != {row["path"] for row in selected["entries"]}:
        raise BiRefNetSnapshotError("private HR800 stage resource inventory changed")
    for row in selected["entries"]:
        path = stage / row["path"]
        source_freeze._safe_ancestors(path)
        info = source_freeze._lstat(path)
        if (not stat.S_ISREG(info.st_mode) or info.st_size != row["bytes"] or
                getattr(info, "st_nlink", 1) != 1):
            raise BiRefNetSnapshotError("private HR800 staged source linked/changed")
    if scan_selected_hr800(
            stage / selected["entries"][0]["path"],
            stage / selected["entries"][1]["path"],
            stage / selected["entries"][2]["path"])["tree_sha256"] != (
                selected["tree_sha256"]):
        raise BiRefNetSnapshotError("private HR800 staged bytes differ from approval")


def _copy_member(src, dest, entry):
    with src.open("rb") as original, dest.open("xb") as target:
        before = os.fstat(original.fileno())
        if before.st_size != entry["bytes"]:
            raise BiRefNetSnapshotError("selected HR800 original resized before private copy")
        remaining = entry["bytes"]
        digest = hashlib.sha256()
        while remaining:
            part = original.read(min(BLOCK, remaining))
            if not part:
                raise BiRefNetSnapshotError("selected HR800 source truncated while staging")
            target.write(part)
            digest.update(part)
            remaining -= len(part)
        if original.read(1) or digest.hexdigest() != entry["sha256"]:
            raise BiRefNetSnapshotError("selected HR800 source drifted during private staging")
        if source_freeze._descriptor_identity(os.fstat(original.fileno())) != (
                source_freeze._descriptor_identity(before)):
            raise BiRefNetSnapshotError("selected HR800 source fd changed during copy")
        target.flush()
        os.fsync(target.fileno())
    source_freeze._safe_ancestors(dest)
    info = source_freeze._lstat(dest)
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
        raise BiRefNetSnapshotError("private HR800 copied resource linked")


def _require_held_hr800_pascal_graph(path):
    """Validate the input and output tensors used by the Pascal runtime.

    The model accepts a float32 RGB tensor of shape 1 × 3 × 800 × 800 named
    `x` or `input`. It returns a single mask of shape 1 × 1 × 800 × 800.
    This check validates the interface. Weight provenance and prediction
    quality require separate verification.
    """


    from . import detect
    detect._admit_onnx_preparse(str(path))
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise BiRefNetSnapshotError(
            "held HR800 compiler ONNX graph cannot be parsed safely") from exc

    def tensor_abi(value_info):
        typed = value_info.type
        if not typed.HasField("tensor_type"):
            return None, None
        tensor = typed.tensor_type
        if not tensor.HasField("shape"):
            return int(tensor.elem_type), None
        shape = []
        for dimension in tensor.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                return int(tensor.elem_type), None
            shape.append(int(dimension.dim_value))
        return int(tensor.elem_type), shape

    weights = {tensor.name for tensor in model.graph.initializer}
    inputs = [info for info in model.graph.input if info.name not in weights]
    if len(inputs) != 1 or inputs[0].name not in ("x", "input") or (
            tensor_abi(inputs[0]) != (onnx.TensorProto.FLOAT, [1, 3, 800, 800])):
        raise BiRefNetSnapshotError(
            "held HR800 source input ABI differs from Pascal float32 RGB NCHW 800")
    outputs = list(model.graph.output)
    if len(outputs) != 1 or not outputs[0].name:
        raise BiRefNetSnapshotError(
            "held HR800 Pascal requires exactly one mask output")
    dtype, shape = tensor_abi(outputs[0])
    if (dtype not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT,
                      onnx.TensorProto.DOUBLE) or
            shape != [1, 1, 800, 800]):
        raise BiRefNetSnapshotError(
            "held HR800 output[0] is not a Pascal one-channel NCHW 800 mask")


@contextmanager
def hold_selected_hr800(model, config, processor, *, stage_parent,
                        approved_tree_sha256):
    """Hold the staged HR800 files open for the caller.

    This operation requires Windows and a local NTFS volume. It checks the
    original and staged files before and after use and retains the staged
    directory even if the caller raises an exception.
    """
    approved = _sha(approved_tree_sha256, "approved HR800 tree")
    if not source_lock._is_windows():
        raise BiRefNetSnapshotError("native Win32 local NTFS required; no unheld fallback")
    selected = scan_selected_hr800(model, config, processor)
    if selected["tree_sha256"] != approved:
        raise BiRefNetSnapshotError("selected HR800 tree differs from approved Preview SHA")
    paths = _members(model, config, processor)
    parent = Path(os.path.abspath(os.fspath(stage_parent)))
    source_freeze._safe_ancestors(parent)
    if not stat.S_ISDIR(source_freeze._lstat(parent).st_mode):
        raise BiRefNetSnapshotError("private HR800 stage parent is not a directory")


    owner = os.path.normcase(str(paths[0][1].parent))
    parent_norm = os.path.normcase(str(parent))
    try:
        inside_owner = os.path.commonpath((owner, parent_norm)) == owner
    except ValueError:
        inside_owner = False
    if inside_owner:
        raise BiRefNetSnapshotError(
            "private HR800 stage parent overlaps selected source owner directory")
    api = source_lock._Kernel32()
    api.assert_local_ntfs(parent)
    if shutil.disk_usage(parent).free < selected["total_bytes"] + DISK_HEADROOM:
        raise BiRefNetSnapshotError("insufficient disk space for selected HR800 stage")


    from .recipe import _unheld_onnx_external_data
    if _unheld_onnx_external_data(str(paths[0][1])):
        raise BiRefNetSnapshotError(
            "HR800 ONNX external_data requires independently selected held sidecars")
    with ExitStack() as handles:


        parent_info = source_freeze._lstat(parent)
        parent_handle = api.kernel.CreateFileW(
            str(parent), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if parent_handle == source_lock.INVALID_HANDLE_VALUE:
            raise BiRefNetSnapshotError(
                "could not hold private HR800 stage parent before creation")
        handles.callback(api.close, parent_handle)
        parent_native = api.FileInfo()

        def check_parent():
            if not api.kernel.GetFileInformationByHandle(
                    parent_handle, ctypes.byref(parent_native)):
                raise BiRefNetSnapshotError("cannot inspect held HR800 private parent")
            if (not parent_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_DIRECTORY
                    or parent_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT
                    or ((parent_native.nFileIndexHigh << 32) | parent_native.nFileIndexLow)
                    != parent_info.st_ino):
                raise BiRefNetSnapshotError("private HR800 parent FileID/reparse differs")
            source_freeze._safe_ancestors(parent)
            observed_parent = source_freeze._lstat(parent)
            if ((observed_parent.st_dev, observed_parent.st_ino) !=
                    (parent_info.st_dev, parent_info.st_ino)):
                raise BiRefNetSnapshotError("private HR800 parent path changed")

        check_parent()
        stage = Path(tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=parent))
        root_info = source_freeze._lstat(stage)
        identity = (root_info.st_dev, root_info.st_ino)


        root = api.kernel.CreateFileW(
            str(stage), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if root == source_lock.INVALID_HANDLE_VALUE:
            raise BiRefNetSnapshotError(
                "could not acquire native HR800 private stage directory handle")
        handles.callback(api.close, root)
        root_native = api.FileInfo()

        def check_root():
            check_parent()
            if not api.kernel.GetFileInformationByHandle(root, ctypes.byref(root_native)):
                raise BiRefNetSnapshotError("HR800 held stage root cannot be inspected")
            if (not root_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_DIRECTORY or
                    root_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT or
                    root_native.dwVolumeSerialNumber != parent_native.dwVolumeSerialNumber or
                    ((root_native.nFileIndexHigh << 32) | root_native.nFileIndexLow)
                    != root_info.st_ino):
                raise BiRefNetSnapshotError("private HR800 native root FileID/reparse differs")
            source_freeze._safe_ancestors(stage)
            now = source_freeze._lstat(stage)
            if (now.st_dev, now.st_ino) != identity:
                raise BiRefNetSnapshotError("HR800 root replaced before/after compiler")

        check_root()
        for (_role, src, _max), entry in zip(paths, selected["entries"]):
            _copy_member(src, stage / entry["path"], entry)
        _private_tree(stage, identity, selected)

        opened = []
        for entry in selected["entries"]:
            source = stage / entry["path"]
            handle = api.open(source)
            handles.callback(api.close, handle)
            native = api.identity(handle)
            file_info = api.FileInfo()
            if (not api.kernel.GetFileInformationByHandle(
                    handle, ctypes.byref(file_info)) or
                    file_info.nNumberOfLinks != 1):
                raise BiRefNetSnapshotError("private HR800 staged member has hardlink alias")
            opened.append((source, handle, native, entry))


        if _unheld_onnx_external_data(str(opened[0][0])):
            raise BiRefNetSnapshotError(
                "held HR800 compiler input contains unselected ONNX external_data sidecars")
        _require_held_hr800_pascal_graph(opened[0][0])

        def audit():
            check_root()
            _private_tree(stage, identity, selected)
            for path, handle, native, entry in opened:
                hf_held_stage._native_check(
                    api, path, handle, native, entry["bytes"], entry["sha256"], BLOCK)
            if scan_selected_hr800(model, config, processor)["tree_sha256"] != approved:
                raise BiRefNetSnapshotError(
                    "original HR800 approved selected tree changed during consumer")
            check_root()

        audit()
        ticket = HeldBiRefNetSnapshot(
            compiler_input=stage / selected["entries"][0]["path"],
            stage=stage, tree_sha256=approved,
            selected_model_sha256=selected["entries"][0]["sha256"])
        try:
            yield ticket
        finally:
            audit()
