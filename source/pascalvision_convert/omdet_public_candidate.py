# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Publish a strictly pinned, unverified OmDet tiny candidate as Pascal v2.

The historical ``omdet-candidate-profile.json`` remains a v1 DISABLED record.
This opt-in route revalidates it and writes a separate Pascal four-input v2
passport into an exclusively created same-parent NTFS stage. A successful
namespace commit does not attest a compiler, native runtime, numerical parity,
source authenticity, redistribution rights, or release readiness.

No source is modified and an incomplete stage is retained on every failure.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

from . import bundle_atomic, manifest, omdet_bundle_profile


SCHEMA = "crosspascal.omdet.public-candidate.v2"
PROFILE_NAME = "omdet-candidate-profile.json"
REVISION = "7fe93cecfb770c4d76cf71163956221249cab566"
LABELS = ("person", "car", "dog", "cat", "bicycle", "motorcycle",
          "airplane", "bus")
INPUTS = (
    ("pixel_values", (1, 3, 640, 640), "NCHW", "float32"),
    ("class_features", (8, 1, 512), "NCF", "float32"),
    ("task_features", (19, 1, 512), "NCF", "float32"),
    ("task_mask", (1, 19), "NT", "int64"),
)
OUTPUTS = (
    ("boxes", (1, 900, 4), "float32", "normalized_cxcywh"),
    ("scores", (1, 900, 8), "float32", "raw_logits_not_sigmoid"),
)
FEATURE_FILES = {
    "class_features": "class_features.f32le",
    "task_features": "task_features.f32le",
    "task_mask": "task_mask.i64le",
}


ASSET_PINS = {
    "processor": (492, "cc6c8ac92ba1a8a62065f3439ebb58f01e4895fa038901e1df14c3427a617e23"),
    "model": (1959, "c0d6546428c6dd7e72d9c8e2123cdd7ff24925dfbff9b535b66cd2f19ca65c46"),
    "features": (2023, "d75049e0e1b0061251cc15bebd87dbe23a8ec91b009a520945dc4f8d58021dae"),
    "class_features": (16384, "6a90c68fcd3c22b71909a4fe38d9cea6145667a5d3aa6a79689d58e11666cb71"),
    "task_features": (38912, "5e6b2db8e8a2a5fa0972569a88fdee7f62aad32d7ad9af1a7dc98af108134949"),
    "task_mask": (152, "71abaaa065c52e925b405818f0e8dfc4af012a3dd317f0dacf647e88ec000013"),
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_JSON_LIMIT = 131072
_BLOCK = 1024 * 1024
_MAX_ONNX_FIELDS = 2_000_000
_MAX_ONNX_DEPTH = 32
_FILE_QUOTA = bundle_atomic.MAX_NATIVE_FAMILY_BYTES


class OmDetPublicCandidateError(ValueError):
    """Fail closed without deleting a private or exposed candidate directory."""


@contextmanager
def _hold_plain_ntfs_directory(directory):
    """Deny rename/delete of one verified directory throughout a write scope.

    Windows directory names can otherwise be exchanged for an NTFS junction
    between `mkdir` and a later `destination.open('xb')`. The latter follows
    that junction and creates a previously absent file OUTSIDE our stage, even
    if the post-write inventory correctly refuses READY. A GENERIC_READ
    CreateFileW with READ|WRITE sharing but NO DELETE sharing keeps the name
    stable while still permitting children to be created in the directory.
    FILE_READ_ATTRIBUTES is NOT enough: on NTFS it permits a directory rename
    even with sharing flags excluding delete (empirically reproduced).

    Hold each ancestor as well as stage/features/audit/source. A pathname
    check alone or a check after writing does not close this race. This is a
    live Win32 namespace hold, NOT immutable source bytes or crash durability.
    """
    if os.name != "nt":
        raise OmDetPublicCandidateError("held NTFS directories require native Windows")
    from ctypes import wintypes

    class FileInfo(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME),
            ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("file_id_high", wintypes.DWORD),
            ("file_id_low", wintypes.DWORD),
        ]

    path = Path(directory)
    before = bundle_atomic._guard_stat(str(path), directory=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(FileInfo)]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(
        str(path), 0x80000000, 0x1 | 0x2, None, 3,
        0x02000000 | 0x00200000, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise OmDetPublicCandidateError(
            "cannot hold directory against junction/delete: %s; winerror=%d" %
            (path, ctypes.get_last_error()))
    try:
        info = FileInfo()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise OmDetPublicCandidateError(
                "cannot inspect held NTFS directory identity: %s" % path)
        if (not info.attributes & 0x10 or info.attributes & 0x400):
            raise OmDetPublicCandidateError(
                "held stage directory is not plain or is a reparse point: %s" % path)
        after = bundle_atomic._guard_stat(str(path), directory=True)
        held_id = (int(info.file_id_high) << 32) | int(info.file_id_low)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or (
                int(after.st_ino) != held_id):
            raise OmDetPublicCandidateError(
                "NTFS directory path identity changed during held open: %s" % path)
        yield
        after = bundle_atomic._guard_stat(str(path), directory=True)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise OmDetPublicCandidateError(
                "held NTFS directory path identity changed before release: %s" % path)
    finally:
        if not kernel.CloseHandle(handle):
            raise OmDetPublicCandidateError(
                "failed to release held NTFS directory handle: %s" % path)


def _strict_object(body: bytes, label: str) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise OmDetPublicCandidateError(
                    "%s contains duplicate JSON key %r" % (label, key))
            result[key] = value
        return result

    try:
        result = json.loads(
            body.decode("utf-8"), object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OmDetPublicCandidateError(
                    "%s contains nonfinite JSON token %s" % (label, token))))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OmDetPublicCandidateError("%s is not strict UTF-8 JSON" % label) from exc
    if not isinstance(result, dict):
        raise OmDetPublicCandidateError("%s must be a JSON object" % label)
    return result


def _paths(candidate_dir, destination):
    root = Path(bundle_atomic._local_absolute(candidate_dir))
    final = Path(bundle_atomic._local_absolute(destination))

    for label, path in (("candidate", root), ("destination parent", final.parent)):
        if (os.path.normcase(os.path.abspath(str(path))) !=
                os.path.normcase(os.path.realpath(str(path)))):
            raise OmDetPublicCandidateError(
                label + " contains redirected or short-name path aliases")
    for path in (root, final.parent):
        bundle_atomic._guard_existing_ancestors(str(path))
        bundle_atomic._require_ntfs(str(path))
    bundle_atomic._guard_stat(str(root), directory=True)
    if os.path.lexists(final):
        raise FileExistsError("destination already exists: %s" % final)
    a, b = os.path.normcase(str(root)), os.path.normcase(str(final))
    try:
        common = os.path.normcase(os.path.commonpath((a, b)))
    except ValueError:
        common = ""
    if common in (a, b):
        raise OmDetPublicCandidateError("candidate input and destination overlap")
    return root, final


def _member(root: Path, relative: str) -> tuple[Path, os.stat_result]:
    relative = bundle_atomic.validate_member_name(relative)
    path = root.joinpath(*relative.split("/"))
    bundle_atomic._guard_existing_ancestors(str(path.parent))
    info = bundle_atomic._guard_stat(str(path), directory=False)
    if info.st_nlink != 1:
        raise OmDetPublicCandidateError("hardlinked candidate member: " + relative)
    return path, info


def _pin(record, label):
    feature_role = next((name for name, file in FEATURE_FILES.items()
                         if label in (name, "features/" + file)), None)
    keys = {"file", "bytes", "sha256"}
    enriched = keys | {"shape", "dtype"}
    if not isinstance(record, dict) or set(record) != (
            enriched if feature_role is not None else keys):
        raise OmDetPublicCandidateError("%s has unsupported or missing pin fields" % label)
    if feature_role is not None:
        expected = next((shape, dtype) for name, shape, _layout, dtype in INPUTS
                        if name == feature_role)
        storage_dtype = "int64le" if expected[1] == "int64" else "float32le"
        if (type(record["shape"]) is not list or
                record["shape"] != list(expected[0]) or
                record["dtype"] != storage_dtype):
            raise OmDetPublicCandidateError(
                "%s feature shape/dtype differs from pinned tiny graph" % label)
    if (type(record["bytes"]) is not int or record["bytes"] < 1 or
            record["bytes"] > _FILE_QUOTA or
            not isinstance(record["sha256"], str) or
            not _SHA.fullmatch(record["sha256"])):
        raise OmDetPublicCandidateError("%s has invalid size/SHA pin" % label)
    bundle_atomic.validate_member_name(record["file"])
    return {key: record[key] for key in ("file", "bytes", "sha256")}


def _small_bytes(root: Path, relative: str, limit: int) -> bytes:
    path, info = _member(root, relative)
    if not 0 < info.st_size <= limit:
        raise OmDetPublicCandidateError("small JSON member empty/oversized: " + relative)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if bundle_atomic._stamp(os.fstat(fd)) != bundle_atomic._stamp(info):
            raise OmDetPublicCandidateError("JSON member changed before read: " + relative)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if (len(data) != info.st_size or
                bundle_atomic._stamp(os.fstat(fd)) != bundle_atomic._stamp(info) or
                bundle_atomic._stamp(bundle_atomic._guard_stat(str(path), directory=False))
                != bundle_atomic._stamp(info)):
            raise OmDetPublicCandidateError("JSON member changed while reading: " + relative)
        return data
    finally:
        os.close(fd)


def _assert_asset(role, record):
    size, digest = ASSET_PINS[role]
    if record["bytes"] != size or record["sha256"] != digest:
        raise OmDetPublicCandidateError(
            "OmDet %s differs from Pascal pinned revision %s" % (role, REVISION))


def _read_varint(stream, end):
    number = 0
    for shift in range(0, 70, 7):
        if stream.tell() >= end:
            raise OmDetPublicCandidateError("truncated ONNX protobuf varint")
        part = stream.read(1)
        if not part:
            raise OmDetPublicCandidateError("truncated ONNX protobuf")
        byte = part[0]
        if shift == 63 and byte > 1:
            raise OmDetPublicCandidateError("ONNX protobuf varint exceeds uint64")
        number |= (byte & 127) << shift
        if byte < 128:
            return number
    raise OmDetPublicCandidateError("ONNX protobuf varint exceeds ten bytes")


def _reject_external_onnx(root: Path, record):
    """Walk bounded ONNX protobuf envelopes, seeking past inline tensors.

    External weights can live in sparse tensors or attributes in nested graphs,
    not just graph.initializer. Reject every such dependency until an independently
    selected and fully pinned sidecar closure is supported here. No third-party
    package, whole-model allocation, filename heuristic or best-effort fallback.
    """
    path, first = _member(root, record["file"])
    if not record["file"].lower().endswith(".onnx"):
        raise OmDetPublicCandidateError("source model must be an ONNX file")
    if first.st_size != record["bytes"]:
        raise OmDetPublicCandidateError("ONNX source size differs from candidate pin")
    children = {
        "model": {7: "graph"},
        "graph": {1: "node", 5: "tensor", 15: "sparse"},
        "node": {5: "attribute"},
        "attribute": {5: "tensor", 6: "graph", 10: "tensor",
                      11: "graph", 22: "sparse", 23: "sparse"},
        "sparse": {1: "tensor", 2: "tensor"},
    }
    visited = 0
    model_graphs = 0

    def walk(stream, kind, end, depth):
        nonlocal visited, model_graphs
        if depth > _MAX_ONNX_DEPTH:
            raise OmDetPublicCandidateError("ONNX graph nesting limit exceeded")
        while stream.tell() < end:
            visited += 1
            if visited > _MAX_ONNX_FIELDS:
                raise OmDetPublicCandidateError("ONNX protobuf field count exceeds bound")
            tag = _read_varint(stream, end)
            number, wire = tag >> 3, tag & 7
            if number == 0 or wire not in (0, 1, 2, 5):
                raise OmDetPublicCandidateError("unsupported/malformed ONNX wire type")
            if kind == "model" and number in (20, 25):
                raise OmDetPublicCandidateError("ONNX training/function graph is out of scope")
            if kind == "tensor" and number == 13:
                raise OmDetPublicCandidateError("ONNX external_data has no pinned closure")
            if kind == "tensor" and number == 14:
                if wire != 0:
                    raise OmDetPublicCandidateError("malformed ONNX data_location")
                if _read_varint(stream, end) != 0:
                    raise OmDetPublicCandidateError("ONNX external data_location refused")
                continue
            child = children.get(kind, {}).get(number)
            if child is not None and wire != 2:
                raise OmDetPublicCandidateError("malformed ONNX nested message")
            if wire == 2:
                length = _read_varint(stream, end)
                payload_end = stream.tell() + length
                if payload_end > end:
                    raise OmDetPublicCandidateError("ONNX protobuf length exceeds envelope")
                if child is not None:
                    if kind == "model" and number == 7:
                        model_graphs += 1
                        if model_graphs != 1:
                            raise OmDetPublicCandidateError("ONNX requires one model graph")
                    walk(stream, child, payload_end, depth + 1)
                else:
                    stream.seek(payload_end)
            elif wire == 0:
                _read_varint(stream, end)
            else:
                amount = 8 if wire == 1 else 4
                if stream.tell() + amount > end:
                    raise OmDetPublicCandidateError("truncated ONNX fixed-width field")
                stream.seek(amount, os.SEEK_CUR)
        if stream.tell() != end:
            raise OmDetPublicCandidateError("ONNX protobuf envelope length mismatch")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if bundle_atomic._stamp(os.fstat(fd)) != bundle_atomic._stamp(first):
            raise OmDetPublicCandidateError("ONNX source changed before structure scan")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            walk(stream, "model", first.st_size, 0)
        if (model_graphs != 1 or
                bundle_atomic._stamp(os.fstat(fd)) != bundle_atomic._stamp(first) or
                bundle_atomic._stamp(bundle_atomic._guard_stat(str(path), directory=False))
                != bundle_atomic._stamp(first)):
            raise OmDetPublicCandidateError("ONNX graph absent or source changed")
    finally:
        os.close(fd)


def _copy_pinned(root: Path, stage: Path, record, name: str):
    record = _pin(record, name)
    relative = bundle_atomic.validate_member_name(name)
    source, before = _member(root, record["file"])
    if before.st_size != record["bytes"]:
        raise OmDetPublicCandidateError("source size changed for " + name)
    destination = stage.joinpath(*relative.split("/"))
    digest = hashlib.sha256()
    count = 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(source, flags)
    try:
        if bundle_atomic._stamp(os.fstat(fd)) != bundle_atomic._stamp(before):
            raise OmDetPublicCandidateError("source identity changed for " + name)
        with os.fdopen(fd, "rb", closefd=False) as reader, destination.open("xb") as writer:
            while True:
                block = reader.read(_BLOCK)
                if not block:
                    break
                count += len(block)
                if count > record["bytes"]:
                    raise OmDetPublicCandidateError("source exceeds approved length: " + name)
                digest.update(block)
                writer.write(block)
        if (count != record["bytes"] or digest.hexdigest() != record["sha256"] or
                bundle_atomic._stamp(os.fstat(fd)) != bundle_atomic._stamp(before) or
                bundle_atomic._stamp(bundle_atomic._guard_stat(str(source), directory=False))
                != bundle_atomic._stamp(before)):
            raise OmDetPublicCandidateError("source SHA/identity changed while copying " + name)
    finally:
        os.close(fd)
    after = bundle_atomic._guard_stat(str(destination), directory=False)
    if after.st_size != count or bundle_atomic._file_sha(str(destination), after) != record["sha256"]:
        raise OmDetPublicCandidateError("staged copy SHA differs from candidate pin: " + name)
    return {"file": relative, "bytes": count, "sha256": record["sha256"]}


def _write_generated(stage: Path, name: str, data: bytes):
    name = bundle_atomic.validate_member_name(name)
    if not data or len(data) > _JSON_LIMIT:
        raise OmDetPublicCandidateError("generated OmDet record has invalid length")
    path = stage.joinpath(*name.split("/"))
    with path.open("xb") as stream:
        stream.write(data)
    info = bundle_atomic._guard_stat(str(path), directory=False)
    digest = hashlib.sha256(data).hexdigest()
    if info.st_size != len(data) or bundle_atomic._file_sha(str(path), info) != digest:
        raise OmDetPublicCandidateError("staged generated record differs: " + name)
    return {"file": name, "bytes": len(data), "sha256": digest}


def _json_output(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _validate_fixed_profile(root: Path, candidate):
    target = candidate["target_profile"]["deployment"]
    if target["os"] != "windows" or target["arch"] != "x86_64":
        raise OmDetPublicCandidateError("Pascal OmDet v2 requires Windows x86-64")
    if candidate["class_labels_in_order"] != list(LABELS):
        raise OmDetPublicCandidateError("pinned eight trained labels/order differ")
    for current, expected in zip(candidate["vm_inputs"], INPUTS):
        if (current["name"], current["shape"], current["dtype"]) != (
                expected[0], list(expected[1]), expected[3]):
            raise OmDetPublicCandidateError("native four-input ABI differs from Pascal tiny")
    for current, expected in zip(candidate["vm_outputs"], OUTPUTS):
        if (current["name"], current["shape"], current["dtype"],
                current["semantics"]) != (
                expected[0], list(expected[1]), expected[2], expected[3]):
            raise OmDetPublicCandidateError("native output ABI differs from Pascal tiny")
    for name, role in (("processor", "processor"), ("model", "model")):
        _assert_asset(role, _pin(candidate["hf_configs"][name], name))
    _assert_asset("features", _pin(candidate["feature_manifest"], "features"))
    for role in FEATURE_FILES:
        _assert_asset(role, _pin(candidate["trained_features"][role], role))
    if (PurePosixPath(candidate["feature_manifest"]["file"]).name !=
            "omdet-feature-assets.json"):
        raise OmDetPublicCandidateError("feature manifest has wrong source basename")
    for role, filename in FEATURE_FILES.items():
        if PurePosixPath(candidate["trained_features"][role]["file"]).name != filename:
            raise OmDetPublicCandidateError("trained feature source basename differs: " + role)
    feature_json = _strict_object(_small_bytes(
        root, candidate["feature_manifest"]["file"], _JSON_LIMIT), "trained features")
    if (feature_json.get("class_labels_in_order") != list(LABELS) or
            feature_json.get("task_phrase") != "detect objects" or
            feature_json.get("profile") != "omdet-turbo-tiny-fixed-eight-classes-only" or
            feature_json.get("status") != "FEATURES_ONLY_NATIVE_NOT_CHECKED" or
            feature_json.get("verified_pascal_facade") is not False or
            feature_json.get("source_npz_sha256") !=
            "09d7c6178b5ce0f2cd2a61a8f8976987b48475ddf3ff307b6f1783da120be959" or
            feature_json.get("source_npz_bytes") != 5015380):
        raise OmDetPublicCandidateError("trained feature provenance/revision differs")
    if candidate["verification"]["pascal_facade"] != "DISABLED":
        raise OmDetPublicCandidateError("v1 Facade status must remain DISABLED")
    return target


def _preflight_candidate_byte_quota(root: Path, profile: dict, *, profile_bytes: int):
    """Stat the COMPLETE selected v1 closure before a potentially huge SHA scan.

    The v1 validator currently streams its original ONNX, native module and
    runtime DLLs without a per-file limit. A maliciously large native binary
    must not be hashed for minutes only to be rejected by the publisher's
    eventual 64-GiB quota. This is a cheap admission bound, NOT a substitute
    for the subsequent exact independent pins, v1 rehash or identity checks.
    """
    if (type(profile) is not dict or type(profile_bytes) is not int or
            not 0 < profile_bytes <= _JSON_LIMIT):
        raise OmDetPublicCandidateError("invalid candidate profile quota input")
    try:
        features = profile["trained_features"]
        libraries = profile["runtime"]["libraries"]
        if (type(features) is not dict or set(features) != set(FEATURE_FILES) or
                type(libraries) is not dict or
                not 3 <= len(libraries) <= 4):
            raise OmDetPublicCandidateError("invalid OmDet selected role inventory")
        selected = [
            ("source_onnx", profile["source_onnx"]),
            ("export_contract", profile["export_contract"]),
            ("feature_manifest", profile["feature_manifest"]),
            ("processor", profile["hf_configs"]["processor"]),
            ("model", profile["hf_configs"]["model"]),
            ("compiled_module", profile["compiled_module"]),
            ("compiled_inputs", profile["compiled_inputs"]),
        ]
        selected.extend((role, features[role]) for role in FEATURE_FILES)
        selected.extend(("runtime_" + role, record)
                        for role, record in libraries.items())
    except (KeyError, TypeError, AttributeError) as exc:
        raise OmDetPublicCandidateError(
            "incomplete selected OmDet quota inventory") from exc

    total = profile_bytes + 3 * _JSON_LIMIT
    if total > _FILE_QUOTA:
        raise OmDetPublicCandidateError(
            "selected OmDet candidate exceeds native-family byte quota before SHA")
    names, identities = set(), set()
    for role, record in selected:
        if (type(record) is not dict or
                type(record.get("bytes")) is not int or record["bytes"] < 1):
            raise OmDetPublicCandidateError(
                "invalid selected OmDet byte-size declaration: " + role)
        filename = record.get("file")
        path, info = _member(root, filename)
        folded = filename.casefold()
        identity = (info.st_dev, info.st_ino)
        if folded in names or identity in identities:
            raise OmDetPublicCandidateError(
                "duplicate selected OmDet source path/identity: " + role)
        names.add(folded)
        identities.add(identity)
        if info.st_size != record["bytes"]:
            raise OmDetPublicCandidateError(
                "selected OmDet source size differs before SHA: " + role)
        if info.st_size > _FILE_QUOTA - total:
            raise OmDetPublicCandidateError(
                "selected OmDet candidate exceeds native-family byte quota before SHA")
        total += info.st_size
    return total


def publish(candidate_dir, destination, *, expected_pins=None) -> dict:
    """Convert a validated v1 candidate into a *new*, unverified Pascal v2 bundle.

    ``expected_pins`` must be an independently obtained v1 pins mapping.
    Matching the pins checks selected bytes. Source origin and compiler identity
    still require independent evidence.
    """
    root, final = _paths(candidate_dir, destination)

    if not isinstance(expected_pins, dict) or not expected_pins:
        raise OmDetPublicCandidateError(
            "an independently supplied full expected_pins closure is required")
    original = _small_bytes(root, PROFILE_NAME, _JSON_LIMIT)
    candidate = _strict_object(original, PROFILE_NAME)
    _preflight_candidate_byte_quota(root, candidate, profile_bytes=len(original))
    candidate = omdet_bundle_profile.validate_candidate_profile(
        candidate, root, expected_pins=expected_pins)
    target = _validate_fixed_profile(root, candidate)
    device = target["device"]
    module_name = "omdet-turbo-tiny.%s.dll" % device
    if candidate["runtime"]["binary_runtime_provenance"] != "NOT_CHECKED":
        raise OmDetPublicCandidateError("candidate runtime provenance was promoted")
    _reject_external_onnx(root, _pin(candidate["source_onnx"], "source_onnx"))


    selected = {
        module_name: candidate["compiled_module"],
        module_name + ".inputs": candidate["compiled_inputs"],
        "preprocessor_config.json": candidate["hf_configs"]["processor"],
        "config.json": candidate["hf_configs"]["model"],
        "features/omdet-feature-assets.json": candidate["feature_manifest"],
        "source/omdet-turbo-tiny.onnx": candidate["source_onnx"],
        "audit/export-contract.json": candidate["export_contract"],
    }
    for role, filename in FEATURE_FILES.items():
        selected["features/" + filename] = candidate["trained_features"][role]
    for role, record in candidate["runtime"]["libraries"].items():
        filename = PurePosixPath(record["file"]).name
        if (filename.casefold() in (item.casefold() for item in selected) or
                filename.casefold() in ("manifest.txt", "manifest.json")):
            raise OmDetPublicCandidateError("runtime DLL destination collides: " + role)
        selected[filename] = record
    approved_names = set()
    source_names = set()
    projected_bytes = 0
    for name, raw in selected.items():
        bundle_atomic.validate_member_name(name)
        folded = name.casefold()
        if folded in approved_names:
            raise OmDetPublicCandidateError("duplicate case-folded destination name")
        approved_names.add(folded)
        pin = _pin(raw, name)
        source_folded = pin["file"].casefold()
        if source_folded in source_names:
            raise OmDetPublicCandidateError("source role paths overlap")
        source_names.add(source_folded)
        projected_bytes += pin["bytes"]
    projected_bytes += len(original) + 3 * _JSON_LIMIT
    if projected_bytes > _FILE_QUOTA:
        raise OmDetPublicCandidateError("selected OmDet bundle exceeds native-family byte quota")
    if os.path.lexists(final):
        raise FileExistsError("destination occupied before staging: %s" % final)


    stage = final.parent / (".crosspascal-stage-" + uuid.uuid4().hex)
    parent_holds = ExitStack()
    stage_holds = ExitStack()
    try:
        for ancestor in reversed((final.parent, *final.parent.parents)):
            parent_holds.enter_context(_hold_plain_ntfs_directory(ancestor))
        stage.mkdir(mode=0o700)
        stage_holds.enter_context(_hold_plain_ntfs_directory(stage))
        for directory in ("features", "audit", "source"):
            child = stage / directory
            child.mkdir()
            stage_holds.enter_context(_hold_plain_ntfs_directory(child))
        transferred = {}
        for name, pin in selected.items():
            transferred[name] = _copy_pinned(root, stage, pin, name)
        archived = _write_generated(
            stage, "audit/omdet-candidate-profile.v1.json", original)
        generated = {"audit/omdet-candidate-profile.v1.json": archived}
        vm_inputs = [
            {"name": name, "shape": list(shape), "layout": layout, "dtype": dtype}
            for name, shape, layout, dtype in INPUTS
        ]
        vm_outputs = [
            {"name": name, "shape": list(shape), "dtype": dtype,
             "semantics": semantics}
            for name, shape, dtype, semantics in OUTPUTS
        ]
        artifact = {
            "os": "windows", "arch": "x86_64", "device": device,
            "dtype": "float32", "file": module_name, "entry": "main",
            "format": "so", "hash": transferred[module_name]["sha256"],
        }
        if device == "cuda":
            artifact["device_arch"] = target["sm"]
        public_contract = {
            "format_version": 2, "task_kind": "object_detection",
            "adapter": "omdet", "adapter_api_version": 2,
            "artifact": artifact, "io": vm_inputs, "omdet_outputs": vm_outputs,
            "omdet": {"profile": "tiny-swin-t-hf-r7",
                      "processor_config": "preprocessor_config.json",
                      "model_config": "config.json", "feature_directory": "features"},
            "provenance": {"source": "source/omdet-turbo-tiny.onnx",
                           "source_hash": transferred["source/omdet-turbo-tiny.onnx"]["sha256"]},
        }
        generated["manifest.txt"] = _write_generated(
            stage, "manifest.txt",
            manifest.emit_manifest_text(public_contract).encode("utf-8"))
        generated["manifest.json"] = _write_generated(
            stage, "manifest.json",
            (manifest.emit_manifest_json(public_contract) + "\n").encode("utf-8"))
        audit = {
            "schema": SCHEMA, "schema_version": 2,
            "source_profile_schema": omdet_bundle_profile.SCHEMA,
            "source_profile_historic_pascal_facade": "DISABLED",
            "source_profile": archived,
            "pinned_hf_revision": REVISION,
            "transferred_members": transferred,
            "verification": "NOT_CHECKED",
            "pascal_facade_uat": "NOT_RUN",
            "independent_numeric_oracle": "NOT_RUN",
            "runtime_provenance": "NOT_ATTESTED",
            "redistribution": "NOT_CHECKED",
            "immutable_source_lock": "NOT_IMPLEMENTED",
        }
        generated["audit/public-candidate.json"] = _write_generated(
            stage, "audit/public-candidate.json", _json_output(audit))

        _reject_external_onnx(
            stage, transferred["source/omdet-turbo-tiny.onnx"])
        required = tuple(transferred) + tuple(generated)
        inventory = bundle_atomic.scan_bundle_tree(
            str(stage), max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)

        expected_files = dict(transferred, **generated)
        actual_files = {item.path: item for item in inventory.members
                        if item.kind == "file"}
        actual_directories = {item.path for item in inventory.members
                              if item.kind == "directory"}
        if (set(actual_files) != set(expected_files) or
                actual_directories != {"features", "audit", "source"}):
            raise OmDetPublicCandidateError(
                "unapproved or missing staged OmDet inventory member")
        for name, approved in expected_files.items():
            measured = actual_files[name]
            if (measured.size != approved["bytes"] or
                    measured.sha256 != approved["sha256"]):
                raise OmDetPublicCandidateError(
                    "staged member SHA/size differs from selected inventory: " + name)

        stage_holds.close()
        receipt = bundle_atomic.commit_staged_bundle(
            str(stage), str(final), inventory, required_files=required,
            max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
    except Exception as exc:
        raise OmDetPublicCandidateError(
            "publication refused; stage/public destination preserved for inspection "
            "(%s): %s" % (stage, exc)) from exc
    finally:
        stage_holds.close()
        parent_holds.close()
    return {
        "schema": SCHEMA, "destination": receipt.destination,
        "status": "NOT_CHECKED", "pascal_facade_uat": "NOT_RUN",
        "runtime_provenance": "NOT_ATTESTED", "source_v1_facade": "DISABLED",
        "publication_status": receipt.status,
        "tree_sha256": receipt.tree_sha256, "member_count": receipt.member_count,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True,
                        help="absolute NTFS directory containing validated candidate v1")
    parser.add_argument("--into", required=True,
                        help="absolute, unoccupied destination directory on NTFS")
    parser.add_argument("--expected-pins", required=False,
                        help="out-of-band JSON pinset file outside candidate and destination")
    args = parser.parse_args(argv)
    try:
        expected = None
        if args.expected_pins:
            file = Path(bundle_atomic._local_absolute(args.expected_pins))
            root, final = _paths(args.candidate_dir, args.into)
            bundle_atomic._guard_existing_ancestors(str(file.parent))
            if (os.path.normcase(os.path.abspath(str(file.parent))) !=
                    os.path.normcase(os.path.realpath(str(file.parent)))):
                raise OmDetPublicCandidateError("pinset parent path has aliases")
            for other in (root, final):
                try:
                    same_tree = os.path.commonpath(
                        (os.path.normcase(str(file)), os.path.normcase(str(other))))
                except ValueError:
                    same_tree = ""
                if same_tree == os.path.normcase(str(other)):
                    raise OmDetPublicCandidateError(
                        "independent pinset cannot be inside candidate/output")
            expected = _strict_object(
                _small_bytes(file.parent, file.name, _JSON_LIMIT),
                "out-of-band expected_pins")
        result = publish(args.candidate_dir, args.into, expected_pins=expected)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print("OMDET_PUBLIC_CANDIDATE_REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
