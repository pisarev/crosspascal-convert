# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

from . import (birefnet_hr800_snapshot, hf_held_stage, pipeline, runtime,
               source_freeze, source_lock, targets)


SCHEMA = "crosspascal.birefnet-hr800.consumer-source.v1"
MAX_CONFIG_BYTES = 1024 * 1024
MAX_BLOCK_BYTES = 1024 * 1024
MAX_COMPILED_MODULE_BYTES = 16 * 1024 ** 3
MAX_INPUT_MANIFEST_BYTES = 65536
_SHA = re.compile(r"[0-9a-f]{64}\Z")

TRAINED_HR800_ORIGINAL_ONNX = (
    932218277, "99c984d124ee1b68aaccbdfc715c1a9b5025b769f8b49d296eaf2ba028d247e6")
TRAINED_HR800_WIN64_CPU_RUNTIME_PINS = {
    "tvm_runtime.dll": (
        2172416, "6cf782d0f46612a43943ed446203d40146391f01d733408ba7f6f9b9da5955ff"),
    "tvm_runtime_extra.dll": (
        902144, "03985616af6092f97b9d6cbcb1d1f841f6ed5a78eed4e84acc2a883cc90a682e"),
    "tvm_ffi.dll": (
        4215296, "45ac8b984f0ca0662a3fa5e45bae8cf0c0fdb63a05d6b05250edd54783314a94"),
    "LICENSE-tvm-LICENSE": (
        12308, "7cd3b9cb57596d291cc44488f7c55b5a91ea652dc2bb0ca6098d5508765c23e5"),
}


class HR800RouteRefused(ValueError):
    pass


def _positive_source(path, role):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise HR800RouteRefused("%s must be an exact absolute model resource" % role)
    chosen = Path(os.path.abspath(path))
    source_freeze._safe_ancestors(chosen)
    if not stat.S_ISREG(source_freeze._lstat(chosen).st_mode):
        raise HR800RouteRefused("%s selected resource is not a regular file" % role)
    return chosen


def _private_stage_parent(source_root, requested_output):
    if not isinstance(requested_output, str) or not os.path.isabs(requested_output):
        raise HR800RouteRefused("--into must be an absolute new output directory")
    output = Path(os.path.abspath(requested_output))


    source_freeze._safe_name(output.name)
    if os.path.lexists(output):
        raise FileExistsError("HR800 selected output already exists: " + str(output))
    parent = output.parent
    source_freeze._safe_ancestors(parent)
    if not stat.S_ISDIR(source_freeze._lstat(parent).st_mode):
        raise HR800RouteRefused("HR800 output parent must be an existing regular directory")

    a, b = os.path.normcase(str(source_root)), os.path.normcase(str(parent))
    try:
        overlaps = os.path.commonpath((a, b)) == a
    except ValueError:
        overlaps = False
    if overlaps:
        raise HR800RouteRefused(
            "private HR800 source stage would be inside selected owner tree")
    return output, parent


@contextmanager
def _hold_output_directory(path, *, final_receipt=None,
                           final_receipt_name="birefnet-selected-source.json"):
    if not source_lock._is_windows():
        raise HR800RouteRefused("native Windows NTFS output directory custody required")
    api = source_lock._Kernel32()
    parent = path.parent
    api.assert_local_ntfs(parent)
    source_freeze._safe_ancestors(parent)
    parent_first = source_freeze._lstat(parent)
    if not stat.S_ISDIR(parent_first.st_mode):
        raise HR800RouteRefused("new HR800 output parent is not a directory")
    with ExitStack() as locks:
        parent_handle = api.kernel.CreateFileW(
            str(parent), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if parent_handle == source_lock.INVALID_HANDLE_VALUE:
            raise HR800RouteRefused("cannot hold HR800 output parent before mkdir")
        locks.callback(api.close, parent_handle)
        parent_native = api.FileInfo()

        def audit_parent():
            if not api.kernel.GetFileInformationByHandle(
                    parent_handle, ctypes.byref(parent_native)):
                raise HR800RouteRefused("cannot inspect held HR800 output parent")
            parent_fileid = ((parent_native.nFileIndexHigh << 32) |
                             parent_native.nFileIndexLow)
            if (not (parent_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_DIRECTORY)
                    or (parent_native.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT)
                    or parent_fileid != parent_first.st_ino):
                raise HR800RouteRefused("held HR800 output parent FileID/reparse differs")
            source_freeze._safe_ancestors(parent)
            observed = source_freeze._lstat(parent)
            if (observed.st_dev, observed.st_ino) != (
                    parent_first.st_dev, parent_first.st_ino):
                raise HR800RouteRefused("HR800 output parent path was redirected")

        audit_parent()


        path.mkdir(exist_ok=False)
        initial = source_freeze._lstat(path)
        if not stat.S_ISDIR(initial.st_mode):
            raise HR800RouteRefused("new HR800 output root is not a directory")
        handle = api.kernel.CreateFileW(
            str(path), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if handle == source_lock.INVALID_HANDLE_VALUE:
            raise HR800RouteRefused("failed to hold HR800 output namespace")
        locks.callback(api.close, handle)

        def audit():
            audit_parent()
            info = api.FileInfo()
            if not api.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise HR800RouteRefused("cannot inspect held HR800 output directory")
            current = source_freeze._lstat(path)
            file_id = (info.nFileIndexHigh << 32) | info.nFileIndexLow
            if (not (info.dwFileAttributes & source_lock.FILE_ATTRIBUTE_DIRECTORY)
                    or (info.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT)
                    or info.dwVolumeSerialNumber != parent_native.dwVolumeSerialNumber
                    or current.st_dev != initial.st_dev
                    or current.st_ino != initial.st_ino
                    or current.st_ino != file_id):
                raise HR800RouteRefused("HR800 output directory identity redirected")
            source_freeze._safe_ancestors(path)
        audit()
        yield api, locks
        audit()
        if final_receipt is not None:
            if len(final_receipt) != 1:
                raise HR800RouteRefused("HR800 final receipt was not approved for publication")
            public_path, public_bytes = final_receipt[0]
            if (final_receipt_name not in (
                    "birefnet-selected-source.json",
                    "omdet-selected-compiler-source.json") or
                    public_path != path / final_receipt_name or
                    type(public_bytes) is not bytes or
                    not 0 < len(public_bytes) <= MAX_CONFIG_BYTES):
                raise HR800RouteRefused("HR800 final receipt publication contract differs")

            _write_receipt_bytes_exclusive(public_path, public_bytes)


def _copy_held_sidecar(stage_file, output_file, expected_sha, expected_size):
    if type(expected_size) is not int or not 0 < expected_size <= MAX_CONFIG_BYTES:
        raise HR800RouteRefused("bad selected HR800 resource byte limit")
    source_freeze._safe_ancestors(stage_file)
    selected = source_freeze._lstat(stage_file)
    if (not stat.S_ISREG(selected.st_mode)
            or getattr(selected, "st_nlink", 1) != 1
            or selected.st_size != expected_size):
        raise HR800RouteRefused("HR800 private sidecar changed before publication")
    digest = hashlib.sha256()
    with stage_file.open("rb") as source, output_file.open("xb") as dest:
        first = os.fstat(source.fileno())
        if ((first.st_dev, first.st_ino, first.st_size)
                != (selected.st_dev, selected.st_ino, expected_size)):
            raise HR800RouteRefused("HR800 selected sidecar descriptor differs")
        remaining = expected_size
        while remaining:
            block = source.read(min(MAX_BLOCK_BYTES, remaining))
            if not block:
                raise HR800RouteRefused("HR800 selected sidecar unexpectedly truncated")
            dest.write(block)
            digest.update(block)
            remaining -= len(block)
        if source.read(1) or digest.hexdigest() != expected_sha:
            raise HR800RouteRefused("HR800 sidecar changed during exact copy")
        after = os.fstat(source.fileno())
        if ((first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise HR800RouteRefused("HR800 sidecar modified during copy")
        dest.flush()
        os.fsync(dest.fileno())


def _pin_output_sidecar(api, locks, path, member):
    source_freeze._safe_ancestors(path)
    handle = api.open(path)
    locks.callback(api.close, handle)
    native = api.identity(handle)
    info = api.FileInfo()
    if (not api.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)) or
            info.nNumberOfLinks != 1):
        raise HR800RouteRefused("HR800 output JSON leaf has hardlink/invalid FileID")
    hf_held_stage._native_check(
        api, path, handle, native, member["bytes"], member["sha256"],
        MAX_BLOCK_BYTES)
    return path, handle, native, member


def _audit_pinned_output_sidecars(api, leaves):
    for path, handle, native, member in leaves:
        hf_held_stage._native_check(
            api, path, handle, native, member["bytes"], member["sha256"],
            MAX_BLOCK_BYTES)


def _write_receipt_bytes_exclusive(path, body):
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _pin_compiler_output(api, locks, path, *, max_bytes):
    source_freeze._safe_ancestors(path)
    before = source_freeze._lstat(path)
    if (not stat.S_ISREG(before.st_mode) or
            getattr(before, "st_nlink", 1) != 1 or
            not 0 < before.st_size <= max_bytes):
        raise HR800RouteRefused("HR800 compiled output leaf missing/linked/oversized")
    handle = api.open(path)
    locks.callback(api.close, handle)
    native = api.identity(handle)
    info = api.FileInfo()
    if (not api.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)) or
            info.nNumberOfLinks != 1 or
            native.file_bytes != before.st_size or
            native.file_id != before.st_ino):
        raise HR800RouteRefused("HR800 compiled output FileID/hardlink changed")
    digest, size = hf_held_stage._hash_win32_handle(
        api, handle, block_bytes=MAX_BLOCK_BYTES, expected_bytes=before.st_size)
    entry = {"bytes": size, "sha256": digest}
    hf_held_stage._native_check(
        api, path, handle, native, size, digest, MAX_BLOCK_BYTES)
    return path, handle, native, entry


def _prestage_exact_trained_hr800_win64_cpu_runtime(output, api, locks):
    if os.name != "nt":
        raise HR800RouteRefused("HR800 exact CPU wheel pre-stage requires Win32")
    inventory, libraries, licences = runtime._plan_runtime_files(
        "cpu", target_dir=str(output))
    rows = libraries + licences
    if (inventory.get("ready") is not True or inventory.get("missing") or
            inventory.get("blocked_existing") or inventory.get("existing_verified") or
            inventory.get("unmanaged_cuda_files") or
            len(rows) != len(TRAINED_HR800_WIN64_CPU_RUNTIME_PINS) or
            {row.get("file") for row in rows} !=
            set(TRAINED_HR800_WIN64_CPU_RUNTIME_PINS)):
        raise HR800RouteRefused("exact trained HR800 CPU wheel/license inventory refused")

    for row in rows:
        filename = row["file"]
        size, sha = TRAINED_HR800_WIN64_CPU_RUNTIME_PINS[filename]
        source = row.get("source")
        if (row.get("bytes") != size or row.get("sha256") != sha or
                not isinstance(source, str) or not os.path.isabs(source)):
            raise HR800RouteRefused("trained HR800 CPU wheel source pin differs: " + filename)
        path = Path(source)
        source_freeze._safe_ancestors(path)
        info = source_freeze._lstat(path)
        if (not stat.S_ISREG(info.st_mode) or
                getattr(info, "st_nlink", 1) != 1 or info.st_size != size or
                os.path.lexists(output / filename)):
            raise HR800RouteRefused(
                "trained HR800 CPU wheel source/destination unsafe: " + filename)

    pinned = []
    for row in rows:
        filename = row["file"]
        size, sha = TRAINED_HR800_WIN64_CPU_RUNTIME_PINS[filename]
        source = Path(row["source"])
        destination = output / filename
        digest = hashlib.sha256()
        with source.open("rb") as reader, destination.open("xb") as writer:
            first = os.fstat(reader.fileno())
            if (source_freeze._descriptor_identity(first) !=
                    source_freeze._descriptor_identity(source_freeze._lstat(source)) or
                    first.st_size != size or getattr(first, "st_nlink", 1) != 1):
                raise HR800RouteRefused("HR800 wheel source changed before copy")
            remaining = size
            while remaining:
                block = reader.read(min(MAX_BLOCK_BYTES, remaining))
                if not block:
                    raise HR800RouteRefused("HR800 wheel source shortened during copy")
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
            if (reader.read(1) or digest.hexdigest() != sha or
                    source_freeze._descriptor_identity(os.fstat(reader.fileno())) !=
                    source_freeze._descriptor_identity(first)):
                raise HR800RouteRefused("HR800 wheel source changed during copy")
            writer.flush()
            os.fsync(writer.fileno())
        pin = _pin_compiler_output(api, locks, destination, max_bytes=size)
        if pin[3] != {"bytes": size, "sha256": sha}:
            raise HR800RouteRefused("HR800 wheel member changed before Win32 pin")
        pinned.append(pin)
    _audit_pinned_output_sidecars(api, pinned)
    return pinned


def _check_no_foreign_loadables(output, compiled_module, target, *, exact_wheel):
    expected = {compiled_module.name.casefold()}
    if exact_wheel:
        expected.update(name.casefold()
                        for name in TRAINED_HR800_WIN64_CPU_RUNTIME_PINS
                        if name.lower().endswith(".dll"))
    else:
        components = runtime.REQUIRED + runtime.REQUIRED_FFI
        if target == "cuda":
            components += runtime.FOR_GPU
        ext = runtime.library_extension()
        for component in components:
            expected.add((component + ext).casefold())
            expected.add(("lib" + component + ext).casefold())
    with os.scandir(output) as files:
        for count, entry in enumerate(files, 1):
            if count > 256:
                raise HR800RouteRefused(
                    "HR800 output has too many sidecars to audit before verification")
            name = entry.name
            if not name.casefold().endswith(
                    (".dll", ".pyd", ".exe", ".ocx", ".cpl", ".so", ".dylib")):
                continue
            if name.casefold() not in expected:
                raise HR800RouteRefused(
                    "unapproved loadable HR800 output child: " + name)
            path = output / name
            source_freeze._safe_ancestors(path)
            info = source_freeze._lstat(path)
            if (not stat.S_ISREG(info.st_mode) or
                    getattr(info, "st_nlink", 1) != 1):
                raise HR800RouteRefused(
                    "linked/nonregular HR800 loadable output child: " + name)


def _trusted_tasklist_path():
    try:
        return source_lock._trusted_tasklist_path()
    except source_lock.SourceLockUnavailable as exc:
        raise HR800RouteRefused(str(exc)) from exc


def _admit_hr800_heavy_source_resources(model_bytes):
    if model_bytes <= 64 * 1024 ** 2:
        return
    from . import detect
    free = detect._available_physical_memory_bytes()


    need = max(5 * 1024 ** 3,
               detect.ONNX_PREPARSE_RAM_MULTIPLIER * model_bytes +
               detect.ONNX_PREPARSE_RAM_HEADROOM_BYTES)
    if free is None or free < need:
        raise HR800RouteRefused(
            "large selected HR800 ONNX/native compiler requires >=5 GiB "
            "available physical RAM and ONNX parse quota %d bytes; observed %s"
            % (need, free))
    tasklist = _trusted_tasklist_path()
    try:
        observed = subprocess.run(
            [str(tasklist), "/NH", "/FO", "CSV"], capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=15, check=False, cwd=str(tasklist.parent))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HR800RouteRefused(
            "cannot establish absence of competing native compute processes") from exc
    if observed.returncode != 0 or not observed.stdout.strip():
        raise HR800RouteRefused("HR800 native process inventory unavailable")
    seen_pid = False
    competitors = {
        "python.exe", "pythonw.exe", "pytest.exe", "dcc32.exe", "dcc64.exe",
        "cl.exe", "link.exe", "msbuild.exe", "ninja.exe", "cmake.exe",
        "backup.exe", "robocopy.exe",
    }
    for row in csv.reader(observed.stdout.splitlines()):
        if len(row) < 2 or not row[1].strip().isdecimal():
            continue
        seen_pid = True
        if (row[0].strip().casefold() in competitors and
                int(row[1].strip()) != os.getpid()):
            raise HR800RouteRefused(
                "competing compute process prevents HR800 native compiler: " +
                row[0].strip())
    if not seen_pid:
        raise HR800RouteRefused("HR800 process enumeration returned no usable PID")


def preview_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pascalvision-convert birefnet-hr800-preview",
        description="Read-only selected HR800 SwinL resources and approval SHA")
    parser.add_argument("--model", required=True, metavar="ABS_ONNX")
    parser.add_argument("--config", required=True, metavar="ABS_JSON")
    parser.add_argument("--processor", required=True, metavar="ABS_JSON")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return 0 if error.code == 0 else 1
    try:
        selected = birefnet_hr800_snapshot.scan_selected_hr800(
            _positive_source(args.model, "model"),
            _positive_source(args.config, "config"),
            _positive_source(args.processor, "processor"))
        print(json.dumps(selected, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False))
        return 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print("HR800 PREVIEW REFUSED, NOT_CHECKED: %s" % exc, file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pascalvision-convert birefnet-hr800-held",
        description="Explicit selected full-HR800 SwinL Win32-held source to TVM core; "
                    "NOT_CHECKED, not a trained-model acceptance")
    parser.add_argument("--model", required=True, metavar="ABS_ONNX")
    parser.add_argument("--config", required=True, metavar="ABS_JSON")
    parser.add_argument("--processor", required=True, metavar="ABS_JSON")
    parser.add_argument("--approved-tree-sha256", required=True, metavar="SHA256")
    parser.add_argument("--into", required=True, metavar="ABS_EMPTY_DIRECTORY")
    parser.add_argument("--target", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sm", default="", help="selected CUDA SM; never CPU fallback")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return 0 if error.code == 0 else 1

    try:
        if not isinstance(args.approved_tree_sha256, str) or (
                _SHA.fullmatch(args.approved_tree_sha256) is None):
            raise HR800RouteRefused("exact approved lowercase HR800 tree SHA required")
        model = _positive_source(args.model, "model")
        config = _positive_source(args.config, "config")
        processor = _positive_source(args.processor, "processor")
        output, parent = _private_stage_parent(model.parent, args.into)


        if not source_lock._is_windows():
            raise HR800RouteRefused("Win32 required for strict selected HR800 custody")
        req_sm = targets.validate_target_sm(args.target, args.sm or None)
        from . import build_plan
        plan = build_plan.resolve(
            targets.host(), route="raw", source=str(model),
            target=args.target, req_sm=req_sm)
        selected = birefnet_hr800_snapshot.scan_selected_hr800(
            model, config, processor)
        if selected["tree_sha256"] != args.approved_tree_sha256:
            raise HR800RouteRefused("HR800 selected resources changed from approved tree SHA")
        if selected["model_profile"] != "FULL_HR800_SWINL_CALLER_SELECTED_NOT_ATTESTED":
            raise HR800RouteRefused("selected source must be full HR800 SwinL")
        selected_model = next(row for row in selected["entries"]
                              if row["role"] == "model")
        _admit_hr800_heavy_source_resources(selected_model["bytes"])
        if shutil.disk_usage(parent).free < (
                selected["total_bytes"] + 256 * 1024 ** 2):
            raise HR800RouteRefused("insufficient disk space for private HR800 source stage")

        final_receipt = []
        with _hold_output_directory(output, final_receipt=final_receipt) as (
                output_api, output_locks):
            copied_leaves = []
            compiled_leaves = []
            selected_cpu_runtime = []
            expected_module = output / (
                "model." + args.target + pipeline.module_extension())
            expected_inputs = output / (expected_module.name + ".inputs")

            def hold_compiled_at_build_done(step):
                if (getattr(step, "name", None) != "Building the module" or
                        getattr(step, "state", None) != "done"):
                    return
                if compiled_leaves:
                    raise HR800RouteRefused(
                        "HR800 repeated completed compiler output callback")
                _check_no_foreign_loadables(
                    output, expected_module, args.target,
                    exact_wheel=bool(selected_cpu_runtime))
                module_pin = _pin_compiler_output(
                    output_api, output_locks, expected_module,
                    max_bytes=MAX_COMPILED_MODULE_BYTES)
                inputs_pin = _pin_compiler_output(
                    output_api, output_locks, expected_inputs,
                    max_bytes=MAX_INPUT_MANIFEST_BYTES)
                compiled_leaves.extend((module_pin, inputs_pin))

            with birefnet_hr800_snapshot.hold_selected_hr800(
                    model, config, processor, stage_parent=parent,
                    approved_tree_sha256=selected["tree_sha256"]) as held:
                original_model = selected_model
                if (args.target == "cpu" and
                        (original_model["bytes"], original_model["sha256"]) ==
                        TRAINED_HR800_ORIGINAL_ONNX):
                    selected_cpu_runtime = _prestage_exact_trained_hr800_win64_cpu_runtime(
                        output, output_api, output_locks)
                compiled = pipeline.run(
                    held.compiler_input, str(output), target=args.target,
                    req_sm=req_sm, source=str(model),
                    approved_build_plan=plan,
                    on_step=hold_compiled_at_build_done)
                if compiled.sound is not True:
                    raise HR800RouteRefused(
                        "HR800 selected compiler/Verify pipeline refused or stopped early")
                if len(compiled_leaves) != 2:
                    raise HR800RouteRefused(
                        "HR800 compiler output was not Win32-held before verification")
                _check_no_foreign_loadables(
                    output, expected_module, args.target,
                    exact_wheel=bool(selected_cpu_runtime))

                summary = getattr(compiled, "summary", None)
                if (not isinstance(summary, dict) or
                        summary.get("module") != str(expected_module) or
                        summary.get("directory") != str(output) or
                        summary.get("build_plan") != plan):
                    raise HR800RouteRefused(
                        "HR800 compiler report does not bind selected output/BuildPlan")
                source_freeze._safe_ancestors(expected_module)
                module_info = source_freeze._lstat(expected_module)
                if not stat.S_ISREG(module_info.st_mode) or module_info.st_size <= 0:
                    raise HR800RouteRefused("HR800 selected compiler artifact missing")

                _audit_pinned_output_sidecars(output_api, compiled_leaves)
                input_name, input_shape = pipeline._read_first_input(
                    str(expected_inputs))
                if input_name not in ("x", "input") or input_shape != [1, 3, 800, 800]:
                    raise HR800RouteRefused(
                        "HR800 compiled native input ABI must be x/input float32 NCHW 800x800")
                for role, filename in (("config", "birefnet-model-config.json"),
                                       ("processor", "birefnet-processor.json")):
                    member = next(row for row in selected["entries"]
                                  if row["role"] == role)
                    _copy_held_sidecar(
                        held.stage / member["path"], output / filename,
                        member["sha256"], member["bytes"])
                    copied_leaves.append(_pin_output_sidecar(
                        output_api, output_locks, output / filename, member))

            _audit_pinned_output_sidecars(output_api, copied_leaves)
            _audit_pinned_output_sidecars(output_api, compiled_leaves)
            _audit_pinned_output_sidecars(output_api, selected_cpu_runtime)
            _check_no_foreign_loadables(
                output, expected_module, args.target,
                exact_wheel=bool(selected_cpu_runtime))
            receipt = {
                "schema": SCHEMA,
                "status": "POST_WIN32_HELD_SNAPSHOT_NOT_RELEASE_GRADE",
                "model_profile": selected["model_profile"],
                "selected_source_tree_sha256": selected["tree_sha256"],
                "selected_model_sha256": held.selected_model_sha256,
                "selected_resource_members": selected["entries"],
                "copied_resource_members": {
                    "birefnet-model-config.json": next(
                        {"bytes": row["bytes"], "sha256": row["sha256"]}
                        for row in selected["entries"] if row["role"] == "config"),
                    "birefnet-processor.json": next(
                        {"bytes": row["bytes"], "sha256": row["sha256"]}
                        for row in selected["entries"] if row["role"] == "processor"),
                },
                "compiler_output_pinned_members": {
                    path.name: dict(entry)
                    for path, _handle, _native, entry in compiled_leaves
                },
                "selected_revision_claim": selected["source_revision"],
                "source_revision_status": "NOT_ATTESTED",
                "compiler_input_private": True,
                "compiled_input_name": input_name,
                "compiled_input_shape": input_shape,
                "pipeline_sound": compiled.sound is True,
                "exact_trained_cpu_wheel_pre_staged": {
                    path.name: dict(info)
                    for path, _handle, _native, info in selected_cpu_runtime
                },
                "stage_retained": True,
                "verification_status": "NOT_CHECKED",
                "native_pascal_run": "NOT_RUN",
                "independent_model_reference": "NOT_ATTESTED",
                "model_license_status": "NOT_ATTESTED",
                "release_grade_immutable": False,
            }

            receipt_path = output / "birefnet-selected-source.json"
            receipt_bytes = (json.dumps(
                receipt, sort_keys=True, indent=2, ensure_ascii=False,
                allow_nan=False) + "\n").encode("utf-8")
            if not 0 < len(receipt_bytes) <= MAX_CONFIG_BYTES:
                raise HR800RouteRefused("HR800 selected receipt exceeds byte quota")
            pending_path = output / "birefnet-selected-source.pending.json"
            pending_receipt = dict(receipt)
            pending_receipt["status"] = "PENDING_HR800_POST_AUDIT_NOT_ACCEPTED"
            pending_bytes = (json.dumps(
                pending_receipt, sort_keys=True, indent=2, ensure_ascii=False,
                allow_nan=False) + "\n").encode("utf-8")
            _write_receipt_bytes_exclusive(pending_path, pending_bytes)
            pinned_pending = _pin_compiler_output(
                output_api, output_locks, pending_path,
                max_bytes=MAX_CONFIG_BYTES)
            if pinned_pending[3] != {
                    "bytes": len(pending_bytes),
                    "sha256": hashlib.sha256(pending_bytes).hexdigest()}:
                raise HR800RouteRefused("HR800 pending receipt changed before custody")

            _audit_pinned_output_sidecars(output_api, copied_leaves)
            _audit_pinned_output_sidecars(output_api, compiled_leaves)
            _audit_pinned_output_sidecars(output_api, selected_cpu_runtime)
            _audit_pinned_output_sidecars(output_api, [pinned_pending])
            _check_no_foreign_loadables(
                output, expected_module, args.target,
                exact_wheel=bool(selected_cpu_runtime))

            final_receipt.append((receipt_path, receipt_bytes))
        print("HR800 HELD SOURCE CONSUMED; native/Pascal/original-ORT "
              "acceptance NOT_CHECKED (advisory exit 2)")
        return 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print("HR800 HELD SOURCE REFUSED, NOT_CHECKED: %s" % exc, file=sys.stderr)
        return 1

