# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Read-only destination-side structural check for a finished model bundle.

Unlike a true native UAT, this check NEVER loads a DLL or runs inference. It
compares the recipe, an optional verified-facade manifest, the actual bundle
files, and an explicitly supplied target-machine report. Even a full static
match MUST NOT be called VERIFIED_LOCAL or VERIFIED_REMOTE.

Machine report (not an attestation):
    {"os":"windows","arch":"x86_64","gpu_sm":"sm_89",
     "cuda_driver_api":12080}

Run on the *destination* or with its freshly measured report:
    python -m pascalvision_convert.destination_preflight
        --bundle <directory> --machine-json <observed-machine-report.json>

Exit 0 means **static structural checks only**; 1 = concrete mismatch or
corrupt/incomplete bundle, 2 = missing measurement / NOT_CHECKED.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

from . import bundle_download, cuda_preflight, runtime, targets
from .recipe_file import read_recipe


_SYSTEM_EXT = {"windows": ".dll", "linux": ".so", "darwin": ".dylib"}
_VERSION = "crosspascal.destination-static-check.v1"


class BundleMismatch(ValueError):
    """A concrete incompatibility or missing/corrupt deliverable."""


class InsufficientMeasurement(ValueError):
    """A fact required to compare the destination was not measured."""


def inspect_current_machine(*, run=None):
    """Read the current machine's hardware details without loading a model.

    GPU discovery uses nvidia-smi and can fall back to a TVM device query.
    The separate Driver API probe calls nvcuda.dll or libcuda. Unavailable
    measurements are reported as unknown. Native inference is required to
    verify the bundle on this destination.
    """
    host = targets.host()
    result = {
        "os": host.get("os"),
        "arch": host.get("arch"),
        "gpus": [],
        "cuda_driver_api": None,
        "observation": {
            "source": "CURRENT_MACHINE_LOCAL_PROCESS_NOT_SIGNED",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_discovery": "NOT_CHECKED",
            "driver_api_discovery": "NOT_CHECKED",
        },
    }
    command = ["nvidia-smi", "--query-gpu=compute_cap,name",
               "--format=csv,noheader"]
    try:
        completed = (run or subprocess.run)(
            command, capture_output=True, text=True, timeout=15)
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if not line.strip():
                    continue
                capability, separator, name = line.partition(",")
                if not separator or not name.strip():
                    continue
                try:
                    sm = targets.normalize_cuda_sm(capability.strip())
                except ValueError:
                    continue
                result["gpus"].append({
                    "sm": sm, "name": name.strip(),
                    "observation": "nvidia-smi",
                })
            result["observation"]["gpu_discovery"] = "nvidia-smi"
        else:
            result["observation"]["gpu_discovery"] = (
                "nvidia-smi exited %s" % completed.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["observation"]["gpu_discovery"] = (
            "nvidia-smi unavailable or timed out: %s" % exc)
    if not result["gpus"] and host.get("cuda_arch"):
        try:
            result["gpus"].append({
                "sm": targets.normalize_cuda_sm(host["cuda_arch"]),
                "name": "TVM device 0, name not measured",
                "observation": "tvm.target.Target.from_device",
            })
        except ValueError:
            pass
    if result["gpus"]:
        api, reason = cuda_preflight.driver_api_version()
        result["cuda_driver_api"] = api
        result["observation"]["driver_api_discovery"] = reason
    return result


def _name(value, where):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or ":" in value
            or "\x00" in value):
        raise BundleMismatch("%s must be one plain bundle filename" % where)
    return value


def _read_json_file(path, where):
    if path.stat().st_size > 1024 * 1024:
        raise BundleMismatch("%s is too large for an ordinary metadata file" % where)
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (UnicodeError, ValueError) as exc:
        raise BundleMismatch("%s is not valid UTF-8 JSON" % where) from exc
    if not isinstance(result, dict):
        raise BundleMismatch("%s must contain one JSON object" % where)
    return result


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _sha_field(value, where):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)):
        raise BundleMismatch("%s requires an exact SHA-256 value" % where)
    return value.lower()


def _required_file(known, name, where, *, expected_sha=None):
    name = _name(name, where)
    item = known.get(name)
    if item is None or item[1] <= 0:
        raise BundleMismatch("%s missing or empty: %s" % (where, name))
    path = item[0]
    digest = _sha(path)
    if expected_sha is not None and digest != _sha_field(expected_sha, where):
        raise BundleMismatch("%s SHA-256 differs: %s" % (where, name))
    return {"file": name, "bytes": item[1], "sha256": digest}


def _runtime_component(known, name, ext):
    names = [name + ext, "lib" + name + ext]
    present = [candidate for candidate in names if candidate in known]
    if len(present) != 1:
        raise BundleMismatch("runtime %s has %d candidate files (expected one)"
                             % (name, len(present)))
    return _required_file(known, present[0], "runtime " + name)


def _deployment(recipe, manifest):
    profile = recipe.get("target_profile")
    if not isinstance(profile, dict):
        raise InsufficientMeasurement("recipe has no validated target_profile")
    deploy = profile.get("deployment")
    if not isinstance(deploy, dict):
        raise BundleMismatch("recipe has no deployment target profile")
    if manifest is not None:
        art = manifest.get("artifact")
        if not isinstance(art, dict):
            raise BundleMismatch("verified facade manifest has no artifact contract")
        if manifest.get("target_profile") != profile:
            raise BundleMismatch("manifest and recipe target_profile disagree")
        for field, expected in (("os", deploy.get("os")),
                                ("arch", deploy.get("arch")),
                                ("device", deploy.get("device"))):
            if art.get(field) != expected:
                raise BundleMismatch("manifest artifact %s differs from recipe" % field)
        if deploy.get("device") == "cuda":
            if art.get("device_arch") != deploy.get("sm"):
                raise BundleMismatch("manifest GPU SM differs from recipe")
        elif art.get("device_arch") not in (None, ""):
            raise BundleMismatch("CPU manifest unexpectedly names a GPU SM")
        mp = manifest.get("provenance")
        if not isinstance(mp, dict) or not mp.get("source_hash"):
            raise InsufficientMeasurement("manifest lacks source SHA provenance")
        if (not isinstance(recipe.get("source_hash"), str) or
                _sha_field(mp["source_hash"], "manifest provenance") !=
                _sha_field(recipe["source_hash"], "recipe source provenance")):
            raise BundleMismatch("recipe and manifest source SHA disagree")
    return deploy


def _compare_machine(deploy, machine):
    if not isinstance(machine, dict):
        raise InsufficientMeasurement("destination machine report is missing")
    actual_os = machine.get("os")
    actual_arch = machine.get("arch")
    if not isinstance(actual_os, str) or not actual_os.strip():
        raise InsufficientMeasurement("destination OS was not measured")
    if not isinstance(actual_arch, str) or not actual_arch.strip():
        raise InsufficientMeasurement("destination CPU architecture was not measured")
    required_os = targets._norm_os(deploy.get("os"))
    required_arch = targets._norm_arch(deploy.get("arch"))
    if required_os not in _SYSTEM_EXT:
        raise BundleMismatch("unsupported target OS: %s" % required_os)
    if targets._norm_os(actual_os) != required_os:
        raise BundleMismatch("destination OS %s differs from build %s"
                             % (actual_os, required_os))
    if targets._norm_arch(actual_arch) != required_arch:
        raise BundleMismatch("destination ISA %s differs from build %s"
                             % (actual_arch, required_arch))

    device = deploy.get("device")
    if device not in ("cpu", "cuda"):
        raise BundleMismatch("unsupported requested device: %r" % device)
    if device == "cuda":
        required_sm = targets.normalize_cuda_sm(deploy.get("sm"))
        if not required_sm:
            raise BundleMismatch("CUDA bundle has no exact target SM")
        reports = machine.get("gpus")
        if reports is None:
            reported_sm = machine.get("gpu_sm")
            if not reported_sm:
                raise InsufficientMeasurement("destination GPU SM was not measured")
            reports = [{"sm": reported_sm}]
        if not isinstance(reports, list):
            raise InsufficientMeasurement("destination GPU list is not measured")
        available = set()
        for item in reports:
            if not isinstance(item, dict) or not item.get("sm"):
                continue
            try:
                available.add(targets.normalize_cuda_sm(item["sm"]))
            except ValueError:
                continue
        if not available:
            raise InsufficientMeasurement("destination GPU SM is unknown")
        if required_sm not in available:
            raise BundleMismatch("no GPU of required SM %s; observed %s"
                                 % (required_sm, sorted(available)))
        required_driver = deploy.get("driver_api")
        reported_driver = machine.get("cuda_driver_api")
        if reported_driver in (None, ""):
            raise InsufficientMeasurement("destination CUDA Driver API was not measured")
        try:
            reported_driver = targets.normalize_driver_api(reported_driver)
        except (ValueError, TypeError) as exc:
            raise InsufficientMeasurement("destination CUDA Driver API invalid") from exc
        if required_driver is None:
            raise InsufficientMeasurement("recipe lacks a required target CUDA Driver API")
        if reported_driver < targets.normalize_driver_api(required_driver):
            raise BundleMismatch("destination driver API %s below requested %s"
                                 % (reported_driver, required_driver))
    return {"os": required_os, "arch": required_arch,
            "device": device, "sm": deploy.get("sm"),
            "driver_api": machine.get("cuda_driver_api") if device == "cuda" else None}


def _evaluate(folder, machine):
    entries = bundle_download.collect_bundle_files(
        folder, allow_download_info=True)
    known = {name: (Path(path), size) for name, path, size in entries}
    transport = {"download_metadata": "NOT_PRESENT",
                 "verified_archive_member_count": 0}
    if bundle_download.INFO_FILE in known:
        supplied = _read_json_file(
            known[bundle_download.INFO_FILE][0],
            bundle_download.INFO_FILE)
        if (type(supplied.get("schema")) is not int or
                supplied["schema"] != 1 or
                not isinstance(supplied.get("job"), str) or
                not supplied["job"] or
                supplied.get("verification_status") not in (
                    "VERIFIED_LOCAL", "NOT_CHECKED") or
                supplied.get("independent_verification_by_downloader") is not False):
            raise BundleMismatch("download transport metadata has invalid schema/status")
        declared = supplied.get("files")
        observed = set(known) - {bundle_download.INFO_FILE}
        if not isinstance(declared, dict) or set(declared) != observed:
            raise BundleMismatch("download transport member list differs from extracted files")
        for name in sorted(observed):
            metadata = declared[name]
            if (not isinstance(metadata, dict) or
                    type(metadata.get("bytes")) is not int or
                    metadata["bytes"] != known[name][1] or
                    _sha_field(metadata.get("sha256"),
                               "download metadata member SHA") !=
                    _sha(known[name][0])):
                raise BundleMismatch("download transport SHA/size mismatch: " + name)
        transport = {"download_metadata": "ALL_MEMBER_SHA256_MATCH",
                     "verified_archive_member_count": len(observed),
                     "claimed_job": supplied["job"],
                     "claimed_builder_verification_status": supplied[
                         "verification_status"],
                     "metadata_is_unsigned": True}
    if "recipe.json" not in known:
        raise BundleMismatch("bundle has no recipe.json")
    recipe = read_recipe(str(known["recipe.json"][0]))
    if not isinstance(recipe.get("recipe_version"), int):
        raise BundleMismatch("recipe version is invalid")
    manifest = (_read_json_file(known["manifest.json"][0], "manifest.json")
                if "manifest.json" in known else None)
    deploy = _deployment(recipe, manifest)

    comparison = _compare_machine(deploy, machine)
    device = comparison["device"]
    ext = _SYSTEM_EXT[comparison["os"]]
    if manifest is None:
        if recipe.get("contract") not in (None, {}):
            raise BundleMismatch("a requested Pascal facade has no verified manifest")
        name = _name(recipe.get("name"), "recipe.name") + "." + device + ext
        facade_scope = "RAW_TENSORS_NO_FACADE_PASSPORT"
        module = _required_file(known, name, "compiled raw model")
    else:
        artifact = manifest["artifact"]
        if not artifact.get("hash"):
            raise BundleMismatch(
                "verified facade passport lacks SHA-256 of its built artifact")
        module = _required_file(known, artifact.get("file"),
                                "manifest artifact", expected_sha=artifact.get("hash"))
        if "manifest.txt" not in known:
            raise BundleMismatch("manifest.json exists without its Pascal text passport")
        _required_file(known, "manifest.txt", "verified facade manifest text")
        if recipe.get("contract") in (None, {}):
            raise BundleMismatch("verified facade manifest but recipe has no contract")
        facade_scope = "PASSED_ORIGINAL_BUILDER_CONTRACT_NOT_DESTINATION_TEST"

    _required_file(known, module["file"] + ".inputs", "model input ABI")
    _required_file(known, "recipe.json", "build recipe")
    _required_file(known, "report.txt", "build report")
    runtime_names = ["tvm_runtime", "tvm_runtime_extra", "tvm_ffi"]
    if device == "cuda":
        runtime_names.append("tvm_runtime_cuda")
    libraries = [_runtime_component(known, component, ext)
                 for component in runtime_names]
    licences = []
    for name in sorted(known):
        if name.startswith("LICENSE-tvm-"):
            licences.append(_required_file(known, name, "redistribution licence"))
        if device == "cuda" and runtime._CUDA_DEPENDENCY.fullmatch(name):
            raise BundleMismatch("unattested standalone CUDA dependency: " + name)
    if not licences:
        raise BundleMismatch("TVM licence files missing from bundle")
    if manifest is not None:
        try:
            bundle_download._validate_pascal_passport_pair(entries)
        except bundle_download.UnsafeBundle as exc:
            raise BundleMismatch(str(exc)) from exc
    return {
        "status": "STATIC_PROFILE_MATCH_NOT_EXECUTED",
        "acceptance_status": "NOT_CHECKED",
        "verified_on_destination": False,
        "native_inference_executed": False,
        "independent_numerical_comparison_executed": False,
        "machine_report_source": (
            machine.get("observation", {}).get("source",
                                                "CALLER_SUPPLIED_NOT_ATTESTED")
            if isinstance(machine.get("observation"), dict)
            else "CALLER_SUPPLIED_NOT_ATTESTED"),
        "runtime_abi_execution": "NOT_CHECKED",
        "source_model_bytes_revalidated_here": False,
        "download_transport": transport,
        "target": comparison,
        "facade": facade_scope,
        "model": module,
        "runtime_libraries": libraries,
        "redistribution_licences": licences,
        "remaining": (
            "Run an independent model oracle and confirm native process exit "
            "on this exact target; this report is NOT a verified deployment."),
    }


def inspect_bundle(folder, machine):
    """Never return VERIFIED_* even for a verified BUILDER's manifest."""
    try:
        result = _evaluate(folder, machine)
    except (InsufficientMeasurement, ValueError, TypeError, OSError) as exc:
        not_checked = isinstance(exc, InsufficientMeasurement)
        result = {
            "status": "NOT_CHECKED" if not_checked else "BLOCKED",
            "reason": str(exc),
        }
    result.setdefault("acceptance_status", "NOT_CHECKED")
    result["schema"] = _VERSION
    result["verified_on_destination"] = False
    result["native_inference_executed"] = False
    result["independent_numerical_comparison_executed"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    machine_choice = parser.add_mutually_exclusive_group(required=True)
    machine_choice.add_argument("--machine-json")
    machine_choice.add_argument(
        "--current-machine", action="store_true",
        help="probe THIS destination OS/CPU/CUDA, then compare its actual profile")
    args = parser.parse_args(argv)
    if args.current_machine:
        report = inspect_bundle(args.bundle, inspect_current_machine())
    else:
        path = Path(args.machine_json)
        if path.is_symlink() or not path.is_file():
            report = {"status": "NOT_CHECKED",
                      "reason": "machine report file missing or symlink"}
        else:
            try:
                report = inspect_bundle(
                    args.bundle, _read_json_file(path, "machine report"))
            except (OSError, ValueError) as exc:
                report = {"status": "NOT_CHECKED", "reason": str(exc)}
    report["schema"] = _VERSION
    report["acceptance_status"] = "NOT_CHECKED"
    report["verified_on_destination"] = False
    report["native_inference_executed"] = False
    report["independent_numerical_comparison_executed"] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return (0 if report["status"] == "STATIC_PROFILE_MATCH_NOT_EXECUTED"
            else 2 if report["status"] == "NOT_CHECKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
