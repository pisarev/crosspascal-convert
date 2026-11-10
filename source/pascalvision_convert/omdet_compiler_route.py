# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys

from . import (birefnet_hr800_route as held_output, build_plan,
               omdet_bundle_profile, omdet_compiler_hold, omdet_public_candidate,
               pipeline, runtime, source_freeze, targets)


SCHEMA = "crosspascal.omdet.compiler-consumer-source.v1"
MAX_DECLARATION_BYTES = 131072
INPUT_ROWS = omdet_public_candidate.INPUTS
STAGED_ROLES = omdet_compiler_hold.STAGED_ROLES
CONSUMER_RESOURCE_DESTINATIONS = {
    "source_onnx": "source/omdet-turbo-tiny.onnx",
    "export_contract": "audit/export-contract.json",
    "feature_manifest": "features/omdet-feature-assets.json",
    "processor": "preprocessor_config.json",
    "model": "config.json",
    "class_features": "features/class_features.f32le",
    "task_features": "features/task_features.f32le",
    "task_mask": "features/task_mask.i64le",
}
TRAINED_DIRECT_ONNX = (215936399, "56a06084c6170770e547ecad84eb44c4e2cbed5ba5d828634c2857ae57384a77")
TRAINED_DIRECT_ASSETS = {
    "processor": (492, "cc6c8ac92ba1a8a62065f3439ebb58f01e4895fa038901e1df14c3427a617e23"),
    "model": (1959, "c0d6546428c6dd7e72d9c8e2123cdd7ff24925dfbff9b535b66cd2f19ca65c46"),
    "features": (2023, "aef1f159663b02748df0f00732015f0273c98bde8b65ddaf88ad7b9ed160e0f6"),
    "class_features": (16384, "7b3eca3894a221f28f927a47719dd8ef89e2efc5f03e42ec66a6fc3f3ade64e2"),
    "task_features": (38912, "af52f13935719717b0c2640dac7677c287e1da1254cf3738db78b4be50038b7c"),
    "task_mask": (152, "71abaaa065c52e925b405818f0e8dfc4af012a3dd317f0dacf647e88ec000013"),
}
TRAINED_DIRECT_CONTRACT = (863, "9982c3e133c81f65c77ca9af058b7c51a38f7aafd6a1d12c7d926f5c5a9678d3")
TRAINED_DIRECT_NPZ_SHA = "906f0fc89f1627da2017e99440cd99fd36cd6f803b93e2bfb4398b8686383e43"

TRAINED_WIN64_CPU_RUNTIME_WHEEL_PINS = {
    "tvm_runtime.dll": (2172416, "6cf782d0f46612a43943ed446203d40146391f01d733408ba7f6f9b9da5955ff"),
    "tvm_runtime_extra.dll": (902144, "03985616af6092f97b9d6cbcb1d1f841f6ed5a78eed4e84acc2a883cc90a682e"),
    "tvm_ffi.dll": (4215296, "45ac8b984f0ca0662a3fa5e45bae8cf0c0fdb63a05d6b05250edd54783314a94"),
    "LICENSE-tvm-LICENSE": (12308, "7cd3b9cb57596d291cc44488f7c55b5a91ea652dc2bb0ca6098d5508765c23e5"),
}


def _direct_revision_pins(pins):
    trained = ((pins["source_onnx"]["bytes"], pins["source_onnx"]["sha256"])
               == TRAINED_DIRECT_ONNX)
    if trained:
        if (pins["export_contract"]["bytes"], pins["export_contract"]["sha256"]) != TRAINED_DIRECT_CONTRACT:
            raise OmDetCompilerRouteRefused("trained OmDet export contract differs from approved ONNX revision")
        return TRAINED_DIRECT_ASSETS, TRAINED_DIRECT_NPZ_SHA
    return (omdet_public_candidate.ASSET_PINS,
            "09d7c6178b5ce0f2cd2a61a8f8976987b48475ddf3ff307b6f1783da120be959")


class OmDetCompilerRouteRefused(ValueError):
    pass


def _unique_pinset(root: Path, output: Path, pin_file):
    if type(pin_file) is not str or not os.path.isabs(pin_file):
        raise OmDetCompilerRouteRefused("OmDet independent pinset needs absolute path")
    path = Path(os.path.abspath(pin_file))
    source_freeze._safe_ancestors(path)
    info = source_freeze._lstat(path)
    if (not stat.S_ISREG(info.st_mode) or
            getattr(info, "st_nlink", 1) != 1 or
            not 0 < info.st_size <= MAX_DECLARATION_BYTES):
        raise OmDetCompilerRouteRefused("OmDet expected_pins JSON missing/oversized")


    if (os.path.normcase(os.path.abspath(str(path.parent))) !=
            os.path.normcase(os.path.realpath(str(path.parent)))):
        raise OmDetCompilerRouteRefused(
            "OmDet expected_pins parent has an 8.3/redirected path alias")
    for ancestor in path.parents:
        if os.path.samefile(ancestor, root):
            raise OmDetCompilerRouteRefused(
                "OmDet independent expected_pins resolves inside candidate owner")


    for protected in (root, output):
        try:
            inside = (os.path.commonpath((os.path.normcase(str(path)),
                                           os.path.normcase(str(protected)))) ==
                      os.path.normcase(str(protected)))
        except ValueError:
            inside = False
        if inside:
            raise OmDetCompilerRouteRefused(
                "OmDet independently supplied expected_pins is self-signed")
    data = omdet_public_candidate._small_bytes(
        path.parent, path.name, MAX_DECLARATION_BYTES)
    return omdet_public_candidate._strict_object(
        data, "independent expected_pins"), hashlib.sha256(data).hexdigest()


def _selection(args):
    if args.source_dir is not None:
        return _selection_direct_eight(args)
    root, output = omdet_public_candidate._paths(args.candidate_dir, args.into)
    if os.path.lexists(output):
        raise FileExistsError("OmDet selected compiler output is occupied")
    expected_pins, pinset_sha = _unique_pinset(
        root, output, args.expected_pins)
    profile_bytes = omdet_public_candidate._small_bytes(
        root, omdet_public_candidate.PROFILE_NAME, MAX_DECLARATION_BYTES)
    profile = omdet_public_candidate._strict_object(
        profile_bytes, omdet_public_candidate.PROFILE_NAME)


    omdet_public_candidate._preflight_candidate_byte_quota(
        root, profile, profile_bytes=len(profile_bytes))
    raw_source = profile.get("source_onnx")
    if (not isinstance(raw_source, dict) or
            type(raw_source.get("bytes")) is not int or
            raw_source["bytes"] > omdet_compiler_hold.MAX_ONNX_BYTES):
        raise OmDetCompilerRouteRefused("selected OmDet ONNX is outside holder quota")
    selected = omdet_bundle_profile.validate_candidate_profile(
        profile, root, expected_pins=expected_pins)
    deployment = omdet_public_candidate._validate_fixed_profile(root, selected)
    if args.target != deployment["device"]:
        raise OmDetCompilerRouteRefused(
            "OmDet selected target differs from approved source candidate profile")
    requested_sm = targets.validate_target_sm(args.target, args.sm or None)
    if requested_sm != deployment["sm"]:
        raise OmDetCompilerRouteRefused(
            "OmDet selected target SM differs from approved candidate")
    eight = {
        "source_onnx": selected["source_onnx"],
        "export_contract": selected["export_contract"],
        "feature_manifest": selected["feature_manifest"],
        "processor": selected["hf_configs"]["processor"],
        "model": selected["hf_configs"]["model"],
        **{role: selected["trained_features"][role]
           for role in omdet_public_candidate.FEATURE_FILES},
    }
    safe = {role: omdet_public_candidate._pin(value, role)
            for role, value in eight.items()}


    omdet_public_candidate._reject_external_onnx(
        root, safe["source_onnx"])
    snapshot = omdet_compiler_hold.scan_selected_omdet(root, safe)
    original_model = root.joinpath(*safe["source_onnx"]["file"].split("/"))
    req_os, req_arch = deployment["os"], deployment["arch"]
    profile_plan = build_plan.resolve(
        targets.host(), route="raw", source=str(original_model),
        target=args.target, req_sm=requested_sm, req_os=req_os,
        req_arch=req_arch, req_driver_api=deployment.get("driver_api"))
    if profile_plan["target_profile"] != selected["target_profile"]:
        raise OmDetCompilerRouteRefused(
            "current BuildPlan target OS/ISA/SM/driver differs from selected OmDet profile")
    return {
        "root": root, "output": output, "original_model": original_model,
        "selected": snapshot, "pins": safe, "plan": profile_plan,
        "source_mode": "EXISTING_V1_CANDIDATE_RECOMPILE",
        "pinset_sha256": pinset_sha,
        "candidate_profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "req_sm": requested_sm, "req_os": req_os, "req_arch": req_arch,
        "req_driver_api": deployment.get("driver_api"),
    }


def _validate_direct_eight_static(root, pins):
    if set(pins) != {role for role, _staged, _max in STAGED_ROLES}:
        raise OmDetCompilerRouteRefused("OmDet direct source requires exact eight pins")
    from . import omdet_bundle_profile as profile

    fixed, approved_npz_sha = _direct_revision_pins(pins)
    roles = {
        "processor": "processor", "model": "model",
        "feature_manifest": "features", "class_features": "class_features",
        "task_features": "task_features", "task_mask": "task_mask",
    }
    for name, asset in roles.items():
        size, sha = fixed[asset]
        if pins[name]["bytes"] != size or pins[name]["sha256"] != sha:
            raise OmDetCompilerRouteRefused(
                "selected OmDet direct %s differs from pinned trained tiny revision" % name)
    feature_name = pins["feature_manifest"]["file"]
    if PurePosixPath(feature_name).name != "omdet-feature-assets.json":
        raise OmDetCompilerRouteRefused("OmDet selected feature manifest name differs")
    for role, name in omdet_public_candidate.FEATURE_FILES.items():
        if PurePosixPath(pins[role]["file"]).name != name:
            raise OmDetCompilerRouteRefused("OmDet selected feature filename differs: " + role)
    def selected_json(role, label):
        body = omdet_public_candidate._small_bytes(
            root, pins[role]["file"], MAX_DECLARATION_BYTES)
        if (len(body) != pins[role]["bytes"] or
                hashlib.sha256(body).hexdigest() != pins[role]["sha256"]):
            raise OmDetCompilerRouteRefused(
                "selected OmDet %s bytes changed during semantic validation" % role)
        return omdet_public_candidate._strict_object(body, label)

    selected_contract = selected_json("export_contract", "OmDet selected export contract")
    feature_record = selected_json("feature_manifest", "OmDet selected trained features")
    if (feature_record.get("class_labels_in_order") != list(omdet_public_candidate.LABELS)
            or feature_record.get("task_phrase") != "detect objects"
            or feature_record.get("profile") !=
            "omdet-turbo-tiny-fixed-eight-classes-only"
            or feature_record.get("status") != "FEATURES_ONLY_NATIVE_NOT_CHECKED"
            or feature_record.get("verified_pascal_facade") is not False
            or feature_record.get("source_npz_sha256") !=
            approved_npz_sha
            or feature_record.get("source_npz_bytes") != 5015380):
        raise OmDetCompilerRouteRefused(
            "OmDet selected feature provenance/labels/revision is not pinned")
    inputs, outputs, assets, labels = profile._inspect_four_inputs(
        root, selected_contract, feature_record, feature_name)
    expected_inputs = [
        {"name": name, "shape": list(shape), "dtype": dtype}
        for name, shape, _layout, dtype in INPUT_ROWS
    ]
    expected_outputs = [
        {"name": name, "shape": list(shape), "dtype": dtype,
         "semantics": meaning}
        for name, shape, dtype, meaning in omdet_public_candidate.OUTPUTS
    ]
    if (inputs != expected_inputs or outputs != expected_outputs or
            labels != list(omdet_public_candidate.LABELS)):
        raise OmDetCompilerRouteRefused("direct eight OmDet graph/label tensor ABI differs")
    for role in omdet_public_candidate.FEATURE_FILES:
        if ({k: assets[role][k] for k in ("file", "bytes", "sha256")} !=
                pins[role]):
            raise OmDetCompilerRouteRefused(
                "OmDet direct selected source and trained feature manifest disagree")


def _selection_direct_eight(args):
    root, output = omdet_public_candidate._paths(args.source_dir, args.into)
    approved, pinset_sha = _unique_pinset(root, output, args.expected_pins)
    selected = omdet_compiler_hold.scan_selected_omdet(root, approved)
    _validate_direct_eight_static(root, approved)
    selected_onnx = next(row for row in selected["members"]
                         if row["role"] == "source_onnx")
    omdet_public_candidate._reject_external_onnx(
        root, {"file": selected_onnx["file"], "bytes": selected_onnx["bytes"],
               "sha256": selected_onnx["sha256"]})
    host = targets.host()
    if (host.get("os") != "windows" or host.get("arch") != "x86_64"):
        raise OmDetCompilerRouteRefused(
            "OmDet direct selected Pascal x86-64 requires native Windows x86-64")
    req_sm = targets.validate_target_sm(args.target, args.sm or None)
    original_model = root.joinpath(*selected_onnx["file"].split("/"))
    plan = build_plan.resolve(
        host, route="raw", source=str(original_model),
        target=args.target, req_sm=req_sm,
        req_os="windows", req_arch="x86_64")
    return {
        "root": root, "output": output, "original_model": original_model,
        "selected": selected, "pins": approved, "plan": plan,
        "source_mode": "DIRECT_EIGHT_EXTERNAL_RESOURCES_FIRST_COMPILER",
        "pinset_sha256": pinset_sha,
        "candidate_profile_sha256": None,
        "req_sm": req_sm, "req_os": "windows", "req_arch": "x86_64",
        "req_driver_api": None,
    }


def _argv(argv, *, preview):
    parser = argparse.ArgumentParser(
        prog=("pascalvision-convert omdet-held-preview" if preview
              else "pascalvision-convert omdet-held-compiler"),
        description="OmDet fixed-eight selected ONNX/features/config to TVM; NOT_CHECKED")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-dir", metavar="ABS_EXISTING_V1_BUNDLE")
    source.add_argument("--source-dir", metavar="ABS_EIGHT_ORIGINAL_RESOURCES")
    parser.add_argument("--expected-pins", required=True, metavar="ABS_PINSET_JSON")
    parser.add_argument("--into", required=True, metavar="ABS_NEW_DIR")
    parser.add_argument("--target", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--sm", default="", help="exact CUDA SM; CPU forbids SM")
    parser.add_argument("--optimized-onnx", metavar="ABS_PINNED_ORT_BASIC_ONNX")
    parser.add_argument("--optimization-receipt", metavar="ABS_PINNED_RECEIPT_JSON")
    if not preview:
        parser.add_argument("--approved-tree-sha256", required=True, metavar="SHA256")
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if type(exc.code) is int else 2


def preview_main(argv=None):
    parsed = _argv(argv, preview=True)
    if isinstance(parsed, int):
        return parsed
    try:
        if ((parsed.optimized_onnx is None) != (parsed.optimization_receipt is None)
                or (parsed.optimized_onnx is not None and parsed.source_dir is None)):
            raise OmDetCompilerRouteRefused("optimized trained compiler requires exact direct-eight source and receipt")
        selected = _selection(parsed)
        if parsed.optimized_onnx is not None:
            omdet_compiler_hold.inspect_exact_optimized_source(
                parsed.optimized_onnx, parsed.optimization_receipt, selected["pins"])
        record = dict(selected["selected"])
        record.update({
            "source_profile": "OMDET_TINY_SWIN_T_FIXED_EIGHT_CALLER_APPROVED_NOT_ATTESTED",
            "source_mode": selected["source_mode"],
            "target": parsed.target, "target_sm": selected["req_sm"],
            "candidate_profile_sha256": selected["candidate_profile_sha256"],
            "expected_pins_sha256": selected["pinset_sha256"],
            "native_pascal_run": "NOT_RUN",
            "independent_model_reference": "NOT_ATTESTED",
        })
        print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
        return 2
    except (ValueError, OSError, TypeError, RuntimeError, KeyError) as error:
        print("OmDet HELD Preview REFUSED, NOT_CHECKED: %s" % error,
              file=sys.stderr)
        return 1


def _four_native_inputs(path):
    source_freeze._safe_ancestors(path)
    observed = source_freeze._lstat(path)
    if not stat.S_ISREG(observed.st_mode) or not 0 < observed.st_size <= 65536:
        raise OmDetCompilerRouteRefused("OmDet four-input native TSV is missing/oversized")
    with path.open("rb") as source:
        raw = source.read(65537)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise OmDetCompilerRouteRefused("OmDet native inputs are not UTF-8") from error
    if len(lines) != len(INPUT_ROWS):
        raise OmDetCompilerRouteRefused("OmDet compiler must output exact FOUR inputs")
    for index, ((name, shape, _layout, _dtype), line) in enumerate(
            zip(INPUT_ROWS, lines)):
        expected = [str(index), name, ",".join(str(n) for n in shape)]
        if line.split("\t") != expected:
            raise OmDetCompilerRouteRefused(
                "OmDet native four-input order/shape differs at index %d" % index)
    return [{"name": name, "shape": list(shape), "dtype": dtype}
            for name, shape, _layout, dtype in INPUT_ROWS]


def _copy_held_consumer_resource(source, destination, selected):
    limit = next(max_bytes for role, _name, max_bytes in STAGED_ROLES
                 if role == selected["role"])
    expected_size, approved = selected["bytes"], selected["sha256"]
    if not 0 < expected_size <= limit:
        raise OmDetCompilerRouteRefused("OmDet copied resource exceeds selected quota")
    source_freeze._safe_ancestors(source)
    source_freeze._safe_ancestors(destination.parent)
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        before = os.fstat(reader.fileno())
        if (before.st_size != expected_size or
                getattr(before, "st_nlink", 1) != 1):
            raise OmDetCompilerRouteRefused("OmDet held consumer member resized")
        remaining = expected_size
        while remaining:
            blob = reader.read(min(omdet_compiler_hold.BLOCK, remaining))
            if not blob:
                raise OmDetCompilerRouteRefused("OmDet held consumer resource shortened")
            writer.write(blob)
            digest.update(blob)
            remaining -= len(blob)
        if reader.read(1) or digest.hexdigest() != approved:
            raise OmDetCompilerRouteRefused("OmDet copied consumer source SHA differs")
        if source_freeze._descriptor_identity(os.fstat(reader.fileno())) != (
                source_freeze._descriptor_identity(before)):
            raise OmDetCompilerRouteRefused("OmDet source changed during consumer copy")
        writer.flush()
        os.fsync(writer.fileno())


def _retain_selected_pipeline_diagnostic(report, output, api, output_locks):
    if not callable(getattr(report, "as_dict", None)):
        raise OmDetCompilerRouteRefused("selected compiler returned no diagnostic report")
    payload = report.as_dict()
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise OmDetCompilerRouteRefused("selected compiler diagnostic schema invalid")
    diagnostic = {
        "schema": "crosspascal.omdet.compiler-diagnostic.v1",
        "status": "ADVISORY_COMPILER_REPORT_NOT_NATIVE_UAT",
        "verified_pascal": False,
        "pipeline": payload,
    }
    encoded = (json.dumps(diagnostic, sort_keys=True, indent=2,
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if not 0 < len(encoded) <= MAX_DECLARATION_BYTES:
        raise OmDetCompilerRouteRefused("selected compiler diagnostic exceeds size quota")
    path = output / "omdet-selected-compiler-diagnostic.json"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    pinned = held_output._pin_compiler_output(
        api, output_locks, path, max_bytes=MAX_DECLARATION_BYTES)
    expected = {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    if pinned[3] != expected:
        raise OmDetCompilerRouteRefused("selected compiler diagnostic changed before native hold")
    held_output._audit_pinned_output_sidecars(api, [pinned])
    return pinned


def _prestage_exact_trained_win64_cpu_runtime(output, api, output_locks):
    if os.name != "nt":
        raise OmDetCompilerRouteRefused("trained runtime pre-stage needs Win32 NTFS")
    inventory, libraries, licences = runtime._plan_runtime_files(
        "cpu", target_dir=str(output))
    rows = libraries + licences
    if (inventory.get("ready") is not True or
            inventory.get("missing") or inventory.get("blocked_existing") or
            inventory.get("existing_verified") or
            inventory.get("unmanaged_cuda_files") or
            len(rows) != len(TRAINED_WIN64_CPU_RUNTIME_WHEEL_PINS) or
            {row.get("file") for row in rows} !=
            set(TRAINED_WIN64_CPU_RUNTIME_WHEEL_PINS)):
        raise OmDetCompilerRouteRefused(
            "exact trained Win64 CPU runtime wheel/license inventory refused")


    for row in rows:
        size, sha = TRAINED_WIN64_CPU_RUNTIME_WHEEL_PINS[row["file"]]
        if (row.get("bytes") != size or row.get("sha256") != sha or
                not isinstance(row.get("source"), str) or
                not os.path.isabs(row["source"])):
            raise OmDetCompilerRouteRefused(
                "approved trained CPU wheel source size/SHA differs: " + row["file"])
        source_freeze._safe_ancestors(Path(row["source"]))
        source_info = source_freeze._lstat(Path(row["source"]))
        if (not stat.S_ISREG(source_info.st_mode) or
                getattr(source_info, "st_nlink", 1) != 1 or
                source_info.st_size != size or
                os.path.lexists(output / row["file"])):
            raise OmDetCompilerRouteRefused(
                "trained CPU wheel source/destination unsafe or already occupied: " +
                row["file"])

    pinned = []
    for row in rows:
        size, sha = TRAINED_WIN64_CPU_RUNTIME_WHEEL_PINS[row["file"]]
        origin = Path(row["source"])
        destination = output / row["file"]
        digest = hashlib.sha256()
        with origin.open("rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            if (source_freeze._descriptor_identity(before) !=
                    source_freeze._descriptor_identity(
                        source_freeze._lstat(origin)) or
                    before.st_size != size or
                    getattr(before, "st_nlink", 1) != 1):
                raise OmDetCompilerRouteRefused(
                    "wheel source changed before selected native copy")
            remaining = size
            while remaining:
                block = reader.read(min(1048576, remaining))
                if not block:
                    raise OmDetCompilerRouteRefused(
                        "trained CPU wheel source shortened during copy")
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
            if (reader.read(1) or digest.hexdigest() != sha or
                    source_freeze._descriptor_identity(os.fstat(reader.fileno())) !=
                    source_freeze._descriptor_identity(before)):
                raise OmDetCompilerRouteRefused(
                    "trained CPU wheel/rights original bytes changed during copy")
            writer.flush()
            os.fsync(writer.fileno())
        pin = held_output._pin_compiler_output(
            api, output_locks, destination, max_bytes=size)
        if pin[3] != {"bytes": size, "sha256": sha}:
            raise OmDetCompilerRouteRefused(
                "selected CPU wheel output changed before Win32 native hold")
        pinned.append(pin)
    held_output._audit_pinned_output_sidecars(api, pinned)
    return pinned


def main(argv=None):
    parsed = _argv(argv, preview=False)
    if isinstance(parsed, int):
        return parsed
    try:
        if ((parsed.optimized_onnx is None) != (parsed.optimization_receipt is None)
                or (parsed.optimized_onnx is not None and parsed.source_dir is None)):
            raise OmDetCompilerRouteRefused("optimized trained compiler requires exact direct-eight source and receipt")
        prepared = _selection(parsed)
        if parsed.optimized_onnx is not None:
            omdet_compiler_hold.inspect_exact_optimized_source(
                parsed.optimized_onnx, parsed.optimization_receipt, prepared["pins"])
        selected = prepared["selected"]
        if selected["tree_sha256"] != omdet_compiler_hold._approved_sha(
                parsed.approved_tree_sha256):
            raise OmDetCompilerRouteRefused(
                "OmDet selected resources changed from approved Preview tree SHA")
        output, root = prepared["output"], prepared["root"]
        stage_parent = output.parent
        if shutil.disk_usage(stage_parent).free < (
                2 * selected["total_bytes"] +
                2 * omdet_compiler_hold.DISK_HEADROOM):
            raise OmDetCompilerRouteRefused("OmDet stage/compiled output disk headroom low")


        final_receipt = []
        with held_output._hold_output_directory(
                output, final_receipt=final_receipt,
                final_receipt_name="omdet-selected-compiler-source.json") as (api, output_locks):
            pinned_outputs = []
            consumer_resources = []
            selected_cpu_runtime = []
            module = output / (
                "omdet-turbo-tiny." + parsed.target + pipeline.module_extension())
            inputs = output / (module.name + ".inputs")

            def hold_compiled_at_build_done(step):
                if (getattr(step, "name", None) != "Building the module" or
                        getattr(step, "state", None) != "done"):
                    return
                if pinned_outputs:
                    raise OmDetCompilerRouteRefused(
                        "OmDet repeated completed compiler output callback")
                pinned_outputs.append(held_output._pin_compiler_output(
                    api, output_locks, module,
                    max_bytes=held_output.MAX_COMPILED_MODULE_BYTES))
                pinned_outputs.append(held_output._pin_compiler_output(
                    api, output_locks, inputs, max_bytes=65536))

            consumer_directories = ExitStack()
            output_locks.callback(consumer_directories.close)
            with omdet_compiler_hold.hold_selected_omdet(
                    root, prepared["pins"], stage_parent=stage_parent,
                    approved_tree_sha256=selected["tree_sha256"],
                    optimized_onnx=parsed.optimized_onnx,
                    optimization_receipt=parsed.optimization_receipt) as held:
                if (parsed.target == "cpu" and parsed.optimized_onnx is not None
                        and prepared["pins"]["source_onnx"]["sha256"] ==
                        TRAINED_DIRECT_ONNX[1]):
                    selected_cpu_runtime = _prestage_exact_trained_win64_cpu_runtime(
                        output, api, output_locks)
                report = pipeline.run(
                    held.compiler_input, str(output), target=parsed.target,
                    module_name="omdet-turbo-tiny", shape=None, contract=None,
                    source=str(prepared["original_model"]),
                    approved_build_plan=prepared["plan"],
                    req_sm=prepared["req_sm"], req_os=prepared["req_os"],
                    req_arch=prepared["req_arch"],
                    req_driver_api=prepared["req_driver_api"],
                    on_step=hold_compiled_at_build_done)
                diagnostic_pin = _retain_selected_pipeline_diagnostic(
                    report, output, api, output_locks)
                summary = getattr(report, "summary", None)
                if (getattr(report, "sound", None) is not True or
                        not isinstance(summary, dict) or
                        summary.get("module") != str(module) or
                        summary.get("directory") != str(output) or
                        summary.get("build_plan") != prepared["plan"]):
                    raise OmDetCompilerRouteRefused(
                        "OmDet real shared compiler refused selected plan/output; "
                        "see omdet-selected-compiler-diagnostic.json")
                if len(pinned_outputs) != 2:
                    raise OmDetCompilerRouteRefused(
                        "OmDet compiler output was not Win32-held before "
                        "runtime placement and model verification")
                held_output._audit_pinned_output_sidecars(api, pinned_outputs)
                four_io = _four_native_inputs(inputs)
                for directory in ("source", "audit", "features"):
                    subdirectory = output / directory
                    subdirectory.mkdir(exist_ok=False)
                    consumer_directories.enter_context(
                        omdet_public_candidate._hold_plain_ntfs_directory(
                            subdirectory))
                for row in selected["members"]:
                    relative = CONSUMER_RESOURCE_DESTINATIONS[row["role"]]
                    destination = output.joinpath(*relative.split("/"))
                    _copy_held_consumer_resource(
                        held.stage / row["stage_name"], destination, row)
                    consumer_resources.append(held_output._pin_compiler_output(
                        api, output_locks, destination,
                        max_bytes=next(bound for role, _staged, bound in
                                       STAGED_ROLES if role == row["role"])))

            consumer_directories.close()
            held_output._audit_pinned_output_sidecars(api, pinned_outputs)
            held_output._audit_pinned_output_sidecars(api, consumer_resources)
            held_output._audit_pinned_output_sidecars(api, selected_cpu_runtime)
            receipt = {
                "schema": SCHEMA,
                "status": "POST_WIN32_OMDET_SELECTED_COMPILER_CUSTODY_NOT_RELEASE_GRADE",
                "source_tree_sha256": selected["tree_sha256"],
                "selected_source_members": selected["members"],
                "source_mode": prepared["source_mode"],
                "candidate_profile_sha256": prepared["candidate_profile_sha256"],
                "expected_pins_sha256": prepared["pinset_sha256"],
                "compiler_private_model_path": True,
                "compiler_diagnostic": dict(diagnostic_pin[3]),
                "exact_trained_cpu_wheel_pre_staged": {
                    path.name: dict(info)
                    for path, _handle, _native, info in selected_cpu_runtime
                },
                "optimized_compiler_input": (
                    {"bytes": omdet_compiler_hold.OPTIMIZED_TRAINED_SOURCE[0],
                     "sha256": omdet_compiler_hold.OPTIMIZED_TRAINED_SOURCE[1],
                     "optimization_receipt": held.optimized_receipt,
                     "status": "SEPARATE_WIN32_HELD_COMPILE_SOURCE_NOT_NATIVE_UAT"}
                    if parsed.optimized_onnx is not None else None),
                "retained_stage": str(held.stage),
                "compiler_output_pinned_members": {
                    path.name: dict(info)
                    for path, _handle, _native, info in pinned_outputs
                },
                "copied_pascal_consumer_resources": {
                    str(path.relative_to(output)).replace("\\", "/"): dict(info)
                    for path, _handle, _native, info in consumer_resources
                },
                "four_input_contract": four_io,
                "selected_build_plan_sha256": prepared["plan"]["plan_sha256"],
                "compiler_report_sound": True,
                "verification_status": "NOT_CHECKED",
                "native_pascal_run": "NOT_RUN",
                "independent_original_ort": "NOT_ATTESTED",
                "trained_model_revision": "NOT_ATTESTED",
                "runtime_dll_provenance": "NOT_ATTESTED",
                "model_license_status": "NOT_ATTESTED",
                "public_pascal_facade_ready": False,
                "release_grade_immutable": False,
            }
            receipt_name = output / "omdet-selected-compiler-source.json"
            encoded = (json.dumps(receipt, indent=2, sort_keys=True,
                                  ensure_ascii=False, allow_nan=False) +
                       "\n").encode("utf-8")
            if not 0 < len(encoded) <= MAX_DECLARATION_BYTES:
                raise OmDetCompilerRouteRefused("OmDet public receipt exceeds byte quota")

            pending_name = output / "omdet-selected-compiler-source.pending.json"
            pending_receipt = dict(receipt)
            pending_receipt["status"] = "PENDING_OMDET_POST_AUDIT_NOT_ACCEPTED"
            pending_encoded = (json.dumps(
                pending_receipt, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if not 0 < len(pending_encoded) <= MAX_DECLARATION_BYTES:
                raise OmDetCompilerRouteRefused("OmDet pending receipt exceeds byte quota")
            held_output._write_receipt_bytes_exclusive(pending_name, pending_encoded)
            held_pending = held_output._pin_compiler_output(
                api, output_locks, pending_name, max_bytes=MAX_DECLARATION_BYTES)
            if held_pending[3] != {
                    "bytes": len(pending_encoded),
                    "sha256": hashlib.sha256(pending_encoded).hexdigest()}:
                raise OmDetCompilerRouteRefused(
                    "OmDet pending diagnostic changed before native handle custody")
            held_output._audit_pinned_output_sidecars(api, pinned_outputs)
            held_output._audit_pinned_output_sidecars(api, consumer_resources)
            held_output._audit_pinned_output_sidecars(api, selected_cpu_runtime)
            held_output._audit_pinned_output_sidecars(api, [held_pending])
            held_output._check_no_foreign_loadables(
                output, module, parsed.target,
                exact_wheel=bool(selected_cpu_runtime))

            final_receipt.append((receipt_name, encoded))
        print("OmDet eight-member selected-source compiler custody completed "
              "(advisory exit 2; Pascal/ORT/native acceptance NOT_CHECKED)")
        return 2
    except (ValueError, OSError, TypeError, RuntimeError, KeyError,
            AttributeError, ImportError) as error:
        print("OmDet held compiler REFUSED, NOT_CHECKED: %s" % error,
              file=sys.stderr)
        return 1
