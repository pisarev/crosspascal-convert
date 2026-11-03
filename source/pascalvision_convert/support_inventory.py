# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Read-only, preliminary I-11 software support/debt inventory.

This is an authored statement of inspected software scope, not a discovery
probe, model runner, verified test report, authenticated release gate or
replacement for the owner's 28-step debt plan. It cannot change artifacts,
user data, status files, snapshots, repositories or network state.

A family can have a working path for a *specific* model/target while the
normative all-family TYPE-1 implementation obligation remains open. Hardware,
Pascal, browser and human acceptance require separate real evidence. In
particular, historical native passes must not be generalized to other models
or machines; a historical strict compiler-window XFAIL stays open.
"""

from __future__ import annotations

import argparse
import json
import sys


SCHEMA = "crosspascal.pascalvision.support-inventory.v1"
SNAPSHOT = "2026-10-18"
_TYPE1 = "IMPLEMENTATION_OPEN"
_NOT_RELEASED = "PRODUCT_PARTIAL_NOT_RELEASED"
_PRODUCT = "tools/pascalvision-convert/"
_PACKAGE = _PRODUCT + "pascalvision_convert/"
_TESTS = _PRODUCT + "tests/"
_PASCAL = "4-ml/pascal/"


def _paths(prefix, names):
    """Create repo-relative owned paths, never absolute or private machine paths."""
    return [prefix + name for name in names]


def _route(status, entry, scope):
    return {"status": status, "entry": entry, "scope": scope}


def _ownership(package=(), product=(), pascal=(), external=(), tests=(),
               pascal_tests=()):
    return {
        "sources": (_paths(_PACKAGE, package) +
                    _paths(_PRODUCT, product) +
                    _paths(_PASCAL, pascal) + list(external)),
        "tests": (_paths(_TESTS, tests) +
                  _paths("4-ml/tests/", pascal_tests)),
    }


def _work(identifier, title, software, native, human, ownership, blockers,
          routes=()):
    return {
        "id": identifier,
        "title": title,
        "classification": "TYPE-1",
        "implementation_status": _TYPE1,
        "software_status": software,
        "native_acceptance": native,
        "human_acceptance": human,
        "ownership": ownership,
        "routes": list(routes),
        "implementation_open_reasons": list(blockers),
        "provenance": [
            "state/PASCALVISION-IMPLEMENTATION-DEBT-PLAN-20261014.md",
            "state/PASCALVISION-TARGET-CONFIG-DNA.md",
        ],
    }


def _family(identifier, name, source_format, task, scope, software, native,
            human, ownership, routes, blockers):
    return {
        "id": identifier,
        "name": name,
        "source_format": source_format,
        "task": task,
        "declared_scope": scope,
        "classification": "TYPE-1",
        "implementation_status": _TYPE1,
        "software_status": software,
        "native_acceptance": native,
        "human_acceptance": human,
        "ownership": ownership,
        "routes": list(routes),
        "implementation_open_reasons": list(blockers),
        "unsupported_expansion": "Unknown architectures, formats, model sizes "
                                 "or target machines are not inferred supported.",
    }


def _stage(identifier, title, related_ids, open_ids):
    return {
        "id": identifier,
        "title": title,
        "implementation_status": (_TYPE1 if any(
            item in open_ids for item in related_ids) else "REVIEW_REQUIRED"),
        "software_status": "PARTIAL",
        "related_work_items": list(related_ids),
        "independent_native_or_human_acceptance": "NOT_GLOBALLY_ACCEPTED",
    }


def build_inventory() -> dict:
    """Return a fresh JSON-serializable advisory inventory without any I/O.

    Declared rows reflect the exact normative I-01..I-11 scope. All TYPE-1
    classifications here are intentionally conservative; a future owner
    review must change the specific row only after the missing connected
    implementation has been proven, not after a mocked test or old UAT.
    """
    work_items = [
        _work(
            "I-01", "Common BuildPlan, recipe replay and target semantics",
            "PARTIAL_CONNECTED", "SCOPED_NATIVE_RESULTS_NOT_GLOBAL",
            "NOT_APPLICABLE",
            _ownership(
                package=("build_plan.py", "targets.py", "recipe_file.py",
                         "recipe.py", "pipeline.py"),
                product=("master.py", "import_adapters.py"),
                external=("1-bindings/tvm/llm/build_llm.py",),
                tests=("test_common_build_plan_step03.py",
                       "test_family_replay_step03.py",
                       "test_pipeline_approved_build_plan_step06.py")),
            ("Unify every supported wizard/CLI/replay/family builder option, "
             "including original-model licence and destination ABI provenance.",
             "Prove each effective selected option reaches generated artifacts "
             "and no missing branch defaults to a different target."),
            (_route("PARTIAL_CONNECTED", "pascalvision-convert --into",
                    "Shared RAW/selected family recipe; no all-family parity"),
             _route("PARTIAL_CONNECTED", "master.py / master_page.html",
                    "Selected typed BuildPlan, not all families")),
        ),
        _work(
            "I-02", "Compiler input closure and immutable selected bytes",
            "PARTIAL_HELD_WIN32", "NO_RELEASE_GRADE_IMMUTABILITY",
            "NOT_APPLICABLE",
            _ownership(
                package=("source_freeze.py", "source_lock.py",
                         "hf_closure.py", "hf_held_stage.py",
                         "onnx_closure.py", "paddle_closure.py",
                         "paddle_held_stage.py", "exported_onnx_hold.py",
                         "recipe.py"),
                product=("master.py", "import_adapters.py"),
                tests=("test_preview_source_content_gap.py",
                       "test_source_lock.py", "test_step08_hf_build_binding.py",
                       "test_step08_hf_preview_selected_tree.py",
                       "test_step08_paddle_held_stage.py",
                       "test_step08_paddle_wizard_selected_binding.py",
                       "test_step08_paddle_exported_onnx_compiler_hold.py",
                       "test_step08_paddle_cli_selected.py",
                       "test_step08_paddle_recipe_replay.py",
                       "test_step08_paddle_cli_exported_onnx_compiler_hold.py")),
            ("Historical STRICT XFAIL: original source can change and be "
             "restored during a compiler window.",
             "Scoped HTTP HF Preview now pins exact Qwen/Gemma selected "
             "weights/tokenizer SHA and requires matching held provenance; "
             "Paddle opt-in now holds lean exported ONNX through TVM Verify, "
             "but external_data and custom plugins remain unsupported.",
             "84MiB+ PP-OCRv5 trained compiler/runtime was NOT_RUN, while "
             "all-family source closure, preexisting mappings and direct "
             "CLI Preview-independent approvals remain TYPE-1 open.",
             "Public Paddle CLI selected-source opt-in and saved recipe "
             "replay now hold self-contained exported ONNX through pipeline "
             "call; a false Verify receipt on unsound pipeline is refused, "
             "but numeric independent original ORT/TVM remains NOT_RUN.",
             "Close all promised external-resource and selected-input custody "
             "paths without modifying the user's source."),
            (_route("OPT_IN_PARTIAL", "Wizard source freeze and protected RAW",
                    "Selected NTFS/RAW path; universal compiler custody open"),
             _route("OPT_IN_PARTIAL", "native_family_atomic",
                    "Selected native HF and consumer hold; not all formats"),
             _route("OPT_IN_PARTIAL", "paddle_native_hold",
                    "Paddle selected original exporter and separate held "
                    "lean ONNX compiler/Verify stage; external_data denied"),
             _route("OPT_IN_PARTIAL", "CLI --paddle-native-hold / --recipe",
                    "Selected original Paddle approval + private exported-"
                    "ONNX Win32 hold for pipeline call, NOT_CHECKED, no "
                    "trained Rec or global release attestation")),
        ),
        _work(
            "I-03", "Atomic complete artifact, runtime and ZIP publication",
            "PARTIAL_CONNECTED", "BUNDLE_NAMESPACE_NOT_RELEASE_DURABLE",
            "NOT_APPLICABLE",
            _ownership(
                package=("bundle_atomic.py", "bundle_publication.py",
                         "bundle_download.py", "runtime.py",
                         "retained_ledger.py", "destination_preflight.py"),
                product=("master.py",),
                tests=("test_bundle_atomic_step08.py",
                       "test_bundle_download.py",
                       "test_bundle_publication_step08.py",
                       "test_step08_legacy_zip_preheader.py",
                       "test_step08_llm_family_atomic_publication.py",
                       "test_runtime_inventory.py")),
            ("Prove exact complete closure/READY and failure/crash behavior "
             "for every claimed RAW, family and composite ZIP route.",
             "Retained failed stages, post-rename integrity and non-durable "
             "power-loss behavior must be reported without owner-data deletion. "
             "Legacy ZIP now pins every selected member SHA, denies unpinned "
             "triples, linked/aliased/NFD members, refuses incomplete nested "
             "enumeration and applies 4096 member/64GiB aggregate bounds "
             "before full hashing; transient refusals remain retryable and "
             "persistent unsafe legacy artefacts revoke download_ready. "
             "Selected Qwen/Gemma native LLM ZIP checks now bind the second "
             "DLL and params.bin size to the initial full-SHA inventory; "
             "BuildPlan/JSON initial byte caps and phase-aware HTTP409 READY "
             "taxonomy have software RED/GREEN evidence only. "
             "This software transport does NOT prove release-grade durability."),
            (_route("PARTIAL_CONNECTED", "Wizard Build/Download ZIP",
                    "Only eligible committed and inspected bundles"),
             _route("PARTIAL_CONNECTED", "pascalvision-convert --into",
                    "Selected RAW/family atomic paths, no universal guarantee")),
        ),
        _work(
            "I-04", "Actual OS/ISA/GPU, toolkit and runtime compatibility",
            "PARTIAL_TARGET_PREFLIGHT", "SCOPED_NATIVE_TARGET_PASSES",
            "NOT_APPLICABLE",
            _ownership(
                package=("environment.py", "targets.py", "cuda_preflight.py",
                         "convert.py", "runtime.py", "tuning.py"),
                external=("4-ml/converters/onnx_to_tvm.py",),
                tests=("test_target_profile_v2.py", "test_targets.py",
                       "test_step08_cpu_foreign_tuning_guard.py",
                       "test_step08_cpu_target_capability_preflight.py")),
            ("Multi-GPU selection and exact CUDA driver/runtime/library ABI "
             "dependency closure are not complete for all declared targets.",
             "PTX/cubin/fatbin and foreign-ISA modes need proven supported "
             "behavior or an explicit source-connected refusal."),
            (_route("PARTIAL_CONNECTED", "Wizard target/SM and target profile",
                    "Machine-derived defaults with known target limitations"),
             _route("PARTIAL_CONNECTED", "pascalvision-convert --target/--sm",
                    "Effective target validation; unsupported options refuse")),
        ),
        _work(
            "I-05", "Executable target-machine Pascal acceptance",
            "PARTIAL_CPU_TVM_RUNNER", "PASCAL_DESTINATION_UAT_NOT_RUN",
            "NOT_APPLICABLE",
            _ownership(
                package=("destination_preflight.py", "destination_native_runner.py",
                         "verify.py", "remote_memtest.py"),
                pascal=("CrossVision.ML.Facade.pas",),
                tests=("test_step08_destination_native_runner.py",
                       "test_step08_destination_native_cli_dispatch.py",
                       "test_step08_remote_contract_cli.py")),
            ("A real target Pascal Facade/TVM DLL inference/ABI/teardown "
             "runner with independent oracle has not been connected for the "
             "declared full scope.",
             "CPU TVM numerical mode and SSH BuildPlan-only probes cannot "
             "grant VERIFIED_REMOTE."),
            (_route("CONNECTED_LIMITED_NOT_VERIFIED",
                    "destination-native-cpu --bundle --original-onnx",
                    "Bounded CPU TVM numeric runner, NOT_CHECKED overall"),
             _route("CONTRACT_CHECK_ONLY", "remote-cpu-contract",
                    "No remote model inference")),
        ),
        _work(
            "I-06", "Public OmDet tiny four-input Pascal detector",
            "PARTIAL_FOUR_INPUT_CONNECTED", "TRAINED_PUBLIC_FACADE_UAT_NOT_RUN",
            "NOT_APPLICABLE",
            _ownership(
                package=("manifest.py", "omdet_public_candidate.py",
                         "omdet_bundle_profile.py"),
                pascal=("CrossVision.ML.OmDet.pas", "CrossVision.ML.Facade.pas",
                        "TVM.OmDet.Pipeline.pas",
                        "TVM.OmDet.ImagePreprocess.pas",
                        "TVM.OmDet.FixedFeatures.pas"),
                external=("4-ml/converters/convert_omdet.py",),
                tests=("test_step08_omdet_public_candidate.py",
                       "test_step08_omdet_pascal_native_manifest.py",
                       "test_step08_omdet_candidate_wizard.py")),
            ("Finish and independently prove public wizard/CLI positive "
             "trained model integration; the candidate package is NOT_CHECKED.",
             "Real image public Pascal Facade versus independent ONNX Runtime "
             "oracle and lifecycle acceptance have not run."),
            (_route("CONNECTED_NOT_CHECKED", "omdet-candidate-bundle "
                    "--candidate-dir --expected-pins --into",
                    "Exact tiny v2 candidate only; no VERIFIED/release"),
             _route("CONNECTED_NOT_CHECKED",
                    "master.py /api/omdet-candidate-preview",
                    "Strict opt-in one-use Preview/Build connected; no ZIP READY or trained native UAT")),
        ),
        _work(
            "I-07", "Unified PaddleOCR Det, Rec, dictionary and CTC",
            "COMPOSITE_DRAFT_PLUS_PNG_ORT_NATIVE_DET_AND_PAIRWISE_ADVISORY",
            "SCOPED_TRAINED_MOBILE_DET_CPU_RAW_RUN_REC_CTC_NOT_RUN",
            "HUMAN_GROUND_TRUTH_NOT_ATTESTED",
            _ownership(
                package=("ocr_composite.py", "ocr_composite_oracle.py",
                         "paddle_closure.py", "ocr_ort_raw_capture.py",
                         "ocr_capture_bridge.py", "ocr_png_preprocess.py",
                         "ocr_native_det_capture_bridge.py",
                         "ocr_det_pairwise.py"),
                pascal=("TVM.PaddleOcr.pas", "CrossVision.ML.Facade.pas"),
                product=("master.py",),
                external=("4-ml/tests/PaddleOcrNativeCapture.dpr",
                          "4-ml/tests/PaddleOcrNativeRecCapture.dpr"),
                tests=("test_step08_ocr_composite_wizard_draft.py",
                       "test_step08_ocr_composite_product_cli.py",
                       "test_paddleocr_stage_contract.py",
                       "test_step08_ocr_ort_capture_cli.py",
                       "test_step08_ocr_ort_capture_bridge.py",
                       "test_step08_ocr_png_preprocess_producer.py",
                       "test_step08_ocr_png_public_cli.py",
                       "test_step08_ocr_native_det_capture_bridge.py",
                       "test_step08_ocr_det_pairwise.py",
                       "test_step08_ocr_det_pairwise_public_cli.py",
                       "test_step08_ocr_ort_dynamic_shape_specialization.py",
                       "test_step08_ocr_stage_ntfs_admission.py",
                       "test_step08_ocr_stage_directory_hold.py",
                       "test_step08_native_rec_raw_dpr_contract.py",
                       "test_step08_native_pascal_path_and_roi_contract.py",
                       "test_step08_ocr_native_rec_preprocess_public_contract.py",
                       "test_step08_native_rec_truthful_product_checkpoint.py"),
                pascal_tests=("test_verify_paddleocr_gpu.py",)),
            ("Real RGB8 PNG-to-f32le DRAFT preprocessing, original ONNX Runtime "
             "CPU raw single-stage execution and read-only capture inspector "
             "are connected; none independently proves vendor/native "
             "preprocessing equivalence, Pascal Det+Rec+CTC or human GT.",
             "Real Delphi37 Pascal Mobile Det CPU raw producer compiled and "
             "executed once on a provenance-labelled RGB8 image, natural exit0; "
             "this does not independently attest its process/runtime origin "
             "or compare the original ONNX model. An independent standalone "
             "PaddleOcrNativeRecCapture.dpr source now calls the production "
             "PpocrPreprocess and preserves same-run raw post-softmax scores, "
             "but Delphi37 Rec compile, actual trained crop/Rec inference, "
             "independent original ORT Rec, CTC, human GT and verified-root "
             "Pascal adapter remain NOT_RUN / TYPE-1 OPEN. "
             "Standalone native/ORT receipt and pairwise inspectors are "
             "advisory and do NOT authenticate supplied producers."),
            (_route("DRAFT_NOT_CHECKED", "ocr-composite",
                    "One Det+Rec+dictionary composite without verified root"),
             _route("CONNECTED_NOT_CHECKED",
                    "ocr-png-input --bundle --stage --image --into",
                    "Actually decodes bounded RGB8 PNG into exact DRAFT Det/Rec "
                    "float32le input, advisory exit2; no original ORT/native "
                    "inference, independent crop/GT or product READY"),
             _route("CONNECTED_NOT_CHECKED",
                    "ocr-ort-raw --model --input --shape --into",
                    "Executable original ONNX Runtime CPU raw producer; selected "
                    "trained Det ORT numerical execution not yet measured, advisory exit2"),
             _route("READ_ONLY_NOT_CHECKED",
                    "ocr_capture_bridge.inspect_ort_capture",
                    "Pure read-only supplied raw ORT tensor inspection; never "
                    "attests native execution or independent human ground truth"),
             _route("READ_ONLY_NOT_CHECKED",
                    "ocr_native_det_capture_bridge.inspect_native_det_capture",
                    "Strict SHA/shape/runtime-assets admission of a supplied "
                    "Pascal Det-only capture; 26 synthetic software tests and "
                    "separate trained native pilot, no process attestation/GT"),
             _route("READ_ONLY_NOT_CHECKED",
                    "ocr-det-pairwise --native-capture --ort-capture "
                    "--native-model --original-onnx --native-sha256 "
                    "--original-sha256 --output-name",
                    "SHA-bound exact-input and all-scalar original ORT/TVM Det "
                    "comparison, advisory exit2 on supplied numeric match; "
                    "no producer attestation, Rec/CTC, human GT or READY"),
             _route("SOURCE_ONLY_NOT_RUN",
                    "4-ml/tests/PaddleOcrNativeRecCapture.dpr",
                    "Standalone Pascal Rec raw source; no Delphi37 compile, "
                    "trained crop, original ORT/CTC/human GT, or READY"),
             _route("ADVISORY_NOT_CHECKED", "ocr-composite-audit --bundle --evidence",
                    "Supplied Det/Rec/CTC/human evidence comparison, always advisory exit2"),
             _route("DRAFT_NOT_CHECKED",
                    "master.py OCR composite Preview/Build",
                    "No human-GT acceptance or public verified facade")),
        ),
        _work(
            "I-08", "Each declared family/format has an honest complete route",
            "MIXED_PARTIAL_FAMILY_ROUTES", "SCOPED_HISTORICAL_NATIVE_PASSES",
            "OCR_HUMAN_ACCEPTANCE_OPEN",
            _ownership(
                package=("detect.py", "convert.py", "pipeline.py",
                         "manifest.py", "llm_family_profile.py"),
                product=("import_adapters.py",),
                pascal=("TVM.Yolo11.pas", "TVM.RtDetr.pas",
                        "TVM.BiRefNet.pas", "TVM.PaddleOcr.pas"),
                tests=("test_verify_tflite_magic_step07.py",
                       "test_rtdetr_output_passport.py",
                       "test_step08_hf_source_output_alias.py")),
            ("The declared family matrix below contains individual TYPE-1 "
             "gaps; unsupported additions must not be silently offered.",
             "TFLite opt-in FP32, full HR800, LLM runtime/licence and "
             "all-family verified publication have incomplete closure."),
            (_route("MIXED_FAMILY_SCOPE", "Wizard source family selection",
                    "Declared route by selected family; refer to families[]"),
             _route("MIXED_FAMILY_SCOPE", "pascalvision-convert --into",
                    "Not all declared formats have a complete consumer path")),
        ),
        _work(
            "I-09", "Wizard, CLI, examples and consumer documentation",
            "PARTIAL_CONNECTED", "BROWSER_ALL_FAMILIES_NOT_ACCEPTED",
            "CONSUMER_ACCEPTANCE_NOT_RUN",
            _ownership(
                package=("cli.py", "recipe.py"),
                product=("master.py", "master_page.html", "README.md",
                         "SUPPORT-MATRIX.md"),
                tests=("test_wizard_next.py", "test_step08_public_workflow_docs.py",
                       "test_step08_omdet_candidate_wizard.py",
                       "test_wizard_contract.py")),
            ("Finish truthful Next/Back and strict Preview→Build→Verify→"
             "Export/unsupported behavior across all declared families.",
             "Provide portable examples and live browser acceptance beyond "
             "a small scoped RAW CPU demo."),
            (_route("PARTIAL_CONNECTED", "master.py / master_page.html",
                    "Six-step Wizard with scoped successful RAW path"),
             _route("PARTIAL_CONNECTED", "pascalvision-convert --into",
                    "Family-specific CLI variants; no universal parity")),
        ),
        _work(
            "I-10", "Selective original integration and portable distribution",
            "PARTIAL_SOURCE_MIGRATION", "CLEAN_ROOM_INSTALL_NOT_RUN",
            "EXTERNAL_RELEASE_NOT_AUTHORIZED",
            _ownership(
                package=("runtime.py", "bundle_download.py",
                         "destination_preflight.py"),
                product=("README.md", "SUPPORT-MATRIX.md"),
                external=("tools/release-audit/", "state/PASCALVISION-TARGET-CONFIG-DNA.md"),
                tests=("test_step08_public_workflow_docs.py",
                       "test_runtime_inventory.py")),
            ("Only selected migration slices exist; remaining semantic "
             "source and binary dependencies need independent review.",
             "No complete clean-path portable source-only distribution "
             "with verified licence notices and native example is accepted."),
            (_route("IMPLEMENTATION_OPEN", "portable clean-room install",
                    "Not yet a supported completed install/consume route"),
             _route("PARTIAL_CONNECTED", "Wizard or CLI bundle export",
                    "Existing selected ZIPs do not imply portable release")),
        ),
        _work(
            "I-11", "Executable global TYPE-1=0 inventory and final gate",
            "SOFTWARE_INVENTORY_PARTIAL", "ALL_SCOPE_ACCEPTANCE_NOT_RUN",
            "HUMAN_RELEASE_APPROVAL_NOT_GIVEN",
            _ownership(
                package=("support_inventory.py",),
                product=("SUPPORT-MATRIX.md", "README.md"),
                tests=("test_step08_support_inventory_cli.py",)),
            ("This module is a static, authored preliminary inventory; "
             "it does not dynamically attest implementation, test ownership, "
             "executables, model acceptance or code provenance.",
             "Final release gate, supported-path strict-XFAIL closure, "
             "signed evidence and all normative obligations remain open."),
            (_route("CONNECTED_PRELIMINARY_ADVISORY",
                    "pascalvision-convert support-inventory",
                    "Read-only CLI exposes I-01..I-11/V1..V8 and refuses --require-type1-zero while OPEN"),),
        ),
    ]

    families = [
        _family(
            "F-YOLO11-DET", "YOLO11 object detection",
            "Ultralytics .pt -> self-contained ONNX", "object_detection",
            "Declared YOLO11 detection shapes and model-selected OS/device",
            "SCOPED_PRODUCT_PATH", "SCOPED_NATIVE_WINDOWS_CUDA_PASSED",
            "NOT_APPLICABLE",
            _ownership(package=("pipeline.py", "verify.py"),
                       product=("import_adapters.py",),
                       pascal=("TVM.Yolo11.pas",),
                       tests=("test_verify.py", "test_wizard_next.py"),
                       pascal_tests=("test_verify_yolo11_gpu.py",)),
            (_route("PARTIAL_CONNECTED", "Wizard selected YOLO .pt source",
                    "Pinned exporter then common core"),
             _route("PARTIAL_CONNECTED", "pascalvision-convert --into",
                    "Per-model contract; not universal YOLO versions")),
            ("All declared decoder/shape/runtime/portable ZIP combinations "
             "are not established by one YOLO11n GPU UAT.",)),
        _family(
            "F-YOLO11-SEG", "YOLO11 instance segmentation",
            "Ultralytics .pt -> self-contained ONNX", "instance_segmentation",
            "Declared YOLO11 segmentation, masks and model-specific output ABI",
            "SCOPED_PRODUCT_PATH", "SCOPED_GPU_ORT_COMPARISON_PASSED",
            "NOT_APPLICABLE",
            _ownership(package=("pipeline.py", "contract.py"),
                       product=("import_adapters.py",),
                       pascal=("TVM.Yolo11.pas",),
                       tests=("test_wizard_next.py", "test_pipeline_manifest_security.py")),
            (_route("PARTIAL_CONNECTED", "Wizard selected YOLO .pt source",
                    "No inference acceptance on all target GPUs"),),
            ("Complete public Pascal segmentation ABI/consumer and "
             "cross-target model-family acceptance remain open.",)),
        _family(
            "F-RTDETR", "RT-DETR fused object detection",
            "Self-contained ONNX", "object_detection",
            "Exact rtdetr_fused_cxcywh_v1 [1,300,6], not arbitrary outputs",
            "SCOPED_PRODUCT_PATH", "SCOPED_WINDOWS_CUDA_PASCAL_PASSED",
            "NOT_APPLICABLE",
            _ownership(package=("manifest.py", "verify.py", "pipeline.py"),
                       pascal=("TVM.RtDetr.pas",),
                       tests=("test_rtdetr_output_passport.py",
                              "test_verify_rtdetr_fused.py")),
            (_route("PARTIAL_CONNECTED", "pascalvision-convert --into",
                    "Raw ONNX; fused adapter metadata must agree"),
             _route("PARTIAL_CONNECTED", "Wizard ONNX/RT-DETR source",
                    "Other RT-DETR variants not inferred")),
            ("All variant/profile/runtime/target end-to-end routes are not "
             "closed by a single fused RT-DETR-L validation.",)),
        _family(
            "F-BIREFNET-HR800", "Full BiRefNet HR800 SwinL",
            "Exact full SwinL ONNX/checkpoint", "foreground_segmentation",
            "Full HR800 800x800 with original checkpoint; IoU >= 0.99 oracle",
            "PARTIAL_SOURCE_ROUTE", "HISTORICAL_SCOPED_IOU_ONLY",
            "NOT_APPLICABLE",
            _ownership(package=("convert.py", "pipeline.py", "verify.py"),
                       pascal=("TVM.BiRefNet.pas",),
                       external=("4-ml/converters/convert_birefnet.py",),
                       tests=("test_verify.py",)),
            (_route("PARTIAL_CONNECTED", "pascalvision-convert --into",
                    "Raw ONNX path; full-model UAT not repeated"),
             _route("PARTIAL_CONNECTED", "Wizard ONNX source",
                    "Never substitute Lite/SwinT for full HR800 SwinL")),
            ("Fresh full HR800 CUDA target and independent PyTorch FP32 "
             "IoU>=0.99 acceptance remain uncompleted.",)),
        _family(
            "F-OMDET-TINY", "OmDet-Turbo tiny exact four-input v2",
            "Pinned four-input ONNX plus feature/config assets",
            "object_detection",
            "Exact eight classes; boxes [1,900,4], raw scores [1,900,8]",
            "PARTIAL_PUBLIC_PASCAL_V2", "HISTORICAL_STANDALONE_NATIVE_PASS",
            "NOT_APPLICABLE",
            _ownership(package=("omdet_public_candidate.py",
                                "omdet_bundle_profile.py", "manifest.py"),
                       pascal=("CrossVision.ML.OmDet.pas",
                               "TVM.OmDet.Pipeline.pas",
                               "TVM.OmDet.FixedFeatures.pas"),
                       tests=("test_step08_omdet_public_candidate.py",
                              "test_step08_omdet_pascal_native_manifest.py",
                              "test_step08_omdet_candidate_wizard.py")),
            (_route("CONNECTED_NOT_CHECKED", "omdet-candidate-bundle",
                    "Requires full independent caller-declared SHA pins"),
             _route("CONNECTED_NOT_CHECKED",
                    "Wizard OmDet candidate Preview/Build",
                    "One-use selected SHA/identity Preview and Build, no native Pascal UAT")),
            ("Actual trained public Pascal IDetector versus independent "
             "reference and target runtime/origin attestation NOT_RUN.",)),
        _family(
            "F-PADDLEOCR-DETREC", "PaddleOCR Det + Rec + CTC dictionary",
            "Paddle inference directory; pinned Det, Rec and UTF-8 dictionary",
            "ocr",
            "PP-OCRv5 det/rec/CTC; composite root must keep independent GT",
            "DRAFT_PLUS_ORIGINAL_ORT_NATIVE_DET_AND_PAIRWISE_ADVISORY",
            "SCOPED_TRAINED_MOBILE_DET_CPU_RAW_RUN_REC_CTC_NOT_RUN",
            "HUMAN_GROUND_TRUTH_NOT_ATTESTED",
            _ownership(package=("paddle_closure.py", "ocr_composite.py",
                                "ocr_composite_oracle.py",
                                "ocr_ort_raw_capture.py", "ocr_capture_bridge.py",
                                "ocr_native_det_capture_bridge.py",
                                "ocr_det_pairwise.py"),
                       pascal=("TVM.PaddleOcr.pas",),
                       product=("import_adapters.py",),
                       external=("4-ml/tests/PaddleOcrNativeCapture.dpr",
                                 "4-ml/tests/PaddleOcrNativeRecCapture.dpr"),
                       tests=("test_step08_ocr_composite_wizard_draft.py",
                              "test_step08_paddle_wizard_selected_binding.py",
                              "test_step08_ocr_ort_capture_cli.py",
                              "test_step08_ocr_ort_capture_bridge.py",
                              "test_step08_ocr_native_det_capture_bridge.py",
                              "test_step08_ocr_det_pairwise.py",
                              "test_step08_ocr_det_pairwise_public_cli.py",
                             "test_step08_ocr_ort_dynamic_shape_specialization.py",
                             "test_step08_native_rec_raw_dpr_contract.py",
                             "test_step08_native_pascal_path_and_roi_contract.py",
                             "test_step08_ocr_native_rec_preprocess_public_contract.py",
                             "test_step08_native_rec_truthful_product_checkpoint.py"),
                       pascal_tests=("test_verify_paddleocr_gpu.py",)),
            (_route("DRAFT_NOT_CHECKED", "ocr-composite",
                    "Det/Rec/dictionary assembly only"),
             _route("CONNECTED_NOT_CHECKED", "ocr-ort-raw",
                    "One original ORT raw model/input capture route; new trained "
                    "ORT comparison NOT_RUN; trained Mobile Det Pascal raw "
                    "producer has executed but is not independently accepted"),
             _route("READ_ONLY_NOT_CHECKED",
                    "ocr_capture_bridge.inspect_ort_capture",
                    "Read-only SHA/shape projection of supplied capture, not "
                    "human GT, source image or native producer attestation"),
             _route("READ_ONLY_NOT_CHECKED",
                    "ocr_native_det_capture_bridge.inspect_native_det_capture",
                    "Det-only Pascal receipt integrity/projection, real native "
                    "Det scoped natural0 but origin NOT_ATTESTED, no root acceptance"),
             _route("READ_ONLY_NOT_CHECKED", "ocr-det-pairwise",
                    "Actual original ORT/Pascal Det scalar comparison adapter, "
                    "supplied-file-only status NOT_CHECKED; Rec/CTC/GT open"),
             _route("SOURCE_ONLY_NOT_RUN",
                    "4-ml/tests/PaddleOcrNativeRecCapture.dpr",
                    "Actual production PpocrPreprocess and post-softmax raw "
                    "score capture are written but NOT compiled/executed; "
                    "no independent original ORT, source crop ROI/GT or READY"),
             _route("ADVISORY_NOT_CHECKED", "ocr-composite-audit",
                    "No independent producers run; supplied numeric matches exit2"),
             _route("DRAFT_NOT_CHECKED", "Wizard OCR composite",
                    "No human GT-backed verified root")),
            ("Verified-root consumer passport/gate, independent native "
             "Det/Rec/CTC capture, coherent CTC sidecars and human-text GT "
             "remain unimplemented or unaccepted.",)),
        _family(
            "F-TFLITE-FP32", "Opt-in TFLite floating-point RAW",
            "Self-contained .tflite", "model_defined",
            "Only strict self-contained FLOAT32, no quantized input/output",
            "PARTIAL_RAW_OPT_IN", "FAMILY_NATIVE_UAT_NOT_RUN",
            "NOT_APPLICABLE",
            _ownership(package=("convert.py", "pipeline.py", "verify.py",
                                "source_freeze.py"),
                       tests=("test_verify_tflite_magic_step07.py",
                              "test_pipeline_manifest_security.py")),
            (_route("OPT_IN_LIMITED", "pascalvision-convert --into",
                    "Exact file/schema admission and independent TFLite oracle"),
             _route("OPT_IN_LIMITED", "Wizard selected .tflite source",
                    "Unsupported schema and quantized variants fail closed")),
            ("Complete selected TFLite schema closure and full public "
             "Pascal deployment/UAT not yet established.",)),
        _family(
            "F-HF-QWEN", "Qwen2.5-0.5B native LLM",
            "Pinned HuggingFace safetensors directory", "text_generation",
            "Exact Qwen2.5-0.5B profile and supported float32/float16",
            "PARTIAL_NATIVE_FAMILY_BUILDER", "SCOPED_LOCAL_NATIVE_HISTORY",
            "NOT_APPLICABLE",
            _ownership(package=("hf_closure.py", "hf_held_stage.py",
                                "llm_bundle_contract.py",
                                "native_consumer_hold.py"),
                       product=("import_adapters.py",),
                       external=("1-bindings/tvm/llm/build_llm.py",),
                       tests=("test_step08_hf_source_output_alias.py",
                              "test_step08_llm_family_atomic_publication.py",
                              "test_llm_bundle_contract_step04.py")),
            (_route("OPT_IN_PARTIAL", "pascalvision-convert --native-family-atomic",
                    "Held private HF selected source/consumer; NOT_CHECKED"),
             _route("OPT_IN_PARTIAL", "Wizard native_family_atomic",
                    "Explicit selected consumer and no-clobber bundle")),
            ("Model licence authenticity, runtime provenance, full target/"
             "tokenizer closure and trained numerical family acceptance OPEN.",)),
        _family(
            "F-HF-GEMMA", "Gemma2-2B native LLM",
            "Pinned HuggingFace safetensors directory", "text_generation",
            "Exact Gemma-2-2B profile and supported float32/float16",
            "PARTIAL_NATIVE_FAMILY_BUILDER", "FAMILY_NATIVE_UAT_NOT_RUN",
            "NOT_APPLICABLE",
            _ownership(package=("hf_closure.py", "hf_held_stage.py",
                                "llm_family_profile.py",
                                "llm_bundle_contract.py"),
                       product=("import_adapters.py",),
                       external=("1-bindings/tvm/llm/build_llm.py",),
                       tests=("test_step08_hf_build_binding.py",
                              "test_native_llm_plan_step04.py")),
            (_route("OPT_IN_PARTIAL", "pascalvision-convert --native-family-atomic",
                    "Exact registry family, not arbitrary Gemma checkpoint"),
             _route("OPT_IN_PARTIAL", "Wizard native_family_atomic",
                    "Same typed plan and pre-staged selected consumer")),
            ("Independent real Gemma native Pascal decode/oracle, licence "
             "and portable runtime acceptance are not demonstrated.",)),
    ]

    work_ids = {row["id"] for row in work_items}
    family_ids = {row["id"] for row in families}
    if (work_ids != {"I-%02d" % n for n in range(1, 12)} or
            len(work_ids) != len(work_items) or
            len(family_ids) != len(families)):
        raise ValueError("normative support-inventory IDs are missing or duplicated")

    type1_open_ids = [
        item["id"] for item in work_items + families
        if item["implementation_status"] == _TYPE1
    ]
    open_work_ids = {item["id"] for item in work_items
                     if item["implementation_status"] == _TYPE1}
    v_stages = [
        _stage("V1", "TargetProfile and common BuildPlan",
               ("I-01", "I-04"), open_work_ids),
        _stage("V2", "Hardware discovery and machine-specific recommendations",
               ("I-01", "I-04", "I-09"), open_work_ids),
        _stage("V3", "Effective TVM target and selected architecture",
               ("I-01", "I-04", "I-08"), open_work_ids),
        _stage("V4", "CUDA runtime closure and portable target behavior",
               ("I-03", "I-04", "I-05"), open_work_ids),
        _stage("V5", "Next-only Wizard with editable defaults",
               ("I-01", "I-02", "I-09"), open_work_ids),
        _stage("V6", "Truthful recipe, manifest and artifact provenance",
               ("I-01", "I-03", "I-10"), open_work_ids),
        _stage("V7", "Independent model, Pascal and target acceptance",
               ("I-05", "I-06", "I-07", "I-08"), open_work_ids),
        _stage("V8", "Safe selective original/source and product integration",
               ("I-08", "I-10", "I-11"), open_work_ids),
    ]

    unsupported = [
        {
            "id": "U-TFLITE-QUANTIZED",
            "format": "TFLite uint8/int8/other quantized tensors",
            "status": "UNSUPPORTED_BY_DESIGN",
            "reason": "Only selected self-contained FP32 route is declared; "
                      "quantized tensors may silently change numerical meaning.",
        },
        {
            "id": "U-UNKNOWN-PT-H5",
            "format": "Arbitrary PyTorch .pt/.pth, Keras .h5 or other HF weights",
            "status": "UNSUPPORTED_BY_DESIGN",
            "reason": "Export-first via an explicitly supported framework is "
                      "required; an unknown family is not auto-converted.",
        },
        {
            "id": "U-FOREIGN-OS-ISA-GPU",
            "format": "Unproven foreign OS/CPU ISA/SM/driver or PTX/fatbin",
            "status": "UNSUPPORTED_OR_RUNNER_NEEDED",
            "reason": "No generic portability guarantee; support is selected "
                      "per compiler, binary, runtime and available runner.",
        },
        {
            "id": "U-OMDET-UNPINNED",
            "format": "Arbitrary OmDet class counts and self-authenticated v1",
            "status": "UNSUPPORTED_BY_DESIGN",
            "reason": "Public four-input v2 is exact tiny/eight-class only; "
                      "independent external full expected_pins are mandatory.",
        },
    ]
    xfails = [
        {
            "id": "XFAIL-COMPILER-WINDOW-SOURCE-REWRITE",
            "status": "STRICT_XFAIL_OPEN",
            "reason": "Selected source temporarily changed and restored "
                      "during compilation is not universally prevented.",
            "test": _TESTS + "test_preview_source_content_gap.py",
            "owner_work_items": ["I-02", "I-11"],
        }
    ]
    
    
    release_blockers = list(type1_open_ids)
    release_blockers.extend(row["id"] for row in xfails
                            if row["status"] == "STRICT_XFAIL_OPEN")
    type1_zero = not type1_open_ids and not release_blockers
    summary = {
        "work_items_declared": len(work_items),
        "v_stages_declared": len(v_stages),
        "families_declared": len(families),
        "type1_open_count": len(type1_open_ids),
        "type1_open_ids": type1_open_ids,
        "strict_xfail_open": bool(xfails),
        "type1_zero": type1_zero,
        "release_blocker_ids": release_blockers,
        "release_status": _NOT_RELEASED,
        "verified_globally": False,
    }
    return {
        "schema": SCHEMA,
        "snapshot_date": SNAPSHOT,
        "inventory_kind": "DECLARED_SOURCE_REVIEW_PRELIMINARY",
        "source_of_truth": [
            "state/PASCALVISION-IMPLEMENTATION-DEBT-PLAN-20261014.md",
            "state/PASCALVISION-TARGET-CONFIG-DNA.md",
            "tools/pascalvision-convert/SUPPORT-MATRIX.md",
        ],
        "normative_scope": {
            "work_items": "I-01..I-11",
            "v_stages": "V1..V8",
            "does_not_replace_28_step_debt_plan": True,
            "global_scope_reduced": False,
            "type1_definition": "IMPLEMENTATION_OPEN includes missing "
                                "code, integration, user route or reliable "
                                "refusal for a declared capability.",
            "type2_definition": "VALIDATION_OPEN requires a complete "
                                "software route; mocks cannot upgrade it "
                                "to independently accepted native or human UAT.",
        },
        "work_items": work_items,
        "v_stages": v_stages,
        "families": families,
        "unsupported": unsupported,
        "historic_strict_xfails": xfails,
        "summary": summary,
        "gate": {
            "kind": "PRELIMINARY_TYPE1_ZERO_ADVISORY",
            "attested": False,
            "is_final_release_gate": False,
            "ready_to_release": False,
            "type1_zero": type1_zero,
            "missing_final_obligations": [
                "Complete and independently audit all I-01..I-11 implementations",
                "Close supported-path strict XFAIL",
                "Run real native/independent-oracle and human acceptance",
                "Obtain separate owner authorization for any publication",
            ],
        },
    }


def main(argv=None) -> int:
    """Print read-only JSON and optionally fail if preliminary TYPE-1 is nonzero."""
    parser = argparse.ArgumentParser(
        prog="pascalvision-convert support-inventory",
        description="Read-only I-01..I-11/V1..V8 advisory support and debt inventory.",
    )
    parser.add_argument(
        "--require-type1-zero", action="store_true",
        help="return nonzero when any declared TYPE-1 or supported-path "
             "strict XFAIL remains; never grants final release approval",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if type(error.code) is int else 2

    inventory = build_inventory()
    print(json.dumps(inventory, sort_keys=True, ensure_ascii=False,
                     indent=2, allow_nan=False))
    if args.require_type1_zero and not inventory["summary"]["type1_zero"]:
        print(
            "TYPE1_ZERO_REFUSED: preliminary software obligations remain OPEN; "
            "inventory is advisory and no release is approved",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
