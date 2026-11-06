# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Read-only inventory of the documented CrossPascal Converter Early Access scope."""

from __future__ import annotations

import argparse
import json
import sys


SCHEMA = "crosspascal.pascalvision.support-inventory.v2"


def build_inventory() -> dict:
    """Describe the documented routes without inspecting the local machine."""
    cpu_profiles = (
        ("squeezenet", "SqueezeNet 1.1", "image_classification"),
        ("resnet50", "ResNet50", "image_classification"),
        ("yolo11n-cls", "YOLO11n classification", "image_classification"),
        ("yolo11n", "YOLO11n detection", "object_detection"),
        ("rtdetr-r18vd", "RT-DETR R18vd", "object_detection"),
    )
    families = [
        {
            "id": profile,
            "name": name,
            "task": task,
            "source_format": "ONNX from the matching CPU model source pack",
            "target": "cpu",
        }
        for profile, name, task in cpu_profiles
    ]
    families.append({
        "id": "qwen2.5-0.5b",
        "name": "Qwen2.5-0.5B-Instruct",
        "task": "text_generation",
        "source_format": "Pinned Hugging Face safetensors",
        "target": "cuda",
        "profile": "qwen2.5-0.5b-cuda-f32-64-v1",
        "architecture": "Detected on the build host or selected for the destination",
    })
    return {
        "schema": SCHEMA,
        "inventory_kind": "DOCUMENTED_EARLY_ACCESS_SCOPE",
        "source_of_truth": ["README.md", "INSTALL.md"],
        "platform": "Windows x64",
        "pascal_examples": ["Delphi 13.2", "Lazarus 4.8"],
        "families": families,
        "limits": [
            "Other included families and compilation paths are outside the verified Early Access scope.",
            "Run the matching Pascal example to verify inference on the destination computer.",
            "The output runtime, device and driver must support the architecture selected for conversion.",
        ],
        "summary": {
            "families_declared": len(families),
            "type1_zero": False,
            "verified_globally": False,
        },
        "gate": {
            "kind": "ADVISORY_SUPPORT_SCOPE",
            "attested": False,
            "is_final_release_gate": False,
            "ready_to_release": False,
            "type1_zero": False,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pascalvision-convert support-inventory",
        description="Show the documented Early Access model conversion routes.",
    )
    parser.add_argument(
        "--require-type1-zero", action="store_true",
        help="require full implementation coverage beyond Early Access",
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
            "TYPE1_ZERO_REFUSED: this advisory inventory covers Early Access only",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
