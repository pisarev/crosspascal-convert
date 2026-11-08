# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Compile configuration-derived Qwen2.5-VL graphs for an explicit CPU target."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import onnx
import tvm
from tvm import relax
from tvm.s_tir import dlight
from tvm.relax.frontend.onnx import from_onnx
from pascalvision_convert.convert import _shared_library_builder
from pascalvision_convert.cpu_vision_schedule import schedule_cpu_vision
from pascalvision_convert.onnx_integer_square import replace_squares


def compile_bundle(folder, cpu_arch, selected=None):
    manifest = json.loads((folder / "model.json").read_bytes())
    if manifest.get("schema") != "crosspascal.qwen2_5_vl.v1":
        raise ValueError("Expected an exported Qwen2.5-VL bundle")
    if selected is not None and selected not in manifest['graphs']:
        raise ValueError('The selected graph has not been exported: ' + selected)
    target = tvm.target.Target({"kind": "llvm", "mcpu": cpu_arch, "num-cores": 4})
    hashes = dict(manifest.get('artifact_sha256', {})) if selected else {}
    if selected and manifest.get('cpu_arch') != cpu_arch:
        raise ValueError('A partial compile must preserve the existing CPU target')
    for kind, graph in manifest["graphs"].items():
        if selected and kind != selected:
            for name in (graph['onnx'], graph['library']):
                with (folder/name).open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest() != hashes.get(name):
                        raise ValueError('Cannot reuse a changed graph: ' + name)
            continue
        model = onnx.load(folder / graph["onnx"], load_external_data=False)
        replace_squares(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Renaming name .*")
            module = from_onnx(model, keep_params_in_input=False)
        with target:
            module = relax.transform.FuseTransposeMatmul()(module)
            module = relax.get_pipeline("zero")(module)
            module = schedule_cpu_vision(module)
            module = dlight.ApplyDefaultSchedule(dlight.cpu.Reduction())(module)
        executable = relax.build(module, target=target)
        graph["library"] = "qwen-vl-" + kind + "-cpu.dll"
        executable.export_library(str(folder / graph["library"]), fcompile=_shared_library_builder())
        print("Compiled " + kind, flush=True)
        for name in (graph["library"], graph["onnx"]):
            with (folder / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    for name in ("parameters.json", "tokenizer.json", "tokenizer_config.json", "config.json",
                 "preprocessor_config.json", "generation_config.json"):
        if (folder / name).is_file():
            with (folder / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest["artifact_sha256"] = hashes
    manifest["cpu_arch"] = cpu_arch
    manifest["compiler"] = {"tvm": tvm.__version__, "target": str(target)}
    (folder / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cpu-arch", default="generic")
    parser.add_argument('--graph', choices=('text','vision','prefill'))
    args = parser.parse_args()
    compile_bundle(args.bundle, args.cpu_arch, args.graph)
