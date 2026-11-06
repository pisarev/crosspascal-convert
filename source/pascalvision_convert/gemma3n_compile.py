# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
import json
import hashlib
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent))
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from pascalvision_convert.gemma_cpu_schedule import schedule_gemma_cpu
from pascalvision_convert.convert import _shared_library_builder

bundle = Path(sys.argv[1])
graph = onnx.load(bundle / "text.onnx", load_external_data=False)
module = from_onnx(graph, keep_params_in_input=False)
print("Imported Gemma 3n token graph", flush=True)
target = tvm.target.Target({"kind": "llvm", "mcpu": sys.argv[2] if len(sys.argv) > 2 else "generic", "num-cores": 4})
with target:
    module = relax.transform.FuseTransposeMatmul()(module)
    module = schedule_gemma_cpu(relax.get_pipeline("zero")(module))
executable = relax.build(module, target=target)
executable.export_library(str(bundle / "gemma3n-cpu.dll"), fcompile=_shared_library_builder())
manifest = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
manifest["artifact_sha256"] = {}
for name in ("gemma3n-cpu.dll", "text.onnx", "parameters.json", "tokenizer.model", "tokenizer-gguf.json"):
    with (bundle / name).open("rb") as stream:
        manifest["artifact_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
manifest["compiler"] = {"tvm": tvm.__version__, "target": str(target)}
(bundle / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("TVM DLL ready", flush=True)
