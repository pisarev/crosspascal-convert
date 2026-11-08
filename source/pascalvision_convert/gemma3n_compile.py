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
from pascalvision_convert.cpu_vision_schedule import schedule_cpu_vision
from tvm.s_tir import dlight
from pascalvision_convert.convert import _shared_library_builder

bundle = Path(sys.argv[1])
manifest = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
cpu = sys.argv[2] if len(sys.argv) > 2 else 'generic'
prefill_only = '--prefill-only' in sys.argv[3:]
if prefill_only and manifest.get('cpu_arch') != cpu:
    raise ValueError('Partial compilation must preserve the token graph CPU target')
target = tvm.target.Target({'kind':'llvm','mcpu':cpu,'num-cores':4})
hashes = dict(manifest.get('artifact_sha256',{})) if prefill_only else {}
graphs = [('text','text.onnx','gemma3n-cpu.dll')]
if manifest.get('prefill'):
    graphs.append(('prefill',manifest['prefill']['onnx'],manifest['prefill']['library']))
elif prefill_only:
    raise ValueError('No exported prefill graph')
for kind, filename, library in graphs:
    if prefill_only and kind == 'text':
        for name in (filename,library):
            with (bundle/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != hashes.get(name):
                    raise ValueError('Cannot reuse a changed token graph: '+name)
        continue
    graph = onnx.load(bundle/filename,load_external_data=False)
    module = from_onnx(graph,keep_params_in_input=False)
    print('Imported Gemma 3n '+kind+' graph',flush=True)
    with target:
        module = relax.transform.FuseTransposeMatmul()(module)
        module = relax.get_pipeline('zero')(module)
        if kind == 'prefill':
            module = dlight.ApplyDefaultSchedule(dlight.cpu.Reduction())(schedule_cpu_vision(module))
        else:
            module = schedule_gemma_cpu(module)
    executable = relax.build(module,target=target)
    executable.export_library(str(bundle/library),fcompile=_shared_library_builder())
    for name in (filename,library):
        with (bundle/name).open('rb') as stream:
            hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
record = json.loads((bundle/'parameters.json').read_bytes())
names = {'parameters.json','tokenizer.model','tokenizer-gguf.json'}
names.update(row['file'] for row in record['weights'].values())
for name in sorted(names):
    with (bundle / name).open("rb") as stream:
        hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
manifest['artifact_sha256'] = hashes
manifest["compiler"] = {"tvm": tvm.__version__, "target": str(target)}
manifest["cpu_arch"] = cpu
(bundle / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("TVM DLL ready", flush=True)
