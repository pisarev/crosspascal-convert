# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.

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
from pascalvision_convert.onnx_integer_square import replace_squares
from pascalvision_convert.targets import normalize_cuda_sm
from pascalvision_convert.gemma_cpu_schedule import schedule_gemma_cpu
from pascalvision_convert.cpu_vision_schedule import schedule_cpu_vision


def compile_vision(folder):
    manifest = json.loads((folder / 'vision.json').read_bytes())
    cuda = manifest['backend'] == 'TVM CUDA FP32'
    if manifest['backend'] not in ('TVM CUDA FP32', 'TVM CPU FP32'):
        raise ValueError('Unsupported vision compilation backend')
    arch = normalize_cuda_sm(manifest.get('cuda_arch')) if cuda else None
    if cuda and not arch:
        raise ValueError('A destination CUDA architecture is required')
    graph = onnx.load(folder / 'vision.onnx')
    replace_squares(graph)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Renaming name .*')
        module = from_onnx(graph, keep_params_in_input=False)
    target = (tvm.target.Target({'kind': 'cuda', 'arch': arch}, host='llvm') if cuda else
              tvm.target.Target({'kind': 'llvm', 'mcpu': manifest.get('cpu_arch', 'generic'), 'num-cores': 4}))
    with target:
        module = relax.transform.FuseTransposeMatmul()(module)
        module = relax.get_pipeline('zero')(module)
        if cuda:
            module = dlight.ApplyDefaultSchedule(dlight.gpu.Matmul(), dlight.gpu.GEMV(),
                dlight.gpu.Reduction(), dlight.gpu.GeneralReduction(), dlight.gpu.Fallback())(module)
        else:
            module = schedule_gemma_cpu(schedule_cpu_vision(module))
    executable = relax.build(module, target=target)
    executable.export_library(str(folder / manifest['library']), fcompile=_shared_library_builder())
    for name in (manifest['library'], manifest['layout'], manifest['parameters'], manifest['hard_embeddings'],
                 manifest['hard_norm'], manifest['projection'], 'vision.onnx'):
        with (folder / name).open('rb') as stream:
            manifest['artifact_sha256'][name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest['compiler'] = {'tvm': tvm.__version__, 'target': str(target)}
    (folder / 'vision.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Vision bundle compiled for ' + str(target), flush=True)


if __name__ == '__main__':
    compile_vision(Path(sys.argv[1]))
