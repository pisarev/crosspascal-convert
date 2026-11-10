# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.

import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import onnx
import tvm
from tvm import relax,te,s_tir
from tvm.s_tir import dlight
from tvm.relax.frontend.onnx import from_onnx


def q8_matvec(shape, rows):
    shape=tuple(int(v) for v in shape)
    width=shape[-1]
    x=te.placeholder(shape,'float32',name='x')
    q=te.placeholder((rows,width),'int8',name='q')
    scale=te.placeholder((rows,),'float32',name='scale')
    reduction=te.reduce_axis((0,width),'k')
    out=te.compute(shape[:-1]+(rows,),lambda *i: te.sum(
        x[i[:-1]+(reduction,)]*q[i[-1],reduction].astype('float32')*scale[i[-1]],
        axis=reduction),name='q8_dot')
    schedule=s_tir.Schedule(te.create_prim_func([x,q,scale,out]))
    block=schedule.get_sblock('q8_dot')
    loops=schedule.get_loops(block)
    row=schedule.fuse(*loops[:-1])
    _,lane=schedule.split(loops[-1],factors=[None,128])
    schedule.bind(row,'blockIdx.x')
    schedule.bind(lane,'threadIdx.x')
    return schedule.mod['main'].without_attr('global_symbol').with_attr('tirx.is_scheduled',True).with_attr('op_pattern',8)


@relax.expr_functor.mutator
class QuantizedMatvec(relax.PyExprMutator):
    def __init__(self,module,selected):
        super().__init__(module)
        self.selected=selected
        self.new_vars={}
        self.kernels={}
        self.replacements=0
        self.bindings=[]

    def visit_function_(self,func):
        parameters=[]
        for param in func.params:
            if param in self.selected:
                shape=param.ty.shape
                q=relax.Var(param.name+'_q8',relax.TensorType(shape,'int8'))
                scale=relax.Var(param.name+'_scale',relax.TensorType([shape[0]],'float32'))
                self.new_vars[param]=(q,scale)
                parameters.extend((q,scale))
                self.bindings.extend(({'source':self.selected[param],'kind':'q8'},
                                      {'source':self.selected[param],'kind':'scale'}))
            else:
                parameters.append(param)
                self.bindings.append({'source':param.name,'kind':'original'})
        body=self.visit_expr(func.body)
        return relax.Function(parameters,body,func.ret_ty,func.is_pure,func.attrs,func.span)

    def visit_call_(self,call):
        if (isinstance(call.op,tvm.ir.Op) and call.op.name=='relax.power' and
                isinstance(call.args[1],relax.Constant)):
            exponent=call.args[1].data.numpy()
            if exponent.size==1 and float(exponent.reshape(-1)[0])==2.0:
                value=self.visit_expr(call.args[0])
                return relax.op.multiply(value,value)
        if (isinstance(call.op,tvm.ir.Op) and call.op.name=='relax.call_tir' and
                call.args[0].name_hint.startswith('NT_matmul')):
            arguments=call.args[1].fields
            if len(arguments)==2 and arguments[1] in self.new_vars:
                x=self.visit_expr(arguments[0])
                weight=arguments[1]
                shape=tuple(int(v) for v in x.ty.shape)
                rows=int(weight.ty.shape[0])
                key=(shape,rows)
                if key not in self.kernels:
                    self.kernels[key]=self.builder_.add_func(q8_matvec(shape,rows),'gemma_q8_'+str(len(self.kernels)))
                self.replacements+=1
                return relax.call_tir(self.kernels[key],(x,*self.new_vars[weight]),out_ty=call.ty)
        return super().visit_call_(call)


def compile_gpu(bundle,arch):
    from pascalvision_convert.convert import _shared_library_builder
    from pascalvision_convert.targets import normalize_cuda_sm
    arch=normalize_cuda_sm(arch)
    if not arch:
        dev=tvm.cuda()
        if not dev.exist:
            raise ValueError('Select a CUDA architecture or attach a CUDA device')
        arch=normalize_cuda_sm(dev.compute_version)
    manifest=json.loads((bundle/'model.json').read_text())
    if not manifest.get('host_embeddings'):
        raise ValueError('GPU export must receive selected host embedding rows')
    record=json.loads((bundle/'parameters.json').read_text())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Renaming name .*')
        module=from_onnx(onnx.load(bundle/'text.onnx'),keep_params_in_input=False)
    module=relax.transform.FuseTransposeMatmul()(module)
    main=module['main']
    names=manifest['input_names']
    if len(main.params)!=len(names):
        raise ValueError('Unexpected imported parameter ordering')
    sources={param.name:name for param,name in zip(main.params,names)}
    selected={param:name for param,name in zip(main.params,names)
              if name in manifest['parameters'] and len(param.ty.shape)==2 and
              int(param.ty.shape[0])*int(param.ty.shape[1])>=256}
    rewriter=QuantizedMatvec(module,selected)
    rewritten=rewriter.visit_expr(main)
    module=rewriter.builder_.get()
    module.update_func(module.get_global_var('main'),rewritten)

    relax.analysis.well_formed(module)
    quantized={}
    for index,name in enumerate(selected.values()):
        spec=manifest['parameters'][name]
        original=record['weights'][spec['tensor']]
        weight=np.memmap(bundle/original['file'],mode='r',dtype='float32',shape=tuple(original['shape']))
        if spec['slice'] is not None:
            weight=weight[spec['slice']]
        qfile=f'gpu-q8-{index:04d}.bin'
        sfile=f'gpu-scale-{index:04d}.bin'
        q=np.memmap(bundle/qfile,mode='w+',dtype='int8',shape=weight.shape)
        scale=np.memmap(bundle/sfile,mode='w+',dtype='float32',shape=(weight.shape[0],))
        for start in range(0,len(weight),128):
            chunk=np.asarray(weight[start:start+128])
            row_scale=np.maximum(np.max(np.abs(chunk),axis=1)/127,np.finfo('float32').tiny)
            scale[start:start+128]=row_scale
            q[start:start+128]=np.clip(np.rint(chunk/row_scale[:,None]),-127,127).astype('int8')
        q.flush();scale.flush()
        quantized[name]={'q8':qfile,'scale':sfile,'shape':list(weight.shape)}
        del q,scale,weight
    target=tvm.target.Target({'kind':'cuda','arch':arch},host='llvm')
    print(f'CUDA graph: {rewriter.replacements} quantized projections, {len(rewriter.kernels)} kernels',flush=True)
    with target:
        module=relax.get_pipeline('zero')(module)
        module=dlight.ApplyDefaultSchedule(dlight.gpu.Matmul(),dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),dlight.gpu.GeneralReduction(),dlight.gpu.Fallback())(module)
    executable=relax.build(module,target=target)
    library='gemma3n-cuda.dll'
    executable.export_library(str(bundle/library),fcompile=_shared_library_builder())
    manifest.update({'backend':'TVM CUDA W8A32','library':library,'cuda_arch':arch,
        'quantization':{'scheme':'symmetric-int8-per-row','activation_dtype':'float32','weights':quantized},
        'gpu_bindings':[dict(item,source=sources.get(item['source'],item['source'])) for item in rewriter.bindings],
        'compiler':{'tvm':tvm.__version__,'target':str(target)}})
    manifest['artifact_sha256']={}
    for name in (library,'text.onnx','parameters.json','tokenizer.model','tokenizer-gguf.json'):
        with (bundle/name).open('rb') as stream:
            manifest['artifact_sha256'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
    (bundle/'model.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('CUDA bundle ready',flush=True)


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    compile_gpu(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else None)
