# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""CUDA execution with persistent compact weights and host embedding rows."""
import atexit
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
from concurrent.futures import Future
import queue
import threading
import time
import numpy as np
from .gemma3n_runtime import Session
from .gemma3n_workspace import CudaWorkspace
from .generation import Generation
import sentencepiece
import tvm
from tvm import relax


def cuda_memory():
    driver=ctypes.WinDLL('nvcuda.dll')
    free,total=ctypes.c_size_t(),ctypes.c_size_t()
    code=driver.cuMemGetInfo_v2(ctypes.byref(free),ctypes.byref(total))
    if code:
        raise RuntimeError(f'CUDA memory query failed: {code}')
    return free.value,total.value


class _CudaSession(Session):
    def __init__(self,folder,device_id=0):
        started=time.perf_counter()
        self.folder=Path(folder).resolve(strict=True)
        self.manifest=self.read_json('model.json')
        if self.manifest.get('backend')!='TVM CUDA W8A32' or not self.manifest.get('host_embeddings'):
            raise ValueError('Expected a native CUDA Gemma 3n bundle')
        for filename,expected in self.manifest['artifact_sha256'].items():
            with self.member(filename).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                    raise ValueError('Bundle artifact changed: '+filename)
        self.record=self.read_json('parameters.json')
        self.metadata=self.read_json('tokenizer-gguf.json')
        self.vocabulary=self.metadata['tokenizer.ggml.tokens']
        self.special={word:index for index,word in enumerate(self.vocabulary)
                      if word in ('<start_of_turn>','<end_of_turn>','<eos>')}
        self.tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(self.member('tokenizer.model')))
        self.device=tvm.cuda(device_id)
        if not self.device.exist:
            raise RuntimeError('The selected CUDA device is unavailable; no CPU fallback is enabled')
                                                                                    
        context_probe=tvm.runtime.tensor(np.zeros(1,dtype='float32'),self.device)
        self.device.sync()
        del context_probe
        self.memory_before=cuda_memory()
        self.arrays={}
        for name,spec in self.record['weights'].items():
            path=self.member(spec['file'])
            dtype=np.dtype(spec.get('dtype','float32'))
            if path.stat().st_size!=math.prod(spec['shape'])*dtype.itemsize:
                raise ValueError('Wrong parameter size: '+spec['file'])
            self.arrays[name]=np.memmap(path,mode='r',dtype=dtype,shape=tuple(spec['shape']))
        self.host_tables=[]
        for name in ('embed_tokens.weight','embed_tokens_per_layer.weight'):
            self.host_tables.append(self.arrays[self.record['parameters'][name]['tensor']])
        self.bindings=self.manifest['gpu_bindings'][len(self.manifest['runtime_inputs']):]
        quantized=self.manifest['quantization']['weights']
        self.host_parameters=[]
        for binding in self.bindings:
            name,kind=binding['source'],binding['kind']
            if kind in ('q8','scale'):
                entry=quantized[name]
                dtype,shape=('int8',tuple(entry['shape'])) if kind=='q8' else ('float32',(entry['shape'][0],))
                path=self.member(entry[kind])
                if path.stat().st_size!=math.prod(shape)*np.dtype(dtype).itemsize:
                    raise ValueError('Wrong quantized parameter size: '+entry[kind])
                value=np.memmap(path,mode='r',dtype=dtype,shape=shape)
            else:
                spec=self.manifest['parameters'][name]
                value=self.arrays[spec['tensor']]
                if spec['slice'] is not None:
                    value=value[spec['slice']]
            self.host_parameters.append(value)
        self.parameter_bytes=sum(value.nbytes for value in self.host_parameters)
        cache_bytes=int(np.prod(self.manifest['cache_shape']))*4*self.manifest['cache_count']
        required=self.parameter_bytes+3*cache_bytes+1024**3
        if self.memory_before[0]<required:
            raise MemoryError(f'CUDA requires about {required/1024**3:.2f} GiB free for this bundle; available {self.memory_before[0]/1024**3:.2f} GiB')
        library=self.member(self.manifest['library'])
        claim=CudaWorkspace.claim(library)
        try:
            self.vm=relax.VirtualMachine(tvm.runtime.load_module(str(library)),self.device)
            self.parameters=[tvm.runtime.tensor(np.asarray(value),self.device) for value in self.host_parameters]
            self.workspace=CudaWorkspace(library,device_id,claimed_key=claim)
        except BaseException:
            CudaWorkspace.release_claim(claim)
            raise
        self.device.sync()
        self.memory_loaded=cuda_memory()
        self.load_seconds=time.perf_counter()-started
        self.vision=None
        if self.manifest.get('vision_manifest'):
            from .gemma3n_vision_runtime import Vision
            if self.manifest['vision_manifest']!='vision.json':
                self.close()
                raise ValueError('Unsupported vision manifest member')
            try:
                self.vision=Vision(self.folder,self.host_tables[0].shape[1],device_id)
            except BaseException:
                self.close()
                raise

    def close(self):
        if getattr(self,'vm',None) is None:
            return
        self.device.sync()
        self.workspace.close()
        self.parameters.clear()
        self.vm=None
        gc.collect()
        self.device.sync()
        self.host_parameters.clear()
        self.host_tables.clear()
        self.arrays.clear()
        self.vision=None

    def __enter__(self):
        return self

    def __exit__(self,*exception):
        self.close()

    def new_cache(self):
        return [tvm.runtime.tensor(np.zeros(self.manifest['cache_shape'],dtype='float32'),self.device)
                for _ in range(self.manifest['cache_count'])]

    def step(self,token,position,caches,embedding=None):
        if self.vm is None:
            raise RuntimeError('This CUDA session is closed')
        if not 0<=position<self.manifest['context']:
            raise ValueError('Token position exceeds the compiled context')
        if embedding is None:
            if any(not 0<=token<len(table) for table in self.host_tables):
                raise ValueError('Token is outside the source embedding tables')
            rows=[np.array(table[token],copy=True) for table in self.host_tables]
        else:
                                                                            
                                                                         
            rows=[np.asarray(embedding,dtype='float32')/math.sqrt(self.host_tables[0].shape[1]),
                  np.array(self.host_tables[1][token if 0<=token<len(self.host_tables[1]) else 0],copy=True)]
        embeddings=[tvm.runtime.tensor(row.reshape(1,1,-1),self.device) for row in rows]
        index=tvm.runtime.tensor(np.array([position],dtype='int64'),self.device)
        try:
            result=self.vm['main'](*embeddings,index,*caches,*self.parameters)
        except Exception:
            self.workspace.check()
            raise
        self.workspace.check()
        return result[0],list(result[1:])

    def generate(self,prompt,max_new_tokens=0,on_text=None,should_stop=None,chat=True,options=None,images=None):
        if type(max_new_tokens) is not int or max_new_tokens<0:
            raise ValueError('Use zero for automatic generation or a positive manual limit')
        if images:
            if self.vision is None:
                raise ValueError('This bundle has no converted vision encoder; remove the image or select an image bundle')
            tokens,overrides,image_report=self.vision.prompt(self,prompt,images,chat,should_stop)
        else:
            tokens=self.encode(prompt,chat)
            overrides,image_report={},[]
        count=len(tokens)
        if count>=self.manifest['context']:
            raise ValueError('The prompt does not fit the compiled context')
        sampling=Generation(options,tokens)
        limit,limit_reason=sampling.limit(max_new_tokens,self.manifest['context']-count)
        cache=self.new_cache()
        self.device.sync()
        started=time.perf_counter()
        output,times,first,stop=[],[],None,'context_limit'
        stop_ids={self.tokenizer.eos_id(),self.special.get('<end_of_turn>')}
        try:
            for position in range(self.manifest['context']-1):
                if should_stop and should_stop():
                    stop='user_stop';break
                logits,cache=self.step(tokens[position],position,cache,overrides.get(position))
                if position+1<count:
                    continue
                logits=logits.numpy()
                self.device.sync()
                elapsed=time.perf_counter()-started
                if first is None:
                    first=elapsed
                if not np.isfinite(logits).all():
                    raise RuntimeError('Non-finite native CUDA output')
                token=sampling.choose(logits)
                times.append(elapsed)
                if token in stop_ids:
                    stop='eos';break
                output.append(token);tokens.append(token)
                if on_text:
                    on_text(self.tokenizer.decode(output))
                if len(output)>=limit:
                    stop=limit_reason;break
        except KeyboardInterrupt:
            stop='user_stop'
        self.device.sync()
        return {'images':image_report,'generation_options':sampling.report(),'backend':self.manifest['backend'],'gpu':self.device.device_name,
                'compiled_arch':self.manifest['cuda_arch'],'load_seconds':self.load_seconds,
                'prompt_tokens':count,'generated_tokens':len(output),'token_ids':output,
                'text':self.tokenizer.decode(output),'stop':stop,'first_token_seconds':first,
                'generation_timestamps':times,'total_seconds':time.perf_counter()-started,
                'parameter_bytes':self.parameter_bytes,'memory_before':self.memory_before,
                'memory_loaded':self.memory_loaded,'memory_after':cuda_memory(),
                'workspace_bytes':self.workspace.bytes,'workspace_errors':list(self.workspace.errors)}


class GpuSession:
    """One persistent CUDA worker with deterministic thread-local cleanup.

    Calls run on one thread so the CUDA context and session-owned workspace
    have a consistent lifetime. Closing explicitly releases device storage,
    restores the bundle's context slots and joins the worker. Errors propagate.
    """
    def __init__(self,folder,device_id=0):
        self._queue=queue.Queue()
        self._state=None
        self._closed=False
        self._worker=threading.Thread(target=self._serve,name='GemmaCUDA',daemon=True)
        self._worker.start()
        try:
            self._state=self._call(_CudaSession,folder,device_id)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _serve(self):
        while True:
            item=self._queue.get()
            if item is None:
                return
            future,func,args,kwargs=item
            try:
                future.set_result(func(*args,**kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                                                                                  
                del item,future,func,args,kwargs

    def _call(self,func,*args,**kwargs):
        if self._closed:
            raise RuntimeError('This CUDA session is closed')
        if threading.current_thread() is self._worker:
            return func(*args,**kwargs)
        future=Future()
        self._queue.put((future,func,args,kwargs))
        return future.result()

    def __getattr__(self,name):
        state=self.__dict__.get('_state')
        if state is None:
            raise AttributeError(name)
        return getattr(state,name)

    def generate(self,*args,**kwargs):
        return self._call(self._state.generate,*args,**kwargs)

    def new_cache(self):
        return self._call(self._state.new_cache)

    def step(self,*args,**kwargs):
        return self._call(self._state.step,*args,**kwargs)

    def close(self):
        if self._closed:
            return
        if threading.current_thread() is self._worker:
            raise RuntimeError('Close the session after its generation callback returns')
        try:
            if self._state is not None:
                self._call(self._state.close)
        finally:
            self._closed=True
            self._queue.put(None)
            self._worker.join()
            atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self,*exception):
        self.close()
