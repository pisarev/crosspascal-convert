# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Text inference in a native TVM Gemma 3n bundle; no PyTorch or Ollama runtime."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path[:0] = [p for p in os.environ.get("CROSSPASCAL_GEMMA_RUNTIME_PACKAGES", "").split(os.pathsep) if p]
import numpy as np
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pascalvision_convert.generation import Generation
import sentencepiece
import tvm
import tvm_ffi
from tvm import relax


class Session:
    def __init__(self, folder):
        started = time.perf_counter()
        self.folder = Path(folder).resolve(strict=True)
        self.manifest = self.read_json("model.json")
        if (self.manifest.get("schema") != "crosspascal.gemma3n.text.v1" or
                self.manifest.get("architecture") != "gemma3n" or
                self.manifest.get("backend") != "TVM LLVM CPU FP32"):
            raise ValueError("Expected a native Gemma 3n text bundle")
        for filename, expected in self.manifest.get("artifact_sha256", {}).items():
            with self.member(filename).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Bundle artifact changed: " + filename)
        self.record = self.read_json("parameters.json")
        self.metadata = self.read_json("tokenizer-gguf.json")
        self.vocabulary = self.metadata["tokenizer.ggml.tokens"]
        self.special = {word: index for index, word in enumerate(self.vocabulary)
                        if word in ("<start_of_turn>", "<end_of_turn>", "<eos>")}
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(self.member("tokenizer.model")))
        self.vm = relax.VirtualMachine(tvm.runtime.load_module(str(self.member("gemma3n-cpu.dll"))), tvm.cpu())
        self.arrays = {}
        for name, spec in self.record["weights"].items():
            path = self.member(spec["file"])
            dtype = np.dtype(spec.get("dtype", "float32"))
            if path.stat().st_size != math.prod(spec["shape"]) * dtype.itemsize:
                raise ValueError("Wrong parameter size: " + spec["file"])
            self.arrays[name] = np.memmap(path, mode="r", dtype=dtype, shape=tuple(spec["shape"]))
        self.parameters = []
        for name in self.manifest["input_names"][len(self.manifest["runtime_inputs"]):]:
            spec = self.manifest["parameters"][name]
            array = self.arrays[spec["tensor"]]
            if spec["slice"] is not None:
                array = array[spec["slice"]]
                                                                                   
            self.parameters.append(tvm_ffi.from_dlpack(array.__dlpack__(max_version=(1, 0))))
        self.prefill_vm, self.prefill_parameters = None, []
        if self.manifest.get('prefill'):
            graph = self.manifest['prefill']
            if type(graph['chunk']) is not int or not 1 < graph['chunk'] <= self.manifest['context']:
                raise ValueError('Invalid CPU prefill chunk')
            required = {graph['onnx'],graph['library']}
            if not required <= self.manifest.get('artifact_sha256',{}).keys():
                raise ValueError('CPU prefill artifact hashes are incomplete')
            for name in graph['input_names'][len(graph['runtime_inputs']):]:
                binding = graph['parameters'][name]
                array = self.arrays[binding['tensor']]
                if binding['slice'] is not None:
                    array = array[binding['slice']]
                self.prefill_parameters.append(tvm_ffi.from_dlpack(array.__dlpack__(max_version=(1,0))))
            self.prefill_vm = relax.VirtualMachine(tvm.runtime.load_module(str(self.member(graph['library']))),tvm.cpu())
        self.host_tables = []
        if self.manifest.get('host_embeddings'):
            for name in ('embed_tokens.weight', 'embed_tokens_per_layer.weight'):
                self.host_tables.append(self.arrays[self.record['parameters'][name]['tensor']])
        self.vision = None
        if self.manifest.get('vision_manifest'):
            from .gemma3n_vision_runtime import Vision
            if self.manifest['vision_manifest'] != 'vision.json' or not self.host_tables:
                raise ValueError('CPU images require a converted host-embedding graph and vision.json')
            self.vision = Vision(self.folder, self.host_tables[0].shape[1])
            if self.vision.manifest['backend'] != 'TVM CPU FP32':
                raise ValueError('A CPU text bundle must use a CPU vision encoder')
        self.load_seconds = time.perf_counter() - started

    def close(self):
        self.parameters.clear()
        self.prefill_parameters.clear()
        self.prefill_vm = None
        self.vm = None
        self.vision = None
        self.host_tables.clear()
        self.arrays.clear()


    def member(self, name):
        path = (self.folder / name).resolve(strict=True)
        if path.parent != self.folder or not path.is_file():
            raise ValueError("Invalid bundle member")
        return path

    def read_json(self, name):
        return json.loads(self.member(name).read_text(encoding="utf-8"))

    def encode(self, prompt, chat=True):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a non-empty text prompt")
        encode = self.tokenizer.encode
        bos = self.tokenizer.bos_id()
        prefix = [bos] if bos >= 0 else []
        if chat:
            if not all(name in self.special for name in ("<start_of_turn>", "<end_of_turn>")):
                raise ValueError("This tokenizer has no Gemma chat markers; use plain completion explicitly")
            start, end = self.special["<start_of_turn>"], self.special["<end_of_turn>"]
            return prefix + [start] + encode("user\n" + prompt) + [end] + encode("\n") + [start] + encode("model\n")
        return prefix + encode(prompt)

    def step(self, token, position, caches, embedding=None):
        if self.vm is None:
            raise RuntimeError('This CPU session is closed')
        if not 0 <= position < self.manifest['context']:
            raise ValueError('Token position exceeds the compiled context')
        if self.host_tables:
            if embedding is None:
                if any(not 0 <= token < len(table) for table in self.host_tables):
                    raise ValueError('Token is outside the source embedding tables')
                rows = [np.array(table[token], copy=True) for table in self.host_tables]
            else:
                rows = [np.asarray(embedding, dtype='float32') / math.sqrt(self.host_tables[0].shape[1]),
                        np.array(self.host_tables[1][token if 0 <= token < len(self.host_tables[1]) else 0], copy=True)]
            inputs = [tvm.runtime.tensor(row.reshape(1, 1, -1)) for row in rows]
        else:
            if embedding is not None:
                raise ValueError('This text graph does not accept image embeddings')
            inputs = [tvm.runtime.tensor(np.array([[token]], dtype='int64'))]
        result = self.vm['main'](*inputs, tvm.runtime.tensor(np.array([position], dtype='int64')),
                                 *caches, *self.parameters)
        return result[0], list(result[1:])

    def prefill(self, tokens, position, caches, overrides):
        rows = [[],[]]
        for index,token in enumerate(tokens,position):
            embedding = overrides.get(index)
            if embedding is None:
                if any(not 0 <= token < len(table) for table in self.host_tables):
                    raise ValueError('Token is outside the source embedding tables')
                for output,table in zip(rows,self.host_tables):
                    output.append(table[token])
            else:
                rows[0].append(np.asarray(embedding,dtype='float32')/math.sqrt(self.host_tables[0].shape[1]))
                rows[1].append(self.host_tables[1][token if 0 <= token < len(self.host_tables[1]) else 0])
        inputs = [tvm.runtime.tensor(np.array(row,dtype='float32',copy=True).reshape(1,len(tokens),-1)) for row in rows]
        result = self.prefill_vm['main'](*inputs,tvm.runtime.tensor(np.arange(position,position+len(tokens),dtype='int64')),
                                         *caches,*self.prefill_parameters)
        return result[0],list(result[1:])

    def generate(self, prompt, max_new_tokens=0, on_text=None, should_stop=None, chat=True, options=None, images=None):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("Max new tokens must be zero (automatic) or a positive integer")
        started = time.perf_counter()
        if images:
            if self.vision is None:
                raise ValueError('This bundle has no converted CPU vision encoder')
            tokens, overrides, image_report = self.vision.prompt(self, prompt, images, chat, should_stop)
        else:
            tokens = self.encode(prompt, chat)
            overrides, image_report = {}, []
        encoding_seconds = time.perf_counter() - started
        prompt_count = len(tokens)
        context = self.manifest["context"]
        if prompt_count >= context:
            raise ValueError(f"Prompt needs {prompt_count} tokens; rebuild with context above {context}")
        sampling = Generation(options, tokens)
        limit, limit_reason = sampling.limit(max_new_tokens, context - prompt_count)
        caches = [tvm.runtime.tensor(np.zeros(self.manifest["cache_shape"], dtype="float32"))
                  for _ in range(self.manifest["cache_count"])]
        stop_ids = {self.tokenizer.eos_id(), self.special.get("<end_of_turn>")}
        output, first, stop = [], None, "context_limit"
        try:
                                                                                
            position = 0
            chunk = self.manifest.get('prefill',{}).get('chunk',1)
            while position < context - 1:
                if should_stop and should_stop():
                    stop = "user_stop"
                    break
                count = chunk if self.prefill_vm is not None and prompt_count-position >= chunk else 1
                if count > 1:
                    logits,caches = self.prefill(tokens[position:position+count],position,caches,overrides)
                else:
                    logits, caches = self.step(tokens[position], position, caches, overrides.get(position))
                position += count
                if position < prompt_count:
                    continue
                logits = logits.numpy()
                if not np.isfinite(logits).all():
                    raise RuntimeError("The native graph produced non-finite logits")
                if first is None:
                    first = time.perf_counter() - started
                token = sampling.choose(logits)
                if token in stop_ids:
                    stop = "eos"
                    break
                output.append(token)
                tokens.append(token)
                if on_text:
                    on_text(self.tokenizer.decode(output))
                if len(output) >= limit:
                    stop = limit_reason
                    break
        except KeyboardInterrupt:
            stop = "user_stop"
        return {"images": image_report, "encoding_seconds": encoding_seconds, "load_seconds": self.load_seconds,
                "generation_options": sampling.report(), "backend": "TVM LLVM CPU FP32", "source_sha256": self.manifest["source_sha256"],
                "prompt_tokens": prompt_count, "generated_tokens": len(output), "token_ids": output,
                "text": self.tokenizer.decode(output), "stop": stop, "first_token_seconds": first,
                "total_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0: EOS or context capacity; Ctrl+C stops")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--plain", action="store_true", help="Use plain completion for a base model instead of Gemma chat markers")
    args = parser.parse_args()
    previous = ""
    def display(text):
        nonlocal previous
        print(text[len(previous):] if text.startswith(previous) else "\n" + text, end="", flush=True)
        previous = text
    manifest = json.loads((args.bundle / "model.json").read_text(encoding="utf-8"))
    if manifest.get("backend") == "TVM CUDA W8A32":
                                                                                   
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pascalvision_convert.gemma3n_gpu_runtime import GpuSession
        session = GpuSession(args.bundle)
    else:
        session = Session(args.bundle)
    report = session.generate(args.prompt, args.max_new_tokens, display, chat=not args.plain)
    print("\n" + json.dumps(report, ensure_ascii=True), flush=True)
    if args.result:
        args.result.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
