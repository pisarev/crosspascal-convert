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
        self.folder = Path(folder).resolve(strict=True)
        self.manifest = self.read_json("model.json")
        if (self.manifest.get("schema") != "crosspascal.gemma3n.text.v1" or
                self.manifest.get("architecture") != "gemma3n"):
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

    def close(self):
        self.parameters.clear()
        self.vm = None
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

    def generate(self, prompt, max_new_tokens=0, on_text=None, should_stop=None, chat=True, options=None):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("Max new tokens must be zero (automatic) or a positive integer")
        started = time.perf_counter()
        tokens = self.encode(prompt, chat)
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
                                                                                
            for position in range(context - 1):
                if should_stop and should_stop():
                    stop = "user_stop"
                    break
                result = self.vm["main"](tvm.runtime.tensor(np.array([[tokens[position]]], dtype="int64")),
                                         tvm.runtime.tensor(np.array([position], dtype="int64")),
                                         *caches, *self.parameters)
                caches = list(result[1:])
                if position + 1 < prompt_count:
                    continue
                logits = result[0].numpy()
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
        return {"generation_options": sampling.report(), "backend": "TVM LLVM CPU FP32", "source_sha256": self.manifest["source_sha256"],
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
