# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Native TVM Qwen2.5-VL CPU sessions; PyTorch is not a runtime dependency."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import tvm
import tvm_ffi
from tvm import relax
from tokenizers import Tokenizer
from .generation import Generation
from .cpu_cancellation import cancellable_cpu_call


class Session:
    def __init__(self, folder):
        started = time.perf_counter()
        self.folder = Path(folder).resolve(strict=True)
        self.manifest = self.read_json("model.json")
        m = self.manifest
        if m.get("schema") != "crosspascal.qwen2_5_vl.v1" or m.get("backend") != "TVM LLVM CPU FP32":
            raise ValueError("Expected a converted Qwen2.5-VL CPU bundle")
        required = {"parameters.json", "tokenizer.json", "tokenizer_config.json", "config.json", "preprocessor_config.json"}
        for graph in m["graphs"].values():
            required.update((graph["onnx"], graph["library"]))
        if not required <= m.get("artifact_sha256", {}).keys():
            raise ValueError("Bundle artifact hashes are incomplete")
        for name, expected in m["artifact_sha256"].items():
            self.check_hash(name, expected)
        self.record = self.read_json("parameters.json")
        self.config = self.read_json("config.json")
        self.preprocessor = self.read_json("preprocessor_config.json")
        self.tokenizer = Tokenizer.from_file(str(self.member("tokenizer.json")))
        self.arrays, self.parameters, self.vms = {}, {}, {}
        self.checked = set()
        self.vision = self
        self.last_image_key, self.last_features = None, None
        for kind, graph in m["graphs"].items():
            values = []
            for name in graph["input_names"][len(graph["runtime_inputs"]):]:
                array = self.array(graph["parameters"][name])
                values.append(tvm_ffi.from_dlpack(array.__dlpack__(max_version=(1, 0))))
            self.parameters[kind] = values
            self.vms[kind] = relax.VirtualMachine(tvm.runtime.load_module(str(self.member(graph["library"]))), tvm.cpu())
        self.embedding = self.array(self.record["weights"]["model.language_model.embed_tokens.weight"])
        text = self.config.get("text_config", self.config)
        eos = text.get("eos_token_id", self.config.get("eos_token_id"))
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        self.stop_ids.add(self.tokenizer.token_to_id("<|im_end|>"))
        self.stop_ids.discard(None)
        self.load_seconds = time.perf_counter() - started

    def member(self, name):
        path = (self.folder / name).resolve(strict=True)
        if path.parent != self.folder or not path.is_file():
            raise ValueError("Invalid bundle member: " + str(name))
        return path

    def read_json(self, name):
        return json.loads(self.member(name).read_text(encoding="utf-8"))

    def check_hash(self, name, expected):
        with self.member(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Bundle artifact changed: " + name)

    def array(self, row):
        filename = row["file"]
        dtype = np.dtype(row["dtype"])
        path = self.member(filename)
        if path.stat().st_size != math.prod(row["shape"]) * dtype.itemsize:
            raise ValueError("Wrong parameter size: " + filename)
        if filename not in self.checked:
            self.check_hash(filename, row["sha256"])
            self.checked.add(filename)
        key = (filename, dtype.str, tuple(row["shape"]))
        if key not in self.arrays:
            self.arrays[key] = np.memmap(path, mode="r", dtype=dtype, shape=tuple(row["shape"]))
        return self.arrays[key]

    def close(self):
        self.parameters.clear()
        self.vms.clear()
        self.embedding = None
        self.arrays.clear()
        self.last_features = None

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def image_pixels(self, path):
        from PIL import Image, ImageOps
        with Image.open(path) as original:
            if original.width * original.height > 64_000_000:
                raise ValueError("Image exceeds 64 megapixels; resize it before attaching")
            image = ImageOps.exif_transpose(original).convert("RGB")
            size = self.manifest["image_size"]
                                                                                           
            image = ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
            pixels = np.asarray(image, dtype="float32")
        p = self.preprocessor
        pixels *= np.float32(p.get("rescale_factor", 1 / 255))
        pixels = (pixels - np.asarray(p["image_mean"], dtype="float32")) / np.asarray(p["image_std"], dtype="float32")
        vision = self.config["vision_config"]
        patch, merge, temporal = vision["patch_size"], vision["spatial_merge_size"], vision["temporal_patch_size"]
        grid = size // patch
        pixels = np.repeat(pixels.transpose(2, 0, 1)[None], temporal, axis=0)
        patches = pixels.reshape(1, temporal, 3, grid // merge, merge, patch, grid // merge, merge, patch)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return np.ascontiguousarray(patches.reshape(grid * grid, 3 * temporal * patch * patch))

    def image_features(self, path, should_stop=None):
        path = Path(path).resolve(strict=True)
        with path.open("rb") as stream:
            key = hashlib.file_digest(stream, "sha256").hexdigest()
        if key == self.last_image_key:
            return self.last_features, {"sha256": key, "reused": True, "seconds": 0}
        started = time.perf_counter()
        pixels = self.image_pixels(path)
        if should_stop and should_stop():
            raise InterruptedError("Image encoding cancelled")
        with cancellable_cpu_call(self.vms["vision"], should_stop):
            features = self.vms["vision"]["main"](tvm.runtime.tensor(pixels), *self.parameters["vision"]).numpy()
        if features.shape != (self.manifest["soft_tokens"], self.embedding.shape[1]) or not np.isfinite(features).all():
            raise RuntimeError("The native vision encoder produced invalid embeddings")
        if should_stop and should_stop():
            raise InterruptedError("Image encoding cancelled")
        self.last_image_key, self.last_features = key, features
        return features, {"sha256": key, "reused": False, "seconds": time.perf_counter() - started, "backend": self.manifest["backend"]}

    def prompt(self, text, images, chat, should_stop):
        tokens, positions, overrides, reports = [], [], {}, []
        next_position = 0

        def append_text(ids):
            nonlocal next_position
            for token in ids:
                tokens.append(token)
                positions.append([next_position] * 3)
                next_position += 1

        if chat:
            for marker in ("<|im_start|>", "<|im_end|>"):
                if self.tokenizer.token_to_id(marker) is None:
                    raise ValueError("This tokenizer has no Qwen chat markers; select plain completion")
            append_text(self.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"))
        needed = len(tokens) + len(images) * (self.manifest["soft_tokens"] + 2) + len(self.encode(text)) + 12
        if needed >= self.manifest["context"]:
            raise ValueError("Text and images do not fit the converted context; choose fewer images or a larger context")
        vision = self.config["vision_config"]
        grid = self.manifest["image_size"] // (vision["patch_size"] * vision["spatial_merge_size"])
        for path in images:
            features, report = self.image_features(path, should_stop)
            reports.append(report)
            append_text([self.config["vision_start_token_id"]])
            for i, feature in enumerate(features):
                overrides[len(tokens)] = feature
                tokens.append(self.config["image_token_id"])
                positions.append([next_position, next_position + i // grid, next_position + i % grid])
            next_position += grid
            append_text([self.config["vision_end_token_id"]])
        append_text(self.encode(text))
        if chat:
            append_text(self.encode("<|im_end|>\n<|im_start|>assistant\n"))
        return tokens, positions, overrides, reports, next_position

    def generate(self, prompt, max_new_tokens=0, on_text=None, should_stop=None, chat=True, options=None, images=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a non-empty prompt")
        images = images or []
        if not isinstance(images, list) or any(not isinstance(p, str) for p in images):
            raise ValueError("Images must be a list of file paths")
        if not self.vms:
            raise RuntimeError("The model has been unloaded")
        started = time.perf_counter()
        tokens, positions, overrides, image_reports, next_position = self.prompt(prompt, images, chat, should_stop)
        encoding_seconds = time.perf_counter() - started
        prompt_count = len(tokens)
        sampling = Generation(options, tokens)
        limit, reason = sampling.limit(max_new_tokens, self.manifest["context"] - prompt_count)
        caches = [tvm.runtime.tensor(np.zeros(self.manifest["cache_shape"], dtype="float32"))
                  for _ in range(self.manifest["cache_count"])]
        output, first, stop = [], None, "context_limit"
        position = 0
        chunk = self.manifest.get('prefill_chunk', 1)
        if type(chunk) is not int or not 1 <= chunk <= self.manifest['context']:
            raise ValueError('Invalid prefill chunk size')
        while position < self.manifest["context"] - 1:
            if should_stop and should_stop():
                stop = "user_stop"
                break
            count = chunk if 'prefill' in self.vms and chunk > 1 and prompt_count-position >= chunk else 1
            kind = 'prefill' if count > 1 else 'text'
            rows = []
            for index in range(position, position+count):
                row = overrides.get(index)
                if row is None:
                    token = tokens[index]
                    if not 0 <= token < len(self.embedding):
                        raise ValueError("Token is outside the embedding table")
                    row = self.embedding[token]
                rows.append(row)
            result = self.vms[kind]["main"](
                tvm.runtime.tensor(np.array(rows, dtype="float32", copy=True).reshape(1, count, -1)),
                tvm.runtime.tensor(np.array(positions[position:position+count], dtype="int64").T.reshape(3, 1, count)),
                tvm.runtime.tensor(np.arange(position, position+count, dtype="int64")), *caches, *self.parameters[kind])
            logits, caches = result[0].numpy(), list(result[1:])
            position += count
            if not np.isfinite(logits).all():
                raise RuntimeError("The native text graph produced non-finite logits")
            if position < prompt_count:
                continue
            if first is None:
                first = time.perf_counter() - started
            token = sampling.choose(logits)
            if token in self.stop_ids:
                stop = "eos"
                break
            output.append(token)
            tokens.append(token)
            positions.append([next_position] * 3)
            next_position += 1
            if on_text:
                on_text(self.tokenizer.decode(output, skip_special_tokens=True))
            if len(output) >= limit:
                stop = reason
                break
        return {"text": self.tokenizer.decode(output, skip_special_tokens=True), "tokens": output,
                "prompt_tokens": prompt_count, "generated_tokens": len(output), "stop": stop,
                "first_token_seconds": first, "seconds": time.perf_counter() - started,
                "encoding_seconds": encoding_seconds, "load_seconds": self.load_seconds,
                "images": image_reports, "settings": sampling.report(), "backend": self.manifest["backend"]}
