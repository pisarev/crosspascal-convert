# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Export Qwen2.5-VL as separate native vision and cached token graphs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path[:0] = [p for p in os.environ.get("CROSSPASCAL_GEMMA_PACKAGES", "").split(os.pathsep) if p]
from pascalvision_convert.qwen_vl_family import profile


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_staged(source, folder, spec):
    source, folder = source.resolve(strict=True), folder.resolve(strict=True)
    record = json.loads((folder / 'parameters.json').read_bytes())
    pins = record['source_sha256']
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json'}
    if not required <= pins.keys() or not any(name.endswith('.safetensors') for name in pins):
        raise ValueError('Staged source hashes are incomplete')
    for name, expected in pins.items():
        original = (source / name).resolve(strict=True)
        if original.parent != source or digest(original) != expected:
            raise ValueError('Original checkpoint differs from the staged source: ' + name)
        if not name.endswith('.safetensors') and name != 'model.safetensors.index.json':
            copied = (folder / name).resolve(strict=True)
            if copied.parent != folder or digest(copied) != expected:
                raise ValueError('Staged metadata changed: ' + name)
    checked = set()
    for row in record['weights'].values():
        filename = (folder / row['file']).resolve(strict=True)
        if filename.parent != folder:
            raise ValueError('Staged parameter escapes the bundle')
        if filename not in checked:
            if digest(filename) != row['sha256']:
                raise ValueError('Staged parameter changed: ' + row['file'])
            checked.add(filename)
    if profile(folder, spec['context'], spec['image_size']) != spec:
        raise ValueError('Staged topology differs from the original checkpoint')


def stage(source, output, spec):
    import numpy as np
    import torch
    from safetensors import safe_open
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_bytes())["weight_map"]
        shards = sorted(set(index.values()))
    else:
        shards = ["model.safetensors"]
        index = None
    weights, source_hashes = {}, {}
    if index_path.is_file():
        source_hashes[index_path.name] = digest(index_path)
    for shard in shards:
        path = (source / shard).resolve(strict=True)
        if path.parent != source:
            raise ValueError("Weight shard escapes the source directory")
        source_hashes[shard] = digest(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if index is not None and index.get(name) != shard:
                    raise ValueError("Shard index disagrees with tensor " + name)
                normalized = name
                if name.startswith("visual."):
                    normalized = "model." + name
                elif name.startswith("model.") and not name.startswith(("model.visual.", "model.language_model.")):
                    normalized = "model.language_model." + name[6:]
                if normalized in weights:
                    raise ValueError("Duplicate source tensor " + normalized)
                tensor = handle.get_tensor(name)
                if not tensor.is_floating_point():
                    raise ValueError("Expected floating-point source weights: " + name)
                array = tensor.to(dtype=torch.float32).contiguous().numpy()
                if not np.isfinite(array).all():
                    raise ValueError("Non-finite source tensor: " + name)
                filename = "weight-%05d.bin" % len(weights)
                array.tofile(output / filename)
                weights[normalized] = {"file": filename, "shape": list(array.shape),
                                      "dtype": "float32", "sha256": digest(output / filename)}
                del array, tensor
        print("Staged " + shard, flush=True)
    if index is not None and sum(1 for _ in index) != len(weights):
        raise ValueError("The weight index is incomplete")
    if spec["text"].get("tie_word_embeddings", spec["config"].get("tie_word_embeddings", False)):
        weights.setdefault("lm_head.weight", weights["model.language_model.embed_tokens.weight"])
    record = {"weights": weights, "source_sha256": source_hashes}
    (output / "parameters.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                 "generation_config.json", "chat_template.json", "LICENSE"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
            source_hashes[name] = digest(source / name)
    (output / "parameters.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def load_model(folder):
    import numpy as np
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLRotaryEmbedding, Qwen2_5_VisionRotaryEmbedding)
    config = Qwen2_5_VLConfig.from_pretrained(folder, local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)
    record = json.loads((folder / "parameters.json").read_bytes())
    arrays = {name: np.memmap(folder / row["file"], mode="r", dtype=row["dtype"], shape=tuple(row["shape"]))
              for name, row in record["weights"].items()}
    state = {name: torch.from_numpy(array) for name, array in arrays.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.model.language_model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config.text_config, device="cpu")
    for layer in model.model.language_model.layers:
        layer.self_attn.rotary_emb = model.model.language_model.rotary_emb
    model.model.visual.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(
        config.vision_config.hidden_size // config.vision_config.num_heads // 2)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Unbound model parameters")
    missing = [name for name, b in model.named_buffers() if b.device.type != "cpu"]
    if missing:
        raise ValueError("Unbound model buffers: " + ", ".join(missing))
    return model.eval(), record, arrays


def export(folder, spec, prefill_only=False):
    import numpy as np
    import onnx
    import torch
    torch.set_num_threads(4)
    model, record, arrays = load_model(folder)

    class Cache:
        def __init__(self, previous, selector, indices=None):
            self.previous, self.selector, self.current = previous, selector, []
            self.indices = indices
        def get_seq_length(self):
            return 0
        def update(self, key, value, layer, options):
            if self.indices is not None:
                key = key.index_select(2, self.indices)
                value = value.index_select(2, self.indices)
            key = torch.where(self.selector, key, self.previous[2 * layer])
            value = torch.where(self.selector, value, self.previous[2 * layer + 1])
            self.current.extend((key, value))
            return key, value

    class TokenGraph(torch.nn.Module):
        def __init__(self, chunk=1):
            super().__init__()
            self.model = model
            self.chunk = chunk
            self.register_buffer("positions", torch.arange(spec["context"]), persistent=False)
        def forward(self, embedding, rotary_position, position, *previous):
            distance = position.reshape(1, 1, self.chunk, 1) - self.positions.reshape(1, 1, 1, -1)
            masks = {"full_attention": torch.where(distance >= 0, 0.0, torch.finfo(torch.float32).min)}
            if self.model.config.text_config.use_sliding_window:
                masks["sliding_attention"] = torch.where(
                    (distance >= 0) & (distance < self.model.config.text_config.sliding_window),
                    0.0, torch.finfo(torch.float32).min)
            if self.chunk == 1:
                cache = Cache(previous, (self.positions == position).reshape(1, 1, -1, 1))
            else:
                relative = self.positions - position[0]
                cache = Cache(previous, ((relative >= 0) & (relative < self.chunk)).reshape(1, 1, -1, 1),
                              relative.clamp(0, self.chunk-1))
            hidden = self.model.model.language_model(inputs_embeds=embedding, attention_mask=masks,
                position_ids=rotary_position, cache_position=position, past_key_values=cache,
                use_cache=False).last_hidden_state
            return (self.model.lm_head(hidden[:, -1, :]), *cache.current)

    class VisionGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            vision = model.model.visual
            grid = torch.tensor([[1, spec["image_size"] // vision.patch_size,
                                  spec["image_size"] // vision.patch_size]])
            index, windows = vision.get_window_index(grid)
            rotary = vision.rot_pos_emb(grid)
            rotary = rotary.reshape(-1, vision.spatial_merge_unit, rotary.shape[-1])[index].reshape_as(rotary)
            emb = torch.cat((rotary, rotary), dim=-1)
            self.register_buffer("index", index, persistent=False)
            self.register_buffer("reverse", torch.argsort(index), persistent=False)
            self.register_buffer("cosine", emb.cos(), persistent=False)
            self.register_buffer("sine", emb.sin(), persistent=False)
            self.register_buffer("windows", torch.unique_consecutive(torch.tensor(windows, dtype=torch.int32)), persistent=False)
            self.register_buffer("full", torch.tensor([0, int(grid.prod())], dtype=torch.int32), persistent=False)
        def forward(self, pixels):
            vision = self.model.model.visual
            hidden = vision.patch_embed(pixels)
            hidden = hidden.reshape(-1, vision.spatial_merge_unit, hidden.shape[-1])[self.index].reshape_as(hidden)
            for i, block in enumerate(vision.blocks):
                hidden = block(hidden, cu_seqlens=self.full if i in vision.fullatt_block_indexes else self.windows,
                               position_embeddings=(self.cosine, self.sine))
            return vision.merger(hidden)[self.reverse]

    def graph_export(graph, args, input_names, output_names, filename):
        graph.eval()
        with torch.inference_mode():
            torch.onnx.export(graph, args, str(folder / filename), export_params=False, dynamo=False,
                opset_version=18, do_constant_folding=False, input_names=input_names, output_names=output_names)
        exported = onnx.load(folder / filename)
        bindings = {}
        buffers = dict(graph.named_buffers())
        for item in list(exported.graph.input)[len(input_names):]:
            name = item.name
            original = name.removeprefix("model.")
            if original in record["weights"]:
                bindings[name] = record["weights"][original]
            elif name in buffers:
                array = buffers[name].detach().numpy()
                filename_buffer = filename.replace(".onnx", "") + "-buffer-%03d.bin" % len(bindings)
                array.tofile(folder / filename_buffer)
                bindings[name] = {"file": filename_buffer, "shape": list(array.shape),
                                  "dtype": str(array.dtype), "sha256": digest(folder / filename_buffer)}
            else:
                raise ValueError("Unbound graph input: " + name)
        print("Exported " + filename, flush=True)
        return {"onnx": filename, "input_names": [v.name for v in exported.graph.input],
                "runtime_inputs": input_names, "parameters": bindings}

    count = spec["text"]["num_hidden_layers"] * 2
    caches = tuple(torch.zeros(spec["cache_shape"]) for _ in range(count))
    text_inputs = ["embedding", "rotary_position", "position"] + ["cache_%d" % i for i in range(count)]
    if prefill_only:
        manifest = json.loads((folder/'model.json').read_bytes())
        if manifest['context'] != spec['context'] or manifest['cache_shape'] != spec['cache_shape']:
            raise ValueError('Prefill topology does not match the existing token graph')
        chunk = 16
        manifest['graphs']['prefill'] = graph_export(TokenGraph(chunk),
            (torch.zeros((1, chunk, spec['text']['hidden_size'])),
             torch.arange(chunk).reshape(1, 1, chunk).expand(3, 1, chunk), torch.arange(chunk), *caches),
            text_inputs, ['logits'] + ['updated_%d' % i for i in range(count)], 'prefill.onnx')
        manifest['prefill_chunk'] = chunk
        manifest['inference_acceptance'] = 'NOT_CHECKED'
        (folder/'model.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return
    text_graph = graph_export(TokenGraph(),
        (torch.zeros((1, 1, spec["text"]["hidden_size"])), torch.zeros((3, 1, 1), dtype=torch.int64),
         torch.zeros(1, dtype=torch.int64), *caches), text_inputs,
        ["logits"] + ["updated_%d" % i for i in range(count)], "text.onnx")
    vision = spec["vision"]
    pixels = torch.zeros(((spec["image_size"] // vision["patch_size"]) ** 2,
                          vision.get("in_channels", 3) * vision["temporal_patch_size"] * vision["patch_size"] ** 2))
    vision_graph = graph_export(VisionGraph(), (pixels,), ["pixels"], ["embeddings"], "vision.onnx")
    chunk = 16
    prefill_graph = graph_export(TokenGraph(chunk),
        (torch.zeros((1, chunk, spec['text']['hidden_size'])),
         torch.arange(chunk).reshape(1, 1, chunk).expand(3, 1, chunk), torch.arange(chunk), *caches),
        text_inputs, ['logits'] + ['updated_%d' % i for i in range(count)], 'prefill.onnx')
    manifest = {"schema": "crosspascal.qwen2_5_vl.v1", "architecture": "qwen2_5_vl",
        "backend": "TVM LLVM CPU FP32", "context": spec["context"], "context_window_size": spec["context"],
        "image_size": spec["image_size"], "soft_tokens": spec["soft_tokens"], "image_policy": spec["image_policy"],
        "cache_shape": spec["cache_shape"], "cache_count": count, "modalities": ["text", "image"],
        "graphs": {"text": text_graph, "vision": vision_graph, "prefill": prefill_graph},
        "prefill_chunk": chunk, "inference_acceptance": "NOT_CHECKED",
        "artifact_sha256": {}, "source_sha256": record["source_sha256"]}
    (folder / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Qwen2.5-VL native graph export complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    args = parser.parse_args()
    spec = profile(args.source, args.context, args.image_size)
    if args.prefill_only and not args.staged:
        raise ValueError('Prefill-only export requires an existing staged bundle')
    if not args.staged:
        args.output.mkdir(parents=True, exist_ok=False)
        stage(args.source.resolve(), args.output, spec)
    else:
        validate_staged(args.source, args.output, spec)
    export(args.output, spec, args.prefill_only)
