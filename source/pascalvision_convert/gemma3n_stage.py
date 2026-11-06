# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Prepare file-backed FP32 Gemma 3n parameters from an architecture-tagged GGUF."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent
sys.path[:0] = [p for p in os.environ.get("CROSSPASCAL_GEMMA_PACKAGES", "").split(os.pathsep) if p]
import gguf
import numpy as np
import torch
from transformers import Gemma3nTextConfig, Gemma3nTextModel


def source_config(reader):
    fields = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    if fields.get("general.architecture") != "gemma3n":
        raise ValueError("Expected the Gemma 3n architecture")
    tensors = {t.name: t for t in reader.tensors}
    def get(key):
        return fields["gemma3n." + key]
    layers = get("block_count")
    sparsity = [0 if not math.isfinite(s) else (1 + math.erf(s / math.sqrt(2))) / 2
                for s in get("activation_sparsity_scale")]
    config = Gemma3nTextConfig(
        vocab_size=int(tensors["token_embd.weight"].shape[-1]),
        vocab_size_per_layer_input=int(tensors["per_layer_token_embd.weight"].shape[-1]),
        hidden_size=get("embedding_length"), num_hidden_layers=layers,
        intermediate_size=[int(tensors[f"blk.{i}.ffn_gate.weight"].shape[1]) for i in range(layers)],
        hidden_size_per_layer_input=get("embedding_length_per_layer_input"),
        num_attention_heads=get("attention.head_count"), num_key_value_heads=get("attention.head_count_kv"),
        head_dim=get("head_dim"), rms_norm_eps=get("attention.layer_norm_rms_epsilon"),
        sliding_window=get("attention.sliding_window"), num_kv_shared_layers=get("attention.shared_kv_layers"),
        layer_types=["sliding_attention" if v else "full_attention"
                     for v in get("attention.sliding_window_pattern")],
        altup_num_inputs=get("altup.num_inputs"), altup_active_idx=get("altup.active_idx"),
        altup_correct_scale=get("altup.correct_scale"),
        laurel_rank=int(tensors["blk.0.laurel_l.weight"].shape[1]),
        max_position_embeddings=get("context_length"), rope_theta=get("rope.freq_base"),
        rope_local_base_freq=get("rope.freq_base_local"), activation_sparsity_pattern=sparsity,
        attn_implementation="eager",
    )
    return config, tensors


def stage(source, output):
    reader = gguf.GGUFReader(source)
    config, tensors = source_config(reader)
    with torch.device("meta"):
        model = Gemma3nTextModel(config)
    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA3N, config.num_hidden_layers)
    mapping = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        index = None
        if parts[0] in ("altup_projections", "altup_unembed_projections"):
            index = int(parts[1])
            translated = mapper.get_name("model." + parts[0] + ".weight", try_suffixes=(".weight",))
        else:
            translated = mapper.get_name("model." + name + (".weight" if name.endswith("_scale") else ""),
                                         try_suffixes=(".weight",))
        if translated not in tensors:
            raise ValueError(f"Missing source tensor for {name}: {translated}")
        shape = tuple(int(x) for x in reversed(tensors[translated].shape))
        if index is not None:
            shape = shape[1:]
        if shape != tuple(param.shape):
            raise ValueError(f"Shape mismatch for {name}")
        mapping[name] = {"tensor": translated, "slice": index, "shape": list(param.shape)}
    selected = sorted({item["tensor"] for item in mapping.values()})
    size = sum(int(tensors[name].n_elements) * 4 for name in selected)
    output.mkdir(exist_ok=False)
    with open(source, "rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    cached_root = os.environ.get("CROSSPASCAL_GEMMA_PARAMETER_CACHE")
    cached = None
    if cached_root:
        cached_root = Path(cached_root).resolve(strict=True)
        cached = json.loads((cached_root / "parameters.json").read_text(encoding="utf-8"))
        if (cached.get("schema") != "crosspascal.gemma3n.parameters.v1" or
                cached.get("source_sha256") != source_hash or cached.get("parameters") != mapping or
                cached.get("config") != json.loads(json.dumps(config.to_dict()))):
            raise ValueError("Parameter cache does not match this source and architecture")
        for name in selected:
            spec = cached["weights"][name]
            item = cached_root / spec["file"]
            shape = [int(v) for v in reversed(tensors[name].shape)]
            if (item.resolve(strict=True).parent != cached_root or spec["shape"] != shape or
                    item.stat().st_size != math.prod(shape) * 4):
                raise ValueError("Invalid cached parameter file")
            with item.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
                    raise ValueError("A cached parameter has changed")
        print("Verified reusable parameters against source, shapes and every file hash", flush=True)
    elif shutil.disk_usage(output).free < size + 4 * 1024**3:
        raise ValueError("Insufficient disk for the file-backed parameter bundle")
    records = {}
    for number, name in enumerate(selected):
        t = tensors[name]
        shape = tuple(int(v) for v in reversed(t.shape))
        path = output / f"weight-{number:04d}.bin"
        if cached is not None:
            spec = cached["weights"][name]
                                                                                         
            os.link(cached_root / spec["file"], path)
            records[name] = {**spec, "file": path.name}
            continue
        target = np.memmap(path, mode="w+", dtype="float32", shape=shape)
                                                                         
        source_rows = t.data.reshape((-1, t.data.shape[-1]))
        dest_rows = target.reshape((-1, shape[-1]))
        for start in range(0, len(dest_rows), 128):
            dest_rows[start:start + 128] = gguf.dequantize(source_rows[start:start + 128], t.tensor_type)
        target.flush()
        del dest_rows, target
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records[name] = {"file": path.name, "shape": shape, "sha256": digest}
        if number % 40 == 0:
            print(f"Prepared {number + 1} tensors", flush=True)
    tokenizer = {k: v.contents() for k, v in reader.fields.items() if k.startswith("tokenizer.")}
    (output / "tokenizer-gguf.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    record = {"schema": "crosspascal.gemma3n.parameters.v1", "config": config.to_dict(),
              "source_sha256": source_hash, "source_format": "GGUF",
              "parameters": mapping, "weights": records}
    (output / "parameters.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Parameter staging complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.source, args.output)
