# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Inspect the architecture and tensor inventory without loading weights."""
import json
import os
from pathlib import Path
import sys
sys.path[:0] = [p for p in os.environ.get("CROSSPASCAL_GEMMA_PACKAGES", "").split(os.pathsep) if p]
import gguf


def inspect(path):
    reader = gguf.GGUFReader(path)
    fields = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    if fields.get("general.architecture") != "gemma3n":
        raise ValueError("This GGUF does not declare the Gemma 3n architecture")
    tensors = {t.name: t for t in reader.tensors}
    def value(key):
        return fields["gemma3n." + key]
    layers = value("block_count")
    if type(layers) is not int or not 1 <= layers <= 128:
        raise ValueError("Invalid Gemma 3n layer count")
    if len(value("attention.sliding_window_pattern")) != layers:
        raise ValueError("Invalid attention layout")
    if len(value("activation_sparsity_scale")) != layers:
        raise ValueError("Invalid activation sparsity layout")
    return {
        "model_type": "gemma3n", "architectures": ["Gemma3nForCausalLM"],
        "source_file": path.name, "num_hidden_layers": layers,
        "hidden_size": value("embedding_length"),
        "vocab_size": int(tensors["token_embd.weight"].shape[-1]),
        "intermediate_size": max(int(tensors[f"blk.{i}.ffn_gate.weight"].shape[1]) for i in range(layers)),
        "num_attention_heads": value("attention.head_count"),
        "num_key_value_heads": value("attention.head_count_kv"),
        "head_dim": value("head_dim"), "sliding_window": value("attention.sliding_window"),
        "max_position_embeddings": value("context_length"),
        "packed_bytes": sum(int(t.n_elements) * 4 for t in reader.tensors),
    }


if __name__ == "__main__":
    print(json.dumps(inspect(Path(sys.argv[1]))))
