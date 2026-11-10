# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.

import argparse
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path[:0] = [p for p in os.environ.get("CROSSPASCAL_GEMMA_PACKAGES", "").split(os.pathsep) if p]
from pascalvision_convert.qwen_vl_family import profile
from pascalvision_convert.llm_helpers import weights


def inspect(source, context, image_size):
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    spec = profile(source, context, image_size)
    source = Path(source).resolve(strict=True)
    config = Qwen2_5_VLConfig.from_pretrained(source, local_files_only=True)
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)
    expected = {name: tuple(p.shape) for name, p in model.state_dict().items()}
    checkpoint = weights.open_source(source)
    mapped = {}
    for name in checkpoint.names:
        normalized = name
        if name.startswith("visual."):
            normalized = "model." + name
        elif name.startswith("model.") and not name.startswith(("model.visual.", "model.language_model.")):
            normalized = "model.language_model." + name[6:]
        if normalized in mapped:
            raise ValueError("Duplicate normalized tensor: " + normalized)
        mapped[normalized] = name
    if config.text_config.tie_word_embeddings and "lm_head.weight" not in mapped:
        expected.pop("lm_head.weight", None)
    if mapped.keys() != expected.keys():
        raise ValueError("Source tensor names differ from this architecture: " +
                         ", ".join(sorted(mapped.keys() ^ expected.keys())[:8]))
    for name, shape in expected.items():
        original = mapped[name]
        if checkpoint.shape_of(original) != shape or checkpoint.dtype_of(original) not in ("F32", "F16", "BF16"):
            raise ValueError("Source tensor shape or dtype mismatch: " + original)
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]
    if hasattr(checkpoint, "weight_map"):
        files += ["model.safetensors.index.json", *sorted(set(checkpoint.weight_map.values()))]
    else:
        files.append("model.safetensors")
    files += [name for name in ("generation_config.json", "chat_template.json", "LICENSE") if (source/name).is_file()]
    for name in files:
        if (source/name).resolve(strict=True).parent != source or (source/name).parent != source:
            raise ValueError("This source profile requires files directly inside the checkpoint folder")
    return {"architecture": "qwen2_5_vl", "layers": config.text_config.num_hidden_layers,
            "hidden_size": config.text_config.hidden_size, "files": files,
            "source_bytes": sum((source/name).stat().st_size for name in files),
            "expanded_bytes": sum(math.prod(shape)*4 for shape in expected.values()),
            "cache_bytes": 2*config.text_config.num_hidden_layers*math.prod(spec['cache_shape'])*4,
            "tensor_count": len(expected), "context": context, "image_size": image_size,
            "soft_tokens": spec["soft_tokens"], "backend": spec["backend"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source, args.context, args.image_size)))
