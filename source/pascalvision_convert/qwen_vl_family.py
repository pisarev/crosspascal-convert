# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Configuration-derived Qwen2.5-VL image/text profile (no checkpoint allowlist)."""
import json
import math
from pathlib import Path


def profile(source, context=2048, image_size=224):
    source = Path(source).resolve(strict=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Expected the Qwen2.5-VL architecture, not an ordinary Qwen text model")
    text = config.get("text_config", config)
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("The source has no vision encoder configuration")
    required = ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size")
    for key in required:
        if type(text.get(key)) is not int or text[key] <= 0:
            raise ValueError("Invalid text topology: " + key)
    if text["hidden_size"] % text["num_attention_heads"]:
        raise ValueError("Attention heads do not divide the hidden size")
    if text["num_attention_heads"] % text["num_key_value_heads"]:
        raise ValueError("Query heads do not divide the key/value groups")
    if vision.get("out_hidden_size") != text["hidden_size"]:
        raise ValueError("Vision merger output does not match the text embedding size")
    head_dim = text["hidden_size"] // text["num_attention_heads"]
    rope = text.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) not in ("default", "mrope"):
        raise ValueError("This conversion profile supports the source default multimodal RoPE")
    sections = rope.get("mrope_section")
    if (not isinstance(sections, list) or len(sections) != 3 or
            any(type(v) is not int or v <= 0 for v in sections) or sum(sections) * 2 != head_dim):
        raise ValueError("Invalid temporal/height/width rotary sections")
    patch, merge = vision.get("patch_size"), vision.get("spatial_merge_size")
    if not all(type(v) is int and v > 0 for v in
               (patch, merge, vision.get("temporal_patch_size"), vision.get("depth"),
                vision.get("hidden_size"), vision.get("num_heads"))):
        raise ValueError("Invalid vision topology")
    if vision['hidden_size'] % vision['num_heads'] or (vision['hidden_size'] // vision['num_heads']) % 4:
        raise ValueError('Invalid vision attention head dimensions')
    if vision.get('in_channels', vision.get('in_chans', 3)) != 3:
        raise ValueError('This image profile requires a three-channel RGB encoder')
    if type(image_size) is not int or not patch * merge <= image_size <= 448 or image_size % (patch * merge):
        raise ValueError("Image size must be a multiple of patch size times merge size, at most 448 pixels")
    soft_tokens = (image_size // (patch * merge)) ** 2
    if type(context) is not int or not soft_tokens + 32 < context <= text["max_position_embeddings"]:
        raise ValueError("Context must fit the image and text and stay within the source position range")
    for key in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        if type(config.get(key)) is not int or not 0 <= config[key] < text["vocab_size"]:
            raise ValueError("Invalid multimodal token: " + key)
    if not (source / "tokenizer.json").is_file() or not (source / "preprocessor_config.json").is_file():
        raise ValueError("Tokenizer and image preprocessing files are required")
    preprocessing = json.loads((source/'preprocessor_config.json').read_text(encoding='utf-8'))
    for key in ('image_mean', 'image_std'):
        values = preprocessing.get(key)
        if (not isinstance(values, list) or len(values) != 3 or
                any(type(v) not in (int, float) or not math.isfinite(v) or
                    (key == 'image_std' and v <= 0) for v in values)):
            raise ValueError('Invalid image normalization: ' + key)
    for key in ('do_rescale', 'do_normalize', 'do_convert_rgb'):
        if preprocessing.get(key, True) is not True:
            raise ValueError('This image profile requires ' + key)
    for key, expected in (('patch_size',patch),('temporal_patch_size',vision['temporal_patch_size']),('merge_size',merge)):
        if preprocessing.get(key, expected) != expected:
            raise ValueError('Image processor disagrees with the vision architecture: ' + key)
    factor = preprocessing.get('rescale_factor',1/255)
    if type(factor) not in (int,float) or not math.isfinite(factor) or factor <= 0:
        raise ValueError('Invalid image rescaling factor')
    return {"architecture": "qwen2_5_vl", "config": config, "text": text, "vision": vision,
            "context": context, "image_size": image_size, "soft_tokens": soft_tokens,
            "head_dim": head_dim, "cache_shape": [1, text["num_key_value_heads"], context, head_dim],
            "modalities": ["text", "image"], "image_policy": "RGB aspect-preserving letterbox",
            "backend": "TVM LLVM CPU FP32"}
