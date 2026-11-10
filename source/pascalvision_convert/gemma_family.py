# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


GEMMA2_ARCH = "Gemma2ForCausalLM"
GEMMA2 = "gemma2"
TEXT_NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
              "post_feedforward_layernorm")


def json_file(path, max_bytes=64 * 1024 * 1024):
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"Excessive JSON input: {path.name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique)


def identify(raw):
    if type(raw) is not dict or type(raw.get("architectures")) is not list:
        raise ValueError("Expected a Hugging Face model config with architectures")
    arch = raw["architectures"]
    family = raw.get("model_type")
    if (type(family) is not str or not re.fullmatch(r"gemma(?:\d+[a-z]?)?(?:_[a-z0-9]+)*", family)
            or len(arch) != 1 or type(arch[0]) is not str
            or not re.fullmatch(r"Gemma[A-Za-z0-9]+", arch[0])):
        raise ValueError(f"Expected a Gemma architecture, got {arch!r}")
    return family


def require_backend(raw):
    family = identify(raw)
    if family != GEMMA2 or raw["architectures"] != [GEMMA2_ARCH]:
        raise ValueError(f"Recognized {family} ({raw['architectures'][0]}). "
                         "A native TVM backend is required for this architecture.")
    return family


def integer(raw, key, low, high):
    value = raw.get(key)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} through {high}")
    return value


def real(raw, key, low, high):
    value = raw.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be finite and in the supported range")
    return value


def gemma2_shapes(raw):
    hidden = integer(raw, "hidden_size", 16, 16384)
    inner = integer(raw, "intermediate_size", 16, 131072)
    layers = integer(raw, "num_hidden_layers", 1, 128)
    qheads = integer(raw, "num_attention_heads", 1, 128)
    kvheads = integer(raw, "num_key_value_heads", 1, 128)
    head = integer(raw, "head_dim", 16, 512)
    vocab = integer(raw, "vocab_size", 256000, 256000)
    window = integer(raw, "sliding_window", 64, 131072)
    context = integer(raw, "max_position_embeddings", 64, 131072)
    if qheads % kvheads or head % 8 or window > context:
        raise ValueError("Incompatible grouped attention or context")
    for key, lower, upper in (("rms_norm_eps", 1e-10, .01), ("rope_theta", 100, 1e9),
                              ("query_pre_attn_scalar", 1, 65536),
                              ("attn_logit_softcapping", .01, 10000),
                              ("final_logit_softcapping", .01, 10000)):
        real(raw, key, lower, upper)
    if (raw.get("tie_word_embeddings", True) is not True or
        raw.get("attention_bias", False) is not False or
        raw.get("rope_scaling") is not None or raw.get("quantization_config") is not None or
        raw.get("hidden_activation", raw.get("hidden_act")) != "gelu_pytorch_tanh" or
        raw.get("bos_token_id") != 2 or raw.get("eos_token_id") not in (1, [1, 107])):
        raise ValueError("Unsupported Gemma 2 text-model semantics")
    shapes = {"model.embed_tokens.weight": (vocab, hidden), "model.norm.weight": (hidden,)}
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        rows = {
            "self_attn.q_proj.weight": (qheads * head, hidden),
            "self_attn.k_proj.weight": (kvheads * head, hidden),
            "self_attn.v_proj.weight": (kvheads * head, hidden),
            "self_attn.o_proj.weight": (hidden, qheads * head),
            "mlp.gate_proj.weight": (inner, hidden),
            "mlp.up_proj.weight": (inner, hidden),
            "mlp.down_proj.weight": (hidden, inner),
            **{f"{norm}.weight": (hidden,) for norm in TEXT_NORMS},
        }
        shapes.update({prefix + key: dims for key, dims in rows.items()})
    return shapes


def validate_source(source, raw, weights):
    shapes = gemma2_shapes(raw)
    checkpoint = weights.open_source(source)
    if set(checkpoint.names) != shapes.keys():
        raise ValueError(f"Tensor names differ from the architecture: {sorted(set(checkpoint.names) ^ shapes.keys())[:8]}")
    for name, dims in shapes.items():
        if checkpoint.shape_of(name) != dims or checkpoint.dtype_of(name) not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Invalid shape or dtype for {name}")
    tokenizer = json_file(source / "tokenizer.json")
    model = tokenizer["model"]
    vocab = model["vocab"]
    normalizer = tokenizer.get("normalizer")
    if (model["type"] != "BPE" or model["byte_fallback"] is not True or
        len(vocab) != raw["vocab_size"] or
        {name: vocab.get(name) for name in ("<bos>", "<eos>", "<end_of_turn>")} !=
        {"<bos>": 2, "<eos>": 1, "<end_of_turn>": 107} or
        tokenizer.get("pre_tokenizer") != {"type": "Split", "pattern": {"String": " "},
                                            "behavior": "MergedWithPrevious", "invert": False} or
        normalizer != {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}):
        raise ValueError("Unsupported Gemma tokenizer semantics")
    files = ["config.json", "tokenizer.json"]
    if hasattr(checkpoint, "weight_map"):
        files += ["model.safetensors.index.json", *sorted(set(checkpoint.weight_map.values()))]
    else:
        files.append("model.safetensors")
    return sum(math.prod(shape) for shape in shapes.values()) * 4, files


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(args, source, raw, byte_count, files, weights):
    import tvm
    from tvm import relax
    import gemma2_model
    from gemma_cpu_schedule import schedule_gemma_cpu
    from pascalvision_convert.convert import _shared_library_builder

    output = args.output.resolve()
    if output == source or source in output.parents or output.exists() or not output.parent.is_dir():
        raise ValueError("Choose a new output directory outside the source with an existing parent")
    if shutil.disk_usage(output.parent).free < byte_count + 1024**3:
        raise ValueError("Insufficient free disk space for the FP32 bundle")
    if args.context > min(raw["sliding_window"], raw["max_position_embeddings"], 1024):
        raise ValueError("The experimental profile only supports a short context")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.cpu):
        raise ValueError("Invalid LLVM CPU target")
    before = {name: digest(source / name) for name in files}
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    print(f"Building in {stage}", flush=True)
    target = tvm.target.Target({"kind": "llvm", "mcpu": args.cpu, "num-cores": args.threads})
    chunk = min(args.context, 64)
    config = gemma2_model.Gemma2Config(raw, context_window_size=args.context, prefill_chunk_size=chunk)
    model = gemma2_model.Gemma2ForCausalLM(config)
    with target:
        graph, params = model.export_tvm(spec=model.get_default_spec())
        graph = relax.transform.FuseTransposeMatmul()(graph)
        graph = schedule_gemma_cpu(relax.get_pipeline("zero")(graph))
    order = [name for name, _ in params]
    shapes = {name: tuple(map(int, tensor.shape)) for name, tensor in params}
    library = "gemma2-cpu.dll"
    with tvm.transform.PassContext(opt_level=3):
        executable = relax.build(graph, target=target)
    executable.export_library(str(stage / library), fcompile=_shared_library_builder())
    del executable, graph, model
    written, count = weights.convert(source, stage, order, shapes, raw["num_hidden_layers"],
                                     plan_fn=weights.plan_gemma2)
    if written != byte_count or count != len(order):
        raise RuntimeError("Emitted parameter layout differs from the source-derived size")
    exporter = Path(__file__).with_name("export_tokenizer_gemma.py")
    subprocess.run([sys.executable, "-B", str(exporter), "--tokenizer",
                    str(source / "tokenizer.json"), "--out", str(stage)], check=True)
    if any(digest(source / name) != sha for name, sha in before.items()):
        raise RuntimeError("Source changed during conversion")
    params_sha = json_file(stage / "params.json")["sha256"]
    record = {
        "schema": "crosspascal.gemma-family.v1", "model_type": GEMMA2,
        "library": library, "target": "cpu", "dtype": "float32",
        "llvm_target": str(target), "experimental": True,
        "attention": "legacy-short-context-without-attention-score-softcap",
        "context_window_size": args.context, "prefill_chunk_size": chunk,
        "vocab_size": raw["vocab_size"], "eos_token_id": raw["eos_token_id"],
        "bos_token_id": raw["bos_token_id"], "param_order": order,
        "params_sha256": params_sha, "source_sha256": before,
        "library_sha256": digest(stage / library),
    }
    (stage / "model.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.rename(stage, output)
    print(json.dumps({"status": "BUILT_EXPERIMENTAL_GEMMA2", "output": str(output),
                      "bytes": written, "parameters": count}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--converter-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", type=int, choices=(64, 128, 256, 512, 1024), default=1024)
    parser.add_argument("--threads", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--cpu", default="generic")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    raw = json_file(source / "config.json", 1024 * 1024)
    try:
        family = require_backend(raw)
    except ValueError as exc:
        parser.error(str(exc))
    helpers = args.converter_source.resolve(strict=True)
    sys.path[:0] = [str(Path(__file__).parent), str(helpers),
                    str(helpers / "pascalvision_convert/llm_helpers")]
    import weights

    total, files = validate_source(source, raw, weights)
    if args.output is None:
        print(json.dumps({"status": "STRUCTURE_VALIDATED", "family": family, "fp32_bytes": total,
                          "layers": raw["num_hidden_layers"], "files": files}, indent=2))
    else:
        convert(args, source, raw, total, files, weights)


if __name__ == "__main__":
    main()
