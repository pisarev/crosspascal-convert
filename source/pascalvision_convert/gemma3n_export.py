# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Export a native text graph with external file-backed parameters."""
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
root = Path(__file__).resolve().parent
sys.path[:0] = [p for p in os.environ.get("CROSSPASCAL_GEMMA_PACKAGES", "").split(os.pathsep) if p]
import numpy as np
import torch
from transformers import Gemma3nTextConfig, Gemma3nTextModel
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextRotaryEmbedding


class SharedPrefix:
    """Full-prefix execution retaining the architecture's shared KV layers."""
    def __init__(self):
        self.shared_layers = {}
    def get_seq_length(self):
        return 0
    def update(self, key, value, layer, options):
        return key, value


def tokenizer(bundle):
    import sentencepiece as sp
    from sentencepiece import sentencepiece_model_pb2 as proto
    metadata = json.loads((bundle / "tokenizer-gguf.json").read_text(encoding="utf-8"))
    model = proto.ModelProto()
    model.trainer_spec.model_type = 2
    model.trainer_spec.byte_fallback = True
    model.trainer_spec.unk_id = metadata.get("tokenizer.ggml.unknown_token_id", 3)
    model.trainer_spec.bos_id = metadata.get("tokenizer.ggml.bos_token_id", 2)
    model.trainer_spec.eos_id = metadata.get("tokenizer.ggml.eos_token_id", 1)
    model.trainer_spec.pad_id = metadata.get("tokenizer.ggml.padding_token_id", 0)
    for name in ("unk", "bos", "eos", "pad"):
        token_id = getattr(model.trainer_spec, name + "_id")
        if token_id >= 0:
            setattr(model.trainer_spec, name + "_piece", metadata["tokenizer.ggml.tokens"][token_id])
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.add_dummy_prefix = metadata.get("tokenizer.ggml.add_space_prefix", False)
    model.normalizer_spec.remove_extra_whitespaces = False
    for word, score, kind in zip(metadata["tokenizer.ggml.tokens"],
                                  metadata["tokenizer.ggml.scores"],
                                  metadata["tokenizer.ggml.token_type"]):
        piece = model.pieces.add()
        piece.piece, piece.score, piece.type = word, score, kind
    raw = model.SerializeToString()
    (bundle / "tokenizer.model").write_bytes(raw)
    return sp.SentencePieceProcessor(model_proto=raw)


def load_model(bundle):
    record = json.loads((bundle / "parameters.json").read_text())
    config = Gemma3nTextConfig(**record["config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma3nTextModel(config)
    arrays = {name: np.memmap(bundle / spec["file"], mode="r", dtype="float32",
                             shape=tuple(spec["shape"]))
              for name, spec in record["weights"].items()}
    state = {}
    for name, spec in record["parameters"].items():
        data = arrays[spec["tensor"]]
        if spec["slice"] is not None:
            data = data[spec["slice"]]
        state[name] = torch.from_numpy(data)
    model.load_state_dict(state, assign=True)
    model.rotary_emb = Gemma3nTextRotaryEmbedding(config, device="cpu")
    local = copy.deepcopy(config)
    local.rope_theta, local.rope_scaling = config.rope_local_base_freq, {"rope_type": "default"}
    model.rotary_emb_local = Gemma3nTextRotaryEmbedding(local, device="cpu")
    model.embed_tokens.embed_scale = torch.tensor(config.hidden_size**0.5)
    model.embed_tokens_per_layer.embed_scale = torch.tensor(config.hidden_size_per_layer_input**0.5)
    model.per_layer_projection_scale = torch.tensor(config.hidden_size**-0.5)
    model.per_layer_input_scale = torch.tensor(2**-0.5)
    for layer in model.layers:
        layer.altup.router_input_scale = torch.tensor(config.hidden_size**-1)
        layer.self_attn.v_norm.weight = torch.tensor(1.0)
        bound = layer.altup.config.altup_coef_clip
        coefficients = layer.altup.correction_coefs.weight.detach().clone()
        if bound is not None:
            coefficients.clamp_(-bound, bound)
        layer.altup.correction_coefs.weight = torch.nn.Parameter(coefficients, requires_grad=False)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())
    return model.eval(), record, arrays


class TextGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model
    def forward(self, ids, full, sliding, last):
        hidden = self.text(ids, attention_mask={"full_attention": full, "sliding_attention": sliding},
                           past_key_values=SharedPrefix(), use_cache=False).last_hidden_state
        current = hidden.index_select(1, last).squeeze(1)
        logits = torch.nn.functional.linear(current, self.text.embed_tokens.weight)
        cap = self.text.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap


class TokenCache:
    def __init__(self, previous, selector):
        self.previous = previous
        self.selector = selector
        self.current = []
        self.shared_layers = {}
    def get_seq_length(self):
        return 0
    def update(self, key, value, layer, options):
        key = torch.where(self.selector, key, self.previous[2 * layer])
        value = torch.where(self.selector, value, self.previous[2 * layer + 1])
        self.current.extend((key, value))
        return key, value


class TokenGraph(torch.nn.Module):
    def __init__(self, model, context):
        super().__init__()
        self.text = model
        self.register_buffer("positions", torch.arange(context), persistent=False)
    def forward(self, ids, position, *previous):
        distance = position.reshape(1, 1, 1, 1) - self.positions.reshape(1, 1, 1, -1)
        full = torch.where(distance >= 0, 0.0, torch.finfo(torch.float32).min)
        sliding = torch.where((distance >= 0) & (distance < self.text.config.sliding_window),
                              0.0, torch.finfo(torch.float32).min)
        cache = TokenCache(previous, (self.positions == position).reshape(1, 1, -1, 1))
        hidden = self.text(ids, attention_mask={"full_attention": full, "sliding_attention": sliding},
                           past_key_values=cache, position_ids=position.reshape(1, 1),
                           cache_position=position, use_cache=False).last_hidden_state
        logits = torch.nn.functional.linear(hidden[:, -1, :], self.text.embed_tokens.weight)
        cap = self.text.config.final_logit_softcapping
        return (torch.tanh(logits / cap) * cap, *cache.current)


class GpuTokenGraph(TokenGraph):
    """Keep large lookup tables in host memory and transfer only selected rows."""
    def forward(self, embedding, per_layer_embedding, position, *previous):
        distance = position.reshape(1, 1, 1, 1) - self.positions.reshape(1, 1, -1)
        full = torch.where(distance >= 0, 0.0, torch.finfo(torch.float32).min)
        sliding = torch.where((distance >= 0) & (distance < self.text.config.sliding_window),
                              0.0, torch.finfo(torch.float32).min)
        cache = TokenCache(previous, (self.positions == position).reshape(1, 1, -1, 1))
        inputs = embedding * self.text.embed_tokens.embed_scale
        per_layer = per_layer_embedding.reshape(1, 1, self.text.config.num_hidden_layers,
                                                self.text.config.hidden_size_per_layer_input)
        per_layer = per_layer * self.text.embed_tokens_per_layer.embed_scale
        hidden = self.text(inputs_embeds=inputs, per_layer_inputs=per_layer,
                           attention_mask={"full_attention": full, "sliding_attention": sliding},
                           past_key_values=cache, position_ids=position.reshape(1, 1),
                           cache_position=position, use_cache=False).last_hidden_state
        logits = torch.nn.functional.linear(hidden[:, -1, :], self.text.embed_tokens.weight)
        cap = self.text.config.final_logit_softcapping
        return (torch.tanh(logits / cap) * cap, *cache.current)


def masks(context, window):
    positions = np.arange(context)
    distance = positions[:, None] - positions[None, :]
    full = np.where(distance >= 0, 0.0, np.finfo("float32").min).astype("float32")
    sliding = np.where((distance >= 0) & (distance < window), 0.0,
                        np.finfo("float32").min).astype("float32")
    return full[None, None], sliding[None, None]


def main():
    bundle = Path(sys.argv[1])
    context = int(sys.argv[2])
    torch.set_num_threads(4)
    model, record, arrays = load_model(bundle)
    sp = tokenizer(bundle)
    pieces = json.loads((bundle / "tokenizer-gguf.json").read_text(encoding="utf-8"))["tokenizer.ggml.tokens"]
    if not 1 <= context <= model.config.max_position_embeddings:
        raise ValueError("The requested context exceeds the architecture's position range")
                                                                         
    def sparsity(self, value):
        return torch.relu(value - (value.mean(-1, keepdim=True) +
                          value.std(-1, keepdim=True, unbiased=False) * self.multiplier))
    for layer in model.layers:
        mlp = layer.mlp
        if mlp.activation_sparsity > 0:
            multiplier = torch.distributions.Normal(0, 1).icdf(torch.tensor(mlp.activation_sparsity))
            mlp.register_buffer("multiplier", multiplier, persistent=False)
            mlp._gaussian_topk = MethodType(sparsity, mlp)
    gpu = "--gpu" in sys.argv[3:]
    token_graph = (GpuTokenGraph if gpu else TokenGraph)(model, context).eval()
    cache_count = 2 * (model.config.num_hidden_layers - model.config.num_kv_shared_layers)
    caches = tuple(torch.zeros((1, model.config.num_key_value_heads, context, model.config.head_dim))
                   for _ in range(cache_count))
    arguments = (torch.tensor([[sp.bos_id()]]), torch.tensor([0]), *caches)
    input_names = ["ids", "position"] + [f"cache_{i}" for i in range(cache_count)]
    if gpu:
        embedding = torch.zeros((1, 1, model.config.hidden_size))
        per_layer = torch.zeros((1, 1, model.config.num_hidden_layers * model.config.hidden_size_per_layer_input))
        arguments = (embedding, per_layer, torch.tensor([0]), *caches)
        input_names = ["embedding", "per_layer_embedding", "position"] + [f"cache_{i}" for i in range(cache_count)]
    output_names = ["logits"] + [f"updated_{i}" for i in range(cache_count)]
    torch.onnx.export(token_graph, arguments, str(bundle / "text.onnx"), export_params=False,
                      dynamo=False, opset_version=18, do_constant_folding=False,
                      input_names=input_names, output_names=output_names)
    print("Exported", flush=True)
    import onnx
    onnx_graph = onnx.load(bundle / "text.onnx")
    names = [v.name for v in onnx_graph.graph.input]
    buffers = dict(token_graph.named_buffers())
    parameters = {}
    for name in names[len(input_names):]:
        original = name.removeprefix("text.")
        if original.endswith(".altup.correction_coefs.weight"):
                                                                                    
                                                                              
            array = token_graph.get_parameter(name).detach().numpy()
            filename = "coefficient-" + str(len(parameters)) + ".bin"
            array.tofile(bundle / filename)
            record["weights"][name] = {"file": filename, "shape": list(array.shape), "dtype": str(array.dtype)}
            parameters[name] = {"tensor": name, "slice": None, "shape": list(array.shape)}
        elif original in record["parameters"]:
            parameters[name] = record["parameters"][original]
        elif name in buffers:
            array = buffers[name].numpy()
            filename = "buffer-" + str(len(parameters)) + ".bin"
            array.tofile(bundle / filename)
            record["weights"][name] = {"file": filename, "shape": list(array.shape), "dtype": str(array.dtype)}
            parameters[name] = {"tensor": name, "slice": None, "shape": list(array.shape)}
        else:
            raise ValueError(f"Unbound ONNX input {name}")
    (bundle / "parameters.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (bundle / "model.json").write_text(json.dumps({
        "schema": "crosspascal.gemma3n.text.v1", "architecture": "gemma3n",
        "context": context, "sliding_window": model.config.sliding_window,
        "input_names": names, "parameters": parameters,
        "library": "gemma3n-cpu.dll", "tokenizer": "tokenizer.model",
        "context_window_size": context, "backend": "TVM LLVM CPU FP32",
        "modalities": ["text"], "inference_acceptance": "NOT_CHECKED",
        "host_embeddings": gpu,
        "runtime_inputs": input_names, "cache_count": cache_count,
        "cache_shape": [1, model.config.num_key_value_heads, context, model.config.head_dim],
        "source_sha256": record["source_sha256"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
