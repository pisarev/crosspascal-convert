# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import hashlib
import math
import os
import struct
from pathlib import Path

KEY = 'qwen2-compatible-local'
PROFILE = 'qwen2-compatible-cuda-f32-2048-v1'
SCHEMA = 'crosspascal.qwen2-compatible.v1'
TOKENIZER_SHA256 = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
TOKENIZER_BYTES = 7031645
MAX_PACKED_BYTES = 12 * 1024**3
MAX_LAYERS = 80
CONTEXT = 2048
VOCAB = 151936
BOS = 151643
EOS = 151645
_MIB = 1024**2


def _int(raw, name, low, high):
    value = raw.get(name)
    if type(value) is not int or not low <= value <= high:
        raise ValueError('unsupported Qwen2-compatible %s (expected integer %d..%d)'
                         % (name, low, high))
    return value


def _real(raw, name, low, high):
    value = raw.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('unsupported Qwen2-compatible %s (not finite/in range)' % name)
    return float(value)


def validate_config(raw):
    if type(raw) is not dict:
        raise ValueError('Qwen2-compatible config must be a JSON object')
    if raw.get('architectures') != ['Qwen2ForCausalLM'] or raw.get('model_type') != 'qwen2':
        raise ValueError('only dense Qwen2ForCausalLM text architecture is supported')
    for field in ('auto_map', 'custom_pipelines', 'trust_remote_code',
                  'quantization_config', 'vision_config', 'audio_config',
                  'image_token_id', 'video_token_id', 'rope_scaling',
                  'text_config', 'num_experts', 'num_local_experts',
                  'experts_per_token', 'router_aux_loss_coef'):
        if raw.get(field) is not None and raw.get(field) is not False:
            raise ValueError('unsupported Qwen2-compatible option: ' + field)
    if raw.get('use_sliding_window', False) is not False:
        raise ValueError('Qwen2-compatible sliding window requires a different graph')
    if raw.get('attention_bias', True) is not True or raw.get('mlp_bias', False) is not False:
        raise ValueError('Qwen2-compatible attention/MLP bias semantics differ')
    if raw.get('hidden_act', 'silu') != 'silu':
        raise ValueError('Qwen2-compatible requires SiLU gated MLP')
    if 'attention_dropout' in raw:
        _real(raw, 'attention_dropout', 0, 0)
    hidden = _int(raw, 'hidden_size', 128, 8192)
    intermediate = _int(raw, 'intermediate_size', 128, 32768)
    layers = _int(raw, 'num_hidden_layers', 1, MAX_LAYERS)
    heads = _int(raw, 'num_attention_heads', 1, 128)
    kv = _int(raw, 'num_key_value_heads', 1, 128)
    vocab = _int(raw, 'vocab_size', VOCAB, VOCAB)
    _int(raw, 'bos_token_id', BOS, BOS)
    _int(raw, 'eos_token_id', EOS, EOS)
    if type(raw.get('tie_word_embeddings')) is not bool:
        raise ValueError('Qwen2-compatible tied/untied embedding flag must be boolean')
    if hidden % heads or heads % kv:
        raise ValueError('Qwen2-compatible invalid head/KV grouping')
    head_dim = hidden // heads
    if not 32 <= head_dim <= 256 or head_dim % 2:
        raise ValueError('Qwen2-compatible head dimension is outside supported rotary layout')
    if 'head_dim' in raw and _int(raw, 'head_dim', head_dim, head_dim) != head_dim:
        raise ValueError('Qwen2-compatible explicit head_dim differs from Qwen2 graph')
    position_limit = raw.get('max_position_embeddings', CONTEXT)
    if type(position_limit) is not int or position_limit < CONTEXT:
        raise ValueError('Qwen2-compatible max positions must cover the %d-token context' % CONTEXT)
    theta = _real(raw, 'rope_theta', 1000, 1000000000)
    eps = _real(raw, 'rms_norm_eps', 1e-9, 1e-3)
    cfg = {'hidden_size': hidden, 'intermediate_size': intermediate,
           'num_hidden_layers': layers, 'num_attention_heads': heads,
           'num_key_value_heads': kv, 'head_dim': head_dim,
           'vocab_size': vocab, 'bos_token_id': BOS, 'eos_token_id': EOS,
           'tie_word_embeddings': raw['tie_word_embeddings'],
           'rope_theta': theta, 'rms_norm_eps': eps}
    packed = packed_layout(cfg)
    amount = sum(4 * math.prod(dims) for _name, dims in packed)
    if amount > MAX_PACKED_BYTES:
        raise ValueError('Qwen2-compatible FP32 parameters exceed 12 GiB safe output quota')
    return cfg


def packed_layout(cfg):
    hidden, inter = cfg['hidden_size'], cfg['intermediate_size']
    kv_width = cfg['num_key_value_heads'] * cfg['head_dim']
    fusion = hidden + 2 * kv_width
    result = [('model.embed_tokens.weight', [cfg['vocab_size'], hidden])]
    for layer in range(cfg['num_hidden_layers']):
        prefix = 'model.layers.%d.' % layer
        result.extend((
            (prefix + 'self_attn.c_attn.weight', [fusion, hidden]),
            (prefix + 'self_attn.c_attn.bias', [fusion]),
            (prefix + 'self_attn.o_proj.weight', [hidden, hidden]),
            (prefix + 'mlp.gate_up_proj.weight', [2 * inter, hidden]),
            (prefix + 'mlp.down_proj.weight', [hidden, inter]),
            (prefix + 'input_layernorm.weight', [hidden]),
            (prefix + 'post_attention_layernorm.weight', [hidden]),
        ))
    result.append(('model.norm.weight', [hidden]))
    if not cfg['tie_word_embeddings']:
        result.append(('lm_head.weight', [cfg['vocab_size'], hidden]))
    return result


def hf_layout(cfg):
    h, i = cfg['hidden_size'], cfg['intermediate_size']
    kv = cfg['num_key_value_heads'] * cfg['head_dim']
    result = {'model.embed_tokens.weight': [cfg['vocab_size'], h],
              'model.norm.weight': [h]}
    for layer in range(cfg['num_hidden_layers']):
        at = 'model.layers.%d.' % layer
        result.update({
            at + 'self_attn.q_proj.weight': [h, h],
            at + 'self_attn.q_proj.bias': [h],
            at + 'self_attn.k_proj.weight': [kv, h],
            at + 'self_attn.k_proj.bias': [kv],
            at + 'self_attn.v_proj.weight': [kv, h],
            at + 'self_attn.v_proj.bias': [kv],
            at + 'self_attn.o_proj.weight': [h, h],
            at + 'mlp.gate_proj.weight': [i, h],
            at + 'mlp.up_proj.weight': [i, h],
            at + 'mlp.down_proj.weight': [h, i],
            at + 'input_layernorm.weight': [h],
            at + 'post_attention_layernorm.weight': [h]})
    if not cfg['tie_word_embeddings']:
        result['lm_head.weight'] = [cfg['vocab_size'], h]
    return result


def requirements(cfg):
    packed = sum(4 * math.prod(dims) for _, dims in packed_layout(cfg))
    kv_cache = (2 * cfg['num_hidden_layers'] * cfg['num_key_value_heads'] *
                cfg['head_dim'] * CONTEXT * 4)
    max_fused = max(4 * math.prod(dims) for _, dims in packed_layout(cfg))
    return {'packed_parameter_bytes': packed, 'tensor_count': len(packed_layout(cfg)),
            'kv_cache_bytes': kv_cache,
            'min_free_gpu_bytes': packed + kv_cache + 512 * _MIB,
            'min_free_host_ram_bytes': max(2048 * _MIB, 3 * max_fused + 1024 * _MIB),
            'min_free_output_disk_bytes': 2 * packed + 512 * _MIB,
            'note': 'Lower-bound FP32 admission estimates, not a guarantee of TVM compiler peak.'}


def admit(cfg, *, disk_free=None, ram_free=None, gpu_free=None):
    need = requirements(cfg)
    for kind, available, requested in (
            ('output disk', disk_free, need['min_free_output_disk_bytes']),
            ('host RAM', ram_free, need['min_free_host_ram_bytes']),
            ('GPU VRAM', gpu_free, need['min_free_gpu_bytes'])):
        if available is not None:
            if type(available) is not int or available < 0:
                raise ValueError('invalid measured free ' + kind)
            if available < requested:
                raise ValueError('insufficient free %s: %d bytes, require at least %d'
                                 % (kind, available, requested))
    return need


def measured_available_ram():
    if os.name != 'nt':
        return None
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    state = MEMORYSTATUSEX()
    state.dwLength = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ValueError('cannot measure Windows free RAM for generic Qwen admission')
    return int(state.ullAvailPhys)


def descriptor(cfg, source_tree_sha256):
    if type(source_tree_sha256) is not str or len(source_tree_sha256) != 64 or any(
            ch not in '0123456789abcdef' for ch in source_tree_sha256):
        raise ValueError('Qwen2-compatible source closure SHA256 missing')
    verified = validate_config({
        **cfg, 'architectures': ['Qwen2ForCausalLM'], 'model_type': 'qwen2'})
    if verified != cfg:
        raise ValueError('Qwen2-compatible descriptor dimensions are not canonical')
    return {'schema': SCHEMA, 'model_key': KEY, 'source_kind': 'local-sha256',
            'source_tree_sha256': source_tree_sha256,
            'operator_contract': 'dense-silu-gqa-rope-normal-v1',
            'tokenizer_contract': 'qwen2-byte-bpe-im-v1',
            'tokenizer_sha256': TOKENIZER_SHA256,
            'context_window_size': CONTEXT, 'prefill_chunk_size': CONTEXT,
            'max_batch_size': 1, **cfg, **requirements(cfg)}


def validate_descriptor(value, model):
    if type(value) is not dict or value.get('schema') != SCHEMA:
        raise ValueError('Qwen2-compatible versioned descriptor missing')
    fields = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'head_dim',
              'vocab_size', 'bos_token_id', 'eos_token_id',
              'tie_word_embeddings', 'rope_theta', 'rms_norm_eps')
    cfg = {name: value.get(name) for name in fields}
    wanted = descriptor(cfg, model.get('source_tree_sha256'))
    if value != wanted:
        raise ValueError('Qwen2-compatible descriptor differs from derived semantics')
    for name in fields:
        if type(model.get(name)) is not type(cfg[name]) or model[name] != cfg[name]:
            raise ValueError('model.json differs from Qwen2-compatible descriptor: ' + name)
    return cfg


def _header(base, name):
    from . import hf_closure
    with hf_closure._opened_regular(base, name, max_bytes=hf_closure.MAX_TOTAL_BYTES) as (stream, info):
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated local Safetensors header: ' + name)
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= hf_closure.MAX_HEADER_BYTES or length > info.st_size - 8:
            raise ValueError('oversized/malformed local Safetensors header: ' + name)
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError('truncated local Safetensors tensor table: ' + name)
    return hf_closure._decode_json(raw, name)


def inspect_local_headers(directory, cfg):
    from . import hf_closure
    base, _ = hf_closure._root(directory)
    expected = hf_layout(cfg)
    selected, sharded = hf_closure._metadata_selection(base)
    if sharded:
        meta, shards = hf_closure._index_map(
            hf_closure._json_file(base, 'model.safetensors.index.json',
                                  hf_closure.MAX_INDEX_BYTES)[0])
    else:
        shards = {'model.safetensors': set(expected)}
    seen = {}
    kind = None
    for filename, registered in sorted(shards.items()):
        table = _header(base, filename)
        if type(table) is not dict:
            raise ValueError('local Safetensors header is not a JSON object')
        tensors = {name: row for name, row in table.items() if name != '__metadata__'}
        if set(tensors) != set(registered):
            raise ValueError('Safetensors shard index/header tensor closure differs: ' + filename)
        for name, item in tensors.items():
            if name in seen or name not in expected:
                raise ValueError('unexpected/duplicate Qwen2-compatible weight: ' + name)
            if type(item) is not dict or set(item) != {'dtype', 'shape', 'data_offsets'}:
                raise ValueError('malformed Qwen2-compatible tensor header: ' + name)
            if type(item['shape']) is not list or item['shape'] != expected[name]:
                raise ValueError('Qwen2-compatible weight shape mismatch: ' + name)
            dtype = item['dtype']
            if dtype not in ('F16', 'BF16', 'F32') or (kind is not None and dtype != kind):
                raise ValueError('mixed/unsupported Qwen2-compatible weight dtype: ' + name)
            kind = dtype
            seen[name] = filename
    if set(seen) != set(expected):
        missing = sorted(set(expected) - set(seen))
        raise ValueError('Qwen2-compatible weights absent: ' + ', '.join(missing[:3]))
    return {'input_dtype': kind, 'source_tensors': len(seen),
            'layout': 'sharded' if sharded else 'single'}


def verify_tokenizer(directory):
    from . import hf_closure
    base, _ = hf_closure._root(directory)
    path, info = hf_closure._regular(base, 'tokenizer.json',
                                    max_bytes=hf_closure.MAX_TOKENIZER_BYTES)
    if info.st_size != TOKENIZER_BYTES:
        raise ValueError('incompatible Qwen2-compatible tokenizer.json length')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            part = stream.read(1048576)
            if not part:
                break
            digest.update(part)
    if digest.hexdigest() != TOKENIZER_SHA256:
        raise ValueError('incompatible tokenizer semantics; only the accepted Qwen byte-BPE chat tokenizer is supported')
    return TOKENIZER_SHA256
