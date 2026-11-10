# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


import math

from tvm import relax as rx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, RopeMode, TIRPagedKVCache
from tvm.target import Target


class Gemma2Config(object):
    def __init__(self, raw, dtype='float32',
                 context_window_size=8192, prefill_chunk_size=512):
        self.hidden_size = raw['hidden_size']
        self.intermediate_size = raw['intermediate_size']
        self.num_hidden_layers = raw['num_hidden_layers']
        self.num_attention_heads = raw['num_attention_heads']
        self.num_key_value_heads = raw['num_key_value_heads']

        self.head_dim = raw['head_dim']
        self.vocab_size = raw['vocab_size']
        self.rms_norm_eps = raw['rms_norm_eps']
        self.rope_theta = raw['rope_theta']

        self.query_pre_attn_scalar = raw.get('query_pre_attn_scalar', self.head_dim)
        self.attn_logit_softcapping = raw.get('attn_logit_softcapping')
        self.final_logit_softcapping = raw.get('final_logit_softcapping')
        self.sliding_window = raw.get('sliding_window')
        self.tie_word_embeddings = raw.get('tie_word_embeddings', True)

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                'REFUSED: num_attention_heads %d is not divisible by num_key_value_heads %d'
                % (self.num_attention_heads, self.num_key_value_heads))

        self.dtype = dtype
        self.context_window_size = context_window_size
        self.prefill_chunk_size = prefill_chunk_size


def require_step08_exact_gemma2_native_attention(config):
    if (config.attn_logit_softcapping != 50 or
            config.final_logit_softcapping != 30 or
            config.sliding_window != 4096 or
            config.context_window_size != 8192 or
            config.prefill_chunk_size != 512 or
            config.num_hidden_layers != 26 or
            config.query_pre_attn_scalar != 256):
        raise ValueError(
            'selected original Gemma2 TYPE-1 requires attention score cap50 '
            '(pre-softmax), final vocabulary cap30 (distinct), alternating '
            'sliding local4096/global8192, chunked prefill512, 26 layers '
            'and query scale 256')
    raise ValueError(
        'selected Gemma2 TYPE-1 native REFUSED before output: installed TVM '
        'fused PagedKVCache does not implement pre-softmax '
        '50*tanh(score/50) and alternating local4096/global8192 windows; '
        'the independent scalar reference is NOT a trained TVM kernel')


def _gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    inner = op.tanh((x + x * x * x * 0.044715) * c)
    return x * (inner + 1.0) * 0.5


class Gemma2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        fused_out = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.c_attn = nn.Linear(config.hidden_size, fused_out,
                                bias=False, dtype=config.dtype)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim,
                                config.hidden_size,
                                bias=False, dtype=config.dtype)

        self.sm_scale = config.query_pre_attn_scalar ** (-0.5)

    def forward(self, hidden_states, paged_kv_cache, layer_id):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + 2 * h_kv, d))
        out = paged_kv_cache.attention_with_fused_qkv(
            layer_id, qkv, h_q, self.sm_scale)
        return self.o_proj(op.reshape(out, (b, s, h_q * d)))


class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size,
                                      2 * config.intermediate_size,
                                      bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size,
                                   bias=False, dtype=config.dtype)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = op.split(gate_up, 2, axis=-1)
        return self.down_proj(_gelu_tanh(gate) * up)


def _rms(config):
    return nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps,
                      bias=False, dtype=config.dtype)


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Gemma2Attention(config)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = _rms(config)
        self.post_attention_layernorm = _rms(config)
        self.pre_feedforward_layernorm = _rms(config)
        self.post_feedforward_layernorm = _rms(config)

    def forward(self, hidden_states, paged_kv_cache, layer_id):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, paged_kv_cache, layer_id)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        hidden_states = residual + h
        return hidden_states


class Gemma2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         dtype=config.dtype)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = _rms(config)

    def forward(self, input_embed, paged_kv_cache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        return self.norm(hidden_states)


def _take_last_position(hidden_states):
    def compute_last(x):
        from tvm import te
        b, s, d = x.shape
        return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k],
                          name='take_last')

    return op.tensor_expr_op(compute_last, name_hint='take_last',
                             args=[hidden_states])


class Gemma2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Gemma2Model(config)
        self.dtype = config.dtype
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.tie_word_embeddings = config.tie_word_embeddings
        self.final_logit_softcapping = config.final_logit_softcapping

        self.embed_scale = math.sqrt(config.hidden_size)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                     bias=False, dtype=config.dtype)

    def _logits(self, hidden_states):
        if self.tie_word_embeddings:
            weight = self.model.embed_tokens.weight
            logits = op.matmul(hidden_states, op.permute_dims(weight),
                               out_dtype='float32')
        else:
            logits = op.astype(self.lm_head(hidden_states), 'float32')
        cap = self.final_logit_softcapping
        if cap:
            logits = op.tanh(logits * (1.0 / cap)) * cap
        return logits

    def embed(self, input_ids):
        return self.model.embed_tokens(input_ids) * self.embed_scale

    def prefill(self, input_embed, paged_kv_cache):
        hidden_states = self.model(input_embed, paged_kv_cache)
        return self._logits(_take_last_position(hidden_states)), paged_kv_cache

    def decode(self, input_embed, paged_kv_cache):
        hidden_states = self.model(input_embed, paged_kv_cache)
        return self._logits(hidden_states), paged_kv_cache

    def create_tir_paged_kv_cache(self, max_batch_size, max_total_seq_len,
                                  prefill_chunk_size, page_size):
        return TIRPagedKVCache(
            attn_kind='mha',
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=0,
            layer_partition=rx.ShapeExpr([0, self.num_hidden_layers]),
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            mla_original_qk_head_dim=0,
            mla_original_v_head_dim=0,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_scaling={},
            rope_ext_factors=rx.prim_value(0),
            rotary_dim=self.head_dim,
            enable_disaggregation=False,
            dtype=self.dtype,
            target=Target.current(),
        )

    def get_default_spec(self):
        batch = 1
        return {
            'embed': {
                'input_ids': nn.spec.Tensor([batch, 'seq_len'], 'int32'),
                '$': {'param_mode': 'packed', 'effect_mode': 'none'},
            },
            'prefill': {
                'input_embed': nn.spec.Tensor(
                    [batch, 'seq_len', self.hidden_size], self.dtype),
                'paged_kv_cache': nn.spec.Object(object_type=PagedKVCache),
                '$': {'param_mode': 'packed', 'effect_mode': 'none'},
            },
            'decode': {
                'input_embed': nn.spec.Tensor(
                    [batch, 1, self.hidden_size], self.dtype),
                'paged_kv_cache': nn.spec.Object(object_type=PagedKVCache),
                '$': {'param_mode': 'packed', 'effect_mode': 'none'},
            },
            'create_tir_paged_kv_cache': {
                'max_batch_size': int,
                'max_total_seq_len': int,
                'prefill_chunk_size': int,
                'page_size': int,
                '$': {'param_mode': 'none', 'effect_mode': 'none'},
            },
        }
