# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Validate the structural profile for a native language model build."""

import json
import os
from pathlib import Path


_PROFILES = {
    'qwen2.5-0.5b': {
        'architectures': ['Qwen2ForCausalLM'],
        'model_type': 'qwen2',
        'hidden_size': 896,
        'intermediate_size': 4864,
        'num_hidden_layers': 24,
        'num_attention_heads': 14,
        'num_key_value_heads': 2,
        'vocab_size': 151936,
        'rope_theta': 1000000,
        'bos_token_id': 151643,
        'eos_token_id': 151645,
        'tie_word_embeddings': True,
    },
    'gemma-2-2b': {
        'architectures': ['Gemma2ForCausalLM'],
        'model_type': 'gemma2',
        'hidden_size': 2304,
        'intermediate_size': 9216,
        'num_hidden_layers': 26,
        'num_attention_heads': 8,
        'num_key_value_heads': 4,
        'head_dim': 256,
        'vocab_size': 256000,
        'rope_theta': 10000,
        'query_pre_attn_scalar': 256,
        'attn_logit_softcapping': 50,
        'final_logit_softcapping': 30,
        'sliding_window': 4096,
        'bos_token_id': 2,
        'eos_token_id': [1, 107],
    },
}


_REAL_SCALARS = frozenset(('rope_theta', 'query_pre_attn_scalar',
                           'attn_logit_softcapping', 'final_logit_softcapping'))


def validate_exact_model(config, model_key):
    """Refuse an unapproved variant with a named, measurable field mismatch."""
    if model_key not in _PROFILES:
        raise ValueError('unsupported native LLM model_key %r' % model_key)
    if not isinstance(config, dict):
        raise ValueError('HF config must be a JSON object')
    for field, expected in _PROFILES[model_key].items():
        observed = config.get(field)
        if field in _REAL_SCALARS:
            agrees = (type(observed) in (int, float) and
                      observed == expected)
        else:
            agrees = (observed == expected and
                      type(observed) is type(expected))
        if not agrees:
            raise ValueError('HF model %s config mismatch in %s: expected %r, '
                             'got %r (different size/revision/model is NOT '
                             'automatically supported)' %
                             (model_key, field, expected, observed))
    return model_key


def identify_model_key(config):
    """Return supported key or None; known family with wrong size is an ERROR."""
    if not isinstance(config, dict):
        raise ValueError('HF model config must be an object')
    declared = config.get('architectures') or []
    if not isinstance(declared, list):
        raise ValueError('HF architectures field must be a list')
    for key, expected in _PROFILES.items():
        if expected['architectures'][0] in declared:
            return validate_exact_model(config, key)
    return None


def load_source_config(directory):
    """Read a bounded local config with strict JSON and identifiable profile."""
    if not isinstance(directory, (str, os.PathLike)):
        raise ValueError('HF source must name a local model directory')
    root = Path(directory)
    path = root / 'config.json'
    if (not root.is_dir() or root.is_symlink() or
            getattr(root, 'is_junction', lambda: False)() or
            path.is_symlink() or getattr(path, 'is_junction', lambda: False)()):
        raise ValueError('HF source is missing, redirected or not a real directory')
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError('missing/unreadable HF config.json: %s' % exc) from exc
    if not 1 <= len(raw) <= 1024 * 1024:
        raise ValueError('HF config.json is empty or exceeds 1MiB')

    def unique_fields(pairs):
        data = {}
        for name, value in pairs:
            if name in data:
                raise ValueError('duplicate HF config field %r' % name)
            data[name] = value
        return data

    def no_nonfinite(value):
        raise ValueError('nonfinite HF config JSON value %r' % value)

    try:
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_fields,
                          parse_constant=no_nonfinite)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError('invalid HF config.json: %s' % exc) from exc
    return identify_model_key(data), data
