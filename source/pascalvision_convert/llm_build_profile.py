# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import hashlib
import os
from pathlib import Path
import re
import stat
import sys

from . import qwen_universal

QWEN_CUDA_PROFILE = "qwen2.5-0.5b-cuda-f32-64-v1"
QWEN_15_CUDA_PROFILE = "qwen2.5-1.5b-cuda-f32-64-v1"
QWEN_GENERIC_CUDA_PROFILE = qwen_universal.PROFILE
_PROFILES = {
    QWEN_CUDA_PROFILE: {
        "profile_id": QWEN_CUDA_PROFILE,
        "model_key": "qwen2.5-0.5b",
        "target": "cuda",
        "dtype": "float32",
        "context_window_size": 64,
        "prefill_chunk_size": 64,
        "max_batch_size": 1,
    },
    QWEN_15_CUDA_PROFILE: {
        "profile_id": QWEN_15_CUDA_PROFILE,
        "model_key": "qwen2.5-1.5b",
        "target": "cuda",
        "dtype": "float32",
        "context_window_size": 64,
        "prefill_chunk_size": 64,
        "max_batch_size": 1,
    },
    QWEN_GENERIC_CUDA_PROFILE: {
        "profile_id": QWEN_GENERIC_CUDA_PROFILE,
        "model_key": qwen_universal.KEY,
        "target": "cuda",
        "dtype": "float32",
        "context_window_size": qwen_universal.CONTEXT,
        "prefill_chunk_size": qwen_universal.CONTEXT,
        "max_batch_size": 1,
    },
}


def select(profile_id, *, model_key=None):
    if profile_id in (None, ""):
        return None
    if type(profile_id) is not str or profile_id not in _PROFILES:
        raise ValueError("Unknown LLM build profile")
    selected = dict(_PROFILES[profile_id])
    if model_key is not None and model_key != selected["model_key"]:
        raise ValueError("The selected LLM profile does not match this model")
    return selected


def validate(options, *, route, target, dtype, sm):
    if options is None:
        return None
    if type(options) is not dict:
        raise ValueError("LLM build options must be an object")
    expected = select(options.get("profile_id"))
    if expected is None or options != expected or any(
            type(options.get(key)) is not type(value)
            for key, value in expected.items()):
        raise ValueError("LLM build options differ from the named profile")
    if route != "family_full_build":
        raise ValueError("LLM build options require a native family build")
    if (target, dtype) != (expected["target"], expected["dtype"]) or not sm:
        raise ValueError("The selected LLM profile requires a CUDA architecture and float32")
    return dict(expected)


def console_python(path=None):
    candidate = Path(path or sys.executable).absolute()
    if candidate.name.casefold() == 'pythonw.exe':
        candidate = candidate.with_name('python.exe')
        if not candidate.is_file():
            raise ValueError('The selected Python distribution is missing python.exe')
    return str(candidate)


def python_snapshot(path=None):
    candidate = Path(console_python(path))
    if not candidate.is_file():
        raise ValueError("The selected Python executable does not exist")
    info = candidate.stat()
    if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 64 * 1024**2:
        raise ValueError("The selected Python executable is not a bounded regular file")
    digest = hashlib.sha256()
    with candidate.open("rb") as reader:
        before = os.fstat(reader.fileno())
        remaining = info.st_size
        while remaining:
            block = reader.read(min(remaining, 1024**2))
            if not block:
                raise ValueError("Python executable changed while reading")
            digest.update(block)
            remaining -= len(block)
        if reader.read(1):
            raise ValueError("Python executable grew while reading")
        after = os.fstat(reader.fileno())
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(info) != identity(before) or identity(before) != identity(after):
        raise ValueError("Python executable changed while reading")
    return {"path": str(candidate), "bytes": info.st_size,
            "sha256": digest.hexdigest()}


def validate_python(record, *, route):
    if record is None:
        return None
    if (route != "family_full_build" or type(record) is not dict or
            set(record) != {"path", "bytes", "sha256"} or
            type(record["path"]) is not str or
            not os.path.isabs(record["path"]) or
            type(record["bytes"]) is not int or
            not 1 <= record["bytes"] <= 64 * 1024**2 or
            type(record["sha256"]) is not str or
            re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
        raise ValueError("Invalid selected LLM Python executable record")
    return dict(record)


def check_python(record):
    validate_python(record, route="family_full_build")
    if record is None or python_snapshot(record["path"]) != record:
        raise ValueError("The selected Python executable changed after Preview")
    return record["path"]

