# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from contextlib import ExitStack, contextmanager
import ctypes
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from . import app_paths, gemma_family, source_lock, targets
from .llm_helpers import weights


_lock = threading.Lock()
_preview = None
_GIB = 1024**3


class _Memory(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("load", ctypes.c_uint32),
                ("values", ctypes.c_uint64 * 7)]


def _admit(raw):
    info = _Memory()
    info.size = ctypes.sizeof(info)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
        raise OSError("Cannot verify Windows memory availability")
    peak_tensor = 4 * max(
        raw["vocab_size"] * raw["hidden_size"],
        2 * raw["intermediate_size"] * raw["hidden_size"],
        (raw["num_attention_heads"] + 2 * raw["num_key_value_heads"]) *
        raw["head_dim"] * raw["hidden_size"],
    )
    required = 8 * _GIB if raw["model_type"] == "gemma3n" else 3 * peak_tensor + 3 * _GIB
    if info.values[1] < required or info.values[3] < required:
        raise ValueError(f"Gemma conversion needs {required / _GIB:.1f} GiB of available RAM and commit")


def _inspect(data):
    if not source_lock._is_windows() or type(data) is not dict:
        raise ValueError("Gemma conversion requires a local Windows checkpoint")
    path = Path(str(data.get("path") or "").strip())
    if not path.is_absolute():
        raise ValueError("Enter the full local checkpoint folder path")
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Select the actual checkpoint folder")
    source = path.resolve(strict=True)
    api = source_lock._Kernel32()
    api.assert_local_ntfs(source)
    if source.is_file():
        inspector = Path(__file__).with_name("gemma3n_inspect.py")
        result = subprocess.run([_gemma_python(), "-B", str(inspector), str(source)],
                                capture_output=True, text=True, timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise ValueError("Gemma 3n inspection failed. " + result.stderr[-1500:])
        raw = json.loads(result.stdout)
        files, total = [source.name], raw["packed_bytes"]
        source = source.parent
    else:
        raw = gemma_family.json_file(source / "config.json", 1024 * 1024)
        gemma_family.require_backend(raw)
        total, files = gemma_family.validate_source(source, raw, weights)
    context = data.get("context", 1024)
    cpu = data.get("cpu", "generic")
    if (type(context) is not int or context not in (64, 128, 256, 512, 1024) or
            context > raw["max_position_embeddings"] or
            (raw["model_type"] == "gemma2" and context > raw["sliding_window"])):
        raise ValueError("Select a supported short context for the checkpoint")
    if type(cpu) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", cpu):
        raise ValueError("Enter an LLVM CPU target or generic")
    requested = data.get("target", "cpu")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Select automatic, CPU or CUDA execution")
    sm = targets.normalize_cuda_sm(data.get("cuda_arch"))
    detected = targets.normalize_cuda_sm(targets.host().get("cuda_arch")) if requested != "cpu" else None
    target = ("cuda" if raw["model_type"] == "gemma3n" and (detected or sm) else "cpu") if requested == "auto" else requested
    if target == "cuda":
        if raw["model_type"] != "gemma3n":
            raise ValueError("This CUDA route implements the Gemma 3n architecture")
        sm = sm or detected
        if not sm:
            raise ValueError("No CUDA architecture detected; enter the destination architecture")
    elif sm:
        raise ValueError("A CUDA architecture cannot be used with CPU execution")
    raw = dict(raw, target=target, cuda_arch=sm)
    return source, raw, context, cpu, total, files


def _gemma_python():
    executable = Path(os.environ.get("CROSSPASCAL_GEMMA_PYTHON", sys.executable))
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("Configure an installed Python with the Gemma conversion dependencies")
    return str(executable)


def _build_gemma3n(source, raw, staged, context, cpu, stream):
    folder = Path(__file__).parent
    steps = [
        [_gemma_python(), "-B", str(folder / "gemma3n_stage.py"), str(source / raw["source_file"]), str(staged)],
        [_gemma_python(), "-B", str(folder / "gemma3n_export.py"), str(staged), str(context), "--host-embeddings"],
        [sys.executable, "-B", str(folder / "gemma3n_compile.py"), str(staged), cpu],
    ]
    if raw["target"] == "cuda":
        steps[1].append("--gpu")
        steps[2] = [sys.executable, "-B", str(folder / "gemma3n_gpu.py"), str(staged), raw["cuda_arch"]]
    for argv in steps:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=1800,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError(f"Gemma 3n conversion failed in {Path(argv[2]).name}")


@contextmanager
def _held_pins(source, files):
    api = source_lock._Kernel32()
    pins = {}
    held = []
    with ExitStack() as stack:
        for member in files:
            path = source / member
            if path.parent != source or path.resolve(strict=True).parent != source:
                raise ValueError("A selected Gemma file is outside the checkpoint")
            handle = api.open(path)
            stack.callback(api.close, handle)
            identity = api.identity(handle)
            source_lock._selected_path_matches_handle(path, identity, identity.file_bytes)
            pins[member] = gemma_family.digest(path)
            held.append((path, handle, identity))
        yield pins
        for path, handle, identity in held:
            source_lock._selected_path_matches_handle(path, identity, identity.file_bytes)
            if api.identity(handle) != identity:
                raise ValueError(f"Selected Gemma file changed during verification: {path.name}")


def preview(data):
    global _preview
    if not _lock.acquire(blocking=False):
        return {"trouble": "A Gemma inspection or build is already running"}
    try:
        _preview = None
        source, raw, context, cpu, total, files = _inspect(data)
        with _held_pins(source, files) as pins:
            fresh = _inspect(data)
            if fresh != (source, raw, context, cpu, total, files):
                raise ValueError("Checkpoint changed while taking the Preview")
        token = secrets.token_hex(24)
        _preview = SimpleNamespace(token=token, expires=time.monotonic() + 600,
                                   data=dict(data), pins=pins)
        return {"preview_token": token, "model_type": raw["model_type"],
                "architecture": raw["architectures"][0], "layers": raw["num_hidden_layers"],
                "hidden_size": raw["hidden_size"], "packed_bytes": total, "context": context,
                "cpu": cpu, "target": raw["target"], "cuda_arch": raw["cuda_arch"],
                "backend": "TVM CUDA W8A32" if raw["target"] == "cuda" else "TVM CPU FP32", "experimental": True}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError,
            subprocess.TimeoutExpired) as exc:
        return {"trouble": str(exc)}
    finally:
        _lock.release()


def build(data):
    global _preview
    if not _lock.acquire(blocking=False):
        return {"trouble": "A Gemma inspection or build is already running"}
    try:
        approved = _preview
        token = data.get("preview_token") if type(data) is dict else None
        if (approved is None or type(token) is not str or
                not secrets.compare_digest(token, approved.token) or time.monotonic() >= approved.expires):
            raise ValueError("Gemma Preview is missing, expired or already used")
        _preview = None
        source, raw, context, cpu, total, files = _inspect(approved.data)
        _admit(raw)
        output = app_paths.data_path("bundles") / (raw["model_type"] + "-" + token[:12])
        log = app_paths.data_path("logs") / (raw["model_type"] + "-" + token[:12] + ".log")
        script = Path(gemma_family.__file__).resolve()
        bundles = output.parent.resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix=".gemma-build-", dir=bundles) as scratch:
            staged = Path(scratch) / "bundle"
            argv = [sys.executable, "-B", str(script), str(source), "--converter-source",
                    str(script.parents[1]), "--output", str(staged), "--context", str(context), "--cpu", cpu]
            with _held_pins(source, files) as pins:
                if pins != approved.pins or _inspect(approved.data) != (source, raw, context, cpu, total, files):
                    raise ValueError("Checkpoint changed after Preview. Inspect it again.")
                with log.open("xb") as stream:
                    if raw["model_type"] == "gemma3n":
                        _build_gemma3n(source, raw, staged, context, cpu, stream)
                    else:
                        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=900,
                                                creationflags=subprocess.CREATE_NO_WINDOW)
                        if result.returncode:
                            raise RuntimeError(f"Gemma compiler exited with code {result.returncode}. Log: {log}")
            model = gemma_family.json_file(staged / "model.json")
            expected = approved.pins[raw["source_file"]] if raw["model_type"] == "gemma3n" else approved.pins
            if model["source_sha256"] != expected:
                raise RuntimeError("The emitted Gemma model failed its source check")
            staged.rename(output)
        return {"status": "BUILT_EXPERIMENTAL_" + raw["model_type"].upper(), "bundle_root": str(output),
                "log": str(log), "backend": model.get("backend", "TVM CPU FP32"), "inference": "NOT_CHECKED"}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError,
            subprocess.TimeoutExpired) as exc:
        return {"trouble": str(exc)}
    finally:
        _lock.release()
