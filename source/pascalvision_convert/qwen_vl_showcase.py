# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Held-source Preview/Build gate for architecture-derived Qwen2.5-VL."""
import ctypes
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from . import app_paths, gemma_showcase, source_lock

_preview = None
_GIB = 1024**3


def _inspect(data):
    if type(data) is not dict or not source_lock._is_windows():
        raise ValueError("Qwen2.5-VL conversion needs a local Windows checkpoint")
    source = Path(str(data.get("path") or "").strip())
    if not source.is_absolute() or source.is_symlink() or getattr(source, "is_junction", lambda: False)():
        raise ValueError("Select the full path of the actual checkpoint folder")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Select the checkpoint folder containing config.json and original Safetensors weights")
    source_lock._Kernel32().assert_local_ntfs(source)
    context, size, cpu = data.get("context", 2048), data.get("image_size", 224), data.get("cpu", "generic")
    if type(context) is not int or type(size) is not int:
        raise ValueError("Context and image dimensions must be integers")
    if type(cpu) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", cpu):
        raise ValueError("Enter an LLVM CPU target or generic")
    result = subprocess.run([gemma_showcase._gemma_python(), "-B",
        str(Path(__file__).with_name("qwen_vl_inspect.py")), str(source),
        "--context", str(context), "--image-size", str(size)], capture_output=True, text=True,
        timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise ValueError("Qwen2.5-VL source inspection failed. " + result.stderr[-1600:])
    report = json.loads(result.stdout)
    return source, report, cpu


def preview(data):
    global _preview
    if not gemma_showcase._lock.acquire(blocking=False):
        return {"trouble": "A language-model inspection or build is already running"}
    try:
        _preview = None
        source, report, cpu = _inspect(data)
        with gemma_showcase._held_pins(source, report["files"]) as pins:
            if _inspect(data) != (source, report, cpu):
                raise ValueError("Source changed while taking the preview")
        token = secrets.token_hex(24)
        _preview = SimpleNamespace(token=token, expires=time.monotonic()+600, data=dict(data), pins=pins,
                                   report=report)
        return dict(report, preview_token=token, cpu=cpu, experimental=True)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError, subprocess.TimeoutExpired) as exc:
        return {"trouble": str(exc)}
    finally:
        gemma_showcase._lock.release()


def build(data):
    global _preview
    if not gemma_showcase._lock.acquire(blocking=False):
        return {"trouble": "A language-model inspection or build is already running"}
    try:
        approved = _preview
        token = data.get("preview_token") if type(data) is dict else None
        if (approved is None or type(token) is not str or not secrets.compare_digest(token, approved.token)
                or time.monotonic() >= approved.expires):
            raise ValueError("Preview is missing, expired or already used; inspect the checkpoint again")
        _preview = None
        source, report, cpu = _inspect(approved.data)
        if report != approved.report:
            raise ValueError("Source topology changed after Preview")
        info = gemma_showcase._Memory()
        info.size = ctypes.sizeof(info)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
            raise OSError("Cannot verify available memory")
        required = report["expanded_bytes"] + 3*report['cache_bytes'] + 2*_GIB
        if min(info.values[1], info.values[3]) < required:
            raise ValueError("This FP32 profile needs %.1f GiB of available RAM and commit" % (required/_GIB))
        app_paths.prepare()
        parent = app_paths.data_path("bundles").resolve(strict=True)
        if shutil.disk_usage(parent).free < report["expanded_bytes"] + 2*_GIB:
            raise ValueError("Insufficient free disk space for the expanded model")
        output = parent / ("qwen-vl-" + token[:12])
        log = app_paths.data_path("logs") / ("qwen-vl-" + token[:12] + ".log")
        folder = Path(__file__).parent
        with tempfile.TemporaryDirectory(prefix=".qwen-vl-build-", dir=parent) as scratch:
            staged = Path(scratch)/"bundle"
            with gemma_showcase._held_pins(source, report["files"]) as pins:
                if pins != approved.pins:
                    raise ValueError("Source files changed after Preview")
                with log.open("xb") as stream:
                    for argv in (
                        [gemma_showcase._gemma_python(), "-B", str(folder/"qwen_vl_export.py"), str(source),
                         str(staged), "--context", str(report["context"]), "--image-size", str(report["image_size"])],
                        [sys.executable, "-B", str(folder/"qwen_vl_compile.py"), str(staged), "--cpu-arch", cpu]):
                        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=1800,
                                                creationflags=subprocess.CREATE_NO_WINDOW)
                        if result.returncode:
                            raise RuntimeError("Qwen2.5-VL conversion failed in " + Path(argv[2]).name + ". Log: " + str(log))
            model = json.loads((staged/"model.json").read_bytes())
            if model["source_sha256"] != approved.pins:
                raise RuntimeError("Converted model source hashes do not match Preview")
            staged.rename(output)
        return {"status": "BUILT_EXPERIMENTAL_QWEN2_5_VL", "bundle_root": str(output), "log": str(log),
                "backend": model["backend"], "inference": "NOT_CHECKED"}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError, subprocess.TimeoutExpired) as exc:
        return {"trouble": str(exc)}
    finally:
        gemma_showcase._lock.release()
