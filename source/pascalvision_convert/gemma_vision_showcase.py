# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Add image support to an existing CUDA text bundle without rebuilding it."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from . import app_paths, gemma_showcase, source_lock

_preview = None


def inspect(data):
    if not isinstance(data, dict):
        raise ValueError('Choose a converted text bundle folder and a matching projector file')
    paths = []
    for field, label in [('bundle', 'converted text bundle folder'), ('projector', 'vision projector file')]:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Choose the ' + label)
        path = Path(value.strip().strip('"'))
        if not path.is_absolute():
            raise ValueError('Use the full local path to the ' + label)
        paths.append(path.resolve(strict=True))
    bundle, projector = paths
    if not bundle.is_dir() or not (bundle / 'model.json').is_file():
        raise ValueError('Select a converted bundle folder containing model.json')
    if not projector.is_file():
        raise ValueError('Select the matching vision projector GGUF file')
    api = source_lock._Kernel32()
    api.assert_local_ntfs(bundle)
    api.assert_local_ntfs(projector)
    model = json.loads((bundle / 'model.json').read_text(encoding='utf-8'))
    if (model.get('architecture') != 'gemma3n' or model.get('backend') != 'TVM CUDA W8A32'
            or not model.get('host_embeddings') or model.get('vision_manifest')):
        raise ValueError('Select a converted Gemma 3n CUDA text bundle without an existing vision add-on')
    script = Path(__file__).with_name('gemma3n_vision_export.py')
    result = subprocess.run([gemma_showcase._gemma_python(), '-B', str(script), str(projector), '--inspect'],
        capture_output=True, text=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise ValueError('Cannot inspect the vision projector. ' + result.stderr[-1500:])
    metadata = json.loads(result.stdout)
    record = json.loads((bundle / 'parameters.json').read_text(encoding='utf-8'))
    if metadata['clip.vision.projection_dim'] != record['config']['hidden_size']:
        raise ValueError('Projector and text hidden dimensions do not match')
    files = sorted(p.name for p in bundle.iterdir() if p.is_file())
    if any(p.is_symlink() or p.is_dir() for p in bundle.iterdir()):
        raise ValueError('Use a flat, regular converted bundle')
    return bundle, projector, model, metadata, files


def preview(data):
    global _preview
    if not gemma_showcase._lock.acquire(blocking=False):
        return {'trouble': 'A Gemma operation is already running'}
    try:
        _preview = None
        bundle, projector, model, metadata, files = inspect(data)
        with gemma_showcase._held_pins(bundle, files) as text_pins, gemma_showcase._held_pins(projector.parent, [projector.name]) as vision_pins:
            token = secrets.token_hex(24)
            _preview = SimpleNamespace(data=dict(data), token=token, expires=time.monotonic()+600,
                                       text_pins=text_pins, vision_pins=vision_pins)
        return {'preview_token': token, 'cuda_arch': model['cuda_arch'], 'image_size': metadata['clip.vision.image_size'],
                'modalities': ['text', 'image']}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {'trouble': str(exc)}
    finally:
        gemma_showcase._lock.release()


def build(data):
    global _preview
    if not gemma_showcase._lock.acquire(blocking=False):
        return {'trouble': 'A Gemma operation is already running'}
    try:
        approved = _preview
        token = data.get('preview_token')
        if (approved is None or not isinstance(token, str) or not secrets.compare_digest(token, approved.token)
                or time.monotonic() >= approved.expires):
            raise ValueError('Inspect the vision selection again; Preview expired or was already used')
        _preview = None
        bundle, projector, model, metadata, files = inspect(approved.data)
        output = app_paths.data_path('bundles') / ('gemma3n-image-' + token[:12])
        log = app_paths.data_path('logs') / ('gemma3n-image-' + token[:12] + '.log')
        folder = Path(__file__).parent
        with tempfile.TemporaryDirectory(prefix='.vision-build-', dir=output.parent) as scratch:
            staged = Path(scratch) / 'bundle'
            staged.mkdir()
            with gemma_showcase._held_pins(bundle, files) as text_pins, gemma_showcase._held_pins(projector.parent, [projector.name]) as vision_pins:
                if text_pins != approved.text_pins or vision_pins != approved.vision_pins:
                    raise ValueError('Selected source changed after Preview')
                required = sum((bundle / name).stat().st_size for name in files) + 4 * 1024**3
                if shutil.disk_usage(staged).free < required:
                    raise ValueError(f'The independent image bundle needs about {required / 1024**3:.1f} GiB of free disk space')
                for name in files:
                    shutil.copy2(bundle / name, staged / name)
                steps = [[gemma_showcase._gemma_python(), '-B', str(folder/'gemma3n_vision_export.py'), str(projector),
                          '--output', str(staged), '--text-bundle', str(bundle)],
                         [sys.executable, '-B', str(folder/'gemma3n_vision_compile.py'), str(staged)]]
                with log.open('xb') as stream:
                    for argv in steps:
                        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=1800,
                                                creationflags=subprocess.CREATE_NO_WINDOW)
                        if result.returncode:
                            raise RuntimeError('Vision conversion failed. Log: ' + str(log))
                vision = json.loads((staged/'vision.json').read_bytes())
                if vision['source_sha256'] != vision_pins[projector.name]:
                    raise ValueError('Converted vision source does not match Preview')
                model['vision_manifest'] = 'vision.json'
                model['artifact_sha256']['vision.json'] = hashlib.sha256((staged/'vision.json').read_bytes()).hexdigest()
                (staged/'model.json').write_text(json.dumps(model, indent=2), encoding='utf-8')
            staged.rename(output)
        return {'status': 'BUILT_EXPERIMENTAL_GEMMA3N_IMAGE', 'bundle_root': str(output)}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {'trouble': str(exc)}
    finally:
        gemma_showcase._lock.release()
