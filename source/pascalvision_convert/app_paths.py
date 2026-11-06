# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Writable per-user paths, separate from installed source and model inputs."""

import os
from pathlib import Path


def data_root():
    configured = os.environ.get("CROSSPASCAL_CONVERT_HOME")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("CROSSPASCAL_CONVERT_HOME must be an absolute directory")
        return root
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "CrossPascal" / "Convert"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "crosspascal-convert"


def data_path(name):
    if name not in ("uploads", "downloads", "bundles", "logs", "evidence", "models"):
        raise ValueError("Unknown converter data directory")
    return data_root() / name


def consumer_kit():
    """Locate the flat consumer kit supplied beside the converter repository."""
    selected = os.environ.get('CROSSPASCAL_CONSUMER_KIT')
    if selected:
        path = Path(selected).expanduser()
        if not path.is_absolute():
            raise ValueError('CROSSPASCAL_CONSUMER_KIT must be an absolute directory')
        return path
    return Path(__file__).resolve().parents[3] / 'crosspascal-tvm' / 'runtime' / 'windows-x64-cuda'


def prepare():
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("Converter data directory must not be a link or junction")
    for name in ("uploads", "downloads", "bundles", "logs", "evidence", "models"):
        folder = data_path(name)
        folder.mkdir(exist_ok=True)
        if (not folder.is_dir() or folder.is_symlink() or
                getattr(folder, "is_junction", lambda: False)()):
            raise ValueError("Converter data subdirectory is redirected")
    return root

