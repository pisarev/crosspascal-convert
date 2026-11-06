# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Pinned public model sources used by the wizard's named profiles."""

import hashlib
import json
from pathlib import Path
import os
import shutil
from urllib.request import Request, urlopen

from . import app_paths

CATALOG = Path(__file__).with_name("model_catalog.json")


def catalog_model(model_key):
    raw = CATALOG.read_bytes()
    if not 1 <= len(raw) <= 128 * 1024:
        raise ValueError("Model catalog exceeds its byte limit")
    data = json.loads(raw.decode("utf-8"))
    if data.get("schema") != "crosspascal.model-catalog.v1":
        raise ValueError("Unknown model catalog schema")
    try:
        return data["models"][model_key]
    except KeyError as exc:
        raise ValueError("This model has no pinned public source profile") from exc


def match_source(scan, model_key):
    """Compare an already measured HF closure; do not rehash large weights."""
    expected = catalog_model(model_key)
    wanted = {
        item["path"]: (item["bytes"], item["sha256"])
        for item in expected["files"]
    }
    rows = scan.get("entries")
    if type(rows) is not list:
        raise ValueError("Missing measured HF source entries")
    observed = {}
    for item in rows:
        if type(item) is not dict or item.get("path") in observed:
            raise ValueError("Invalid or duplicate measured HF source entry")
        observed[item.get("path")] = (item.get("bytes"), item.get("sha256"))
    if (scan.get("model_key") != model_key or observed != wanted or
            scan.get("tree_sha256") != expected["tree_sha256"]):
        raise ValueError("The selected model files do not match the pinned official revision")
    return {
        "status": "PINNED_OFFICIAL_REVISION_MATCH",
        "repository": expected["repository"],
        "revision": expected["revision"],
        "source_tree_sha256": expected["tree_sha256"],
    }


def _hash(path, size):
    if (path.is_symlink() or getattr(path, "is_junction", lambda: False)() or
            not path.is_file() or path.stat().st_size != size):
        raise ValueError("Model file is missing, redirected, or has the wrong size: " + path.name)
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        remaining = size
        while remaining:
            block = reader.read(min(1024**2, remaining))
            if not block:
                raise ValueError("Model file shortened during verification")
            digest.update(block)
            remaining -= len(block)
        if reader.read(1):
            raise ValueError("Model file grew during verification")
    return digest.hexdigest()


def download(model_key, *, progress=None, opener=urlopen):
    """Download one immutable revision with bounded I/O and no file replacement."""
    selected = catalog_model(model_key)
    files = selected["files"] + [selected["license_file"]]
    root = app_paths.data_path("models")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("Model download root is redirected")
    destination = root / (model_key + "-" + selected["revision"][:7])
    total = sum(item["bytes"] for item in files)
    if destination.exists():
        if (destination.is_symlink() or getattr(destination, "is_junction", lambda: False)() or
                not destination.is_dir() or
                {p.name for p in destination.iterdir()} != {p["path"] for p in files}):
            raise ValueError("An incomplete or different model directory already exists: " + str(destination))
        for item in files:
            if _hash(destination / item["path"], item["bytes"]) != item["sha256"]:
                raise ValueError("An existing model file has a different checksum")
        return {"path": str(destination), "bytes": total,
                "repository": selected["repository"], "revision": selected["revision"],
                "status": "SOURCE_FILES_VERIFIED", "inference_status": "NOT_RUN"}
    if shutil.disk_usage(root).free < total + 128 * 1024**2:
        raise ValueError("Not enough free disk space for this model download")
    destination.mkdir(exist_ok=False)
    received = 0
    try:
        for item in files:
            name = item["path"]
            if name != Path(name).name or name in ("", ".", ".."):
                raise ValueError("Invalid model catalog filename")
            expected_url = ("https://huggingface.co/" + selected["repository"] +
                            "/resolve/" + selected["revision"] + "/" + name)
            if item["url"] != expected_url:
                raise ValueError("Model catalog URL is not the pinned immutable revision")
            digest = hashlib.sha256()
            count = 0
            request = Request(expected_url, headers={"User-Agent": "CrossPascal-Convert/0.2"})
            with opener(request, timeout=30) as response, (destination / name).open("xb") as writer:
                if not response.geturl().startswith("https://"):
                    raise ValueError("Model download redirected away from HTTPS")
                while count < item["bytes"]:
                    block = response.read(min(1024**2, item["bytes"] - count))
                    if not block:
                        raise ValueError("Model download ended before its declared size")
                    writer.write(block)
                    digest.update(block)
                    count += len(block)
                    received += len(block)
                    if progress is not None:
                        progress({"file": name, "received_bytes": received, "total_bytes": total})
                if response.read(1):
                    raise ValueError("Model download exceeds its pinned byte size")
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Downloaded model checksum differs from the pinned revision: " + name)
        return {"path": str(destination), "bytes": total,
                "repository": selected["repository"], "revision": selected["revision"],
                "status": "SOURCE_FILES_VERIFIED", "inference_status": "NOT_RUN"}
    except Exception as exc:
        raise RuntimeError("Model download failed; incomplete files remain at %s: %s" %
                           (destination, exc)) from exc

