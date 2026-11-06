# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""One local wizard instance per user, with a bounded discovery receipt."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
import webbrowser

from . import app_paths

SESSION_FILE = "wizard-session.json"


def session_path():
    return app_paths.data_root() / SESSION_FILE


@contextmanager
def instance_lock():
    root = app_paths.prepare()
    key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:24]
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        api.CreateMutexW.restype = wintypes.HANDLE
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        api.CloseHandle.restype = wintypes.BOOL
        handle = api.CreateMutexW(None, False, "Local\\CrossPascalConvert-" + key)
        if not handle:
            raise OSError(ctypes.get_last_error(), "Cannot create converter instance mutex")
        acquired = ctypes.get_last_error() != 183
        try:
            yield acquired
        finally:
            api.CloseHandle(handle)
    else:
        import fcntl
        with (root / "wizard.lock").open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
            try:
                yield acquired
            finally:
                if acquired:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_session():
    path = session_path()
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Wizard session file is redirected")
    if not path.is_file() or not 1 <= path.stat().st_size <= 4096:
        raise ValueError("The existing converter is still starting; open it again shortly")
    item = json.loads(path.read_text(encoding="utf-8"))
    if (item.get("app") != "CrossPascal Convert" or
            type(item.get("port")) is not int or not 1024 <= item["port"] <= 65535 or
            type(item.get("pid")) is not int or item["pid"] <= 0 or
            type(item.get("instance")) is not str or len(item["instance"]) != 32):
        raise ValueError("Invalid wizard session record")
    return item


def open_existing(open_browser=True):
    expected = read_session()
    url = "http://127.0.0.1:%d/" % expected["port"]
    with urlopen(url + "api/identity", timeout=3) as reply:
        raw = reply.read(4097)
    if len(raw) > 4096 or json.loads(raw.decode("utf-8")) != expected:
        raise ValueError("The recorded port belongs to a different service")
    if open_browser:
        webbrowser.open(url)
    return expected


def publish(server):
    item = {"app": "CrossPascal Convert", "pid": os.getpid(),
            "port": server.server_port, "instance": server.instance_id}
    path = session_path()
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Wizard session file is redirected")
    temporary = path.with_name(path.name + "." + server.instance_id + ".tmp")
    with temporary.open("x", encoding="utf-8") as writer:
        json.dump(item, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, path)
    return item


def clear(instance_id):
    path = session_path()
    try:
        if read_session()["instance"] == instance_id:
            path.unlink()
    except (OSError, ValueError):
        pass

