# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Fail-closed SSH *probe* transport with local-only evidence.

Remote execution is restricted to fixed no-argument introspection commands.
It does not transfer models, run arbitrary shell, create remote files, install
packages, or attest any native model. OpenSSH itself may make OS-managed logs;
this API cannot guarantee a remote computer is physically diskless.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import time
from . import app_paths


class RemoteProbeRefused(RuntimeError):
    pass


_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}\Z")
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_PROBES = {
    "kernel": "uname -sr",
    "architecture": "uname -m",
    "cpu": "nproc",
    "gpu": "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",


    "windows_gpu": "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
    "windows_nvcc": "nvcc --version",
    "windows_identity": "powershell.exe -NoProfile -NonInteractive -Command \"[Environment]::MachineName; [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture\"",
}
_MAX_OUTPUT = 1024 * 1024


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    exit_code: int
    stdout_path: str
    stderr_path: str
    elapsed_seconds: float
    verification_status: str = "NOT_CHECKED"


def probe_ssh(*, host, user, known_hosts, local_evidence_dir, probe,
              port=22, timeout_seconds=20, max_output_bytes=65536,
              ssh_executable="ssh"):
    """Run one immutable probe; evidence directory must exist and be empty.

    A known_hosts file containing an independently verified pinned host key is
    mandatory. No password, key, token or host fingerprint is saved here.
    Caller provides an already-configured noninteractive SSH identity.
    """
    if (type(host) is not str or not _IDENTITY.fullmatch(host) or
            type(user) is not str or not _USER.fullmatch(user) or
            type(probe) is not str or probe not in _PROBES or
            type(port) is not int or not 1 <= port <= 65535
            or type(timeout_seconds) not in (int, float)
            or not 1 <= timeout_seconds <= 120 or
            type(max_output_bytes) is not int or
            not 1 <= max_output_bytes <= _MAX_OUTPUT or
            ssh_executable != "ssh"):
        raise RemoteProbeRefused("invalid SSH target, bounded probe or transport")
    identity = Path(known_hosts)
    evidence = Path(local_evidence_dir)
    if (not identity.is_file() or identity.is_symlink() or
            identity.stat().st_size == 0 or not evidence.is_dir() or
            evidence.is_symlink() or list(evidence.iterdir())):
        raise RemoteProbeRefused("pinned known_hosts and an empty local evidence directory required")

    root = app_paths.data_path("evidence").resolve()
    if root not in evidence.resolve().parents:
        raise RemoteProbeRefused("all probe artifacts must stay in the converter evidence directory")
    command = [
        "ssh", "-F", "none", "-T", "-n",
        "-o", "BatchMode=yes", "-o", "NumberOfPasswordPrompts=0",
        "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
        "-o", "GlobalKnownHostsFile=none",
        "-o", "UserKnownHostsFile=" + str(identity.resolve()),
        "-o", "ControlMaster=no", "-o", "ClearAllForwardings=yes",
        "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
        "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no",
        "-o", "ConnectTimeout=" + str(min(15, int(timeout_seconds))),
        "-p", str(port), "-l", user, "--", host, _PROBES[probe],
    ]
    start = time.monotonic()
    stdout_path, stderr_path = evidence / "probe.stdout", evidence / "probe.stderr"
    try:
        with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
            with subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  shell=False, env={**os.environ,
                                                    "SSH_ASKPASS": "",
                                                    "SSH_ASKPASS_REQUIRE": "never"}) as child:
                from threading import Thread, Lock
                failures = []
                total_output = 0
                output_lock = Lock()
                def drain(pipe, target):
                    nonlocal total_output
                    try:
                        while True:
                            chunk = pipe.read(8192)
                            if not chunk:
                                break
                            with output_lock:
                                total_output += len(chunk)
                                exceeded = total_output > max_output_bytes
                                if exceeded:
                                    failures.append("SSH combined output exceeded bounded quota")
                            if exceeded:
                                child.kill()
                                break
                            target.write(chunk)
                    except (OSError, ValueError) as exc:
                        failures.append(str(exc))
                        child.kill()
                streams = [Thread(target=drain, args=(child.stdout, out)),
                           Thread(target=drain, args=(child.stderr, err))]
                for thread in streams:
                    thread.start()
                try:
                    exit_code = child.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    child.kill()
                    child.wait()
                    raise RemoteProbeRefused("SSH probe timeout; no verification") from exc
                finally:
                    for thread in streams:
                        thread.join()
                if failures or exit_code != 0:
                    raise RemoteProbeRefused("SSH probe refused (exit=%s, stream=%s)" %
                                             (exit_code, bool(failures)))
    except (OSError, subprocess.SubprocessError) as exc:
        raise RemoteProbeRefused("SSH probe unavailable") from exc
    return ProbeResult(probe, exit_code, str(stdout_path), str(stderr_path),
                       time.monotonic() - start)
