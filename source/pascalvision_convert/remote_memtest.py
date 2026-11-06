"""Fixed Windows CPU BuildPlan contract smoke over pinned SSH, memory only.

This executes exact SHA-pinned first-party Python sources inside a remote
Python 3.12 process. It does not transfer models or execute native inference.
The result is CONTRACT_CHECK_ONLY / NOT_CHECKED, never VERIFIED.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from threading import Lock, Thread
from . import app_paths

SOURCE_HASHES = {
    "targets.py": "a1ad54622864de6db31f0e3fab90e77f83878a7ec0545d2fddcda04298632ab8",
    "llm_build_profile.py": "deae1e348c42fe41db1a9dee33d178d08d144f29d7019de110fbda3d4476ec8d",
    "build_plan.py": "7b622ff4475a0a024712a49fb9f8beced20ac3aa0a393f361b9a00cbbac99892",
}
ROOT = Path(__file__).resolve().parent
EVIDENCE_ROOT = app_paths.data_path("evidence")
MAX_INPUT = 256 * 1024
MAX_OUTPUT = 64 * 1024
WINDOWS_PYTHON312 = "python"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}\Z")
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")


class RemoteContractRefused(RuntimeError):
    pass

REMOTE_SCRIPT = r"""
import base64, hashlib, json, platform, sys, types
def refuse(reason):
    print(json.dumps({"status":"CONTRACT_REFUSED","verification_status":"NOT_CHECKED",
                      "reason":reason}),flush=True)
    sys.exit(1)
try:
    if platform.system() != "Windows" or sys.version_info[:2] != (3,12):
        refuse("requires Windows Python 3.12")
    raw=sys.stdin.buffer.read(262145)
    if len(raw)>262144: refuse("input limit")
    item=json.loads(raw.decode("utf-8"))
    if set(item)!={"sources","hashes","operation"} or item["operation"]!="cpu_generic_plan_v1":
        refuse("unsupported contract operation")
    names=("targets.py","llm_build_profile.py","build_plan.py")
    if set(item["sources"])!=set(names) or set(item["hashes"])!=set(names):
        refuse("source names")
    package=types.ModuleType("pascalvision_convert")
    package.__path__=[]
    sys.modules["pascalvision_convert"]=package
    for name in names:
        src=base64.b64decode(item["sources"][name],validate=True)
        if hashlib.sha256(src).hexdigest()!=item["hashes"][name]: refuse("source SHA")
        modname="pascalvision_convert."+name[:-3]
        module=types.ModuleType(modname)
        module.__package__="pascalvision_convert"
        sys.modules[modname]=module
        exec(compile(src, "<approved-"+name+">","exec"),module.__dict__)
    from pascalvision_convert import build_plan
    host={"os":"windows","arch":"x86_64","cuda_arch":None}
    result=build_plan.resolve(host,target="cpu",route="raw",tune=False,source="cpu-contract-only")
    build_plan.validate_recorded(result)
    if result["compiler_target"]!="llvm" or result["verification_status"]!="NOT_CHECKED":
        refuse("unexpected CPU contract")
    print(json.dumps({"status":"CONTRACT_CHECK_ONLY","verification_status":"NOT_CHECKED",
                      "native_inference_executed":False,"operation":item["operation"],
                      "plan_sha256":result["plan_sha256"],"source_hashes":item["hashes"]}),
          flush=True)
except Exception as exc:
    refuse("contract failed: "+type(exc).__name__)
"""


def _sources():
    if (set(SOURCE_HASHES) != {"targets.py", "llm_build_profile.py", "build_plan.py"} or
            any(type(value) is not str or
                re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in SOURCE_HASHES.values())):
        raise RemoteContractRefused("approved source pins require exact 64 lowercase SHA-256 hex")
    result = {}
    for name, expected in SOURCE_HASHES.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise RemoteContractRefused("approved source absent or linked")
        before = path.stat()
        source = path.read_bytes()
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RemoteContractRefused("approved source changed while reading")
        if hashlib.sha256(source).hexdigest() != expected:
            raise RemoteContractRefused("approved source SHA differs")
        result[name] = base64.b64encode(source).decode("ascii")
    payload = json.dumps({"operation": "cpu_generic_plan_v1",
                          "sources": result, "hashes": SOURCE_HASHES},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_INPUT:
        raise RemoteContractRefused("source payload exceeds input cap")
    return payload


def run(*, host, user, known_hosts, local_evidence_dir, port=22,
        timeout_seconds=30, popen=None):
    """Write evidence only into a preexisting empty evidence folder; fail closed."""
    if (type(host) is not str or not _NAME.fullmatch(host) or
            type(user) is not str or not _USER.fullmatch(user) or
            type(port) is not int or not 1 <= port <= 65535 or
            type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120):
        raise RemoteContractRefused("invalid remote identity or execution limits")
    pinned = Path(known_hosts)
    evidence = Path(local_evidence_dir)
    if (pinned.is_symlink() or not pinned.is_file() or pinned.stat().st_size < 1 or
            evidence.is_symlink() or not evidence.is_dir() or
            list(evidence.iterdir()) or
            EVIDENCE_ROOT.resolve() not in evidence.resolve().parents):
        raise RemoteContractRefused("pinned host key and a unique empty local evidence directory are required")
    payload = _sources()
    script = base64.b64encode(REMOTE_SCRIPT.encode("utf-8")).decode("ascii")

    fixed_command = ('"' + WINDOWS_PYTHON312 + '" -B -c '
                     '"import base64;exec(base64.b64decode(\'' + script + '\'))"')
    argv = ["ssh", "-F", "none", "-T", "-o", "BatchMode=yes",
            "-o", "NumberOfPasswordPrompts=0", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no", "-o", "GlobalKnownHostsFile=none",
            "-o", "UserKnownHostsFile=" + str(pinned.resolve()),
            "-o", "ControlMaster=no", "-o", "ClearAllForwardings=yes",
            "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
            "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no",
            "-o", "ConnectTimeout=15", "-p", str(port), "-l", user,
            "--", host, fixed_command]


    try:
        child = (popen or subprocess.Popen)(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=False,
            env={**os.environ, "SSH_ASKPASS": "", "SSH_ASKPASS_REQUIRE": "never"})
        buffers = [bytearray(), bytearray()]
        fault = []
        quota = Lock()

        def drain(index, pipe):
            try:
                while True:
                    block = pipe.read(8192)
                    if not block:
                        return
                    with quota:
                        if len(buffers[0]) + len(buffers[1]) + len(block) > MAX_OUTPUT:
                            fault.append("remote combined output limit")
                            child.kill()
                            return
                        buffers[index].extend(block)
            except (OSError, ValueError) as exc:
                fault.append(type(exc).__name__)
                child.kill()

        readers = [Thread(target=drain, args=(0, child.stdout), daemon=True),
                   Thread(target=drain, args=(1, child.stderr), daemon=True)]
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + timeout_seconds
        sender_done = __import__("threading").Event()

        def send():
            try:
                child.stdin.write(payload)
                child.stdin.close()
            except (OSError, ValueError):
                fault.append("SSH stdin closed before contract transfer")
            finally:
                sender_done.set()

        sender = Thread(target=send, daemon=True)
        sender.start()
        timed_out = False
        if not sender_done.wait(max(0, deadline - time.monotonic())):
            timed_out = True
            child.kill()
        try:
            child.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
        finally:
            for reader in readers:
                reader.join(timeout=max(0, deadline - time.monotonic()))
            sender.join(timeout=max(0, deadline - time.monotonic()))
        stdout, stderr = bytes(buffers[0]), bytes(buffers[1])


        (evidence / "contract.stdout").write_bytes(stdout)
        (evidence / "contract.stderr").write_bytes(stderr)
        if timed_out or not sender_done.is_set() or any(r.is_alive() for r in readers):
            child.kill()
            try:
                child.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise RemoteContractRefused("remote contract timed out")
        if fault:
            child.kill()
            try:
                child.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise RemoteContractRefused("remote contract stream failed or exceeded cap")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RemoteContractRefused("SSH contract execution failed") from exc
    if len(stdout) + len(stderr) > MAX_OUTPUT:
        raise RemoteContractRefused("remote output exceeded combined cap")

    if child.returncode != 0 or stderr:
        raise RemoteContractRefused("remote contract nonzero exit or stderr")
    try:
        result = json.loads(stdout.decode("utf-8"))
        if (type(result) is not dict or
                result.get("status") != "CONTRACT_CHECK_ONLY" or
                result.get("verification_status") != "NOT_CHECKED" or
                result.get("native_inference_executed") is not False or
                result.get("operation") != "cpu_generic_plan_v1" or
                result.get("source_hashes") != SOURCE_HASHES or
                not re.fullmatch(r"[0-9a-f]{64}", result.get("plan_sha256", ""))):
            raise ValueError("untrusted response fields")
    except (UnicodeError, ValueError, TypeError) as exc:
        raise RemoteContractRefused("invalid remote contract response") from exc
    return result


def main(argv=None):
    """Installed, fixed-purpose SSH contract command, never a model UAT."""
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert remote-cpu-contract',
        description=('Run a fixed SHA-pinned CPU BuildPlan contract in remote '
                     'Windows Python 3.12 memory; evidence stays in the local data directory. '
                     'This never performs trained/native model verification.'))
    parser.add_argument('--host', required=True,
                        help='approved SSH hostname or IP (no shell fragments)')
    parser.add_argument('--user', required=True,
                        help='existing noninteractive SSH username; no password')
    parser.add_argument('--known-hosts', required=True,
                        help='local nonempty known_hosts with independently pinned key')
    parser.add_argument('--evidence-dir', required=True,
                        help='pre-created empty directory beneath the converter evidence directory')
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--timeout-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    try:
        outcome = run(host=args.host, user=args.user,
                      known_hosts=args.known_hosts,
                      local_evidence_dir=args.evidence_dir,
                      port=args.port, timeout_seconds=args.timeout_seconds)
    except (RemoteContractRefused, OSError, ValueError) as exc:
        print(json.dumps({'status': 'CONTRACT_REFUSED',
                          'verification_status': 'NOT_CHECKED',
                          'native_inference_executed': False,
                          'reason': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
