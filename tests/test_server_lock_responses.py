"""Test actual Responder methods with a blocked fake writer; never import master/http/TVM."""
import argparse
import ast
import builtins
import copy
import hashlib
import hmac
from http.client import HTTPMessage
import json
from pathlib import Path
import threading
import types
from urllib.parse import urlparse

BASELINE_SHA = "c39ae95e957b5b56d09e949b76c3146079451f24b69d0ff44bd6541139aa18c4"
METHODS = {"_send", "do_GET", "do_POST", "_handle_post"}
SESSION = "isolated-regression-token-not-a-live-session"
BUSY_MESSAGE = "An operation is still running. Wait for it to finish before closing the converter."
CASES = (
    {"name": "get_while_closing", "method": "do_GET", "path": "/api/progress?job=3",
     "closing": True, "active": 2, "jobs": {}, "downloads": {}, "code": 503,
     "body": "Converter is closing"},
    {"name": "post_while_closing", "method": "do_POST", "path": "/api/build",
     "closing": True, "active": 2, "jobs": {}, "downloads": {}, "code": 503,
     "body": "Converter is closing"},
    {"name": "busy_quit", "method": "do_POST", "path": "/api/quit",
     "closing": False, "active": 1, "jobs": {"3": {"finished": False}},
     "downloads": {}, "code": 200, "body": {"trouble": BUSY_MESSAGE}},
    {"name": "existing_model_download", "method": "do_POST", "path": "/api/model-download",
     "closing": False, "active": 2, "jobs": {},
     "downloads": {"existing-download": {"finished": False, "status": "DOWNLOADING"}},
     "code": 200, "body": {"job": "existing-download"}},
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def forbidden(*args, **kwargs):
    raise AssertionError("A forbidden import, job thread, token creation or server action was reached")


class BlockedWriter:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.body = bytearray()

    def write(self, data):
        require(type(data) is bytes, "Response writer expects bytes")
        self.entered.set()
        require(self.release.wait(5.0), "Fake writer was not released within five seconds")
        self.body.extend(data)
        return len(data)


class FakeBase:
    def send_response(self, code):
        self.codes.append(code)

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        self.headers_finished += 1

    def send_error(self, code, message):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

    def _read_body(self):
        return {"model_key": "qwen2.5-0.5b"}

    def _handle_get(self):
        raise AssertionError("A closing GET must return before routing")


def read_source(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(256 * 1024 + 1)
    require(len(raw) <= 256 * 1024, "Source exceeds the static test input cap")
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def selected_responder(source, source_name, case):
    module = ast.parse(source, filename=source_name)
    found = [node for node in module.body
             if isinstance(node, ast.ClassDef) and node.name == "Responder"]
    require(len(found) == 1, "Expected exactly one Responder class")
    methods = [node for node in found[0].body
               if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    require({node.name for node in methods} == METHODS and len(methods) == len(METHODS),
            "Expected the four actual methods, once each")
    require(all(not node.decorator_list for node in methods), "Unexpected method decorator")
    fixture = ast.ClassDef(name="Responder", bases=[ast.Name(id="FakeBase", ctx=ast.Load())],
                           keywords=[], body=methods, decorator_list=[], type_params=[])
    executable = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))
    sandbox_builtins = dict(vars(builtins))
    sandbox_builtins["__import__"] = forbidden
    namespace = {
        "__name__": "isolated_master_lock_regression",
        "__builtins__": sandbox_builtins,
        "FakeBase": FakeBase,
        "LOCK": threading.Lock(),
        "SHUTTING_DOWN": case["closing"],
        "ACTIVE_REQUESTS": case["active"],
        "JOBS": copy.deepcopy(case["jobs"]),
        "MODEL_DOWNLOADS": copy.deepcopy(case["downloads"]),
        "SESSION_TOKEN": SESSION,
        "json": json,
        "urlparse": urlparse,
        "secrets": types.SimpleNamespace(compare_digest=hmac.compare_digest, token_hex=forbidden),
        "threading": types.SimpleNamespace(Thread=forbidden),
        "_JSONPostRejected": type("_JSONPostRejected", (Exception,), {}),
    }


    exec(compile(executable, source_name, "exec"), namespace)
    return namespace["Responder"], namespace


def exercise(source, source_name, case, expect_released):
    responder_type, namespace = selected_responder(source, source_name, case)
    handler = responder_type()
    handler.path = case["path"]
    handler.headers = HTTPMessage()
    handler.headers["Host"] = "127.0.0.1:8766"
    handler.headers["Origin"] = "http://127.0.0.1:8766"
    handler.headers["X-CrossPascal-Session"] = SESSION
    handler.server = types.SimpleNamespace(server_port=8766, shutdown=forbidden)
    handler.codes, handler.response_headers, handler.headers_finished = [], [], 0
    handler.wfile = BlockedWriter()
    failures = []

    def respond():
        try:
            getattr(handler, case["method"])()
        except BaseException as error:
            failures.append(type(error).__name__ + ": " + str(error))

    worker = threading.Thread(target=respond, daemon=True, name="isolated-fake-response")
    worker.start()
    acquired = False
    writer_was_blocked = False
    try:
        require(handler.wfile.entered.wait(2.0), "Response did not reach the fake writer")
        writer_was_blocked = worker.is_alive() and not handler.wfile.release.is_set()
        require(writer_was_blocked, "Writer was not blocked at the lock check")

        acquired = namespace["LOCK"].acquire(timeout=0.2)
        if acquired:
            namespace["LOCK"].release()
    finally:
        handler.wfile.release.set()
        worker.join(2.0)
    require(not worker.is_alive(), "Responder thread did not terminate")
    require(not failures, "Responder failed: " + repr(failures))
    require(acquired is expect_released,
            "Shared LOCK availability differs while writer blocks: " + case["name"])
    require(handler.codes == [case["code"]], "Response status changed")
    body = bytes(handler.wfile.body).decode("utf-8")
    if case["code"] == 200:
        require(json.loads(body) == case["body"], "Response JSON changed")
        headers = dict(handler.response_headers)
        require(headers.get("Content-Type") == "application/json; charset=utf-8",
                "JSON content type changed")
        require(headers.get("Content-Length") == str(len(handler.wfile.body)),
                "JSON content length changed")
    else:
        require(body == case["body"], "Error message changed")
    require(namespace["ACTIVE_REQUESTS"] == case["active"], "Request counter leaked or changed")
    require(namespace["SHUTTING_DOWN"] is case["closing"], "Shutdown state changed")
    require(namespace["JOBS"] == case["jobs"], "Jobs changed")
    require(namespace["MODEL_DOWNLOADS"] == case["downloads"], "Download reservation changed")
    return {"case": case["name"], "blocked_writer": writer_was_blocked,
            "another_thread_acquired_LOCK": acquired, "http_status": handler.codes[0],
            "response_and_state_preserved": True}


def run_version(path, expect_released, baseline=False):
    source, digest = read_source(path)
    if baseline:
        require(digest == BASELINE_SHA, "Baseline source SHA differs from reviewed original")
    return {"source": str(Path(path).absolute()), "sha256": digest,
            "expected_LOCK_available": expect_released,
            "cases": [exercise(source, str(path), case, expect_released) for case in CASES]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Review copy or applied source to test")
    parser.add_argument("--baseline", help="Optional untouched original for four negative controls")
    args = parser.parse_args()
    result = {"schema": "crosspascal.server-lock-writer-regression.v1",
              "status": "FAIL", "actual_http_requests": 0, "model_runs": 0, "versions": []}
    try:
        if args.baseline:
            result["versions"].append(run_version(args.baseline, False, baseline=True))
        result["versions"].append(run_version(args.source, True))
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
