import ast
import builtins
import copy
import hashlib
import io
import json
import linecache
import os
from pathlib import Path
import threading
import types
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
PRIVATE_VALUE = "LOCAL_VALUE_MUST_NOT_BE_CAPTURED_8175"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parsed(path):
    raw = path.read_bytes()
    require(len(raw) <= 256 * 1024, "Static source exceeds test cap")
    return ast.parse(raw.decode("utf-8"), filename=str(path)), hashlib.sha256(raw).hexdigest()


def execute_nodes(nodes, namespace, filename):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, filename, "exec"), namespace)


def restricted_import(name, *args, **kwargs):
    require(name == "traceback", "Unexpected import in extracted code: " + name)
    return builtins.__import__(name, *args, **kwargs)


def source_fixtures():
    pipeline_path = SOURCE / "pascalvision_convert/pipeline.py"
    master_path = SOURCE / "master.py"
    report_path = SOURCE / "pascalvision_convert/report.py"
    pipeline, pipeline_sha = parsed(pipeline_path)
    master, master_sha = parsed(master_path)
    report, report_sha = parsed(report_path)
    restricted = dict(vars(builtins))
    restricted["__import__"] = restricted_import
    common = {"__builtins__": restricted, "__name__": "isolated_failure_fixture"}
    report_nodes = [node for node in report.body
                    if isinstance(node, ast.ClassDef) and node.name in ("Step", "Report")
                    or isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name)
                        and target.id in ("DONE", "REFUSED", "UNCHECKED", "STATE_NAMES")
                        for target in node.targets)]
    report_ns = dict(common)
    execute_nodes(report_nodes, report_ns, "<isolated report classes>")
    helpers = [node for node in pipeline.body
               if isinstance(node, ast.FunctionDef) and node.name == "_build_failure_diagnostic"]
    require(len(helpers) == 1, "Missing actual diagnostic helper")
    runs = [node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    require(len(runs) == 1, "Missing actual pipeline.run")
    compile_tries = [node for node in runs[0].body if isinstance(node, ast.Try)
                     and any(isinstance(child, ast.ImportFrom) and child.module == "convert"
                             for child in ast.walk(node))]
    require(len(compile_tries) == 1, "Compile exception boundary is not unique")
    handlers = [node for node in compile_tries[0].handlers
                if isinstance(node.type, ast.Name) and node.type.id == "Exception"
                and node.name == "trouble"]
    require(len(handlers) == 1, "Missing actual compiler exception handler")
    pipeline_ns = dict(common, os=os, __file__=str(pipeline_path))
    execute_nodes(helpers, pipeline_ns, str(pipeline_path))
    synthetic_catch = ast.FunctionDef(
        name="capture",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
                           ("build", "step", "report", "finished")],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[ast.Try(body=[ast.Expr(value=ast.Call(func=ast.Name(id="build", ctx=ast.Load()),
                                                   args=[], keywords=[]))],
                      handlers=handlers, orelse=[], finalbody=[])],
        decorator_list=[], returns=None, type_comment=None, type_params=[])
    execute_nodes([synthetic_catch], pipeline_ns, "<isolated actual compile catch>")
    record_nodes = [node for node in master.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_record"]
    require(len(record_nodes) == 1, "Missing actual _record")

    def recorder(stream, step):
        namespace = dict(common, json=json, sys=types.SimpleNamespace(stderr=stream),
                         LOCK=threading.Lock(), JOBS={"1": {"steps": [], "summary": {}}})
        execute_nodes(record_nodes, namespace, str(master_path))
        return namespace, lambda: namespace["_record"]("1", step)

    pins = {"master.py": master_sha, "pipeline.py": pipeline_sha, "report.py": report_sha}
    return report_ns, pipeline_ns, recorder, pins


class BlockedLog:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.message, self.flushed = "", False

    def write(self, text):
        self.entered.set()
        require(self.release.wait(5.0), "Fake log writer was not released")
        self.message += text
        return len(text)

    def flush(self):
        self.flushed = True


class FailedLog:
    def write(self, text):
        raise OSError("Synthetic unavailable diagnostic sink")

    def flush(self):
        raise AssertionError("Failed write must not reach flush")


class ShortLog:
    def write(self, text):
        return 0

    def flush(self):
        raise AssertionError("Short write must not reach flush")


class ForbiddenLog:
    def write(self, text):
        raise AssertionError("A successful step must not write diagnostics")

    def flush(self):
        raise AssertionError("A successful step must not flush diagnostics")


def run_checks():
    report_ns, pipeline_ns, recorder, pins = source_fixtures()
    report_type = report_ns["Report"]
    converter_path = str((SOURCE / "pascalvision_convert/convert.py").resolve())
    synthetic = (
        "def inner_failure():\n"
        "    private_local = " + repr(PRIVATE_VALUE) + "\n"
        "    raise ValueError('synthetic inner failure')\n"
        "def convert_onnx():\n"
        "    try:\n"
        "        inner_failure()\n"
        "    except ValueError as cause:\n"
        "        raise RuntimeError('synthetic compiler refusal') from cause\n"
    )
    previous_cache = linecache.cache.get(converter_path)
    linecache.cache[converter_path] = (len(synthetic), None, synthetic.splitlines(True), converter_path)
    synthetic_ns = {}
    exec(compile(synthetic, converter_path, "exec"), synthetic_ns)
    checks = []
    try:
        report = report_type("Synthetic build")
        step = report.step("Building the module")
        callbacks = []
        returned = pipeline_ns["capture"](synthetic_ns["convert_onnx"], step, report,
                                           lambda value: callbacks.append(value))
        require(returned is report and callbacks == [step] and not report.sound,
                "Actual catch changed report/callback/refusal behavior")
        require(step.lines == ["synthetic compiler refusal"], "Original UI error changed")
        diagnostic = step._build_failure_diagnostic
        require(diagnostic["exception_type"] == "builtins.RuntimeError", "Wrong exception type")
        site = diagnostic["call_site"]
        require(diagnostic["call_site_scope"] == "converter_call"
                and site["file"] == converter_path and site["function"] == "convert_onnx"
                and site["line"] == 8 and "raise RuntimeError" in site["source"],
                "Actual converter call site was not retained")
        trace = diagnostic["traceback"]
        require("ValueError: synthetic inner failure" in trace
                and "RuntimeError: synthetic compiler refusal" in trace
                and "direct cause" in trace and PRIVATE_VALUE not in trace,
                "Full chain missing or local variables captured")
        public = json.dumps(report.as_dict()) + report.as_text()
        require("Traceback" not in public and "_build_failure_diagnostic" not in public,
                "Private traceback leaked into public report/UI serialization")
        checks.append({"id": "actual_catch_private_trace_and_call_site", "status": "PASS",
                       "cause_chain_preserved": True, "locals_captured": False})

        stream = io.StringIO()
        state, record = recorder(stream, step)
        record()
        logged = stream.getvalue()
        header = json.loads(logged.splitlines()[2])
        require(header["job"] == "1" and header["step"] == step.name
                and header["call_site"] == site and trace in logged,
                "Existing log did not receive the complete labelled diagnostic")
        require(state["JOBS"]["1"]["steps"] == [step.as_dict()]
                and "Traceback" not in json.dumps(state["JOBS"]),
                "Diagnostic leaked into normal job state")
        checks.append({"id": "existing_log_complete_private_diagnostic", "status": "PASS"})

        stream = BlockedLog()
        state, record = recorder(stream, step)
        errors = []

        def write_diagnostic():
            try:
                record()
            except BaseException as error:
                errors.append(type(error).__name__ + ": " + str(error))

        worker = threading.Thread(target=write_diagnostic, daemon=True, name="fake-diagnostic-log")
        worker.start()
        acquired = False
        recorded_before_write_finished = False
        try:
            require(stream.entered.wait(2.0), "Record did not reach fake log")
            acquired = state["LOCK"].acquire(timeout=0.2)
            if acquired:
                try:
                    recorded_before_write_finished = state["JOBS"]["1"]["steps"] == [step.as_dict()]
                finally:
                    state["LOCK"].release()
        finally:
            stream.release.set()
            worker.join(2.0)
        require(acquired and recorded_before_write_finished and not worker.is_alive()
                and not errors and stream.flushed, "Log I/O held LOCK or hid the refused step")
        checks.append({"id": "blocked_log_does_not_hold_LOCK", "status": "PASS",
                       "refused_step_already_visible": True})

        for name, sink in (("failed_log", FailedLog()), ("short_log_write", ShortLog())):
            state, record = recorder(sink, step)
            record()
            job = state["JOBS"]["1"]
            require(job["steps"] == [step.as_dict()] and not report.sound,
                    "Log failure replaced the original refusal")
            require(job["diagnostic_log_error"]["exception_type"] == "builtins.OSError"
                    and job["diagnostic_log_error"]["step"] == step.name,
                    "Diagnostic sink failure was not retained")
            job.update({"summary": {"verification_status": "NOT_CHECKED"}})
            require("diagnostic_log_error" in job, "Final summary replacement lost log error")
            checks.append({"id": name + "_preserves_original_refusal", "status": "PASS"})

        good = report_ns["Step"]("Successful ordinary step").done()
        state, record = recorder(ForbiddenLog(), good)
        record()
        require(state["JOBS"]["1"] == {"steps": [good.as_dict()], "summary": {}},
                "Successful progress gained logging/error side effects")
        checks.append({"id": "successful_step_has_no_diagnostic_io", "status": "PASS"})

        original_capture = pipeline_ns["_build_failure_diagnostic"]

        def fail_capture(error):
            raise OSError("Synthetic diagnostic capture failure")

        pipeline_ns["_build_failure_diagnostic"] = fail_capture
        try:
            failed_capture_report = report_type("Synthetic diagnostic failure")
            failed_capture_step = failed_capture_report.step("Building the module")
            returned = pipeline_ns["capture"](synthetic_ns["convert_onnx"], failed_capture_step,
                                               failed_capture_report, lambda value: value)
        finally:
            pipeline_ns["_build_failure_diagnostic"] = original_capture
        require(returned is failed_capture_report and not failed_capture_report.sound
                and failed_capture_step.lines == ["synthetic compiler refusal"]
                and failed_capture_step._build_failure_diagnostic["capture_error"] == "builtins.OSError",
                "Diagnostic capture failure masked the compiler error")
        checks.append({"id": "capture_failure_preserves_original_refusal", "status": "PASS"})
    finally:
        if previous_cache is None:
            linecache.cache.pop(converter_path, None)
        else:
            linecache.cache[converter_path] = previous_cache
    return pins, checks


def main():
    result = {"schema": "crosspascal.private-build-diagnostics-regression.v1",
              "status": "FAIL", "actual_http_requests": 0, "model_runs": 0, "checks": []}
    try:
        result["source_pins"], result["checks"] = run_checks()
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": "".join(traceback.format_exception(error))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
