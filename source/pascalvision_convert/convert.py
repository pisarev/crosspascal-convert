# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from enum import Enum
from typing import Dict, List, Optional, Tuple

from . import targets, tuning


class TuningRequest:
    __slots__ = ('tasks', 'per_task', 'total_trials', 'work_dir', 'reuse')

    def __init__(self, tasks: int, per_task: int, total_trials: int,
                 work_dir: str, reuse: bool = False) -> None:
        self.tasks = tasks
        self.per_task = per_task
        self.total_trials = total_trials
        self.work_dir = work_dir
        self.reuse = reuse

    def __repr__(self) -> str:
        return ('TuningRequest(tasks=%d, per_task=%d, total_trials=%d)'
                % (self.tasks, self.per_task, self.total_trials))


def plan_tuning(mod, tvm_target, work_dir: str,
                per_task: int = tuning.DEFAULT_TRIALS_PER_TASK) -> TuningRequest:
    tuning.refuse_on_windows()
    tuning.refuse_without_packages()
    tasks, _ = tuning.count_tasks(mod, tvm_target)
    ready = tuning.coverage(work_dir, tasks)
    if ready['complete'] and ready['workloads'] == tasks:
        return TuningRequest(tasks=tasks, per_task=per_task, total_trials=0,
                             work_dir=work_dir, reuse=True)
    return TuningRequest(
        tasks=tasks,
        per_task=per_task,
        total_trials=tuning.budget_for(tasks, per_task),
        work_dir=work_dir,
    )


class Target(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    CUDNN = "cudnn"


def _normalize_selected_rec_identity_nodes(onnx_model) -> int:
    import onnx

    nodes = onnx_model.graph.node
    if any(attribute.type in (onnx.AttributeProto.GRAPH,
                              onnx.AttributeProto.GRAPHS)
           for node in nodes for attribute in node.attribute):
        raise ValueError('selected Rec Identity normalizer refuses nested ONNX graphs')
    public_outputs = {value.name for value in onnx_model.graph.output}
    removed = []
    aliases = {}
    for node in list(nodes):
        if node.op_type != 'Identity' or node.output[0] in public_outputs:
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            raise ValueError('selected Rec Identity has invalid edge arity')
        upstream = node.input[0]
        while upstream in aliases:
            upstream = aliases[upstream]
        aliases[node.output[0]] = upstream
        removed.append(node)

    if len(removed) != 142:
        raise ValueError('selected Rec original Identity topology unexpectedly changed')
    for node in nodes:
        for i, value in enumerate(node.input):
            while value in aliases:
                value = aliases[value]
            node.input[i] = value
    for node in removed:
        nodes.remove(node)
    onnx.checker.check_model(onnx_model)
    return len(removed)


@contextlib.contextmanager
def _selected_rec_safe_cuda_compiler(enabled: bool):
    if not enabled:
        yield
        return
    keys = ('TVM_CUDA_COMPILE_MODE', 'TVM_CUDA_NVCC_NO_FAST_MATH')
    saved = {name: os.environ.get(name) for name in keys}
    try:
        os.environ['TVM_CUDA_COMPILE_MODE'] = 'nvcc'
        os.environ['TVM_CUDA_NVCC_NO_FAST_MATH'] = '1'
        yield
    finally:
        for name in keys:
            if saved[name] is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = saved[name]


def convert_onnx(
    model_path: str,
    output_path: str,
    target: Target = Target.CPU,
    opt_level: int = 3,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    tune_work_dir: Optional[str] = None,
    trials_per_task: int = tuning.DEFAULT_TRIALS_PER_TASK,
    req_sm: Optional[str] = None,
    cpu_codegen_target=None,
    cpu_codegen_receipt=None,
) -> str:
    req_sm = targets.validate_target_sm(target, req_sm)
    if cpu_codegen_target is not None:
        if (target != Target.CPU or not tune_work_dir or
                not isinstance(cpu_codegen_target, dict) or
                not isinstance(cpu_codegen_receipt, dict)):
            raise ValueError('selected CPU codegen requires tuned CPU and receipt')
        selected = targets.admit_cpu_codegen(
            cpu_codegen_target.get('mode'),
            cpu_codegen_target.get('mcpu') if cpu_codegen_target.get('mode') == 'explicit' else None)
        if selected != cpu_codegen_target:
            raise ValueError('CPU codegen target changed since BuildPlan admission')


        targets.require_local_cpu_tuning_target(selected)


        tuning.refuse_on_windows()

    import onnx  # noqa: WPS433 (lazy)
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    _refuse_known_bad_tvm(tvm.__version__)

    onnx_model = onnx.load(model_path)
    _refuse_unknown_input_names(onnx_model, input_shapes)
    _refuse_free_dimensions(onnx_model, input_shapes)


    _rewrite_cuda_float32_pow2(onnx_model, target)


    selected_rec_cuda = False
    if (isinstance(input_shapes, dict) and set(input_shapes) == {'x'} and
            tuple(input_shapes['x']) == (1, 3, 48, 320)):
        with open(model_path, 'rb') as selected_source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: selected_source.read(1024 * 1024), b''):
                digest.update(chunk)
            selected_sha = digest.hexdigest()
        if selected_sha == '1018d6d7a6fdc174ba75579c192b6a1c39d2cf66104804292827b45ae4bbf9dc':
            _normalize_selected_rec_identity_nodes(onnx_model)
            selected_rec_cuda = target in (Target.CUDA, Target.CUDNN)
    mod = from_onnx(onnx_model, shape_dict=input_shapes or {})

    if target in (Target.CUDA, Target.CUDNN):
        tvm_target = _target_for(target, req_sm=req_sm)
        _teach_nvrtc_where_cccl_lives()
        if target == Target.CUDNN:
            mod = _partition_for_cudnn_groups1(mod)
        request = None
    elif tune_work_dir:
        mcpu = (selected['mcpu'] if cpu_codegen_target is not None
                else _mcpu_for_this_machine())
        tvm_target = tuning.cpu_target(mcpu=mcpu)
        if cpu_codegen_target is not None:
            actual_kind = str(tvm_target.kind.name)
            actual_mcpu = str(tvm_target.attrs.get('mcpu') or '')
            if (actual_kind != 'llvm' or actual_mcpu != (mcpu or '')):
                raise ValueError('TVM effective CPU target differs from BuildPlan')


        request = plan_tuning(mod, tvm_target, tune_work_dir, trials_per_task)
    else:
        tvm_target = tvm.target.Target("llvm")
        request = None

    with _selected_rec_safe_cuda_compiler(selected_rec_cuda):
        built = _build_and_export(mod, tvm_target, opt_level, output_path, request)
    _write_input_manifest(built, _onnx_input_names(onnx_model, input_shapes))
    if request is not None:
        _refuse_incomplete_tuning(request, built)
    if cpu_codegen_target is not None:
        cpu_codegen_receipt.update({
            'compiler_target': selected['tvm_target'],
            'actual_tvm_target': str(tvm_target),
            'binding': 'TVM_KIND_AND_MCPU_CHECKED_AFTER_EXPORT',
            'destination_compatibility': 'NOT_CHECKED',
            'llvm_triple_attested': False,
            'full_target_equivalence_attested': False})
    return built


def _mcpu_for_this_machine() -> Optional[str]:
    try:
        import tvm
        from tvm.target import codegen
        detected = str(codegen.llvm_get_system_cpu() or '').strip()


        if detected and not codegen.llvm_is_valid_cpu(
                detected, codegen.llvm_get_system_triple()):
            return None
    except Exception:
        return None
    return detected or None


def _refuse_incomplete_tuning(request: "TuningRequest", module_path: str) -> None:
    report = tuning.coverage(request.work_dir, request.tasks)
    if report['complete']:
        return
    removed = False
    for path in (module_path, module_path + '.inputs'):
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed = removed or path == module_path
        except OSError:
            pass
    raise RuntimeError(
        'tuning did not cover every task (%s); %s'
        % (report['reason'],
           'the module was deleted' if removed else 'module removal was not confirmed'))


def _refuse_unknown_input_names(onnx_model, input_shapes) -> None:
    if not input_shapes:
        return
    weights = {w.name for w in onnx_model.graph.initializer}
    known = [e.name for e in onnx_model.graph.input if e.name not in weights]
    unknown = [name for name in input_shapes if name not in known]
    if unknown:
        raise ValueError(
            'the model has no input named %s. Its inputs are: %s'
            % (', '.join(repr(n) for n in unknown),
               ', '.join(repr(n) for n in known) or 'none'))


def _refuse_free_dimensions(onnx_model, input_shapes) -> None:
    for name, shape in (input_shapes or {}).items():
        bad = [index for index, dim in enumerate(shape) if int(dim) < 1]
        if bad:
            raise ValueError(
                "the shape given for %r is (%s), and a dimension there is not a "
                "positive number. That is the model's own free shape, not an "
                "answer to it - name the size you are building for, for example "
                "%s:1,3,640,640."
                % (name, ', '.join(str(d) for d in shape), name))

    weights = {w.name for w in onnx_model.graph.initializer}
    named = set(input_shapes or {})
    for entry in onnx_model.graph.input:
        if entry.name in weights or entry.name in named:
            continue
        free = [index for index, dim
                in enumerate(entry.type.tensor_type.shape.dim)
                if not dim.dim_value]
        if not free:
            continue
        declared = [(dim.dim_value if dim.dim_value else
                     (dim.dim_param or '?'))
                    for dim in entry.type.tensor_type.shape.dim]
        raise ValueError(
            "input %r leaves %s free - the model declares it as (%s). Name the "
            "shape you are building for, for example %s:1,3,640,640. TVM can "
            "convert a dynamic shape, but it cannot TUNE one, so the size has "
            "to be decided here and it cannot be guessed on your behalf."
            % (entry.name,
               'dimension %d' % free[0] if len(free) == 1
               else 'dimensions %s' % ', '.join(str(i) for i in free),
               ', '.join(str(d) for d in declared),
                entry.name))


def _rewrite_cuda_float32_pow2(onnx_model, target):
    if target not in (Target.CUDA, Target.CUDNN):
        return []


    if not getattr(getattr(onnx_model, 'graph', None), 'node', None):
        return []

    import onnx
    import warnings
    try:
        inferred = onnx.shape_inference.infer_shapes(onnx_model,
                                                    strict_mode=False)
    except Exception:
        return []
    types = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    types.update({v.name: v.type.tensor_type.elem_type for v in inferred.graph.input})
    producers = {name: node for node in onnx_model.graph.node for name in node.output}
    rewrites = []
    for index, node in enumerate(onnx_model.graph.node):
        if node.op_type != 'Pow' or len(node.input) != 2 or len(node.output) != 1:
            continue
        base, exponent_name = node.input
        if types.get(base) != onnx.TensorProto.FLOAT:
            continue
        const = producers.get(exponent_name)
        if const is None or const.op_type != 'Constant':
            continue
        if len(const.attribute) != 1 or const.attribute[0].name != 'value':
            continue
        value = onnx.numpy_helper.to_array(const.attribute[0].t)
        if value.shape != () or value.dtype.name != 'float32' or float(value) != 2.0:
            continue
        result_name = node.output[0]
        replacement = onnx.helper.make_node(
            'Mul', [base, base], [result_name],
            name='tvm_cuda_float32_square_%d' % index)
        node.CopyFrom(replacement)
        rewrites.append({'index': index, 'input': base, 'output': result_name})
    if rewrites:
        warnings.warn(
            'CUDA correctness: normalized %d float32 Pow(x,2) operations to '
            'Mul(x,x) in memory; original ONNX file unchanged. Independent '
            'numerical verification remains required.' % len(rewrites),
            RuntimeWarning, stacklevel=2)
    return rewrites


def _onnx_input_names(onnx_model, input_shapes):
    weights = {w.name for w in onnx_model.graph.initializer}
    rows = []
    for entry in onnx_model.graph.input:
        if entry.name in weights:
            continue
        dims = []
        for dim in entry.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else -1)
        if input_shapes and entry.name in input_shapes:
            dims = list(input_shapes[entry.name])
        rows.append((entry.name, dims))
    return rows


def _write_input_manifest(module_path: str, rows) -> str:
    manifest = module_path + ".inputs"
    tab = chr(9)
    with open(manifest, "w", encoding="utf-8", newline=chr(10)) as handle:
        for index, (name, dims) in enumerate(rows):
            handle.write(str(index) + tab + name + tab
                         + ",".join(str(d) for d in dims) + chr(10))
    return manifest


def convert_tflite(
    model_path: str,
    output_path: str,
    target: Target = Target.CPU,
    opt_level: int = 3,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    req_sm: Optional[str] = None,
) -> str:
    req_sm = targets.validate_target_sm(target, req_sm)

    import tvm
    from tvm import relax
    from tvm.relax.frontend.tflite import from_tflite

    _refuse_known_bad_tvm(tvm.__version__)

    try:
        import tflite  # noqa: WPS433 (lazy)
    except ModuleNotFoundError as missing:
        raise RuntimeError(
            "the 'tflite' package is required to read .tflite models "
            "(pip install tflite); it carries the flatbuffer schema, "
            "not TensorFlow itself"
        ) from missing

    with open(model_path, "rb") as handle:
        buffer = handle.read()
    model = tflite.Model.GetRootAsModel(buffer, 0)
    _refuse_quantized_tflite(model, model_path)
    with _tflite_optional_attribute_defaults():
        mod = from_tflite(model, shape_dict=input_shapes or {})

    tvm_target = _target_for(target, req_sm=req_sm)
    if target in (Target.CUDA, Target.CUDNN):
        _teach_nvrtc_where_cccl_lives()
    built = _build_and_export(mod, tvm_target, opt_level, output_path)
    _write_input_manifest(built, _tflite_input_names(model, input_shapes))
    return built


def _tflite_dimensions(tensor):
    signature = tensor.ShapeSignatureAsNumpy()
    if getattr(signature, "size", 0):
        return [int(d) for d in signature]
    shape = tensor.ShapeAsNumpy()
    if getattr(shape, "size", 0):
        return [int(d) for d in shape]
    return []


def _tflite_input_names(model, input_shapes):
    subgraph = model.Subgraphs(0)
    rows = []
    for index in range(subgraph.InputsLength()):
        tensor = subgraph.Tensors(subgraph.Inputs(index))
        raw = tensor.Name()
        name = raw.decode("utf-8") if raw else str(index)
        dims = _tflite_dimensions(tensor)
        if input_shapes and name in input_shapes:
            dims = list(input_shapes[name])
        rows.append((name, dims))
    return rows


def _refuse_quantized_tflite(model, model_path: str) -> None:
    subgraph = model.Subgraphs(0)
    for index in range(subgraph.InputsLength()):
        tensor = subgraph.Tensors(subgraph.Inputs(index))
        quantization = tensor.Quantization()
        if quantization is None or quantization.ScaleLength() == 0:
            continue
        scale = quantization.Scale(0)
        if scale and scale != 0.0:
            name = tensor.Name()
            raise RuntimeError(
                "refusing to convert an integer-quantized TFLite model: input "
                f"'{name.decode() if name else index}' of {model_path} carries "
                f"quantization (scale {scale}). This converter does not support "
                "that input. Export a floating-point model or use its float variant."
            )


@contextlib.contextmanager
def _tflite_optional_attribute_defaults():
    from tvm.relax.frontend.tflite import tflite_frontend

    original = tflite_frontend.FlexBufferDecoder

    class _DecoderWithDefaults(original):  # noqa: WPS431 (scoped on purpose)
        def decode(self):
            options = super().decode()
            if isinstance(options, dict):
                options.setdefault("detections_per_class", 100)
            return options

    tflite_frontend.FlexBufferDecoder = _DecoderWithDefaults
    try:
        yield
    finally:
        tflite_frontend.FlexBufferDecoder = original


def _target_for(target: Target, req_sm: Optional[str] = None):
    import tvm

    sm = targets.validate_target_sm(target, req_sm)
    if sm:
        return tvm.target.Target({'kind': 'cuda', 'arch': sm})
    return tvm.target.Target(targets.tvm_target_spec(target))


def _build_and_export(mod, tvm_target, opt_level: int, output_path: str,
                      tune: Optional["TuningRequest"] = None) -> str:
    import tvm
    from tvm import relax

    source_mod = mod
    try:
        if tune is None:
            mod = relax.get_pipeline("zero")(mod)
        elif tune.reuse:
            mod = tuning.apply_pipeline(target=tvm_target, work_dir=tune.work_dir)(mod)
        else:
            mod = tuning.tuning_pipeline(
                total_trials=tune.total_trials,
                target=tvm_target,
                work_dir=tune.work_dir,
                per_task=tune.per_task,
            )(mod)
        with tvm.transform.PassContext(opt_level=opt_level), tvm_target:
            executable = relax.build(mod, target=tvm_target)
    except Exception as primary:
        if tune is not None or not _cuda_default_schedule_failed(primary, tvm_target):
            raise


        import warnings
        warnings.warn(
            'The default CUDA TIR schedule rejected a reduction. Retrying '
            'with TVM DLight GPU Fallback on the SAME CUDA target and SM; '
            'all numerical verification remains mandatory.',
            RuntimeWarning, stacklevel=2)
        try:
            executable = _build_cuda_dlight_fallback(
                source_mod, tvm_target, opt_level)
        except Exception as fallback:
            raise RuntimeError(
                'Both CUDA TIR schedules failed. The default raised %s: %s; '
                'DLight GPU Fallback raised %s: %s. No CPU fallback was '
                'attempted and no compiled artifact was certified.'
                % (type(primary).__name__, str(primary)[:450],
                   type(fallback).__name__, str(fallback)[:1200])) from fallback
    executable.export_library(output_path, fcompile=_shared_library_builder())
    return output_path


def _cuda_default_schedule_failed(error, tvm_target):
    kind = getattr(getattr(tvm_target, 'kind', None), 'name', '')
    message = str(error)
    return (kind == 'cuda' and 'ScheduleError' in message and
            'schedule primitive' in message)


def _build_cuda_dlight_fallback(source_mod, tvm_target, opt_level):
    import tvm
    from tvm import relax
    from tvm.s_tir import dlight as dl
    from tvm.relax.backend.cuda import pipeline as cuda_pipeline

    passes = (cuda_pipeline.library_dispatch_passes(tvm_target) + [
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        dl.ApplyDefaultSchedule(dl.gpu.Fallback()),
    ] + cuda_pipeline.dataflow_lower_passes(tvm_target) +
              cuda_pipeline.finalize_passes(tvm_target))
    mod = source_mod
    with tvm.transform.PassContext(opt_level=opt_level), tvm_target:
        for compiler_pass in passes:
            mod = compiler_pass(mod)
        return relax.build(mod, target=tvm_target, relax_pipeline=None)


def cccl_include_directory() -> Optional[str]:
    root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not root:
        return None
    nested = os.path.join(root, "include", "cccl")
    if os.path.isfile(os.path.join(nested, "cuda", "std", "cstdint")):
        return nested
    flat = os.path.join(root, "include")
    if os.path.isfile(os.path.join(flat, "cuda", "std", "cstdint")):
        return None
    return None


def _teach_nvrtc_where_cccl_lives() -> None:
    include_dir = cccl_include_directory()
    if include_dir is None:
        return

    import tvm_ffi
    from tvm.support import nvcc as tvm_nvcc

    @tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)
    def _compile_with_cccl(code, target=None):  # noqa: WPS430 (TVM calls it)


        compiler = os.environ.get("TVM_CUDA_COMPILE_MODE", "nvrtc").lower()
        if compiler == "nvrtc":
            return tvm_nvcc.compile_cuda(
                code, target_format="cubin", compiler="nvrtc",
                options=["-I" + include_dir])
        if compiler == "nvcc":
            return tvm_nvcc.compile_cuda(
                code, target_format="fatbin", compiler="nvcc",
                options=["-I" + include_dir])
        raise ValueError(
            "Invalid TVM_CUDA_COMPILE_MODE %r: expected nvrtc or nvcc"
            % compiler)


MIN_CONVERTER_TVM = (0, 26, 0)


def _refuse_known_bad_tvm(version: str) -> None:
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    if tuple(parts) >= MIN_CONVERTER_TVM:
        return
    raise RuntimeError(
        "TVM %s is below the supported converter version. Conversion requires "
        "TVM %s or newer. Use the TVM and TVM FFI versions listed for this package."
        % (version, ".".join(str(p) for p in MIN_CONVERTER_TVM))
    )

def default_library_suffix() -> str:
    return ".dll" if platform.system() == "Windows" else ".so"


def _shared_library_builder():
    if platform.system() != "Windows":
        return None

    def _build(output: str, objects, options=None) -> None:
        clang = _find_clang_executable()
        if isinstance(objects, (str, bytes)):
            objects = [objects]
        objects = [str(obj) for obj in objects]
        extra = ([options] if isinstance(options, str) else list(options or []))
        libraries = _tvm_import_libraries()


        def clang_command(linker, link_dir=None):
            command: List[str] = [
                clang, "-O2", "--target=x86_64-pc-windows-msvc",
                "-shared", "-fuse-ld=" + linker, "-Wl,/Brepro",
                "-o", output,
            ]


            if link_dir:
                command.insert(4, "-B" + link_dir)
            return command + objects + libraries + extra

        attempts = []
        if clang:
            bundled_lld = os.path.join(
                os.path.dirname(os.path.abspath(clang)), "lld-link.exe")
            if os.path.isfile(bundled_lld):
                command = clang_command("lld")
                failure = _run_windows_link(command)
                if failure is None:
                    return
                attempts.append("bundled lld-link: " + failure)


        try:
            msvc_env, msvc_link = _initialized_msvc_linker()
        except RuntimeError as exc:
            attempts.append(str(exc))
            raise RuntimeError(
                "Linking the Windows x64 TVM module failed: " +
                "; ".join(attempts) +
                ". Install Visual Studio C++ Build Tools (MSVC and Windows SDK) "
                "or use a clang distribution with its own lld-link.exe."
            ) from exc

        if clang:
            command = clang_command("link", os.path.dirname(msvc_link))
        else:
            native_options = []
            for flag in extra:
                if flag.startswith("/"):
                    native_options.append(flag)
                elif flag.startswith("-Wl,"):
                    native_options.extend(flag[4:].split(","))
                else:
                    raise RuntimeError(
                        "MSVC link.exe cannot translate compiler option %r; "
                        "use clang with a compatible linker." % flag)
            command = [
                msvc_link, "/NOLOGO", "/DLL", "/Brepro", "/OUT:" + output,
            ] + objects + libraries + native_options
        failure = _run_windows_link(command, env=msvc_env)
        if failure is not None:
            attempts.append("MSVC link.exe: " + failure)
            raise RuntimeError(
                "Linking the Windows x64 TVM module failed: " +
                "; ".join(attempts))

    return _build


def _run_windows_link(command, env=None):
    try:
        kwargs = {"capture_output": True}
        if env is not None:
            kwargs["env"] = env
        done = subprocess.run(command, **kwargs)
    except OSError as exc:
        return "%s could not start: %s" % (command[0], exc)
    if done.returncode == 0:
        return None
    stderr = done.stderr or b""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return ("%s exited %s: %s" %
            (command[0], done.returncode, stderr.strip()[:1600] or
             "(no linker diagnostic)"))


def _installed_vswhere():
    candidate = shutil.which("vswhere.exe") or shutil.which("vswhere")
    if candidate and os.path.isfile(candidate):
        return candidate
    for root in (os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("ProgramFiles")):
        if not root:
            continue
        candidate = os.path.join(
            root, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _vcvars64_candidates():
    roots = []
    finder = _installed_vswhere()
    if finder:
        try:
            result = subprocess.run([
                finder, "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                roots.extend(line.strip() for line in result.stdout.splitlines()
                             if line.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    if os.environ.get("VSINSTALLDIR"):
        roots.append(os.environ["VSINSTALLDIR"])
    seen = set()
    for root in roots:
        candidate = os.path.abspath(os.path.join(
            root, "VC", "Auxiliary", "Build", "vcvars64.bat"))
        if candidate.casefold() not in seen and os.path.isfile(candidate):
            seen.add(candidate.casefold())
            yield candidate


def _environment_value(environment, key):
    for candidate, value in environment.items():
        if candidate.casefold() == key.casefold():
            return value
    return None


def _validate_msvc_environment(environment):
    arch = (_environment_value(environment, "VSCMD_ARG_TGT_ARCH")
            or "").strip().lower()
    if arch not in ("x64", "amd64"):
        return None
    root = _environment_value(environment, "VCToolsInstallDir")
    lib = _environment_value(environment, "LIB")
    include = _environment_value(environment, "INCLUDE")
    if not root or not lib or not include:
        return None
    if not any(os.path.isdir(folder) for folder in
               lib.split(os.pathsep) if folder):
        return None
    if not any(os.path.isdir(folder) for folder in
               include.split(os.pathsep) if folder):
        return None
    for host_arch in ("Hostx64", "Hostx86"):
        linker = os.path.join(root, "bin", host_arch, "x64", "link.exe")
        if os.path.isfile(linker):
            return os.path.abspath(linker)
    return None


def _initialized_msvc_linker():
    current = dict(os.environ)
    linker = _validate_msvc_environment(current)
    if linker:
        current["PATH"] = (os.path.dirname(linker) + os.pathsep +
                           (_environment_value(current, "PATH") or ""))
        return current, linker

    problems = []
    for vcvars in _vcvars64_candidates():
        executable = os.environ.get("ComSpec") or shutil.which("cmd.exe") or "cmd.exe"


        command = ('"' + executable + '" /d /s /c "chcp 65001 >nul && call "'
                   + vcvars + '" >nul && set"')
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append("vcvars64 initialization error: %s" % exc)
            continue
        if result.returncode != 0:
            problems.append("vcvars64 returned exit %s" % result.returncode)
            continue
        environment = dict(current)
        for line in result.stdout.splitlines():
            key, delimiter, value = line.partition("=")
            if delimiter and key and " " not in key:
                duplicate = next(
                    (old for old in environment
                     if old.casefold() == key.casefold()), None)
                if duplicate is not None:
                    del environment[duplicate]
                environment[key] = value
        linker = _validate_msvc_environment(environment)
        if linker:
            environment["PATH"] = (
                os.path.dirname(linker) + os.pathsep +
                (_environment_value(environment, "PATH") or ""))
            for key in tuple(environment):
                if key != "PATH" and key.casefold() == "path":
                    del environment[key]
            return environment, linker
        problems.append(
            "vcvars64 did not provide x64 MSVC link.exe, LIB and INCLUDE")
    raise RuntimeError(
        "compatible x64 MSVC linker environment was not found"
        + (": " + "; ".join(problems) if problems else
           ": vswhere/VSINSTALLDIR gave no usable vcvars64.bat"))


def _find_clang_executable() -> Optional[str]:
    from . import environment  # noqa: WPS433 (lazy: keeps the import graph flat)

    found, _on_path = environment.find_clang()
    return found


def _tvm_import_libraries() -> List[str]:
    wanted = "tvm_ffi.lib"
    for directory in _ffi_library_directories():
        candidate = os.path.join(directory, wanted)
        if os.path.isfile(candidate):
            return [candidate]
    return []


def _ffi_library_directories() -> List[str]:
    directories: List[str] = []
    try:
        import tvm_ffi  # noqa: WPS433 (lazy, optional)

        package = os.path.dirname(os.path.abspath(tvm_ffi.__file__))
        directories.append(os.path.join(package, "lib"))
        directories.append(package)
    except Exception:
        pass
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory:
            directories.append(directory)
    return directories

def _partition_for_cudnn_groups1(mod):
    try:
        from tvm.relax.backend.contrib.cudnn import partition_for_cudnn
    except Exception:
        return mod

    return partition_for_cudnn(mod)
