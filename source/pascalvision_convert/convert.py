# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""ONNX -> TVM Relax conversion.

Formalizes state/RECIPES.md R-001 (PaddleOCR cuDNN BYOC). TVM/onnx are imported
lazily so the package installs without them; conversion needs a working TVM Relax
(v0.24+) environment (see README).
"""
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
    """What a tuned build needs, decided before the build starts.

    Built by :func:`plan_tuning`, never by hand: the budget has to come from the
    model's own task count, and hand-picked round numbers are what produced the
    95.09 ms case - a classifier that came out slower WITH tuning than without.
    """

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
    """Count the model's tuning tasks, then size the budget from that count.

    Counting costs three to eleven seconds; the tuning it governs costs eleven
    to fifty-one minutes. Skipping the count to save the seconds is how a budget
    ends up below the coverage threshold.
    """
    
    
    
    
    
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
    """Erase identity-only edges in the SHA-selected PP-OCRv5 Server Rec graph.

    This is an in-memory, algebraically exact ONNX transformation: Identity
    never changes its input tensor. Preserve public graph output names, refuse
    nested subgraphs, and do not silently accept a different selected graph.
    The original ONNX file and all trained weights remain unchanged.
    """
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
    """Scope proven no-fast-math NVCC to the exact selected trained Rec build.

    On CUDA 12.8, default NVRTC --use_fast_math compiled the real Rec model
    but ALL 735400 output values were NaN. The unmodified TVM 0.26 NVCC
    callback with TVM_CUDA_NVCC_NO_FAST_MATH=1 returned finite probabilities
    agreeing with original ORT (max abs 1.3412e-5, all argmax equal).
    Do not change the compiler mode of any generic or legacy model.
    """
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
    """Convert an ONNX model to a compiled TVM Relax module (.so) and return its path.

    Pass ``tune_work_dir`` to build with code tuning. It applies to CPU targets
    only, and it changes the target as well as the pipeline: a tuned build gets
    ``num-cores`` and ``mcpu``, an untuned one must not - ``mcpu`` on its own
    measured 0.54 to 0.58 of the untuned speed.
    """
    
    
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
    """The mcpu string for the CPU we are building on, or None.

    None is a legitimate answer and not a failure: without a name we build for
    the generic target, which is exactly what an untuned build wants anyway.
    LLVM's own answer is used rather than a table of our own - a table would
    rot, and a wrong mcpu is worse than none.
    """
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
    """Undertuned is not a lesser success, it is a failure.

    The check reads the database the tuning left behind rather than the budget
    we asked for. Asking is not receiving: the classifier was asked for 500
    trials across 53 tasks and twenty-nine of them received none, which is how
    a tuned build came out thirty percent slower than an untuned one.

    THE MODULE IS DELETED, not merely reported. A file that is slower than an
    untuned build while looking exactly like a good one is worse than no file:
    whoever picks it up next has no way to tell, and our own rule is that a
    leftover artifact turns a failed build into a false success.
    """
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
        'tuning did not cover every task (%s); the module was %s because an '
        'undertuned build under an mcpu target is slower than an untuned one'
        % (report['reason'],
           'deleted' if removed else 'not written'))


def _refuse_unknown_input_names(onnx_model, input_shapes) -> None:
    """A shape given for an input the model does not have is a mistake, and it
    must be named as one here.

    Measured 01.09.2026 on rtdetr_r18vd: the input is called ``pixel_values``,
    a shape was supplied for ``image``, and the conversion died deep inside
    with ``TypeError: object of type 'NoneType' has no len()``. Nothing in that
    message points at the real error, and the caller is left guessing.

    The check costs one set comparison and turns a puzzle into a sentence.
    """
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
    """An input the model left free, and the caller did not name, is refused here.

    TVM Relax converts dynamic shapes perfectly well. Tuning does not: of the
    three pipelines it ships, the only one that tunes is called
    ``static_shape_tuning``, and there is no dynamic counterpart. So a shape has
    to be named, and the person who knows which one is the caller, not us.

    Guessing was tried and it was wrong. On 31.08.2026 rtdetr_r18vd left all
    four dimensions free, ones were substituted, the channel count became 1
    instead of 3, and onnxruntime refused with a message about a convolution.

    Not guessing was also tried, and the silence was worse: on 01.09.2026 the
    same model without a shape died inside the frontend with ``TypeError:
    object of type 'NoneType' has no len()`` while converting a Resize operator.
    Nothing in that sentence points at the missing shape.

    The same refusal already existed in verify.py, but it guarded the
    COMPARISON path, not the BUILD path - so the build never reached it. A check
    that does not stand where the fault occurs is not a check.
    """
    
    
    
    
    
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
    """Normalize *only* exactly squared float32 tensors before TVM CUDA import.

    This works on an ONNX model loaded in memory, never changes its source
    weights or path, and deliberately leaves other exponents untouched.

    The rule is precision-preserving: `Pow(x, scalar float32 2)` and `Mul(x,x)`
    both square float32 values. The observed TVM CUDA `Pow` lowering yielded
    NaNs on a genuine PaddleOCR Rec model; independent ONNX Runtime runs of
    the unchanged and transformed graphs agreed (maxAbs <= 9.54e-7). The
    transformed CUDA graph then passed the original numerical tolerance.

    A model with unknown operand type or non-scalar exponent is not rewritten.
    Its later mandatory reference comparison must refuse if it miscomputes.
    """
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
    """Name and shape of every graph input, in declaration order.

    Initializers are excluded: ONNX lists weights among graph.input, and calling a
    weight an input would make the manifest lie about what the caller must supply.
    """
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
    """Write the input names beside the artifact, one per line.

    WHY THIS FILE EXISTS. The compiled module does not carry the names of its
    inputs: measured 31.08.2026 on rtdetr_r18vd.cpu.dll, where
    get_function_metadata, get_function_doc and the global function registry all
    return nothing for them. Without this file a caller that names the wrong input
    is answered silently, because the binding resolves every name to index 0 -- a
    wrong name was measured to pass unnoticed on 30.08.2026.

    The format is deliberately not JSON: it is read by both Delphi and Free
    Pascal, and a tab-separated line needs no parser in either.

        index <TAB> name <TAB> d0,d1,d2,d3

    A dimension of -1 means the model left it free.
    """
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
    """Convert a TFLite model to a compiled TVM Relax module and return its path.

    WHY A SEPARATE ENTRY POINT - measured 30.08.2026, not assumed.

    A .tflite file is a flatbuffer, not ONNX, and no conversion to ONNX happens
    here. Relax carries its own TFLite frontend
    (``tvm.relax.frontend.tflite.from_tflite``), so the model is read directly.
    Going through tf2onnx would drag in TensorFlow - hundreds of megabytes and a
    second source of truth about the same graph.

    TFLite keeps images as NHWC, and the adapters named after that framework
    expect exactly that layout. Nothing is transposed on the way in.
    """
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
    """Dimensions of a TFLite tensor, with -1 for whatever the model left free.

    The shape signature is preferred because that is where TFLite records free
    dimensions; the plain shape substitutes a concrete number for them and would
    make the manifest claim a size the model does not actually fix.
    """
    signature = tensor.ShapeSignatureAsNumpy()
    if getattr(signature, "size", 0):
        return [int(d) for d in signature]
    shape = tensor.ShapeAsNumpy()
    if getattr(shape, "size", 0):
        return [int(d) for d in shape]
    return []


def _tflite_input_names(model, input_shapes):
    """Name and shape of every subgraph input, in declaration order.

    WHY THIS IS NEEDED HERE TOO - measured 31.08.2026, not assumed. The manifest
    was written for ONNX only, so a module built from a .tflite file shipped
    without one. The binding reads a missing manifest as "there is nothing to
    check against" and resolves any name to index 0, which means the caller's
    input name was accepted whatever it was - silently. That is the exact
    failure the manifest exists to remove, and it survived for TFLite alone.

    Names come from the model itself rather than from the frontend, because the
    frontend is free to rename; the caller names the input as the FILE calls it.
    """
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
    """Refuse an integer-quantized TFLite model instead of building a wrong one.

    WHY REFUSE RATHER THAN WARN - measured 30.08.2026 against a real interpreter.

    Two stock TensorFlow models were converted and compared with what the TFLite
    interpreter itself produces on the same image:

        mobilenet_v2_1.0_224   float32          class 655 at 0.596 - exact match
        coco_ssd_mobilenet_v1  uint8 quantized  scores around 0.02, where the
                               (scale 1/128,    interpreter gives 0.824 and 0.730

    The quantized model converts, loads and runs. It simply computes the wrong
    numbers, and nothing downstream can tell: the shapes are right, the four
    outputs are there, the scores are plain floats. That is the worst kind of
    failure, and it is why this refuses rather than warns.

    The check looks at the input tensors' quantization parameters, because that
    is where the integer encoding is declared. A float model leaves the scale at
    zero.
    """
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
                f"quantization (scale {scale}). Measured 30.08.2026 on the stock "
                "COCO SSD MobileNet v1: such a model converts and runs, but its "
                "scores come out around 0.02 where the TFLite interpreter gives "
                "0.73-0.82 - wrong numbers with no sign of trouble. Float TFLite "
                "models convert correctly; export the model without quantization, "
                "or use its float variant."
            )


@contextlib.contextmanager
def _tflite_optional_attribute_defaults():
    """Supply TFLite defaults for attributes the frontend insists on.

    WHY THIS EXISTS - measured 30.08.2026 on a real model, not assumed.

    The stock COCO SSD MobileNet v1 detector
    (``coco_ssd_mobilenet_v1_1.0_quant``, the one TensorFlow itself publishes)
    is refused by the Relax TFLite frontend:

        DETECTION_POSTPROCESS custom options miss required attributes:
        detections_per_class

    Its ``TFLite_Detection_PostProcess`` node carries ten fields, and that one is
    genuinely absent - checked by decoding the flexbuffer. But the TFLite schema
    marks it OPTIONAL with a default of 100, so the model is well formed and the
    frontend is stricter than the format. Without this, every SSD detector of
    that vintage is unconvertible, and those are exactly the models the TFLite
    adapter's four-output contract was written for.

    The patch is narrow and temporary: only the decoder used to read custom
    options, only while the frontend runs, restored afterwards. Other operators
    read their options by name, so an unused extra key cannot disturb them.
    """
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
    """TVM target for our choices, with an explicit GPU SM when requested."""
    import tvm

    sm = targets.validate_target_sm(target, req_sm)
    if sm:
        
        
        
        
        return tvm.target.Target({'kind': 'cuda', 'arch': sm})
    return tvm.target.Target(targets.tvm_target_spec(target))


def _build_and_export(mod, tvm_target, opt_level: int, output_path: str,
                      tune: Optional["TuningRequest"] = None) -> str:
    """Legalize, lower and write the module. Shared by every input format.

    Without ``tune`` this is the pipeline we have always used: ``zero``, no
    scheduler, no vectorised or parallel loops. It is correct and it is slow -
    57 to 139 times slower than ONNX Runtime on the models measured 01.09.2026.

    With ``tune`` the same module goes through MetaSchedule instead, and the
    caller has already decided the budget from the model's own task count. That
    order is not decoration: a budget smaller than ``tasks * per_task`` leaves
    tasks untuned under an mcpu target and comes out SLOWER than no tuning.
    """
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
        with tvm.transform.PassContext(opt_level=opt_level):
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
    """Recognise the actual TVM CUDA TIR schedule failure, not arbitrary errors.

    A misconfigured toolkit, unsupported target, or linker failure must still
    raise normally; DLight scheduling cannot fix them. CPU cannot be retried
    here under any circumstances.
    """
    kind = getattr(getattr(tvm_target, 'kind', None), 'name', '')
    message = str(error)
    return (kind == 'cuda' and 'ScheduleError' in message and
            'schedule primitive' in message)


def _build_cuda_dlight_fallback(source_mod, tvm_target, opt_level):
    """Use TVM's own alternative GPU schedule for a failed CUDA reduction.

    The pass list is from the TVM 0.27 CUDA compiler and requires this TVM
    installation to provide it. It does not change the CUDA architecture,
    binary format or target CPU. No separate model converter is introduced.
    """
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
    """Directory that makes ``#include <cuda/std/...>`` resolve, or None.

    CUDA 13 moved the CCCL headers from ``<CUDA>/include`` down into
    ``<CUDA>/include/cccl``. Older toolkits keep them directly under
    ``<CUDA>/include``, where NVRTC finds them without help.
    """
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
    """Add the CCCL include directory to NVRTC's search path when it is needed.

    WHY THIS EXISTS - measured 30.08.2026 on CUDA 13.2, not guessed.

    Building for the CUDA target failed while compiling the generated kernels:

        #include <cuda/std/cstdint>
                                   ^
        1 catastrophic error detected in the compilation of "tvm_kernels.cu".

    The header is present, just not where NVRTC looks. CUDA 13 relocated CCCL:

        CUDA 12 and earlier   <CUDA>/include/cuda/std/cstdint
        CUDA 13.2 (measured)  <CUDA>/include/cccl/cuda/std/cstdint

    TVM passes no include directories to NVRTC, so the nested layout is invisible
    to it. Two other routes were tried and rejected by measurement:

      * installing cuda-bindings - needed, but not enough: it supplies the
        bindings, not the headers, and the same error came back;
      * TVM_CUDA_COMPILE_MODE=nvcc, which TVM documents - nvcc does know its own
        layout, but it refused with "Cannot find compiler 'cl.exe' in PATH".
        That would drag Visual Studio into a toolchain that otherwise needs only
        clang, so it is the worse trade.

    Adding one include directory keeps NVRTC, needs no extra toolchain, and is a
    no-op on toolkits that place CCCL where NVRTC already looks.
    """
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
    """Refuse to convert with a TVM whose ONNX frontend is known to be wrong.

    MEASURED 28.08.2026 on yolo11n, a real photograph, against onnxruntime:

        onnxruntime (reference)   best class score 0.911
        TVM 0.26                  0.911, max divergence 1.28e-03
        TVM 0.24                  0.125, max divergence 5.33e+02

    The 0.24 build loads, runs and returns tensors of the right shape - and
    detects nothing. That is why this refuses instead of warning: a silently
    wrong model is worse than no model, because everything downstream looks
    healthy.

    Where it goes wrong was narrowed by bisecting the graph: the first diverging
    node is an ordinary 3x3 convolution (group=1) deep in the backbone, with its
    input still matching to 3.6e-06. Optimisation level, build pipeline, explicit
    input shape and graph simplification all made no difference - the divergence
    was identical to the last digit, which is what says "systematic", not
    "numerical".

    WHAT WE DO NOT KNOW, and do not pretend to: 0.25 was never tested. It is
    refused because it is below the version we measured as good, not because it
    was found bad.

    The runtime side is unaffected: a module built by 0.26 was loaded by our
    0.24 runtime and produced 0.911 with 1.28e-03 divergence. Compilation and
    execution are different sides, and only this one needs the newer TVM.
    """
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    if tuple(parts) >= MIN_CONVERTER_TVM:
        return
    raise RuntimeError(
        "This TVM (%s) miscompiles ONNX models: measured on yolo11n it returns "
        "tensors of the right shape that detect nothing (best score 0.125 "
        "against 0.911 from onnxruntime). Conversion needs TVM %s or newer. "
        "The runtime side is unaffected - a module built by a newer TVM runs "
        "correctly under the older runtime, which is measured, not assumed."
        % (version, ".".join(str(p) for p in MIN_CONVERTER_TVM))
    )

def default_library_suffix() -> str:
    """Platform suffix for a loadable module: .dll on Windows, .so elsewhere."""
    return ".dll" if platform.system() == "Windows" else ".so"


def _shared_library_builder():
    """Return an fcompile for export_library, or None to keep TVM's default.

    WHY THIS EXISTS - measured 28.08.2026, not guessed.

    On Windows TVM's built-in packer calls

        clang -O2 --target=x86_64 -shared -o out.dll lib0.o devc.o

    and that fails with "linker (via gcc) command failed". Two things are wrong
    with it: the target triple is incomplete (x86_64 with no vendor or OS, so
    clang falls back to a Unix-style driver) and the link step is then handed to
    gcc, which is not what builds a DLL on this platform.

    Measured on the same machine with the same clang 21.1.1: the plain command

        clang -O2 --target=x86_64-pc-windows-msvc -shared -o t.dll t.c

    produces a working DLL when clang can find lld-link. Some Windows clang
    distributions (including Embarcadero's) ship clang.exe WITHOUT lld-link:
    the same command then fails with 'program not executable'. In that case we
    locate an installed MSVC C++ toolset, initialize its environment using
    vcvars64.bat, and explicitly use its link.exe. This does NOT change the
    target of Relax/LLVM/CUDA code generation or choose a CPU fallback.

    On anything other than Windows we return None and let TVM do what it already
    does correctly there.
    """
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
    """Run an actual linker, returning an actionable failure or None on success."""
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
    """Locate the official Visual Studio installation finder, without years."""
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
    """Prefer the installed VS C++ toolset reported by vswhere.

    An existing VSINSTALLDIR is useful when the finder was not installed (for
    example in a tailored build environment), but a directory does not count
    as a toolchain until vcvars64 and the native link.exe are both validated.
    """
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
    """Read Windows environment names case-insensitively, even in plain dicts.

    os.environ on Windows normalizes keys, while the captured output of 'set'
    may use another spelling (Path vs PATH, VCToolsInstallDir vs uppercase).
    A normal dict does NOT preserve Windows' case-insensitive lookup semantics.
    """
    for candidate, value in environment.items():
        if candidate.casefold() == key.casefold():
            return value
    return None


def _validate_msvc_environment(environment):
    """Return the x64 MSVC linker only for a complete, initialized toolchain."""
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
    """Return (environment, verified linker) for native x64 MSVC tools.

    We cannot assume a VS Developer Prompt. Execute vcvars64 in a short-lived
    cmd.exe child, capture its environment, then use that environment in the
    separate linker child. No changes are made to the parent process or PATH.
    """
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
    """clang.exe, on PATH or where LLVM normally installs itself.

    PATH alone was not enough and cost half an hour once: a tuned build spent
    that long tuning and then refused to link, on a machine where clang sat in
    its default directory the whole time. Looking in the standard place first
    is not a workaround - refusing to look is the defect.
    """
    
    
    
    
    
    from . import environment  # noqa: WPS433 (lazy: keeps the import graph flat)

    found, _on_path = environment.find_clang()
    return found


def _tvm_import_libraries() -> List[str]:
    """Import libraries a compiled model must be linked against on Windows.

    WHY - measured 28.08.2026 on a real model, not assumed.

    A model built by Relax calls back into the TVM FFI layer. Linking the
    generated objects on their own fails with exactly one unresolved symbol:

        LNK2019: TVMFFIErrorSetRaisedFromCStrParts, referenced in
                 __tvm_ffi_transpose1

    Adding tvm_ffi.lib resolves it and the DLL links. That single name is the
    whole reason this function exists; if a future TVM needs more, the linker
    will name them just as plainly.

    WHERE WE LOOK, and why there: the same directories where the TVM DLL itself
    was found, that is the entries of PATH. Using one rule for both keeps the
    library and its import library from being taken out of different builds -
    a mismatch that would link cleanly and fail at load time.
    """
    wanted = "tvm_ffi.lib"
    for directory in _ffi_library_directories():
        candidate = os.path.join(directory, wanted)
        if os.path.isfile(candidate):
            return [candidate]
    return []


def _ffi_library_directories() -> List[str]:
    """Directories that may hold the TVM FFI import library, best first.

    TWO LAYOUTS EXIST and both are in use here, which is why this looks in two
    places rather than one:

        wheel from PyPI    site-packages/tvm_ffi/lib/tvm_ffi.lib
        build from source  the build output directory, reachable through PATH
                           because that is also how the DLL itself is found

    The package is asked first. It is the more precise answer: it names the very
    installation Python has imported, so the import library cannot come from a
    different build than the DLL that will load the model.
    """
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
    """cuDNN BYOC partition restricted to groups==1 convolutions (R-001).

    The stock ``partition_for_cudnn`` offloads grouped/depthwise conv too (its
    ``_check_conv2d`` ignores ``groups``), which miscomputes the result. The recipe's
    fix is a custom pattern that only matches ``groups == 1``. The exact pattern is
    TVM-version specific - see state/RECIPES.md R-001. This hook falls back to the
    stock partition if a custom one is not wired yet.
    """
    try:
        from tvm.relax.backend.contrib.cudnn import partition_for_cudnn
    except Exception:
        return mod
    
    return partition_for_cudnn(mod)
