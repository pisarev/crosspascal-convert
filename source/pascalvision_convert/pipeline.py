# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""The core: recognise -> build -> place the runtime -> compare with the reference.

WHY A CORE RATHER THAN A RETELLING IN EACH SHELL. The same job is presented by
two faces: the build recipe on somebody else's machine and the wizard on this
one. While the order of steps lived in the recipe's shell script, only the recipe
knew it, and the wizard would have had to retell it in its own words - a second
account of one job.

THIS IS ARITHMETIC, NOT A WORRY. On 31.08.2026 the recipe turned out to hold
seven defects, and THREE of them lived precisely in that shell text:

    the TFL3 marker was looked for in the first four bytes, and it sits at 4;
    the output name was assembled without an extension, and the loader will
        not take such a file;
    the linker's leftovers, .exp and .lib, matched the module's name pattern.

Each of the three would have repeated in the wizard word for word had it retold
the recipe. Here they are fixed ONCE.

The same disease is described at the head of our own test builder: there were
twenty-six recipes, each carrying its copy of the module path list; the list
moved and nothing built at all. What rots is not the path, it is the copying.

PRINTS NOTHING. Returns a report; printing belongs to whoever shows it.
"""

import datetime
import hashlib
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path

from . import detect
from . import build_plan
from . import runtime
from . import targets
from . import verify
from .contract import (
    build_vision_contract, requires_adapter_output_check,
    validate_adapter_outputs, requires_paddleocr_output_check,
    validate_paddleocr_outputs, validate_vision_adapter,
)
from .manifest import emit_manifest_json, emit_manifest_text
from .recipe_file import build_recipe, write_recipe
from .report import Report

MODULE_EXTENSION = {
    'Windows': '.dll',
    'Darwin': '.dylib',
}
DEFAULT_MODULE_EXTENSION = '.so'





ARTIFACT_DTYPE = 'float32'




LINKER_LEFTOVERS = ('.exp', '.lib')






MAX_URL_MODEL_BYTES = 64 * 1024 ** 3
MIN_URL_DISK_HEADROOM_BYTES = 256 * 1024 ** 2
URL_READ_BLOCK_BYTES = 1024 * 1024


def module_extension():
    return MODULE_EXTENSION.get(platform.system(), DEFAULT_MODULE_EXTENSION)


def _native_artifact_arch():
    arch = targets._norm_arch(platform.machine())
    if arch not in ('x86_64', 'arm64'):
        raise ValueError('unsupported native CPU ISA %r for an auditable '
                         'CrossPascal artifact passport' % platform.machine())
    return arch


def _unsafe_output_directory(directory):
    """Detect linked ancestors without following them into unrelated trees.

    A user-supplied destination through a symlink or NTFS junction can redirect
    a supposedly isolated build into a different user's directory. Do not
    resolve the link (that would hide it); check each lexical ancestor before
    any mkdir, native compiler import or output write.
    """
    path = Path(directory).absolute()
    for member in (path, *path.parents):
        if member.is_symlink() or getattr(member, 'is_junction', lambda: False)():
            return str(member)
    return None


def parse_shape(text):
    """A shape out of "images:1,3,640,640" or "1,3,640,640".

    The name is cut at the LAST colon: inputs of TensorFlow models end in ":0"
    themselves, and splitting at the first colon tore the name in half.
    """
    if not text:
        return None, None
    name, _, dims = text.rpartition(':')
    shape = tuple(int(d) for d in dims.split(',') if d.strip())
    return (name or None), shape


def download(url, target_dir):
    """Fetch with exact bounded bytes and exclusive no-clobber publication.

    The partially written newly owned destination is RETAINED on failure.
    It never replaces an existing model.bin or follows a redirected output
    directory. This bounds transport, not selected compiler-source custody.
    """
    import urllib.request
    path = os.path.join(target_dir, 'model.bin')
    redirected = _unsafe_output_directory(target_dir)
    if redirected:
        raise ValueError('URL model output directory resolves through '
                         'a symlink/junction: ' + redirected)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError('URL model download needs an existing output '
                                'directory: ' + str(target_dir))
    if os.path.lexists(path):
        raise FileExistsError('URL model.bin output already exists; refusing '
                              'to clobber owner bytes: ' + path)
    available = shutil.disk_usage(target_dir).free
    if available <= MIN_URL_DISK_HEADROOM_BYTES:
        raise ValueError('insufficient physical disk free space for safe '
                         'URL model download headroom')
    quota = min(MAX_URL_MODEL_BYTES,
                available - MIN_URL_DISK_HEADROOM_BYTES)
    if quota < 1:
        raise ValueError('URL model download byte quota is unavailable')
    with urllib.request.urlopen(url, timeout=300) as answer:
        headers = getattr(answer, 'headers', {})
        declared_raw = (headers.get('Content-Length')
                        if callable(getattr(headers, 'get', None)) else None)
        declared = None
        if declared_raw is not None:
            encoded = str(declared_raw).strip()
            if not re.fullmatch(r'[0-9]+', encoded):
                raise ValueError('malformed URL model Content-Length')
            declared = int(encoded)
            if declared <= 0:
                raise ValueError('invalid zero-size URL model Content-Length')
            if declared > quota:
                raise ValueError('URL model Content-Length exceeds admitted '
                                 'disk/transport byte quota')
        
        
        
        
        with open(path, 'xb') as handle:
            total = 0
            while True:
                amount = min(URL_READ_BLOCK_BYTES, quota - total + 1)
                block = answer.read(amount)
                if not block:
                    break
                if not isinstance(block, bytes) or len(block) > amount:
                    raise ValueError('URL model body violated bounded byte '
                                     'read contract')
                if (len(block) > quota - total or
                        (declared is not None and
                         len(block) > declared - total)):
                    raise ValueError('URL model streamed body exceeded '
                                     'Content-Length or bounded byte quota')
                handle.write(block)
                total += len(block)
            if total == 0:
                raise ValueError('empty URL model response cannot be compiled')
            if declared is not None and total != declared:
                raise ValueError('URL model body ended short of Content-Length')
            handle.flush()
            os.fsync(handle.fileno())
    return path


def _remove_leftovers(base):
    """Drop the .exp and .lib the linker left behind."""
    removed = []
    for suffix in LINKER_LEFTOVERS:
        candidate = base + suffix
        if os.path.isfile(candidate):
            os.remove(candidate)
            removed.append(os.path.basename(candidate))
    return removed


def _sha256_file(path):
    """The SHA-256 hex digest of a file, read in blocks so a large module is not
    pulled into memory whole. Only the standard library is used - the converter's
    promise that nothing has to be installed reaches down to here too."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _toolchain_versions():
    """The TVM and ONNX versions that produced the module - best effort.

    NEVER RAISES. A package that will not import records its field as ``unknown``
    rather than sinking a build that already succeeded (the passport is issued for
    a model that passed, and a missing version does not unmake that). tvm.runtime
    is added only when it advertises its own __version__; in TVM 0.26 it does not,
    so the field is simply left out rather than guessed.
    """
    versions = {}
    try:
        import tvm
        versions['tvm'] = getattr(tvm, '__version__', None) or 'unknown'
    except Exception:
        versions['tvm'] = 'unknown'
    try:
        import onnx
        versions['onnx'] = getattr(onnx, '__version__', None) or 'unknown'
    except Exception:
        versions['onnx'] = 'unknown'
    try:
        import tvm.runtime as tvm_runtime
        runtime_version = getattr(tvm_runtime, '__version__', None)
        if runtime_version:
            versions['tvm_runtime'] = runtime_version
    except Exception:
        pass
    return versions


def _utc_now_iso():
    """The current moment as ISO 8601 UTC, e.g. 2026-09-06T12:00:00Z - seconds
    precision, a trailing Z rather than +00:00, so it reads as one token."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime('%Y-%m-%dT%H:%M:%SZ')


def _gather_provenance(built, model_path, source, source_closure=None):
    """The audit facts the passport and the recipe share, gathered ONCE so the two
    files agree to the byte: the built module's fingerprint, the source as recorded
    and its own fingerprint when it is a local file, the toolchain, and the moment.

    NEVER RAISES. Each fact is best effort - a fingerprint that cannot be taken is
    recorded as None (and then omitted downstream), a version that will not import
    as ``unknown`` - because this feeds the recipe, which is written for every
    build that got this far, success or not; a stumble here must not unmake it.
    """
    facts = {
        'artifact_hash': _try(lambda: _sha256_file(built)),
        'source': source if source is not None else model_path,
        'source_hash': (_try(lambda: _sha256_file(model_path))
                        if os.path.isfile(model_path) else None),
        'built_at': _utc_now_iso(),
        'toolchain': _toolchain_versions(),
    }
    
    
    
    
    if source_closure is not None:
        facts['source_closure'] = source_closure
    return facts


def _try(thunk):
    """Run *thunk*, returning its value or None if it raised - the best-effort
    wrapper the provenance facts use so none of them can sink a finished build."""
    try:
        return thunk()
    except Exception:
        return None


def _require_selected_cpu_codegen_route(plan, tuned_onnx_cpu):
    """Reject an unimplemented CPU target route before any TVM compilation."""
    if 'cpu_codegen' in plan and not tuned_onnx_cpu:
        raise ValueError('selected CPU codegen requires ONNX tuned CPU '
                         'compilation; this model route cannot honour it')


def run(model_path, target_dir, target='cpu', shape=None, module_name='model',
        on_step=None, tune_work_dir=None, trials_per_task=None, contract=None,
        req_sm=None, source=None, license=None, req_os=None, req_arch=None,
        req_driver_api=None, approved_build_plan=None, cpu_codegen=None,
        approved_source_closure=None, approved_paddle_source=None):
    """Walk the whole path and return a report.

    model_path       a file on disk (already downloaded)
    target_dir       where the module, runtime and licence go
    target           cpu or cuda
    shape            "images:1,3,640,640" or None
    source           the model as the user named it - a URL or a path - for the
                     build recipe to record, so a repeat fetches the model afresh
                     rather than reaching for a local download that may be gone.
                     None means the recipe records model_path itself, which is
                     right when the caller was handed a local file to begin with.
    license          the source model's licence (an SPDX id or a name), recorded
                     in the passport and the recipe, or None to leave it out. It
                     is the one audit fact the build cannot discover for itself -
                     provenance, the hashes and the toolchain are gathered here;
                     the licence is whatever the caller was told.
    tune_work_dir    tune the CPU build, keeping the database here; None to skip
    trials_per_task  trials each tuning task gets; the total budget is that
                     figure times the model's own task count
    contract         the model-contract input from the recipe, or None. When it
                     names a task, a bundle manifest (manifest.txt + manifest.json)
                     is written beside the module after a successful build; when
                     it is None or names no task, nothing is emitted and the run
                     behaves exactly as it always has. See _emit_manifest.
    req_sm           the CUDA architecture to pass to the TVM compiler (e.g.
                     "sm_89"), or None to resolve from the measured host GPU.
                     A CUDA build without either an explicit or a measured SM
                     is refused before compilation; it must not silently use
                     an unspecified host GPU. Supplying an SM for CPU refuses.
    req_os, req_arch  optional deployment OS/CPU architecture overrides.
                     Defaults are measured from this build host. Every new
                     build records the same versioned target_profile in recipe
                     v2; older recipe v1 files remain readable for replay.
    req_driver_api    optional CUDA Driver API version of the destination.
                      This is a deployment requirement, not an installed
                      compiler version; it is recorded in the profile and
                      checked before compiling for that destination.

    Tuning is off by default and that is deliberate. It costs eleven to fifty
    minutes of machine time against five seconds for an untuned build, and the
    person who just wants to see whether a model converts at all should not pay
    that. What it buys, measured 01.09.2026: 32 to 58 times faster on CPU.
    """
    report = Report('Converting a model to the TVM format')

    def finished(step):
        if on_step:
            on_step(step)
        return step

    try:
        if str(target).strip().lower() not in ('cpu', 'cuda'):
            raise ValueError('the pipeline target must be cpu or cuda, got %r'
                             % target)
        req_sm = targets.validate_target_sm(target, req_sm)
    except ValueError as trouble:
        finished(report.step('Checking target reachability').refuse(str(trouble)))
        return report
    step = report.step('Recognising the model')
    try:
        recognised = detect.recognise(model_path)
    except OSError as trouble:
        finished(step.refuse('the file cannot be read: %s' % trouble))
        return report
    step.say('recognised: %s' % recognised.detail)
    
    
    
    if os.path.isfile(model_path):
        step.say('size %.1f MB' % (os.path.getsize(model_path) / 1048576.0))
    step.numbers['kind'] = recognised.kind
    step.numbers['convertible'] = recognised.convertible
    if not recognised.convertible:
        for line in recognised.advice:
            step.say(line)
        if os.path.isfile(model_path):
            step.say('first bytes: %s' % detect.first_bytes(model_path))
        finished(step.refuse('this file is not something we can compile'))
        return report
    finished(step.done())

    source_closure = None
    paddle_selected = None
    if approved_paddle_source is not None:
        check = report.step('Checking selected original Paddle model receipt')
        try:
            from . import paddle_closure
            from .recipe_file import validate_paddle_selected_source
            paddle_selected = validate_paddle_selected_source(
                approved_paddle_source, source=source)
            if (source_closure is not None or
                    approved_source_closure is not None or
                    recognised.kind != detect.ONNX or
                    not os.path.isfile(model_path)):
                raise ValueError('Paddle selected receipt requires canonical '
                                 'exported ONNX, not a different source format')
            observed = paddle_closure.audit_selected_paddle_source(
                source, role=paddle_selected['role'],
                dictionary=(paddle_selected['dictionary'] or {}).get('reference'),
                expected_tree_sha256=paddle_selected['tree_sha256'])
            if observed != paddle_selected:
                raise ValueError('Paddle selected receipt differs before native build')
            check.say('exact selected original Paddle SHA %s'
                      % paddle_selected['tree_sha256'])
            check.say('NOT_CHECKED: original scan does not hold compiler input')
        except (OSError, ValueError, TypeError, KeyError) as trouble:
            finished(check.refuse(str(trouble)))
            return report
        finished(check.done())
        report.summary['paddle_selected_source'] = {
            'tree_sha256': paddle_selected['tree_sha256'],
            'role': paddle_selected['role'],
            'verification_status': 'NOT_CHECKED',
            'release_grade_immutable': False}
    if approved_source_closure is not None:
        check = report.step('Checking selected ONNX external source closure')
        try:
            from . import onnx_closure
            from .recipe_file import validate_selected_source_closure

            
            
            
            
            source_closure = validate_selected_source_closure(
                approved_source_closure)
            if (recognised.kind != detect.ONNX or source is None or
                    os.path.abspath(str(source))
                    != source_closure['source_model']):
                raise ValueError('approved ONNX source closure has different '
                                 'recognised format/original provenance')
            stage = onnx_closure.audit_selected_closure(
                os.path.dirname(os.path.abspath(model_path)),
                expected_tree_sha256=source_closure['tree_sha256'])
            if (os.path.normcase(os.path.abspath(stage['compiler_input']))
                    != os.path.normcase(os.path.abspath(model_path)) or
                    stage['model_rel'] != source_closure['model_rel'] or
                    stage['dependencies'] != source_closure['dependencies'] or
                    stage['file_count'] != source_closure['file_count'] or
                    stage['total_bytes'] != source_closure['total_bytes']):
                raise ValueError('private ONNX/sidecar stage differs from approved '
                                 'external dependency closure')
            check.say('approved selected graph and sidecar SHA-256 tree %s'
                      % source_closure['tree_sha256'])
            check.say('not a release-grade lock or independent numeric model oracle')
        except (OSError, ValueError, TypeError, KeyError) as trouble:
            finished(check.refuse(str(trouble)))
            return report
        finished(check.done())
        report.summary['source_closure'] = source_closure

    
    
    
    
    
    
    
    
    host_facts = targets.host()
    
    
    
    
    if approved_build_plan is not None and 'cpu_codegen' in approved_build_plan:
        recorded = approved_build_plan['cpu_codegen']
        if not isinstance(recorded, dict):
            finished(report.step('Checking target reachability').refuse(
                'approved CPU codegen record is malformed'))
            return report
        replay = {'mode': recorded.get('mode'),
                  'mcpu': recorded.get('mcpu') if recorded.get('mode') == 'explicit' else None}
        if cpu_codegen is not None and cpu_codegen != replay:
            finished(report.step('Checking target reachability').refuse(
                'CPU codegen request differs from approved Preview'))
            return report
        cpu_codegen = replay
    try:
        chosen_plan = build_plan.resolve(
            host_facts, target=target, req_sm=req_sm,
            req_os=req_os, req_arch=req_arch,
            req_driver_api=req_driver_api,
            source=source if source is not None else model_path,
            shape=shape, tune=bool(tune_work_dir),
            trials_per_task=trials_per_task, contract=contract,
            cpu_codegen=cpu_codegen)
    except ValueError as trouble:
        finished(report.step('Checking target reachability').refuse(str(trouble)))
        return report
    
    
    
    
    if approved_build_plan is not None and chosen_plan != approved_build_plan:
        finished(report.step('Checking target reachability').refuse(
            'effective BuildPlan differs from reviewed Preview: source, '
            'model options or measured target changed; request a fresh Preview'))
        return report
    target_profile = chosen_plan['target_profile']
    req_sm = target_profile['deployment']['sm']
    level = target_profile['reachability']['level']
    reason = target_profile['reachability']['reason']
    selected_os = target_profile['deployment']['os']
    selected_arch = target_profile['deployment']['arch']
    report.summary['build_plan'] = chosen_plan
    reach = report.step('Checking target reachability')
    reach.say('host os=%s arch=%s cuda=%s'
              % (host_facts['os'], host_facts['arch'],
                 host_facts['cuda_arch'] or 'none'))
    reach.say('target os=%s arch=%s device=%s sm=%s'
              % (selected_os, selected_arch, target,
                 req_sm or '(host default)'))
    
    
    tuned_cpu = (target.lower() == 'cpu' and bool(tune_work_dir)
                 and recognised.kind != detect.TFLITE)
    try:
        _require_selected_cpu_codegen_route(chosen_plan, tuned_cpu)
    except ValueError as trouble:
        finished(reach.refuse(str(trouble)))
        return report
    if tuned_cpu:
        reach.say('compiler target %s%s' % (
            chosen_plan['compiler_target'],
            ' (actual codegen pending)' if 'cpu_codegen' in chosen_plan else
            ' (legacy host tuning)'))
    else:
        reach.say('compiler target %s' % targets.tvm_target_spec(target, req_sm))
    reach.numbers['level'] = level
    if level in (targets.RUNNER_NEEDED, targets.UNSUPPORTED_V1):
        reach.say('level %s' % level)
        finished(reach.refuse(reason))
        return report
    if (target_profile['deployment'].get('driver_api') is not None
            and target.lower() == 'cuda'):
        
        
        
        
        from . import cuda_preflight, environment
        compiler = environment.find_cuda()
        toolkit = cuda_preflight.installed_toolkit_version(
            compiler, environment._ask_program)
        compatibility, statement = cuda_preflight.compare_driver_toolkit(
            target_profile['deployment']['driver_api'], toolkit)
        reach.say('deployment CUDA driver API: %s; build toolkit: %s'
                  % (target_profile['deployment']['driver_api'],
                     '.'.join(str(n) for n in toolkit) if toolkit else 'unknown'))
        if compatibility != 'candidate':
            finished(reach.refuse(
                'the target CUDA environment is not confirmed compatible: ' +
                statement))
            return report
        reach.say('target driver/toolkit capability check: ' + statement)
        reach.say('the destination model still requires a real runtime test')
    reach.say('level %s: %s' % (level, reason))
    if level == targets.COMPILE_ONLY:
        reach.say('this machine can COMPILE the module, but tuning, benchmarking '
                  'and verify need the target device and cannot run here')
    finished(reach.done())

    
    
    
    if contract and contract.get('task'):
        adapter_check = report.step('Checking Pascal facade adapter availability')
        try:
            scope = validate_vision_adapter(contract['task'],
                                            contract.get('adapter'))
        except ValueError as trouble:
            finished(adapter_check.refuse(str(trouble)))
            return report
        adapter_check.numbers['contract_scope'] = scope
        report.summary['contract_scope'] = scope
        if scope == 'INDIVIDUAL_RAW_STAGE_ONLY':
            adapter_check.say('PaddleOCR stage only: numeric raw tensor '
                              'verification can be independent, but the '
                              'Det+Rec+dictionary Pascal composite still '
                              'needs separate native CTC acceptance')
        else:
            adapter_check.say('task/adapter pair has a current Pascal '
                              'TModelLoader.Build* implementation')
        finished(adapter_check.done())

    
    
    
    
    
    
    
    adapter_output_mode = None
    adapter_output_signature = None
    if contract and requires_adapter_output_check(
            contract.get('task'), contract.get('adapter')):
        check = report.step('Checking Pascal adapter output signature')
        try:
            signature = _source_output_signature(model_path, recognised.kind)
            _, chosen_shape = parse_shape(shape)
            input_batch = chosen_shape[0] if chosen_shape else None
            adapter_output_mode = validate_adapter_outputs(
                contract['task'], contract['adapter'], signature,
                input_batch=input_batch)
            if (adapter_output_mode == 'rtdetr_fused'
                    and recognised.kind != detect.ONNX):
                raise ValueError(
                    'the strict RT-DETR fused independent numerical oracle '
                    'currently supports ONNX only; this TFLite model cannot '
                    'receive a verified rtdetr manifest. Omit the task/adapter '
                    'contract to compile raw tensors.')
        except Exception as trouble:
            finished(check.refuse(str(trouble)))
            return report
        check.numbers['adapter_output_mode'] = adapter_output_mode
        report.summary['adapter_output_mode'] = adapter_output_mode
        if adapter_output_mode == 'rtdetr_fused':
            
            
            adapter_output_signature = signature
        check.say('Pascal %s output shape compatible (%s): %s' % (
            contract['adapter'], adapter_output_mode,
            '; '.join('%s %s %s' % (
                item['name'], item['shape'], item['dtype'])
                for item in signature)))
        if adapter_output_mode == 'rtdetr_fused':
            check.say(
                'fused decoder interprets columns as '
                'cx,cy,width,height,confidence,class_id; structural check '
                'does not establish numerical accuracy. Strict independent '
                'fused detection comparison must PASS before verified manifest.')
        finished(check.done())

    if contract and requires_paddleocr_output_check(
            contract.get('task'), contract.get('adapter')):
        check = report.step('Checking individual PaddleOCR stage output signature')
        try:
            if recognised.kind != detect.ONNX:
                raise ValueError('individual PaddleOCR native stage passport '
                                 'requires ONNX source; unsupported foreign '
                                 'formats may only compile without this contract')
            signature = _source_output_signature(model_path, recognised.kind)
            _, chosen_shape = parse_shape(shape)
            adapter_output_mode = validate_paddleocr_outputs(
                contract['task'], contract['adapter'], signature,
                input_shape=chosen_shape)
        except Exception as trouble:
            finished(check.refuse(str(trouble)))
            return report
        check.numbers['adapter_output_mode'] = adapter_output_mode
        report.summary['adapter_output_mode'] = adapter_output_mode
        check.say('PaddleOCR %s structurally matches original ONNX: %s' % (
            adapter_output_mode,
            '; '.join('%s %s %s' % (
                item['name'], item['shape'], item['dtype']) for item in signature)))
        check.say('This proves ONE raw stage ABI only. Independent numerical '
                  'comparison and a separate Det+Rec+dictionary native CTC '
                  'oracle are required for the final composite passport.')
        finished(check.done())

    step = report.step('Building the module')
    
    
    
    
    if (not isinstance(module_name, str) or
            not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', module_name) or
            module_name in ('.', '..') or module_name.endswith('.')):
        finished(step.refuse('unsafe module_name: expected one plain filename '
                             'without path separators, parent traversal or '
                             'trailing dot'))
        return report
    
    
    
    dos_base = module_name.split('.', 1)[0].upper()
    if (dos_base in ('CON', 'PRN', 'AUX', 'NUL') or
            re.fullmatch(r'(?:COM|LPT)[1-9]', dos_base)):
        finished(step.refuse('unsafe module_name: reserved Windows device name '
                             + dos_base))
        return report
    try:
        linked = _unsafe_output_directory(target_dir)
    except (OSError, ValueError, TypeError) as trouble:
        finished(step.refuse('cannot safely inspect destination: %s' % trouble))
        return report
    if linked:
        finished(step.refuse('output destination contains a symlink/junction '
                             '(%s); no build files were written' % linked))
        return report
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    module_path = os.path.join(target_dir,
                               module_name + '.' + target + module_extension())
    
    
    reserved = (
        module_path, module_path + '.inputs',
        os.path.join(target_dir, 'manifest.txt'),
        os.path.join(target_dir, 'manifest.json'),
        os.path.join(target_dir, 'recipe.json'),
        os.path.join(target_dir, 'report.txt'),
    )
    
    
    previous = [item for item in reserved if os.path.lexists(item)]
    if previous:
        finished(step.refuse(
            'this directory contains a previous build (%s). Nothing was '
            'replaced; select a fresh output directory.'
            % ', '.join(os.path.basename(item) for item in previous)))
        return report
    input_name, dims = parse_shape(shape)
    shapes = {input_name: dims} if (input_name and dims) else None
    step.say('target %s, extension %s' % (target, module_extension()))
    if req_sm:
        step.say('CUDA compiler architecture %s' % req_sm)

    
    

    if shapes:
        step.say('input shape given: %s = %s'
                 % (input_name, ','.join(str(d) for d in dims)))
    try:
        from .convert import Target, convert_onnx, convert_tflite
        tvm_target = Target.CUDA if target.lower() == 'cuda' else Target.CPU
        build = convert_tflite if recognised.kind == detect.TFLITE else convert_onnx
        extra = {}
        cpu_receipt = {}
        if req_sm is not None:
            extra['req_sm'] = req_sm
        if tune_work_dir and build is convert_onnx and tvm_target is Target.CPU:
            if 'cpu_codegen' in chosen_plan:
                extra['cpu_codegen_target'] = chosen_plan['cpu_codegen']
                extra['cpu_codegen_receipt'] = cpu_receipt
            step.say('tuning is on; the database goes to %s' % tune_work_dir)
            step.say('an untuned build takes seconds, a tuned one tens of '
                     'minutes - and comes out 32 to 58 times faster on the '
                     'three models measured 01.09.2026')
            extra['tune_work_dir'] = tune_work_dir
            if trials_per_task:
                extra['trials_per_task'] = trials_per_task
        built = build(model_path, module_path, target=tvm_target,
                      input_shapes=shapes, **extra)
        if 'cpu_codegen' in chosen_plan:
            if (cpu_receipt.get('compiler_target') != chosen_plan['compiler_target']
                    or cpu_receipt.get('binding') !=
                    'TVM_KIND_AND_MCPU_CHECKED_AFTER_EXPORT'):
                raise ValueError('actual tuned CPU compiler target not attested')
            report.summary['effective_cpu_codegen'] = cpu_receipt.copy()
        if tune_work_dir and 'tune_work_dir' in extra:
            from . import tuning as tuning_module
            covered = tuning_module.coverage(tune_work_dir, 0)
            step.say('tuning covered %s of %s tasks%s'
                     % (covered['tuned'], covered['workloads'],
                        '' if covered['complete'] else
                        ' - INCOMPLETE: ' + str(covered['reason'])))
    except Exception as trouble:
        finished(step.refuse(str(trouble)))
        return report
    step.say('produced %s' % os.path.basename(built))
    step.say('size %.1f MB' % (os.path.getsize(built) / 1048576.0))
    removed = _remove_leftovers(os.path.splitext(built)[0])
    if removed:
        step.say('linker leftovers removed: %s' % ', '.join(removed))
    manifest = built + '.inputs'
    if os.path.isfile(manifest):
        with open(manifest, encoding='utf-8') as handle:
            names = [line.split(chr(9))[1]
                     for line in handle.read().splitlines() if chr(9) in line]
        step.say('input names written alongside: %s' % ', '.join(names))
    report.summary['module'] = built
    if not tuned_cpu:
        report.summary['target_spec'] = targets.tvm_target_spec(target, req_sm)
    report.summary['req_sm'] = req_sm
    finished(step.done())

    runtime_step = runtime.place(target_dir, target)
    report.steps.append(runtime_step)
    finished(runtime_step)

    
    
    if target.lower() == 'cuda' and req_sm and level == targets.COMPILE_ONLY:
        
        
        verify_step = report.step('Comparison with the reference')
        verify_step.say('NOT CHECKED: built for %s, host GPU is %s. Verification '
                        'requires a device matching the requested architecture.'
                        % (req_sm, host_facts['cuda_arch'] or 'unavailable'))
    else:
        device_kind = 'cuda' if target.lower() == 'cuda' else 'cpu'
        if adapter_output_mode == 'rtdetr_fused':
            verify_step = verify.verify(
                model_path, built, dims, device_kind,
                comparison_profile='rtdetr_fused')
        else:
            verify_step = verify.verify(model_path, built, dims, device_kind)
        report.steps.append(verify_step)
    finished(verify_step)
    
    
    
    report.summary['verification_status'] = (
        'VERIFIED_LOCAL' if verify_step.state == 'done' else
        'NOT_CHECKED' if verify_step.state == 'unchecked' else 'FAILED')
    if paddle_selected is not None:
        
        
        
        
        report.summary['exported_onnx_comparison_status'] = (
            report.summary['verification_status'])
        report.summary['verification_status'] = 'NOT_CHECKED'
        report.summary['verification_note'] = (
            'Exported ONNX comparison is not independent Paddle original '
            'inference/CTC verification; source-level SHA remains nonimmutable')

    
    
    
    
    
    
    
    
    
    if paddle_selected is not None:
        check = report.step('Rechecking selected Paddle source after native build')
        try:
            from . import paddle_closure
            observed = paddle_closure.audit_selected_paddle_source(
                source, role=paddle_selected['role'],
                dictionary=(paddle_selected['dictionary'] or {}).get('reference'),
                expected_tree_sha256=paddle_selected['tree_sha256'])
            if observed != paddle_selected:
                raise ValueError('Paddle source/dictionary changed while compiling')
        except (OSError, ValueError, TypeError, KeyError) as trouble:
            finished(check.refuse(str(trouble)))
            return report
        finished(check.done())
    if source_closure is None:
        provenance = _gather_provenance(built, model_path, source)
    else:
        provenance = _gather_provenance(built, model_path, source,
                                        source_closure=source_closure)
    if paddle_selected is not None:
        provenance['paddle_selected_source'] = paddle_selected

    
    
    
    
    
    
    
    
    
    
    
    if contract and contract.get('task'):
        manifest_step = report.step('Writing the bundle manifest')
        if paddle_selected is not None:
            manifest_step.say(
                'not issued: selected Paddle input has no independent '
                'original det/rec/CTC numerical oracle; exported ONNX '
                'comparison is not a VERIFIED product passport')
            finished(manifest_step)
        elif not report.sound or verify_step.state != 'done':
            manifest_step.say(
                'not issued: independent verification did not PASS '
                '(state=%s); the verified passport is withheld'
                % verify_step.state)
            finished(manifest_step)
        else:
            try:
                _emit_manifest(target_dir, built, target, contract, manifest_step,
                               provenance=provenance, license=license,
                               req_sm=req_sm, target_profile=target_profile,
                               adapter_output_mode=adapter_output_mode,
                               adapter_output_signature=adapter_output_signature)
                finished(manifest_step.done())
            except Exception as trouble:
                finished(manifest_step.refuse(str(trouble)))
                return report

    
    
    
    
    
    
    
    
    
    
    recipe_step = report.step('Writing the build recipe')
    try:
        recipe = build_recipe(
            source if source is not None else model_path, target, shape,
            module_name, bool(tune_work_dir), trials_per_task, req_sm, contract,
            provenance=provenance, license=license,
            target_profile=target_profile, build_plan=chosen_plan,
            effective_cpu_codegen=(report.summary.get('effective_cpu_codegen')
                                   if 'cpu_codegen' in chosen_plan else None))
        write_recipe(recipe, os.path.join(target_dir, 'recipe.json'))
        recipe_step.say('source %s' % recipe['source'])
        recipe_step.say('target %s, sm %s, tune %s'
                        % (target, req_sm or '(host default)', recipe['tune']))
        recipe_step.say('toolchain tvm=%s onnx=%s'
                        % (provenance['toolchain'].get('tvm', 'unknown'),
                           provenance['toolchain'].get('onnx', 'unknown')))
        recipe_step.say('wrote recipe.json (build inputs, separate from the manifest)')
        finished(recipe_step.done())
    except OSError as trouble:
        finished(recipe_step.refuse('the recipe could not be written: %s' % trouble))

    report.summary['directory'] = target_dir
    report.summary['files'] = sorted(os.listdir(target_dir))
    return report


def _source_output_signature(model_path, kind):
    """Inspect actual source graph outputs without loading TVM or compiling.

    The signature is used only for adapters with known Pascal output contracts.
    All dimensions remain as declared: integers for fixed sizes, ONNX symbolic
    names for shared dynamic sizes, and None for an unconstrained dimension.
    Inferring values from example data would wrongly certify incompatible graphs.
    """
    if kind == detect.ONNX:
        import onnx

        
        graph = onnx.load(model_path, load_external_data=False).graph
        rows = []
        for value in graph.output:
            tensor = (value.type.tensor_type
                      if value.type.HasField('tensor_type') else None)
            shape = None
            if tensor is not None and tensor.HasField('shape'):
                shape = []
                for dim in tensor.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(int(dim.dim_value))
                    elif dim.HasField('dim_param') and dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(None)
            dtype = {
                onnx.TensorProto.FLOAT: 'float32',
                onnx.TensorProto.FLOAT16: 'float16',
            }.get(tensor.elem_type) if tensor is not None else None
            rows.append({'name': value.name, 'shape': shape, 'dtype': dtype})
        return rows

    if kind == detect.TFLITE:
        import tflite

        with open(model_path, 'rb') as handle:
            buffer = handle.read()
        model = tflite.Model.GetRootAsModel(buffer, 0)
        subgraph = model.Subgraphs(0)
        if subgraph is None:
            raise ValueError('the TFLite graph has no first subgraph')
        tensor_types = tflite.TensorType
        
        if hasattr(tensor_types, 'TensorType'):
            tensor_types = tensor_types.TensorType
        dtypes = {tensor_types.FLOAT32: 'float32',
                  tensor_types.FLOAT16: 'float16'}
        rows = []
        for index in range(subgraph.OutputsLength()):
            tensor = subgraph.Tensors(subgraph.Outputs(index))
            raw_name = tensor.Name()
            signature = tensor.ShapeSignatureAsNumpy()
            if not getattr(signature, 'size', 0):
                signature = tensor.ShapeAsNumpy()
            shape = ([int(d) if int(d) > 0 else None for d in signature]
                     if signature is not None else [])
            rows.append({
                'name': (raw_name.decode('utf-8')
                         if raw_name else str(index)),
                'shape': shape,
                'dtype': dtypes.get(tensor.Type()),
            })
        return rows

    raise ValueError('cannot inspect output signature of model kind %r' % kind)


def _read_first_input(inputs_path):
    """Return the ONLY supported public Pascal input with strict TSV parsing.

    The sidecar is the TSV the converter writes beside every module -
    ``index<TAB>name<TAB>d0,d1,...``. The manifest's io line describes the tensor
    the caller must supply. The public vision Facade currently describes ONE
    input; accepting the first of four OmDet tensors would create a false
    manifest saying a multi-input model only needs one caller tensor.
    A free dimension -1 is retained explicitly rather than guessing a value.
    """
    candidate = Path(inputs_path)
    if (candidate.is_symlink() or
            getattr(candidate, 'is_junction', lambda: False)()):
        raise ValueError('linked native .inputs is not a trusted Pascal IO contract')
    with open(inputs_path, 'rb') as handle:
        raw = handle.read(65537)
    if not raw or len(raw) > 65536:
        raise ValueError('native .inputs missing, empty or larger than 64KiB')
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeDecodeError as error:
        raise ValueError('native .inputs is not valid UTF-8') from error
    if len(lines) != 1:
        raise ValueError('public Pascal vision passport requires exactly ONE '
                         'native .inputs row (multi-input adapter unsupported)')
    parts = lines[0].split(chr(9))
    if len(parts) != 3 or parts[0] != '0' or not parts[1]:
        raise ValueError('native .inputs needs exact index 0, name, shape TSV')
    name, encoded = parts[1], parts[2]
    if (any(ord(letter) < 32 for letter in name) or not encoded or
            not re.fullmatch(r'-?[0-9]+(?:,-?[0-9]+){0,15}', encoded)):
        raise ValueError('native .inputs name/shape contains unsupported syntax')
    shape = [int(value) for value in encoded.split(',')]
    if any(value == 0 or value < -1 or value > 2147483647 for value in shape):
        raise ValueError('native .inputs dimensions must be positive or -1')
    return name, shape


def _emit_manifest(target_dir, built, target, contract, step,
                   provenance=None, license=None, req_sm=None,
                   target_profile=None, adapter_output_mode=None,
                   adapter_output_signature=None):
    """Write manifest.txt and manifest.json beside a freshly built vision module.

    The artifact quad is filled here from what the build knows: os and native
    CPU architecture from the running platform, dtype fixed to float32,
    device from the chosen target, file from the module's own name, and format
    and entry from the vision conventions build_vision_contract bakes in. The io
    name and shape come from the .inputs sidecar; the layout comes from the
    recipe, because the file does not record it.

    provenance and license enrich the passport into a VERIFIABLE artifact (see
    _enrich_contract): the built module's hash, where the model came from and
    when, the toolchain, and the licence. They are optional here on purpose - the
    pipeline always passes provenance, so a real build gets the rich passport;
    a direct caller (a test exercising the emit path alone) may omit it and get
    the plain manifest, exactly as before.
    """
    linked = _unsafe_output_directory(target_dir)
    if linked:
        raise ValueError('cannot issue Pascal passport through linked output '
                         'directory: %s' % linked)
    inputs_path = built + '.inputs'
    if not os.path.isfile(inputs_path):
        raise ValueError('the input sidecar %s is missing, so the manifest cannot '
                         'describe the model input' % os.path.basename(inputs_path))
    io_name, io_shape = _read_first_input(inputs_path)

    if not contract.get('adapter'):
        raise ValueError('a task was given but no adapter; the manifest needs '
                         'both. Pass the adapter, e.g. yolo11 or tensorflow.')

    native_arch = _native_artifact_arch()
    if target_profile is not None:
        declared_arch = targets._norm_arch(
            target_profile.get('deployment', {}).get('arch'))
        if declared_arch != native_arch:
            raise ValueError('model artifact CPU ISA differs from native build '
                             'host: declared %r, observed %r' %
                             (declared_arch, native_arch))

    built_contract = build_vision_contract(
        contract['task'], contract['adapter'], contract.get('layout') or 'NCHW',
        os.path.basename(built), io_name, io_shape,
        adapter_api_version=contract.get('adapter_api_version', 1),
        os_=platform.system().lower(), arch=native_arch,
        device=target.lower(), dtype=ARTIFACT_DTYPE,
        score_threshold=contract.get('score_threshold'),
        nms_threshold=contract.get('nms_threshold'),
        mean=contract.get('mean'), std=contract.get('std'),
        keypoint_count=contract.get('keypoint_count'))

    if adapter_output_mode == 'rtdetr_fused':
        if (not isinstance(adapter_output_signature, (tuple, list))
                or len(adapter_output_signature) != 1):
            raise ValueError('verified RT-DETR fused output signature is absent')
        
        
        
        validate_adapter_outputs(
            contract['task'], contract['adapter'], adapter_output_signature,
            input_batch=io_shape[0] if io_shape else None)
        output = adapter_output_signature[0]
        built_contract['output'] = {
            'name': output['name'],
            'shape': list(output['shape']),
            'dtype': output['dtype'],
            'format': 'rtdetr_fused_cxcywh_v1',
            'fields': ['cx', 'cy', 'width', 'height', 'confidence', 'class_id'],
        }
    elif adapter_output_signature is not None:
        raise ValueError('output signature present without validated fused mode')

    req_sm = targets.validate_target_sm(target, req_sm)
    if req_sm:
        built_contract['artifact']['device_arch'] = req_sm
    if target_profile is not None:
        targets.validate_recorded_profile(
            target_profile, target=target, req_sm=req_sm)
        
        
        
        built_contract['target_profile'] = target_profile

    _enrich_contract(built_contract, provenance, license)

    
    
    
    
    
    
    manifest_txt = os.path.join(target_dir, 'manifest.txt')
    manifest_json = os.path.join(target_dir, 'manifest.json')
    if any(os.path.lexists(path) for path in (manifest_txt, manifest_json)):
        raise FileExistsError('another build already owns the manifest filename; '
                              'choose a fresh output directory')
    txt_body = emit_manifest_text(built_contract, newline=chr(10)).encode('utf-8')
    json_body = (emit_manifest_json(built_contract) + chr(10)).encode('utf-8')
    with tempfile.TemporaryDirectory(prefix='.pascalvision-passport-',
                                     dir=target_dir) as staging:
        scratch_txt = Path(staging) / 'manifest.txt'
        scratch_json = Path(staging) / 'manifest.json'
        with scratch_txt.open('xb') as file:
            file.write(txt_body)
        with scratch_json.open('xb') as file:
            file.write(json_body)
        if (scratch_txt.read_bytes() != txt_body or
                scratch_json.read_bytes() != json_body):
            raise OSError('staged Pascal manifest bytes are incomplete')
        os.link(scratch_json, manifest_json)
        os.link(scratch_txt, manifest_txt)

    step.say('task %s, adapter %s'
             % (built_contract['task_kind'], built_contract['adapter']))
    step.say('artifact os=%s arch=%s device=%s dtype=%s'
             % (platform.system().lower(), native_arch, target.lower(),
                ARTIFACT_DTYPE))
    if req_sm:
        step.say('artifact deviceArch=%s' % req_sm)
    step.say('io name=%s shape=%s layout=%s'
             % (io_name, ','.join(str(d) for d in io_shape),
                built_contract['io']['layout']))
    if built_contract.get('output'):
        row = built_contract['output']
        step.say('verified output ABI %s name=%s shape=%s dtype=%s fields=%s'
                 % (row['format'], row['name'],
                    ','.join(str(d) for d in row['shape']), row['dtype'],
                    ','.join(row['fields'])))
    artifact_hash = built_contract['artifact'].get('hash')
    if artifact_hash:
        step.say('artifact sha256=%s' % artifact_hash)
    if built_contract.get('provenance'):
        prov = built_contract['provenance']
        step.say('provenance source=%s builtAt=%s'
                 % (prov.get('source'), prov.get('built_at')))
    if built_contract.get('toolchain'):
        tc = built_contract['toolchain']
        step.say('toolchain tvm=%s onnx=%s'
                 % (tc.get('tvm', 'unknown'), tc.get('onnx', 'unknown')))
    if built_contract.get('license'):
        step.say('license %s' % built_contract['license'])
    step.say('wrote manifest.txt and manifest.json')
    labels = contract.get('labels')
    if labels:
        
        
        
        
        step.say('labels source recorded: %s (not yet emitted into the manifest)'
                 % labels)


def _enrich_contract(built_contract, provenance, license):
    """Fold the audit facts into a built vision contract, in place, so the emitter
    writes them as the extra manifest lines (and the JSON companion mirrors them).

    Each field is added ONLY when it has a value, which is what keeps the
    optional-in-the-emitter promise intact: a contract enriched with nothing is
    the contract build_vision_contract made, and renders exactly as the shipped
    bundles do. provenance is the dict _gather_provenance built (or None); license
    is the caller's string (or None). The internal keys stay snake_case, matching
    the rest of the contract; the emitter renders them camelCase.
    """
    if provenance:
        artifact_hash = provenance.get('artifact_hash')
        if artifact_hash:
            built_contract['artifact']['hash'] = artifact_hash
        section = {'source': provenance.get('source')}
        if provenance.get('source_hash'):
            section['source_hash'] = provenance['source_hash']
        if provenance.get('built_at'):
            section['built_at'] = provenance['built_at']
        if section.get('source'):
            built_contract['provenance'] = section
        toolchain = provenance.get('toolchain')
        if toolchain:
            built_contract['toolchain'] = dict(toolchain)
    if license:
        built_contract['license'] = license
