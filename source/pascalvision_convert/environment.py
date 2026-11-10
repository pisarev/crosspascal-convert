# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import os
import platform
import shutil
import subprocess
import sys

from .report import Report
from . import cuda_preflight


def llvm_locations():
    places = []
    for variable in ('ProgramFiles', 'ProgramFiles(x86)', 'LOCALAPPDATA'):
        root = os.environ.get(variable, '')
        if root:
            places.append(os.path.join(root, 'LLVM', 'bin'))


    places.extend((
        os.path.join('C:' + os.sep, 'Program Files', 'LLVM', 'bin'),
        os.path.join('C:' + os.sep, 'Program Files (x86)', 'LLVM', 'bin'),
        os.path.join(os.sep, 'usr', 'bin'),
        os.path.join(os.sep, 'usr', 'local', 'bin'),
    ))
    seen = set()
    unique = []
    for place in places:
        if place not in seen:
            seen.add(place)
            unique.append(place)
    return tuple(unique)


LLVM_LOCATIONS = llvm_locations()


def _package_version(name):
    try:
        package = __import__(name)
    except Exception:
        return None
    return getattr(package, '__version__', 'present, version not declared')


def _ask_program(path, args):
    try:
        finished = subprocess.run([path] + args, capture_output=True, timeout=30)
    except Exception:
        return None
    if finished.returncode != 0:
        return None
    text = (finished.stdout or finished.stderr or b'').decode('utf-8', 'replace')
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def find_clang():
    on_path = shutil.which('clang')
    if on_path:
        return on_path, True
    for location in llvm_locations():
        for name in ('clang.exe', 'clang'):
            candidate = os.path.join(location, name)
            if os.path.isfile(candidate):
                return candidate, False
    return None, False


def find_cuda():
    nvcc = shutil.which('nvcc')
    if nvcc:
        return nvcc
    root = os.environ.get('CUDA_PATH', '')
    if root:
        for name in ('nvcc.exe', 'nvcc'):
            candidate = os.path.join(root, 'bin', name)
            if os.path.isfile(candidate):
                return candidate
    return None


def find_gpu():
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    return _ask_program(smi, ['--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'])


def find_gpu_arch():
    from pascalvision_convert import targets
    through_tvm = targets.host().get('cuda_arch')
    if through_tvm:
        return through_tvm
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    answer = _ask_program(smi, ['--query-gpu=compute_cap', '--format=csv,noheader'])


    if not answer or not answer.strip():
        return None


    first = answer.strip().splitlines()[0].strip()
    parts = first.split('.')
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return 'sm_%s%s' % (parts[0], parts[1])


def core_verdict(python_ok, tvm_ok, onnx_reader, tflite_reader, linker_ok):
    common = [name for name, ok in (('Python', python_ok), ('TVM', tvm_ok),
                                    ('the linker', linker_ok)) if not ok]
    if common:
        return {
            'ready': False, 'formats': (),
            'reason': ('%s missing from this interpreter, so no model of any '
                       'format can be compiled here' % ', '.join(common)),
        }
    formats = tuple(name for name, ok in (('onnx', onnx_reader),
                                          ('tflite', tflite_reader)) if ok)
    if len(formats) == 2:
        return {'ready': True, 'formats': formats,
                'reason': 'this interpreter can compile ONNX and TFLite'}
    if formats:
        absent = 'tflite' if formats[0] == 'onnx' else 'onnx'
        return {'ready': True, 'formats': formats,
                'reason': ('this interpreter can compile %s; the %r package is '
                           'missing, so a %s model cannot be read here'
                           % (formats[0].upper(), absent, absent.upper()))}
    return {'ready': False, 'formats': (),
            'reason': ('neither onnx nor tflite is installed, so no model can be '
                       'read here - the compiler is fine, there is nothing to feed it')}


def reference_verdict(onnx_runner, tflite_runner):
    if onnx_runner and tflite_runner:
        return {'ready': True, 'formats': ('onnx', 'tflite'),
                'reason': 'ONNX and TFLite builds can both be compared with their models'}
    if onnx_runner:
        return {'ready': True, 'formats': ('onnx',),
                'reason': ('ONNX builds can be compared; ai_edge_litert is missing, '
                           'so a TFLite build would go unchecked')}
    if tflite_runner:
        return {'ready': True, 'formats': ('tflite',),
                'reason': ('TFLite builds can be compared; onnxruntime is missing, '
                           'so an ONNX build would go unchecked')}
    return {'ready': False, 'formats': (),
            'reason': ('neither onnxruntime nor ai_edge_litert is installed, so a '
                       'build cannot be compared with the model it came from')}


def full_build_verdict(interpreter, builder, tvm_in_builder, linker):
    missing = []
    if not interpreter:
        missing.append('its interpreter is not where it is expected')
    if not builder:
        missing.append('the builder script is missing')
    if interpreter and not tvm_in_builder:
        missing.append('that interpreter carries no TVM')
    if not linker:
        missing.append('clang is not on this machine, and the bundle\'s module is '
                       'linked with it at the very end of the build')
    if not missing:
        return {'ready': True,
                'reason': 'the family builder, its interpreter and the linker are in place'}
    return {'ready': False,
            'reason': ('a full native build is unavailable: %s' % ', '.join(missing))}


def gpu_verdict(card, nvcc):
    if not nvcc:
        return {
            'ready': False,
            'needs_arch': False,
            'reason': ('no NVIDIA toolkit: nvcc is not on PATH and CUDA_PATH names '
                       'no compiler, and the GPU kernels are compiled through it'),
        }
    if not card:
        return {
            'ready': True,
            'needs_arch': True,
            'reason': ('toolkit present, no card visible: a GPU build must name '
                       'its target architecture, and the bundle it '
                       'makes will not run on this machine'),
        }
    return {'ready': True, 'needs_arch': False, 'reason': card}


def cudnn_locations():
    places = []
    for variable in ('CUDNN_PATH', 'CUDA_PATH'):
        root = os.environ.get(variable, '')
        if root:
            places.append(os.path.join(root, 'bin'))
            places.append(os.path.join(root, 'lib', 'x64'))


    for variable in ('ProgramFiles', 'ProgramW6432'):
        root = os.environ.get(variable, '')
        if root:
            places.append(os.path.join(root, 'NVIDIA', 'CUDNN'))
    places.append(os.path.join('C:' + os.sep, 'Program Files', 'NVIDIA', 'CUDNN'))
    for entry in os.environ.get('PATH', '').split(os.pathsep):
        if entry and 'cud' in entry.lower():
            places.append(entry)
    try:
        import nvidia
        for root in getattr(nvidia, '__path__', []):
            places.append(os.path.join(root, 'cudnn', 'bin'))
    except ImportError:
        pass


    seen = set()
    unique = []
    for place in places:
        if place and place not in seen:
            seen.add(place)
            unique.append(place)
    return unique


def find_cudnn():
    names = ('cudnn64_9.dll', 'cudnn64_8.dll', 'libcudnn.so', 'libcudnn.so.9',
             'libcudnn.so.8', 'libcudnn.dylib')
    places = cudnn_locations()
    for place in places:
        if not os.path.isdir(place):
            continue
        for name in names:
            candidate = os.path.join(place, name)
            if os.path.isfile(candidate):
                return candidate, places
        for entry in os.listdir(place):
            if entry.lower().startswith('cudnn') and entry.lower().endswith(
                    ('.dll', '.so', '.dylib')):
                return os.path.join(place, entry), places
    return None, places


def tvm_cudnn_support():
    try:
        import tvm
        from tvm.contrib import cudnn
    except Exception:
        return None, None
    try:
        built_with = tvm.support.libinfo().get('USE_CUDNN')
    except Exception:
        built_with = None
    try:
        usable = bool(cudnn.exists())
    except Exception:
        usable = False
    return built_with, usable


def _has_wsl():
    if platform.system() != 'Windows':
        return None
    wsl = shutil.which('wsl')
    if not wsl:
        return False
    try:
        finished = subprocess.run([wsl, '-e', 'true'], capture_output=True, timeout=60)
    except Exception:
        return False
    return finished.returncode == 0


def collect():
    report = Report('Machine report')

    step = report.step('Machine and Python')
    step.say('%s %s, %s' % (platform.system(), platform.release(), platform.machine()))
    step.say('Python %s' % platform.python_version())
    step.say('interpreter %s' % sys.executable)
    step.numbers['python'] = platform.python_version()
    step.numbers['system'] = platform.system()
    if sys.version_info >= (3, 10):
        step.done()
    else:
        step.refuse('Python 3.10 or newer is required - the TVM wheels start there')

    step = report.step('TVM, which does the compiling')
    version = _package_version('tvm')
    if version:
        step.numbers['tvm'] = version
        step.done('apache-tvm %s' % version)
        ffi = _package_version('tvm_ffi')
        if ffi:
            step.say('tvm-ffi %s' % ffi)
            step.numbers['tvm_ffi'] = ffi
    else:
        step.refuse('no tvm package; it installs as apache-tvm==0.26.0')


    step = report.step('Reading models')
    for name, purpose in (('onnx', 'ONNX models'),
                          ('tflite', 'TFLite models')):
        version = _package_version(name)
        step.say('%-16s %s   %s' % (name, version or 'MISSING', purpose))
    onnx_reader = bool(_package_version('onnx'))
    tflite_reader = bool(_package_version('tflite'))


    if onnx_reader and tflite_reader:
        step.done()
    elif onnx_reader or tflite_reader:
        present = 'onnx' if onnx_reader else 'tflite'
        absent = 'tflite' if onnx_reader else 'onnx'
        step.done('%s models can be read; %s is missing, so models of that '
                  'format cannot' % (present.upper(), absent))
    else:
        step.refuse('missing: onnx, tflite - no model of either format can be read')

    step = report.step('Reference runners, for checking a build against its model')
    for name, purpose in (('onnxruntime', 'reference for ONNX'),
                          ('ai_edge_litert', 'reference for TFLite')):
        version = _package_version(name)
        step.say('%-16s %s   %s' % (name, version or 'MISSING', purpose))
    reference = reference_verdict(bool(_package_version('onnxruntime')),
                                  bool(_package_version('ai_edge_litert')))
    if reference['ready']:
        step.done(reference['reason'])
    else:
        step.say(reference['reason'])
        step.say('a build still runs; what it loses is the comparison, and the '
                 'comparison step says so for itself')
        step.refuse('nothing to compare a build against')

    step = report.step('Linker')
    if platform.system() == 'Windows':
        clang, on_path = find_clang()
        if clang and on_path:
            step.done(_ask_program(clang, ['--version']) or 'version not reported')
            step.say(clang)
        elif clang:
            os.environ['PATH'] = os.path.dirname(clang) + os.pathsep + os.environ.get('PATH', '')
            step.done('found off the path; the path was added for this run')
            step.say(clang)
            step.say('to stop this recurring, add that directory to PATH for good')
        else:
            step.refuse('Windows needs clang from LLVM: without it the module cannot be linked')
    else:
        step.done('on this system TVM links by itself; no separate linker is needed')


    step = report.step('GPU')
    card = find_gpu()
    nvcc = find_cuda()
    cudnn_path, cudnn_places = find_cudnn()
    cudnn_built, cudnn_usable = tvm_cudnn_support()

    if card:
        step.say('card:    %s' % card)
    else:
        step.say('card:    nvidia-smi did not answer, so no card is visible')

    if nvcc:
        version = _ask_program(nvcc, ['--version'])
        step.say('toolkit: %s' % nvcc)
        if version:
            step.say('         %s' % version)
    else:
        step.say('toolkit: not found - building for the GPU is unavailable')

    if cudnn_path:
        step.say('cuDNN:   %s' % cudnn_path)
    else:
        step.say('cuDNN:   not found. Searched in:')


        shown = cudnn_places[:6]
        for place in shown:
            step.say('           %s' % place)
        if len(cudnn_places) > len(shown):
            step.say('           ... and %d more place(s), not listed here'
                     % (len(cudnn_places) - len(shown)))
        step.say('         (that is a statement about our search, not about the machine)')


    if cudnn_built is not None:
        step.say('this TVM was built with USE_CUDNN=%s, and reports cuDNN as %s'
                 % (cudnn_built, 'usable' if cudnn_usable else 'unusable'))
        if cudnn_path and not cudnn_usable:
            step.say('so cuDNN is on the machine but out of this compiler reach:'
                     ' a cuDNN target would fail, and it is not offered')

    step.numbers['card'] = card or ''
    step.numbers['cuda'] = bool(nvcc)
    step.numbers['cudnn_on_machine'] = bool(cudnn_path)
    step.numbers['cudnn_usable'] = bool(cudnn_usable)
    if not nvcc:
        step.say('none of this affects building for the CPU')
    step.done()

    step = report.step('WSL, needed only for PaddlePaddle models')
    present = _has_wsl()
    if present is None:
        step.done('not Windows: Paddle converts right here')
    elif present:
        step.done('present')
    else:
        step.say('absent; ONNX and TFLite models are unaffected')
        step.say('install WSL for the documented PaddlePaddle conversion route')
        step.done()


    def _step_ok(name):
        return all(s.state != 'refused' for s in report.steps if s.name == name)

    core = core_verdict(_step_ok('Machine and Python'),
                        _step_ok('TVM, which does the compiling'),
                        onnx_reader, tflite_reader,
                        _step_ok('Linker'))


    report.summary['cpu_ready'] = core['ready']
    report.summary['core_ready'] = core['ready']
    report.summary['core_reason'] = core['reason']


    report.summary['core_formats'] = list(core['formats'])


    report.summary['reference_ready'] = reference['ready']
    report.summary['reference_reason'] = reference['reason']
    report.summary['reference_formats'] = list(reference['formats'])


    report.summary['has_cuda'] = bool(nvcc)
    report.summary['card'] = card or ''


    report.summary['card_arch'] = find_gpu_arch() or ''
    report.summary['cudnn_on_machine'] = bool(cudnn_path)
    report.summary['cudnn_usable'] = bool(cudnn_usable)


    verdict = gpu_verdict(card, nvcc)
    report.summary['card_present'] = bool(card)
    report.summary['gpu_ready'] = verdict['ready']
    report.summary['gpu_needs_arch'] = verdict['needs_arch']
    report.summary['gpu_reason'] = verdict['reason']


    check = cuda_preflight.check_cuda_toolkit(nvcc, _ask_program)
    report.summary['cuda_compatibility'] = check['status']
    report.summary['cuda_compatibility_reason'] = check['reason']
    report.summary['cuda_driver_api'] = check['driver_api']
    report.summary['cuda_toolkit_version'] = check['toolkit_version']
    report.summary['cuda_runtime_version'] = check['runtime_version']
    report.summary['cuda_compatibility_scope'] = 'installed toolkit only, not exported TVM module'
    step.say('driver/runtime: %s - %s' % (check['status'], check['reason']))

    from . import targets
    report.summary['os'] = platform.system().lower()
    report.summary['arch'] = targets._norm_arch(platform.machine())
    return report


if __name__ == '__main__':
    print(collect().as_text())
