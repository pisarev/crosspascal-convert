# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""What this machine has for converting a model, and what it lacks.

WHY. On 31.08.2026 a build quietly failed because the linker was not on the
path, and it looked like the MODEL being refused. An hour went into a model that
had nothing to do with it. This report answers that question in a second: every
part is named, and said to be present or not.

ONE RULE HERE, AND IT IS THE MAIN ONE: DO NOT GUESS. A part that cannot be asked
gets "not checked" - a state of its own, equal to neither "present" nor
"missing". Filing the unchecked under success is how a silent gate is born.
"""

import os
import platform
import shutil
import subprocess
import sys

from .report import Report
from . import cuda_preflight


def llvm_locations():
    """Where clang is looked for besides PATH, best first.

    A FUNCTION AND NOT A CONSTANT, because two of these come from the machine:
    a system drive is not always C:, and LLVM installed "for me" lands under
    LOCALAPPDATA, which a fixed list had no way to name.

    ONE LIST FOR BOTH SIDES. The build had its own search (convert.py) and this
    file had another, and they differed - the build looked under LOCALAPPDATA,
    the diagnosis did not. On a machine with a per-user LLVM the wizard would
    therefore report "no linker" while the build found one and compiled happily:
    a false refusal, from two instruments answering one question. Both now ask
    here.
    """
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
    """Version of an installed package, or None. Asked by import, not by files."""
    try:
        package = __import__(name)
    except Exception:
        return None
    return getattr(package, '__version__', 'present, version not declared')


def _ask_program(path, args):
    """First non-empty line a program answers with, or None."""
    try:
        finished = subprocess.run([path] + args, capture_output=True, timeout=30)
    except Exception:
        return None
    text = (finished.stdout or finished.stderr or b'').decode('utf-8', 'replace')
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def find_clang():
    """The linker: where it is, and whether it is on the path.

    WHY TWO ANSWERS RATHER THAN ONE. "Absent" and "present but off the path" are
    different troubles with different cures, and merging them sends a reader off
    to install what they already have. Measured 31.08.2026: clang sat in
    the system LLVM installation directory while the wizard's environment could not see it,
    and the report said "clang is needed" - advice pointing the wrong way.
    """
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
    """The NVIDIA toolkit, if visible. We look for the compiler, not a variable.

    An environment variable survives an uninstalled toolkit, so the presence of
    nvcc decides; the variable is only a second place to look.
    """
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
    """The card itself, asked of the driver. Returns its line, or None.

    The toolkit and the card are different facts: a machine can carry the
    toolkit with no card in it, and a card with no toolkit installed. Reporting
    one as the other names a false cause, and a false cause sends the reader to
    the wrong place.
    """
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    return _ask_program(smi, ['--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'])


def find_gpu_arch():
    """The card's compute capability as a CUDA architecture - 'sm_75' for a GTX
    1650 - or None when there is no card to ask.

    WHY IT IS ASKED AND NOT TYPED IN. A GPU build compiles for one architecture,
    and the wizard has to offer a default that fits the card in front of the
    user; a number they have to look up is a number they will get wrong, and a
    wrong one only surfaces on the far side of a build.

    TVM FIRST, THE DRIVER SECOND. targets.host() already names the card the way
    the LLM builder does - tvm.cuda(0), then Target.from_device - and that is the
    instrument the build itself will use, so its answer is the one that matters.
    It needs TVM, which this converter does not require; where TVM is absent the
    driver is asked instead, through nvidia-smi, so the wizard can still offer a
    default. One order of preference, named, rather than two detectors drifting
    apart.
    """
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
    """What the core can compile IN THIS INTERPRETER, and what stands in the way.

    Pure, like gpu_verdict, and deliberately blind to everything the other path
    needs: a family's own builder runs elsewhere, and no fact about it belongs in
    this answer. See full_build_verdict for the other half.

    THE ANSWER IS A LIST OF FORMATS, NOT A YES. The two readers serve different
    formats and are imported in different branches - onnx in the ONNX path
    (convert.py:96), tflite in the TFLite path (convert.py:339) - so neither is
    needed to build the other's models. Demanding both, as one flag used to,
    tells a machine that carries tflite and no onnx that it can build nothing,
    while a TFLite model would compile there perfectly well. What IS shared is
    Python, TVM and the linker: without any of those no format survives.
    """
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
    """Can a finished build be COMPARED with the model it came from.

    A third ability, and a third pure function, deliberately apart from the two
    that answer whether anything can be built at all. Its facts are the two
    reference runners and nothing else: this verdict can neither block a build
    nor be blocked by one.

    HALF IS NOT NOTHING. The runners cover different formats - onnxruntime an
    ONNX model, ai_edge_litert a TFLite one - so with one of them present, builds
    of that format are still checked. Reporting "cannot compare" in that case
    would name a false cause for every ONNX model on a machine that checks them
    perfectly well.
    """
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
    """Can a family's own builder produce a whole bundle, and what is missing.

    WHY THIS IS A SEPARATE VERDICT. A full build - an LLM - is not compiled by
    the core at all: the family's builder runs in ITS OWN pinned interpreter and
    writes the whole bundle. So it needs neither the TVM of this process nor the
    ONNX and TFLite readers. Folding the two into one "ready" flag, as cpu_ready
    did, makes a wizard with no onnx package announce that it can build nothing -
    while the full build beside it would have succeeded. One flag answering for
    two abilities names a false cause the moment the two differ.

    THE LINKER IS THE ONE SHARED TOOL, and it belongs here on its own merits,
    not by inheritance. The family builder exports its module through the very
    function the ONNX core uses (convert._shared_library_builder), and that
    function needs clang: without it the build dies at the export, minutes in,
    with the weights already converted. A shared fact is not the same thing as a
    merged verdict - what was wrong before was one flag standing for two
    abilities, not two abilities needing one tool.

    WHAT IT DOES *NOT* REQUIRE, deliberately. There is a downstream record (P1)
    about clang on Windows: TVM's own MetaSchedule runner calls it with an
    incomplete triple and without the runtime import libraries, and tuning then
    measures nothing while reporting success. That defect lives in TVM's tuning
    path, not in ours - our exporter already passes the full
    x86_64-pc-windows-msvc triple and adds tvm_ffi.lib itself. So this verdict
    asks for a usable clang and stops there. Demanding the P1 configuration would
    guard a state the product does not yet enter (that record is `proposed`), and
    a gate watching for something nobody does is a gate that only ever misleads.

    The other facts are about that other interpreter, and this function sees
    nothing else: where it is, whether the builder script is there, and whether
    that interpreter actually carries TVM (a path that exists is not the same
    fact as an environment that can build).
    """
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
    """Can this machine build for the GPU, and what exactly stands in the way.

    A PURE function of two facts, so the verdict can be proved for every pair
    without a machine that has them - the shape targets.classify_target uses.

    WHAT IT REFUSES TO MERGE. The verdict used to be "the CPU path is ready AND
    nvcc exists", which merged three separate things and named a false cause
    twice over. It let a machine with the toolkit and NO CARD look fully ready,
    and it locked the GPU on a machine that HAS both, because the wizard happened
    to run in an interpreter without TVM - TVM the wizard needs for the ONNX
    core, not for the GPU, and the LLM builder runs in its own interpreter
    anyway. Whether the ONNX core can compile at all is reported as cpu_ready,
    separately, where it belongs.

    WHAT DECIDES. The kernels are compiled through the NVIDIA toolkit, so nvcc is
    what makes a GPU build possible at all. The CARD is a different fact: without
    one a build is still possible - that is what naming an architecture is for,
    and the builder takes --arch precisely so a bundle can be made for a card
    this machine does not have - but nothing built that way can be RUN here. So
    a missing card narrows the choice and is said out loud; it does not lock it.

    Returns ready (may the GPU be chosen), needs_arch (an architecture must be
    named because no card can supply one) and reason (what to show the user).
    """
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
                       'the architecture it is for (sm_89), and the bundle it '
                       'makes will not run on this machine'),
        }
    return {'ready': True, 'needs_arch': False, 'reason': card}


def cudnn_locations():
    """Where cuDNN is looked for. Named out loud so a miss can be argued with."""
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
    """cuDNN libraries on this machine, and the places that were searched.

    A MISS HERE IS ABOUT OUR SEARCH, NOT ABOUT THE MACHINE, so the places are
    reported alongside. Measured 31.08.2026: cuDNN 9 was present and this report
    said nothing about it, because it sat in a TVM build directory rather than in
    any of the usual homes.
    """
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
    """Whether OUR TVM can use cuDNN at all - a separate question entirely.

    cuDNN sitting on the machine does not mean the compiler can reach it: the
    wheel is built with USE_CUDNN either on or off, and ours (apache-tvm 0.26.0)
    is OFF - measured 31.08.2026. Offering a cuDNN target on such a wheel would
    be a promise the build cannot keep.
    """
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
    """Whether this Windows has a working WSL. Needed ONLY for Paddle models."""
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
    """A report about this machine. Prints nothing and changes nothing."""
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
        step.say('under Windows the paddle2onnx wheel does not load - measured 31.08.2026')
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
    step.say('driver/runtime: %s — %s' % (check['status'], check['reason']))
    
    from . import targets
    report.summary['os'] = platform.system().lower()
    report.summary['arch'] = targets._norm_arch(platform.machine())
    return report


if __name__ == '__main__':
    print(collect().as_text())
