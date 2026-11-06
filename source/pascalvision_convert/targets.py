# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Describe build and verification support for a selected deployment target.

The classifier uses the supplied host and target facts. A CUDA build can
target another architecture, but local execution checks require a matching
device. A different operating system requires a separate build runner.
The host probe reports the first CUDA device that the TVM runtime can use.
"""

import platform
import re

LOCAL = 'LOCAL'
COMPILE_ONLY = 'COMPILE_ONLY'
CROSS_CPU = 'CROSS_CPU'
RUNNER_NEEDED = 'RUNNER_NEEDED'
UNSUPPORTED_V1 = 'UNSUPPORTED_V1'

_ARCH_ALIASES = {
    'amd64': 'x86_64',
    'x64': 'x86_64',
    'x86_64': 'x86_64',
    'em64t': 'x86_64',
    'intel64': 'x86_64',
    'arm64': 'arm64',
    'aarch64': 'arm64',
}


def _norm_os(value):
    return (value or '').strip().lower()


def _norm_arch(value):
    text = (value or '').strip().lower()
    return _ARCH_ALIASES.get(text, text)


_SM_CODE = re.compile(r'(?:sm_|compute_)?([1-9][0-9]{1,2}[a-z]?)\Z')
_SM_CAPABILITY = re.compile(r'([1-9][0-9]?)\.([0-9])\Z')


def normalize_cuda_sm(value):
    """Canonical CUDA architecture or None: sm_89, 89 and 8.9 -> sm_89.

    Validate the *whole* value before passing it to TVM. Extracting digits from
    arbitrary text used to accept typos such as 'sm_89garbage' as a real GPU.
    The format is future-compatible (no hard-coded architecture whitelist) and
    retains architecture suffixes such as sm_90a rather than silently dropping
    them. Support for an architecture is ultimately checked by the TVM/CUDA
    toolchain, not inferred from its syntactic validity.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('invalid CUDA SM %r: expected sm_89, 89 or 8.9' % value)
    text = str(value).strip().lower()
    if not text:
        return None
    code = _SM_CODE.fullmatch(text)
    if code:
        return 'sm_' + code.group(1)
    capability = _SM_CAPABILITY.fullmatch(text)
    if capability:
        return 'sm_' + capability.group(1) + capability.group(2)
    raise ValueError('invalid CUDA SM %r: expected sm_89, 89 or 8.9' % value)


def _norm_sm(value):
    """Legacy comparison helper; bad host probes mean an *unknown* GPU."""
    try:
        return normalize_cuda_sm(value) or ''
    except ValueError:
        return ''


def validate_target_sm(req_device, req_sm=None):
    """Validate device/SM together and return the canonical SM (or None).

    A CUDA SM given with a CPU build is an error, not an ignored option.
    """
    device = getattr(req_device, 'value', req_device) or 'cpu'
    device = str(device).strip().lower()
    if device not in ('cpu', 'cuda', 'cudnn'):
        raise ValueError('unknown target device %r (expected cpu, cuda or cudnn)'
                         % req_device)
    sm = normalize_cuda_sm(req_sm)
    if device == 'cpu' and sm is not None:
        raise ValueError('CUDA SM %s cannot be used with a CPU target' % sm)
    return sm


def tvm_target_spec(req_device, req_sm=None):
    """The *actual* TVM target string used by both model frontends."""
    sm = validate_target_sm(req_device, req_sm)
    device = getattr(req_device, 'value', req_device) or 'cpu'
    if str(device).strip().lower() == 'cpu':
        return 'llvm'
    return 'cuda -arch=' + sm if sm else 'cuda'


def recommend_target(host_facts, req_device=None, req_sm=None):
    """Pure machine-first suggestion with an explicit user override.

    host_facts is the shape returned by host(). The detected CUDA architecture
    is offered as a *build target* when available, otherwise generic CPU is
    offered. UI callers still need to check the CUDA toolchain is installed;
    the hardware probe alone does not establish that it can compile.
    """
    detected_sm = _norm_sm(host_facts.get('cuda_arch')) or None
    requested_sm = normalize_cuda_sm(req_sm)
    if req_device is None:
        device = ('cuda' if requested_sm or detected_sm else 'cpu')
    else:
        device = getattr(req_device, 'value', req_device)
    sm = validate_target_sm(device, requested_sm)
    if str(device).strip().lower() in ('cuda', 'cudnn') and sm is None:
        sm = detected_sm
    device = str(device).strip().lower()
    level, reason = classify_target(
        req_os=host_facts['os'], req_arch=host_facts['arch'],
        req_device=device, req_sm=sm,
        host_os=host_facts['os'], host_arch=host_facts['arch'],
        host_cuda_arch=host_facts.get('cuda_arch'))
    return {
        'target': device,
        'req_sm': sm,
        'level': level,
        'reason': reason,
        'source': ('explicit' if req_device is not None or requested_sm is not None
                   else 'detected'),
        'cuda_toolchain_check_required': device in ('cuda', 'cudnn'),
    }

TARGET_PROFILE_VERSION = 1


def _cpu_codegen_module():
    """Lazy LLVM codegen module; keeping imports local preserves stdlib callers."""
    from tvm.target import codegen
    return codegen


def admit_cpu_codegen(mode='host', mcpu=None, *, host_arch=None, llvm_codegen=None):
    """Admit an x86_64 LLVM CPU codegen target, without claiming destination UAT.

    BuildPlan uses this check before admitting an explicit tuned CPU target.
    A valid LLVM CPU name identifies a compiler target. Runtime compatibility
    requires a separate check on the machine that will execute the module.

    `host` retains the historically optimized tuned target, `generic` uses the
    x86-64 baseline, and `explicit` requires a caller-selected LLVM CPU name.
    An explicit request is never replaced with the host CPU after a failure.
    """
    if type(mode) is not str or mode not in ('host', 'generic', 'explicit'):
        raise ValueError('CPU codegen mode must be host, generic or explicit')
    if _norm_arch(host_arch or platform.machine()) != 'x86_64':
        raise ValueError('CPU mcpu admission currently requires x86_64 build host')
    if mode != 'explicit' and mcpu is not None:
        raise ValueError('CPU mcpu is only accepted in explicit mode')
    if mode == 'explicit' and (type(mcpu) is not str or
                               not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]{0,79}', mcpu)):
        raise ValueError('explicit CPU mcpu must be one bounded LLVM CPU name')
    if mode == 'generic':
        return {'mode': mode, 'mcpu': None, 'tvm_target': 'llvm',
                'destination_compatibility': 'NOT_CHECKED'}
    if llvm_codegen is None:
        try:
            llvm_codegen = _cpu_codegen_module()
        except (ImportError, AttributeError) as exc:
            raise ValueError('LLVM CPU admission unavailable; no target fallback') from exc
    try:
        triple = str(llvm_codegen.llvm_get_system_triple() or '').strip()
        actual = (str(llvm_codegen.llvm_get_system_cpu() or '').strip()
                  if mode == 'host' else mcpu)
        valid = bool(triple and actual and
                     llvm_codegen.llvm_is_valid_cpu(actual, triple))
    except Exception as exc:
        raise ValueError('LLVM CPU target could not be validated; no fallback') from exc
    if not valid:
        raise ValueError('unsupported LLVM CPU mcpu %r; no host fallback' % actual)
    return {'mode': mode, 'mcpu': actual,
            'tvm_target': 'llvm -mcpu=' + actual,
            'destination_compatibility': 'NOT_CHECKED'}


def require_local_cpu_tuning_target(selected, *, llvm_codegen=None):
    """Refuse foreign explicit mcpu BEFORE any local MetaSchedule runner runs.

    LLVM accepting a CPU *name* proves a compiler can emit that ISA, not that
    the host can execute generated kernels.  The current tuning pipeline runs
    native candidates on its build host.  Until a separate destination tuner
    is implemented, an explicit CPU must match the observed LLVM host CPU.
    That equality is a conservative admission rule, NOT a destination feature
    attestation or a native numerical verification.

    `admit_cpu_codegen` intentionally remains the pure syntax/LLVM-target
    capability check; it may describe a foreign compiler intention.  A real
    tuned build has the additional requirement enforced here and at the
    compiler entry, so no public caller can bypass this runtime safety gate.
    """
    if type(selected) is not dict or selected.get('mode') not in (
            'host', 'generic', 'explicit'):
        raise ValueError('CPU tuning target is not an admitted mode')
    if selected['mode'] != 'explicit':
        return selected
    requested = selected.get('mcpu')
    if type(requested) is not str or not requested:
        raise ValueError('explicit CPU tuning target lacks an LLVM CPU name')
    try:
        codegen = llvm_codegen if llvm_codegen is not None else _cpu_codegen_module()
        triple = str(codegen.llvm_get_system_triple() or '').strip()
        measured = str(codegen.llvm_get_system_cpu() or '').strip()
        valid = bool(triple and measured and
                     codegen.llvm_is_valid_cpu(measured, triple))
    except Exception as exc:
        raise ValueError('local CPU tuning compatibility could not be measured; '
                         'no foreign ISA fallback') from exc
    if not valid:
        raise ValueError('local LLVM host CPU/triple could not be validated; '
                         'no foreign ISA fallback')
    if measured != requested:
        raise ValueError('cannot tune explicit CPU %r on local LLVM CPU %r: '
                         'MetaSchedule executes generated kernels here; a '
                         'separate matching CPU tuner/runner is required'
                         % (requested, measured))
    return selected


def normalize_driver_api(value):
    """CUDA Driver API number (e.g. 12080), or None when it is unknown."""
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError('target CUDA driver API must be an integer version')
    if isinstance(value, int):
        version = value
    elif isinstance(value, str) and re.fullmatch(r'[0-9]{4,6}', value.strip()):
        version = int(value.strip())
    else:
        raise ValueError('target CUDA driver API must be an integer like 12080')
    if version < 1000:
        raise ValueError('target CUDA driver API is not a valid version')
    return version


def resolve_target_profile(host_facts, req_device='cpu', req_sm=None,
                           req_os=None, req_arch=None, req_driver_api=None):
    """Produce one versioned target description for preview/build/recipe.

    Every piece comes from observed host facts or an explicit user override.
    This is deliberately independent of GPU/Toolkit discovery and can be
    exercised with mock hosts without silently selecting the developer's GPU.
    """
    host_os = _norm_os(host_facts.get('os'))
    host_arch = _norm_arch(host_facts.get('arch'))
    host_sm = _norm_sm(host_facts.get('cuda_arch')) or None
    if not host_os or not host_arch:
        raise ValueError('build host OS and CPU architecture must be known')
    target_os = _norm_os(req_os or host_os)
    target_arch = _norm_arch(req_arch or host_arch)
    device = getattr(req_device, 'value', req_device) or 'cpu'
    device = str(device).strip().lower()
    selected_sm = validate_target_sm(device, req_sm)
    target_driver_api = normalize_driver_api(req_driver_api)
    if device == 'cpu' and target_driver_api is not None:
        raise ValueError('CUDA driver API cannot be used with a CPU target')
    if device in ('cuda', 'cudnn') and selected_sm is None:
        selected_sm = host_sm
    level, reason = classify_target(
        req_os=target_os, req_arch=target_arch, req_device=device,
        req_sm=selected_sm, host_os=host_os, host_arch=host_arch,
        host_cuda_arch=host_sm)
    deployment = {
        'os': target_os, 'arch': target_arch, 'device': device,
        'sm': selected_sm,
    }
    if target_driver_api is not None:
        deployment['driver_api'] = target_driver_api
    return {
        'target_profile_version': TARGET_PROFILE_VERSION,
        'build_host': {
            'os': host_os, 'arch': host_arch, 'gpu_sm': host_sm,
        },
        'deployment': deployment,
        'reachability': {'level': level, 'reason': reason},
        'requested_tvm_target': tvm_target_spec(device, selected_sm),
    }


def validate_recorded_profile(profile, target=None, req_sm=None):
    """Reject tampered or contradictory versioned target profiles.

    The profile is a recipe input, not a security credential. This check
    detects inconsistent edits where a user changes only the top-level SM,
    OS, architecture or TVM target string, accidentally replaying another
    architecture while believing the original target is being reproduced.
    """
    if not isinstance(profile, dict):
        raise ValueError('target_profile must be an object')
    if profile.get('target_profile_version') != TARGET_PROFILE_VERSION:
        raise ValueError('unknown target_profile_version %r'
                         % profile.get('target_profile_version'))
    host = profile.get('build_host')
    deploy = profile.get('deployment')
    if not isinstance(host, dict) or not isinstance(deploy, dict):
        raise ValueError('target_profile has no build_host or deployment object')
    try:
        expected = resolve_target_profile(
            {'os': host['os'], 'arch': host['arch'],
             'cuda_arch': host.get('gpu_sm')},
            req_device=deploy['device'], req_sm=deploy.get('sm'),
            req_os=deploy['os'], req_arch=deploy['arch'],
            req_driver_api=deploy.get('driver_api'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid target_profile: %s' % exc) from exc
    for field in ('build_host', 'deployment', 'reachability',
                  'requested_tvm_target'):
        if profile.get(field) != expected[field]:
            raise ValueError('target_profile %s differs from resolved target'
                             % field)
    if target is not None and str(target).strip().lower() != deploy['device']:
        raise ValueError('recipe target and target_profile device disagree')
    if req_sm is not None or target is not None:
        canonical = validate_target_sm(deploy['device'], req_sm)
        if canonical != deploy.get('sm'):
            raise ValueError('recipe SM and target_profile architecture disagree')
    return profile


def classify_target(req_os, req_arch, req_device, req_sm,
                    host_os, host_arch, host_cuda_arch):
    """Classify a requested build target against a host. PURE.

    req_os, req_arch, req_device, req_sm
        the target: an OS, a CPU architecture, a device ("cpu" or "cuda"), and -
        for CUDA - a specific sm such as "sm_89" (empty or None when no particular
        card is named).
    host_os, host_arch, host_cuda_arch
        the machine that would build it: its OS, its CPU architecture, and the CUDA
        architecture of its first card, or None when there is none (or none could
        be detected).

    Returns (level, reason): one of the five module constants and a short honest
    sentence. Nothing here reads the running machine - the host is entirely in the
    arguments - so a given target and host always yield the same verdict.
    """
    arch_match = _norm_arch(req_arch) == _norm_arch(host_arch)
    os_match = _norm_os(req_os) == _norm_os(host_os)
    device = getattr(req_device, 'value', req_device) or 'cpu'
    device = str(device).strip().lower()
    req_sm = validate_target_sm(device, req_sm)

    if not arch_match:
        return (UNSUPPORTED_V1,
                "target architecture %s is not this host's %s; a foreign CPU "
                "architecture needs a cross-toolchain and a matching sysroot, "
                "which is not part of the local v1" % (req_arch, host_arch))

    if not os_match:
        return (RUNNER_NEEDED,
                "target OS %s is not this host's %s; building for another OS "
                "needs a native, VM or CI runner of it, which the local v1 does "
                "not orchestrate" % (req_os, host_os))


    if device in ('cuda', 'cudnn'):
        return _classify_cuda(req_sm, host_cuda_arch)

    return (LOCAL,
            "a CPU target on this host's own OS and architecture; it builds, "
            "tunes and verifies here")


def _classify_cuda(req_sm, host_cuda_arch):
    """LOCAL when the host's card matches the request, COMPILE_ONLY otherwise.

    Compilation of a CUDA module does not need a card (TVM emits the kernels), but
    tuning, benchmarking and verify all RUN on the device - so a card that is
    absent or of another sm turns a reachable build into a compile-only one.
    """
    want = _norm_sm(req_sm)
    have = _norm_sm(host_cuda_arch)

    if have and (not want or want == have):
        if want:
            reason = ("this host has a CUDA card (%s) matching the requested %s, "
                      "so it builds, tunes and verifies here"
                      % (host_cuda_arch, req_sm))
        else:
            reason = ("this host has a CUDA card (%s) and no particular sm was "
                      "requested, so it builds, tunes and verifies here"
                      % host_cuda_arch)
        return (LOCAL, reason)

    if not have:
        gap = "the current runtime could not use a local CUDA device"
    else:
        gap = ("this host's CUDA card is %s, not the requested %s"
               % (host_cuda_arch, req_sm))
    built_for = ("for %s " % req_sm) if req_sm else ""
    return (COMPILE_ONLY,
            "a CUDA module %scompiles on this host, but %s; tuning, benchmarking "
            "and verify run ON the card and need that exact one, so they cannot "
            "happen here" % (built_for, gap))


def host():
    """This machine's own facts: OS, CPU architecture, and its CUDA sm if any.

    NOT PURE, unlike classify_target: it reads the running machine and may import
    TVM to ask about the card. That is why it is a separate function - so the
    verdict logic stays pure and testable while the machine-reading lives in one
    place.

        os         platform.system() lowered - "windows", "linux", "darwin"
        arch       platform.machine() normalised - "AMD64" and "x86_64" both come
                   back "x86_64", so a Windows and a Linux x86 host agree
        cuda_arch  the first CUDA card's architecture ("sm_89"), or None

    cuda_arch is BEST EFFORT. Detecting it needs TVM, which the converter does not
    require to be installed; when TVM is absent, or no card is present, or the
    query raises for any reason, cuda_arch is None. A None is not an error - it
    reads downstream as "no local card", the honest thing to say when none was
    found.
    """
    return {
        'os': _norm_os(platform.system()),
        'arch': _norm_arch(platform.machine()),
        'cuda_arch': _detect_cuda_arch(),
    }


def _detect_cuda_arch():
    """The first CUDA card's architecture through TVM, or None. Never raises.

    Mirrors how llm/build_llm.py names a card: tvm.cuda(0).exist to see whether one
    is there, then Target.from_device to read its arch. Every failure - TVM not
    installed, no driver, no card, an API that has moved - collapses to None,
    because the caller needs one plain answer: is there a local card we can name.
    """
    try:
        import tvm
    except Exception:
        return None
    try:
        device = tvm.cuda(0)
        if not getattr(device, 'exist', False):
            return None
        target = tvm.target.Target.from_device(device)
        arch = target.attrs.get('arch')
        return str(arch) if arch else None
    except Exception:
        return None
