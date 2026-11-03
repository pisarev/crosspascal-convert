# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""One versioned *intention* for the wizard, CLI, core and family builders.

The TargetProfile resolver in targets.py remains the ONLY target classifier.
BuildPlan adds the actual user choices and a deterministic accidental-change
digest; it does not claim immutable source bytes or prove a model ran. A client
must still bind HTTP Preview to source SHA and its one-use token, and a native
verification result must be independently earned after the build.
"""

import hashlib
import json
import os
import stat
from pathlib import Path

from . import targets


BUILD_PLAN_VERSION = 1
_ROUTES = frozenset(('raw', 'family_full_build'))
_DTYPES = {
    'raw': frozenset(('float32',)),
    'family_full_build': frozenset(('float32', 'float16')),
}


def _checksum(record):
    body = {key: value for key, value in record.items()
            if key != 'plan_sha256'}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode('utf-8')).hexdigest()


def resolve(host_facts, *, target='cpu', req_sm=None, req_os=None,
            req_arch=None, req_driver_api=None, route='raw', source=None,
            shape=None, dtype='float32', tune=False, trials_per_task=None,
            contract=None, cpu_codegen=None):
    """Resolve effective target and selections; refuse unsupported requests.

    No numerical success is possible here. A saved plan is valid only as a
    replayable request; its SHA is NOT an authenticating signature against a
    malicious editor, just an accidental/stale-field consistency guard.
    """
    if route not in _ROUTES:
        raise ValueError('unsupported BuildPlan route %r' % route)
    if type(target) is not str or target.strip().lower() not in ('cpu', 'cuda'):
        raise ValueError('BuildPlan target must be cpu or cuda')
    target = target.strip().lower()
    if type(dtype) is not str or dtype not in _DTYPES[route]:
        raise ValueError('BuildPlan %s does not implement dtype %r' % (route, dtype))
    if source is not None and (type(source) is not str or not source.strip()):
        raise ValueError('BuildPlan source must be a nonempty string or None')
    if shape is not None and type(shape) is not str:
        raise ValueError('BuildPlan shape must be a string or None')
    if contract is not None and not isinstance(contract, dict):
        raise ValueError('BuildPlan model contract must be an object or None')
    if type(tune) is not bool:
        raise ValueError('BuildPlan tune must be boolean')
    if trials_per_task is not None and (type(trials_per_task) is not int
                                        or trials_per_task <= 0):
        raise ValueError('BuildPlan trials_per_task must be a positive integer')
    
    
    
    if cpu_codegen is not None and (route != 'raw' or target != 'cpu' or not tune):
        raise ValueError('CPU codegen choice requires tuned raw CPU route')
    if route == 'family_full_build':
        
        
        
        if shape or contract or tune or trials_per_task is not None:
            raise ValueError('LLM family builder cannot honour a model shape, '
                             'vision contract or tuning controls in this route')
        if req_driver_api is not None:
            raise ValueError('LLM family builder does not implement a selected '
                             'CUDA driver API: do not ignore that override')

    admitted_cpu = None
    if cpu_codegen is not None:
        if (type(cpu_codegen) is not dict or
                set(cpu_codegen) != {'mode', 'mcpu'}):
            raise ValueError('CPU codegen choice requires exact mode and mcpu')
        if route != 'raw' or target != 'cpu' or not tune:
            raise ValueError('CPU codegen choice requires tuned raw CPU route')
        admitted_cpu = targets.admit_cpu_codegen(
            cpu_codegen['mode'], cpu_codegen['mcpu'],
            host_arch=host_facts.get('arch'))
        
        
        
        targets.require_local_cpu_tuning_target(admitted_cpu)
        
        
        
        
        from . import tuning
        try:
            tuning.refuse_on_windows()
        except RuntimeError as exc:
            raise ValueError('selected CPU codegen tuning unavailable: %s' % exc) from exc

    profile = targets.resolve_target_profile(
        host_facts, req_device=target, req_sm=req_sm,
        req_os=req_os, req_arch=req_arch,
        req_driver_api=req_driver_api)
    level = profile['reachability']['level']
    if level in (targets.RUNNER_NEEDED, targets.UNSUPPORTED_V1):
        raise ValueError(profile['reachability']['reason'])
    if target == 'cuda' and not profile['deployment']['sm']:
        raise ValueError('CUDA target SM is unknown: no CUDA card architecture '
                         'was measured and none was selected; choose the target '
                         'SM before Build, without a CPU fallback')

    output = {
        'build_plan_version': BUILD_PLAN_VERSION,
        'route': route,
        'source': source,
        'target_profile': profile,
        'compiler_target': profile['requested_tvm_target'],
        'options': {
            'shape': shape,
            'dtype': dtype,
            'tune': tune,
            'trials_per_task': trials_per_task,
            'contract': contract,
        },
        'verification_status': 'NOT_CHECKED',
        'verification_tier': (
            'FAMILY_ORACLE_NOT_IMPLEMENTED' if route == 'family_full_build' else
            'DESTINATION_ORACLE_REQUIRED' if level == targets.COMPILE_ONLY else
            'LOCAL_ORACLE_PENDING'),
    }
    if admitted_cpu is not None:
        
        
        
        output['cpu_codegen'] = admitted_cpu
        output['compiler_target'] = admitted_cpu['tvm_target']
        output['compiler_binding_status'] = 'PENDING_ACTUAL_CODEGEN'
    output['plan_sha256'] = _checksum(output)
    return output


def validate_recorded(plan, *, source=None, target=None, req_sm=None,
                      target_profile=None, shape=None, dtype=None,
                      tune=None, trials_per_task=None, contract=None):
    """Validate a stored plan against its own inputs and optional recipe fields.

    Cannot validate the *current* source/model bytes: that belongs to the
    later immutable-source workflow. Native VERIFIED is NEVER a BuildPlan field.
    """
    if not isinstance(plan, dict):
        raise ValueError('build_plan must be a JSON object')
    if plan.get('build_plan_version') != BUILD_PLAN_VERSION:
        raise ValueError('unknown build_plan_version')
    if plan.get('verification_status') != 'NOT_CHECKED':
        raise ValueError('BuildPlan must not claim model execution or VERIFIED')
    if plan.get('plan_sha256') != _checksum(plan):
        raise ValueError('BuildPlan digest differs from recorded choices')
    profile = plan.get('target_profile')
    if not isinstance(profile, dict) or not isinstance(profile.get('build_host'), dict):
        raise ValueError('BuildPlan lacks recorded host and target profile')
    deploy = profile.get('deployment')
    if not isinstance(deploy, dict):
        raise ValueError('BuildPlan lacks recorded deployment target')
    options = plan.get('options')
    if not isinstance(options, dict):
        raise ValueError('BuildPlan options must be an object')
    selected_cpu = plan.get('cpu_codegen')
    if selected_cpu is not None:
        if (type(selected_cpu) is not dict or
                set(selected_cpu) != {'mode', 'mcpu', 'tvm_target',
                                      'destination_compatibility'} or
                plan.get('compiler_binding_status') != 'PENDING_ACTUAL_CODEGEN'):
            raise ValueError('selected CPU compiler target is not pending/bound')
    elif 'compiler_binding_status' in plan:
        raise ValueError('unexpected CPU compiler binding status')
    built = resolve(
        {'os': profile['build_host'].get('os'),
         'arch': profile['build_host'].get('arch'),
         'cuda_arch': profile['build_host'].get('gpu_sm')},
        target=deploy.get('device'), req_sm=deploy.get('sm'),
        req_os=deploy.get('os'), req_arch=deploy.get('arch'),
        req_driver_api=deploy.get('driver_api'),
        route=plan.get('route'), source=plan.get('source'),
        shape=options.get('shape'), dtype=options.get('dtype'),
        tune=options.get('tune'), trials_per_task=options.get('trials_per_task'),
        contract=options.get('contract'),
        cpu_codegen=({'mode': selected_cpu['mode'],
                      'mcpu': (selected_cpu['mcpu'] if selected_cpu['mode'] == 'explicit'
                               else None)} if selected_cpu is not None else None))
    if built != plan:
        raise ValueError('BuildPlan differs from effective target or options')
    if source is not None and source != plan.get('source'):
        raise ValueError('recipe source and BuildPlan source disagree')
    if target is not None and target != deploy['device']:
        raise ValueError('recipe target and BuildPlan target disagree')
    if req_sm is not None and targets.validate_target_sm(deploy['device'], req_sm) != deploy.get('sm'):
        raise ValueError('recipe SM and BuildPlan SM disagree')
    if target_profile is not None and target_profile != profile:
        raise ValueError('recipe target_profile and BuildPlan target_profile disagree')
    if shape is not None and shape != options.get('shape'):
        raise ValueError('recipe shape and BuildPlan shape disagree')
    if dtype is not None and dtype != options.get('dtype'):
        raise ValueError('recipe dtype and BuildPlan dtype disagree')
    if tune is not None and tune != options.get('tune'):
        raise ValueError('recipe tuning and BuildPlan tuning disagree')
    if trials_per_task is not None and trials_per_task != options.get('trials_per_task'):
        raise ValueError('recipe tuning budget and BuildPlan budget disagree')
    if contract is not None and contract != options.get('contract'):
        raise ValueError('recipe model contract and BuildPlan contract disagree')
    return plan


def validate_native_family_execution(plan, *, source, target, req_sm,
                                     dtype, host_os, host_arch,
                                     actual_tvm_target=None):
    """Native HF builder-side validation, separate from wizard assurances.

    Refuse *before loading weights* when the compiler process, actual local OS,
    ISA, requested GPU and data source differ from the reviewed BuildPlan.
    A plan is still not a model-source lock, UAT or a verification passport.
    """
    validate_recorded(plan, source=source, target=target,
                      req_sm=req_sm, dtype=dtype)
    if plan['route'] != 'family_full_build':
        raise ValueError('native LLM process requires a family_full_build plan')
    profile = plan['target_profile']
    requested_sm = targets.validate_target_sm(target, req_sm)
    if requested_sm != profile['deployment'].get('sm'):
        raise ValueError('native builder CLI CUDA SM differs from approved '
                         'BuildPlan (missing --arch may silently select GPU0)')
    host = profile['build_host']
    local_os = targets._norm_os(host_os)
    local_arch = targets._norm_arch(host_arch)
    if not local_os or not local_arch:
        raise ValueError('native builder cannot establish host OS/CPU architecture')
    if host['os'] != local_os or host['arch'] != local_arch:
        raise ValueError('native builder host OS/ISA differs from source BuildPlan')
    deployment = profile['deployment']
    if deployment['os'] != local_os or deployment['arch'] != local_arch:
        raise ValueError('native builder cannot compile for foreign OS/CPU ISA '
                         'without a measured matching toolchain and runner')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('native builder requires a local HF source folder')
    if (not isinstance(plan['source'], str) or
            os.path.normcase(os.path.abspath(source)) !=
            os.path.normcase(os.path.abspath(plan['source']))):
        raise ValueError('native builder source differs from approved BuildPlan')
    if actual_tvm_target is not None:
        expected = profile['requested_tvm_target']
        if actual_tvm_target != expected:
            raise ValueError('native compiled TVM target %r differs from '
                             'approved BuildPlan target %r'
                             % (actual_tvm_target, expected))
    return plan


def read_native_plan(path, *, expected_sha256=None):
    """Load a bounded exact plan without duplicate JSON keys or NaN.

    Only the native family builder consumes this transfer file; the outer
    wizard stores an independently audited copy with its own bundle receipt.
    """
    candidate = Path(path).absolute()
    if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
        raise ValueError('linked native BuildPlan transfer file is untrusted')
    try:
        info = candidate.stat()
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 1024 * 1024:
            raise ValueError('native BuildPlan must be a regular file <= 1MiB')
        
        
        
        
        
        with candidate.open('rb') as stream:
            before_read = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before_read.st_mode) or
                    (before_read.st_dev, before_read.st_ino,
                     before_read.st_size) !=
                    (info.st_dev, info.st_ino, info.st_size)):
                raise ValueError('native BuildPlan changed while reading')
            remaining = info.st_size
            parts = []
            while remaining:
                part = stream.read(min(64 * 1024, remaining))
                if not part or len(part) > remaining:
                    raise ValueError('native BuildPlan changed while reading')
                parts.append(part)
                remaining -= len(part)
            if stream.read(1):
                raise ValueError('native BuildPlan changed while reading')
            after_read = os.fstat(stream.fileno())
        if ((before_read.st_dev, before_read.st_ino,
             before_read.st_size, before_read.st_mtime_ns) !=
                (after_read.st_dev, after_read.st_ino,
                 after_read.st_size, after_read.st_mtime_ns)):
            raise ValueError('native BuildPlan changed while reading')
        raw = b''.join(parts)
        if expected_sha256 is not None:
            if (type(expected_sha256) is not str or
                    len(expected_sha256) != 64 or
                    any(ch not in '0123456789abcdef' for ch in expected_sha256)):
                raise ValueError('native BuildPlan approved SHA-256 is invalid')
            if hashlib.sha256(raw).hexdigest() != expected_sha256:
                
                
                raise ValueError(
                    'native BuildPlan content changed since initial ZIP inventory')
    except OSError as exc:
        raise ValueError('native BuildPlan is unavailable: %s' % exc) from exc

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate BuildPlan JSON key %r' % key)
            result[key] = value
        return result

    def no_constants(value):
        raise ValueError('nonfinite BuildPlan JSON value %r' % value)

    try:
        plan = json.loads(raw.decode('utf-8'),
                          object_pairs_hook=no_duplicates,
                          parse_constant=no_constants)
    except (UnicodeError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError('native BuildPlan malformed: %s' % exc) from exc
    return validate_recorded(plan)
