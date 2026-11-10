# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


import ctypes
import json
import os
import re
import subprocess
import sys

from .llm_build_profile import console_python


def parse_nvcc_release(text):
    match = re.search(r'\brelease\s+(\d+)\.(\d+)\b', text or '', re.I)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def installed_toolkit_version(nvcc_path, ask_program):
    if not nvcc_path:
        return None
    first = ask_program(nvcc_path, ['--version'])
    toolkit = parse_nvcc_release(first)
    if toolkit:
        return toolkit


    try:
        result = subprocess.run([nvcc_path, '--version'],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return parse_nvcc_release(result.stdout + '\n' + result.stderr)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def driver_api_version():
    try:
        dll = (ctypes.WinDLL('nvcuda.dll') if sys.platform == 'win32'
               else ctypes.CDLL('libcuda.so.1'))
        initialized = dll.cuInit(0)
        if initialized != 0:
            return None, 'CUDA Driver API cuInit returned %s' % initialized
        version = ctypes.c_int(-1)
        result = dll.cuDriverGetVersion(ctypes.byref(version))
        if result != 0 or version.value <= 0:
            return None, 'CUDA Driver API cuDriverGetVersion returned %s' % result
        return version.value, 'CUDA Driver API answered'
    except (OSError, AttributeError, ValueError) as exc:
        return None, 'CUDA Driver API was not available: %s' % exc


def compare_driver_toolkit(driver_api, toolkit_release):
    if driver_api is None or toolkit_release is None:
        return 'unknown', 'CUDA driver or toolkit version could not be determined'
    major, minor = toolkit_release
    driver_major = driver_api // 1000
    expected = major * 1000 + minor * 10
    if driver_major < major:
        return ('incompatible', 'CUDA toolkit %d.%d requires a newer driver '
                'than the installed CUDA driver API %s' % (major, minor, driver_api))
    if driver_api < expected:
        return ('unknown', 'CUDA driver API %s is older than toolkit %d.%d; '
                'minor-version compatibility must be checked separately'
                % (driver_api, major, minor))
    return 'candidate', 'CUDA driver API %s covers toolkit %d.%d' % (
        driver_api, major, minor)


def runtime_in_toolkit(nvcc_path, toolkit_release):
    if not nvcc_path or not toolkit_release:
        return None
    root = os.path.dirname(os.path.abspath(nvcc_path))
    major = toolkit_release[0]
    expected = ('cudart64_%s.dll' % major if sys.platform == 'win32'
                else 'libcudart.so.%s' % major)
    for folder in (root, os.path.join(root, 'x64')):
        candidate = os.path.join(folder, expected)
        if os.path.isfile(candidate):
            return candidate

    candidate = os.path.join(os.path.dirname(root), 'lib64', expected)
    return candidate if os.path.isfile(candidate) else None


_RUNTIME_PROBE = r'''
import ctypes,json,sys
lib = (ctypes.WinDLL(sys.argv[1]) if sys.platform == "win32"
       else ctypes.CDLL(sys.argv[1]))
version=ctypes.c_int(-1)
count=ctypes.c_int(-1)
version_rc=lib.cudaRuntimeGetVersion(ctypes.byref(version))
count_rc=lib.cudaGetDeviceCount(ctypes.byref(count))
print(json.dumps({"version_rc":version_rc,"runtime_api":version.value,
                  "devices_rc":count_rc,"devices":count.value}))
'''


def check_cuda_toolkit(nvcc_path, ask_program, run=None):
    driver, driver_reason = driver_api_version()
    toolkit = installed_toolkit_version(nvcc_path, ask_program)
    state, reason = compare_driver_toolkit(driver, toolkit)
    report = {
        'status': state, 'reason': reason, 'driver_api': driver,
        'driver_reason': driver_reason,
        'toolkit_version': (list(toolkit) if toolkit else None),
        'runtime_version': None, 'device_count': None,
        'runtime_probe': 'not checked',
    }
    if state != 'candidate':
        return report
    library = runtime_in_toolkit(nvcc_path, toolkit)
    if not library:
        report['status'] = 'unknown'
        report['reason'] = 'Toolkit runtime library was not located for the selected nvcc'
        return report
    runner = run or subprocess.run
    try:
        result = runner([console_python(), '-B', '-c', _RUNTIME_PROBE, library],
                        capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            report['status'] = 'incompatible'
            report['reason'] = ('CUDA runtime probe process failed (exit %s)'
                                % result.returncode)
            return report
        answer = json.loads((result.stdout or '').strip())
        report['runtime_probe'] = answer
        required = ('version_rc', 'runtime_api', 'devices_rc', 'devices')
        if (not isinstance(answer, dict) or
                any(key not in answer or type(answer[key]) is not int
                    for key in required)):
            report['status'] = 'unknown'
            report['reason'] = ('CUDA runtime probe returned incomplete or invalid '
                                'fields; compatibility is NOT confirmed')
            return report
        report['runtime_version'] = answer.get('runtime_api')
        report['device_count'] = answer.get('devices')
        if answer.get('version_rc') or answer.get('devices_rc'):
            report['status'] = 'incompatible'
            report['reason'] = ('CUDA Runtime returned version status %s, '
                                'device status %s'
                                % (answer.get('version_rc'), answer.get('devices_rc')))
        elif not isinstance(answer.get('devices'), int) or answer['devices'] <= 0:
            report['status'] = 'unknown'
            report['reason'] = 'CUDA runtime answered but detected no device'
        elif answer['runtime_api'] != (toolkit[0] * 1000 + toolkit[1] * 10):
            report['status'] = 'unknown'
            report['reason'] = ('CUDA runtime version %s differs from the selected '
                                'toolkit %d.%d; a mismatched runtime must not '
                                'be certified as compatible'
                                % (answer['runtime_api'], toolkit[0], toolkit[1]))
        else:
            report['status'] = 'compatible'
            report['reason'] = 'CUDA driver and selected toolkit runtime probe passed'
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        report['status'] = 'unknown'
        report['reason'] = 'CUDA runtime probe did not complete: %s' % exc
    return report
