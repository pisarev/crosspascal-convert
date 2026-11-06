# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Recognise a model file by its bytes.

The TFLite identifier sits at offset four, after the FlatBuffer root-table
offset. PyTorch weights need an ONNX export before they can be compiled.
Unrecognised files are reported with their leading bytes for diagnosis.
"""

import os
import posixpath
import re
import stat
import struct
import sys
import zipfile

ONNX = 'onnx'
PADDLE = 'paddle'
TFLITE = 'tflite'
PYTORCH = 'pytorch'
SAFETENSORS = 'safetensors'
UNKNOWN = 'unknown'


TFLITE_MARKER = b'TFL3'
TFLITE_MARKER_OFFSET = 4

ZIP_MARKER = b'PK' + bytes([3, 4])


PADDLE_WEIGHTS = 'inference.pdiparams'
PADDLE_GRAPHS = ('inference.json', 'inference.pdmodel')
PADDLE_PIR_MARKER = b'"magic":"pir"'


PICKLE_FIRST = 0x80
PICKLE_PROTOCOLS = (2, 3, 4, 5)


PYTORCH_FAMILIES = (
    (b'ultralytics', 'ultralytics',
     'yolo export model=MODEL.pt format=onnx opset=12'),
    (b'models.yolo', 'yolov5',
     'python export.py --weights MODEL.pt --include onnx --opset 12'),
    (b'models.common', 'yolov5',
     'python export.py --weights MODEL.pt --include onnx --opset 12'),
)


MAX_PT_ARCHIVE_MEMBERS = 8192
MAX_PT_PICKLE_METADATA_BYTES = 128 * 1024 * 1024
MAX_PT_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024


MAX_ONNX_PREPARSE_BYTES = 8 * 1024 ** 3
SMALL_ONNX_PREPARSE_BYTES = 64 * 1024 ** 2
MIN_LARGE_ONNX_FREE_RAM_BYTES = 5 * 1024 ** 3
ONNX_PREPARSE_RAM_MULTIPLIER = 8
ONNX_PREPARSE_RAM_HEADROOM_BYTES = 512 * 1024 ** 2


def _linux_onnx_available_bytes(*, read_text=None, rlimit_as=None,
                                 page_bytes=None):
    """Conservative Linux process headroom, not host MemFree or a RAM promise.

    Use reclaimable-cache-aware MemAvailable, the visible cgroup v2/v1 memory
    hierarchy (including finite ancestors), and finite RLIMIT_AS minus actual
    virtual memory size. This read-only admission is intentionally NOT an
    immutable-source lock, guaranteed RSS bound or numerical model test.
    Tests may supply a synthetic /proc/cgroup reader on non-Linux hosts.
    """
    if read_text is None:
        def read_text(path):
            with open(path, 'r', encoding='ascii', errors='strict') as source:
                return source.read(256 * 1024 + 1)

    def small(path, maximum=256 * 1024):
        try:
            value = read_text(path)
        except (OSError, UnicodeError, ValueError):
            return None
        return value if isinstance(value, str) and len(value) <= maximum else None

    meminfo = small('/proc/meminfo', 64 * 1024)
    if meminfo is None:
        return None
    match = re.search(r'(?m)^MemAvailable:[ \t]*([0-9]+)[ \t]+kB[ \t]*$',
                      meminfo)
    if match is None or int(match.group(1)) <= 0:
        return None
    available = int(match.group(1)) * 1024

    members = small('/proc/self/cgroup', 64 * 1024)
    mounts = small('/proc/self/mountinfo', 256 * 1024)
    if members is None or mounts is None:
        return None

    v2 = None
    v1 = None
    for line in members.splitlines():
        row = line.split(':', 2)
        if len(row) != 3:
            return None
        if row[0] == '0' and not row[1]:
            v2 = row[2]
        if 'memory' in row[1].split(','):
            v1 = row[2]
    mode, member = (('v2', v2) if v2 is not None else
                    ('v1', v1) if v1 is not None else (None, None))

    if mode is not None:
        def mount_path(value):

            value = re.sub(r'\\([0-7]{3})',
                           lambda m: chr(int(m.group(1), 8)), value)
            if (not value.startswith('/') or '\x00' in value or
                    '..' in value.split('/')):
                return None
            return posixpath.normpath(value)

        selected = None
        member = mount_path(member)
        if member is None:
            return None
        for line in mounts.splitlines():
            halves = line.split(' - ', 1)
            if len(halves) != 2:
                continue
            left, right = halves[0].split(), halves[1].split()
            if len(left) < 5 or len(right) < 3:
                continue
            if mode == 'v2' and right[0] != 'cgroup2':
                continue
            if mode == 'v1' and (right[0] != 'cgroup' or
                                  'memory' not in right[2].split(',')):
                continue
            root, mounted = mount_path(left[3]), mount_path(left[4])
            if root is None or mounted is None:
                continue
            if member != root and not member.startswith(root.rstrip('/') + '/'):
                continue
            if selected is None or len(root) > len(selected[0]):
                selected = (root, mounted)


        if selected is None:
            return None
        root, mounted = selected
        suffix = member[len(root):].lstrip('/') if root != '/' else member.lstrip('/')
        group = posixpath.normpath(posixpath.join(mounted, suffix))
        if group != mounted and not group.startswith(mounted.rstrip('/') + '/'):
            return None

        limit_name, usage_name = (
            ('memory.max', 'memory.current') if mode == 'v2' else
            ('memory.limit_in_bytes', 'memory.usage_in_bytes'))
        for _ in range(128):
            raw_limit = small(posixpath.join(group, limit_name), 4096)
            if raw_limit is None:
                return None
            raw_limit = raw_limit.strip()
            if raw_limit == 'max' and mode == 'v2':
                limit = None
            elif re.fullmatch(r'[0-9]+', raw_limit):
                limit = int(raw_limit)

                if mode == 'v1' and limit >= (1 << 60):
                    limit = None
            else:
                return None
            if limit is not None:
                raw_usage = small(posixpath.join(group, usage_name), 4096)
                if raw_usage is None or not re.fullmatch(
                        r'[0-9]+', raw_usage.strip()):
                    return None
                remaining = max(0, limit - int(raw_usage.strip()))
                available = min(available, remaining)
            if group == mounted:
                break
            group = posixpath.dirname(group)
        else:
            return None

    if rlimit_as is None:
        try:
            import resource
            rlimit_as = resource.getrlimit(resource.RLIMIT_AS)
        except (ImportError, AttributeError, OSError, ValueError):
            return None
    if not isinstance(rlimit_as, (tuple, list)) or len(rlimit_as) != 2:
        return None
    soft = rlimit_as[0]
    if type(soft) is not int:
        return None

    if 0 <= soft < (1 << 60):
        if page_bytes is None:
            try:
                page_bytes = int(os.sysconf('SC_PAGE_SIZE'))
            except (OSError, TypeError, ValueError, AttributeError):
                return None
        if type(page_bytes) is not int or not 512 <= page_bytes <= 65536:
            return None
        statm = small('/proc/self/statm', 4096)
        if statm is None or not statm.split() or not re.fullmatch(
                r'[0-9]+', statm.split()[0]):
            return None
        virtual_size = int(statm.split()[0]) * page_bytes
        available = min(available, max(0, soft - virtual_size))
    return available


def _available_physical_memory_bytes():
    """Best-effort portable *physical* availability; None is not approval."""
    if os.name == 'nt':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        try:
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
            kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
            if kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, ValueError):
            pass
        return None

    if sys.platform.startswith('linux'):
        return _linux_onnx_available_bytes()


    try:
        import psutil
        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        pages = int(os.sysconf('SC_AVPHYS_PAGES'))
        size = int(os.sysconf('SC_PAGE_SIZE'))
        if pages > 0 and size > 0:
            return pages * size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None


class OnnxAdmissionError(ValueError):
    """Inspection was deferred by a capacity guard; no graph was parsed."""


def _admit_onnx_preparse(path):
    """Refuse unsupported size/RAM BEFORE any full ONNX protobuf allocation.

    This check estimates capacity before parsing. The converter validates the
    selected source and any external data separately.
    """
    source = os.stat(path)
    if not stat.S_ISREG(source.st_mode) or source.st_size <= 0:
        raise ValueError('ONNX pre-parse needs a nonempty regular source file')
    size = source.st_size
    if size > MAX_ONNX_PREPARSE_BYTES:
        raise OnnxAdmissionError(
            'ONNX pre-parse source size exceeds bounded %d-byte '
            'protobuf admission quota' % MAX_ONNX_PREPARSE_BYTES)
    if size > SMALL_ONNX_PREPARSE_BYTES:
        available = _available_physical_memory_bytes()
        needed = max(MIN_LARGE_ONNX_FREE_RAM_BYTES,
                     ONNX_PREPARSE_RAM_MULTIPLIER * size +
                     ONNX_PREPARSE_RAM_HEADROOM_BYTES)
        if available is None or available < needed:
            raise OnnxAdmissionError(
                'ONNX pre-parse cannot establish sufficient free '
                'physical RAM before materializing %d bytes: '
                'need at least %d, observed %s' % (size, needed, available))
    return size


class Recognised(object):
    """What the file is, and - when we cannot use it - what to do about it."""

    def __init__(self, kind, detail='', advice=(), family='', status=''):
        self.kind = kind
        self.detail = detail
        self.advice = list(advice)
        self.family = family
        self.status = status

    @property
    def convertible(self):
        return self.kind in (ONNX, TFLITE)

    def as_dict(self):
        answer = {
            'kind': self.kind,
            'detail': self.detail,
            'advice': list(self.advice),
            'family': self.family,
            'convertible': self.convertible,
        }
        if self.status:
            answer['status'] = self.status
        return answer


def _paddle_answer(where):
    """PaddlePaddle: a real road, but it goes through ONNX and not from here."""
    advice = [
        'PaddlePaddle inference models are converted through an intermediate ONNX.',
        'The export command is:',
        '    paddle2onnx --model_dir DIR --model_filename inference.json'
        ' --params_filename inference.pdiparams --save_file model.onnx'
        ' --opset_version 11',
        'On Windows, run the Paddle export in WSL with Ubuntu 24.04.',
        'PP-OCR recognition models require ONNX opset 11 or later and removal'
        ' of internal Identity nodes.',
    ]
    return Recognised(PADDLE, 'PaddlePaddle inference model in %s' % where, advice)


def _paddle_directory(path):
    """Is this directory a Paddle inference model? Returns the graph file or None."""
    if not os.path.isdir(path):
        return None
    if not os.path.isfile(os.path.join(path, PADDLE_WEIGHTS)):
        return None
    for name in PADDLE_GRAPHS:
        if os.path.isfile(os.path.join(path, name)):
            return name
    return None


def _looks_like_safetensors(head):
    """An 8-byte little-endian header length, then JSON starting with '{'."""
    if len(head) < 9:
        return False
    length = struct.unpack('<Q', head[:8])[0]
    return 0 < length < (1 << 32) and head[8:9] == b'{'


def _admit_pytorch_zip_directory(stream):
    """Bound ZIP (including ZIP64) metadata BEFORE ZipFile creates ZipInfo.

    zipfile.ZipFile() eagerly reads the complete central directory and creates
    one Python object per real entry; its later `len(infolist())` limit is too
    late. Reuse Python's bounded EOCD/ZIP64 parser, then walk actual variable-
    length central-directory headers without materializing metadata or weights.
    Both metadata *bytes* and actual entries must fit separate generous quotas.
    Multi-GiB tensor storage payloads remain untouched by this preflight.
    """
    eocd = getattr(zipfile, '_EndRecData', None)
    if eocd is None:
        raise ValueError('ZIP central-directory preflight is unavailable; '
                         'cannot safely parse PyTorch source')
    end = eocd(stream)
    if end is None:
        raise zipfile.BadZipFile('no bounded ZIP end-of-directory record')
    count = end[zipfile._ECD_ENTRIES_TOTAL]
    amount = end[zipfile._ECD_SIZE]
    if count < 0 or amount < 0:
        raise zipfile.BadZipFile('negative ZIP central-directory metadata')
    if count > MAX_PT_ARCHIVE_MEMBERS:
        raise ValueError('PyTorch ZIP advertised too many members before '
                         'central-directory allocation')
    if amount > MAX_PT_CENTRAL_DIRECTORY_BYTES:
        raise ValueError('PyTorch ZIP central-directory metadata exceeds '
                         'bounded 64MiB quota before ZipInfo allocation')
    if (end[zipfile._ECD_DISK_NUMBER] != 0 or
            end[zipfile._ECD_DISK_START] != 0):
        raise zipfile.BadZipFile('multi-disk ZIP is not an approved source')


    overhead = (zipfile.sizeEndCentDir64 + zipfile.sizeEndCentDir64Locator
                if end[zipfile._ECD_SIGNATURE] == zipfile.stringEndArchive64
                else 0)
    directory_start = end[zipfile._ECD_LOCATION] - amount - overhead
    if directory_start < 0 or directory_start > end[zipfile._ECD_LOCATION]:
        raise zipfile.BadZipFile('invalid ZIP central-directory offset')
    stream.seek(directory_start)
    remaining = amount
    walked = 0
    while remaining:
        if remaining < zipfile.sizeCentralDir:
            raise zipfile.BadZipFile('truncated ZIP central-directory header')
        header = stream.read(zipfile.sizeCentralDir)
        if (len(header) != zipfile.sizeCentralDir or
                header[:4] != zipfile.stringCentralDir):
            raise zipfile.BadZipFile('invalid ZIP central-directory signature')

        name_bytes, extra_bytes, comment_bytes = struct.unpack_from(
            '<HHH', header, 28)
        span = (zipfile.sizeCentralDir + name_bytes + extra_bytes +
                comment_bytes)
        if span > remaining:
            raise zipfile.BadZipFile('ZIP entry exceeds bounded directory extent')
        stream.seek(span - zipfile.sizeCentralDir, os.SEEK_CUR)
        remaining -= span
        walked += 1
        if walked > MAX_PT_ARCHIVE_MEMBERS:
            raise ValueError('PyTorch ZIP has too many actual central-directory '
                             'members before ZipInfo allocation')
    if walked != count:
        raise ValueError('PyTorch ZIP advertised central-directory member '
                         'count differs from actual bounded walk')
    stream.seek(0)


def _pytorch_family(path):
    """Which family the weights come from, with bounded ZIP metadata first."""
    try:
        with open(path, 'rb') as source:
            _admit_pytorch_zip_directory(source)
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
                if len(members) > MAX_PT_ARCHIVE_MEMBERS:
                    raise ValueError('PyTorch ZIP has too many members for '
                                     'bounded selected model detection')
                pickles = [info for info in members
                           if info.filename.endswith('data.pkl')]
                if not pickles:
                    return None, ''
                if len(pickles) != 1:
                    raise ValueError('PyTorch ZIP has ambiguous data.pkl members')
                selected = pickles[0]
                if selected.file_size > MAX_PT_PICKLE_METADATA_BYTES:
                    raise ValueError('PyTorch ZIP pickle metadata exceeds bounded '
                                     '128MiB family-detection quota')
                with archive.open(selected) as tensor_stream:
                    body = tensor_stream.read(MAX_PT_PICKLE_METADATA_BYTES + 1)
                if (len(body) > MAX_PT_PICKLE_METADATA_BYTES or
                        len(body) != selected.file_size):
                    raise ValueError('PyTorch ZIP pickle metadata exceeds '
                                     'selected bounded read or declared size')
    except (zipfile.BadZipFile, OSError, KeyError):
        return None, ''
    for marker, family, command in PYTORCH_FAMILIES:
        if marker in body:
            return family, command
    return '', ''


def _pytorch_answer(family, command):
    detail = 'PyTorch weights'
    if family:
        detail = 'PyTorch weights, %s' % family
    advice = [
        'These are trained weights, not a computation graph:'
        ' there is nothing here to compile yet.',
        'Export the model to ONNX first, then bring the .onnx file back here.',
    ]
    if command:
        advice.append('For this family the export command is:')
        advice.append('    ' + command)
    else:
        advice.append('Ultralytics models:  yolo export model=MODEL.pt format=onnx opset=12')
        advice.append('YOLOv5 repository:   python export.py --weights MODEL.pt --include onnx')
    advice.append('We do not run the export here: it needs PyTorch itself,'
                  ' and installing it to convert one file is the wrong trade.')
    return Recognised(PYTORCH, detail, advice, family)


def _onnx_answer(path):
    """ONNX is confirmed by parsing, not by elimination."""
    try:
        _admit_onnx_preparse(path)
    except OnnxAdmissionError as why:
        return Recognised(
            UNKNOWN, 'ONNX inspection deferred by the capacity guard',
            [str(why), 'No ONNX parsing was attempted; file validity is not established.',
             'Retry only after the reported capacity requirement is met.'],
            status='DEFERRED')
    except (OSError, ValueError) as why:
        return Recognised(
            UNKNOWN, 'ONNX inspection could not read the selected source',
            [str(why), 'No ONNX parsing was attempted.'])
    try:
        import onnx
    except ImportError:
        return Recognised(UNKNOWN, 'the onnx package is missing, so nothing can confirm this file',
                          ['Install onnx, then try again.'])
    try:
        onnx.load(path, load_external_data=False)
    except MemoryError:
        return Recognised(
            UNKNOWN, 'ONNX inspection ran out of memory',
            ['The parser could not complete; file validity is not established.',
             'Free memory before trying Recognise again.'], status='DEFERRED')
    except Exception as why:
        return Recognised(
            UNKNOWN,
            'not a format we recognise, and it does not parse as ONNX either',
            ['The ONNX parser said: %s' % why,
             'Recognised formats: ONNX (.onnx) and TFLite (.tflite).',
             'PyTorch, Keras and Paddle files have to be exported to ONNX first.'])
    return Recognised(ONNX, 'ONNX graph')


def recognise(path):
    """Recognise a model. Reads bytes, never names - with one exception.

    The exception is PaddlePaddle, whose inference model is a DIRECTORY rather
    than a file. There the layout itself is the marker: weights beside a graph.
    """
    graph = _paddle_directory(path)
    if graph:
        return _paddle_answer('this directory (%s)' % graph)
    if os.path.isdir(path):
        return Recognised(UNKNOWN, 'a directory, and not a Paddle model',
                          ['Point us at a model FILE, or at a Paddle inference'
                           ' directory holding %s.' % PADDLE_WEIGHTS])


    candidate = os.stat(path)
    if not stat.S_ISREG(candidate.st_mode):
        raise ValueError('selected model path must be a regular file, not a '
                         'FIFO, device, socket or directory')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError('selected model descriptor is not a regular file')
        with os.fdopen(fd, 'rb') as handle:
            fd = -1
            head = handle.read(64)
    finally:
        if fd >= 0:
            os.close(fd)

    if os.path.basename(path).lower().endswith('.pdmodel'):
        return _paddle_answer('the directory beside this file')
    if PADDLE_PIR_MARKER in head:
        return _paddle_answer('the directory beside this file')

    if head[TFLITE_MARKER_OFFSET:TFLITE_MARKER_OFFSET + 4] == TFLITE_MARKER:
        return Recognised(TFLITE, 'TFLite flatbuffer')

    if head[:4] == ZIP_MARKER:
        family, command = _pytorch_family(path)
        if family is not None:
            return _pytorch_answer(family, command)
        return Recognised(
            UNKNOWN, 'a zip archive, but not PyTorch weights',
            ['Zip archives are not models by themselves.',
             'If this is an exported bundle, unpack it and point us at the model inside.'])

    if head[:1] == bytes([PICKLE_FIRST]) and len(head) > 1 and head[1] in PICKLE_PROTOCOLS:
        return _pytorch_answer('', '')

    if _looks_like_safetensors(head):
        return Recognised(
            SAFETENSORS, 'safetensors weights',
            ['Safetensors holds weights only - there is no graph to compile.',
             'Export the model to ONNX from the framework that produced it.'])

    return _onnx_answer(path)


def first_bytes(path, count=16):
    """First bytes as hex - printed whenever we could not recognise a file."""
    with open(path, 'rb') as handle:
        head = handle.read(count)
    return ' '.join('%02x' % b for b in head)
