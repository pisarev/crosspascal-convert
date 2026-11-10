# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-


import ctypes
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

from pascalvision_convert.llm_build_profile import console_python

HERE = os.path.dirname(os.path.abspath(__file__))


LLM_HELPERS = os.path.join(HERE, 'pascalvision_convert', 'llm_helpers')


def verify_installed_llm_helpers():
    locked = os.path.join(LLM_HELPERS, 'HELPER-SHA256.json')
    if not os.path.isfile(locked):
        raise RuntimeError('LLM helper lock is missing from installed converter: ' + locked)
    with open(locked, encoding='utf-8') as reader:
        expected = json.load(reader)
    names = ('build_llm.py', 'qwen2_model.py', 'gemma2_model.py', 'weights.py')
    if expected.get('schema') != 'crosspascal-firstparty-llm-helpers-v1' or (
            set(expected.get('sha256', {})) != set(names)):
        raise RuntimeError('LLM helper lock has an incomplete or unsupported schema')
    for name in names:
        path = os.path.join(LLM_HELPERS, name)
        if not os.path.isfile(path) or sha256_of(path) != expected['sha256'][name]:
            raise RuntimeError('LLM helper missing or SHA-256 mismatch: ' + name)
    return os.path.join(LLM_HELPERS, 'build_llm.py')


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


MAX_YOLO_EXPORTED_ONNX_BYTES = 64 * 1024 ** 3
MIN_YOLO_OUTPUT_HEADROOM = 256 * 1024 ** 2


@contextmanager
def _hold_yolo_directory_identity(directory, creation_info, *, label):
    from pascalvision_convert import source_freeze, source_lock

    def path_audit():
        source_freeze._safe_ancestors(directory)
        info = source_freeze._lstat(directory)
        if (not stat.S_ISDIR(info.st_mode) or
                (info.st_dev, info.st_ino) !=
                (creation_info.st_dev, creation_info.st_ino)):
            raise RuntimeError('%s directory FileID/ancestry was replaced' % label)

    path_audit()
    if not source_lock._is_windows():
        try:
            yield False
        finally:
            path_audit()
        return

    api = source_lock._Kernel32()
    api.assert_local_ntfs(directory)
    with ExitStack() as handles:
        opened = api.kernel.CreateFileW(
            str(directory), source_lock.GENERIC_READ,
            source_lock.FILE_SHARE_READ, None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if opened == source_lock.INVALID_HANDLE_VALUE:
            raise RuntimeError('%s Win32 directory hold failed: %d' %
                               (label, ctypes.get_last_error()))
        handles.callback(api.close, opened)

        def native_audit():
            observed = api.FileInfo()
            if not api.kernel.GetFileInformationByHandle(
                    opened, ctypes.byref(observed)):
                raise RuntimeError('%s held directory FileID unavailable' % label)
            file_id = ((observed.nFileIndexHigh << 32) |
                       observed.nFileIndexLow)
            if (file_id != creation_info.st_ino or
                    not (observed.dwFileAttributes &
                         source_lock.FILE_ATTRIBUTE_DIRECTORY) or
                    (observed.dwFileAttributes &
                     source_lock.FILE_ATTRIBUTE_REPARSE_POINT)):
                raise RuntimeError('%s held directory FileID/reparse mismatch' % label)
            path_audit()

        native_audit()
        try:
            yield True
        finally:
            native_audit()


@contextmanager
def _hold_yolo_private_child_source(stage, root_info, model, model_size, sha,
                                    *, copied_identity=None):
    from pascalvision_convert import source_freeze, source_lock

    def stat_root():
        source_freeze._safe_ancestors(stage)
        observed = source_freeze._lstat(stage)
        if (not stat.S_ISDIR(observed.st_mode) or
                (observed.st_dev, observed.st_ino) !=
                (root_info.st_dev, root_info.st_ino)):
            raise RuntimeError('private YOLO stage root identity/ancestry changed')

    def stat_model():
        source_freeze._safe_ancestors(model)
        observed = source_freeze._lstat(model)
        if (not stat.S_ISREG(observed.st_mode) or
                getattr(observed, 'st_nlink', 1) != 1 or
                observed.st_size != model_size):
            raise RuntimeError('private YOLO .pt is linked or changed')
        return observed

    stat_root()
    before = stat_model()
    if (copied_identity is not None and
            (before.st_dev, before.st_ino) != copied_identity):
        raise RuntimeError('private YOLO .pt copy FileID was replaced '
                           'before native child custody')
    if sha256_of(model) != sha:
        raise RuntimeError('private YOLO .pt SHA drifted before exporter')
    if not source_lock._is_windows():
        try:
            yield False
        finally:
            stat_root()
            after = stat_model()
            if ((after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                    or sha256_of(model) != sha):
                raise RuntimeError('private YOLO .pt changed during POSIX exporter')
        return

    api = source_lock._Kernel32()
    api.assert_local_ntfs(stage)
    with ExitStack() as handles:
        directory_handle = api.kernel.CreateFileW(
            str(stage), source_lock.GENERIC_READ, source_lock.FILE_SHARE_READ,
            None, source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT | 0x02000000, None)
        if directory_handle == source_lock.INVALID_HANDLE_VALUE:
            raise RuntimeError('cannot hold private YOLO stage directory '
                               'on Win32: %d' % ctypes.get_last_error())
        handles.callback(api.close, directory_handle)

        def check_directory_handle():
            info = api.FileInfo()
            if not api.kernel.GetFileInformationByHandle(
                    directory_handle, ctypes.byref(info)):
                raise RuntimeError('cannot inspect held YOLO stage FileID')
            if ((info.dwFileAttributes & source_lock.FILE_ATTRIBUTE_REPARSE_POINT)
                    or not (info.dwFileAttributes &
                            source_lock.FILE_ATTRIBUTE_DIRECTORY)
                    or ((info.nFileIndexHigh << 32) | info.nFileIndexLow) !=
                    root_info.st_ino):
                raise RuntimeError('held YOLO stage is reparse/not approved FileID')
            stat_root()

        check_directory_handle()
        selected_handle = api.open(model)
        handles.callback(api.close, selected_handle)
        selected_identity = api.identity(selected_handle)
        if selected_identity.file_bytes != model_size:
            raise RuntimeError('held YOLO private .pt has wrong size')
        if (copied_identity is not None and
                selected_identity.file_id != copied_identity[1]):
            raise RuntimeError('private YOLO .pt copy FileID replaced '
                               'before Win32 handle acquisition')

        def audit():
            check_directory_handle()
            if stat_model().st_ino != selected_identity.file_id:
                raise RuntimeError('held YOLO private .pt pathname/FileID replaced')
            file_info = api.FileInfo()
            if (not api.kernel.GetFileInformationByHandle(
                    selected_handle, ctypes.byref(file_info)) or
                    file_info.nNumberOfLinks != 1):
                raise RuntimeError('private YOLO staged .pt has hardlink alias')

            from pascalvision_convert import hf_held_stage
            hf_held_stage._native_check(
                api, model, selected_handle, selected_identity,
                model_size, sha, 1024 * 1024)
            check_directory_handle()

        audit()
        try:
            yield True
        finally:
            audit()


class ImportAdapter(object):
    name = ''
    kind = 'export'

    def matches(self, path, recognised):
        raise NotImplementedError

    def export(self, path, out_dir):
        raise NotImplementedError


class YoloImportAdapter(ImportAdapter):
    name = 'yolo'
    OPSET = 17


    VENV = os.environ.get('PASCALVISION_YOLO_VENV') or sys.prefix

    def _yolo_exe(self):
        return os.path.join(self.VENV, 'Scripts', 'yolo.exe')

    def _python(self):
        return os.path.join(self.VENV, 'Scripts', 'python.exe')

    def matches(self, path, recognised):
        if not os.path.isfile(path) or not path.lower().endswith(('.pt', '.pth')):
            return False


        if (getattr(recognised, 'convertible', False) or
                getattr(recognised, 'kind', '') in ('onnx', 'tflite')):
            return False
        family = (getattr(recognised, 'family', '') or '').lower()
        detail = (getattr(recognised, 'detail', '') or '').lower()
        base = os.path.basename(path).lower()
        return ('ultralytics' in family or 'yolov5' in family
                or 'ultralytics' in detail or 'yolo' in base)

    def export(self, path, out_dir):
        yolo = self._yolo_exe()
        if not os.path.isfile(yolo):
            raise RuntimeError(
                'YOLO export toolchain is missing at %s. It is a dedicated venv '
                'with ultralytics + torch (CPU is enough); create it before the '
                'first full-cycle YOLO conversion.' % self.VENV)
        from pascalvision_convert import source_freeze
        original_file = Path(os.path.abspath(path))
        selected_output = Path(os.path.abspath(out_dir))
        source_freeze._safe_ancestors(original_file)
        source_stat = os.lstat(original_file)
        if (not stat.S_ISREG(source_stat.st_mode) or
                getattr(source_stat, 'st_nlink', 1) != 1 or
                not 0 < source_stat.st_size <= 16 * 1024 ** 3):
            raise RuntimeError('selected YOLO owner model must be one regular '
                               'non-hardlinked bounded file (<=16GiB)')


        ancestor = selected_output
        while not os.path.lexists(ancestor):
            if ancestor.parent == ancestor:
                raise RuntimeError('YOLO output has no existing safe ancestor')
            ancestor = ancestor.parent
        source_freeze._safe_ancestors(ancestor)


        stage_parent = selected_output.parent
        if stage_parent == selected_output:
            raise RuntimeError('YOLO final output may not be a volume root: '
                               'private exporter stage must be outside final')
        produced = os.path.splitext(path)[0] + '.onnx'
        onnx_path = str(selected_output / os.path.basename(produced))


        if os.path.lexists(onnx_path):
            raise FileExistsError('YOLO selected output ONNX is occupied: '
                                  + onnx_path)
        os.makedirs(selected_output, exist_ok=True)
        source_freeze._safe_ancestors(selected_output)
        source_freeze._safe_ancestors(stage_parent)


        with _hold_yolo_directory_identity(
                selected_output, os.lstat(selected_output),
                label='YOLO selected output before exporter'):
            return self._start_yolo_private_export(
                original_file=original_file, stage_parent=stage_parent,
                source_stat=source_stat, selected_output=selected_output,
                onnx_path=onnx_path, yolo=yolo, source_path=path)

    def _start_yolo_private_export(self, *, original_file, stage_parent,
                                   source_stat, selected_output, onnx_path,
                                   yolo, source_path):
        if shutil.disk_usage(stage_parent).free < (
                source_stat.st_size + MIN_YOLO_OUTPUT_HEADROOM):
            raise RuntimeError('not enough disk space for retained private '
                               'YOLO input stage and headroom')
        selected_sha = sha256_of(original_file)


        stage = Path(tempfile.mkdtemp(
            prefix='.crosspascal-yolo-export-', dir=stage_parent))
        stage_info = os.lstat(stage)


        with _hold_yolo_directory_identity(
                stage, stage_info, label='YOLO private stage'):
            return self._finish_yolo_private_export(
                original_file=original_file, stage=stage,
                stage_info=stage_info, source_stat=source_stat,
                selected_sha=selected_sha, selected_output=selected_output,
                onnx_path=onnx_path, yolo=yolo, source_path=source_path)

    def _finish_yolo_private_export(self, *, original_file, stage, stage_info,
                                    source_stat, selected_sha, selected_output,
                                    onnx_path, yolo, source_path):
        from pascalvision_convert import source_freeze
        staged_model = stage / original_file.name
        with open(original_file, 'rb') as source:
            source_fd = os.fstat(source.fileno())
            if (source_fd.st_dev != source_stat.st_dev or
                    source_fd.st_ino != source_stat.st_ino or
                    source_fd.st_size != source_stat.st_size or
                    source_fd.st_mtime_ns != source_stat.st_mtime_ns):
                raise RuntimeError('YOLO original model identity changed '
                                   'before private source copy')
            digest = hashlib.sha256()
            with open(staged_model, 'xb') as held_copy:
                remaining = source_fd.st_size
                while remaining:
                    block = source.read(min(1048576, remaining))
                    if not block:
                        raise RuntimeError('YOLO original model shortened '
                                           'during private source copy')
                    held_copy.write(block)
                    digest.update(block)
                    remaining -= len(block)
                if source.read(1):
                    raise RuntimeError('YOLO original model grew during '
                                       'private source copy')
                held_copy.flush()
                os.fsync(held_copy.fileno())


                copied_info = os.fstat(held_copy.fileno())
            source_after = os.fstat(source.fileno())
            if (source_after.st_dev != source_fd.st_dev or
                    source_after.st_ino != source_fd.st_ino or
                    source_after.st_size != source_fd.st_size or
                    source_after.st_mtime_ns != source_fd.st_mtime_ns or
                    digest.hexdigest() != selected_sha):
                raise RuntimeError('YOLO original model changed while '
                                   'copying into retained private stage')
        if (sha256_of(original_file) != selected_sha or
                sha256_of(staged_model) != selected_sha):
            raise RuntimeError('YOLO private model copy is not the approved '
                               'selected source bytes before child')
        staged_produced = staged_model.with_suffix('.onnx')
        if os.path.lexists(staged_produced):
            raise RuntimeError('private YOLO generated output unexpectedly '
                               'exists before the exporter child')
        cmd = [yolo, 'export', 'model=%s' % staged_model, 'format=onnx',
               'opset=%d' % self.OPSET]
        with _hold_yolo_private_child_source(
                stage, stage_info, staged_model, source_stat.st_size,
                selected_sha, copied_identity=(copied_info.st_dev,
                                               copied_info.st_ino)) as native_held:
            done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or '')[-2000:]
            raise RuntimeError('yolo export failed (exit %d):\n%s'
                               % (done.returncode, tail))
        if not os.path.lexists(staged_produced):
            raise RuntimeError(
                'yolo export reported success but no .onnx was written at %s'
                % staged_produced)
        source_freeze._safe_ancestors(staged_produced)
        original_info = os.lstat(staged_produced)
        if (not stat.S_ISREG(original_info.st_mode) or
                getattr(original_info, 'st_nlink', 1) != 1 or
                original_info.st_size <= 0):
            raise RuntimeError('YOLO exporter returned an invalid, linked '
                               'or empty generated ONNX file')
        if original_info.st_size > MAX_YOLO_EXPORTED_ONNX_BYTES:
            raise RuntimeError('YOLO exported ONNX exceeds explicit 64GiB '
                               'post-export size admission; stage retained')


        source_freeze._safe_ancestors(stage)
        stage_after = os.lstat(stage)
        if ((stage_after.st_dev, stage_after.st_ino) !=
                (stage_info.st_dev, stage_info.st_ino)):
            raise RuntimeError('private YOLO export root identity changed')
        with os.scandir(stage) as entries:
            names = []
            for _ in range(3):
                item = next(entries, None)
                if item is None:
                    break
                names.append(item.name)
        if sorted(names) != sorted((staged_model.name,
                                    staged_produced.name)):
            raise RuntimeError('private YOLO exporter created unexpected '
                               'members; retained for owner inspection')
        source_freeze._safe_ancestors(staged_model)
        staged_info = source_freeze._lstat(staged_model)
        if (not stat.S_ISREG(staged_info.st_mode) or
                getattr(staged_info, 'st_nlink', 1) != 1 or
                sha256_of(staged_model) != selected_sha or
                sha256_of(original_file) != selected_sha):
            raise RuntimeError('selected YOLO original/private source '
                               'changed during exporter child')


        source_freeze._safe_ancestors(selected_output)
        output_info = os.lstat(selected_output)
        with _hold_yolo_directory_identity(
                selected_output, output_info, label='YOLO selected output'):
            return self._publish_private_yolo_graph(
                staged_produced=staged_produced, original_info=original_info,
                selected_output=selected_output, onnx_path=onnx_path,
                original_file=original_file, selected_sha=selected_sha,
                source_path=source_path, stage=stage,
                native_held=native_held)

    def _publish_private_yolo_graph(self, *, staged_produced, original_info,
                                    selected_output, onnx_path, original_file,
                                    selected_sha, source_path, stage, native_held):
        if shutil.disk_usage(selected_output).free < (
                original_info.st_size + MIN_YOLO_OUTPUT_HEADROOM):
            raise RuntimeError('insufficient post-export disk space/headroom '
                               'to copy selected YOLO ONNX without filling volume; '
                               'private stage retained')
        with open(staged_produced, 'rb') as original:
            before = os.fstat(original.fileno())
            if (before.st_dev != original_info.st_dev or
                    before.st_ino != original_info.st_ino or
                    before.st_size != original_info.st_size):
                raise RuntimeError('YOLO private exported ONNX changed '
                                   'before exclusive destination publication')
            with open(onnx_path, 'xb') as selected:
                remaining = before.st_size
                while remaining:
                    chunk = original.read(min(1048576, remaining))
                    if not chunk:
                        raise RuntimeError('private YOLO exported ONNX was '
                                           'shortened while copying')
                    selected.write(chunk)
                    remaining -= len(chunk)
                if original.read(1):
                    raise RuntimeError('private YOLO exported ONNX grew after '
                                       'the selected copy extent')
                selected.flush()
                os.fsync(selected.fileno())
            after = os.fstat(original.fileno())
            if (after.st_dev != before.st_dev or
                    after.st_ino != before.st_ino or
                    after.st_size != before.st_size or
                    after.st_mtime_ns != before.st_mtime_ns):
                raise RuntimeError('YOLO private ONNX changed during '
                                   'exclusive copy; retain partial stage')
        provenance = self._provenance(source_path, onnx_path)
        if ('source_sha256' in provenance and
                provenance['source_sha256'] != selected_sha):
            raise RuntimeError('YOLO original source changed before '
                               'provenance receipt')
        provenance['private_export_stage'] = {
            'status': ('WIN32_PRIVATE_STAGE_PT_HELD_DURING_EXPORT_NOT_RELEASE_GRADE'
                       if native_held else 'PRIVATE_SOURCE_COPY_NOT_NATIVE_HELD'),
            'selected_source_sha256': selected_sha,
            'retained_stage': str(stage),
            'verification_status': 'NOT_CHECKED',
            'release_grade_immutable': False,
            'retained_after_context': True,
        }
        return {'onnx': onnx_path, 'provenance': provenance}

    def _provenance(self, source, onnx_path):
        lock = os.path.join(self.VENV, 'requirements.lock.txt')
        return {
            'adapter': self.name,
            'command': 'ultralytics export format=onnx opset=%d' % self.OPSET,
            'opset': self.OPSET,
            'toolchain': self._versions(),
            'lock_file': lock if os.path.isfile(lock) else None,
            'source': os.path.abspath(source),
            'source_sha256': sha256_of(source),
            'onnx_sha256': sha256_of(onnx_path),
        }

    def _versions(self):
        code = ("import json,ultralytics,torch,onnx;"
                "print(json.dumps({'ultralytics':ultralytics.__version__,"
                "'torch':torch.__version__,'onnx':onnx.__version__}))")
        out = subprocess.run([self._python(), '-c', code],
                             capture_output=True, text=True)
        try:
            return json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            return {}


class LLMImportAdapter(ImportAdapter):
    name = 'llm'
    kind = 'full_build'


    BUILDER = os.path.join(LLM_HELPERS, 'build_llm.py')


    PYTHON = console_python(os.environ.get('PASCALVISION_TVM_PYTHON'))


    TARGETS = ('cpu', 'cuda')


    ARCH_SHAPE = re.compile(r'^sm_\d{2,3}[a-z]?$')


    CUDA_ESCAPE = 'PASCALVISION_ALLOW_CUDA_WITHOUT_CARD'


    DTYPES = ('float32', 'float16')

    def _key_for(self, path, *, generic=False):
        cfg = os.path.join(path, 'config.json')
        if not os.path.isfile(cfg):
            return None


        from pascalvision_convert.llm_family_profile import load_source_config
        key, _config = load_source_config(path, generic=generic)
        return key

    def matches(self, path, recognised):
        return os.path.isdir(path) and self._key_for(path) is not None

    def _cuda_visible(self, python_executable=None):
        code = ('import sys, tvm; '
                'sys.stdout.write("yes" if tvm.cuda(0).exist else "no")')
        try:
            done = subprocess.run([python_executable or self.PYTHON, '-c', code], capture_output=True,
                                  text=True, encoding='utf-8', errors='replace',
                                  timeout=300)
        except Exception:
            return None
        if done.returncode != 0:
            return None
        answer = (done.stdout or '').strip()
        if answer.endswith('yes'):
            return True
        if answer.endswith('no'):
            return False
        return None

    def refuse_unusable_dtype(self, dtype):
        if dtype not in self.DTYPES:
            raise RuntimeError(
                'unknown weight precision %r. build_llm.py converts to %s and to '
                'nothing else, so anything outside that pair would be dropped on '
                'the way and the bundle would not say so. Note that the spelling '
                'is the builder\'s own - float16, not fp16 or half.'
                % (dtype, ' or '.join(self.DTYPES)))

    def refuse_unusable_device(self, target, arch, python_executable=None):
        selected_python = python_executable or self.PYTHON
        if target not in self.TARGETS:
            raise RuntimeError(
                'unknown build device %r. This adapter builds for %s and hands '
                'the builder nothing else; a device it does not know would be '
                'dropped on the way, and the bundle would not say so.'
                % (target, ' or '.join(self.TARGETS)))
        if arch and not self.ARCH_SHAPE.match(arch):
            raise RuntimeError(
                'a CUDA architecture is written sm_<number>. %r is not '
                'of that shape, and TVM would only reject it deep inside the '
                'build, after the weights are loaded.' % arch)
        if target == 'cpu':
            if arch:
                raise RuntimeError(
                    'CUDA architecture %s requires the cuda target. Remove '
                    'the architecture to build for the processor.' % arch)
            return
        if os.environ.get(self.CUDA_ESCAPE) == '1':
            return
        if not os.path.isfile(selected_python):
            raise RuntimeError(
                'the cuda device was requested and the build interpreter is not '
                'at %s, so whether a card is reachable cannot even be asked. '
                'Point PASCALVISION_TVM_PYTHON at the TVM environment, or build '
                'for the processor.' % selected_python)
        from pascalvision_convert import environment
        if environment.find_cuda() is None:
            raise RuntimeError(
                'the cuda device was requested and no NVIDIA toolkit is visible: '
                'nvcc is not on PATH and CUDA_PATH names no compiler. The '
                'builder compiles its kernels through nvrtc, so the run would '
                'fail minutes in, with the weights already loaded. Install the '
                'toolkit, or build for the processor. Set %s=1 to try anyway.'
                % self.CUDA_ESCAPE)
        if arch:
            return
        visible = self._cuda_visible(selected_python)
        if visible is None:
            raise RuntimeError(
                'the cuda device was requested and %s could not answer whether a '
                'CUDA device is reachable - the check itself failed, which is not '
                'the same as "no card", so it is not read as one. Repair that '
                'environment, name the destination CUDA architecture to '
                'build without asking, or build for the processor. Set %s=1 to '
                'skip this check.' % (selected_python, self.CUDA_ESCAPE))
        if not visible:
            raise RuntimeError(
                'the cuda device was requested, %s reports no reachable CUDA '
                'device, and no architecture was named to build for an absent '
                'card. build_llm.py ends the same way ("cuda target, no card and '
                'no --arch"), but only after loading the model, so the refusal is '
                'made here instead of after the wait. Name an architecture '
                'for the destination to cross-build, or build for the '
                'processor. Set %s=1 to skip this check.'
                % (selected_python, self.CUDA_ESCAPE))

    def build(self, path, out_dir, target='cpu', arch='', dtype='float32',
              build_plan=None, *, consumer_kit_source=None,
              approved_consumer_inventory=None,
              approved_hf_source_tree_sha256=None):
        from pascalvision_convert import qwen_universal
        requested = ((build_plan or {}).get('options', {}).get('llm') or {}).get('model_key')
        key = self._key_for(path, generic=(requested == qwen_universal.KEY))
        if key is None:
            raise RuntimeError('not a recognised LLM directory: %s' % path)
        installed_builder = verify_installed_llm_helpers()
        from pascalvision_convert import llm_build_profile
        selected_python = (build_plan or {}).get('options', {}).get('compiler_python')
        builder_python = (llm_build_profile.check_python(selected_python)
                          if selected_python is not None else self.PYTHON)
        if not os.path.isfile(builder_python):
            raise RuntimeError('LLM builder interpreter missing at %s' % builder_python)
        if not os.path.samefile(installed_builder, self.BUILDER):
            raise RuntimeError('LLM builder path differs from its installed SHA-locked source')
        target = (target or 'cpu').strip().lower()
        arch = (arch or '').strip().lower()
        dtype = (dtype or 'float32').strip().lower()
        self.refuse_unusable_device(target, arch, builder_python)
        self.refuse_unusable_dtype(dtype)
        if approved_hf_source_tree_sha256 is not None:
            if (build_plan is None or
                    type(approved_hf_source_tree_sha256) is not str or
                    re.fullmatch(r'[0-9a-f]{64}',
                                 approved_hf_source_tree_sha256) is None):
                raise RuntimeError('Preview-approved selected HF source requires '
                                   'a recorded family BuildPlan and exact SHA-256')
        selected_consumer = None
        if consumer_kit_source is not None or approved_consumer_inventory is not None:
            if (build_plan is None or consumer_kit_source is None or
                    approved_consumer_inventory is None):
                raise RuntimeError('selected native consumer kit requires a BuildPlan, '
                                   'source directory and approved Inventory')
            from pascalvision_convert import bundle_publication, bundle_atomic
            if not isinstance(approved_consumer_inventory, bundle_atomic.Inventory):
                raise RuntimeError('selected native consumer kit Inventory is invalid')
            source_root = os.path.normcase(os.path.realpath(consumer_kit_source))
            output_root_check = os.path.normcase(os.path.realpath(out_dir))
            if source_root == output_root_check:
                raise RuntimeError('selected consumer source must differ from build stage')
            selected_consumer = bundle_publication.inspect_selected_native_consumer_kit(
                consumer_kit_source, model_key=key, device=target,
                expected_tree_sha256=approved_consumer_inventory.tree_sha256)
            if selected_consumer != approved_consumer_inventory:
                raise RuntimeError('selected native consumer Inventory changed')


            if (bundle_publication.inspect_selected_native_consumer_kit(
                    out_dir, model_key=key, device=target) != selected_consumer):
                raise RuntimeError('pre-staged consumer bytes differ from selection')
        if build_plan is not None:
            from pascalvision_convert.build_plan import validate_recorded
            verified_plan = validate_recorded(
                build_plan, source=path, target=target,
                req_sm=arch or None, dtype=dtype)
            if verified_plan['route'] != 'family_full_build':
                raise RuntimeError('an LLM adapter requires its native family '
                                   'BuildPlan, not a raw ONNX compiler plan')
            selected_llm = verified_plan['options'].get('llm')
            if selected_llm is not None and selected_llm['model_key'] != key:
                raise RuntimeError('selected LLM build profile differs from source model')


        if os.name == 'nt' and (
                str(out_dir).replace('/', '\\').startswith('\\') or
                str(path).replace('/', '\\').startswith('\\')):
            raise RuntimeError('native LLM device/UNC namespace for source or '
                               'output is unsupported; use canonical local paths')
        output_root = os.path.abspath(out_dir)
        owner_root = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        actual_output = os.path.normcase(os.path.realpath(output_root))
        try:
            shared_root = os.path.commonpath((owner_root, actual_output))
            overlaps_owner = shared_root in (owner_root, actual_output)
        except ValueError:
            overlaps_owner = False
        if overlaps_owner:
            raise RuntimeError('native LLM output must not overlap original '
                               'HF model source directory or its ancestors')
        if os.path.lexists(output_root):
            info = os.lstat(output_root)
            if (not os.path.isdir(output_root) or os.path.islink(output_root) or
                    getattr(info, 'st_file_attributes', 0) & 0x400):
                raise RuntimeError('native LLM output root is not a plain directory')
        loader_stem = 'qwen2' if key in ('qwen2.5-0.5b', 'qwen2.5-1.5b',
                                       qwen_universal.KEY) else key
        child_outputs = (
            '%s-%s.dll' % (loader_stem, target), 'model.json',
            'params.bin', 'params.json', 'params.index',
            'manifest.txt', 'manifest.json',
        )


        family_receipts = ('recipe.json', 'build-plan.json',
                           'import-provenance.json', 'report.txt')
        for leaf in child_outputs + family_receipts:
            if os.path.lexists(os.path.join(output_root, leaf)):
                raise RuntimeError('native LLM output collision: %s already exists'
                                   % leaf)


        from pascalvision_convert import hf_closure
        source_scan = hf_closure.scan_selected_hf_source(
            path, expected_model_key=key)
        source_tree_sha256 = source_scan['tree_sha256']
        source_identity = None
        if (build_plan or {}).get('options', {}).get('llm') is not None and key != qwen_universal.KEY:
            from pascalvision_convert import model_sources
            source_identity = model_sources.match_source(source_scan, key)
        if (approved_hf_source_tree_sha256 is not None and
                source_tree_sha256 != approved_hf_source_tree_sha256):
            raise hf_closure.HfClosureChanged(
                'selected HF weights/tokenizer differ from Preview-approved '
                'tree SHA; refusing before any native held stage or subprocess')
        cmd = [builder_python, '-B', self.BUILDER, '--model', key,
               '--source', path, '--out', out_dir, '--target', target,
               '--dtype', dtype,
               '--hf-source-tree-sha256', source_tree_sha256]
        if arch:
            cmd += ['--arch', arch]
        selected_llm = (build_plan or {}).get('options', {}).get('llm')
        if selected_llm is not None:
            cmd += ['--context', str(selected_llm['context_window_size']),
                    '--chunk', str(selected_llm['prefill_chunk_size'])]


        def run_and_validate(command):
            if selected_python is not None:
                llm_build_profile.check_python(selected_python)
            child_env = dict(os.environ)
            child_env['PYTHONPATH'] = HERE
            child_env['PYTHONDONTWRITEBYTECODE'] = '1'
            native = subprocess.run(
                command, capture_output=True, text=True,
                encoding='utf-8', errors='replace', env=child_env)
            if native.returncode != 0:
                tail = (native.stderr or native.stdout or '')[-2000:]
                raise RuntimeError('LLM build failed (exit %d):\n%s' %
                                   (native.returncode, tail))
            hf_closure.audit_selected_hf_source(
                path, expected_tree_sha256=source_tree_sha256,
                expected_model_key=key)
            if build_plan is not None:
                from pascalvision_convert.llm_bundle_contract import validate_llm_output
                validate_llm_output(out_dir, build_plan, key,
                                    require_consumer=selected_consumer is not None)
                if selected_consumer is not None:
                    from pascalvision_convert import bundle_publication
                    if bundle_publication.inspect_selected_native_consumer_kit(
                            consumer_kit_source, model_key=key, device=target) != selected_consumer:
                        raise RuntimeError('selected consumer source changed during build')
                    for member in selected_consumer.members:
                        if (member.kind != 'file' or
                                sha256_of(os.path.join(out_dir, member.path)) != member.sha256):
                            raise RuntimeError('pre-staged consumer bytes changed during build')
            return native

        held_receipt = None
        tokenizer_consumer_binding = None
        if build_plan is not None:
            from pascalvision_convert import hf_held_stage
            stage_parent = os.path.dirname(os.path.abspath(out_dir))


            with hf_held_stage.hold_private_selected_hf_stage(
                    path, stage_parent=stage_parent,
                    expected_tree_sha256=source_tree_sha256,
                    expected_model_key=key) as ticket:
                if (ticket.source_tree_sha256 != source_tree_sha256 or
                        ticket.model_key != key or
                        ticket.file_count != source_scan['file_count'] or
                        ticket.total_bytes != source_scan['total_bytes'] or
                        ticket.release_grade_immutable is not False or
                        ticket.verification_status != 'NOT_CHECKED'):
                    raise RuntimeError('private HF Win32 source-stage ticket '
                                       'does not match selected owner bytes')
                if selected_consumer is not None:
                    from pascalvision_convert import hf_tokenizer_consumer_binding
                    tokenizer_consumer_binding = (
                        hf_tokenizer_consumer_binding.bind_held_native_pascal_tokenizer(
                            ticket.compiler_input, out_dir, source_scan['entries'],
                            selected_consumer, model_key=key))


                plan_payload = (json.dumps(
                    build_plan, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                if not 0 < len(plan_payload) <= 1024 * 1024:
                    raise RuntimeError('native BuildPlan transfer exceeds 1MiB')
                tmp = tempfile.mkdtemp(prefix='pascalvision-llm-plan-')
                plan_path = os.path.join(tmp, 'approved-build-plan.json')
                with open(plan_path, 'xb') as handle:
                    handle.write(plan_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                done = run_and_validate(
                    cmd + ['--plan-json', plan_path,
                           '--held-hf-source', ticket.compiler_input])
                held_receipt = {
                    'status': ticket.status,
                    'tree_sha256': ticket.source_tree_sha256,
                    'file_count': ticket.file_count,
                    'total_bytes': ticket.total_bytes,
                    'verification_status': 'NOT_CHECKED',
                    'release_grade_immutable': False,
                    'revision_status': 'NOT_ATTESTED',
                    'stage_storage': 'RETAINED_PRIVATE_NO_AUTOMATIC_CLEANUP',
                    'plan_transfer_storage': 'RETAINED_PRIVATE_NO_AUTOMATIC_CLEANUP',
                }
        else:
            done = run_and_validate(cmd)
        return {'bundle_dir': out_dir, 'key': key,
                'target': target, 'arch': arch, 'dtype': dtype,
                'log_tail': (done.stdout or '')[-1000:],
                'provenance': {'adapter': self.name, 'model_key': key,
                               'builder': 'build_llm.py',
                               'compiler_python': selected_python,
                               'llm_build_profile': selected_llm,
                               'official_source': source_identity,


                               'target': target,
                               'arch': arch or None,
                               'dtype': dtype,
                               'command': ' '.join(['build_llm.py', '--model', key,
                                                    '--target', target,
                                                    '--dtype', dtype]
                                                   + (['--arch', arch] if arch else [])),
                               'source': os.path.abspath(path),
                               'build_plan_sha256': (build_plan or {}).get('plan_sha256'),
                               'hf_source_closure': {
                                   field: source_scan[field] for field in (
                                       'schema', 'schema_version', 'model_key',
                                       'layout', 'entries', 'file_count',
                                       'tensor_count', 'total_bytes',
                                       'tree_sha256', 'status',
                                       'verification_status',
                                       'release_grade_immutable')},
                               'compiler_held_source': held_receipt,
                               'tokenizer_consumer_binding': tokenizer_consumer_binding,
                               'selected_consumer_kit': (
                                   {'tree_sha256': selected_consumer.tree_sha256,
                                    'status': 'SCANNED_NOT_HELD',
                                    'verification_status': 'NOT_CHECKED'}
                                   if selected_consumer is not None else None)}}


class PaddleOcrImportAdapter(ImportAdapter):
    name = 'paddleocr'
    kind = 'export'
    OPSET = 11


    VENV = os.environ.get('PASCALVISION_PADDLE_VENV', '~/venvs/export-paddle')


    _WSL_VENV_EXPANSION = (
        'case "$venv" in "~") venv="$HOME" ;; '
        '"~/"*) venv="$HOME/${venv:2}" ;; esac; ')

    def _wsl_path(self, win_path):
        full = os.path.abspath(win_path)
        drive, rest = os.path.splitdrive(full)
        return '/mnt/%s%s' % (drive.rstrip(':').lower(), rest.replace('\\', '/'))

    def _graph_file(self, path):
        for name in ('inference.json', 'inference.pdmodel'):
            if os.path.isfile(os.path.join(path, name)):
                return name
        return None

    def matches(self, path, recognised):
        if not os.path.isdir(path):
            return False
        if (getattr(recognised, 'kind', '') or '').lower() == 'paddle':
            return True
        return (self._graph_file(path) is not None
                and os.path.isfile(os.path.join(path, 'inference.pdiparams')))

    def export(self, path, out_dir, *, role=None, dictionary=None,
               approved_tree_sha256=None, output_stem=None):
        graph = self._graph_file(path)
        if graph is None:
            raise RuntimeError(
                'no Paddle graph (inference.json or inference.pdmodel) in %s' % path)
        selected = None
        selected_scanner = None
        if any(value is not None for value in (
                role, dictionary, approved_tree_sha256)):
            if role not in ('det', 'rec'):
                raise ValueError('Paddle selected source requires explicit '
                                 'role det/rec before exporter access')
            if not approved_tree_sha256:
                raise ValueError('Paddle selected source requires an approved '
                                 'exact selected-tree SHA-256 digest')
            from pascalvision_convert import paddle_closure as selected_scanner
            selected = selected_scanner.audit_selected_paddle_source(
                path, role=role, dictionary=dictionary,
                expected_tree_sha256=approved_tree_sha256)
            if selected['graph_file'] != graph:
                raise ValueError('Paddle selected graph differs from the '
                                 'exporter-selected inference graph')


        source_root = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        output_root = os.path.normcase(os.path.realpath(os.path.abspath(out_dir)))
        try:
            output_in_source = (
                os.path.commonpath((source_root, output_root)) == source_root)
        except ValueError:
            output_in_source = False
        if output_in_source:
            raise ValueError('Paddle export output cannot be inside the owner '
                             'source directory or overwrite its model bytes')
        if (os.path.islink(out_dir) or
                getattr(os.path, 'isjunction', lambda _path: False)(out_dir)):
            raise ValueError('Paddle export output cannot be a linked/junction '
                             'directory redirecting native model artifacts')
        stem = os.path.basename(path.rstrip('\\/')) or 'paddle_model'
        if output_stem is not None:
            from pascalvision_convert import source_freeze
            if type(output_stem) is not str:
                raise ValueError('Paddle held output stem must be a filename')
            stem = source_freeze._safe_name(output_stem)
        raw_path = os.path.join(out_dir, stem + '.raw.onnx')
        onnx_path = os.path.join(out_dir, stem + '.onnx')
        for member in (raw_path, onnx_path):
            if os.path.lexists(member):
                raise FileExistsError(
                    'Paddle export output collision: refusing to clobber '
                    'existing owner file ' + member)
        os.makedirs(out_dir, exist_ok=True)
        wsl_dir = self._wsl_path(path)
        wsl_out = self._wsl_path(out_dir)
        raw = '%s/%s.raw.onnx' % (wsl_out, stem)
        lean = '%s/%s.onnx' % (wsl_out, stem)


        script = ('set -e; venv="$5"; ' + self._WSL_VENV_EXPANSION +
                  'cd -- "$1"; '
                  '"$venv/bin/paddle2onnx" --model_dir . --model_filename "$2" '
                  '--params_filename inference.pdiparams --save_file "$3" '
                  '--opset_version "$6"; '
                  '"$venv/bin/onnxslim" "$3" "$4"')


        done = subprocess.run(['wsl', '--exec', 'bash', '-lc', script, '--',
                               wsl_dir, graph, raw, lean, self.VENV,
                               str(self.OPSET)],
                              capture_output=True, text=True,
                              encoding='utf-8', errors='replace')
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or '')[-2000:]
            raise RuntimeError('paddle2onnx/onnxslim failed (exit %d):\n%s'
                               % (done.returncode, tail))
        if not os.path.isfile(onnx_path):
            raise RuntimeError(
                'paddle export reported success but no ONNX at %s' % onnx_path)
        provenance = self._provenance(path, onnx_path, graph)
        if selected is not None:
            observed = selected_scanner.audit_selected_paddle_source(
                path, role=role, dictionary=dictionary,
                expected_tree_sha256=approved_tree_sha256)
            if (observed['entries'] != selected['entries'] or
                    observed['dictionary'] != selected['dictionary'] or
                    observed['graph_file'] != selected['graph_file']):
                raise selected_scanner.PaddleClosureChanged(
                    'Paddle selected resources differ after exporter')
            provenance['selected_source_closure'] = {
                key: observed[key] for key in (
                    'schema', 'schema_version', 'source_root', 'role',
                    'graph_file', 'entries', 'file_count', 'total_bytes',
                    'dictionary', 'tree_sha256', 'status',
                    'verification_status', 'release_grade_immutable')}
        return {'onnx': onnx_path,
                'provenance': provenance}

    def export_held(self, path, out_dir, *, stage_parent, role,
                    dictionary=None, approved_tree_sha256):
        from pascalvision_convert import paddle_closure, paddle_held_stage

        try:
            original = paddle_closure.audit_selected_paddle_source(
                path, role=role, dictionary=dictionary,
                expected_tree_sha256=approved_tree_sha256)
        except paddle_closure.PaddleClosureError as error:
            raise paddle_held_stage.PaddleHeldStageChanged(
                'original selected Paddle source not approved: %s' % error) from error
        stem = os.path.basename(path.rstrip('\\/'))
        with paddle_held_stage.hold_private_selected_paddle_stage(
                path, stage_parent=stage_parent, role=role,
                dictionary=dictionary,
                expected_tree_sha256=approved_tree_sha256) as ticket:
            stage_selected = paddle_closure.scan_selected_paddle_source(
                ticket.compiler_input, role=role,
                dictionary=ticket.staged_dictionary)
            if (stage_selected['entries'] != original['entries'] or
                    stage_selected['graph_file'] != original['graph_file']):
                raise paddle_held_stage.PaddleHeldStageChanged(
                    'private stage does not reproduce approved Paddle bytes')
            exported = self.export(
                ticket.compiler_input, out_dir, role=role,
                dictionary=ticket.staged_dictionary,
                approved_tree_sha256=stage_selected['tree_sha256'],
                output_stem=stem)
            provenance = exported.get('provenance') or {}
            if provenance.get('selected_source_closure') != stage_selected:
                raise paddle_held_stage.PaddleHeldStageChanged(
                    'WSL exporter did not attest the actual native-held stage')
            provenance['staged_selected_source_closure'] = stage_selected
            provenance['selected_source_closure'] = original
            provenance['original_source'] = os.path.abspath(path)
            provenance['compiler_held_source'] = {
                'compiler_input': ticket.compiler_input,
                'source_tree_sha256': ticket.source_tree_sha256,
                'role': ticket.role, 'graph_file': ticket.graph_file,
                'file_count': ticket.file_count,
                'total_bytes': ticket.total_bytes,
                'status': ticket.status,
                'verification_status': ticket.verification_status,
                'release_grade_immutable': ticket.release_grade_immutable,
                'retained_after_context': ticket.retained_after_context,
            }
            exported['provenance'] = provenance


        return exported

    def _provenance(self, source, onnx_path, graph):
        lock = self.VENV + '/requirements.lock.txt'
        return {
            'adapter': self.name,
            'command': ('paddle2onnx --opset_version %d, then onnxslim, in WSL'
                        % self.OPSET),
            'opset': self.OPSET,
            'graph_file': graph,
            'toolchain': self._versions(),
            'lock_file': lock,
            'source': os.path.abspath(source),
            'source_sha256': sha256_of(os.path.join(source, 'inference.pdiparams')),
            'onnx_sha256': sha256_of(onnx_path),
            'source_revision_status': 'NOT_ATTESTED',
        }

    def _versions(self):
        script = ('set -eo pipefail; venv="$1"; ' +
                  self._WSL_VENV_EXPANSION +
                  '"$venv/bin/paddle2onnx" --version 2>&1 | tail -1')
        out = subprocess.run(
            ['wsl', '--exec', 'bash', '-lc', script, '--', self.VENV],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        return {'paddle2onnx': (out.stdout or '').strip()[-120:]}


ADAPTERS = [
    YoloImportAdapter(),
    LLMImportAdapter(),
    PaddleOcrImportAdapter(),
]


def find_adapter(path, recognised):
    for adapter in ADAPTERS:
        try:
            if adapter.matches(path, recognised):
                return adapter
        except Exception:
            continue
    return None
