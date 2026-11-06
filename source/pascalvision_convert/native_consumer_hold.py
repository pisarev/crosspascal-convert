# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Win32-held, caller-prepared PRIVATE native LLM consumer kit.

Point-in-time Inventory is required; holding all private files denies new
write/delete opens during the caller's work. This is neither numerical
verification nor custody of the original owner-selected consumer directory.
"""

import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import bundle_atomic, bundle_publication, source_lock
from .hf_held_stage import _hash_win32_handle


class NativeConsumerHoldError(RuntimeError):
    pass


def _snapshot(root, inventory, model_key, device, *, allow_native_outputs=False):
    if allow_native_outputs:
        observed = bundle_atomic.scan_bundle_tree(
            str(root), max_bytes=bundle_atomic.MAX_NATIVE_FAMILY_BYTES)
        from .llm_bundle_contract import _FAMILIES
        library = '%s-%s.dll' % (_FAMILIES[model_key][0], device)

        outputs = {library, '%s-%s.lib' % (_FAMILIES[model_key][0], device),
                   'model.json', 'params.bin', 'params.json',
                   'params.index', 'manifest.json', 'manifest.txt',
                   'recipe.json', 'build-plan.json',
                   'import-provenance.json', 'report.txt'}
        expected = {m.path: m for m in inventory.members}
        actual = {m.path: m for m in observed.members}
        if (any(actual.get(name) != member for name, member in expected.items())
                or any(name not in expected and name not in outputs
                       for name in actual)
                or len(actual) != len(observed.members)):
            raise NativeConsumerHoldError('native outputs altered selected consumer files')
    else:
        observed = bundle_publication.inspect_selected_native_consumer_kit(
            str(root), model_key=model_key, device=device,
            expected_tree_sha256=inventory.tree_sha256)
        if observed != inventory:
            raise NativeConsumerHoldError('selected private consumer Inventory changed')
    if any(m.kind != 'file' or '/' in m.path for m in inventory.members):
        raise NativeConsumerHoldError('native consumer hold requires flat regular files')
    return observed


@contextmanager
def hold_private_native_consumer_stage(private_stage, approved_inventory, *,
                                       model_key, device):
    """Yield the unchanged private stage while every approved leaf is Win32-held.

    Caller must prepare this PRIVATE stage using stage_selected_native_consumer_kit.
    The stage is NEVER removed; a failure forbids caller publication/READY.
    """
    if not isinstance(approved_inventory, bundle_atomic.Inventory):
        raise NativeConsumerHoldError('approved Inventory required')
    if not source_lock._is_windows():
        raise NativeConsumerHoldError('native Win32 required, no unlocked fallback')
    try:
        root = Path(bundle_atomic._local_absolute(private_stage))
        api = source_lock._Kernel32()
        api.assert_local_ntfs(root)
        if not bundle_atomic.STAGE_NAME.fullmatch(root.name):
            raise NativeConsumerHoldError('private UUID stage name required')
        if (root.is_symlink() or getattr(root, 'is_junction', lambda: False)()
                or os.path.normcase(os.path.realpath(root)) != os.path.normcase(str(root))):
            raise NativeConsumerHoldError('private stage redirects through a link')
        base = os.lstat(root)
        if not stat.S_ISDIR(base.st_mode):
            raise NativeConsumerHoldError('private stage is not a regular directory')
        _snapshot(root, approved_inventory, model_key, device)
        paths = [root / member.path for member in approved_inventory.members]
        folded = [member.path.casefold() for member in approved_inventory.members]
        if len(set(folded)) != len(folded):
            raise NativeConsumerHoldError('case-fold alias in consumer inventory')
        with ExitStack() as handles:
            opened = []
            for path, member in zip(paths, approved_inventory.members):
                info = os.lstat(path)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or getattr(info, 'st_file_attributes', 0) & 0x400):
                    raise NativeConsumerHoldError('consumer member is linked/reparse')
                handle = api.open(path)
                handles.callback(api.close, handle)
                identity = api.identity(handle)
                if identity.file_bytes != member.size:
                    raise NativeConsumerHoldError('Win32 held consumer size changed')
                opened.append((path, handle, identity, member))

            def audit(*, allow_native_outputs=False):
                directory = os.lstat(root)
                if (directory.st_dev, directory.st_ino) != (base.st_dev, base.st_ino):
                    raise NativeConsumerHoldError('private stage directory identity changed')
                _snapshot(root, approved_inventory, model_key, device,
                          allow_native_outputs=allow_native_outputs)
                for path, handle, identity, member in opened:
                    if api.identity(handle) != identity:
                        raise NativeConsumerHoldError('consumer Win32 FileID/ChangeTime changed')
                    source_lock._selected_path_matches_handle(path, identity, member.size)
                    digest, size = _hash_win32_handle(
                        api, handle, block_bytes=1024 * 1024, expected_bytes=member.size)
                    if digest != member.sha256 or size != member.size:
                        raise NativeConsumerHoldError('held consumer SHA changed')
                    if api.identity(handle) != identity:
                        raise NativeConsumerHoldError('consumer Win32 metadata changed during read')

            audit()
            try:
                yield str(root)
            finally:
                audit(allow_native_outputs=True)
    except (OSError, ValueError, source_lock.SourceLockError) as error:
        raise NativeConsumerHoldError(str(error)) from error
