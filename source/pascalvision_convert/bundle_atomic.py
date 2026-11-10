# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import ntpath
import os
import re
import stat
import unicodedata


SCHEMA = 'crosspascal.atomic-bundle-inventory'
SCHEMA_VERSION = 1
STATUS = 'ATOMIC_NAMESPACE_COMMITTED_NOT_IMMUTABLE_OR_DURABLE'
STAGE_NAME = re.compile(r'\.crosspascal-stage-[0-9a-f]{32}\Z')
MAX_MEMBERS = 512
MAX_BYTES = 512 * 1024 * 1024

MAX_NATIVE_FAMILY_BYTES = 64 * 1024 * 1024 * 1024
REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {
    f'{base}{suffix}' for base in ('COM', 'LPT')
    for suffix in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '¹', '²', '³')
}


class UnsafeAtomicBundle(ValueError):
    pass


@dataclass(frozen=True)
class Member:
    path: str
    kind: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Inventory:
    schema: str
    version: int
    members: tuple[Member, ...]
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True)
class CommitReceipt:
    destination: str
    tree_sha256: str
    member_count: int
    status: str = STATUS


def validate_member_name(relative: str) -> str:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise UnsafeAtomicBundle('member path is not canonical relative POSIX')
    if relative.startswith('/') or ':' in relative or '\x00' in relative:
        raise UnsafeAtomicBundle('absolute, ADS, drive or NUL member path')
    if unicodedata.normalize('NFC', relative) != relative:
        raise UnsafeAtomicBundle('noncanonical Unicode member path')
    parts = relative.split('/')
    for part in parts:
        if part.casefold().endswith(('.part', '.tmp')):
            raise UnsafeAtomicBundle('temporary or partial staging member')
        if (part in ('', '.', '..') or part.endswith((' ', '.'))
                or part.split('.')[0].upper() in _RESERVED
                or any(ord(ch) < 32 for ch in part)):
            raise UnsafeAtomicBundle('unsafe Windows member component')
    return relative


def _local_absolute(path) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise UnsafeAtomicBundle('path is not a local filesystem path') from exc
    if not isinstance(raw, str) or not ntpath.isabs(raw):
        raise UnsafeAtomicBundle('bundle path must be absolute')
    drive, suffix = ntpath.splitdrive(raw)
    if not re.fullmatch(r'[A-Za-z]:', drive) or not suffix.startswith(('\\', '/')):
        raise UnsafeAtomicBundle('only a local Windows drive is supported')

    for part in [p for p in suffix.replace('/', '\\').split('\\') if p]:
        validate_member_name(part)
    canonical = os.path.abspath(raw)
    drive, suffix = ntpath.splitdrive(canonical)
    for part in [p for p in suffix.replace('/', '\\').split('\\') if p]:
        validate_member_name(part)
    return canonical


def _guard_stat(path, *, directory: bool):
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise UnsafeAtomicBundle('bundle path unavailable') from exc
    if stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) & REPARSE):
        raise UnsafeAtomicBundle('symlink, mount point or NTFS reparse is forbidden')
    if directory and not stat.S_ISDIR(info.st_mode):
        raise UnsafeAtomicBundle('expected a regular directory')
    if not directory and not stat.S_ISREG(info.st_mode):
        raise UnsafeAtomicBundle('expected a regular file')
    return info


def _guard_existing_ancestors(path):
    drive, tail = ntpath.splitdrive(path)
    current = drive + '\\'
    _guard_stat(current, directory=True)
    for part in [p for p in tail.replace('/', '\\').split('\\') if p]:
        current = os.path.join(current, part)
        _guard_stat(current, directory=True)


def _require_ntfs(parent):
    if os.name != 'nt':
        raise UnsafeAtomicBundle('atomic no-clobber namespace pilot requires native Windows')
    drive = ntpath.splitdrive(parent)[0]
    fsname = ctypes.create_unicode_buffer(32)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_info = kernel.GetVolumeInformationW
    get_info.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_wchar_p, ctypes.c_uint32]
    get_info.restype = ctypes.c_int
    if not get_info(drive + '\\', None, 0, None, None, None, fsname, len(fsname)):
        raise UnsafeAtomicBundle('unable to verify volume filesystem')
    if fsname.value.upper() != 'NTFS':
        raise UnsafeAtomicBundle('atomic namespace pilot restricted to NTFS')


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_nlink)


def _file_sha(path, first):
    if first.st_nlink != 1:
        raise UnsafeAtomicBundle('linked file can be written outside selected tree')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags)
        try:
            if _stamp(os.fstat(fd)) != _stamp(first):
                raise UnsafeAtomicBundle('file identity changed before SHA read')
            digest = hashlib.sha256()
            with os.fdopen(fd, 'rb', closefd=False) as reader:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            if (_stamp(os.fstat(fd)) != _stamp(first)
                    or _stamp(_guard_stat(path, directory=False)) != _stamp(first)):
                raise UnsafeAtomicBundle('file changed while SHA was measured')
            return digest.hexdigest()
        finally:
            os.close(fd)
    except OSError as exc:
        raise UnsafeAtomicBundle('file unavailable during SHA inventory') from exc


def scan_bundle_tree(root, *, max_members=MAX_MEMBERS, max_bytes=MAX_BYTES):
    root = _local_absolute(root)
    if not isinstance(max_members, int) or not 1 <= max_members <= MAX_MEMBERS:
        raise UnsafeAtomicBundle('unsupported member quota')
    if (type(max_bytes) is not int or
            not 1 <= max_bytes <= MAX_NATIVE_FAMILY_BYTES):
        raise UnsafeAtomicBundle('unsupported byte quota')
    _guard_existing_ancestors(root)
    seen_inodes = set()
    members = []
    total = 0
    pending = [(root, '')]
    while pending:
        directory, prefix = pending.pop()
        _guard_stat(directory, directory=True)
        try:
            with os.scandir(directory) as stream:
                listing = []
                for item in stream:
                    if len(members) + len(listing) + 1 > max_members:
                        raise UnsafeAtomicBundle('bundle exceeds member quota')
                    listing.append(item)
        except OSError as exc:
            raise UnsafeAtomicBundle('cannot enumerate bundle directory') from exc
        alias = set()
        for item in listing:
            relative = validate_member_name(prefix + item.name)
            folded = unicodedata.normalize('NFC', item.name).casefold()
            if folded in alias:
                raise UnsafeAtomicBundle('case/Unicode aliased bundle entries')
            alias.add(folded)
            if len(members) + 1 > max_members:
                raise UnsafeAtomicBundle('bundle exceeds member quota')
            info = _guard_stat(item.path, directory=item.is_dir(follow_symlinks=False))
            if stat.S_ISDIR(info.st_mode):
                members.append(Member(relative, 'directory', 0, ''))
                pending.append((item.path, relative + '/'))
            elif stat.S_ISREG(info.st_mode):
                if info.st_size < 0 or total + info.st_size > max_bytes:
                    raise UnsafeAtomicBundle('bundle exceeds byte quota')
                key = (info.st_dev, info.st_ino)
                if key in seen_inodes:
                    raise UnsafeAtomicBundle('hardlinked bundle member is forbidden')
                seen_inodes.add(key)
                members.append(Member(relative, 'file', info.st_size,
                                      _file_sha(item.path, info)))
                total += info.st_size
            else:
                raise UnsafeAtomicBundle('special file in proposed bundle')
    if not any(m.kind == 'file' for m in members):
        raise UnsafeAtomicBundle('empty bundle has no user artefacts')
    ordered = tuple(sorted(members, key=lambda m: m.path.encode('utf-8')))
    payload = {'schema': SCHEMA, 'version': SCHEMA_VERSION,
               'members': [m.__dict__ for m in ordered], 'total_bytes': total}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(',', ':'), ensure_ascii=False
                                        ).encode('utf-8')).hexdigest()
    return Inventory(SCHEMA, SCHEMA_VERSION, ordered, total, digest)


def _same_volume_private_stage(stage, destination):
    stage = _local_absolute(stage)
    destination = _local_absolute(destination)
    parent = os.path.dirname(stage)
    if os.path.normcase(parent) != os.path.normcase(os.path.dirname(destination)):
        raise UnsafeAtomicBundle('stage and destination must share one parent')
    if not STAGE_NAME.fullmatch(os.path.basename(stage)):
        raise UnsafeAtomicBundle('stage must be an owned private UUID directory')
    if STAGE_NAME.fullmatch(os.path.basename(destination)):
        raise UnsafeAtomicBundle('destination cannot be another private stage')
    _guard_existing_ancestors(parent)
    _require_ntfs(parent)
    if _guard_stat(stage, directory=True).st_dev != _guard_stat(parent, directory=True).st_dev:
        raise UnsafeAtomicBundle('stage is on a different device')
    if os.path.lexists(destination):
        raise UnsafeAtomicBundle('destination already exists; never clobber')
    return stage, destination


def commit_staged_bundle(stage, destination, approved: Inventory, *,
                         required_files=('recipe.json', 'report.txt'),
                         max_members=MAX_MEMBERS, max_bytes=MAX_BYTES):
    stage, destination = _same_volume_private_stage(stage, destination)
    if not isinstance(approved, Inventory) or approved.schema != SCHEMA or approved.version != SCHEMA_VERSION:
        raise UnsafeAtomicBundle('unrecognised preapproved bundle inventory')
    required = {validate_member_name(name) for name in required_files}
    available = {m.path for m in approved.members if m.kind == 'file'}
    if not required or not required.issubset(available):
        raise UnsafeAtomicBundle('approved bundle lacks mandatory members')
    before = scan_bundle_tree(stage, max_members=max_members, max_bytes=max_bytes)
    if before != approved:
        raise UnsafeAtomicBundle('staged bundle changed after approval')
    inode = _guard_stat(stage, directory=True)


    try:
        os.rename(stage, destination)
    except OSError as exc:
        raise UnsafeAtomicBundle('atomic directory rename refused; no READY') from exc
    try:
        dest_identity = _guard_stat(destination, directory=True)
        if ((inode.st_dev, inode.st_ino) != (dest_identity.st_dev, dest_identity.st_ino)
                or scan_bundle_tree(destination, max_members=max_members,
                                    max_bytes=max_bytes) != approved):
            raise UnsafeAtomicBundle('committed bytes or directory identity changed')
    except Exception as exc:
        raise UnsafeAtomicBundle(
            'post-commit audit failed; no READY; exposed destination preserved '
            '(no rollback of potentially user-modified contents)') from exc
    return CommitReceipt(destination, approved.tree_sha256, len(approved.members))
