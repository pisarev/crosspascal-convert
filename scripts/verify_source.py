from pathlib import Path
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata


INDEX_NAME = "SOURCE-INDEX.json"
INDEX_FIELDS = {INDEX_NAME: "members", "ARCHIVE-MANIFEST.json": "files"}
ARCHIVE_SCHEMAS = {
    "crosslept-stage20-001-archive-manifest-v1",
    "stage10-attempt002-selected-leptonica-preview-subset-v2",
}
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RESERVED_NAMES = {"con", "prn", "aux", "nul"}
RESERVED_NAMES.update(f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10))


def _canonical_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a nonempty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("path must use NFC Unicode")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("path must be canonical and relative")
    if any(ord(char) < 32 or ord(char) == 127 or char in '\\<>:"|?*' for char in value):
        raise ValueError("path contains a nonportable character")
    if any(part.endswith((" ", ".")) for part in parts):
        raise ValueError("path component ends with a space or dot")
    if any(part.split(".")[0].casefold() in RESERVED_NAMES for part in parts):
        raise ValueError("path contains a reserved device name")
    return parts


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonstandard JSON numeric constant")


def _validate_index(index, index_name):
    if not isinstance(index, dict):
        raise ValueError("index must be an object")
    member_field = INDEX_FIELDS[index_name]
    if not isinstance(index.get(member_field), list):
        raise ValueError(f"{member_field} must be an array")
    if any(field in index for field in set(INDEX_FIELDS.values()) - {member_field}):
        raise ValueError("index contains conflicting inventory fields")
    if index_name == "ARCHIVE-MANIFEST.json":
        if not isinstance(index.get("schema"), str) or index["schema"] not in ARCHIVE_SCHEMAS:
            raise ValueError("unsupported archive inventory schema")
        if index.get("manifest_self_excluded") is not True:
            raise ValueError("archive manifest_self_excluded must be true")
    if not isinstance(index.get("local_metadata"), list):
        raise ValueError("local_metadata must be an explicit array")
    members = {}
    namespace = {}
    for number, record in enumerate(index[member_field], 1):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"member {number} must contain path, bytes and sha256")
        parts = _canonical_path(record["path"])
        relative = record["path"]
        if any(part.casefold() == ".git" for part in parts):
            raise ValueError(f"member {number} contains Git metadata")
        if parts[0].casefold() == index_name.casefold():
            raise ValueError("selected root inventory is the unindexed self-index")
        if type(record["bytes"]) is not int or record["bytes"] < 0:
            raise ValueError(f"member {number} bytes must be a nonnegative integer")
        if not isinstance(record["sha256"], str) or not SHA256_PATTERN.fullmatch(record["sha256"]):
            raise ValueError(f"member {number} sha256 must be 64 lowercase hex digits")
        if relative in members:
            raise ValueError(f"member {number} duplicates a path")
        for end in range(1, len(parts) + 1):
            prefix = "/".join(parts[:end])
            kind = "file" if end == len(parts) else "directory"
            previous = namespace.get(prefix.casefold())
            if previous is not None and previous != (prefix, kind):
                raise ValueError(f"member {number} has a path alias or file/directory collision")
            namespace[prefix.casefold()] = (prefix, kind)
        members[relative] = record
    metadata = index["local_metadata"]
    if len(metadata) > 1:
        raise ValueError("local_metadata allows at most the root .git entry")
    for record in metadata:
        if not isinstance(record, dict) or set(record) != {"path", "kind", "reason"}:
            raise ValueError("local_metadata entry must contain path, kind and reason")
        if record["path"] != ".git" or record["kind"] != "file_or_directory":
            raise ValueError("only root .git with kind file_or_directory is permitted")
        if not isinstance(record["reason"], str) or not record["reason"].strip():
            raise ValueError("local_metadata reason must be nonempty")
    directories = {path for path, kind in namespace.values() if kind == "directory"}
    return members, directories, bool(metadata)


def _is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & REPARSE_POINT)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read_regular(path, before, collect=False):
    if _is_link(before) or not stat.S_ISREG(before.st_mode):
        raise OSError("not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _is_link(opened) or not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise OSError("file changed before reading")
        digest = hashlib.sha256()
        size = 0
        chunks = [] if collect else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        if _identity(os.fstat(descriptor)) != _identity(opened):
            raise OSError("file changed while reading")
        return size, digest.hexdigest(), b"".join(chunks) if collect else None
    finally:
        os.close(descriptor)


def verify_repository(root, index_name=INDEX_NAME):
    root = Path(root).absolute()
    result = {
        "ok": False,
        "expected_members": 0,
        "observed_payload_files": 0,
        "checked_members": 0,
        "self_index_files": 0,
        "local_metadata_roots": [],
        "local_metadata_files": 0,
        "local_metadata_directories": 0,
        "findings": [],
    }

    def finding(code, path, message):
        result["findings"].append({"code": code, "path": path, "message": message})

    if index_name not in INDEX_FIELDS:
        finding("unsupported_index", ".", "Index must be SOURCE-INDEX.json or ARCHIVE-MANIFEST.json.")
        return result
    result["index_name"] = index_name
    result["inventory_field"] = INDEX_FIELDS[index_name]
    try:
        root_info = root.lstat()
        if _is_link(root_info) or not stat.S_ISDIR(root_info.st_mode):
            finding("invalid_root", ".", "Repository root must be an ordinary directory.")
            return result
        index_path = root / index_name
        index_info = index_path.lstat()
        if _is_link(index_info) or not stat.S_ISREG(index_info.st_mode):
            finding("invalid_index", index_name, "Self-index must be an ordinary file.")
            return result
        _, index_sha256, raw = _read_regular(index_path, index_info, collect=True)
    except OSError:
        finding("index_io", index_name, "Cannot read the repository root or self-index.")
        return result
    try:
        index = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        expected, expected_directories, git_declared = _validate_index(index, index_name)
    except (UnicodeError, ValueError, RecursionError) as error:
        message = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else "Invalid UTF-8 JSON index."
        finding("invalid_manifest", index_name, message)
        return result
    result["expected_members"] = len(expected)
    result["index_sha256"] = index_sha256
    actual = {}
    observed_names = {}
    pending = [(root, "", False)]
    while pending:
        directory, prefix, inside_metadata = pending.pop()
        try:
            directory_info = directory.lstat()
            if _is_link(directory_info) or not stat.S_ISDIR(directory_info.st_mode):
                finding("linked_or_changed_directory", prefix or ".", "Directory is linked, reparse-backed or changed.")
                continue
            with os.scandir(directory) as entries:
                names = sorted(entry.name for entry in entries)
        except OSError:
            finding("directory_io", prefix or ".", "Cannot enumerate directory.")
            continue
        for name in names:
            relative = prefix + "/" + name if prefix else name
            path = directory / name
            try:
                parts = _canonical_path(relative)
            except ValueError as error:
                finding("invalid_disk_path", relative, str(error))
                continue
            alias = observed_names.get(relative.casefold())
            if alias is not None and alias != relative:
                finding("disk_path_alias", relative, "Filesystem paths differ only by case.")
            observed_names[relative.casefold()] = relative
            try:
                info = path.lstat()
            except OSError:
                finding("entry_io", relative, "Cannot inspect filesystem entry.")
                continue
            if _is_link(info):
                finding("linked_entry", relative, "Symlinks, junctions and other reparse points are not allowed.")
                continue
            if any(part.casefold() == ".git" for part in parts[1:]):
                finding("nested_git_metadata", relative, "Only the explicitly declared root .git is permitted.")
                continue
            is_metadata = inside_metadata
            if len(parts) == 1 and parts[0].casefold() == ".git":
                if relative != ".git" or not git_declared:
                    finding("undeclared_git_metadata", relative, "Root Git metadata is not explicitly declared with this exact path.")
                    continue
                is_metadata = True
                result["local_metadata_roots"].append(relative)
            if stat.S_ISDIR(info.st_mode):
                if is_metadata:
                    result["local_metadata_directories"] += 1
                elif relative not in expected_directories:
                    finding("unexpected_directory", relative, "Directory is not implied by the indexed payload.")
                pending.append((path, relative, is_metadata))
            elif stat.S_ISREG(info.st_mode):
                if is_metadata:
                    result["local_metadata_files"] += 1
                else:
                    actual[relative] = (path, info)
            else:
                finding("special_entry", relative, "Only ordinary files and directories are allowed.")
    result["self_index_files"] = int(index_name in actual)
    result["observed_payload_files"] = len(actual) - result["self_index_files"]
    for relative in sorted(set(actual) - set(expected) - {index_name}):
        finding("unexpected_file", relative, "File is not indexed.")
    for relative in sorted(set(expected) - set(actual)):
        finding("missing_file", relative, "Indexed file is missing or is not an ordinary file.")
    if index_name not in actual or _identity(actual[index_name][1]) != _identity(index_info):
        finding("changed_index", index_name, "Self-index is missing or changed during verification.")
    for relative in sorted(set(expected) & set(actual)):
        path, info = actual[relative]
        try:
            size, digest, _ = _read_regular(path, info)
        except OSError:
            finding("file_io", relative, "Cannot safely read a stable ordinary file.")
            continue
        result["checked_members"] += 1
        if size != expected[relative]["bytes"]:
            finding("size_mismatch", relative, "File size differs from the index.")
        if digest != expected[relative]["sha256"]:
            finding("sha256_mismatch", relative, "File SHA-256 differs from the index.")
    result["ok"] = not result["findings"]
    return result


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    index_name = INDEX_NAME
    if arguments in (["-h"], ["--help"]):
        print("Usage: python scripts/verify_source.py [--index SOURCE-INDEX.json|ARCHIVE-MANIFEST.json]")
        print("Verify the selected inventory against this repository, including hidden files.")
        return 0
    if len(arguments) == 2 and arguments[0] == "--index" and arguments[1] in INDEX_FIELDS:
        index_name = arguments[1]
    elif arguments:
        print("FAIL: unsupported arguments or index; use --help.")
        return 1
    root = Path(__file__).resolve().parents[1]
    result = verify_repository(root, index_name)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
