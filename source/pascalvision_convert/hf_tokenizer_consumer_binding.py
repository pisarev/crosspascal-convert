# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Selected HF tokenizer.json -> exact Pascal native tokenizer byte closure.

This compares the output of the EXISTING Qwen/Gemma tokenizer-export formats
with the selected consumer kit while the caller holds both PRIVATE source
and native kit through Win32 read/share-only descriptors. A matching byte
closure is NOT model numeric/Pascal UAT, a signed checkpoint revision, or a
redistribution licence. No owner source or consumer files are modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from . import bundle_atomic, hf_closure, llm_bundle_contract, source_freeze


MAX_GENERATED_TOKENIZER_BYTES = 256 * 1024 * 1024
MAX_TOKEN_IDS = 512000
MAX_TOKENIZER_MERGES = 1000000


class HfTokenizerConsumerMismatch(ValueError):
    """Native tokenizer kit does not match the approved HF tokenizer bytes."""


def _refuse(reason):
    raise HfTokenizerConsumerMismatch("selected HF/Pascal tokenizer: " + reason)


def _words(vocab):
    if not isinstance(vocab, dict) or not 0 < len(vocab) <= MAX_TOKEN_IDS:
        _refuse("BPE vocab is absent or out of quota")
    identifiers = set()
    for token, index in vocab.items():
        if (not isinstance(token, str) or type(index) is not int or
                not 0 <= index < MAX_TOKEN_IDS or index in identifiers):
            _refuse("BPE token IDs are invalid/duplicated")
        identifiers.add(index)
    highest = max(identifiers)
    if len(identifiers) != highest + 1:
        _refuse("BPE token IDs contain holes")
    ordered = [None] * (highest + 1)
    for token, index in vocab.items():
        ordered[index] = token
    return ordered


def _merges(vocab, merges):
    if not isinstance(merges, list) or len(merges) > MAX_TOKENIZER_MERGES:
        _refuse("BPE merge list outside quota")
    triples = bytearray()
    count = 0


    for item in merges:
        if isinstance(item, str):
            left, separator, right = item.partition(" ")
            if not separator or not left or not right:
                _refuse("BPE merge string must contain two nonempty tokens")
        elif isinstance(item, list) and len(item) == 2 and all(
                isinstance(s, str) for s in item):
            left, right = item
            if not left or not right:
                _refuse("BPE merge pair must contain two nonempty tokens")
        else:
            _refuse("BPE merge has unsupported format")
        joined = left + right
        if left in vocab and right in vocab and joined in vocab:
            triples.extend(struct.pack("<iii", vocab[left], vocab[right], vocab[joined]))
            count += 1
    return bytes(triples), count


def _flatten_tokens(raw):
    blob = bytearray()
    entries = bytearray()
    for piece in raw:
        if len(blob) + len(piece) > MAX_GENERATED_TOKENIZER_BYTES:
            _refuse("Pascal token byte payload outside quota")
        entries.extend(struct.pack("<II", len(blob), len(piece)))
        blob.extend(piece)
    return bytes(blob), bytes(entries)


def _gpt2_backmap():

    visible = (list(range(ord("!"), ord("~") + 1))
               + list(range(ord("\xa1"), ord("\xac") + 1))
               + list(range(ord("\xae"), ord("\xff") + 1)))
    used = list(visible)
    mapped = list(visible)
    extra = 0
    for value in range(256):
        if value not in used:
            used.append(value)
            mapped.append(256 + extra)
            extra += 1
    return {chr(mapped[i]): original for i, original in enumerate(used)}


def _qwen_bytes(spec, vocab, ordered, merges):
    back = _gpt2_backmap()
    raw = []
    single = set()
    for word in ordered:
        try:
            piece = bytes(back[ch] for ch in word)
        except KeyError as exc:
            _refuse("Qwen BPE word is outside GPT-2 byte alphabet")
        raw.append(piece)
        if len(piece) == 1:
            single.add(piece[0])
    if len(single) != 256:


        _refuse("Qwen BPE lacks complete 256-byte single-token alphabet")
    tokens, index = _flatten_tokens(raw)
    triples, count = _merges(vocab, merges)
    specials = spec.get("added_tokens", [])
    if not isinstance(specials, list):
        _refuse("Qwen added_tokens must be a list")
    lines = [str(len(specials))]
    for item in specials:
        if not isinstance(item, dict):
            _refuse("Qwen added token must be a record")
        lines.append("%d %s" % (item["id"], item["content"]))
    pre = spec.get("pre_tokenizer") or {}
    if not isinstance(pre, dict):
        _refuse("Qwen pre_tokenizer must be a record")
    pattern = ""
    for stage in pre.get("pretokenizers", []):
        if stage.get("type") == "Split":
            pattern = stage["pattern"]["Regex"]
    normalizer = spec.get("normalizer") or {}
    if not isinstance(normalizer, dict):
        _refuse("Qwen normalizer must be a record")
    meta = [
        "tokens %d" % len(raw), "merges %d" % count,
        "specials %d" % len(specials), "bytes %d" % len(tokens),
        "normalizer %s" % normalizer.get("type", "none"),
        "pattern %s" % pattern,
    ]
    return {
        "tokenizer-tokens.bin": tokens,
        "tokenizer-tokens.index": index,
        "tokenizer-merges.bin": triples,
        "tokenizer-specials.txt": ("\n".join(lines) + "\n").encode("utf-8"),
        "tokenizer.index": ("\n".join(meta) + "\n").encode("utf-8"),
    }


def _gemma_bytes(spec, vocab, ordered, merges):
    if spec["model"].get("byte_fallback") is not True:
        _refuse("Gemma BPE requires selected byte_fallback=true")
    raw = []
    chars = []
    fallback = [-1] * 256
    for index, word in enumerate(ordered):
        is_fallback = (len(word) == 6 and word.startswith("<0x")
                       and word.endswith(">"))
        if is_fallback:
            try:
                value = int(word[3:5], 16)
                if not 0 <= value <= 255 or fallback[value] != -1:
                    _refuse("Gemma BPE duplicate or invalid byte fallback")
            except ValueError as exc:
                raise HfTokenizerConsumerMismatch(
                    "selected Gemma invalid <0xXX> fallback token") from exc
            fallback[value] = index
            raw.append(bytes([value]))
        else:
            raw.append(word.encode("utf-8"))
        if len(word) == 1:
            chars.append((ord(word), index))
    chars.sort()
    tokens, index = _flatten_tokens(raw)
    triples, count = _merges(vocab, merges)
    chars_blob = b"".join(struct.pack("<Ii", cp, ident) for cp, ident in chars)
    fallback_blob = b"".join(struct.pack("<i", ident) for ident in fallback)
    specials = spec.get("added_tokens", [])
    if not isinstance(specials, list):
        _refuse("Gemma added_tokens must be a list")
    lines = [str(len(specials))]
    for item in specials:
        if not isinstance(item, dict):
            _refuse("Gemma added token must be a record")
        content = str(item["content"]).replace("\n", "\\n")
        lines.append("%d %d %s" % (
            item["id"], 1 if item.get("special") else 0, content))
    bos = -1
    postprocessor = spec.get("post_processor") or {}
    if not isinstance(postprocessor, dict):
        _refuse("Gemma post_processor must be a record")
    for name, info in (postprocessor.get("special_tokens") or {}).items():
        if name == "<bos>" and info.get("ids"):
            bos = info["ids"][0]
    normalizer = spec.get("normalizer") or {}
    if not isinstance(normalizer, dict):
        _refuse("Gemma normalizer must be a record")
    meta = [
        "style sentencepiece", "tokens %d" % len(raw),
        "merges %d" % count, "chars %d" % len(chars),
        "specials %d" % len(specials), "bytes %d" % len(tokens),
        "bos %d" % bos, "space_meta 2581",
        "normalizer %s" % normalizer.get("type", "none"),
    ]
    return {
        "tokenizer-tokens.bin": tokens,
        "tokenizer-tokens.index": index,
        "tokenizer-merges.bin": triples,
        "tokenizer-chars.bin": chars_blob,
        "tokenizer-bytes.bin": fallback_blob,
        "tokenizer-specials.txt": ("\n".join(lines) + "\n").encode("utf-8"),
        "tokenizer.index": ("\n".join(meta) + "\n").encode("utf-8"),
    }


def derive_pascal_tokenizer_bytes(model_key, spec):
    """Reproduce the legacy exported Pascal bytes with zero file writes."""
    if model_key not in ("qwen2.5-0.5b", "qwen2.5-1.5b", "gemma-2-2b"):
        _refuse("unsupported Pascal tokenizer model family")
    if not isinstance(spec, dict) or not isinstance(spec.get("model"), dict):
        _refuse("HF tokenizer.json lacks model section")
    model = spec["model"]
    if model.get("type") != "BPE":
        _refuse("HF tokenizer model is not the selected BPE format")
    vocab = model.get("vocab")
    ordered = _words(vocab)
    merges = model.get("merges")
    try:
        result = (_qwen_bytes(spec, vocab, ordered, merges)
                  if model_key in ("qwen2.5-0.5b", "qwen2.5-1.5b")
                  else _gemma_bytes(spec, vocab, ordered, merges))
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError) as exc:
        if isinstance(exc, HfTokenizerConsumerMismatch):
            raise
        raise HfTokenizerConsumerMismatch(
            "selected HF token metadata unsupported by Pascal exporter") from exc
    expected_files = set(llm_bundle_contract._FAMILIES[model_key][3])
    if set(result) != expected_files:
        _refuse("derived native Pascal files differ from supported family contract")
    if sum(map(len, result.values())) > MAX_GENERATED_TOKENIZER_BYTES:
        _refuse("derived Pascal tokenizer binary aggregate exceeds quota")
    return result


def bind_held_native_pascal_tokenizer(stage_hf, stage_native, selected_hf_entries,
                                      approved_consumer_inventory, *, model_key):
    """Byte-bind selected held HF tokenizer to already-held selected native kit.

    The invoking caller must hold EVERY private HF and consumer-kit file under
    Win32 read/share-only handles from before this check through native child
    exit. This verifies content identity now; it cannot verify Pascal runtime.
    """
    if not isinstance(approved_consumer_inventory, bundle_atomic.Inventory):
        _refuse("independently selected consumer Inventory is required")
    if not isinstance(selected_hf_entries, list):
        _refuse("selected HF source member inventory absent")
    selected = [row for row in selected_hf_entries if row.get("path") == "tokenizer.json"]
    if len(selected) != 1:
        _refuse("selected HF tokenizer.json SHA member absent")
    item = selected[0]
    if (not isinstance(item.get("sha256"), str) or
            len(item["sha256"]) != 64 or
            type(item.get("bytes")) is not int or
            not 0 < item["bytes"] <= hf_closure.MAX_TOKENIZER_BYTES):
        _refuse("selected tokenizer original SHA/size pin is malformed")
    path = Path(stage_hf) / "tokenizer.json"
    source_freeze._safe_ancestors(path)
    source_info = source_freeze._lstat(path)
    if (not stat.S_ISREG(source_info.st_mode) or
            getattr(source_info, "st_nlink", 1) != 1 or
            source_info.st_size != item["bytes"]):
        _refuse("selected held tokenizer source leaf is missing/unsafe")
    with path.open("rb") as stream:
        fd_info = os.fstat(stream.fileno())
        if source_freeze._descriptor_identity(fd_info) != (
                source_freeze._descriptor_identity(source_info)):
            _refuse("held tokenizer source changed before read")
        payload = stream.read(item["bytes"] + 1)
        if (len(payload) != item["bytes"] or
                hashlib.sha256(payload).hexdigest() != item["sha256"] or
                source_freeze._descriptor_identity(os.fstat(stream.fileno())) !=
                source_freeze._descriptor_identity(fd_info)):
            _refuse("held tokenizer source does not equal approved SHA")
    try:
        spec = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise HfTokenizerConsumerMismatch(
            "selected held tokenizer.json is not strict UTF-8 JSON") from exc
    derived = derive_pascal_tokenizer_bytes(model_key, spec)
    selected_native = {
        member.path: member for member in approved_consumer_inventory.members
        if member.kind == "file"
    }
    result = {}
    for filename, expected in derived.items():
        member = selected_native.get(filename)
        if member is None or member.size != len(expected):
            _refuse("selected Pascal %s native size differs from HF tokenizer" %
                    filename)
        file = Path(stage_native) / filename
        source_freeze._safe_ancestors(file)
        before = source_freeze._lstat(file)
        if (not stat.S_ISREG(before.st_mode) or
                getattr(before, "st_nlink", 1) != 1 or
                before.st_size != len(expected)):
            _refuse("selected native Pascal tokenizer leaf missing/linked")
        actual_digest = hashlib.sha256()
        amount = 0
        with file.open("rb") as reader:
            fd_first = os.fstat(reader.fileno())
            if source_freeze._descriptor_identity(fd_first) != (
                    source_freeze._descriptor_identity(before)):
                _refuse("native Pascal tokenizer leaf changed before compare")
            while True:
                block = reader.read(1024 * 1024)
                if not block:
                    break
                actual_digest.update(block)
                amount += len(block)
                if amount > len(expected):
                    _refuse("native Pascal tokenizer leaf grew beyond approved size")
            if source_freeze._descriptor_identity(os.fstat(reader.fileno())) != (
                    source_freeze._descriptor_identity(fd_first)):
                _refuse("native Pascal tokenizer leaf changed during comparison")
        wanted = hashlib.sha256(expected).hexdigest()
        if (amount != len(expected) or actual_digest.hexdigest() != wanted or
                member.sha256 != wanted):
            _refuse("selected Pascal %s bytes differ from held HF tokenizer" %
                    filename)
        result[filename] = {"bytes": len(expected), "sha256": wanted}
    return {
        "schema": "crosspascal.hf-pascal-tokenizer-binding.v1",
        "status": "POST_WIN32_HELD_EXACT_TOKENIZER_BYTES_NOT_MODEL_UAT",
        "model_key": model_key,
        "source_tokenizer_sha256": item["sha256"],
        "native_tokenizer_files": result,
        "verification_status": "NOT_CHECKED",
        "pascal_inference": "NOT_RUN",
        "release_grade_immutable": False,
    }
