# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import numpy as np
import tvm
import tvm_ffi
from tvm import relax
from .generation import Generation


class Tokenizer:
    def __init__(self, member):
        self.raw = member('tokenizer-tokens.bin').read_bytes()
        self.spans = list(struct.iter_unpack('<II', member('tokenizer-tokens.index').read_bytes()))
        self.chars = dict(struct.iter_unpack('<Ii', member('tokenizer-chars.bin').read_bytes()))
        self.bytes = struct.unpack('<256i', member('tokenizer-bytes.bin').read_bytes())
        triples = struct.iter_unpack('<iii', member('tokenizer-merges.bin').read_bytes())
        self.merges = {(left, right): (rank, joined) for rank, (left, right, joined) in enumerate(triples)}
        lines = member('tokenizer-specials.txt').read_text(encoding='utf-8').splitlines()
        self.special, self.controls = {}, set()
        for line in lines[1:]:
            token, control, word = line.split(' ', 2)
            self.special[word.replace('\\n', '\n')] = int(token)
            if int(control):
                self.controls.add(int(token))
        if any(offset + count > len(self.raw) for offset, count in self.spans):
            raise ValueError('Invalid tokenizer offsets')

    def encode(self, text):
        ids = []
        for char in text.replace(' ', '\u2581'):
            if ord(char) in self.chars:
                ids.append(self.chars[ord(char)])
            else:
                ids.extend(self.bytes[value] for value in char.encode('utf-8'))
        if any(value < 0 for value in ids):
            raise ValueError('The tokenizer cannot encode an input character')
        while len(ids) > 1:
            found = [(self.merges[pair][0], i, self.merges[pair][1])
                     for i, pair in enumerate(zip(ids, ids[1:])) if pair in self.merges]
            if not found:
                break
            _, index, joined = min(found)
            ids[index:index + 2] = [joined]
        return ids

    def decode(self, ids):
        pieces = []
        for token in ids:
            if token not in self.controls:
                offset, count = self.spans[token]
                pieces.append(self.raw[offset:offset + count])
        return b''.join(pieces).decode('utf-8', errors='ignore').replace('\u2581', ' ')


class Session:
    def __init__(self, folder):
        start = time.perf_counter()
        self.folder = Path(folder).resolve(strict=True)
        self.manifest = json.loads(self.member('model.json').read_bytes())
        if self.manifest.get('model_type') != 'gemma2' or self.manifest.get('target') != 'cpu':
            raise ValueError('Expected an experimental Gemma 2 CPU bundle')
        for name, digest in ((self.manifest['library'], self.manifest['library_sha256']),
                             ('params.bin', self.manifest['params_sha256'])):
            with self.member(name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('Bundle hash mismatch: ' + name)
        layout = json.loads(self.member('params.json').read_bytes())['tensors']
        self.arrays = []
        self.parameters = []
        total = self.member('params.bin').stat().st_size
        for spec in layout:
            dtype = np.dtype(spec['dtype'])
            if dtype != np.dtype('float32') or spec['offset'] < 0 or spec['offset'] + math.prod(spec['shape']) * dtype.itemsize > total:
                raise ValueError('Invalid CPU parameter layout')
            array = np.memmap(self.member('params.bin'), mode='r', offset=spec['offset'], dtype=dtype, shape=tuple(spec['shape']))
            self.arrays.append(array)
            self.parameters.append(tvm_ffi.from_dlpack(array.__dlpack__(max_version=(1, 0))))
        self.vm = relax.VirtualMachine(tvm.runtime.load_module(str(self.member(self.manifest['library']))), tvm.cpu())
        self.tokenizer = Tokenizer(self.member)
        self.manifest['context'] = self.manifest['context_window_size']
        self.manifest['backend'] = 'TVM LLVM CPU FP32 (experimental Gemma 2)'
        self.load_seconds = time.perf_counter() - start

    def member(self, name):
        path = (self.folder / name).resolve(strict=True)
        if path.parent != self.folder or not path.is_file():
            raise ValueError('Invalid bundle member')
        return path

    def close(self):
        self.vm = None
        self.parameters.clear()
        self.arrays.clear()
        gc.collect()

    def generate(self, prompt, max_new_tokens=0, on_text=None, should_stop=None, chat=True, options=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Enter a non-empty prompt')
        encode = self.tokenizer.encode
        tokens = [self.manifest['bos_token_id']]
        if chat:
            start = self.tokenizer.special['<start_of_turn>']
            end = self.tokenizer.special['<end_of_turn>']
            tokens += [start] + encode('user\n' + prompt) + [end] + encode('\n') + [start] + encode('model\n')
        else:
            tokens += encode(prompt)
        context = self.manifest['context']
        if len(tokens) >= context:
            raise ValueError('The prompt does not fit the bundle context')
        sampling = Generation(options, tokens)
        limit, reason = sampling.limit(max_new_tokens, context - len(tokens))
        shape = tvm.runtime.ShapeTuple
        chunk = self.manifest['prefill_chunk_size']
        cache = self.vm['create_tir_paged_kv_cache'](shape([1]), shape([context]), shape([chunk]), shape([16]))
        tvm.get_global_func('vm.builtin.kv_state_add_sequence')(cache, 0)
        begin = tvm.get_global_func('vm.builtin.kv_state_begin_forward')
        end = tvm.get_global_func('vm.builtin.kv_state_end_forward')
        started = time.perf_counter()
        first, output, stop, position = None, [], reason, 0
        current = tokens[:chunk]
        stops = set(self.manifest['eos_token_id'])
        try:
            while len(output) < limit:
                if should_stop and should_stop():
                    stop = 'user_stop'
                    break
                embedded = self.vm['embed'](tvm.runtime.tensor(np.asarray([current], dtype='int32')), self.parameters)
                begin(cache, shape([0]), shape([len(current)]))
                try:
                    row, cache = self.vm['prefill' if position < len(tokens) else 'decode'](embedded, cache, self.parameters)
                finally:
                    end(cache)
                position += len(current)
                if position < len(tokens):
                    current = tokens[position:position + chunk]
                    continue
                token = sampling.choose(row.numpy())
                if first is None:
                    first = time.perf_counter() - started
                if token in stops:
                    stop = 'eos'
                    break
                output.append(token)
                if on_text:
                    on_text(self.tokenizer.decode(output))
                current = [token]
        finally:
            tvm.get_global_func('vm.builtin.kv_state_remove_sequence')(cache, 0)
        return dict(text=self.tokenizer.decode(output), token_ids=output, prompt_tokens=len(tokens),
                    generated_tokens=len(output), stop=stop, generation_options=sampling.report(),
                    first_token_seconds=first, total_seconds=time.perf_counter()-started,
                    backend=self.manifest['backend'], experimental=True)
