# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Image embeddings from a separately converted Gemma 3n vision bundle."""
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import tvm
from tvm import relax
from .gemma3n_workspace import CudaWorkspace


class Vision:
    def __init__(self, folder, hidden_size, device_id=0):
        self.folder = Path(folder).resolve(strict=True)
        self.manifest = json.loads(self.member('vision.json').read_text(encoding='utf-8'))
        m = self.manifest
        if m.get('schema') != 'crosspascal.gemma3n.vision.v1' or m.get('text_hidden_size') != hidden_size:
            raise ValueError('Vision bundle does not match this text architecture')
        if m.get('backend') not in ('TVM CUDA FP32', 'TVM CPU FP32'):
            raise ValueError('Unsupported vision backend')
        required = {m['library'], m['layout'], m['parameters'], m['hard_embeddings'],
                    m['hard_norm'], m['projection']}
        if not required <= m['artifact_sha256'].keys():
            raise ValueError('Vision artifact hashes are incomplete')
        for name, expected in m['artifact_sha256'].items():
            with self.member(name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Vision artifact changed: ' + name)
        self.layout = json.loads(self.member(m['layout']).read_text(encoding='utf-8'))
        self.hard = np.load(self.member(m['hard_embeddings']), allow_pickle=False)
        self.norm = np.load(self.member(m['hard_norm']), allow_pickle=False)
        self.projection = np.load(self.member(m['projection']), allow_pickle=False)
        if self.hard.shape[1:] != self.norm.shape or self.projection.shape != (hidden_size, self.hard.shape[1]):
            raise ValueError('Incompatible vision projection shapes')
        self.device_id = device_id
        self.hidden_size = hidden_size
        self.last_key, self.last_features = None, None

    def member(self, name):
        path = (self.folder / name).resolve(strict=True)
        if path.parent != self.folder or not path.is_file():
            raise ValueError('Invalid vision bundle member')
        return path

    def hard_embedding(self, token):
        index = token - self.manifest['vocab_offset']
        if not 0 <= index < len(self.hard):
            raise ValueError('Token is outside the vision embedding table')
        x = self.hard[index].copy()
        eps = self.manifest['rms_norm_eps']
        x *= 1 / np.sqrt(np.mean(x * x) + eps)
        x *= self.norm
        x = self.projection @ x
        return x / np.sqrt(np.mean(x * x) + eps)

    def prompt(self, session, text, images, chat, should_stop):
        if not isinstance(images, list) or not images or any(not isinstance(p, str) for p in images):
            raise ValueError('Images must be a non-empty list of file paths')
        vocabulary = session.vocabulary
        markers = {word: vocabulary.index(word) for word in
                   ('<start_of_image>', '<end_of_image>', '<image_soft_token>')}
        if markers['<end_of_image>'] != self.manifest['vocab_offset']:
            raise ValueError('Vision vocabulary does not match the text tokenizer')
        encode = session.tokenizer.encode
        bos = session.tokenizer.bos_id()
        tokens = [bos] if bos >= 0 else []
        if chat:
            tokens += [session.special['<start_of_turn>']] + encode('user\n')
        needed = len(tokens) + len(images) * (self.manifest['soft_tokens'] + 2) + len(encode(text)) + 12
        if needed >= session.manifest['context']:
            raise ValueError('Text and image tokens do not fit the converted context; use fewer images or a larger context')
        overrides, report = {}, []
        for path in images:
            features, info = self.encode(path, should_stop)
            report.append(info)
            tokens.append(markers['<start_of_image>'])
            for row in features:
                overrides[len(tokens)] = row
                tokens.append(markers['<image_soft_token>'])
            overrides[len(tokens)] = self.hard_embedding(markers['<end_of_image>'])
            tokens.append(markers['<end_of_image>'])
        tokens += encode('\n' + text)
        if chat:
            tokens += [session.special['<end_of_turn>']] + encode('\n') + [session.special['<start_of_turn>']] + encode('model\n')
        return tokens, overrides, report

    def encode(self, path, should_stop=None):
        from PIL import Image, ImageOps
        path = Path(path).resolve(strict=True)
        with path.open('rb') as stream:
            key = hashlib.file_digest(stream, 'sha256').hexdigest()
        if key == self.last_key:
            return self.last_features, {'reused': True, 'seconds': 0, 'sha256': key}
        started = time.perf_counter()
        with Image.open(path) as original:
            if original.width * original.height > 64_000_000:
                raise ValueError('Image is too large; resize it below 64 megapixels')
            image = ImageOps.exif_transpose(original).convert('RGB')
            size = self.manifest['image_size']
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype='float32') / 255
        pixels = (pixels - np.asarray(self.manifest['mean'], dtype='float32')) / np.asarray(self.manifest['std'], dtype='float32')
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        if should_stop and should_stop():
            raise InterruptedError('Image encoding cancelled')
        arrays = [pixels]
        total = self.member(self.manifest['parameters']).stat().st_size
        for row in self.layout:
            count = math.prod(row['shape']) * 4
            if row['offset'] < 0 or row['offset'] + count > total:
                raise ValueError('Vision parameter range is invalid')
            arrays.append(np.memmap(self.member(self.manifest['parameters']), mode='r', dtype='float32',
                                    offset=row['offset'], shape=tuple(row['shape'])))
        device = tvm.cuda(self.device_id) if self.manifest['backend'] == 'TVM CUDA FP32' else tvm.cpu()
        if not device.exist:
            raise RuntimeError('Selected vision device is unavailable')
        values, vm, workspace, claim = [], None, None, None
        library = self.member(self.manifest['library'])
        try:
            values = [tvm.runtime.tensor(a, device) for a in arrays]
            if self.manifest['backend'] == 'TVM CUDA FP32':
                claim = CudaWorkspace.claim(library)
            vm = relax.VirtualMachine(tvm.runtime.load_module(str(library)), device)
            if claim is not None:
                workspace = CudaWorkspace(library, self.device_id, maximum_bytes=2 * 1024**3, claimed_key=claim)
            features = vm['main'](*values).numpy()
            device.sync()
            if workspace:
                workspace.check()
        finally:
            if workspace:
                workspace.close()
            elif claim:
                CudaWorkspace.release_claim(claim)
            values.clear()
            vm = None
            gc.collect()
            device.sync()
        if features.shape != (1, self.manifest['soft_tokens'], self.hidden_size) or not np.isfinite(features).all():
            raise RuntimeError('Invalid native vision embeddings')
        if should_stop and should_stop():
            raise InterruptedError('Image encoding cancelled')
        self.last_key, self.last_features = key, features[0]
        return features[0], {'reused': False, 'seconds': time.perf_counter() - started,
                             'sha256': key, 'backend': self.manifest['backend']}
