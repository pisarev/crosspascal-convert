# Original integration Copyright (C) 2026 Yuriy Pisarev, MIT; see LICENSE.

# Copyright 2019 Ross Wightman. Licensed under Apache-2.0; see licenses/timm-LICENSE.txt.


import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.path[:0] = [p for p in os.environ.get('CROSSPASCAL_GEMMA_PACKAGES', '').split(os.pathsep) if p]


def inspect(source):
    import gguf
    reader = gguf.GGUFReader(source)
    metadata = {k: v.contents() for k, v in reader.fields.items() if k.startswith('clip.vision.')}
    if metadata.get('clip.vision.projector_type') != 'gemma3nv':
        raise ValueError('Select a Gemma 3n vision projector GGUF')
    names = {t.name for t in reader.tensors}
    for name in ('mm.soft_emb_norm.weight', 'mm.hard_emb_norm.weight', 'mm.embedding.weight', 'mm.input_projection.weight'):
        if name not in names:
            raise ValueError('Incomplete vision projector: ' + name)
    return metadata


def export(source, output, text_manifest, vocabulary):
    import gguf
    import numpy as np
    import torch
    from torch import nn
    import timm
    import onnx
    import onnxsim
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    timm.layers.set_fused_attn(False)
    metadata = inspect(source)
    reader = gguf.GGUFReader(source)
    tensors = {t.name: t for t in reader.tensors}

    def tensor(name, shape=None):
        t = tensors[name]
        value = gguf.quants.dequantize(t.data, t.tensor_type).astype('float32')
        if shape is not None:
            if value.size != np.prod(shape):
                raise ValueError('Incompatible vision tensor: ' + name)
            value = value.reshape(shape)
        return torch.from_numpy(value.copy())

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device('meta'):
                self.encoder = timm.create_model('mobilenetv5_300m_enc', pretrained=False)
            state = {name: tensor('v.' + name.replace('blocks.', 'blk.'), tuple(value.shape))
                     for name, value in self.encoder.state_dict().items()}
            self.encoder.load_state_dict(state, strict=True, assign=True)
            self.norm = nn.Parameter(tensor('mm.soft_emb_norm.weight'))
            self.proj = nn.Parameter(tensor('mm.input_projection.weight'))

        def forward(self, pixels):
            x = self.encoder(pixels)
            x = x.flatten(2).transpose(1, 2) * (x.shape[1] ** 0.5)
            eps = metadata['clip.vision.attention.layer_norm_epsilon']
            x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * self.norm
            x = torch.nn.functional.linear(x, self.proj)
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)


    def static_fusion(self, inputs):
        size = tuple(int(v) for v in inputs[0].shape[-2:])
        resized = [torch.nn.functional.interpolate(x, size=size, mode=self.interpolation_mode)
                   if tuple(int(v) for v in x.shape[-2:]) != size else x for x in inputs]
        x = self.ffn(torch.cat(resized, dim=1))
        target = tuple(int(v) for v in self.output_resolution)
        if size != target:
            if any(a % b for a, b in zip(size, target)):
                x = torch.nn.functional.interpolate(x, size=target, mode='bilinear')
            else:
                stride = tuple(a // b for a, b in zip(size, target))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=stride, stride=stride)
        return self.norm(x)

    model = Encoder().eval()
    model.encoder.msfa.forward = types.MethodType(static_fusion, model.encoder.msfa)
    size = metadata['clip.vision.image_size']
    pixels = torch.zeros((1, 3, size, size))
    with torch.inference_mode():
        reference = model(pixels)
        if not torch.isfinite(reference).all():
            raise ValueError('Non-finite source vision encoder')
        torch.onnx.export(model, (pixels,), str(output / 'vision.onnx'), export_params=False,
            input_names=['pixels'] + [name for name, _ in model.named_parameters()], output_names=['embeddings'],
            opset_version=18, do_constant_folding=False, dynamo=False)
    graph = onnx.load(output / 'vision.onnx')
    graph, checked = onnxsim.simplify(graph)
    if not checked:
        raise ValueError('Vision shape simplification failed validation')
    onnx.save(graph, output / 'vision.onnx')
    state = model.state_dict()
    layout = []
    with (output / 'vision-params.bin').open('xb') as stream:
        for inp in graph.graph.input[1:]:
            array = state[inp.name].detach().numpy().astype('float32', copy=False)
            layout.append({'name': inp.name, 'offset': stream.tell(), 'shape': list(array.shape), 'dtype': 'float32'})
            stream.write(array.tobytes())
    (output / 'vision-parameters.json').write_text(json.dumps(layout, indent=2), encoding='utf-8')
    for source_name, name in [('mm.embedding.weight', 'vision-hard.npy'), ('mm.hard_emb_norm.weight', 'vision-hard-norm.npy'),
                              ('mm.input_projection.weight', 'vision-projection.npy')]:
        np.save(output / name, tensor(source_name).numpy())
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    cuda = text_manifest['backend'] == 'TVM CUDA W8A32'
    manifest = {'schema': 'crosspascal.gemma3n.vision.v1', 'architecture': 'gemma3nv',
        'backend': 'TVM CUDA FP32' if cuda else 'TVM CPU FP32',
        'cuda_arch': text_manifest.get('cuda_arch') if cuda else None,
        'cpu_arch': text_manifest.get('cpu_arch', 'generic'),
        'library': 'gemma3n-vision-cuda.dll' if cuda else 'gemma3n-vision-cpu.dll', 'layout': 'vision-parameters.json',
        'parameters': 'vision-params.bin', 'hard_embeddings': 'vision-hard.npy', 'hard_norm': 'vision-hard-norm.npy',
        'projection': 'vision-projection.npy', 'image_size': size, 'mean': metadata['clip.vision.image_mean'],
        'std': metadata['clip.vision.image_std'], 'rms_norm_eps': metadata['clip.vision.attention.layer_norm_epsilon'],
        'soft_tokens': int(reference.shape[1]), 'text_hidden_size': int(reference.shape[2]),
        'vocab_offset': vocabulary.index('<end_of_image>'), 'source_sha256': digest,
        'conversion_versions': {'timm': timm.__version__, 'torch': torch.__version__, 'onnx': onnx.__version__},
        'artifact_sha256': {}}
    (output / 'vision.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Vision graph and external parameters ready', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--text-bundle', type=Path)
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect(args.source)))
    else:
        export(args.source, args.output, json.loads((args.text_bundle / 'model.json').read_bytes()),
               json.loads((args.text_bundle / 'tokenizer-gguf.json').read_bytes())['tokenizer.ggml.tokens'])
