# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
# -*- coding: utf-8 -*-
"""Build a registered language model with the installed CrossPascal toolchain."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import stat
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


import tvm
from tvm import relax
from tvm.target import Target

from pascalvision_convert.convert import _shared_library_builder
from pascalvision_convert.convert import _teach_nvrtc_where_cccl_lives as teach_nvrtc
from pascalvision_convert.tuning import cpu_target
from pascalvision_convert.contract import build_text_contract
from pascalvision_convert.manifest import emit_manifest_text, emit_manifest_json
from pascalvision_convert import build_plan as product_build_plan
from pascalvision_convert import targets as product_targets
from pascalvision_convert import hf_closure
from pascalvision_convert import hf_held_stage, source_lock
from pascalvision_convert import llm_build_profile
from pascalvision_convert import model_sources
from pascalvision_convert import qwen_universal
from pascalvision_convert.llm_family_profile import load_source_config
from pascalvision_convert.llm_bundle_contract import validate_llm_output

import qwen2_model
import gemma2_model
import weights


ZOO = os.environ.get('PASCALVISION_MODEL_ZOO', '')


def zoo_path(*parts):
    """Resolve a model directory only when its root was explicitly configured."""
    if not ZOO:
        sys.exit(
            'REFUSED: select --source and --out, or set PASCALVISION_MODEL_ZOO. '
            'No model directory was configured.')
    return os.path.join(ZOO, *parts)


REGISTRY = {
    'qwen2.5-0.5b': dict(
        name='Qwen2.5-0.5B-Instruct', dir='qwen-0.5b',
        repo='Qwen/Qwen2.5-0.5B-Instruct',
        config=qwen2_model.Qwen2Config, model=qwen2_model.Qwen2ForCausalLM,
        plan=None, context=2048, chunk=512,
        tokenizer='byte_bpe', adapter='qwen2'),
    'qwen2.5-1.5b': dict(
        name='Qwen2.5-1.5B-Instruct', dir='qwen-1.5b',
        repo='Qwen/Qwen2.5-1.5B-Instruct',
        config=qwen2_model.Qwen2Config, model=qwen2_model.Qwen2ForCausalLM,
        plan=None, context=64, chunk=64,
        tokenizer='byte_bpe', adapter='qwen2'),
    qwen_universal.KEY: dict(
        name='Local Qwen2-compatible dense text', dir='qwen-compatible',
        repo=None,
        config=qwen2_model.Qwen2Config, model=qwen2_model.Qwen2ForCausalLM,
        plan=None, context=qwen_universal.CONTEXT, chunk=qwen_universal.CONTEXT,
        tokenizer='byte_bpe', adapter='qwen2'),
    'gemma-2-2b': dict(
        name='Gemma-2-2b-it', dir='gemma-2-2b',
        repo='google/gemma-2-2b-it',
        config=gemma2_model.Gemma2Config, model=gemma2_model.Gemma2ForCausalLM,
        plan=weights.plan_gemma2, context=8192, chunk=512,
        tokenizer='sentencepiece', adapter='gemma2'),
}


def _refuse_native_output_collisions(out_dir, model_key, target):
    """Refuse child-owned filenames before compilation can overwrite owner data.

    This is an early no-clobber preflight, not atomic namespace custody: another
    process can still race the subsequent native library/weights writers.
    """
    if os.path.lexists(out_dir):
        info = os.lstat(out_dir)
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or
                getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('native LLM output root is not a plain directory')
    stem = 'qwen2' if model_key in ('qwen2.5-0.5b', 'qwen2.5-1.5b',
                                   qwen_universal.KEY) else model_key
    for name in ('%s-%s.dll' % (stem, target), 'model.json',
                 'params.bin', 'params.json', 'params.index',
                 'manifest.txt', 'manifest.json'):
        if os.path.lexists(os.path.join(out_dir, name)):
            raise ValueError('native LLM output collision: %s already exists' % name)


def _refuse_native_source_output_overlap(source, out_dir):
    """Keep output separate from the HF owner tree and all its ancestors."""
    if os.name == 'nt' and (
            str(source).replace('/', '\\').startswith('\\') or
            str(out_dir).replace('/', '\\').startswith('\\')):


        raise ValueError('native LLM device/UNC namespace is not a safe '
                         'local HF source/output path')
    owner = os.path.normcase(os.path.realpath(os.path.abspath(source)))
    destination = os.path.normcase(os.path.realpath(os.path.abspath(out_dir)))
    try:


        shared_root = os.path.commonpath((owner, destination))
        overlaps = shared_root in (owner, destination)
    except ValueError:
        overlaps = False
    if overlaps:
        raise ValueError('native LLM output directory overlaps original HF '
                         'source or its ancestors')


def index_header_trouble(path, dtype, target):
    """Check the dtype and device consumed by the Pascal weight loader."""
    if not os.path.isfile(path):
        return 'weight index is missing: %s' % path
    with open(path, encoding='utf-8') as handle:
        lines = handle.read().splitlines()
    if len(lines) < 2:
        return 'weight index has fewer than two lines: %s' % path
    words = lines[1].split()
    if len(words) < 3:
        return ('weight index header %r has no explicit device in its third field'
                % lines[1])
    if words[1] != dtype or words[2] != target:
        return ('weight index declares %s %s, but the build produced %s %s'
                % (words[1], words[2], dtype, target))
    return ''


_HF_PRIVATE_STAGE_LEAF = re.compile(
    r'\.crosspascal-hf-stage-[A-Za-z0-9_-]{8,}\Z')


def checked_native_held_hf_source(owner_source, stage_source, tree_sha, model_key):
    """Refuse a digest-identical but *unheld* source before native TVM work.

    A parent-held Win32 CreateFileW(GENERIC_READ, FILE_SHARE_READ) for every
    selected stage member must cause a fresh GENERIC_WRITE open to fail with
    ERROR_SHARING_VIOLATION (32). This is a necessary sharing check, NOT
    cryptographic parent identity, release-grade immutability or model UAT.
    The owner path remains the BuildPlan source; the returned path alone feeds
    config loading and weights conversion.
    """
    if not source_lock._is_windows():
        raise ValueError('held HF source requires native Windows NTFS')
    api = source_lock._Kernel32()
    owner = hf_held_stage._directory(
        owner_source, 'approved original HF owner', api)
    stage = hf_held_stage._directory(
        stage_source, 'native held HF stage', api)
    if (stage == owner or owner in stage.parents or
            not _HF_PRIVATE_STAGE_LEAF.fullmatch(stage.name)):
        raise ValueError('held HF stage is not a distinct canonical private directory')


    hf_closure.audit_selected_hf_source(
        owner, expected_tree_sha256=tree_sha,
        expected_model_key=model_key)
    inventory = hf_closure.audit_selected_hf_source(
        stage, expected_tree_sha256=tree_sha,
        expected_model_key=model_key)
    if (not 1 <= inventory['file_count'] <=
            hf_held_stage.MAX_HF_SELECTED_FILES or
            inventory['total_bytes'] > hf_held_stage.MAX_HF_SELECTED_BYTES):
        raise ValueError('held HF stage exceeds approved selected resource quota')
    for item in inventory['entries']:
        leaf = stage.joinpath(*item['path'].split('/'))
        api.assert_local_ntfs(leaf)


        candidate = api.kernel.CreateFileW(
            str(leaf), 0x40000000, source_lock.FILE_SHARE_READ, None,
            source_lock.OPEN_EXISTING,
            source_lock.FILE_FLAG_OPEN_REPARSE_POINT, None)
        if candidate != source_lock.INVALID_HANDLE_VALUE:
            api.close(candidate)
            raise ValueError('private HF stage member is not held read/share-only: '
                             + item['path'])
        error = ctypes.get_last_error()
        if error != 32:


            raise ValueError('private HF stage has no proven sharing violation '
                             'for member %s (winerror=%d)' % (item['path'], error))
    return str(stage)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=list(REGISTRY))
    p.add_argument('--source', default='',
                   help='local HF directory; otherwise use PASCALVISION_MODEL_ZOO')
    p.add_argument('--target', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--arch', default='', help='target CUDA architecture in sm_<number> form')
    p.add_argument('--shared', type=int, default=49152)
    p.add_argument('--threads', type=int, default=1024)
    p.add_argument('--dtype', default='float32', choices=['float32', 'float16'])
    p.add_argument('--context', type=int, default=0, help='context tokens; 0 uses the registry default')
    p.add_argument('--chunk', type=int, default=0, help='prefill tokens per chunk; 0 uses the registry default')
    p.add_argument('--layers', type=int, default=0, help='reduce layers for a diagnostic build')
    p.add_argument('--out', default='',
                   help='output directory; otherwise use PASCALVISION_MODEL_ZOO')
    p.add_argument('--skip-weights', action='store_true')
    p.add_argument('--apply-tuning', default='')
    p.add_argument('--plan-json', default='',
                   help='verified BuildPlan handed off by wizard/recipe CLI; '
                        'native builder checks it again before weights')
    p.add_argument('--hf-source-tree-sha256', default='',
                   help='parent-selected HF config/tokenizer/weight bytes SHA; '
                        'scanned source SHA; never by itself native-locked/VERIFIED')
    p.add_argument('--held-hf-source', default='',
                   help='separate parent-held PRIVATE Win32 HF byte root. '
                        '--source remains original BuildPlan owner path. '
                        'Valid only with --plan-json plus selected SHA.')
    return p.parse_args()


def main():
    args = parse()
    spec = REGISTRY[args.model]
    source = args.source or zoo_path(spec['dir'], 'hf')
    owner_plan = None
    if args.plan_json:
        owner_plan = product_build_plan.read_native_plan(args.plan_json)


        if (args.shared != 49152 or args.threads != 1024 or
                args.layers or args.skip_weights or
                args.apply_tuning):
            raise ValueError('native LLM BuildPlan does not yet represent '
                             'shared/threads/layers/skip-weights/'
                             'tuning overrides: do not silently ignore them')
        selected_python = owner_plan['options'].get('compiler_python')
        if selected_python is not None:
            executable = llm_build_profile.check_python(selected_python)
            if not os.path.samefile(sys.executable, executable):
                raise ValueError('native builder is running in a different Python interpreter')
        product_build_plan.validate_native_family_execution(
            owner_plan, source=source, target=args.target,
            req_sm=args.arch or None, dtype=args.dtype,
            host_os=platform.system(), host_arch=platform.machine(),
            model_key=args.model, context=args.context, chunk=args.chunk)
        if not args.hf_source_tree_sha256:
            raise ValueError('native LLM BuildPlan requires the parent HF '
                             'source SHA-256; refusing unpinned protected build')
        if not args.held_hf_source:
            raise ValueError('native LLM BuildPlan requires a separate held HF '
                             'private source; no fallback to mutable original')
    elif args.held_hf_source:
        raise ValueError('held HF private source requires owner BuildPlan and '
                         'selected HF SHA-256; direct legacy CLI cannot claim lock')
    if args.model == 'qwen2.5-1.5b' and (
            owner_plan is None or
            (owner_plan['options'].get('llm') or {}).get('profile_id') !=
            llm_build_profile.QWEN_15_CUDA_PROFILE):
        raise ValueError('Qwen2.5-1.5B requires its exact approved CUDA FP32 '
                         '64-token BuildPlan profile')
    if args.model == qwen_universal.KEY and (
            owner_plan is None or args.target != 'cuda' or
            args.dtype != 'float32' or
            (owner_plan['options'].get('llm') or {}).get('profile_id') !=
            qwen_universal.PROFILE):
        raise ValueError('Local Qwen2-compatible build requires its explicit '
                         'CUDA FP32 BuildPlan')
    compiler_source = source
    if args.held_hf_source:
        compiler_source = checked_native_held_hf_source(
            source, args.held_hf_source, args.hf_source_tree_sha256,
            args.model)


    source_identity = None
    if args.hf_source_tree_sha256:
        source_scan = hf_closure.audit_selected_hf_source(
            compiler_source, expected_tree_sha256=args.hf_source_tree_sha256,
            expected_model_key=args.model)
        if args.model == qwen_universal.KEY:
            source_identity = {
                'source_tree_sha256': source_scan['tree_sha256'],
                'source_kind': 'local-sha256'}
        elif owner_plan is not None and owner_plan['options'].get('llm') is not None:
            source_identity = model_sources.match_source(source_scan, args.model)
    context = args.context or spec['context']
    chunk = args.chunk or spec['chunk']
    if not (0 < chunk <= context):
        raise ValueError('LLM context/chunk must be positive and chunk must not exceed context')

    measured_model_key, raw = load_source_config(
        compiler_source, generic=(args.model == qwen_universal.KEY))
    if measured_model_key != args.model:
        raise ValueError('requested native LLM builder %s differs from exact '
                         'HF model profile %r; do not compile unrelated weights'
                         % (args.model, measured_model_key))
    generic_config = None
    if args.model == qwen_universal.KEY:
        generic_config = qwen_universal.validate_config(raw)
        qwen_universal.verify_tokenizer(compiler_source)
        qwen_universal.inspect_local_headers(compiler_source, generic_config)
        # Recheck host admission before graph export. GPU free memory is
        # independently checked by the Pascal runtime, not guessed here.
        import shutil
        qwen_universal.admit(
            generic_config, disk_free=shutil.disk_usage(
                os.path.dirname(os.path.abspath(args.out))).free,
            ram_free=qwen_universal.measured_available_ram())
    if args.layers:
        raw = dict(raw); raw['num_hidden_layers'] = args.layers
        print('REDUCED to %d layers for a diagnostic build' % args.layers)

    config = spec['config'](raw, dtype=args.dtype,
                            context_window_size=context, prefill_chunk_size=chunk)


    if args.model == 'gemma-2-2b' and owner_plan is not None:
        gemma2_model.require_step08_exact_gemma2_native_attention(config)
    print('model %s: %d layers, hidden size %d, head size %d, vocabulary %d'
          % (spec['name'], config.num_hidden_layers, config.hidden_size,
             config.head_dim, config.vocab_size))

    short = {'float32': 'f32', 'float16': 'f16'}[args.dtype]
    out_dir = args.out or zoo_path(
        spec['dir'], '%s-%s%s'
        % (args.target, short, ('-' + args.arch) if args.arch else ''))
    _refuse_native_source_output_overlap(source, out_dir)


    _refuse_native_source_output_overlap(compiler_source, out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    _refuse_native_output_collisions(out_dir, args.model, args.target)

    if args.target == 'cuda':
        if args.arch:
            tvm_target = Target({'kind': 'cuda', 'arch': args.arch,
                'max_shared_memory_per_block': args.shared,
                'max_threads_per_block': args.threads,
                'max_num_threads': args.threads, 'thread_warp_size': 32})
        else:
            device = tvm.cuda(0)
            if not device.exist:
                sys.exit('REFUSED: CUDA target requested, no device found, and --arch is unset')
            tvm_target = Target.from_device(device)
        teach_nvrtc()
        device_name = 'cuda'
        print('GPU: %s, shared-memory limit %s'
              % (tvm_target.attrs.get('arch'),
                 tvm_target.attrs.get('max_shared_memory_per_block')))
    else:
        tvm_target = cpu_target()
        device_name = 'cpu'
    target = tvm_target


    build_arch = str(tvm_target.attrs.get('arch') or '') if device_name == 'cuda' else ''
    artifact_cpu_arch = product_targets._norm_arch(platform.machine())
    if artifact_cpu_arch not in ('x86_64', 'arm64'):
        raise ValueError('native builder cannot describe CPU ISA %r in a '
                         'portable Pascal passport' % platform.machine())
    if owner_plan is not None:
        product_build_plan.validate_native_family_execution(
            owner_plan, source=source, target=device_name,
            req_sm=build_arch or None, dtype=args.dtype,
            host_os=platform.system(), host_arch=platform.machine(),
            actual_tvm_target=product_targets.tvm_target_spec(
                device_name, build_arch or None),
            model_key=args.model, context=args.context, chunk=args.chunk)
    print('build target: %s%s' % (device_name, (', ' + build_arch) if build_arch else ''))

    model = spec['model'](config)
    model.to(args.dtype)

    began = time.time()
    with target:
        mod, named_params = model.export_tvm(spec=model.get_default_spec())
    print('graph created in %.1f s: %d functions, %d weights'
          % (time.time() - began, len(mod.functions), len(named_params)))

    began = time.time()
    with target:
        if args.apply_tuning:
            print('applying tuning from %s' % args.apply_tuning)
            mod = relax.get_pipeline(
                'static_shape_tuning', total_trials=0, target=target,
                work_dir=args.apply_tuning, max_trials_per_task=None)(mod)
        else:
            mod = relax.get_pipeline('zero')(mod)
    with target, tvm.transform.PassContext(opt_level=3):
        executable = relax.build(mod, target=target)


    loader_stem = ('qwen2' if args.model in ('qwen2.5-0.5b', 'qwen2.5-1.5b',
                                           qwen_universal.KEY)
                   else args.model)
    library = os.path.join(out_dir, '%s-%s.dll' % (loader_stem, device_name))
    with tempfile.TemporaryDirectory(prefix='.link-', dir=out_dir) as link_dir:
        linked = os.path.join(link_dir, os.path.basename(library))
        executable.export_library(linked, fcompile=_shared_library_builder())
        os.rename(linked, library)
    print('module built in %.1f s: %s, %.1f MiB'
          % (time.time() - began, library, os.path.getsize(library) / 1048576.0))

    order = [name for name, _ in named_params]
    shapes = dict((name, tuple(int(d) for d in par.shape))
                  for name, par in named_params)

    manifest = {
        'model': spec['name'], 'model_key': args.model,
        'library': os.path.basename(library), 'target': device_name,
        'arch': build_arch,
        'dtype': args.dtype, 'built': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size,
        'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'num_key_value_heads': config.num_key_value_heads,
        'head_dim': config.head_dim, 'vocab_size': config.vocab_size,
        'context_window_size': config.context_window_size,
        'prefill_chunk_size': config.prefill_chunk_size,
        'rope_theta': config.rope_theta,
        'tie_word_embeddings': config.tie_word_embeddings,
        'tokenizer': spec['tokenizer'],
        'eos_token_id': raw.get('eos_token_id'),
        'bos_token_id': raw.get('bos_token_id'),
        'param_order': order,
    }
    if generic_config is not None:
        manifest['qwen_compatibility'] = qwen_universal.descriptor(
            generic_config, source_identity['source_tree_sha256'])
        manifest['source_kind'] = 'local-sha256'
        manifest['rope_theta'] = generic_config['rope_theta']
        manifest['rms_norm_eps'] = generic_config['rms_norm_eps']
    if owner_plan is not None:
        manifest['build_plan_sha256'] = owner_plan['plan_sha256']
        manifest['build_host_cpu_arch'] = artifact_cpu_arch
        selected_llm = owner_plan['options'].get('llm')
        if selected_llm is not None:
            manifest['llm_profile_id'] = selected_llm['profile_id']
            manifest['max_batch_size'] = selected_llm['max_batch_size']
    if source_identity is not None:
        if 'repository' in source_identity:
            manifest['source_repo'] = source_identity['repository']
            manifest['source_revision'] = source_identity['revision']
        manifest['source_tree_sha256'] = source_identity['source_tree_sha256']
    with open(os.path.join(out_dir, 'model.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print('model metadata written: model.json')


    if args.hf_source_tree_sha256:
        hf_closure.audit_selected_hf_source(
            compiler_source, expected_tree_sha256=args.hf_source_tree_sha256,
            expected_model_key=args.model)
    if args.skip_weights:
        print('weights skipped as requested'); return 0

    began = time.time()
    total, count = weights.convert(
        compiler_source, out_dir, order, shapes, config.num_hidden_layers,
        args.dtype, device_name, plan_fn=spec['plan'])
    if args.hf_source_tree_sha256:
        hf_closure.audit_selected_hf_source(
            compiler_source, expected_tree_sha256=args.hf_source_tree_sha256,
            expected_model_key=args.model)
    print('weights converted in %.1f s: %d tensors, %.1f MiB'
          % (time.time() - began, count, total / 1048576.0))


    with open(os.path.join(out_dir, 'params.json'), encoding='utf-8') as f:
        written = json.load(f)
    built_dtype = written.get('dtype')
    odd = sorted(set(t.get('dtype') for t in written.get('tensors', []))
                 - {built_dtype})
    if built_dtype != args.dtype or odd:
        sys.exit('REFUSED: requested dtype %s, wrote %s%s.'
                 % (args.dtype, built_dtype,
                    (', other tensor dtypes: ' + ', '.join(map(str, odd))) if odd else ''))
    print('weight dtype matches the request: %s, %d tensors'
          % (built_dtype, len(written.get('tensors', []))))
    if (written.get('total_bytes') != total or
            not isinstance(written.get('sha256'), str) or
            len(written['sha256']) != 64 or
            any(ch not in '0123456789abcdef' for ch in written['sha256'])):
        raise ValueError('The serialized weight digest is missing or invalid')
    manifest['params_sha256'] = written['sha256']
    manifest['params_bytes'] = total
    with open(os.path.join(out_dir, 'model.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)

    trouble = index_header_trouble(os.path.join(out_dir, 'params.index'),
                                   built_dtype, device_name)
    if trouble:
        sys.exit('REFUSED: %s' % trouble)
    print('weight index header checked: %d %s %s'
          % (len(written.get('tensors', [])), built_dtype, device_name))


    with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as f:
        description = json.load(f)
    bos = description['bos_token_id']
    if isinstance(bos, list):
        bos = bos[0]
    eos = description['eos_token_id']
    if isinstance(eos, list):
        eos = eos[0]
    contract = build_text_contract(
        spec['adapter'], description['library'],
        description['context_window_size'], bos, eos, spec['tokenizer'],
        os_=platform.system().lower(), arch=artifact_cpu_arch,
        device=device_name, dtype=built_dtype,


        device_arch=build_arch or None)


    module_hash = hashlib.sha256()
    with open(library, 'rb') as module_stream:
        for data in iter(lambda: module_stream.read(1024 * 1024), b''):
            module_hash.update(data)
    contract['artifact']['hash'] = module_hash.hexdigest()
    with open(os.path.join(out_dir, 'manifest.txt'), 'w',
              encoding='utf-8', newline='') as f:
        f.write(emit_manifest_text(contract))
    with open(os.path.join(out_dir, 'manifest.json'), 'w',
              encoding='utf-8', newline='') as f:
        f.write(emit_manifest_json(contract))
    print('runtime contract written: manifest.txt, manifest.json')
    if owner_plan is not None:
        validate_llm_output(out_dir, owner_plan, args.model,
                            require_consumer=False)
        print('native BuildPlan/model/weights/manifest consistency PASS; '
              'independent model inference NOT_CHECKED')
    if args.held_hf_source:


        hf_closure.audit_selected_hf_source(
            source, expected_tree_sha256=args.hf_source_tree_sha256,
            expected_model_key=args.model)
    return 0


if __name__ == '__main__':
    sys.exit(main())
