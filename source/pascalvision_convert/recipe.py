# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""The core on the command line - the face the build recipe wears.

    python -m pascalvision_convert.recipe URL-OR-PATH --into out --target cpu

WHY A SEPARATE ENTRY RATHER THAN LINES IN THE RECIPE ITSELF. While the order of
steps lived in shell text, only the recipe knew it. Three of the seven defects
found on 31.08.2026 sat exactly there - and would have repeated in the wizard
word for word, because the wizard would have retold the recipe in its own words.

Here the shell calls the CORE. The wizard calls THE SAME core. There is nowhere
for them to drift apart.

EXIT CODE: zero when every step passed, one when any of them refused. The report
goes both to the screen and to a file - the file travels with the artifact, so
whoever downloads the module can see what it was checked against.
"""

import argparse
from contextlib import nullcontext
import json
import os
import re
import shutil
import sys
import tempfile

from . import build_plan, detect, pipeline, targets
from .contract import build_contract_input, parse_float_csv
from .recipe_file import (contract_from_recipe, read_recipe,
                          write_family_receipt)
from .targets import validate_target_sm


def _unheld_onnx_external_data(path):
    """Find external ONNX tensors without ever dereferencing their sidecars.

    A recipe field can be removed/forged, so *absence of source_closure* is NOT
    proof of a self-contained graph. Until CLI has its own compiler-held
    multi-file stage, it must reject direct ONNX external_data even when the
    saved recipe intentionally omits the original wizard provenance.
    """
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
        return False
    
    
    
    with open(path, 'rb') as handle:
        head = handle.read(64)
    if (head[4:8] == b'TFL3' or head[:4] == b'PK\x03\x04' or
            (head[:1] == b'\x80' and len(head) > 1 and
             head[1] in (2, 3, 4, 5)) or detect._looks_like_safetensors(head)):
        return False
    
    
    
    detect._admit_onnx_preparse(path)
    import onnx
    from google.protobuf.message import DecodeError
    try:
        model = onnx.load(path, load_external_data=False)
    except DecodeError as error:
        
        
        
        
        
        
        raise ValueError('cannot prove unheld ONNX source self-contained '
                         'after protobuf DecodeError: %s' % error) from error
    except Exception as error:
        
        
        
        
        raise ValueError('cannot prove unheld ONNX source is self-contained '
                         'after an uncertain read/parse failure: %s'
                         % error) from error

    seen = [0]

    def tensor_ref(tensor):
        return bool(tensor.external_data or int(tensor.data_location) == 1)

    def sparse_ref(sparse):
        return tensor_ref(sparse.values) or tensor_ref(sparse.indices)

    def attribute_ref(attr, depth):
        return (attr.HasField('t') and tensor_ref(attr.t) or
                any(tensor_ref(t) for t in attr.tensors) or
                attr.HasField('sparse_tensor') and sparse_ref(attr.sparse_tensor) or
                any(sparse_ref(t) for t in attr.sparse_tensors) or
                attr.HasField('g') and graph_ref(attr.g, depth + 1) or
                any(graph_ref(g, depth + 1) for g in attr.graphs))

    def node_ref(node, depth):
        return any(attribute_ref(attr, depth) for attr in node.attribute)

    def graph_ref(graph, depth):
        seen[0] += len(graph.initializer) + len(graph.sparse_initializer) + len(graph.node)
        if depth > 32 or seen[0] > 20000:
            raise ValueError('unbounded ONNX graph cannot be proved self-contained')
        return (any(tensor_ref(t) for t in graph.initializer) or
                any(sparse_ref(t) for t in graph.sparse_initializer) or
                any(node_ref(node, depth) for node in graph.node))

    if graph_ref(model.graph, 0):
        return True
    for info in model.training_info:
        if graph_ref(info.initialization, 0) or graph_ref(info.algorithm, 0):
            return True
    for function in model.functions:
        if any(node_ref(node, 0) for node in function.node):
            return True
        if any(attribute_ref(attr, 0) for attr in function.attribute_proto):
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='pascalvision_convert.recipe',
        description='Build a model into the TVM format and compare it with the original')
    parser.add_argument('model', nargs='?', default='',
                        help='path to a file, or a URL pointing at one; omit it '
                             'only when repeating a saved build with --recipe')
    parser.add_argument('--recipe', default='', metavar='RECIPE_JSON',
                        help='repeat a saved build: read its inputs (source, '
                             'target, sm, shape, name, tuning, model contract) '
                             'from RECIPE_JSON and build from those, ignoring the '
                             'model argument and the model-shaping options. --into '
                             'and --report still say where the output goes')
    parser.add_argument('--into', default='out', help='directory for module and runtime')
    parser.add_argument('--native-family-atomic', action='store_true',
                        help='opt in to a complete native LLM family bundle committed '
                             'once on Windows NTFS; requires an ABSOLUTE new --into, '
                             '--consumer-kit-source and --consumer-kit-tree-sha256; '
                             'never claims numerical VERIFIED or crash durability')
    parser.add_argument('--consumer-kit-source', default='', metavar='ABS_DIRECTORY',
                        help='caller-selected absolute, flat native LLM tokenizer/'
                             'TVM-runtime/licence kit (read-only owner source)')
    parser.add_argument('--consumer-kit-tree-sha256', default='', metavar='SHA256',
                        help='exact SHA-256 inventory of selected native consumer '
                             'kit; mandatory with --native-family-atomic')
    parser.add_argument('--hf-selected-tree-sha256', default='', metavar='SHA256',
                        help='explicit Qwen/Gemma source-tree approval from '
                             'hf-selected-preview; opt-in with native-family-atomic '
                             'to bind model weights/tokenizer/config to held native '
                             'compiler input, not an independently trained-model PASS')
    parser.add_argument('--target', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--sm', default='', metavar='SM',
                        help='the CUDA architecture to compile for (sm_89, 89 or '
                             '8.9); controls the TVM target and bundle metadata. '
                             'Requires --target cuda. Omit for the legacy CUDA '
                             'compiler default.')
    parser.add_argument('--target-os', default='', metavar='OS',
                        help='target operating system; default is this machine. '
                             'A foreign OS needs a separate native build runner.')
    parser.add_argument('--target-cpu-arch', default='', metavar='ARCH',
                        help='target CPU architecture; default is this machine. '
                             'A foreign architecture needs a matching toolchain.')
    parser.add_argument('--cpu-codegen-mode', default='',
                        choices=('host', 'generic', 'explicit'),
                        help='opt-in tuned ONNX CPU LLVM target: host preserves '
                             'local tuning, generic uses baseline, explicit '
                             'requires --cpu-mcpu; destination execution NOT_CHECKED')
    parser.add_argument('--cpu-mcpu', default='', metavar='LLVM_CPU',
                        help='LLVM CPU name for --cpu-codegen-mode explicit; '
                             'never silently substituted with the host CPU')
    parser.add_argument('--target-driver-api', default='', metavar='CUDA_API',
                        help='optional destination CUDA driver API, e.g. 12080; '
                             'checking it prevents shipping a CUDA 13 build to '
                             'a CUDA 12-only driver. Omit when unknown.')
    parser.add_argument('--shape', default='', help='for example images:1,3,640,640')
    parser.add_argument('--dtype', default='float32', choices=('float32', 'float16'),
                        help='LLM family weight precision; raw ONNX/TFLite uses float32')
    parser.add_argument('--name', default='model', help='base name of the module')
    parser.add_argument('--report', default='', help='where to write the report')
    parser.add_argument('--tune', default='', metavar='WORK_DIR',
                        help='tune the cpu build, keeping the database in WORK_DIR')
    parser.add_argument('--trials-per-task', type=int, default=0, metavar='N',
        help='trials per tuning task; the budget is N times the '
             'number of tasks the model turns out to have')
    parser.add_argument('--paddle-role', default='', choices=('det', 'rec'),
                        help='explicit Paddle inference role for selected source')
    parser.add_argument('--paddle-dictionary', default='', metavar='PATH',
                        help='explicit recognition charset path (required for rec)')
    parser.add_argument('--paddle-tree-sha256', default='', metavar='SHA256',
                        help='approved full selected graph/weights/dictionary digest')
    parser.add_argument('--paddle-native-hold', action='store_true',
                        help='require native Win32-held selected Paddle stage during WSL '
                             'export; no unheld fallback (not release grade)')

    
    
    
    contract_group = parser.add_argument_group(
        'model contract',
        'describe the model so a bundle manifest can be written; --task turns '
        'this on, and without it nothing changes')
    contract_group.add_argument(
        '--task', default='',
        help='task kind, e.g. object_detection, classification, pose, '
             'semantic_segmentation; enables writing the bundle manifest')
    contract_group.add_argument(
        '--adapter', default='',
        help='pre/post-processor id, e.g. yolo11, rtdetr, tensorflow, birefnet '
             '(required when --task is given)')
    contract_group.add_argument(
        '--adapter-api-version', type=int, default=1,
        help='adapter pre/postprocess version (default 1)')
    contract_group.add_argument(
        '--layout', default='NCHW', choices=['NCHW', 'NHWC'],
        help='input tensor layout (default NCHW)')
    contract_group.add_argument(
        '--score-threshold', type=float, default=None,
        help='detection score threshold (object_detection)')
    contract_group.add_argument(
        '--nms-threshold', type=float, default=None,
        help='detection NMS threshold (object_detection; some detectors have none)')
    contract_group.add_argument(
        '--mean', default='',
        help='per-channel normalisation mean, comma separated (classification)')
    contract_group.add_argument(
        '--std', default='',
        help='per-channel normalisation std, comma separated (classification)')
    contract_group.add_argument(
        '--keypoint-count', type=int, default=None,
        help='number of keypoints (pose)')
    contract_group.add_argument(
        '--labels', default='',
        help='path to a labels file to record with the bundle')
    contract_group.add_argument(
        '--license', default='',
        help='caller-declared source model licence (e.g. Apache-2.0), saved in '
             'the recipe; RAW manifest may record this label, native LLM '
             'manifest does not attest model licence or redistribution rights')
    args = parser.parse_args(argv)
    if args.recipe and (args.cpu_codegen_mode or args.cpu_mcpu):
        parser.error('saved --recipe defines its own CPU codegen target; '
                     'do not override it with --cpu-codegen-mode/--cpu-mcpu')
    if args.cpu_mcpu and args.cpu_codegen_mode != 'explicit':
        parser.error('--cpu-mcpu requires --cpu-codegen-mode explicit')
    if args.cpu_codegen_mode == 'explicit' and not args.cpu_mcpu:
        parser.error('--cpu-codegen-mode explicit requires --cpu-mcpu')
    if args.cpu_codegen_mode and (args.target != 'cpu' or not args.tune):
        parser.error('--cpu-codegen-mode requires --target cpu and --tune')
    native_family_atomic = args.native_family_atomic
    if args.hf_selected_tree_sha256:
        if (not re.fullmatch(r'[0-9a-f]{64}', args.hf_selected_tree_sha256)
                or not native_family_atomic or args.recipe):
            parser.error('HF source-tree Preview SHA requires a complete lowercase '
                         'SHA256 and direct --native-family-atomic build, not '
                         'an unrecorded saved recipe or unheld legacy route')
    if native_family_atomic:
        if (not args.consumer_kit_source or
                not re.fullmatch(r'[0-9a-f]{64}', args.consumer_kit_tree_sha256)):
            parser.error('native LLM atomic opt-in requires --consumer-kit-source '
                         'and its full lowercase --consumer-kit-tree-sha256')
        if (not os.path.isabs(args.into) or
                not os.path.isabs(args.consumer_kit_source)):
            parser.error('native LLM atomic output and selected consumer kit '
                         'must use absolute local paths')
    elif args.consumer_kit_source or args.consumer_kit_tree_sha256:
        parser.error('selected native consumer kit flags require the explicit '
                     '--native-family-atomic opt-in')
    paddle_flags = (args.paddle_role or args.paddle_dictionary or
                    args.paddle_tree_sha256 or args.paddle_native_hold)
    if args.recipe and paddle_flags:
        parser.error('Paddle direct source options cannot override an unapproved '
                     'saved recipe; selected Paddle CLI replay is not yet wired')
    saved_paddle = None

    
    
    
    
    if args.recipe:
        if args.license:
            parser.error('saved --recipe defines its own model license; '
                         '--license cannot silently override the saved input')
        if args.dtype != 'float32':
            parser.error('saved --recipe defines its own dtype; --dtype must '
                         'not be silently ignored')
        if args.model:
            parser.error('--recipe repeats a saved build and takes the source '
                         'from the recipe; do not also pass a model argument')
        try:
            recipe = read_recipe(args.recipe)
            contract = contract_from_recipe(recipe)
            saved_paddle = recipe.get('paddle_selected_source')
        except (OSError, ValueError) as trouble:
            parser.error(str(trouble))
        if recipe.get('source_closure') is not None:
            
            
            
            
            
            parser.error('saved ONNX external_data source_closure requires '
                         'native multi-file staged replay, not yet wired to '
                         'plain CLI; use a new wizard Preview/Build instead')
        source = recipe.get('source')
        if not source:
            parser.error('the recipe names no source, so there is nothing to build')
        target = recipe.get('target') or 'cpu'
        family_recipe = recipe.get('family_builder')
        expected_family_plan = recipe.get('build_plan') if family_recipe else None
        saved_raw_plan = recipe.get('build_plan') if not family_recipe else None
        cpu_codegen = None
        if saved_raw_plan is not None and 'cpu_codegen' in saved_raw_plan:
            admitted = saved_raw_plan['cpu_codegen']
            cpu_codegen = {'mode': admitted['mode'],
                           'mcpu': admitted['mcpu'] if admitted['mode'] == 'explicit' else None}
        dtype = family_recipe['dtype'] if family_recipe else 'float32'
        shape = recipe.get('shape') or None
        name = recipe.get('name') or 'model'
        req_sm = recipe.get('sm') or None
        saved_profile = recipe.get('target_profile') or {}
        deployment = saved_profile.get('deployment') or {}
        req_os = deployment.get('os') or None
        req_arch = deployment.get('arch') or None
        req_driver_api = deployment.get('driver_api')
        if args.target_os or args.target_cpu_arch or args.target_driver_api:
            parser.error('the saved recipe already defines its destination; '
                         'do not override it silently')
        trials = recipe.get('trials_per_task') or None
        
        
        
        
        license = recipe.get('license') or None
        
        
        
        
        tune_work_dir = (os.path.join(args.into, 'tuning-db')
                         if recipe.get('tune') else None)
    else:
        if not args.model:
            parser.error('a model (a path or a URL) is required unless --recipe '
                         'is given')
        
        
        
        
        try:
            contract = build_contract_input(
                args.task, args.adapter,
                adapter_api_version=args.adapter_api_version,
                layout=args.layout,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                mean=parse_float_csv(args.mean),
                std=parse_float_csv(args.std),
                keypoint_count=args.keypoint_count,
                labels=args.labels)
        except ValueError as trouble:
            parser.error(str(trouble))
        source = args.model
        target = args.target
        family_recipe = None
        expected_family_plan = None
        saved_raw_plan = None
        cpu_codegen = ({'mode': args.cpu_codegen_mode,
                        'mcpu': args.cpu_mcpu if args.cpu_codegen_mode == 'explicit' else None}
                       if args.cpu_codegen_mode else None)
        dtype = args.dtype
        shape = args.shape or None
        name = args.name
        req_sm = args.sm or None
        req_os = args.target_os or None
        req_arch = args.target_cpu_arch or None
        req_driver_api = args.target_driver_api or None
        trials = args.trials_per_task or None
        tune_work_dir = args.tune or None
        license = args.license or None

    if native_family_atomic:
        
        
        if (not isinstance(source, str) or not os.path.isabs(source) or
                not os.path.isdir(source) or source.lower().startswith(
                    ('http://', 'https://'))):
            parser.error('native LLM atomic opt-in needs an absolute local '
                         'HuggingFace model directory')
        if paddle_flags or saved_paddle is not None or tune_work_dir:
            parser.error('native LLM atomic opt-in cannot combine Paddle '
                         'selection or a tuning database')
        if os.path.lexists(args.into):
            print('FAILED native family atomic: final --into is already occupied; '
                  'owner bytes must not be overwritten', file=sys.stderr)
            return 1
        from . import bundle_atomic
        try:
            bundle_atomic._local_absolute(args.into)
            bundle_atomic._local_absolute(args.consumer_kit_source)
            
            
            parent = os.path.normcase(os.path.realpath(
                os.path.dirname(os.path.abspath(args.into))))
            for owner in (source, args.consumer_kit_source):
                original = os.path.normcase(os.path.realpath(owner))
                try:
                    overlapping = os.path.commonpath((original, parent)) == original
                except ValueError:
                    overlapping = False  
                if overlapping:
                    raise ValueError('private native family stage would be '
                                     'inside an original selected owner tree')
        except (OSError, ValueError, TypeError) as trouble:
            print('FAILED native family atomic path preflight: %s' % trouble,
                  file=sys.stderr)
            return 1

    if cpu_codegen is not None and (target != 'cpu' or not tune_work_dir or
                                    family_recipe is not None):
        parser.error('selected CPU codegen requires a tuned raw CPU ONNX build')
    if cpu_codegen is not None and isinstance(source, str) and os.path.isdir(source):
        parser.error('selected CPU codegen requires an ONNX file, not a model directory')

    
    try:
        req_sm = validate_target_sm(target, req_sm)
    except ValueError as trouble:
        parser.error(str(trouble))

    
    
    
    if isinstance(source, str) and os.path.isfile(source):
        try:
            if cpu_codegen is not None and detect.recognise(source).kind != detect.ONNX:
                parser.error('selected CPU codegen requires ONNX, not TFLite/other format')
            if _unheld_onnx_external_data(source):
                parser.error('ONNX external_data requires an explicit selected '
                             'model+sidecar Win32 held stage; plain CLI has '
                             'no supported safe replay/opt-in route yet')
        except (ImportError, OSError, ValueError) as trouble:
            parser.error('cannot prove direct CLI source self-contained: %s' % trouble)

    
    
    
    
    paddle_approved = None
    selected_role = (saved_paddle['role'] if saved_paddle is not None
                     else args.paddle_role)
    selected_dictionary = (
        (saved_paddle['dictionary'] or {}).get('reference', '')
        if saved_paddle is not None else args.paddle_dictionary)
    selected_tree_sha = (
        saved_paddle['tree_sha256'] if saved_paddle is not None
        else args.paddle_tree_sha256)
    selected_hold = (saved_paddle is not None or args.paddle_native_hold)
    if isinstance(source, str) and os.path.isdir(source):
        try:
            import import_adapters
            discovered = import_adapters.find_adapter(
                source, detect.recognise(source))
        except (ImportError, OSError, ValueError) as trouble:
            print('FAILED early source discovery: %s' % trouble, file=sys.stderr)
            return 1
        if discovered is not None and discovered.name == 'paddleocr':
            if (args.recipe and saved_paddle is None or
                    not (selected_hold and selected_role and selected_tree_sha)):
                print('FAILED: Paddle CLI requires explicitly approved role, '
                      'selected-tree SHA-256 and --paddle-native-hold; saved '
                      'recipe replay requires a recorded selected closure',
                      file=sys.stderr)
                return 1
            if (selected_role == 'rec' and not selected_dictionary or
                    selected_role == 'det' and selected_dictionary):
                print('FAILED: Paddle rec requires caller-selected dictionary; '
                      'det rejects a dictionary', file=sys.stderr)
                return 1
            if os.path.lexists(args.into):
                print('FAILED: selected Paddle CLI output already exists; '
                      'refusing to overwrite owner files', file=sys.stderr)
                return 1
            from . import paddle_closure
            try:
                paddle_approved = paddle_closure.audit_selected_paddle_source(
                    source, role=selected_role,
                    dictionary=selected_dictionary or None,
                    expected_tree_sha256=selected_tree_sha)
                if saved_paddle is not None and paddle_approved != saved_paddle:
                    raise ValueError('saved Paddle selected closure differs from '
                                     'actual original model/dictionary bytes')
                
                
                preflight = build_plan.resolve(
                    targets.host(), route='raw', source=source,
                    target=target, req_sm=req_sm, req_os=req_os,
                    req_arch=req_arch, req_driver_api=req_driver_api,
                    shape=shape, tune=bool(tune_work_dir),
                    trials_per_task=trials, contract=contract)
                if saved_paddle is not None and preflight != recipe['build_plan']:
                    raise ValueError('saved Paddle BuildPlan/target differs on replay')
                source_root = os.path.normcase(os.path.realpath(source))
                destination = os.path.normcase(os.path.realpath(args.into))
                try:
                    shared_root = os.path.commonpath((source_root, destination))
                except ValueError:
                    
                    
                    
                    
                    
                    shared_root = None
                if shared_root == source_root:
                    raise ValueError('selected Paddle output inside original model')
            except (OSError, ValueError, TypeError) as trouble:
                print('FAILED selected Paddle CLI preflight: %s' % trouble,
                      file=sys.stderr)
                return 1
        elif paddle_flags or saved_paddle is not None:
            print('FAILED: Paddle selection options require a registered '
                  'Paddle inference directory', file=sys.stderr)
            return 1
    elif paddle_flags or saved_paddle is not None:
        print('FAILED: Paddle options require a local Paddle directory',
              file=sys.stderr)
        return 1

    
    
    
    
    
    
    path = source
    if path.lower().startswith(('http://', 'https://')):
        print('Downloading the model...')
        try:
            with tempfile.TemporaryDirectory(prefix='pascalvision-cli-source-') as temporary:
                downloaded = pipeline.download(path, temporary)
                if _unheld_onnx_external_data(downloaded):
                    print('FAILED: downloaded ONNX external_data cannot be compiled '
                          'through unheld CLI; use local wizard selective freeze',
                          file=sys.stderr)
                    return 1
                if not os.path.isfile(downloaded):
                    raise ValueError('download did not produce a regular model file')
                if (cpu_codegen is not None and
                        detect.recognise(downloaded).kind != detect.ONNX):
                    raise ValueError('selected CPU codegen requires a downloaded ONNX model')
                if not os.path.isdir(args.into):
                    os.makedirs(args.into)
                path = os.path.join(args.into, 'model.bin')
                
                with open(downloaded, 'rb') as original, open(path, 'xb') as output:
                    shutil.copyfileobj(original, output, length=1048576)
        except (ImportError, OSError, ValueError) as trouble:
            print('FAILED: cannot safely stage downloaded model: %s'
                  % trouble, file=sys.stderr)
            return 1
    elif not native_family_atomic and not os.path.isdir(args.into):
        os.makedirs(args.into)

    
    
    
    
    adapter = None
    if (family_recipe or os.path.isdir(path) or
            str(path).lower().endswith(('.pt', '.pth', '.pdmodel', '.safetensors'))):
        try:
            import import_adapters
            recognition = detect.recognise(path)
            adapter = import_adapters.find_adapter(path, recognition)
        except (ImportError, OSError, ValueError) as trouble:
            print('FAILED family source discovery: %s' % trouble,
                  file=sys.stderr)
            return 1

    if (native_family_atomic and
            (adapter is None or getattr(adapter, 'kind', None) != 'full_build')):
        print('FAILED native family atomic: source has no registered native '
              'full-build LLM adapter; no output created', file=sys.stderr)
        return 1

    if family_recipe or (adapter is not None and
                         getattr(adapter, 'kind', None) == 'full_build'):
        if adapter is None or getattr(adapter, 'kind', None) != 'full_build':
            print('FAILED: saved LLM family builder was not found for this source',
                  file=sys.stderr)
            return 1
        key = adapter._key_for(path)
        if family_recipe and family_recipe['model_key'] != key:
            print('FAILED: saved model-family key disagrees with source: %r vs %r'
                  % (family_recipe['model_key'], key), file=sys.stderr)
            return 1
        
        
        
        if args.hf_selected_tree_sha256:
            try:
                if adapter.name != 'llm' or key not in ('qwen2.5-0.5b', 'gemma-2-2b'):
                    raise ValueError('selected HF SHA is only supported for Qwen/Gemma')
                from . import hf_closure
                chosen_hf = hf_closure.audit_selected_hf_source(
                    path, expected_tree_sha256=args.hf_selected_tree_sha256,
                    expected_model_key=key)
                if (chosen_hf['tree_sha256'] != args.hf_selected_tree_sha256 or
                        chosen_hf['status'] != 'SCANNED_NOT_LOCKED' or
                        chosen_hf['verification_status'] != 'NOT_CHECKED' or
                        chosen_hf['release_grade_immutable'] is not False):
                    raise ValueError('HF selected source Preview closure invalid')
            except (OSError, ValueError, TypeError, KeyError) as trouble:
                print('FAILED approved HF CLI Preview before any native stage: %s'
                      % trouble, file=sys.stderr)
                return 1
        if (args.report and os.path.abspath(args.report) !=
                os.path.abspath(os.path.join(args.into, 'report.txt'))):
            print('FAILED: native family receipt currently writes its report '
                  'inside the bundle; custom --report cannot be ignored',
                  file=sys.stderr)
            return 1
        private_stage = None
        private_stage_identity = None
        try:
            plan = build_plan.resolve(
                targets.host(), target=target, req_sm=req_sm,
                req_os=req_os, req_arch=req_arch,
                req_driver_api=req_driver_api, route='family_full_build',
                source=source, dtype=dtype,
                shape=shape, tune=bool(tune_work_dir),
                trials_per_task=trials)
            if expected_family_plan is not None:
                selected = plan['target_profile']['deployment']
                original = expected_family_plan['target_profile']['deployment']
                if (selected != original or
                        plan['options'] != expected_family_plan['options']):
                    raise ValueError('replayed family BuildPlan destination/choices '
                                     'differ from saved recipe')
            arch = plan['target_profile']['deployment']['sm'] or ''
            output = args.into
            consumer_inventory = None
            kwargs = {}
            if args.hf_selected_tree_sha256:
                
                
                
                kwargs['approved_hf_source_tree_sha256'] = (
                    args.hf_selected_tree_sha256)
            if native_family_atomic:
                from . import bundle_publication, bundle_atomic
                consumer_inventory = bundle_publication.inspect_selected_native_consumer_kit(
                    args.consumer_kit_source, model_key=key, device=target,
                    expected_tree_sha256=args.consumer_kit_tree_sha256)
                if os.path.lexists(args.into):
                    raise FileExistsError('native family final output is occupied')

                def created(path, identity):
                    nonlocal private_stage, private_stage_identity
                    private_stage = path
                    private_stage_identity = identity

                output = bundle_publication.create_private_stage(
                    args.into, capacity_profile='native_family',
                    on_created=created)
                if (private_stage != output or private_stage_identity is None):
                    raise RuntimeError('native family stage creation identity missing')
                observed = bundle_atomic._guard_stat(output, directory=True)
                if ((observed.st_dev, observed.st_ino) != private_stage_identity):
                    raise RuntimeError('native family private stage identity changed')
                bundle_publication.stage_selected_native_consumer_kit(
                    args.consumer_kit_source, output, consumer_inventory,
                    model_key=key, device=target)
                
                
                
                kwargs.update({
                    'consumer_kit_source': args.consumer_kit_source,
                    'approved_consumer_inventory': consumer_inventory})
            if native_family_atomic:
                from . import native_consumer_hold
                custody = native_consumer_hold.hold_private_native_consumer_stage(
                    output, consumer_inventory, model_key=key, device=target)
            else:
                custody = nullcontext()
            
            
            
            
            with custody:
                built = adapter.build(path, output, target=target,
                                      arch=arch, dtype=dtype, build_plan=plan,
                                      **kwargs)
                if args.hf_selected_tree_sha256:
                    
                    
                    
                    provenance = built.get('provenance') if type(built) is dict else None
                    selected_source = (provenance.get('hf_source_closure')
                                       if type(provenance) is dict else None)
                    locked = (provenance.get('compiler_held_source')
                              if type(provenance) is dict else None)
                    tokenizer_binding = (provenance.get('tokenizer_consumer_binding')
                                         if type(provenance) is dict else None)
                    selected_tokenizer = next(
                        (row for row in chosen_hf['entries']
                         if row['path'] == 'tokenizer.json'), None)
                    native_tokenizer = (tokenizer_binding.get('native_tokenizer_files')
                                        if type(tokenizer_binding) is dict else None)
                    from .llm_bundle_contract import _FAMILIES
                    exact_tokenizer_names = set(_FAMILIES[key][3])
                    approved_kit_files = {
                        member.path: member
                        for member in consumer_inventory.members if member.kind == 'file'}
                    if (not isinstance(selected_source, dict) or
                            not isinstance(locked, dict) or
                            not isinstance(selected_tokenizer, dict) or
                            not isinstance(tokenizer_binding, dict) or
                            type(native_tokenizer) is not dict or
                            set(native_tokenizer) != exact_tokenizer_names or
                            any((type(native_tokenizer[name]) is not dict or
                                 native_tokenizer[name].get('bytes') != approved_kit_files[name].size or
                                 native_tokenizer[name].get('sha256') != approved_kit_files[name].sha256)
                                for name in exact_tokenizer_names) or
                            tokenizer_binding.get('schema') !=
                            'crosspascal.hf-pascal-tokenizer-binding.v1' or
                            tokenizer_binding.get('status') !=
                            'POST_WIN32_HELD_EXACT_TOKENIZER_BYTES_NOT_MODEL_UAT' or
                            tokenizer_binding.get('model_key') != key or
                            tokenizer_binding.get('source_tokenizer_sha256') !=
                            selected_tokenizer['sha256'] or
                            tokenizer_binding.get('verification_status') != 'NOT_CHECKED' or
                            tokenizer_binding.get('pascal_inference') != 'NOT_RUN' or
                            tokenizer_binding.get('release_grade_immutable') is not False or
                            selected_source.get('tree_sha256') != args.hf_selected_tree_sha256 or
                            selected_source.get('model_key') != key or
                            selected_source.get('verification_status') != 'NOT_CHECKED' or
                            locked.get('tree_sha256') != args.hf_selected_tree_sha256 or
                            locked.get('status') !=
                            'WIN32_HF_SELECTED_PRIVATE_STAGE_CANDIDATE_NOT_RELEASE_GRADE' or
                            locked.get('verification_status') != 'NOT_CHECKED' or
                            locked.get('release_grade_immutable') is not False):
                        raise ValueError('native HF builder did not return exact '
                                         'Preview-bound Win32 held-source receipt')
                for option, expected in (('target', target), ('arch', arch),
                                         ('dtype', dtype), ('key', key)):
                    if option in built and built[option] != expected:
                        raise ValueError('native family returned %s=%r, expected %r'
                                         % (option, built[option], expected))
                write_family_receipt(
                    plan, output, model_key=key,
                    provenance=built['provenance'], license=license)
            if native_family_atomic:
                
                
                after = bundle_publication.inspect_selected_native_consumer_kit(
                    args.consumer_kit_source, model_key=key, device=target,
                    expected_tree_sha256=args.consumer_kit_tree_sha256)
                if after != consumer_inventory:
                    raise RuntimeError('selected native consumer bytes changed')
                inventory, receipt = bundle_publication.commit_native_family_stage(
                    output, args.into, plan, model_key=key,
                    approved_consumer_inventory=consumer_inventory)
                if (receipt.destination != bundle_atomic._local_absolute(args.into)
                        or receipt.tree_sha256 != inventory.tree_sha256):
                    raise RuntimeError('native family atomic commit receipt mismatch')
                print('FAMILY ATOMIC NAMESPACE COMMITTED, NOT_CHECKED: '
                      'no independent model oracle; not immutable or power-loss durable')
                return 0
        except Exception as trouble:
            print('FAILED native family build/replay: %s' % trouble, file=sys.stderr)
            if native_family_atomic and private_stage is not None:
                
                
                
                try:
                    retained = bundle_atomic._guard_stat(private_stage,
                                                         directory=True)
                    if ((retained.st_dev, retained.st_ino) ==
                            private_stage_identity):
                        print('Retained private native family stage for owner '
                              'inspection: %s (not verified; no cleanup)' %
                              private_stage, file=sys.stderr)
                    else:
                        print('Private stage candidate identity changed; '
                              'ownership unproven, no cleanup', file=sys.stderr)
                except (OSError, ValueError):
                    print('Private stage candidate is absent or unproven; '
                          'no cleanup performed', file=sys.stderr)
            return 1
        print('FAMILY BUILT, NOT_CHECKED: no independent model oracle performed')
        return 0

    if dtype != 'float32':
        print('FAILED: --dtype float16 belongs to the native LLM family builder; '
              'RAW ONNX/TFLite compiler supports float32 only', file=sys.stderr)
        return 1

    selected_paddle_export_provenance = None
    if adapter is not None:
        if getattr(adapter, 'kind', None) != 'export':
            print('FAILED: unsupported registered source adapter type',
                  file=sys.stderr)
            return 1
        try:
            if paddle_approved is not None:
                if adapter.name != 'paddleocr':
                    raise ValueError('approved Paddle route changed since preflight')
                again = paddle_closure.audit_selected_paddle_source(
                    path, role=selected_role,
                    dictionary=selected_dictionary or None,
                    expected_tree_sha256=selected_tree_sha)
                if again != paddle_approved:
                    raise ValueError('approved Paddle source changed before export')
                exported = adapter.export_held(
                    path, args.into,
                    stage_parent=os.path.dirname(os.path.abspath(args.into)),
                    role=selected_role,
                    dictionary=selected_dictionary or None,
                    approved_tree_sha256=selected_tree_sha)
                provenance = exported.get('provenance') or {}
                held = provenance.get('compiler_held_source') or {}
                if (provenance.get('selected_source_closure') != paddle_approved or
                        not isinstance(held, dict) or
                        held.get('source_tree_sha256') != paddle_approved['tree_sha256'] or
                        held.get('status') !=
                        'WIN32_PADDLE_SELECTED_HELD_CANDIDATE_NOT_RELEASE_GRADE' or
                        held.get('verification_status') != 'NOT_CHECKED' or
                        held.get('release_grade_immutable') is not False or
                        held.get('retained_after_context') is not True):
                    raise ValueError('Paddle CLI private held-source receipt is invalid')
                after = paddle_closure.audit_selected_paddle_source(
                    source, role=selected_role,
                    dictionary=selected_dictionary or None,
                    expected_tree_sha256=selected_tree_sha)
                if after != paddle_approved:
                    raise ValueError('Paddle source changed during WSL export')
            elif adapter.name == 'paddleocr':
                raise ValueError('Paddle export requires selected held-source approval')
            else:
                exported = adapter.export(path, args.into)
            path = exported['onnx']
            if paddle_approved is not None:
                
                
                selected_paddle_export_provenance = exported['provenance']
            else:
                with open(os.path.join(args.into, 'import-provenance.json'), 'x',
                          encoding='utf-8', newline='\n') as handle:
                    json.dump(exported['provenance'], handle,
                              ensure_ascii=False, indent=2)
                    handle.write('\n')
        except (OSError, ValueError, RuntimeError, KeyError) as trouble:
            print('FAILED source family export: %s' % trouble, file=sys.stderr)
            return 1

        
        
        
        if paddle_approved is None:
            try:
                if _unheld_onnx_external_data(path):
                    print('FAILED: exported ONNX external_data requires a held '
                          'multi-file source closure; CLI export is not yet wired',
                          file=sys.stderr)
                    return 1
            except (ImportError, OSError, ValueError) as trouble:
                print('FAILED: cannot prove exported ONNX self-contained: %s'
                      % trouble, file=sys.stderr)
                return 1

    run_kwargs = {}
    if paddle_approved is not None:
        run_kwargs['approved_paddle_source'] = paddle_approved
    def run_pipeline(compiler_input):
        return pipeline.run(compiler_input, args.into, target=target,
                            shape=shape, module_name=name,
                            tune_work_dir=tune_work_dir,
                            trials_per_task=trials,
                            contract=contract, req_sm=req_sm, source=source,
                            license=license, req_os=req_os, req_arch=req_arch,
                            req_driver_api=req_driver_api,
                            cpu_codegen=cpu_codegen,
                            approved_build_plan=saved_raw_plan, **run_kwargs)

    if paddle_approved is not None:
        
        
        
        
        
        try:
            from . import exported_onnx_hold, paddle_consumer_dictionary
            provenance = selected_paddle_export_provenance
            if (type(provenance) is not dict or
                    provenance.get('selected_source_closure') != paddle_approved):
                raise ValueError('selected Paddle original exporter receipt missing')
            source_exported_onnx = os.path.abspath(path)
            
            
            
            exported_onnx_hold.await_post_wsl_ram_admission(
                os.path.getsize(source_exported_onnx))
            with exported_onnx_hold.hold_private_exported_onnx(
                    source_exported_onnx,
                    stage_parent=os.path.dirname(os.path.abspath(args.into)),
                    expected_sha256=provenance['onnx_sha256'],
                    expected_output_directory=os.path.abspath(args.into)) as ticket, \
                    paddle_consumer_dictionary.hold_selected_rec_dictionary(
                        paddle_approved,
                        os.path.abspath(args.into)) as dictionary_receipt:
                if (os.path.normcase(os.path.abspath(
                        ticket.source_exported_onnx)) !=
                        os.path.normcase(source_exported_onnx) or
                        ticket.original_export_sha256 != provenance['onnx_sha256'] or
                        ticket.staged_model_sha256 != provenance['onnx_sha256'] or
                        ticket.status !=
                        'WIN32_EXPORTED_ONNX_COMPILER_HELD_NOT_RELEASE_GRADE' or
                        ticket.verification_status != 'NOT_CHECKED' or
                        ticket.release_grade_immutable is not False or
                        ticket.retained_after_context is not True):
                    raise RuntimeError('Paddle CLI exported-ONNX held stage '
                                       'does not match selected exporter bytes')
                report = run_pipeline(ticket.compiler_input)
                after = paddle_closure.audit_selected_paddle_source(
                    source, role=selected_role,
                    dictionary=selected_dictionary or None,
                    expected_tree_sha256=selected_tree_sha)
                if after != paddle_approved:
                    raise RuntimeError('Paddle original selected graph, weights '
                                       'or dictionary changed during compiler')
            
            
            provenance['compiler_window_held_exported_onnx'] = {
                'status': 'POST_LOCK_AUDIT_PASSED_NOT_RELEASE_GRADE',
                'exporter_sha256': ticket.original_export_sha256,
                'staged_model_sha256': ticket.staged_model_sha256,
                'bytes': ticket.bytes,
                
                
                
                
                'held_during_pipeline_run': True,
                'pipeline_sound': report.sound is True,
                'verification_status': 'NOT_CHECKED',
                'release_grade_immutable': False,
                'retained_after_context': True,
            }
            if dictionary_receipt is not None:
                
                
                
                
                if (dictionary_receipt['status'] !=
                        'POST_WIN32_HELD_SELECTED_REC_DICTIONARY_NOT_MODEL_UAT' or
                        dictionary_receipt['source_tree_sha256'] !=
                        paddle_approved['tree_sha256'] or
                        dictionary_receipt['sha256'] !=
                        paddle_approved['dictionary']['sha256'] or
                        dictionary_receipt['verification_status'] != 'NOT_CHECKED' or
                        dictionary_receipt['native_pascal_ctc'] != 'NOT_RUN'):
                    raise RuntimeError('selected Paddle Rec native consumer '
                                       'dictionary compiler-window receipt invalid')
                provenance['compiler_window_held_rec_dictionary'] = dictionary_receipt
            with open(os.path.join(args.into, 'import-provenance.json'), 'x',
                      encoding='utf-8', newline='\n') as handle:
                json.dump(provenance, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
        except Exception as trouble:
            
            
            print('FAILED selected Paddle exported-ONNX compiler custody: %s'
                  % trouble, file=sys.stderr)
            return 1
    else:
        report = run_pipeline(path)
    text = report.as_text()
    print(text)

    report_path = args.report or os.path.join(args.into, 'report.txt')
    with open(report_path, 'w', encoding='utf-8', newline=chr(10)) as handle:
        handle.write(text + chr(10))

    return 0 if report.sound else 1


if __name__ == '__main__':
    sys.exit(main())
