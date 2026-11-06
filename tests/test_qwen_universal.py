"""Cheap synthetic local Qwen2-compatible contract tests; no model/TVM/CUDA."""

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

from pascalvision_convert import (build_plan, hf_closure, llm_build_profile,
                                  llm_bundle_contract, llm_family_profile,
                                  qwen_universal)
import import_adapters
import master


def compatible_raw(*, untied=False):
    return {
        'architectures': ['Qwen2ForCausalLM'], 'model_type': 'qwen2',
        'hidden_size': 1024, 'intermediate_size': 4096,
        'num_hidden_layers': 3, 'num_attention_heads': 16,
        'num_key_value_heads': 4, 'vocab_size': 151936,
        'rope_theta': 500000, 'rms_norm_eps': 1e-6,
        'bos_token_id': 151643, 'eos_token_id': 151645,
        'tie_word_embeddings': not untied, 'hidden_act': 'silu',
        'attention_bias': True, 'mlp_bias': False, 'attention_dropout': 0.0,
        'max_position_embeddings': 32768,
    }


def save_safetensors(path, contents):
    """Only a few bytes; all large shapes are tested with a mocked table."""
    spans, data = {}, b''
    for name, (dtype, shape, values) in contents.items():
        begin = len(data)
        data += values
        spans[name] = {'dtype': dtype, 'shape': shape,
                       'data_offsets': [begin, len(data)]}
    header = json.dumps(spans, separators=(',', ':')).encode('utf-8')
    path.write_bytes(struct.pack('<Q', len(header)) + header + data)


class LocalQwenCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen-universal-')
        self.root = Path(self.temp.name)
        self.raw = compatible_raw()
        self.cfg = qwen_universal.validate_config(self.raw)

    def tearDown(self):
        self.temp.cleanup()

    def test_third_genuinely_distinct_topology_has_no_catalog_requirement(self):
        cfg = self.cfg
        self.assertEqual((cfg['hidden_size'], cfg['num_hidden_layers'],
                          cfg['num_attention_heads'], cfg['num_key_value_heads']),
                         (1024, 3, 16, 4))
        self.assertEqual(
            llm_family_profile.identify_model_key(self.raw), qwen_universal.KEY)
        self.assertNotIn(qwen_universal.KEY, llm_family_profile._PROFILES)
        spec = qwen_universal.descriptor(cfg, '1' * 64)
        self.assertEqual(spec['schema'], qwen_universal.SCHEMA)
        self.assertEqual(spec['source_kind'], 'local-sha256')
        self.assertEqual(spec['tensor_count'], 3 * 7 + 2)
        self.assertEqual(qwen_universal.packed_layout(cfg)[-1][0],
                         'model.norm.weight')
        for key in ('qwen2.5-0.5b', 'qwen2.5-1.5b'):
            self.assertEqual(llm_family_profile.identify_model_key(
                llm_family_profile._PROFILES[key]), key)
            self.assertEqual(llm_family_profile.identify_model_key(
                {**llm_family_profile._PROFILES[key], 'rms_norm_eps': 1e-6},
                generic=True),
                qwen_universal.KEY)

    def test_tied_and_untied_packed_head_and_hf_tensor_shapes(self):
        generic = self.cfg
        self.assertNotIn('lm_head.weight', qwen_universal.hf_layout(generic))
        untied = qwen_universal.validate_config(compatible_raw(untied=True))
        self.assertEqual(qwen_universal.packed_layout(untied)[-1],
                         ('lm_head.weight', [151936, 1024]))
        self.assertEqual(qwen_universal.hf_layout(untied)['lm_head.weight'],
                         [151936, 1024])
        self.assertEqual(qwen_universal.descriptor(
            untied, '2' * 64)['tensor_count'], 3 * 7 + 3)

    def test_reject_unsupported_operators_modalities_and_bad_dimensions(self):
        invalid = [
            ('hidden_act', 'gelu'), ('quantization_config', {'quant_method': 'gptq'}),
            ('rope_scaling', {'type': 'linear'}), ('vision_config', {}),
            ('trust_remote_code', True), ('attention_dropout', 0.1),
            ('num_experts', 8), ('hidden_size', 1000),
            ('num_key_value_heads', 3), ('head_dim', 128),
            ('bos_token_id', 0), ('vocab_size', 151000),
            ('tie_word_embeddings', 'true'),
        ]
        for field, value in invalid:
            with self.subTest(field=field), self.assertRaises(ValueError):
                qwen_universal.validate_config({**self.raw, field: value})
        with self.assertRaises(ValueError):
            qwen_universal.validate_config({
                **self.raw, 'architectures': ['Qwen2MoeForCausalLM']})

    def test_generic_buildplan_target_and_resource_refusal(self):
        profile = llm_build_profile.select(qwen_universal.PROFILE,
                                           model_key=qwen_universal.KEY)
        host = {'os': 'windows', 'arch': 'x86_64', 'cuda_arch': 'sm_75'}
        args = {'target': 'cuda', 'route': 'family_full_build',
                'req_sm': 'sm_89', 'source': str(self.root), 'dtype': 'float32',
                'llm_options': profile}
        plan = build_plan.resolve(host, **args)
        self.assertEqual(build_plan.validate_recorded(plan), plan)
        self.assertEqual(plan['target_profile']['deployment']['sm'], 'sm_89')
        for change in ({'dtype': 'float16'}, {'target': 'cpu', 'req_sm': None},
                       {'route': 'raw'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_plan.resolve(host, **{**args, **change})
        need = qwen_universal.requirements(self.cfg)
        self.assertGreater(need['packed_parameter_bytes'], 0)
        for key, value in (('disk_free', need['min_free_output_disk_bytes'] - 1),
                           ('ram_free', need['min_free_host_ram_bytes'] - 1),
                           ('gpu_free', need['min_free_gpu_bytes'] - 1)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,
                                                                'insufficient'):
                qwen_universal.admit(self.cfg, **{key: value})
        self.assertEqual(qwen_universal.admit(
            self.cfg, disk_free=need['min_free_output_disk_bytes'],
            ram_free=need['min_free_host_ram_bytes'],
            gpu_free=need['min_free_gpu_bytes']), need)

    def test_recognise_generic_local_config_and_explicit_curated_override(self):
        (self.root / 'config.json').write_text(
            json.dumps(self.raw), encoding='utf-8')
        adapter = import_adapters.LLMImportAdapter()
        self.assertEqual(adapter._key_for(str(self.root)), qwen_universal.KEY)
        self.assertEqual(master._selected_llm_key({
            'path': str(self.root), 'llm_profile': qwen_universal.PROFILE}),
            qwen_universal.KEY)
        self.assertEqual(adapter._key_for(str(self.root), generic=True),
                         qwen_universal.KEY)
        curated = {**llm_family_profile._PROFILES['qwen2.5-0.5b'],
                   'rms_norm_eps': 1e-6}
        (self.root / 'config.json').write_text(
            json.dumps(curated), encoding='utf-8')
        self.assertEqual(adapter._key_for(str(self.root)), 'qwen2.5-0.5b')
        self.assertEqual(adapter._key_for(str(self.root), generic=True),
                         qwen_universal.KEY)

    def test_generic_build_reproduces_the_preview_before_reserving_output(self):
        (self.root / 'config.json').write_text(json.dumps(self.raw), encoding='utf-8')
        data = {'path': str(self.root), 'llm_profile': qwen_universal.PROFILE}
        details = {'model_key': qwen_universal.KEY}
        with mock.patch.multiple(
            master,
            _source_fingerprint=lambda path: {'extent': 'full small-file SHA-256'},
            _preview_build_plan=lambda options: {
                'ready_for_build': True, 'warnings': [],
                'output_directory': str(self.root / 'bundle'),
            },
            _selected_family_consumer=lambda options, report: None,
            _selected_hf_preview=lambda options: details,
            _hf_preview_display=lambda selected: selected,
            _selected_paddle_preview=lambda options: None,
            _frozen_raw_preview=lambda options, reserved_output: None,
            _output_parent_identity=mock.Mock(side_effect=[('original',), ('changed',)]),
        ), mock.patch.object(master.onnx_source_pack, 'selected', return_value=None), \
            mock.patch.object(qwen_universal, 'admit', return_value=qwen_universal.requirements(self.cfg)):
            preview = master._issue_preview(data)
            self.assertEqual(preview['qwen_compatibility']['profile_id'], qwen_universal.PROFILE)
            refusal = master._build_from_preview({**data, 'preview_token': preview['preview_token']})
        self.assertIn('Preview output parent changed', refusal['trouble'])

    def test_small_actual_single_and_sharded_safetensors_headers(self):
        # Exercise the real local file/header parser without 1 GiB allocations.
        expected = {'embed': [2, 2], 'norm': [2]}
        first = {'embed': ('F32', [2, 2], b'\0' * 16)}
        second = {'norm': ('F32', [2], b'\0' * 8)}
        path = self.root / 'model.safetensors'
        save_safetensors(path, {**first, **second})
        with mock.patch.object(qwen_universal, 'hf_layout', return_value=expected):
            found = qwen_universal.inspect_local_headers(self.root, self.cfg)
            self.assertEqual((found['layout'], found['source_tensors']),
                             ('single', 2))
            save_safetensors(path, {'embed': ('F32', [2, 3], b'\0' * 24),
                                    **second})
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                qwen_universal.inspect_local_headers(self.root, self.cfg)
        path.unlink()
        save_safetensors(self.root / 's1.safetensors', first)
        save_safetensors(self.root / 's2.safetensors', second)
        index = {'metadata': {'total_size': 24},
                 'weight_map': {'embed': 's1.safetensors',
                                'norm': 's2.safetensors'}}
        index_path = self.root / 'model.safetensors.index.json'
        index_path.write_text(json.dumps(index), encoding='utf-8')
        with mock.patch.object(qwen_universal, 'hf_layout', return_value=expected):
            self.assertEqual(qwen_universal.inspect_local_headers(
                self.root, self.cfg)['layout'], 'sharded')
            index['weight_map']['norm'] = 's1.safetensors'
            index_path.write_text(json.dumps(index), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'closure differs'):
                qwen_universal.inspect_local_headers(self.root, self.cfg)

    def test_manifest_descriptor_and_generic_parameter_index_refuse_tamper(self):
        cfg = self.cfg
        spec = qwen_universal.descriptor(cfg, '3' * 64)
        layout = qwen_universal.packed_layout(cfg)
        expected_bytes = spec['packed_parameter_bytes']
        model = {'model_key': qwen_universal.KEY, 'source_kind': 'local-sha256',
                 'source_tree_sha256': '3' * 64, 'qwen_compatibility': spec,
                 'llm_profile_id': qwen_universal.PROFILE,
                 'param_order': [name for name, _ in layout],
                 'params_bytes': expected_bytes, 'params_sha256': '4' * 64,
                 **cfg}
        llm_bundle_contract._validate_qwen_metadata(model, qwen_universal.KEY)
        self.assertEqual(llm_bundle_contract._qwen_parameter_layout(
            qwen_universal.KEY, model), layout)
        for change in (
                ('hidden_size', 999), ('source_tree_sha256', '5' * 64),
                ('tie_word_embeddings', False)):
            broken = copy.deepcopy(model)
            broken[change[0]] = change[1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                llm_bundle_contract._validate_qwen_metadata(
                    broken, qwen_universal.KEY)

        offset = 0
        tensors = []
        lines = ['# count dtype target, then: name offset nbytes ndim dims...',
                 '%d float32 cuda' % len(layout)]
        for name, dims in layout:
            nbytes = 4
            for dim in dims:
                nbytes *= dim
            tensors.append({'name': name, 'dtype': 'float32',
                            'shape': dims, 'offset': offset, 'nbytes': nbytes})
            lines.append('%s %d %d %d %s' % (
                name, offset, nbytes, len(dims),
                ' '.join(str(value) for value in dims)))
            offset += nbytes
        self.assertEqual(offset, expected_bytes)
        params = {'dtype': 'float32', 'sha256': '4' * 64,
                  'total_bytes': offset, 'tensors': tensors}
        index = ('\n'.join(lines) + '\n').encode('utf-8')
        with (mock.patch.object(llm_bundle_contract, '_json', return_value=params),
              mock.patch.object(llm_bundle_contract, '_regular',
                                return_value=(self.root / 'params.bin', offset)),
              mock.patch.object(llm_bundle_contract, '_read_bounded',
                                return_value=index),
              mock.patch.object(llm_bundle_contract, '_hash_file',
                                return_value='4' * 64)):
            llm_bundle_contract._check_parameters(
                self.root, model, 'float32', 'cuda',
                model_key=qwen_universal.KEY)
            tensors[1]['shape'] = [1, 2]
            with self.assertRaisesRegex(ValueError, 'tensor shape'):
                llm_bundle_contract._check_parameters(
                    self.root, model, 'float32', 'cuda',
                    model_key=qwen_universal.KEY)
            tensors[1]['shape'] = layout[1][1]
            with mock.patch.object(llm_bundle_contract, '_read_bounded',
                                   return_value=index.replace(
                                       b'float32 cuda', b'float32 cpu')):
                with self.assertRaisesRegex(ValueError, 'header'):
                    llm_bundle_contract._check_parameters(
                        self.root, model, 'float32', 'cuda',
                        model_key=qwen_universal.KEY)

    def test_tokenizer_identity_refuses_incompatible_small_source(self):
        (self.root / 'tokenizer.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            qwen_universal.verify_tokenizer(self.root)
        self.assertEqual(len(qwen_universal.TOKENIZER_SHA256), 64)

    def test_pascal_consumer_declares_independent_generic_checks(self):
        pascal = ROOT.parent / 'crosspascal-tvm' / 'pascal' / 'QwenCuda.pas'
        text = pascal.read_text(encoding='utf-8')
        for fragment in ('QwenGenericSchema', 'QwenGenericProfileID',
                         'Result.HiddenSize := BoundedInteger',
                         'packed_parameter_bytes', 'min_free_gpu_bytes',
                         'min_free_host_ram_bytes', 'GenericMaxPackedBytes',
                         'source_tree_sha256'):
            self.assertIn(fragment, text)
        # Static source inspection is not a Pascal compile or CUDA acceptance.


if __name__ == '__main__':
    unittest.main()
