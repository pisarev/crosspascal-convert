"""Small contract tests; no model, TVM/ORT import, server, or compiler run."""

import copy
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
ATTEMPTED_NATIVE_IMPORTS = []


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tvm', 'tvm_ffi', 'onnxruntime', 'torch', 'tensorflow'):
            ATTEMPTED_NATIVE_IMPORTS.append(fullname)
            raise RuntimeError('Model runtime import is forbidden in these unit tests: ' + fullname)
        return None


sys.meta_path.insert(0, NoModelImports())
import master
import import_adapters
from pascalvision_convert import app_paths, build_plan, consumer_acceptance
from pascalvision_convert import cuda_preflight, llm_build_profile, llm_family_profile
from pascalvision_convert import model_sources, recipe, wizard_session, bundle_publication
from pascalvision_convert.llm_helpers import weights

HOST = {'os': 'windows', 'arch': 'x86_64', 'cuda_arch': 'sm_75'}
PROFILE = llm_build_profile.QWEN_CUDA_PROFILE


class WizardContractTests(unittest.TestCase):
    def setUp(self):
        workspace = os.environ.get('CROSSPASCAL_TEST_WORKSPACE')
        self.temporary = tempfile.TemporaryDirectory(prefix='wizard-case-', dir=workspace)
        self.root = Path(self.temporary.name)
        self.environment = mock.patch.dict(os.environ, {'CROSSPASCAL_CONVERT_HOME': str(self.root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def plan(self, **changes):
        options = dict(target='cuda', req_sm='sm_75', route='family_full_build',
                       source=str(self.root), dtype='float32',
                       llm_options=llm_build_profile.select(PROFILE))
        options.update(changes)
        return build_plan.resolve(HOST, **options)

    def test_real_wizard_import_and_packaged_helpers(self):
        self.assertEqual(ATTEMPTED_NATIVE_IMPORTS, [])
        self.assertTrue(Path(master.PAGE).is_file())
        self.assertEqual(Path(import_adapters.verify_installed_llm_helpers()).name, 'build_llm.py')
        self.assertTrue((ROOT / 'pyproject.toml').is_file())

    def test_profile_roundtrip_and_native_context_binding(self):
        plan = self.plan()
        self.assertEqual(build_plan.validate_recorded(plan), plan)
        args = dict(source=str(self.root), target='cuda', req_sm='sm_75', dtype='float32',
                    host_os='Windows', host_arch='AMD64', model_key='qwen2.5-0.5b',
                    context=64, chunk=64)
        build_plan.validate_native_family_execution(plan, **args)
        for key, value in (('context', 2048), ('chunk', 512), ('model_key', 'gemma-2-2b')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_plan.validate_native_family_execution(plan, **dict(args, **{key: value}))

    def test_profile_refuses_precision_target_and_rehashed_tampering(self):
        for changes in ({'dtype': 'float16'}, {'target': 'cpu', 'req_sm': None},
                        {'route': 'raw'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)
        plan = self.plan()
        plan['options']['llm']['context_window_size'] = 128
        plan['plan_sha256'] = build_plan._checksum(plan)
        with self.assertRaises(ValueError):
            build_plan.validate_recorded(plan)
        chosen = llm_build_profile.select(PROFILE)
        chosen['max_batch_size'] = True
        with self.assertRaises(ValueError):
            self.plan(llm_options=chosen)

    def test_profile_respects_local_and_explicit_architectures(self):
        local = self.plan(req_sm=None)
        other = self.plan(req_sm='sm_89')
        self.assertEqual(local['target_profile']['deployment']['sm'], HOST['cuda_arch'])
        self.assertEqual(other['target_profile']['deployment']['sm'], 'sm_89')
        self.assertEqual(local['options']['llm'], other['options']['llm'])
        self.assertEqual(build_plan.validate_recorded(other), other)

    def test_legacy_plan_has_no_implicit_new_profile(self):
        plan = build_plan.resolve(HOST, source='fixture', route='family_full_build')
        self.assertEqual(set(plan['options']), {'shape', 'dtype', 'tune', 'trials_per_task', 'contract'})
        build_plan.validate_recorded(plan)
        with self.assertRaises(ValueError):
            build_plan.validate_native_family_execution(
                plan, source='fixture', target='cpu', req_sm=None, dtype='float32',
                host_os='Windows', host_arch='AMD64', context=64, chunk=64)

    def test_interpreter_pin_and_pythonw_pair(self):
        executable = self.root / 'python.exe'
        executable.write_bytes(b'fixture executable, never run')
        windowed = self.root / 'pythonw.exe'
        windowed.write_bytes(b'fixture windowed executable, never run')
        self.assertEqual(llm_build_profile.console_python(windowed), str(executable))
        record = llm_build_profile.python_snapshot(windowed)
        self.assertEqual(llm_build_profile.check_python(record), str(executable))
        executable.write_bytes(b'changed fixture executable')
        with self.assertRaises(ValueError):
            llm_build_profile.check_python(record)
        executable.unlink()
        with self.assertRaises(ValueError):
            llm_build_profile.console_python(windowed)

    def test_gui_options_bind_real_config_profile_and_python(self):
        config = copy.deepcopy(llm_family_profile._PROFILES['qwen2.5-0.5b'])
        (self.root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        executable = self.root / 'compiler.exe'
        executable.write_bytes(b'not executed')
        selection = master._llm_plan_arguments({
            'full_build': True, 'path': str(self.root),
            'llm_profile': PROFILE, 'llm_python': str(executable)})
        self.assertEqual(selection['llm_options']['context_window_size'], 64)
        self.assertEqual(selection['compiler_python']['path'], str(executable))
        config['num_hidden_layers'] = 12
        (self.root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        with self.assertRaises(ValueError):
            master._llm_plan_arguments({'full_build': True, 'path': str(self.root), 'llm_profile': PROFILE})
        with self.assertRaises(ValueError):
            master._llm_plan_arguments({'full_build': False, 'llm_profile': PROFILE})

    def test_saved_recipe_refuses_profile_override_before_io(self):
        with mock.patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as stopped:
            recipe.main(['--recipe', 'not-opened.json', '--llm-profile', PROFILE])
        self.assertEqual(stopped.exception.code, 2)

    def family_preview_request(self):
        source = self.root / 'model'
        source.mkdir()
        config = copy.deepcopy(llm_family_profile._PROFILES['qwen2.5-0.5b'])
        (source / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        executable = self.root / 'compiler.exe'
        executable.write_bytes(b'fixture interpreter, never executed')
        return {'full_build': True, 'native_family_atomic': True,
                'path': str(source), 'target': 'cuda', 'arch': 'sm_75',
                'target_os': 'windows', 'target_cpu_arch': 'x86_64',
                'target_driver_api': '', 'dtype': 'float32',
                'llm_profile': PROFILE, 'llm_python': str(executable),
                'output_directory': str(self.root / 'converted')}

    def test_family_preview_empty_driver_matches_omitted_option(self):
        data = self.family_preview_request()
        with (mock.patch.object(master.targets, 'host', return_value=HOST),
              mock.patch('subprocess.run', side_effect=AssertionError('No child allowed'))):
            empty = master._preview_build_plan(data)
            del data['target_driver_api']
            omitted = master._preview_build_plan(data)
        self.assertNotIn('trouble', empty)
        self.assertNotIn('trouble', omitted)
        self.assertEqual(empty['build_plan'], omitted['build_plan'])
        self.assertIsNone(empty['target'].get('driver_api'))
        self.assertTrue(empty['ready_for_build'])
        self.assertEqual(build_plan.validate_recorded(empty['build_plan']), empty['build_plan'])
        self.assertEqual(ATTEMPTED_NATIVE_IMPORTS, [])

    def test_family_preview_refuses_output_inside_original_model(self):
        data = self.family_preview_request()
        source = Path(data['path'])
        destination = source / 'converted'
        data['output_directory'] = str(destination)
        with (mock.patch.object(master.targets, 'host', return_value=HOST),
              mock.patch('subprocess.run', side_effect=AssertionError('No child allowed'))):
            preview = master._preview_build_plan(data)
        self.assertIn('outside the original LLM source directory', preview.get('trouble', ''))
        self.assertFalse(destination.exists())

    def test_family_preview_retains_explicit_driver_override_refusal(self):
        data = self.family_preview_request()
        with (mock.patch.object(master.targets, 'host', return_value=HOST),
              mock.patch('subprocess.run', side_effect=AssertionError('No child allowed'))):
            for value in ('13020', 13020, 0, False, ' '):
                with self.subTest(value=value):
                    data['target_driver_api'] = value
                    result = master._preview_build_plan(data)
                    self.assertIn('does not implement a selected CUDA driver API',
                                  result.get('trouble', ''))
        self.assertEqual(ATTEMPTED_NATIVE_IMPORTS, [])

    def test_required_checks_never_pass_when_missing(self):
        result = consumer_acceptance.aggregate(PROFILE, [])
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertFalse(result['required_checks_pass'])
        self.assertEqual(set(result['missing_checks']), set(consumer_acceptance.REQUIRED))
        result = consumer_acceptance.after_native_build(self.plan())
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertEqual(next(row for row in result['checks'] if row['id'] == 'consumer_runtime')['status'], 'NOT_RUN')

    def test_diagnostics_are_retained_without_overriding_required_failure(self):
        checks = [{'id': key, 'role': 'required', 'status': 'PASS'}
                  for key in consumer_acceptance.REQUIRED]
        checks.append({'id': 'raw_diagnostic', 'role': 'diagnostic', 'status': 'FAIL'})
        result = consumer_acceptance.aggregate(PROFILE, checks)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['diagnostic_failures'], ['raw_diagnostic'])
        checks[0]['status'] = 'FAIL'
        self.assertEqual(consumer_acceptance.aggregate(PROFILE, checks)['status'], 'FAIL')
        checks[1]['status'] = 'ERROR'
        self.assertEqual(consumer_acceptance.aggregate(PROFILE, checks)['status'], 'ERROR')
        checks[0]['role'] = 'diagnostic'
        with self.assertRaises(ValueError):
            consumer_acceptance.aggregate(PROFILE, checks)

    def test_official_source_identity_uses_all_measured_pins(self):
        catalog = model_sources.catalog_model('qwen2.5-0.5b')
        scan = {'entries': [{key: row[key] for key in ('path', 'bytes', 'sha256')}
                            for row in catalog['files']],
                'tree_sha256': catalog['tree_sha256'], 'model_key': 'qwen2.5-0.5b'}
        result = model_sources.match_source(scan, 'qwen2.5-0.5b')
        self.assertEqual(result['revision'], '7ae557604adf67be50417f59c2c2f167def9a775')
        scan['entries'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            model_sources.match_source(scan, 'qwen2.5-0.5b')

    def tiny_catalog(self):
        revision = 'a' * 40
        payloads = {'config.json': b'{"fixture":true}', 'LICENSE': b'test license'}
        rows = [{'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                 'url': 'https://huggingface.co/Test/Tiny/resolve/' + revision + '/' + name}
                for name, raw in payloads.items()]
        return {'repository': 'Test/Tiny', 'revision': revision,
                'files': rows[:1], 'license_file': rows[1]}, payloads

    def test_download_exact_bytes_and_verified_cache_without_network(self):
        catalog, payloads = self.tiny_catalog()
        requests = []
        def opener(request, timeout):
            requests.append(request.full_url)
            response = io.BytesIO(payloads[request.full_url.rsplit('/', 1)[-1]])
            response.geturl = lambda: request.full_url
            return response
        with mock.patch.object(model_sources, 'catalog_model', return_value=catalog):
            first = model_sources.download('tiny', opener=opener)
            second = model_sources.download('tiny', opener=lambda *a, **k: self.fail('Cached bytes must not redownload'))
        self.assertEqual(len(requests), 2)
        self.assertEqual(first, second)
        self.assertEqual(first['inference_status'], 'NOT_RUN')

    def test_download_refuses_truncation_and_preserves_partial_files(self):
        catalog, _ = self.tiny_catalog()
        def opener(request, timeout):
            response = io.BytesIO(b'x')
            response.geturl = lambda: request.full_url
            return response
        with mock.patch.object(model_sources, 'catalog_model', return_value=catalog):
            with self.assertRaisesRegex(RuntimeError, 'ended before'):
                model_sources.download('tiny', opener=opener)
            with self.assertRaises(ValueError):
                model_sources.download('tiny', opener=opener)
        self.assertTrue(any(app_paths.data_path('models').iterdir()))

    def test_consumer_kit_preserves_licenses_and_refuses_extra_files(self):
        kit = self.root / 'kit'
        kit.mkdir()
        names = {'tvm_runtime.dll', 'tvm_runtime_extra.dll', 'tvm_ffi.dll', 'tvm_runtime_cuda.dll',
                 'tokenizer-tokens.bin', 'tokenizer-tokens.index', 'tokenizer-merges.bin',
                 'tokenizer-specials.txt', 'tokenizer.index', 'LICENSE-tvm-test.txt',
                 'NOTICE-tvm-test.txt', 'LICENSE-Qwen2.5-0.5B-Instruct.txt', 'MODEL-SOURCE.json'}
        for name in names:
            (kit / name).write_bytes(b'fixture data, never loaded')
        inventory = bundle_publication.inspect_selected_native_consumer_kit(
            str(kit), model_key='qwen2.5-0.5b', device='cuda')
        self.assertEqual({item.path for item in inventory.members}, names)
        (kit / 'unexpected.dll').write_bytes(b'extra')
        with self.assertRaises(ValueError):
            bundle_publication.inspect_selected_native_consumer_kit(
                str(kit), model_key='qwen2.5-0.5b', device='cuda')

    def test_weight_serialization_hash_matches_written_order(self):
        raw = struct.pack('<fff', 1.5, -3.0, 7.0)
        header = json.dumps({'first': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]},
                             'second': {'dtype': 'F32', 'shape': [1], 'data_offsets': [8, 12]}}).encode()
        source = self.root / 'model.safetensors'
        source.write_bytes(struct.pack('<Q', len(header)) + header + raw)
        output = self.root / 'out'
        def plan(reader, num_layers):
            return {'first': lambda: reader.read('first'), 'second': lambda: reader.read('second')}
        with mock.patch('sys.stdout', io.StringIO()):
            total, count = weights.convert(str(source), str(output), ['second', 'first'],
                                           {'second': (1,), 'first': (2,)}, 0, plan_fn=plan)
        expected = struct.pack('<fff', 7.0, 1.5, -3.0)
        self.assertEqual((total, count), (12, 2))
        self.assertEqual((output / 'params.bin').read_bytes(), expected)
        metadata = json.loads((output / 'params.json').read_text())
        self.assertEqual(metadata['sha256'], hashlib.sha256(expected).hexdigest())

    def test_cuda_probe_uses_console_child_without_loading_cuda(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout=json.dumps(
                {'version_rc': 0, 'runtime_api': 13020, 'devices_rc': 0, 'devices': 1}))
        with (mock.patch.object(cuda_preflight, 'driver_api_version', return_value=(13020, 'fixture')),
             mock.patch.object(cuda_preflight, 'installed_toolkit_version', return_value=(13, 2)),
             mock.patch.object(cuda_preflight, 'runtime_in_toolkit', return_value='fixture.dll'),
             mock.patch.object(cuda_preflight, 'console_python', return_value='selected-python.exe')):
            result = cuda_preflight.check_cuda_toolkit('fixture-nvcc', None, run=run)
        self.assertEqual(calls[0][0], 'selected-python.exe')
        self.assertEqual(result['status'], 'compatible')

    def test_quit_requires_session_and_refuses_running_job(self):
        responder = object.__new__(master.Responder)
        responder.path = '/api/quit'
        responder.server = SimpleNamespace(server_port=8765, shutdown=mock.Mock())
        responder.headers = {'Host': '127.0.0.1:8765', 'Origin': 'http://127.0.0.1:8765',
                             'X-CrossPascal-Session': 'wrong'}
        responder.send_error = mock.Mock()
        sent = []
        responder._send = lambda text: sent.append(json.loads(text))
        with (mock.patch.dict(master.JOBS, {'1': {'finished': False}}, clear=True),
             mock.patch.dict(master.MODEL_DOWNLOADS, {}, clear=True),
             mock.patch.object(master, 'ACTIVE_REQUESTS', 0),
             mock.patch.object(master, 'SHUTTING_DOWN', False),
             mock.patch.object(master.threading, 'Thread') as thread):
            responder.do_POST()
            responder.send_error.assert_called_once()
            self.assertEqual(responder.send_error.call_args.args[0], 403)
            responder.headers['X-CrossPascal-Session'] = master.SESSION_TOKEN
            responder.do_POST()
            self.assertIn('trouble', sent[-1])
            thread.assert_not_called()
            master.JOBS['1']['finished'] = True
            responder.do_POST()
            self.assertEqual(sent[-1], {'status': 'CLOSING'})
            thread.return_value.start.assert_called_once()

    def test_session_record_retains_identity_and_clears_only_own_instance(self):
        app_paths.prepare()
        server = SimpleNamespace(server_port=8765, instance_id='a' * 32)
        recorded = wizard_session.publish(server)
        self.assertEqual(wizard_session.read_session(), recorded)
        wizard_session.clear('b' * 32)
        self.assertTrue(wizard_session.session_path().exists())
        wizard_session.clear(server.instance_id)
        self.assertFalse(wizard_session.session_path().exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
