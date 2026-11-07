"""CPU checks for the GPU 2 queue; real SDEs and samplers, no checkpoints.

GPU2_TEST_INPUTS may point to copies of the real run_manifest.json files.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if importlib.util.find_spec('tensorflow') is None:  # Outside the Docker image.
    import torch._dynamo  # noqa: F401, its import scans module specs, which stubs lack.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())

import celeba_followup_eval as followup
import celeba_gpu2_eval as gpu2
import celeba_kappa_eval as kappa
import celeba_stage0_eval as stage0
import run_lib
import sampling
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.fox.celeba_kaffine import get_config as get_kaffine

HI = 10.575749165617202
LO = 9.115429865459795
A, K = 'A_c80_lo', 'k002_c8_hi_em'


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def write_manifests(target):
    source = os.environ.get('GPU2_TEST_INPUTS')
    for work, config in ((followup.WORKDIRS['A'], get_a()), (kappa.WORKDIRS['a0'], get_kaffine('a0')),
                         (kappa.WORKDIRS['k002'], get_kaffine('a0'))):
        if source:
            shutil.copy2(str(Path(source) / (work + '.json')), str(target / (work + '.json')))
            continue
        if work == kappa.WORKDIRS['k002']:
            config.model.fox_matern_length_scale = kappa.LENGTH_SCALES['k002']
        resolved = _canonicalize(config)
        resolved.pop('eval')
        resolved['device'] = 'cuda:0'
        (target / (work + '.json')).write_text(json.dumps(dict(config=resolved, model_manifest=build_model_manifest(config))))


def on_cpu(config):
    config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
    return config


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        write_manifests(inputs)
        for module, target, value in ((gpu2, 'ROOT', self.root), (gpu2, 'LOGS', self.root / 'logs'),
                                      (followup, 'TRAINING_INPUTS', inputs), (kappa, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, name):
        return on_cpu(gpu2.get_config(name))


class Plan(Base):
    def test_cells_and_their_order(self):
        self.assertEqual(list(gpu2.CELLS), [A, K])

    def test_a_is_a_c60_lo_except_the_checkpoint(self):
        mine = _canonicalize(gpu2.get_config(A))
        theirs = _canonicalize(followup.get_config('A_c60_lo'))
        self.assertEqual((mine['eval'].pop('begin_ckpt'), mine['eval'].pop('end_ckpt')), (80, 80))
        self.assertEqual((theirs['eval'].pop('begin_ckpt'), theirs['eval'].pop('end_ckpt')), (60, 60))
        self.assertEqual(mine, theirs)

    def test_a_in_explicit_values(self):
        config = gpu2.get_config(A)
        self.assertEqual((config.sampling.method, config.sampling.logsnr_min, config.sampling.logsnr_max),
                         ('ode_dpmpp_2m', -10.0, LO))
        self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.sampling_seed),
                         (64, 10000, 0))

    def test_a_rebuilds_the_trained_model(self):
        manifest = json.loads(followup.manifest_path('A').read_text())
        self.assertEqual(build_model_manifest(gpu2.get_config(A))['model_protocol_sha256'],
                         manifest['model_manifest']['model_protocol_sha256'])

    def test_kappa_rebuilds_the_trained_model(self):
        manifest = json.loads(kappa.manifest_path('k002').read_text())
        self.assertEqual(build_model_manifest(gpu2.get_config(K))['model_protocol_sha256'],
                         manifest['model_manifest']['model_protocol_sha256'])

    def test_kappa_sampling_equals_a0_em(self):
        mine, theirs = gpu2.get_config(K), followup.get_config('a0_c8_hi_em')
        for section in ('sampling', 'eval'):
            self.assertEqual(_canonicalize(mine[section]), _canonicalize(theirs[section]), section)

    def test_kappa_differs_from_a0_only_in_the_length_scale(self):
        mine, theirs = _canonicalize(gpu2.get_config(K).model), _canonicalize(followup.get_config('a0_c8_hi_em').model)
        self.assertEqual({key for key in set(mine) | set(theirs) if mine.get(key) != theirs.get(key)},
                         {'fox_matern_length_scale'})
        self.assertEqual(mine['fox_matern_length_scale'], 0.00692820323)

    def test_kappa_in_explicit_values(self):
        config = gpu2.get_config(K)
        expected = dict(method='pc', predictor='euler_maruyama', corrector='none', noise_removal=True,
                        probability_flow=False, time_grid='uniform_logsnr', logsnr_min=-10.0, logsnr_max=HI)
        for key, value in expected.items():
            self.assertEqual(config.sampling[key], value, key)
        self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.batch_size,
                          config.eval.sampling_seed), (1000, 10000, 512, 0))
        self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), ('x_mean', 'x_t', 'tweedie_xt'))
        self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (8, 8))

    def test_refuses_another_run_behind_kappa(self):
        path = kappa.manifest_path('k002')
        manifest = json.loads(path.read_text())
        manifest['config']['model']['fox_matern_length_scale'] = kappa.LENGTH_SCALES['a0']
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unexpected run'):
            gpu2.get_config(K)

    def test_inception_folder_is_the_hash_of_the_url(self):
        url = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
        self.assertEqual(hashlib.sha1(url.encode()).hexdigest(), gpu2.INCEPTION_MODULE)
        self.assertIn("INCEPTION_TFHUB = '{}'".format(url), (REPO / 'evaluation.py').read_text())


class Samplers(Base):
    def test_em_at_full_length_makes_1001_calls_with_three_outputs(self):
        config = self.config(K)
        sde, eps = run_lib.get_sde(config, n_override=gpu2.CELLS[K]['calls'])
        outputs, nfe = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)(ZeroEps())
        self.assertEqual(nfe, 1001)
        self.assertEqual(set(outputs), set(config.eval.endpoint_denoise_outputs))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()))

    def test_every_cell_grid_spans_its_window_and_its_sampler_builds(self):
        for name, spec in gpu2.CELLS.items():
            config = self.config(name)
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)


class Loading(Base):
    def test_exact_steps(self):
        for name, step in ((A, 400001), (K, 200001)):
            self.assertEqual(gpu2.CELLS[name]['step'], step)

    def test_each_model_uses_its_loader_and_a_neighbouring_step_is_refused(self):
        with patch.object(stage0, 'load_model', return_value=('m', dict(step=400000, checkpoint_id='i'))) as a, \
                patch.object(kappa, 'load_model', return_value=('m', dict(step=200002, checkpoint_id='i'))) as k:
            for name in (A, K):
                with self.assertRaisesRegex(RuntimeError, 'Expected step'):
                    gpu2.load_model('c', gpu2.CELLS[name])
        a.assert_called_once_with('c', gpu2.WORKDIRS['A'], 80)
        k.assert_called_once_with('c', gpu2.WORKDIRS['k002'], 8)

    def test_exact_step_passes(self):
        with patch.object(stage0, 'load_model', return_value=('m', dict(step=400001, checkpoint_id='i'))):
            self.assertEqual(gpu2.load_model('c', gpu2.CELLS[A])[0], 'm')


class Preflight(Base):
    def run_preflight(self, load, cache=True):
        module = self.root / 'cache' / gpu2.INCEPTION_MODULE
        (module / 'variables').mkdir(parents=True)
        if cache:
            (module / 'saved_model.pb').write_text('x')
        original = gpu2.get_config
        with patch.object(gpu2, 'load_model', load), \
                patch.object(gpu2, 'get_config', lambda name: on_cpu(original(name))), \
                patch.object(followup, 'smoke') as self.smoke, patch('torch.cuda.is_available', lambda: True), \
                patch('torch.cuda.device_count', lambda: 1), patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch('torch.cuda.empty_cache'), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0', TFHUB_CACHE_DIR=str(self.root / 'cache')), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(sampling, 'get_sampling_fn'):
            return gpu2.preflight()

    @staticmethod
    def good(config, spec):
        return ZeroEps(), dict(step=spec['step'], checkpoint_id='id')

    def saved(self):
        return json.loads((gpu2.LOGS / 'preflight.json').read_text())

    def test_passes_and_smokes_only_the_ode_cell(self):
        self.assertEqual(self.run_preflight(self.good), 0)
        self.assertEqual(sorted(self.saved()['checkpoints']), ['A_80', 'k002_8'])
        self.assertEqual(self.smoke.call_count, 1)

    def test_missing_checkpoint_80_fails_the_preflight_and_only_cell_a(self):
        def load(config, spec):
            if spec['model'] == 'A':
                raise RuntimeError('Missing checkpoint')
            return self.good(config, spec)

        self.assertEqual(self.run_preflight(load), 1)
        self.assertEqual(self.saved()['status'], 'failed')
        self.assertEqual(list(self.saved()['failures']), [A])
        self.assertEqual(list(self.saved()['checkpoints']), ['k002_8'])
        self.assertTrue(gpu2.failure_path(A).exists())

    def test_truncated_checkpoint_is_a_cell_failure(self):
        def load(config, spec):
            if spec['model'] == 'k002':
                raise EOFError('truncated')
            return self.good(config, spec)

        self.assertEqual(self.run_preflight(load), 1)
        self.assertEqual(list(self.saved()['failures']), [K])

    def test_no_local_inception_stops_both_cells_and_clears_the_old_preflight(self):
        (gpu2.LOGS / 'preflight.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'No local Inception'):
            self.run_preflight(self.good, cache=False)
        self.assertFalse((gpu2.LOGS / 'preflight.json').exists())
        self.assertEqual(gpu2.queue(), 1)


class Report(Base):
    def prepare(self, name, nfe=None, step=None, shards=None):
        spec = gpu2.CELLS[name]
        config = gpu2.get_config(name)
        directory = self.root / 'workdirs' / gpu2.WORKDIRS[spec['model']] / gpu2.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range(shards or 20):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        resolved = json.loads(json.dumps(_canonicalize(config)))
        resolved['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved, source_sha256='source')))
        path = gpu2.LOGS / 'preflight.json'
        saved = json.loads(path.read_text()) if path.exists() else dict(source_sha256='source', checkpoints={}, failures={})
        saved['checkpoints'][gpu2.identity_key(spec)] = dict(step=spec['step'], checkpoint_id='id')
        path.write_text(json.dumps(saved))
        (gpu2.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=spec['count'], nfe=nfe or spec['calls'] + 1, checkpoint_step=step or spec['step'],
                      checkpoint_id='id', dataset_stats_id=gpu2.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in config.eval.endpoint_denoise_outputs:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)

    def test_em_report_expects_1001_calls_and_three_outputs(self):
        self.prepare(K)
        gpu2.report(K)
        result = json.loads(gpu2.result_path(K).read_text())
        self.assertEqual((result['nfe'], result['fid_x_mean'], result['checkpoint_step']), (1001, 2.5, 200001))

    def test_a_report_expects_65_calls(self):
        self.prepare(A)
        gpu2.report(A)
        result = json.loads(gpu2.result_path(A).read_text())
        self.assertEqual((result['nfe'], result['checkpoint_step'], result['lambda_window']), (65, 400001, [-10.0, LO]))

    def test_refuses_wrong_calls_step_and_shards(self):
        for arguments, message in ((dict(nfe=1000), 'report identity'), (dict(step=395001), 'report identity'),
                                   (dict(shards=3), 'shards')):
            self.prepare(K, **arguments)
            with self.assertRaisesRegex(RuntimeError, message):
                gpu2.report(K)
            shutil.rmtree(str(self.root / 'workdirs' / gpu2.WORKDIRS['k002'] / gpu2.eval_folder(K)))

    def test_refuses_another_protocol(self):
        self.prepare(A)
        path = self.root / 'workdirs' / gpu2.WORKDIRS['A'] / gpu2.eval_folder(A) / 'manifest.json'
        saved = json.loads(path.read_text())
        saved['config']['sampling']['logsnr_max'] = HI
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, 'report config'):
            gpu2.report(A)

    def test_summary_keeps_the_fields_of_the_night_queue(self):
        self.prepare(A)
        gpu2.report(A)
        summary = json.loads((gpu2.LOGS / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'running')
        self.assertLessEqual({'status', 'completed_cells', 'total_cells', 'cells', 'point_contrasts',
                              'reference_differences', 'confidence_intervals_computed', 'scope'}, set(summary))
        self.assertAlmostEqual(summary['reference_differences'][0]['value'], 2.5 - 2.6024169921875)
        lines = (gpu2.LOGS / 'summary.csv').read_text().splitlines()
        self.assertEqual(lines[0], 'cell,model,step,lambda_max,method,sampling_seed,num_samples,nfe,'
                                   'fid_tweedie,fid_x_t,kid_tweedie,is_tweedie')
        self.assertEqual(len(lines), 2)

    def test_summary_names_the_failed_cell(self):
        self.prepare(K)
        gpu2.report(K)
        gpu2.write_summary([A])
        summary = json.loads((gpu2.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['status'], summary['failed_cells'], summary['completed_cells']), ('failed', [A], 1))


class Queue(Base):
    def write_preflight(self, failures=None):
        (gpu2.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=failures or {}, checkpoints={})))

    def run_queue(self, fake):
        with patch.object(gpu2, 'run', side_effect=fake) as run, patch.object(gpu2, 'report', lambda name: None), \
                patch.object(gpu2, 'write_summary') as self.summary:
            code = gpu2.queue()
        return code, [call[0][0] for call in run.call_args_list]

    def test_a_failure_of_the_first_cell_does_not_cancel_the_second(self):
        def fake(name):
            if name == A:
                raise subprocess.CalledProcessError(1, 'main.py')
            gpu2.result_path(name).write_text('{}')

        self.write_preflight()
        code, called = self.run_queue(fake)
        self.assertEqual((code, called), (1, [A, K, A]))
        self.assertEqual(json.loads(gpu2.failure_path(A).read_text())['attempt'], 2)

    def test_second_pass_can_recover(self):
        seen = []

        def fake(name):
            seen.append(name)
            if seen.count(A) == 2 or name == K:
                gpu2.result_path(name).write_text('{}')
            elif name == A:
                raise subprocess.CalledProcessError(1, 'main.py')

        self.write_preflight()
        code, called = self.run_queue(fake)
        self.assertEqual((code, called), (0, [A, K, A]))
        self.assertFalse(gpu2.failure_path(A).exists())

    def test_a_cell_blocked_by_preflight_is_not_run_and_the_other_is(self):
        self.write_preflight({A: 'wrong step'})
        code, called = self.run_queue(lambda name: gpu2.result_path(name).write_text('{}'))
        self.assertEqual((code, called), (1, [K]))
        self.summary.assert_called_once_with([A])

    def test_finished_cells_are_skipped(self):
        self.write_preflight()
        for name in gpu2.CELLS:
            gpu2.result_path(name).write_text('{}')
        with patch.object(gpu2, 'run') as run, patch.object(gpu2, 'write_summary'):
            self.assertEqual(gpu2.queue(), 0)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
