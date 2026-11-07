"""CPU checks for the CelebA follow-up; real SDEs and samplers, no checkpoints.

FOLLOWUP_TEST_INPUTS may point to copies of the real run_manifest.json files.
"""
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
import celeba_kappa_eval as kappa
import celeba_matched35_rtx_eval as rtx
import celeba_stage0_eval as stage0
import run_lib
import sampling
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.fox.celeba_kaffine import get_config as get_kaffine

HI = 10.575749165617202
LO = 9.115429865459795
VP_CELLS = [name for name, spec in followup.CELLS.items() if spec['protocol'] == 'matched35']


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def write_manifests(target):
    source = os.environ.get('FOLLOWUP_TEST_INPUTS')
    for model, config in (('A', get_a()), ('a0', get_kaffine('a0'))):
        work = followup.WORKDIRS[model]
        if source:
            shutil.copy2(str(Path(source) / (work + '.json')), str(target / (work + '.json')))
            continue
        resolved = _canonicalize(config)
        resolved.pop('eval')
        resolved['device'] = 'cuda:0'
        (target / (work + '.json')).write_text(json.dumps(dict(config=resolved, model_manifest=build_model_manifest(config))))


def flatten(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix[:-1]: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, prefix + key + '.'))
    return result


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
        for module, target, value in ((followup, 'ROOT', self.root), (followup, 'LOGS', self.root / 'logs'),
                                      (followup, 'TRAINING_INPUTS', inputs), (kappa, 'TRAINING_INPUTS', inputs),
                                      (followup, 'WARM_PAUSE_SECONDS', 0)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, name):
        return on_cpu(followup.get_config(name))


class Plan(Base):
    def test_cells_and_their_order(self):
        self.assertEqual(list(followup.CELLS), [
            'A_c60_lo', 'vp_c40_hi_ode_dpmpp_2m_seed0', 'vp_c40_hi_ode_edm_heun_seed0',
            'vp_c40_hi_ode_dpmpp_2m_seed1', 'vp_c40_hi_ode_edm_heun_seed1', 'a0_c8_hi_em'])

    def test_a_is_the_stage_0_protocol_at_checkpoint_60(self):
        config = followup.get_config('A_c60_lo')
        fox = stage0.get_config('fox_c40_hi')
        for key in ('method', 'probability_flow', 'time_grid', 'logsnr_min', 'noise_removal', 'predictor', 'corrector'):
            self.assertEqual(config.sampling[key], fox.sampling[key], key)
        for key in ('sampling_num_scales', 'sampling_seed', 'num_samples', 'batch_size', 'enable_loss', 'enable_bpd',
                    'enable_sampling', 'endpoint_denoise_variants', 'save_endpoint_states'):
            self.assertEqual(config.eval[key], fox.eval[key], key)
        self.assertEqual(config.sampling.logsnr_max, LO)
        self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (60, 60))
        self.assertEqual(config.eval.sampling_num_scales, 64)

    def test_a_rebuilds_the_trained_model(self):
        manifest = json.loads(followup.manifest_path('A').read_text())
        self.assertEqual(build_model_manifest(followup.get_config('A_c60_lo'))['model_protocol_sha256'],
                         manifest['model_manifest']['model_protocol_sha256'])

    def test_refuses_another_run_behind_a(self):
        path = followup.manifest_path('A')
        manifest = json.loads(path.read_text())
        manifest['config']['seed'] = 43
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unexpected run'):
            followup.get_config('A_c60_lo')

    def test_vp_equals_matched35_except_the_checkpoint(self):
        for name in VP_CELLS:
            spec = followup.CELLS[name]
            mine = flatten(_canonicalize(followup.get_config(name)))
            theirs = flatten(_canonicalize(rtx.config_for((34, 50000, 'vp', spec['method'], spec['seed']))))
            changed = {key for key in set(mine) | set(theirs) if mine.get(key) != theirs.get(key)}
            self.assertEqual(changed, {'eval.begin_ckpt', 'eval.end_ckpt'})
            self.assertEqual((mine['eval.begin_ckpt'], theirs['eval.begin_ckpt']), (40, 260))

    def test_vp_protocol_in_explicit_values(self):
        for name in VP_CELLS:
            spec = followup.CELLS[name]
            config = followup.get_config(name)
            self.assertEqual(_canonicalize(config.model), _canonicalize(stage0.native_config('vp').model))
            self.assertEqual(config.sampling.method, spec['method'])
            self.assertEqual(config.sampling.time_grid,
                             'karras' if spec['method'] == 'ode_edm_heun' else 'uniform_logsnr')
            self.assertEqual((config.sampling.logsnr_min, config.sampling.logsnr_max), (-10.0, HI))
            self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.batch_size),
                             (34, 50000, 512))
            self.assertEqual(config.eval.sampling_seed, spec['seed'])
            self.assertTrue(config.allow_tf32 and config.cudnn_benchmark)

    def test_em_is_the_e3b_protocol_on_a0(self):
        config = followup.get_config('a0_c8_hi_em')
        expected = dict(method='pc', predictor='euler_maruyama', corrector='none', noise_removal=True,
                        probability_flow=False, time_grid='uniform_logsnr', logsnr_min=-10.0, logsnr_max=HI)
        for key, value in expected.items():
            self.assertEqual(config.sampling[key], value, key)
        self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.batch_size,
                          config.eval.sampling_seed), (1000, 10000, 512, 0))
        self.assertTrue(config.eval.endpoint_denoise_variants)
        self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (8, 8))
        self.assertEqual(config.model.fox_drift_schedule, 'affine')
        self.assertEqual(config.model.fox_matern_length_scale, kappa.LENGTH_SCALES['a0'])

    def test_a0_at_ode_cell_is_unchanged(self):
        self.assertEqual(kappa.get_config('a0_c8_hi').sampling.method, stage0.METHOD)


class Samplers(Base):
    def test_em_sampler_builds_and_gives_exactly_the_requested_outputs(self):
        config = self.config('a0_c8_hi_em')
        sde, eps = run_lib.get_sde(config, n_override=4)
        sampler = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)
        outputs, nfe = sampler(ZeroEps())
        self.assertEqual(nfe, 5)  # 1001 at 1000 scales: evaluate_checkpoint requires scales + 1.
        self.assertEqual(set(outputs), set(config.eval.endpoint_denoise_outputs))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()))

    def test_em_at_full_length_makes_1001_calls_on_a_strict_grid(self):
        config = self.config('a0_c8_hi_em')
        sde, eps = run_lib.get_sde(config, n_override=followup.EM_CALLS)
        grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device='cpu', dtype=torch.float32, N=sde.N + 1)
        self.assertTrue(bool(torch.all(grid[1:] < grid[:-1])))
        outputs, nfe = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)(ZeroEps())
        self.assertEqual(nfe, 1001)
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()))

    def test_every_cell_grid_spans_its_window_and_its_sampler_builds(self):
        for name, spec in followup.CELLS.items():
            config = self.config(name)
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)

    def test_karras_grid_outside_the_window_is_refused(self):
        name = 'vp_c40_hi_ode_edm_heun_seed0'
        config = self.config(name)
        sde, eps = run_lib.get_sde(config, n_override=34)
        config.sampling.logsnr_max = LO
        with self.assertRaisesRegex(RuntimeError, 'Karras'):
            followup.check_cell_grid(config, sde, eps, followup.CELLS[name])


class Preflight(Base):
    def run_preflight(self):
        loaded = []

        def load(config, spec):
            loaded.append(followup.identity_key(spec))
            return ZeroEps(), dict(step=1, checkpoint_id='id')

        original = followup.get_config
        with patch.object(followup, 'load_model', load), \
                patch.object(followup, 'get_config', lambda name: on_cpu(original(name))), \
                patch.object(followup, 'smoke') as smoke, patch('torch.cuda.is_available', lambda: True), \
                patch('torch.cuda.device_count', lambda: 1), patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch('torch.cuda.empty_cache'), patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0'), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(sampling, 'get_sampling_fn') as build:
            followup.preflight()
        return loaded, smoke, build

    def test_loads_each_checkpoint_once_and_skips_the_ode_smoke_for_em(self):
        loaded, smoke, build = self.run_preflight()
        self.assertEqual(loaded, ['A_60', 'vp_40', 'a0_8'])
        self.assertEqual(smoke.call_count, 5)
        self.assertEqual(build.call_count, 6)
        self.assertEqual(sorted(json.loads((followup.LOGS / 'preflight.json').read_text())['checkpoints']),
                         ['A_60', 'a0_8', 'vp_40'])

    def test_smoke_runs_both_ode_solvers(self):
        for name in ('vp_c40_hi_ode_dpmpp_2m_seed0', 'vp_c40_hi_ode_edm_heun_seed0'):
            config = self.config(name)
            _, eps = run_lib.get_sde(config, n_override=34)
            followup.smoke(config, ZeroEps(), followup.CELLS[name]['method'], eps)

    def test_loaders_accept_their_own_snapshot_scale(self):
        with patch.object(kappa, 'load_model', return_value='kappa') as a0, \
                patch.object(stage0, 'load_model', return_value='stage0') as other:
            self.assertEqual(followup.load_model('c', followup.CELLS['a0_c8_hi_em']), 'kappa')
            self.assertEqual(followup.load_model('c', followup.CELLS['A_c60_lo']), 'stage0')
        a0.assert_called_once_with('c', followup.WORKDIRS['a0'], 8)
        other.assert_called_once_with('c', followup.WORKDIRS['A'], 60)


class Report(Base):
    def prepare(self, name, nfe=None, step=200001, shards=None):
        spec = followup.CELLS[name]
        config = followup.get_config(name)
        directory = self.root / 'workdirs' / followup.WORKDIRS[spec['model']] / followup.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range(shards or (spec['count'] + 511) // 512):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        resolved = json.loads(json.dumps(_canonicalize(config)))
        resolved['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved, source_sha256='source')))
        path = followup.LOGS / 'preflight.json'
        saved = json.loads(path.read_text()) if path.exists() else dict(source_sha256='source', checkpoints={})
        saved['checkpoints'][followup.identity_key(spec)] = dict(step=step, checkpoint_id='id')
        path.write_text(json.dumps(saved))
        (followup.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=spec['count'], nfe=nfe or spec['calls'] + 1, checkpoint_step=step,
                      checkpoint_id='id', dataset_stats_id=followup.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in config.eval.endpoint_denoise_outputs:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)

    def test_em_report_expects_1001_calls_and_three_outputs(self):
        self.prepare('a0_c8_hi_em')
        followup.report('a0_c8_hi_em')
        result = json.loads(followup.result_path('a0_c8_hi_em').read_text())
        self.assertEqual(result['nfe'], 1001)
        self.assertEqual(result['fid_x_mean'], 2.5)

    def test_em_report_refuses_1000_calls(self):
        self.prepare('a0_c8_hi_em', nfe=1000)
        with self.assertRaisesRegex(RuntimeError, 'report identity'):
            followup.report('a0_c8_hi_em')

    def test_vp_report_expects_35_calls_and_98_shards(self):
        name = 'vp_c40_hi_ode_edm_heun_seed1'
        self.prepare(name)
        followup.report(name)
        result = json.loads(followup.result_path(name).read_text())
        self.assertEqual((result['nfe'], result['num_samples'], result['sampling_seed']), (35, 50000, 1))

    def test_report_refuses_missing_shards(self):
        self.prepare('A_c60_lo', shards=3)
        with self.assertRaisesRegex(RuntimeError, 'shards'):
            followup.report('A_c60_lo')

    def test_report_refuses_another_checkpoint(self):
        self.prepare('A_c60_lo', step=300001)
        path = followup.LOGS / 'preflight.json'
        saved = json.loads(path.read_text())
        saved['checkpoints']['A_60']['checkpoint_id'] = 'other'
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, 'report identity'):
            followup.report('A_c60_lo')

    def test_report_refuses_another_protocol(self):
        self.prepare('A_c60_lo')
        path = self.root / 'workdirs' / followup.WORKDIRS['A'] / followup.eval_folder('A_c60_lo') / 'manifest.json'
        saved = json.loads(path.read_text())
        saved['config']['sampling']['logsnr_max'] = HI
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, 'report config'):
            followup.report('A_c60_lo')

    def test_summary_has_contrasts_and_references(self):
        for name in ('A_c60_lo', 'vp_c40_hi_ode_dpmpp_2m_seed0', 'vp_c40_hi_ode_edm_heun_seed0'):
            self.prepare(name)
            followup.report(name)
        summary = json.loads((followup.LOGS / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'running')
        self.assertEqual([c['contrast'] for c in summary['point_contrasts']], ['vp200k_dpmpp_minus_heun_seed0'])
        self.assertAlmostEqual(summary['reference_differences'][0]['value'], 2.5 - 2.7945523262023926)
        self.assertEqual(len((followup.LOGS / 'summary.csv').read_text().splitlines()), 4)


class Queue(Base):
    def test_a_failed_cell_gets_a_second_pass_after_the_others(self):
        called = []
        followup.result_path('A_c60_lo').write_text('{}')

        def fake(name):
            called.append(name)
            if name == 'vp_c40_hi_ode_edm_heun_seed0' and called.count(name) == 1:
                raise subprocess.CalledProcessError(1, 'main.py')
            followup.result_path(name).write_text('{}')

        with patch.object(followup, 'run', fake), patch.object(followup, 'report', lambda name: None), \
                patch.object(followup, 'warm_inception') as warm:
            self.assertEqual(followup.queue(), 0)
        self.assertEqual(called, list(followup.CELLS)[1:] + ['vp_c40_hi_ode_edm_heun_seed0'])
        self.assertEqual(warm.call_count, 2)
        self.assertFalse(followup.failure_path('vp_c40_hi_ode_edm_heun_seed0').exists())

    def test_a_cell_that_fails_twice_is_reported(self):
        def fake(name):
            if name == 'a0_c8_hi_em':
                raise subprocess.CalledProcessError(1, 'main.py')
            followup.result_path(name).write_text('{}')

        with patch.object(followup, 'run', side_effect=fake) as run, \
                patch.object(followup, 'report', lambda name: None), patch.object(followup, 'warm_inception'):
            self.assertEqual(followup.queue(), 1)
        self.assertEqual(run.call_count, 7)
        self.assertEqual(json.loads(followup.failure_path('a0_c8_hi_em').read_text())['attempt'], 2)

    def test_finished_queue_does_not_touch_the_network(self):
        for name in followup.CELLS:
            followup.result_path(name).write_text('{}')
        with patch.object(followup, 'run') as run, patch.object(followup, 'warm_inception') as warm:
            self.assertEqual(followup.queue(), 0)
        run.assert_not_called()
        warm.assert_not_called()


class Warmup(Base):
    def test_retries_without_a_gpu_until_the_download_works(self):
        results = [MagicMock(returncode=1), subprocess.TimeoutExpired('python', 1), MagicMock(returncode=0)]
        with patch.object(followup.subprocess, 'run', side_effect=results) as run:
            self.assertTrue(followup.warm_inception())
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args[1]['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertIn('get_inception_model', run.call_args[0][0][2])

    def test_gives_up_without_stopping_the_queue(self):
        with patch.object(followup.subprocess, 'run', return_value=MagicMock(returncode=1)) as run:
            self.assertFalse(followup.warm_inception())
        self.assertEqual(run.call_count, followup.WARM_ATTEMPTS)


if __name__ == '__main__':
    unittest.main()
