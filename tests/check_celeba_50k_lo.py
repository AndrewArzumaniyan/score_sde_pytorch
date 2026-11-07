"""CPU checks for the 50k queues on the lower window; real SDEs and samplers, no checkpoints.

LO50K_TEST_INPUTS may point to a copy of the real run_manifest.json of A.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
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

import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_matched35_rtx_eval as rtx
import celeba_stage0_eval as stage0
import run_lib
import sampling
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a

LO = 9.115429865459795
HI = 10.575749165617202
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
A0 = lo.cell_name('A', DPM, 0)
A1 = lo.cell_name('A', HEUN, 0)
V0 = lo.cell_name('vp', DPM, 0)
F0 = lo.cell_name('fox', DPM, 0)


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def write_manifest(target):
    work = followup.WORKDIRS['A']
    source = os.environ.get('LO50K_TEST_INPUTS')
    if source:
        shutil.copy2(str(Path(source) / (work + '.json')), str(target / (work + '.json')))
        return
    config = get_a()
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
        write_manifest(inputs)
        for module, target, value in ((lo, 'ROOT', self.root), (lo, 'LOGS', self.root / 'logs'),
                                      (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, name):
        return on_cpu(lo.get_config(name))


class Plan(Base):
    def test_lanes_split_the_twelve_cells_in_order(self):
        expected = [lo.cell_name(model, method, seed) for model in ('A', 'vp', 'fox')
                    for seed in (0, 1) for method in (DPM, HEUN)]
        self.assertEqual(list(lo.CELLS), expected)
        self.assertEqual(lo.lane_cells('gpu1'), expected[:8])
        self.assertEqual(lo.lane_cells('gpu2'), expected[8:])

    def test_vp_equals_the_night_cell_except_the_window(self):
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = _canonicalize(lo.get_config(lo.cell_name('vp', method, seed)))
                theirs = _canonicalize(followup.get_config('vp_c40_hi_{}_seed{}'.format(method, seed)))
                self.assertEqual(mine['sampling'].pop('logsnr_max'), LO)
                self.assertEqual(theirs['sampling'].pop('logsnr_max'), HI)
                self.assertEqual(mine, theirs)

    def test_fox_equals_matched35_except_the_window(self):
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = _canonicalize(lo.get_config(lo.cell_name('fox', method, seed)))
                theirs = _canonicalize(rtx.config_for((34, 50000, 'fox', method, seed)))
                self.assertEqual(mine['sampling'].pop('logsnr_max'), LO)
                self.assertEqual(theirs['sampling'].pop('logsnr_max'), HI)
                self.assertEqual(mine, theirs)

    def test_a_shares_every_protocol_value_with_vp(self):
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = lo.get_config(lo.cell_name('A', method, seed))
                theirs = lo.get_config(lo.cell_name('vp', method, seed))
                for key in ('method', 'probability_flow', 'time_grid', 'logsnr_min', 'logsnr_max', 'noise_removal',
                            'predictor', 'corrector'):
                    self.assertEqual(mine.sampling[key], theirs.sampling[key], key)
                for key in ('sampling_num_scales', 'sampling_seed', 'num_samples', 'batch_size', 'enable_loss',
                            'enable_bpd', 'enable_sampling', 'endpoint_denoise_variants', 'save_endpoint_states',
                            'include_final_checkpoint'):
                    self.assertEqual(mine.eval[key], theirs.eval[key], key)
                self.assertEqual(tuple(mine.eval.endpoint_denoise_outputs), tuple(theirs.eval.endpoint_denoise_outputs))
                for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
                    self.assertEqual(mine[key], theirs[key], key)

    def test_explicit_values(self):
        for name, spec in lo.CELLS.items():
            config = lo.get_config(name)
            self.assertEqual((config.sampling.logsnr_min, config.sampling.logsnr_max), (-10.0, LO))
            self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.batch_size),
                             (34, 50000, 512))
            self.assertEqual(config.eval.sampling_seed, spec['seed'])
            self.assertEqual(config.sampling.method, spec['method'])
            self.assertEqual(config.sampling.time_grid, 'karras' if spec['method'] == HEUN else 'uniform_logsnr')
            self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (spec['ckpt'], spec['ckpt']))
            self.assertTrue(config.eval.save_endpoint_states)
            self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), ('x_t', 'tweedie_xt'))

    def test_a_rebuilds_the_trained_model(self):
        manifest = json.loads(followup.manifest_path('A').read_text())
        self.assertEqual(build_model_manifest(lo.get_config(A0))['model_protocol_sha256'],
                         manifest['model_manifest']['model_protocol_sha256'])

    def test_native_models_are_the_stage_0_ones(self):
        for model in ('vp', 'fox'):
            config = lo.get_config(lo.cell_name(model, DPM, 0))
            self.assertEqual(_canonicalize(config.model), _canonicalize(stage0.native_config(model).model))

    def test_identities(self):
        self.assertEqual({m: (v['ckpt'], v['step']) for m, v in lo.MODELS.items()},
                         {'A': (80, 400001), 'vp': (40, 200001), 'fox': (260, 1300002)})
        self.assertEqual(lo.MODELS['A']['checkpoint_id'], '7de461feeceb4a488583921d9f7f132d')


class Samplers(Base):
    def test_every_cell_grid_spans_the_window_and_its_sampler_builds(self):
        for name, spec in lo.CELLS.items():
            config = self.config(name)
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)

    def test_smoke_runs_every_model_and_solver(self):
        for model in lo.MODELS:
            for method in (DPM, HEUN):
                config = self.config(lo.cell_name(model, method, 0))
                _, eps = run_lib.get_sde(config, n_override=34)
                followup.smoke(config, ZeroEps(), method, eps)

    def test_full_sampler_calls_and_outputs(self):
        for model in lo.MODELS:
            for method in (DPM, HEUN):
                config = self.config(lo.cell_name(model, method, 0))
                sde, eps = run_lib.get_sde(config, n_override=34)
                outputs, nfe = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)(ZeroEps())
                self.assertEqual(nfe, 35)
                self.assertEqual(set(outputs), {'x_t', 'tweedie_xt'})


class Loading(Base):
    def identity(self, **changes):
        return ('m', dict(dict(step=400001, checkpoint_id=lo.MODELS['A']['checkpoint_id']), **changes))

    def test_exact_identity_passes(self):
        with patch.object(stage0, 'load_model', return_value=self.identity()) as load:
            self.assertEqual(lo.load_model('c', lo.CELLS[A0])[0], 'm')
        load.assert_called_once_with('c', lo.MODELS['A']['work'], 80)

    def test_wrong_step_or_id_is_refused(self):
        for changes, message in ((dict(step=400000), 'Expected step'), (dict(checkpoint_id='other'), 'checkpoint id')):
            with patch.object(stage0, 'load_model', return_value=self.identity(**changes)):
                with self.assertRaisesRegex(RuntimeError, message):
                    lo.load_model('c', lo.CELLS[A0])


class Preflight(Base):
    def run_preflight(self, lane, load, cache=True):
        module = self.root / 'cache' / lo.INCEPTION_MODULE
        (module / 'variables').mkdir(parents=True, exist_ok=True)
        if cache:
            (module / 'saved_model.pb').write_text('x')
        self.loaded = []

        def counted(config, spec):
            self.loaded.append(spec['model'])
            return load(config, spec)

        original = lo.get_config
        with patch.object(lo, 'load_model', counted), \
                patch.object(lo, 'get_config', lambda name: on_cpu(original(name))), \
                patch.object(followup, 'smoke') as self.smoke, patch('torch.cuda.is_available', lambda: True), \
                patch('torch.cuda.device_count', lambda: 1), patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch('torch.cuda.empty_cache'), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0', TFHUB_CACHE_DIR=str(self.root / 'cache')), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(sampling, 'get_sampling_fn'):
            return lo.preflight(lane)

    @staticmethod
    def good(config, spec):
        expected = lo.MODELS[spec['model']]
        return ZeroEps(), dict(step=expected['step'], checkpoint_id=expected['checkpoint_id'])

    def saved(self):
        return json.loads((lo.LOGS / 'preflight.json').read_text())

    def test_gpu1_loads_each_model_once_and_smokes_every_cell(self):
        self.assertEqual(self.run_preflight('gpu1', self.good), 0)
        self.assertEqual(self.loaded, ['A', 'vp'])
        self.assertEqual(self.smoke.call_count, 8)
        self.assertEqual(sorted(self.saved()['checkpoints']), ['A', 'vp'])

    def test_gpu2_touches_only_fox(self):
        self.assertEqual(self.run_preflight('gpu2', self.good), 0)
        self.assertEqual(self.loaded, ['fox'])

    def test_a_bad_model_blocks_only_its_cells(self):
        def load(config, spec):
            if spec['model'] == 'A':
                raise RuntimeError('Expected step')
            return self.good(config, spec)

        self.assertEqual(self.run_preflight('gpu1', load), 1)
        self.assertEqual(sorted(self.saved()['failures']), sorted(lo.lane_cells('gpu1')[:4]))
        self.assertEqual(list(self.saved()['checkpoints']), ['vp'])

    def test_truncated_checkpoint_is_a_cell_failure(self):
        def load(config, spec):
            raise EOFError('truncated')

        self.assertEqual(self.run_preflight('gpu2', load), 1)
        self.assertEqual(len(self.saved()['failures']), 4)

    def test_no_local_inception_stops_the_lane_and_clears_the_old_preflight(self):
        (lo.LOGS / 'preflight.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'No local Inception'):
            self.run_preflight('gpu1', self.good, cache=False)
        self.assertFalse((lo.LOGS / 'preflight.json').exists())
        self.assertEqual(lo.queue('gpu1'), 1)


class Report(Base):
    def prepare(self, name, nfe=None, step=None, shards=98, features=True):
        spec = lo.CELLS[name]
        expected = lo.MODELS[spec['model']]
        config = lo.get_config(name)
        directory = self.root / 'workdirs' / expected['work'] / lo.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range(shards):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        arrays = {lo.FEATURES if features else 'pool_3_x_t': np.zeros((512, 2048), np.float32)}
        np.savez(str(rounds / 'statistics_0.npz'), **arrays)
        resolved = json.loads(json.dumps(_canonicalize(config)))
        resolved['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved, source_sha256='source')))
        (lo.LOGS / 'preflight.json').write_text(json.dumps(dict(source_sha256='source', checkpoints={}, failures={})))
        (lo.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=50000, nfe=nfe or 35, checkpoint_step=step or expected['step'],
                      checkpoint_id=expected['checkpoint_id'], dataset_stats_id=lo.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in config.eval.endpoint_denoise_outputs:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_report_of_each_model(self):
        for name, lane, step in ((A0, 'gpu1', 400001), (V0, 'gpu1', 200001), (F0, 'gpu2', 1300002)):
            self.prepare(name)
            lo.report(name, lane)
            result = json.loads(lo.result_path(name).read_text())
            self.assertEqual((result['nfe'], result['num_samples'], result['checkpoint_step'], result['lambda_window']),
                             (35, 50000, step, [-10.0, LO]))

    def test_refuses_wrong_identity_shards_and_missing_features(self):
        for arguments, message in ((dict(nfe=65), 'report identity'), (dict(step=395001), 'report identity'),
                                   (dict(shards=20), 'shards'), (dict(features=False), 'bootstrap features')):
            directory = self.prepare(A0, **arguments)
            with self.assertRaisesRegex(RuntimeError, message):
                lo.report(A0, 'gpu1')
            shutil.rmtree(str(directory))

    def test_refuses_the_upper_window(self):
        directory = self.prepare(V0)
        saved = json.loads((directory / 'manifest.json').read_text())
        saved['config']['sampling']['logsnr_max'] = HI
        (directory / 'manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, 'report config'):
            lo.report(V0, 'gpu1')

    def test_summary_fields_contrasts_and_references(self):
        self.prepare(A0)
        lo.report(A0, 'gpu1')
        self.prepare(A1)
        lo.report(A1, 'gpu1')
        self.prepare(V0)
        lo.report(V0, 'gpu1')
        summary = json.loads((lo.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['status'], summary['completed_cells'], summary['total_cells']), ('running', 3, 8))
        self.assertLessEqual({'status', 'completed_cells', 'total_cells', 'cells', 'point_contrasts',
                              'reference_differences', 'confidence_intervals_computed', 'scope'}, set(summary))
        self.assertEqual(sorted(c['contrast'] for c in summary['point_contrasts']),
                         ['A_dpmpp_minus_heun_seed0', 'A_minus_vp_ode_dpmpp_2m_seed0'])
        values = {r['cell']: r['value'] for r in summary['reference_differences']}
        self.assertAlmostEqual(values[A0], 2.5 - 2.57734751701355)
        self.assertAlmostEqual(values[V0], 2.5 - 3.3268163204193115)
        lines = (lo.LOGS / 'summary.csv').read_text().splitlines()
        self.assertEqual(lines[0], 'cell,model,step,lambda_max,method,sampling_seed,num_samples,nfe,'
                                   'fid_tweedie,fid_x_t,kid_tweedie,is_tweedie')
        self.assertEqual(len(lines), 4)

    def test_fox_reference(self):
        self.prepare(F0)
        lo.report(F0, 'gpu2')
        summary = json.loads((lo.LOGS / 'summary.json').read_text())
        self.assertAlmostEqual(summary['reference_differences'][0]['value'], 2.5 - 2.4948720932006836)
        self.assertEqual(summary['total_cells'], 4)

    def test_summary_of_a_lane_ignores_the_other_lane(self):
        self.prepare(A0)
        lo.report(A0, 'gpu1')
        lo.write_summary('gpu2')
        self.assertEqual(json.loads((lo.LOGS / 'summary.json').read_text())['completed_cells'], 0)


class Queue(Base):
    def write_preflight(self, failures=None):
        (lo.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=failures or {}, checkpoints={})))

    def run_queue(self, fake, lane='gpu1'):
        with patch.object(lo, 'run', side_effect=fake) as run, patch.object(lo, 'report', lambda name, lane: None), \
                patch.object(lo, 'write_summary') as self.summary:
            code = lo.queue(lane)
        return code, [call[0][0] for call in run.call_args_list]

    def test_a_failed_cell_does_not_cancel_the_next_and_gets_a_second_pass(self):
        def fake(name):
            if name == A0:
                raise subprocess.CalledProcessError(1, 'main.py')
            lo.result_path(name).write_text('{}')

        self.write_preflight()
        code, called = self.run_queue(fake)
        self.assertEqual((code, called), (1, lo.lane_cells('gpu1') + [A0]))
        self.assertEqual(json.loads(lo.failure_path(A0).read_text())['attempt'], 2)
        self.summary.assert_called_once_with('gpu1', [A0])

    def test_second_pass_can_recover(self):
        seen = []

        def fake(name):
            seen.append(name)
            if name != A0 or seen.count(A0) == 2:
                lo.result_path(name).write_text('{}')
            else:
                raise subprocess.CalledProcessError(1, 'main.py')

        self.write_preflight()
        code, called = self.run_queue(fake)
        self.assertEqual((code, called[-1]), (0, A0))
        self.assertFalse(lo.failure_path(A0).exists())

    def test_cells_blocked_by_preflight_are_not_run_and_the_others_are(self):
        blocked = lo.lane_cells('gpu1')[:4]
        self.write_preflight({name: 'wrong step' for name in blocked})
        code, called = self.run_queue(lambda name: lo.result_path(name).write_text('{}'))
        self.assertEqual((code, called), (1, lo.lane_cells('gpu1')[4:]))
        self.summary.assert_called_once_with('gpu1', blocked)

    def test_finished_cells_are_skipped(self):
        self.write_preflight()
        for name in lo.lane_cells('gpu2'):
            lo.result_path(name).write_text('{}')
        code, called = self.run_queue(lambda name: None, 'gpu2')
        self.assertEqual((code, called), (0, []))


class ContainerState(unittest.TestCase):
    """Docker 29 prints an empty line and exits 1 for an unknown name."""

    def state(self, script, output, code):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(root))
        (root / 'docker').write_text('#!/usr/bin/env bash\nprintf "{}"\nexit {}\n'.format(output, code))
        (root / 'docker').chmod(0o755)
        definition = re.search(r'^state\(\) .*$', (REPO / script).read_text(), re.M).group(0)
        result = subprocess.run(['bash', '-c', 'after=x\n{}\nstate'.format(definition)], stdout=subprocess.PIPE,
                                env=dict(os.environ, PATH='{}:{}'.format(root, os.environ['PATH'])))
        return result.stdout.decode()

    def test_both_launchers(self):
        for script in ('run_celeba_50k_lo_rtx.sh', 'run_celeba_gpu2_rtx.sh'):
            self.assertEqual(self.state(script, r'\n', 1), 'gone\n', script)
            self.assertEqual(self.state(script, '', 1), 'gone\n', script)
            self.assertEqual(self.state(script, r'true\n', 0), 'true\n', script)
            self.assertEqual(self.state(script, r'false\n', 0), 'false\n', script)


if __name__ == '__main__':
    unittest.main()
