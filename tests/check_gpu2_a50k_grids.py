"""CPU checks for the grids and the CelebA A 50k cells on GPU 2 and their launcher; no GPU, data or checkpoints."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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

import afhqv2_rtx_train as afhq
import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import gpu2_a50k_grids as g
import gpu2_queue_eval as q
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b
from reproducibility import derive_seed

DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
FIDS = {80: 2.577, 100: 2.744, 120: 2.830, 140: 2.90, 160: 3.00}


def resolved_manifest(config, protocol):
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256=protocol)


def factorial(get, seed=42, freq=5000):
    config = get()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.seed = seed
    config.training.snapshot_freq = freq
    return config


def without_range(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    resolved['eval'].pop('begin_ckpt')
    resolved['eval'].pop('end_ckpt')
    return resolved


def q_spec(name, **changes):
    """A cell of the GPU 2 queue with another checkpoint, as gpu2_queue_eval.resolve fills it."""
    return dict(q.CELLS[name], **changes)


def cpu(config):
    config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'


class Half(torch.nn.Module):
    """Stands in for the score network: any deterministic function of the state."""
    def forward(self, x, labels):
        return 0.5 * x


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        for path in ('logs', 'grids', 'acurve', 'table4', 'refs'):
            (self.root / path).mkdir()
        for key, arm in (('afhq_a', 'a'), ('afhq_vp', 'vp'), ('afhq_fox', 'fox')):
            (inputs / (key + '.json')).write_text(json.dumps(resolved_manifest(afhq.get_config(arm), 'protocol-' + arm)))
        (inputs / 'celeba_b.json').write_text(json.dumps(resolved_manifest(factorial(get_b), 'protocol-b')))
        a_manifest = json.dumps(resolved_manifest(factorial(get_a), 'protocol-a'))
        for key in ('celeba_a_first', 'celeba_a', followup.WORKDIRS['A']):
            (inputs / (key + '.json')).write_text(a_manifest)
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (g, 'ROOT', self.root), (g, 'LOGS', self.root / 'logs'),
                                      (g, 'GRIDS_DIR', self.root / 'grids'), (g, 'ACURVE', self.root / 'acurve'),
                                      (g, 'TABLE4', self.root / 'table4'), (g, 'REFS', self.root / 'refs'),
                                      (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def select(self, ckpt):
        (g.LOGS / g.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=ckpt)))

    def curve(self, fids=FIDS, status='completed', failed=(), verdict='matched', selected=None):
        cells = [dict(ckpt=c, cell='A_c{}_lo'.format(c), step=5000 * c + 1, fid_tweedie_xt=v) for c, v in fids.items()]
        if selected is None and fids:
            best = min(fids.values())
            selected = max(5000 * c for c, v in fids.items() if v <= best + 0.2 + 1e-9)
        (g.ACURVE / 'summary.json').write_text(json.dumps(dict(
            status=status, failed_cells=list(failed), cells=cells, total_cells=5, control=dict(verdict=verdict),
            step_selection=None if verdict == 'failed' else dict(selected=dict(A=dict(step=selected))))))


class Plan(Base):
    def test_grids_and_cells(self):
        self.assertEqual(g.KNOWN, ('celeba_vp', 'celeba_b', 'celeba_dial', 'afhq_vp', 'afhq_a', 'afhq_dial'))
        self.assertEqual(g.SHEETS['celeba'], ('celeba_vp', 'celeba_a', 'celeba_b', 'celeba_dial'))
        self.assertEqual(list(g.CELLS), ['celeba_a_cbest_m35_{}_seed{}'.format(m, s) for s in (0, 1) for m in (DPM, HEUN)])
        self.assertEqual(g.TABLE4_CELLS['celeba_a_cbest_m35_ode_edm_heun_seed1'], 'A_c80_lo_ode_edm_heun_seed1')

    def test_grid_models_steps_and_sources(self):
        seen = {name: (g.grid_spec(name)['source'], g.grid_spec(name)['step']) for name in g.KNOWN}
        self.assertEqual(seen, dict(celeba_vp=('celeba_vp', 200001), celeba_b=('celeba_b', 800001),
                                    celeba_dial=('celeba_fox', 800001), afhq_vp=('afhq_vp', 250001),
                                    afhq_a=('afhq_a', 250001), afhq_dial=('afhq_fox', 200001)))
        with self.assertRaisesRegex(ValueError, 'No step selected'):
            g.grid_spec('celeba_a')
        for ckpt, source in ((80, 'celeba_a_first'), (100, 'celeba_a'), (160, 'celeba_a')):
            self.select(ckpt)
            for spec in (g.grid_spec('celeba_a'), g.resolve('celeba_a_cbest_m35_ode_edm_heun_seed1')):
                self.assertEqual((spec['ckpt'], spec['source'], spec['work'], spec['step']), (ckpt, source, source, 5000 * ckpt + 1))
            self.assertEqual(q.training_protocol(source), 'protocol-a')

    def test_a_cells_equal_the_a_cells_of_table_4(self):
        self.select(80)
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = g.get_config('celeba_a_cbest_m35_{}_seed{}'.format(method, seed))
                theirs = lo.get_config('A_c80_lo_{}_seed{}'.format(method, seed))
                self.assertEqual(_canonicalize(mine), _canonicalize(theirs), (method, seed))
        self.select(120)
        mine = g.get_config('celeba_a_cbest_m35_ode_edm_heun_seed1')
        self.assertEqual(without_range(mine), without_range(lo.get_config('A_c80_lo_ode_edm_heun_seed1')))
        self.assertEqual((mine.eval.begin_ckpt, mine.eval.end_ckpt), (120, 120))

    def test_a_cells_equal_celeba_b_m35_of_the_gpu2_queue(self):
        self.select(140)
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = without_range(g.get_config('celeba_a_cbest_m35_{}_seed{}'.format(method, seed)))
                theirs = without_range(q.get_config('celeba_b_c160_m35_{}_seed{}'.format(method, seed)))
                for section in ('sampling', 'eval', 'data'):
                    self.assertEqual(mine[section], theirs[section], (method, seed, section))
                self.assertEqual((mine['eval']['sampling_num_scales'], mine['eval']['num_samples'],
                                  mine['eval']['batch_size'], mine['sampling']['logsnr_max']), (34, 50000, 512, 9.115429865459795))

    def test_grid_configs_equal_the_seed0_dpm_50k_cells(self):
        expected = dict(
            celeba_vp=lo.get_config('vp_c40_lo_ode_dpmpp_2m_seed0'),
            celeba_b=q.get_config('celeba_b_c160_m35_ode_dpmpp_2m_seed0'),
            celeba_dial=q.config_of(q_spec('celeba_b_c160_m35_ode_dpmpp_2m_seed0', source='celeba_fox')),
            afhq_dial=q.get_config('afhq_fox_c4_m35_ode_dpmpp_2m_seed0'),
            afhq_a=q.config_of(q_spec('afhq_a_cbest_m35_ode_dpmpp_2m_seed0', ckpt=5, source='afhq_a')),
            afhq_vp=q.config_of(q_spec('afhq_fox_c4_m35_ode_dpmpp_2m_seed0', ckpt=5, source='afhq_vp')))
        for name, theirs in expected.items():
            mine = g.config_of(g.grid_spec(name))
            self.assertEqual(_canonicalize(mine), _canonicalize(theirs), name)
            self.assertEqual((mine.sampling.method, mine.eval.sampling_seed, mine.eval.sampling_num_scales,
                              mine.eval.batch_size, mine.sampling.logsnr_min, mine.sampling.logsnr_max),
                             (DPM, 0, 34, 512, -10.0, 9.115429865459795), name)
        self.select(100)
        mine = g.config_of(g.grid_spec('celeba_a'))
        self.assertEqual(without_range(mine), without_range(g.get_config('celeba_a_cbest_m35_ode_dpmpp_2m_seed0')))

    def test_every_config_passes_the_dataset_check_of_the_evaluation(self):
        import endpoint_denoise_eval
        self.select(100)
        configs = [g.config_of(g.grid_spec(name)) for name in g.GRIDS] + [g.get_config(name) for name in g.CELLS]
        for config in configs:
            config.eval.batch_size = 0
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})


class Selection(Base):
    def test_rule_over_400k_to_800k(self):
        self.curve()
        result = g.a_selection()
        self.assertTrue(result['complete'], result)
        self.assertEqual((result['selected_step'], result['selected_ckpt'], result['source']), (500000, 100, 'celeba_a'))
        self.curve({**FIDS, 100: 2.9, 120: 2.95})
        result = g.a_selection()
        self.assertEqual((result['selected_step'], result['source']), (400000, 'celeba_a_first'))
        self.curve({**FIDS, 160: 2.777})  # Equality counts.
        self.assertEqual(g.a_selection()['selected_step'], 800000)

    def test_unclean_curve_selects_nothing(self):
        cases = (dict(status='running'), dict(failed=['A_c160_lo']), dict(verdict='failed'),
                 dict(selected=600000), dict(fids={c: v for c, v in FIDS.items() if c != 140}))
        for case in cases:
            self.curve(**case)
            result = g.a_selection()
            self.assertFalse(result['complete'], case)
            self.assertIn('reason', result)
        (g.ACURVE / 'summary.json').unlink()
        self.assertEqual(g.a_selection()['reason'], 'No curve summary.')

    def test_complete_selection_is_kept(self):
        self.curve()
        with patch.object(g, 'write_summary'):
            self.assertEqual(g.select_step(), 0)
            self.curve({**FIDS, 160: 2.0})
            self.assertEqual(g.select_step(), 0)
        self.assertEqual(json.loads((g.LOGS / g.SELECTION).read_text())['selected_step'], 500000)

    def test_incomplete_selection_is_recomputed(self):
        self.curve(status='running')
        with patch.object(g, 'write_summary') as summary:
            self.assertEqual(g.select_step(), 1)
            summary.assert_called_once_with(['selection'])
            self.curve()
            self.assertEqual(g.select_step(), 0)


class Grids(Base):
    def sample(self, names):
        def load(config, spec):
            return Half(), dict(checkpoint_id='id-' + spec['source'], step=spec['step'])

        original = g.config_of

        def on_cpu(spec):
            config = original(spec)
            cpu(config)
            return config

        with patch.object(g, 'config_of', on_cpu), patch.object(q, 'load_model', load):
            return [g.sample_grid(name) for name in names]

    def test_real_sampler_shared_noise_and_files(self):
        records = self.sample(('celeba_vp', 'celeba_b', 'afhq_dial'))
        seed = derive_seed(0, 'sampling-round', 0)
        torch.manual_seed(seed)
        white = torch.randn(512, 3, 64, 64)[:64].numpy()
        self.assertEqual({r['noise_sha256'] for r in records}, {__import__('hashlib').sha256(white.tobytes()).hexdigest()})
        for record in records:
            self.assertEqual((record['nfe'], record['round_seed'], record['batch'], record['shown'], record['method']),
                             (35, seed, 512, 64, DPM))
            with np.load(str(self.root / record['files']['npz'])) as data:
                for variant in g.VARIANTS:
                    self.assertEqual((data[variant].shape, data[variant].dtype), ((64, 64, 64, 3), np.uint8))
                np.testing.assert_array_equal(data['white_noise'], white)
                self.assertFalse(np.array_equal(data['x_t'], data['tweedie_xt']))
        from PIL import Image
        for path, size in ((self.root / records[0]['files']['x_t'], (530, 530)),
                           (g.write_sheet('celeba'), (76 + 2 * 8 * 66 + 12, 16 + 2 + 2 * 66)),
                           (g.write_sheet('afhqv2'), (76 + 2 * 8 * 66 + 12, 16 + 2 + 66))):
            with Image.open(str(path)) as image:
                self.assertEqual(image.size, size)

    def test_images_are_the_endpoint_conversion_of_the_same_sampler(self):
        import datasets
        import sampling
        from reproducibility import isolated_torch_rng
        from run_lib import get_sde
        record, = self.sample(('afhq_a',))
        spec = g.grid_spec('afhq_a')
        config = g.config_of(spec)
        cpu(config)
        sde, eps = get_sde(config, n_override=34)
        fn = sampling.get_sampling_fn(config, sde, (512, 3, 64, 64), datasets.get_data_inverse_scaler(config), eps)
        with isolated_torch_rng(record['round_seed']):
            outputs, _ = fn(Half())
        with np.load(str(self.root / record['files']['npz'])) as data:
            for variant in g.VARIANTS:
                expected = np.clip(outputs[variant].permute(0, 2, 3, 1).numpy() * 255., 0, 255).astype(np.uint8)[:64]
                np.testing.assert_array_equal(data[variant], expected)

    def test_comparison_with_the_50k_round(self):
        record, = self.sample(('afhq_vp',))
        with np.load(str(self.root / record['files']['npz'])) as data:
            pad = np.zeros((448, 64, 64, 3), np.uint8)
            samples = {'samples_' + v: np.concatenate([data[v], pad]) for v in g.VARIANTS}
        (g.REFS / 'afhq_vp').mkdir()
        np.savez(str(g.REFS / 'afhq_vp/samples_0.npz'), checkpoint_step=np.asarray(250001), round_id=np.asarray(0),
                 sampling_seed=np.asarray(record['round_seed']), **samples)
        result = g.compare('afhq_vp', record)
        self.assertTrue(result['identity_matches'])
        self.assertEqual((result['x_t']['max_abs_difference'], result['tweedie_xt']['equal_fraction']), (0, 1.0))
        np.savez(str(g.REFS / 'afhq_vp/samples_0.npz'), checkpoint_step=np.asarray(200001), round_id=np.asarray(0),
                 sampling_seed=np.asarray(1), **samples)
        self.assertFalse(g.compare('afhq_vp', record)['identity_matches'])
        self.assertIsNone(g.compare('celeba_dial', dict(record, ckpt=160)))
        g.write_summary()
        summary = json.loads((g.LOGS / 'summary.json').read_text())
        self.assertEqual(list(summary['grids']), ['afhq_vp'])
        self.assertTrue(summary['shared_noise']['afhqv2']['shared'])

    def test_a_reference_follows_the_selection(self):
        self.assertEqual(g.reference_path('celeba_a', dict(ckpt=80, source='celeba_a_first')),
                         g.REFS / 'celeba_a_400k/samples_0.npz')
        self.assertEqual(g.reference_path('celeba_a', dict(ckpt=120, source='celeba_a')),
                         self.root / 'workdirs/celeba_a/eval_gpu2a_celeba_a_cbest_m35_ode_dpmpp_2m_seed0/ckpt_120/samples_0.npz')

    def test_a_failed_grid_does_not_stop_the_next(self):
        order = []

        def sample(name):
            order.append(name)
            if name == 'celeba_b':
                raise RuntimeError('bad')
            g.grid_record_path(name).write_text(json.dumps(dict(noise_sha256='n', step=1)))
            return dict(label='x', dataset='d', step=1, noise_sha256='n')

        with patch.object(g, 'sample_grid', sample), patch.object(g, 'write_sheet'), \
                patch.object(g, 'compare', return_value=None):
            self.assertEqual(g.grids(), 1)
        self.assertEqual(order, ['celeba_vp', 'celeba_b', 'celeba_b', 'celeba_dial', 'afhq_vp', 'afhq_a', 'afhq_dial'])
        self.assertTrue(g.grid_failure_path('celeba_b').exists())


class Phase2(Base):
    def run_a(self, ckpt, blocked=None, grid_ok=True):
        self.select(ckpt)
        (g.LOGS / 'preflight_a.json').write_text(json.dumps(dict(failures=blocked or {})))
        order = []

        def evaluate(name):
            order.append(name)
            return True

        def grid(name):
            order.append('grid ' + name)
            return grid_ok

        with patch.object(g, 'evaluate', evaluate), patch.object(g, 'run_grid', grid), \
                patch.object(g, 'write_summary') as summary:
            code = g.a_queue()
        return code, order, summary

    def test_grid_then_four_cells(self):
        code, order, _ = self.run_a(140)
        self.assertEqual((code, order), (0, ['grid celeba_a'] + list(g.CELLS)))

    def test_failed_grid_does_not_block_the_cells(self):
        code, order, summary = self.run_a(140, grid_ok=False)
        self.assertEqual((code, order), (1, ['grid celeba_a'] + list(g.CELLS)))
        summary.assert_called_once_with(['grid celeba_a'])

    def test_failed_preflight_blocks_the_cells_not_the_grid(self):
        code, order, _ = self.run_a(140, blocked=dict(celeba_a_m35='bad'))
        self.assertEqual((code, order), (1, ['grid celeba_a']))

    def test_400k_reuses_table_4(self):
        code, order, _ = self.run_a(80)
        self.assertEqual((code, order), (0, ['grid celeba_a']))
        for theirs in g.TABLE4_CELLS.values():
            (g.TABLE4 / (theirs + '_result.json')).write_text(json.dumps(dict(cell=theirs, fid_tweedie_xt=1.7)))
        g.write_summary()
        summary = json.loads((g.LOGS / 'summary.json').read_text())
        self.assertEqual(len(summary['a50k_table4']['cells']), 4)
        self.assertEqual(summary['a50k_cells'], [])

    def test_preflight_a_at_400k_needs_no_weights(self):
        self.select(80)
        self.assertEqual(g.preflight_a(), 0)
        self.assertEqual(json.loads((g.LOGS / 'preflight_a.json').read_text())['failures'], {})


class Preflight(Base):
    def run_preflight(self, ckpt, smoke):
        def load(config, spec):
            return 'model', dict(checkpoint_id='id', step=spec['step'])

        original = g.config_of

        def on_cpu(spec):
            config = original(spec)
            cpu(config)
            return config

        with patch('torch.cuda.is_available', lambda: True), patch('torch.cuda.device_count', lambda: 1), \
                patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0'), patch.object(lo, 'check_inception'), \
                patch('evaluation.dataset_stats_fingerprint', lambda c: q.REFERENCE_SHA), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(g, 'config_of', on_cpu), patch.object(q, 'load_model', load), \
                patch.object(followup, 'smoke'), patch.object(q, 'endpoint_smoke', side_effect=smoke) as call:
            code = g.preflight(ckpt, 'preflight.json')
        return code, json.loads((g.LOGS / 'preflight.json').read_text()), call

    def test_both_solvers_on_h200_weights_of_a(self):
        code, saved, call = self.run_preflight(g.A_SMOKE, lambda config, model, spec, directory: 3.0)
        self.assertEqual((code, saved['status'], saved['source'], saved['step']), (0, 'passed', 'celeba_a', 500001))
        seen = [(args[2]['source'], args[2]['ckpt'], args[2]['method'], args[0].eval.sampling_num_scales,
                 args[0].sampling.time_grid, args[0].training.factorial_arm) for args, _ in call.call_args_list]
        self.assertEqual(seen, [('celeba_a', 100, DPM, 34, 'uniform_logsnr', 'A_vp_pvp'),
                                ('celeba_a', 100, HEUN, 34, 'karras', 'A_vp_pvp')])

    def test_failure_is_recorded(self):
        def smoke(config, model, spec, directory):
            raise ValueError('heun')

        code, saved, _ = self.run_preflight(120, smoke)
        self.assertEqual((code, sorted(saved['failures'])), (1, ['celeba_a_m35']))


class Report(Base):
    def prepare(self, name, step=None, features=True):
        spec = g.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / g.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range((spec['count'] + 511) // 512):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{q.FEATURES if features else 'other': np.zeros((512, 2048))})
        (directory / 'manifest.json').write_text(json.dumps(dict(config=without_range(g.config_of(spec)))))
        (g.LOGS / 'preflight_a.json').write_text(json.dumps(dict(dataset_stats_id=dict(CELEBA=q.REFERENCE_SHA))))
        (g.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=spec['count'], nfe=spec['calls'] + 1, checkpoint_step=step or spec['step'],
                      dataset_stats_id=q.REFERENCE_SHA, training_protocol_sha256='protocol-a', checkpoint_id='id')
        for metric in ('IS', 'fid', 'kid'):
            for variant in q.stage0.VARIANTS:
                values[metric + '_' + variant] = 2.0
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_report_and_summary(self):
        self.select(140)
        name = 'celeba_a_cbest_m35_ode_edm_heun_seed0'
        self.prepare(name)
        g.report(name)
        summary = json.loads((g.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['a50k_completed'], summary['a50k_total']), (1, 4))
        self.assertEqual(summary['a50k_cells'][0]['step'], 700001)
        self.assertEqual(len((g.LOGS / 'summary.csv').read_text().splitlines()), 2)

    def test_refusals(self):
        self.select(140)
        for name, changes, message in (('celeba_a_cbest_m35_ode_dpmpp_2m_seed0', dict(step=600001), 'report identity'),
                                       ('celeba_a_cbest_m35_ode_dpmpp_2m_seed1', dict(features=False), 'bootstrap')):
            directory = self.prepare(name, **changes)
            with self.assertRaisesRegex(RuntimeError, message):
                g.report(name)
            shutil.rmtree(str(directory))

    def test_400k_checks_the_table_4_checkpoint_id(self):
        self.select(80)
        spec = g.resolve('celeba_a_cbest_m35_ode_dpmpp_2m_seed0')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint id'):
            q.check_identity(spec, 400001, 'protocol-a', 'other')
        q.check_identity(spec, 400001, 'protocol-a', lo.MODELS['A']['checkpoint_id'])
        vp = g.grid_spec('celeba_vp')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint id'):
            q.check_identity(vp, 200001, None, 'other')
        q.check_identity(vp, 200001, None, lo.MODELS['vp']['checkpoint_id'])


DOCKER = r"""#!/usr/bin/env bash
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.polls; do [[ -e "$f" ]] || continue; n="$(basename "$f" .polls)"; (( $(cat "$f") > 0 )) && echo "$n"; done ;;
  inspect)
    [[ -e "$STATE/$last.polls" ]] || { echo; exit 1; }
    case "$*" in
      *State.Running*) n="$(cat "$STATE/$last.polls")"; if (( n > 0 )); then echo $((n - 1)) > "$STATE/$last.polls"; echo true; else echo false; fi ;;
      *ExitCode*) cat "$STATE/$last.code" ;;
      *DeviceIDs*) cat "$STATE/$last.gpus" ;;
      *.Name*) echo "/$last" ;;
    esac ;;
  run)
    name=""; prev=""; for arg in "$@"; do [[ "$prev" == --name ]] && name="$arg"; prev="$arg"; done
    [[ -z "$name" ]] && name=check
    echo "run $name" >> "$STATE/events"; printf '%s\n' "$@" > "$STATE/$name.args" ;;
esac
exit 0
"""
STUBS = {
    'git': '#!/usr/bin/env bash\n[[ "$1" == rev-parse ]] && echo commit\nexit 0\n',
    'nvidia-smi': ('#!/usr/bin/env bash\nif [[ "$*" == *name* ]]; then echo "${GPU_NAME:-NVIDIA RTX PRO 6000 Blackwell}"; '
                   'else echo "${FREE_MIB:-64000}"; fi\n'),
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
}
RUNS = {'celeba_vp': 'score_sde_pytorch/workdirs/celeba_vp_continuous_50k',
        'celeba_fox': 'score_sde_pytorch/workdirs/celeba_fox_matern32_kappa1000_50k',
        'celeba_b': 'celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42',
        'celeba_a_first': 'celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42',
        'celeba_a': 'celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42',
        'afhq_vp': 'afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42',
        'afhq_a': 'afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42',
        'afhq_fox': 'afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42'}
CKPTS = {'celeba_vp': 40, 'celeba_fox': 160, 'celeba_b': 160, 'celeba_a': 100, 'afhq_vp': 5, 'afhq_a': 5, 'afhq_fox': 4}
REFS = {'celeba_vp': 'celeba_50k_lo_20261107/gpu1/workdirs/celeba_vp_continuous_50k/eval_50k_lo_vp_c40_lo_ode_dpmpp_2m_seed0/ckpt_40',
        'celeba_a_400k': 'celeba_50k_lo_20261107/gpu1/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42/'
                         'eval_50k_lo_A_c80_lo_ode_dpmpp_2m_seed0/ckpt_80',
        'celeba_b': 'gpu2_queue_20261114/workdirs/celeba_b/eval_gpu2q_celeba_b_c160_m35_ode_dpmpp_2m_seed0/ckpt_160',
        'afhq_dial': 'gpu2_queue_20261114/workdirs/afhq_fox/eval_gpu2q_afhq_fox_c4_m35_ode_dpmpp_2m_seed0/ckpt_4',
        'afhq_a': 'gpu2_queue_20261114/workdirs/afhq_a/eval_gpu2q_afhq_a_cbest_m35_ode_dpmpp_2m_seed0/ckpt_5',
        'afhq_vp': 'gpu1_vp_tail_20261114/workdirs/afhq_vp/eval_gpu1vp_afhq_vp_cbest_m35_ode_dpmpp_2m_seed0/ckpt_5'}
ROOT_IN = '/workspace/score_sde_pytorch'


class Launcher(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.bin, self.state, self.base = self.root / 'bin', self.root / 'state', self.root / 'base'
        for path in (self.bin, self.state, self.root / 'worktree'):
            path.mkdir()
        for name, text in dict(STUBS, docker=DOCKER).items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        shutil.copy(str(REPO / 'run_gpu2_a50k_grids_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / lo.INCEPTION_MODULE
        self.curve_logs = self.base / 'celeba_a_800k_h200_20261113/eval/logs'
        table4 = self.base / 'celeba_50k_lo_20261107/gpu1/logs'
        for path in (main / 'assets/stats', main / 'celeba', tfhub, self.curve_logs, table4):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt',
                     tfhub / 'saved_model.pb', table4 / 'A_c80_lo_ode_dpmpp_2m_seed0_result.json',
                     self.curve_logs / 'summary.json'):
            path.write_text('x')
        for work, run in RUNS.items():
            (self.base / run / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (self.base / run / 'run_manifest.json').write_text('manifest of ' + work)
            if work in CKPTS:
                (self.base / run / 'checkpoints/checkpoint_{}.pth'.format(CKPTS[work])).write_text('x')
        for ref, path in REFS.items():
            (self.base / path).mkdir(parents=True)
            (self.base / path / 'samples_0.npz').write_text('x')
        self.container('celeba-a-curve-h200', 2, 0, '3,')
        self.container('celeba-a-800k-h200', 99, 0, '3,')

    def container(self, name, polls, code, gpus):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))
        (self.state / (name + '.gpus')).write_text(gpus)

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           AGRID_SERVER_BASE=str(self.base), AGRID_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_gpu2_a50k_grids_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def args(self, name):
        return (self.state / (name + '.args')).read_text()

    def test_start_runs_the_grids_with_read_only_sources(self):
        result = self.launch('start', '2')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run celeba-gpu2-grids'])
        args = self.args('celeba-gpu2-grids')
        self.assertIn('device=2', args)
        for work, run in RUNS.items():
            self.assertIn('{}/checkpoints,dst={}/workdirs/{}/checkpoints,readonly'.format(run, ROOT_IN, work), args)
        for ref, path in REFS.items():
            self.assertIn('{},dst={}/refs/{},readonly'.format(path, ROOT_IN, ref), args)
        self.assertIn('eval/logs,dst={}/acurve,readonly'.format(ROOT_IN), args)
        self.assertIn('gpu1/logs,dst={}/table4,readonly'.format(ROOT_IN), args)
        self.assertIn('gpu2_a50k_grids.py preflight; python -u gpu2_a50k_grids.py grids', args)
        inputs = self.base / 'gpu2_a50k_grids_20261114/workdirs/_training_inputs'
        self.assertEqual(sorted(p.name for p in inputs.iterdir()),
                         ['afhq_a.json', 'afhq_fox.json', 'afhq_vp.json', 'celeba_a.json', 'celeba_a_first.json', 'celeba_b.json'])

    def test_waits_for_the_curve_and_the_grids_then_starts_a(self):
        self.container('celeba-gpu2-grids', 3, 0, '2,')
        result = self.launch('_wait', '2')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run celeba-gpu2-a50k'])
        args = self.args('celeba-gpu2-a50k')
        self.assertIn('device=2', args)
        self.assertIn('gpu2_a50k_grids.py select && { python -u gpu2_a50k_grids.py preflight-a;', args)
        self.assertNotIn('celeba-a-curve-h200', args)

    def test_unclean_curve_starts_nothing(self):
        self.container('celeba-a-curve-h200', 0, 3, '3,')
        self.assertIn(b'exited with 3; nothing started', self.launch('_wait', '2').stdout)
        self.container('celeba-a-curve-h200', 0, 0, '3,')
        (self.curve_logs / 'summary.json').unlink()
        self.assertIn(b'Missing input', self.launch('start-a', '2').stdout)
        self.container('celeba-a-curve-h200', 99, 0, '3,')
        self.assertIn(b'still runs; nothing started', self.launch('_wait', '2', AGRID_WAIT_MAX_SECONDS='0').stdout)
        self.assertIn(b'is true; the A step needs its finished curve', self.launch('start-a', '2').stdout)
        (self.state / 'celeba-a-curve-h200.polls').unlink()
        self.assertIn(b'is gone', self.launch('start-a', '2').stdout)
        self.assertEqual(self.events(), [])

    def test_refusals(self):
        self.assertEqual(self.launch('start', '3', GPU_NAME='NVIDIA H200 NVL').returncode, 1)
        self.assertEqual(self.launch('start', '1', '2').returncode, 2)
        self.container('celeba-a-curve-h200', 0, 0, '3,')
        self.container('celeba-gpu2-grids', 99, 0, '2,')
        self.assertIn(b'runs containers: celeba-gpu2-grids', self.launch('start-a', '2').stdout)
        self.assertIn(b'Container exists: celeba-gpu2-grids', self.launch('start', '2').stdout)
        (self.state / 'celeba-gpu2-grids.polls').unlink()
        self.assertIn(b'need 60000', self.launch('start', '2', FREE_MIB='50000').stdout)
        self.assertEqual(self.events(), [])

    def test_after_a_returns_at_once_and_allows_the_grids(self):
        self.container('celeba-gpu2-grids', 1, 0, '2,')
        result = self.launch('after-a', '2')
        self.assertIn(b'Waiting for celeba-a-curve-h200', result.stdout)
        for _ in range(100):
            if self.events():
                break
            time.sleep(0.1)
        self.assertEqual(self.events(), ['run celeba-gpu2-a50k'])


if __name__ == '__main__':
    unittest.main()
