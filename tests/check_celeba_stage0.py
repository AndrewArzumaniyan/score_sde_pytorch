"""CPU checks for CelebA stage 0; real SDEs and network, no checkpoints.

STAGE0_TEST_MANIFESTS may point to copies of the real run_manifest.json files.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if importlib.util.find_spec('tensorflow') is None:  # Outside the Docker image.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())

import celeba_final_ode_eval as final
import celeba_ode_eval as controlled
import celeba_stage0_eval as stage0
import ode_sampling
import run_lib
import sampling
import sde_lib
from artifact_utils import _canonicalize, build_model_manifest
from models import utils as mutils, ncsnpp  # noqa: F401
from splice_model import SplicedEpsModel


def write_manifests(target):
    source = os.environ.get('STAGE0_TEST_MANIFESTS')
    if source:
        for work in stage0.WORKDIRS.values():
            if work not in (stage0.VP, stage0.FOX):
                shutil.copy2(str(Path(source) / (work + '.json')), str(target / (work + '.json')))
        return
    from configs.factorial import (celeba_a_vp_pvp as a, celeba_b_vp_pfox as b, celeba_c_normfox_pvp as c,
                                   celeba_d_normfox_pfox as d, celeba_e_realfox_pfox as e)
    for model, module in dict(A=a, B=b, C=c, D=d, E=e, F=a, G=d).items():
        config = module.get_config()
        if model in 'FG':
            config.model.noise_conditioning = 'time'
        (target / (stage0.WORKDIRS[model] + '.json')).write_text(
            json.dumps({'config': config.to_dict()}, default=str))


class GaussianEps(torch.nn.Module):
    """Exact noise prediction for N(mean, variance) data under the given process."""

    def __init__(self, process, mean=.3, variance=.25):
        super().__init__()
        self.process, self.mean, self.variance = process, mean, variance

    def forward(self, x, labels):
        if getattr(self.process, 'noise_conditioning', 'time') == 'logsnr':
            low, high = self.process.conditioning_logsnr_min, self.process.conditioning_logsnr_max
            t = self.process.time_from_log_snr(high - labels / 999 * (high - low))
        else:
            t = labels / 999
        one = torch.ones_like(x[:, :1, :1, :1])
        alpha, sigma = self.process.marginal_prob(one, t)
        sigma = sigma[:, None, None, None]
        total = alpha.square() * self.variance + sigma.square()
        return sigma * (x - alpha * self.mean) / total


def exact_moments(process, t, mean=.3, variance=.25):
    one = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    alpha, sigma = process.marginal_prob(one, torch.tensor([t], dtype=torch.float64))
    return float(alpha) * mean, float(alpha) ** 2 * variance + float(sigma) ** 2


class Stage0Case(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'inputs').mkdir()
        (self.root / 'logs').mkdir()
        write_manifests(self.root / 'inputs')
        for patcher in (patch.object(stage0, 'TRAINING_INPUTS', self.root / 'inputs'),
                        patch.object(stage0, 'LOGS', self.root / 'logs'),
                        patch.object(stage0, 'ROOT', self.root),
                        patch.object(stage0, 'DEVICE', 'cpu')):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)


class Matrix(unittest.TestCase):
    def test_cells_and_order(self):
        names = list(stage0.CELLS)
        self.assertEqual(len(names), 33)
        self.assertEqual(names[:2], list(stage0.GATE))
        self.assertEqual(names[2:6], ['A_c40_lo', 'B_c40_lo', 'vp_c40_lo', 'fox_c40_lo'])
        self.assertEqual(names[-1], 'B_c20_lo')
        self.assertEqual(stage0.ENERGY_CELLS, tuple(names[:12]))
        self.assertNotIn('B_c20_lo', stage0.ENERGY_CELLS)
        for model in 'ABCDEFG':
            self.assertIn(model + '_c10_lo', names)

    def test_contrasts_name_existing_cells(self):
        for _, left, right, _ in stage0.CONTRASTS:
            self.assertIn(left, stage0.CELLS)
            self.assertIn(right, stage0.CELLS)

    def test_trained_on_lo_never_on_hi(self):
        for spec in stage0.CELLS.values():
            if spec['model'] in stage0.TRAINED_ON_LO:
                self.assertEqual(spec['window'], 'lo')


class Configs(Stage0Case):
    def sections(self, config):
        result = {s: _canonicalize(config[s]) for s in ('model', 'data', 'sampling', 'eval')}
        for key in ('begin_ckpt', 'end_ckpt', 'sampling_num_scales'):
            result['eval'].pop(key)
        return result

    def test_native_cells_match_the_final_ode_protocol(self):
        for arm in ('vp', 'fox'):
            mine = stage0.get_config(arm + '_c260_hi')
            theirs = final.get_config('ode_dpmpp_2m_10000_' + arm)
            self.assertEqual(self.sections(mine), self.sections(theirs))
            self.assertEqual(mine.eval.sampling_num_scales, 64)
            self.assertEqual((mine.eval.begin_ckpt, mine.eval.end_ckpt), (260, 260))
            for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
                self.assertEqual(mine[key], theirs[key])

    def test_A_B_cells_match_the_controlled_ode_protocol(self):
        (self.root / 'training_inputs').mkdir()
        for arm in 'AB':
            shutil.copy2(str(self.root / 'inputs' / (stage0.WORKDIRS[arm] + '.json')),
                         str(self.root / 'training_inputs/42_{}.json'.format(arm)))
        with patch.object(controlled, 'ROOT', self.root):
            for arm in 'AB':
                theirs = controlled.get_config('ode_dpmpp_2m_256_42_' + arm)
                self.assertEqual(self.sections(stage0.get_config(arm + '_c40_lo')), self.sections(theirs))

    def test_hi_window_refused_for_trained_on_lo(self):
        stage0.CELLS['A_c40_hi'] = dict(model='A', ckpt=40, window='hi', route=None)
        self.addCleanup(stage0.CELLS.pop, 'A_c40_hi')
        with self.assertRaises(ValueError):
            stage0.get_config('A_c40_hi')

    def test_expected_factors_in_each_model(self):
        expected = dict(A=('vpsde', 'logsnr', 'vp_time'), B=('vpsde', 'logsnr', 'normalized_fox_time'),
                        C=('foxvpsde', 'logsnr', 'vp_time'), D=('foxvpsde', 'logsnr', 'normalized_fox_time'),
                        E=('foxvpsde', 'logsnr', 'normalized_fox_time'), F=('vpsde', 'time', 'vp_time'),
                        G=('foxvpsde', 'time', 'normalized_fox_time'))
        for model, (sde, label, allocation) in expected.items():
            config = stage0.get_config(model + '_c10_lo')
            self.assertEqual(config.training.sde, sde)
            self.assertEqual(config.model.noise_conditioning, label)
            self.assertEqual(config.training.noise_distribution, allocation)
        self.assertTrue(stage0.get_config('C_c10_lo').model.fox_normalize_scale)
        self.assertTrue(stage0.get_config('D_c10_lo').model.fox_normalize_scale)
        self.assertFalse(stage0.get_config('E_c10_lo').model.fox_normalize_scale)
        self.assertTrue(stage0.get_config('G_c10_lo').model.fox_normalize_scale)

    def test_every_grid_spans_its_window(self):
        for name, spec in stage0.CELLS.items():
            config = stage0.get_config(name)
            process, eps = run_lib.get_sde(config, n_override=64)
            grid, levels = stage0.check_grid(config, process, eps)
            self.assertEqual(len(grid), 65)
            self.assertAlmostEqual(float(levels[-1]), stage0.WINDOWS[spec['window']], delta=1e-3)
            if getattr(config.model, 'noise_conditioning', 'time') == 'logsnr':
                labels = mutils.logsnr_to_label(levels, config.model.logsnr_min, config.model.logsnr_max)
                self.assertAlmostEqual(float(labels[0]), 999, delta=.01)
                self.assertAlmostEqual(float(labels[-1]), 0, delta=.01)

    def test_every_config_serializes_for_the_eval_manifest(self):
        for name in stage0.CELLS:
            json.dumps(_canonicalize(stage0.get_config(name)), sort_keys=True)

    def test_model_hash_matches_the_training_manifest(self):
        for model in 'ABCDEFG':
            saved = json.loads((stage0.TRAINING_INPUTS / (stage0.WORKDIRS[model] + '.json')).read_text())
            if 'model_manifest' not in saved:
                self.skipTest('Generated manifests have no recorded hash.')
            config = stage0.get_config(model + '_c10_lo')
            self.assertEqual(build_model_manifest(config)['model_protocol_sha256'],
                             saved['model_manifest']['model_protocol_sha256'], model)

    def test_grid_outside_support_refused(self):
        config = stage0.get_config('A_c10_lo')
        process, eps = run_lib.get_sde(config, n_override=64)
        config.sampling.logsnr_max = 10.5
        process.sampling_logsnr_max = 10.5
        with self.assertRaises(RuntimeError):
            stage0.check_grid(config, process, eps)


class Solver(Stage0Case):
    def run_exact(self, name):
        config = stage0.get_config(name)
        process, eps = run_lib.get_sde(config, n_override=128)
        shape = (20000, 1, 1, 1)
        torch.manual_seed(0)
        sampler = ode_sampling.get_vp_ode_sampler(process, shape, lambda x: x, stage0.METHOD, eps, 'cpu')
        outputs, calls = sampler(GaussianEps(process))
        self.assertEqual(calls, 129)
        end = float(process.sampling_time_grid(eps, grid='uniform_logsnr', device='cpu',
                                               dtype=torch.float64, N=129)[-1])
        mean, variance = exact_moments(process, end)
        x = outputs['x_t'].double()
        self.assertAlmostEqual(float(x.mean()), mean, delta=.02, msg=name)
        self.assertAlmostEqual(float(x.var()), variance, delta=.02, msg=name)

    def test_exact_gaussian_flow_for_every_model_type(self):
        for name in ('vp_c260_hi', 'fox_c260_hi', 'vp_c260_lo', 'fox_c260_lo', 'A_c10_lo', 'B_c10_lo',
                     'C_c10_lo', 'D_c10_lo', 'E_c10_lo', 'F_c10_lo', 'G_c10_lo'):
            self.run_exact(name)

    def spliced(self, name, budget=32):
        config = stage0.get_config(name)
        vp, eps = run_lib.get_sde(config, n_override=budget)
        alternate = config.eval.splice_alt
        fox, _ = run_lib.get_sde(alternate)
        model = SplicedEpsModel(GaussianEps(vp), GaussianEps(fox), vp, fox, mode=config.eval.splice_mode,
                                threshold=config.eval.splice_lambda, sampling_eps=eps,
                                time_grid=config.sampling.time_grid, device='cpu').eval()
        return config, vp, fox, eps, model

    def test_splice_route_matches_the_ode_calls(self):
        for name in ('splice_head_c260_lo', 'splice_tail_c260_lo', 'splice_full_c260_lo'):
            _, vp, _, eps, model = self.spliced(name)
            sampler = ode_sampling.get_vp_ode_sampler(vp, (8, 1, 1, 1), lambda x: x, stage0.METHOD, eps, 'cpu')
            _, calls = sampler(model)
            self.assertEqual(calls, 33)
            metadata = model.checked_route_metadata()
            self.assertEqual(metadata['splice_vp_calls'] + metadata['splice_fox_calls'], 33)
            if name == 'splice_head_c260_lo':
                self.assertGreater(metadata['splice_fox_calls'], 0)
                self.assertFalse(model.route_history[-1])
            if name == 'splice_tail_c260_lo':
                self.assertTrue(model.route_history[-1])

    def test_full_route_equals_native_fox_with_exact_scores(self):
        _, vp, _, eps, model = self.spliced('splice_full_c260_lo', budget=64)
        config = stage0.get_config('fox_c260_lo')
        fox, fox_eps = run_lib.get_sde(config, n_override=64)
        torch.manual_seed(1)
        z = torch.randn(4096, 1, 1, 1)
        first = ode_sampling.get_vp_ode_sampler(vp, z.shape, lambda x: x, stage0.METHOD, eps, 'cpu')(model, z=z)[0]
        t0 = fox.sampling_time_grid(fox_eps, grid='uniform_logsnr', device='cpu', dtype=torch.float64, N=65)[0:1]
        alpha, sigma = fox.marginal_prob(torch.ones(1, 1, 1, 1, dtype=torch.float64), t0)
        scale = float(torch.sqrt(alpha.square().flatten() + sigma.square()))
        second = ode_sampling.get_vp_ode_sampler(fox, z.shape, lambda x: x, stage0.METHOD, fox_eps, 'cpu')(
            GaussianEps(fox), z=z * scale)[0]
        difference = (first['tweedie_xt'] - second['tweedie_xt']).abs().max()
        self.assertLess(float(difference), 2e-2)

    def test_dispatch_needs_the_flag_and_dpmpp(self):
        config = stage0.get_config('splice_head_c260_lo')
        process, eps = run_lib.get_sde(config, n_override=4)
        self.assertTrue(callable(sampling.get_sampling_fn(config, process, (1, 3, 64, 64), lambda x: x, eps)))
        config.eval.splice_allow_ode = False
        with self.assertRaises(ValueError):
            sampling.get_sampling_fn(config, process, (1, 3, 64, 64), lambda x: x, eps)
        config.eval.splice_allow_ode = True
        config.sampling.method = 'ode_heun'
        with self.assertRaises(ValueError):
            sampling.get_sampling_fn(config, process, (1, 3, 64, 64), lambda x: x, eps)

    def test_real_network_through_the_project_dispatch(self):
        for name in ('vp_c260_hi', 'fox_c260_hi', 'A_c10_lo', 'E_c10_lo', 'G_c10_lo'):
            config = stage0.get_config(name)
            process, eps = run_lib.get_sde(config, n_override=2)
            model = mutils.create_model(config).eval()
            fn = sampling.get_sampling_fn(config, process, (1, 3, 64, 64), lambda x: x, eps)
            outputs, calls = fn(model)
            self.assertEqual(calls, 3)
            self.assertEqual(set(outputs), set(stage0.VARIANTS))
            self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()), name)

    def test_energy_rows_and_probe_keep_outputs(self):
        config = stage0.get_config('fox_c260_lo')
        process, eps = run_lib.get_sde(config, n_override=8)
        network = GaussianEps(process)
        z = torch.randn(64, 1, 1, 1)
        sampler = ode_sampling.get_vp_ode_sampler(process, z.shape, lambda x: x, stage0.METHOD, eps, 'cpu')
        plain = sampler(network, z=z.clone())[0]
        probe = stage0.energy_probe(network)
        probed = sampler(probe, z=z.clone())[0]
        torch.testing.assert_close(plain['tweedie_xt'], probed['tweedie_xt'], rtol=0, atol=0)
        rows = stage0.energy_rows(process, config, eps, probe.rows)
        self.assertEqual(len(rows), 9)
        self.assertAlmostEqual(rows[0]['lam'], -10, delta=1e-3)
        self.assertAlmostEqual(rows[-1]['lam'], stage0.WINDOWS['lo'], delta=1e-3)
        self.assertLess(rows[4]['scale_sq'], 1)
        self.assertAlmostEqual(rows[0]['normalized_state_sq'], 1, delta=.5)


class Intake(Stage0Case):
    def fake_cell(self, name, fid=3.0):
        spec = stage0.CELLS[name]
        directory = stage0.cell_directory(name)
        shards = directory / 'ckpt_{}'.format(spec['ckpt'])
        shards.mkdir(parents=True)
        for index in range(20):
            (shards / 'statistics_{}.npz'.format(index)).write_bytes(b'x')
        config = stage0.get_config(name)
        recorded = {section: _canonicalize(config[section]) for section in ('model', 'data', 'sampling', 'eval')}
        recorded['eval'].pop('begin_ckpt')  # build_eval_manifest leaves these two out.
        recorded['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(source_sha256='src', config=recorded)))
        values = {metric + '_' + variant: np.asarray(fid) for metric in ('IS', 'fid', 'kid')
                  for variant in stage0.VARIANTS}
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), num_samples=np.asarray(10000), nfe=np.asarray(65),
                 checkpoint_step=np.asarray(spec['ckpt'] * 5000 + 1), checkpoint_id=np.asarray('id'),
                 dataset_stats_id=np.asarray(stage0.REFERENCE_SHA), **values)
        (stage0.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=1.0)))

    def preflight(self):
        keys = {'{}_{}'.format(s['model'], s['ckpt']): dict(checkpoint_id='id', step=s['ckpt'] * 5000 + 1)
                for s in stage0.CELLS.values()}
        (stage0.LOGS / 'preflight.json').write_text(json.dumps(dict(source_sha256='src', checkpoints=keys)))

    def test_report_and_contrasts(self):
        self.preflight()
        self.fake_cell('B_c40_lo', 9.0)
        self.fake_cell('A_c40_lo', 3.0)
        stage0.report('B_c40_lo')
        stage0.report('A_c40_lo')
        summary = json.loads((stage0.LOGS / 'summary.json').read_text())
        self.assertEqual(summary['completed_cells'], 2)
        self.assertEqual(summary['status'], 'running')
        contrast = {c['contrast']: c['value'] for c in summary['point_contrasts']}
        self.assertEqual(contrast['B_vs_A_200k_lo'], 6.0)
        self.assertIn('A_c40_lo', (stage0.LOGS / 'summary.csv').read_text())

    def test_report_rejects_other_sources_and_shards(self):
        self.preflight()
        self.fake_cell('A_c40_lo')
        directory = stage0.cell_directory('A_c40_lo')
        (directory / 'ckpt_40/statistics_0.npz').unlink()
        with self.assertRaisesRegex(RuntimeError, 'statistics shards'):
            stage0.report('A_c40_lo')
        (directory / 'ckpt_40/statistics_0.npz').write_bytes(b'x')
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['source_sha256'] = 'other'
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, 'different sources'):
            stage0.report('A_c40_lo')

    def test_report_rejects_changed_protocol(self):
        self.preflight()
        self.fake_cell('A_c40_lo')
        path = stage0.cell_directory('A_c40_lo') / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['config']['sampling']['logsnr_max'] = 10.0
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, 'report config: sampling'):
            stage0.report('A_c40_lo')

    def test_report_rejects_wrong_weights(self):
        self.preflight()
        self.fake_cell('A_c40_lo')
        saved = json.loads((stage0.LOGS / 'preflight.json').read_text())
        saved['checkpoints']['A_40']['checkpoint_id'] = 'other'
        (stage0.LOGS / 'preflight.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, 'identity: checkpoint_id'):
            stage0.report('A_c40_lo')

    def fake_queue(self, fid, fail=()):
        def run(name):
            if name in fail:
                raise subprocess.CalledProcessError(1, 'main.py')

        def report(name):
            value = stage0.GATE.get(name, 3.0) + (fid if name in stage0.GATE else 0)
            stage0.result_path(name).write_text(json.dumps(dict(fid_tweedie_xt=value)))
        return patch.object(stage0, 'run', run), patch.object(stage0, 'report', report)

    def test_queue_stops_when_the_gate_fails(self):
        first, second = self.fake_queue(.5)
        with first, second, self.assertRaises(SystemExit):
            stage0.queue()
        self.assertFalse(stage0.result_path('A_c40_lo').exists())

    def test_queue_stops_when_a_gate_cell_crashes(self):
        first, second = self.fake_queue(0, fail=('vp_c260_hi',))
        with first, second, self.assertRaises(SystemExit):
            stage0.queue()

    def test_queue_continues_after_a_failed_cell_and_resumes(self):
        first, second = self.fake_queue(.001, fail=('F_c10_lo',))
        with first, second:
            self.assertEqual(stage0.queue(), 1)
        self.assertTrue((stage0.LOGS / 'F_c10_lo_failed.json').exists())
        self.assertTrue(stage0.result_path('splice_full_c260_lo').exists())
        seen = []
        with patch.object(stage0, 'run', seen.append), patch.object(stage0, 'report', lambda name:
                stage0.result_path(name).write_text(json.dumps(dict(fid_tweedie_xt=3.0)))):
            self.assertEqual(stage0.queue(), 0)
        self.assertEqual(seen, ['F_c10_lo'])
        self.assertFalse((stage0.LOGS / 'F_c10_lo_failed.json').exists())

    def test_queue_continues_after_a_value_error(self):
        first, second = self.fake_queue(.001)
        with first, second:
            write_result = stage0.report

            def report(name):
                if name == 'F_c10_lo':
                    raise ValueError('The eval folder belongs to a different protocol.')
                write_result(name)

            with patch.object(stage0, 'report', report):
                self.assertEqual(stage0.queue(), 1)
        self.assertTrue((stage0.LOGS / 'F_c10_lo_failed.json').exists())
        self.assertTrue(stage0.result_path('B_c20_lo').exists())

    def test_write_json_keeps_the_old_file_when_interrupted(self):
        path = stage0.LOGS / 'atomic.json'
        stage0.write_json(path, dict(value=1))
        with patch.object(stage0.os, 'replace', side_effect=OSError('killed')):
            with self.assertRaises(OSError):
                stage0.write_json(path, dict(value=2))
        self.assertEqual(json.loads(path.read_text()), dict(value=1))


class Launcher(unittest.TestCase):
    def test_syntax_and_read_only_inputs(self):
        path = REPO / 'run_celeba_stage0_rtx.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)
        text = path.read_text()
        for work in stage0.WORKDIRS.values():
            self.assertIn(work, text)
        self.assertIn('dst=$root/workdirs/$work/checkpoints,readonly', text)
        self.assertIn('Run from a separate worktree', text)


if __name__ == '__main__':
    unittest.main(verbosity=1)
