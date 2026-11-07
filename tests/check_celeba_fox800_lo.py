"""CPU checks for the FOX 800k cells at 50k; real SDEs and samplers, no checkpoints."""
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

import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_fox800_lo_eval as fox
import celeba_stage0_eval as stage0
import run_lib
import sampling
from artifact_utils import _canonicalize

LO = 9.115429865459795
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
F0 = fox.cell_name(DPM, 0)
F1 = fox.cell_name(HEUN, 0)


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def on_cpu(config):
    config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
    return config


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / 'logs').mkdir()
        for target, value in (('ROOT', self.root), ('LOGS', self.root / 'logs')):
            patcher = patch.object(fox, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class Plan(Base):
    def test_four_cells_in_the_order_of_the_50k_queue(self):
        self.assertEqual(list(fox.CELLS), ['fox_c160_lo_{}_seed{}'.format(method, seed)
                                           for seed in (0, 1) for method in (DPM, HEUN)])

    def test_equals_the_1300k_cell_except_the_checkpoint(self):
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = _canonicalize(fox.get_config(fox.cell_name(method, seed)))
                theirs = _canonicalize(lo.get_config(lo.cell_name('fox', method, seed)))
                self.assertEqual((mine['eval'].pop('begin_ckpt'), mine['eval'].pop('end_ckpt')), (160, 160))
                self.assertEqual((theirs['eval'].pop('begin_ckpt'), theirs['eval'].pop('end_ckpt')), (260, 260))
                self.assertEqual(mine, theirs)

    def test_explicit_values(self):
        for name, spec in fox.CELLS.items():
            config = fox.get_config(name)
            self.assertEqual((config.sampling.logsnr_min, config.sampling.logsnr_max), (-10.0, LO))
            self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.batch_size),
                             (34, 50000, 512))
            self.assertEqual(config.eval.sampling_seed, spec['seed'])
            self.assertEqual(config.sampling.time_grid, 'karras' if spec['method'] == HEUN else 'uniform_logsnr')
            self.assertTrue(config.eval.save_endpoint_states)
            self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), ('x_t', 'tweedie_xt'))

    def test_every_cell_grid_spans_the_window_and_its_sampler_builds(self):
        for name, spec in fox.CELLS.items():
            config = on_cpu(fox.get_config(name))
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)

    def test_full_sampler_calls_and_outputs(self):
        for method in (DPM, HEUN):
            config = on_cpu(fox.get_config(fox.cell_name(method, 0)))
            sde, eps = run_lib.get_sde(config, n_override=34)
            outputs, nfe = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)(ZeroEps())
            self.assertEqual(nfe, 35)
            self.assertEqual(set(outputs), {'x_t', 'tweedie_xt'})


class Loading(Base):
    def identity(self, **changes):
        return 'm', dict(dict(step=800001, checkpoint_id=fox.CHECKPOINT_ID), **changes)

    def test_exact_identity_passes(self):
        with patch.object(stage0, 'load_model', return_value=self.identity()) as load:
            self.assertEqual(fox.load_model('c')[0], 'm')
        load.assert_called_once_with('c', 'celeba_fox_matern32_kappa1000_50k', 160)

    def test_wrong_step_or_id_is_refused(self):
        for changes, message in ((dict(step=800002), 'Expected step'), (dict(checkpoint_id='other'), 'checkpoint id')):
            with patch.object(stage0, 'load_model', return_value=self.identity(**changes)):
                with self.assertRaisesRegex(RuntimeError, message):
                    fox.load_model('c')


class Preflight(Base):
    def run_preflight(self, load, cache=True):
        module = self.root / 'cache' / lo.INCEPTION_MODULE
        (module / 'variables').mkdir(parents=True, exist_ok=True)
        if cache:
            (module / 'saved_model.pb').write_text('x')
        self.loaded = []

        def counted(config):
            self.loaded.append(1)
            return load(config)

        original = fox.get_config
        with patch.object(fox, 'load_model', counted), \
                patch.object(fox, 'get_config', lambda name: on_cpu(original(name))), \
                patch.object(followup, 'smoke') as self.smoke, patch('torch.cuda.is_available', lambda: True), \
                patch('torch.cuda.device_count', lambda: 1), patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0', TFHUB_CACHE_DIR=str(self.root / 'cache')), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(sampling, 'get_sampling_fn'):
            return fox.preflight()

    @staticmethod
    def good(config):
        return ZeroEps(), dict(step=fox.STEP, checkpoint_id=fox.CHECKPOINT_ID)

    def saved(self):
        return json.loads((fox.LOGS / 'preflight.json').read_text())

    def test_loads_once_and_smokes_every_cell(self):
        self.assertEqual(self.run_preflight(self.good), 0)
        self.assertEqual((len(self.loaded), self.smoke.call_count), (1, 4))
        self.assertEqual(self.saved()['checkpoints']['fox']['step'], 800001)

    def test_truncated_checkpoint_blocks_the_cells(self):
        def load(config):
            raise EOFError('truncated')

        self.assertEqual(self.run_preflight(load), 1)
        self.assertEqual(len(self.saved()['failures']), 4)

    def test_no_local_inception_stops_the_queue_and_clears_the_old_preflight(self):
        (fox.LOGS / 'preflight.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'No local Inception'):
            self.run_preflight(self.good, cache=False)
        self.assertFalse((fox.LOGS / 'preflight.json').exists())
        self.assertEqual(fox.queue(), 1)


class Report(Base):
    def prepare(self, name, nfe=None, step=None, shards=98, features=True):
        config = fox.get_config(name)
        directory = self.root / 'workdirs' / fox.WORK / fox.eval_folder(name)
        rounds = directory / 'ckpt_160'
        rounds.mkdir(parents=True)
        for index in range(shards):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        arrays = {fox.FEATURES if features else 'pool_3_x_t': np.zeros((512, 2048), np.float32)}
        np.savez(str(rounds / 'statistics_0.npz'), **arrays)
        resolved = json.loads(json.dumps(_canonicalize(config)))
        resolved['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved, source_sha256='source')))
        (fox.LOGS / 'preflight.json').write_text(json.dumps(dict(source_sha256='source', checkpoints={}, failures={})))
        (fox.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=50000, nfe=nfe or 35, checkpoint_step=step or fox.STEP,
                      checkpoint_id=fox.CHECKPOINT_ID, dataset_stats_id=fox.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in config.eval.endpoint_denoise_outputs:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_160.npz'), **values)
        return directory

    def test_report_and_summary(self):
        self.prepare(F0)
        fox.report(F0)
        self.prepare(F1)
        fox.report(F1)
        result = json.loads(fox.result_path(F0).read_text())
        self.assertEqual((result['nfe'], result['num_samples'], result['checkpoint_step'], result['lambda_window']),
                         (35, 50000, 800001, [-10.0, LO]))
        summary = json.loads((fox.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['status'], summary['completed_cells'], summary['total_cells']), ('running', 2, 4))
        self.assertEqual([c['contrast'] for c in summary['point_contrasts']], ['fox_dpmpp_minus_heun_seed0'])
        self.assertAlmostEqual(summary['reference_differences'][0]['value'], 2.5 - 2.4593145847320557)
        self.assertEqual(len((fox.LOGS / 'summary.csv').read_text().splitlines()), 3)

    def test_refuses_wrong_identity_shards_and_missing_features(self):
        for arguments, message in ((dict(nfe=65), 'report identity'), (dict(step=1300002), 'report identity'),
                                   (dict(shards=20), 'shards'), (dict(features=False), 'bootstrap features')):
            directory = self.prepare(F0, **arguments)
            with self.assertRaisesRegex(RuntimeError, message):
                fox.report(F0)
            shutil.rmtree(str(directory))

    def test_refuses_the_upper_window(self):
        directory = self.prepare(F0)
        saved = json.loads((directory / 'manifest.json').read_text())
        saved['config']['sampling']['logsnr_max'] = 10.575749165617202
        (directory / 'manifest.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, 'report config'):
            fox.report(F0)


class Queue(Base):
    def write_preflight(self, failures=None):
        (fox.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=failures or {}, checkpoints={})))

    def run_queue(self, fake):
        with patch.object(fox, 'run', side_effect=fake) as run, patch.object(fox, 'report', lambda name: None), \
                patch.object(fox, 'write_summary') as self.summary:
            code = fox.queue()
        return code, [call[0][0] for call in run.call_args_list]

    def test_a_failed_cell_does_not_cancel_the_next_and_gets_a_second_pass(self):
        def fake(name):
            if name == F0:
                raise subprocess.CalledProcessError(1, 'main.py')
            fox.result_path(name).write_text('{}')

        self.write_preflight()
        code, called = self.run_queue(fake)
        self.assertEqual((code, called), (1, list(fox.CELLS) + [F0]))
        self.summary.assert_called_once_with([F0])

    def test_blocked_cells_are_not_run(self):
        self.write_preflight({name: 'wrong step' for name in fox.CELLS})
        code, called = self.run_queue(lambda name: None)
        self.assertEqual((code, called), (1, []))

    def test_finished_cells_are_skipped(self):
        self.write_preflight()
        for name in fox.CELLS:
            fox.result_path(name).write_text('{}')
        self.assertEqual(self.run_queue(lambda name: None), (0, []))


if __name__ == '__main__':
    unittest.main()
