"""CPU checks for the CelebA row of kappa; real SDEs and sampler, no checkpoints.

KAPPA_TEST_INPUTS may point to copies of the real run_manifest.json files.
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

import celeba_kappa_eval as kappa
import celeba_stage0_eval as stage0
import ode_sampling
import run_lib
from artifact_utils import _canonicalize, build_model_manifest
from configs.fox.celeba_kaffine import get_config as get_kaffine


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def write_manifests(target):
    source = os.environ.get('KAPPA_TEST_INPUTS')
    for model, work in kappa.WORKDIRS.items():
        path = target / (work + '.json')
        if source:
            shutil.copy2(str(Path(source) / (work + '.json')), str(path))
            continue
        config = get_kaffine('a0')
        config.model.fox_matern_length_scale = kappa.LENGTH_SCALES[model]
        resolved = _canonicalize(config)
        resolved.pop('eval')
        resolved['device'] = 'cuda:0'
        path.write_text(json.dumps(dict(config=resolved, model_manifest=build_model_manifest(config))))


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        write_manifests(inputs)
        for target, value in (('ROOT', self.root), ('LOGS', self.root / 'logs'), ('TRAINING_INPUTS', inputs)):
            patcher = patch.object(kappa, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, name):
        config = kappa.get_config(name)
        config.device = 'cpu'
        return config


class Protocol(Base):
    def test_a0_at_100k_is_first_and_all_windows_are_numeric(self):
        self.assertEqual(list(kappa.CELLS)[0], 'a0_c4_hi')
        self.assertEqual(len(kappa.CELLS), 6)
        for spec in kappa.CELLS.values():
            self.assertIn(spec['window'], kappa.WINDOWS)

    def test_config_matches_stage_0_protocol(self):
        fox = stage0.get_config('fox_c40_hi')
        for name in kappa.CELLS:
            config = kappa.get_config(name)
            for key in ('method', 'probability_flow', 'time_grid', 'logsnr_min', 'noise_removal',
                        'predictor', 'corrector'):
                self.assertEqual(config.sampling[key], fox.sampling[key], key)
            for key in ('sampling_num_scales', 'sampling_seed', 'num_samples', 'batch_size', 'enable_loss',
                        'enable_bpd', 'enable_sampling', 'endpoint_denoise_variants', 'save_endpoint_states'):
                self.assertEqual(config.eval[key], fox.eval[key], key)
            self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), kappa.VARIANTS)
            self.assertEqual(config.sampling.logsnr_max, kappa.WINDOWS[kappa.CELLS[name]['window']])
            self.assertEqual(config.eval.begin_ckpt, kappa.CELLS[name]['ckpt'])
            self.assertEqual(config.eval.end_ckpt, kappa.CELLS[name]['ckpt'])

    def test_config_rebuilds_the_trained_model(self):
        for model in kappa.WORKDIRS:
            manifest = json.loads(kappa.manifest_path(model).read_text())
            config = kappa.get_config('{}_c8_hi'.format(model))
            self.assertEqual(build_model_manifest(config)['model_protocol_sha256'],
                             manifest['model_manifest']['model_protocol_sha256'])

    def test_refuses_another_run_behind_the_name(self):
        path = kappa.manifest_path('k002')
        path.write_text(kappa.manifest_path('a0').read_text())
        with self.assertRaisesRegex(ValueError, 'Unexpected run'):
            kappa.get_config('k002_c8_hi')

    def test_every_grid_spans_its_window(self):
        for name in kappa.CELLS:
            config = self.config(name)
            sde, eps = run_lib.get_sde(config, n_override=kappa.BUDGET)
            kappa.check_grid(config, sde, eps)

    def test_sampler_runs_on_the_affine_schedule(self):
        for name in ('a0_c8_hi', 'k002_c8_hi', 'k002_c8_lo'):
            config = self.config(name)
            sde, eps = run_lib.get_sde(config, n_override=4)
            sampler = ode_sampling.get_vp_ode_sampler(sde, (2, 3, 8, 8), lambda x: x, kappa.METHOD, eps, config.device)
            outputs, calls = sampler(ZeroEps(), z=torch.ones(2, 3, 8, 8))
            self.assertEqual(calls, 5)
            self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()))


class Steps(Base):
    def load(self, step):
        config = self.config('a0_c4_hi')
        directory = self.root / 'workdirs' / kappa.WORKDIRS['a0'] / 'checkpoints'
        directory.mkdir(parents=True)
        (directory / 'checkpoint_4.pth').write_text('x')
        state = dict(step=step, checkpoint_id='id')
        with patch('models.utils.create_model', lambda c: torch.nn.Linear(2, 2)), \
                patch('utils.restore_checkpoint', lambda *a, **k: state), \
                patch('artifact_utils.build_model_manifest', lambda c: dict(model_protocol_sha256='x')), \
                patch('models.ema.ExponentialMovingAverage.copy_to', lambda *a: None):
            return kappa.load_model(config, kappa.WORKDIRS['a0'], 4)

    def test_checkpoint_4_is_100k(self):
        self.assertEqual(self.load(100001)[1]['step'], 100001)

    def test_refuses_a_step_on_the_5k_scale(self):
        with self.assertRaisesRegex(RuntimeError, 'Unexpected step'):
            self.load(20001)


class Report(Base):
    def prepare(self, name, nfe=kappa.BUDGET + 1):
        spec = kappa.CELLS[name]
        directory = self.root / 'workdirs' / kappa.WORKDIRS[spec['model']] / kappa.eval_folder(name)
        shards = directory / 'ckpt_{}'.format(spec['ckpt'])
        shards.mkdir(parents=True)
        for index in range((kappa.COUNT + kappa.BATCH - 1) // kappa.BATCH):
            (shards / 'statistics_{}.npz'.format(index)).write_text('x')
        config = json.loads(json.dumps(_canonicalize(kappa.get_config(name))))
        config['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        config['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=config, source_sha256='source')))
        key = '{}_{}'.format(spec['model'], spec['ckpt'])
        (kappa.LOGS / 'preflight.json').write_text(json.dumps(dict(
            source_sha256='source', checkpoints={key: dict(step=200001, checkpoint_id='id')})))
        (kappa.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=kappa.COUNT, nfe=nfe, checkpoint_step=200001, checkpoint_id='id',
                      dataset_stats_id=kappa.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in kappa.VARIANTS:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)

    def test_report_writes_result_and_contrast(self):
        for name in ('a0_c8_hi', 'k002_c8_hi'):
            self.prepare(name)
            kappa.report(name)
        summary = json.loads((kappa.LOGS / 'summary.json').read_text())
        self.assertEqual(summary['completed_cells'], 2)
        self.assertEqual([c['contrast'] for c in summary['point_contrasts']], ['k002_vs_a0_200k_hi'])
        self.assertEqual(summary['point_contrasts'][0]['value'], 0.0)

    def test_report_refuses_other_identity(self):
        self.prepare('a0_c8_hi', nfe=35)
        with self.assertRaisesRegex(RuntimeError, 'report identity'):
            kappa.report('a0_c8_hi')


class Queue(Base):
    def test_finished_cells_are_skipped_and_a_failure_does_not_stop_the_row(self):
        kappa.result_path('a0_c4_hi').write_text('{}')
        called = []

        def fake(name):
            called.append(name)
            if name == 'a0_c8_hi':
                raise subprocess.CalledProcessError(1, 'main.py')

        with patch.object(kappa, 'run', fake), patch.object(kappa, 'report', lambda name: None):
            self.assertEqual(kappa.queue(), 1)
        self.assertEqual(called, list(kappa.CELLS)[1:])
        self.assertTrue(kappa.failure_path('a0_c8_hi').exists())


if __name__ == '__main__':
    unittest.main()
