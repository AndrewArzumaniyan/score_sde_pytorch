"""CPU checks for the CIFAR matched protocol; real SDEs and sampler, no checkpoints.

CIFAR35_TEST_INPUTS must point to copies of the real run_manifest.json files.
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

import cifar_matched35_eval as cifar
import ode_sampling
import run_lib
from artifact_utils import _canonicalize, build_model_manifest
from models import ddpm, ncsnpp  # noqa: F401


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        for work in {value[0] for value in cifar.MODELS.values()}:
            shutil.copy2(str(Path(os.environ['CIFAR35_TEST_INPUTS']) / (work + '.json')), str(inputs / (work + '.json')))
        for target, value in (('ROOT', self.root), ('LOGS', self.root / 'logs'), ('TRAINING_INPUTS', inputs)):
            patcher = patch.object(cifar, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self, name):
        config = cifar.get_config(name)
        config.device = 'cpu'
        return config


class Plan(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(len(cifar.CELLS), 22)
        self.assertEqual(list(cifar.CELLS)[0], cifar.GATE_CELL)
        cells = list(cifar.CELLS.values())
        for model in cifar.MODELS:
            matched = [c for c in cells if c['model'] == model and c['calls'] == 34]
            self.assertEqual({(c['method'], c['seed']) for c in matched},
                             {(m, s) for m in (cifar.DPM, cifar.HEUN) for s in (0, 1)})
        self.assertEqual({c['model'] for c in cells if c['calls'] == 64}, {'ncsn_fox600', 'ncsn_vp400'})

    def test_every_contrast_names_real_cells(self):
        for _, left, right in cifar.CONTRASTS:
            self.assertIn(left, cifar.CELLS)
            self.assertIn(right, cifar.CELLS)


class Protocol(Base):
    def test_gate_config_is_the_old_cell(self):
        config = cifar.get_config(cifar.GATE_CELL)
        self.assertEqual((config.seed, config.allow_tf32, config.cudnn_benchmark), (42, False, False))
        self.assertEqual((config.eval.sampling_num_scales, config.eval.sampling_seed), (64, 2))
        self.assertEqual((config.eval.num_samples, config.eval.batch_size), (50000, 512))
        self.assertEqual(config.sampling.time_grid, 'uniform_logsnr')
        self.assertEqual(config.sampling.logsnr_max, 10.575749165617202)

    def test_heun_uses_the_karras_grid(self):
        self.assertEqual(cifar.get_config('ncsn_vp600_{}_n34_seed0'.format(cifar.HEUN)).sampling.time_grid, 'karras')

    def test_config_rebuilds_each_trained_model(self):
        for model, (work, _, _) in cifar.MODELS.items():
            name = next(n for n, spec in cifar.CELLS.items() if spec['model'] == model)
            manifest = json.loads((cifar.TRAINING_INPUTS / (work + '.json')).read_text())
            self.assertEqual(build_model_manifest(cifar.get_config(name))['model_protocol_sha256'],
                             manifest['model_manifest']['model_protocol_sha256'], model)

    def test_every_grid_spans_the_window_and_the_sampler_runs(self):
        for name, spec in cifar.CELLS.items():
            if spec['seed'] == 1:
                continue
            config = self.config(name)
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            if spec['method'] == cifar.DPM:
                cifar.check_grid(config, sde, eps, spec['calls'])
            small, _ = run_lib.get_sde(config, n_override=4)
            sampler = ode_sampling.get_vp_ode_sampler(small, (2, 3, 8, 8), lambda x: x, spec['method'], eps, config.device)
            outputs, calls = sampler(ZeroEps(), z=torch.ones(2, 3, 8, 8))
            self.assertEqual(calls, 5, name)
            self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()), name)


class Report(Base):
    def prepare(self, name, fid=3.4352, nfe=None):
        spec = cifar.CELLS[name]
        work, ckpt, step = cifar.MODELS[spec['model']]
        directory = self.root / 'workdirs' / work / cifar.eval_folder(name)
        shards = directory / 'ckpt_{}'.format(ckpt)
        shards.mkdir(parents=True)
        for index in range(98):
            (shards / 'statistics_{}.npz'.format(index)).write_text('x')
        config = json.loads(json.dumps(_canonicalize(cifar.get_config(name))))
        config['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        config['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=config, source_sha256='source')))
        key = '{}_{}'.format(spec['model'], spec['method'])
        path = cifar.LOGS / 'preflight.json'
        pre = json.loads(path.read_text()) if path.exists() else dict(source_sha256='source', checkpoints={})
        pre['checkpoints'][key] = dict(step=step, checkpoint_id='id')
        path.write_text(json.dumps(pre))
        (cifar.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=cifar.COUNT, nfe=nfe or spec['calls'] + 1, checkpoint_step=step,
                      checkpoint_id='id', dataset_stats_id=cifar.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in cifar.VARIANTS:
                values[metric + '_' + variant] = fid
        np.savez(str(directory / 'report_{}.npz'.format(ckpt)), **values)

    def test_report_and_contrast(self):
        left = 'ncsn_fox600_{}_n34_seed0'.format(cifar.DPM)
        right = 'ncsn_vp600_{}_n34_seed0'.format(cifar.DPM)
        self.prepare(left, fid=3.0)
        cifar.report(left)
        self.prepare(right, fid=3.5)
        cifar.report(right)
        summary = json.loads((cifar.LOGS / 'summary.json').read_text())
        self.assertEqual(summary['completed_cells'], 2)
        self.assertEqual(len(summary['point_contrasts']), 1)
        self.assertAlmostEqual(summary['point_contrasts'][0]['value'], -0.5)

    def test_report_refuses_other_identity(self):
        self.prepare(cifar.GATE_CELL, nfe=35)
        with self.assertRaisesRegex(RuntimeError, 'report identity'):
            cifar.report(cifar.GATE_CELL)

    def test_gate(self):
        self.prepare(cifar.GATE_CELL, fid=cifar.GATE_FID + 0.01)
        cifar.report(cifar.GATE_CELL)
        cifar.check_gate()
        cifar.result_path(cifar.GATE_CELL).write_text(json.dumps(dict(fid_tweedie_xt=cifar.GATE_FID + 0.05)))
        with self.assertRaises(SystemExit) as stop:
            cifar.check_gate()
        self.assertEqual(stop.exception.code, cifar.GATE_EXIT)


class Queue(Base):
    def test_a_failed_gate_cell_stops_the_queue(self):
        def fail(name):
            raise subprocess.CalledProcessError(1, 'main.py')
        with patch.object(cifar, 'run', fail), self.assertRaises(SystemExit) as stop:
            cifar.queue()
        self.assertEqual(stop.exception.code, cifar.GATE_EXIT)

    def test_other_failures_do_not_stop_the_queue(self):
        cifar.result_path(cifar.GATE_CELL).write_text(json.dumps(dict(fid_tweedie_xt=cifar.GATE_FID)))
        second = list(cifar.CELLS)[1]
        called = []

        def fake(name):
            called.append(name)
            if name == second:
                raise subprocess.CalledProcessError(1, 'main.py')

        with patch.object(cifar, 'run', fake), patch.object(cifar, 'report', lambda name: None):
            self.assertEqual(cifar.queue(), 1)
        self.assertEqual(called, list(cifar.CELLS)[1:])


if __name__ == '__main__':
    unittest.main()
