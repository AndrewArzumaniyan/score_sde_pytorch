"""CPU-only checks of the EDM-protocol queue; no checkpoints or GPU required."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import celeba_edm_protocol_eval as edm
import celeba_ode_eval
from check_celeba_ode import NumericalChecks, training_manifests  # noqa: F401 (also run here)

LAUNCHER = 'run_celeba_low_nfe_overnight.sh'


class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        training_manifests(self.root)
        self.patch = patch.object(celeba_ode_eval, 'ROOT', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_matrix(self):
        self.assertEqual(len(edm.CELLS), 13)
        self.assertEqual(len(set(edm.CELLS)), 13)
        self.assertEqual(edm.CELLS[:3], [(10000, 2, 'edm', 'native'), (10000, 20, 'vp', 'edm'), (10000, 20, 'fox', 'edm')])
        self.assertEqual(sum(c[0] for c in edm.CELLS), 9 * 10000 + 4 * 50000)
        ab = [c for c in edm.CELLS if c[3] == 'ab']
        self.assertEqual([c[2] for c in ab], ['A42', 'B42', 'A43', 'B43', 'A44', 'B44'])
        self.assertEqual([c[1] for c in ab], [40, 40, 4, 4, 4, 4])
        self.assertEqual(len(edm.weights()), 11)
        self.assertAlmostEqual(edm.WINDOWS['edm'][0], -8.764053269347762, places=12)
        self.assertAlmostEqual(edm.WINDOWS['edm'][1], 12.429216196844383, places=12)

    def test_configs(self):
        for count, ckpt, arm, window in edm.CELLS:
            cfg = edm.get_config(edm.cell_name(count, ckpt, arm, window))
            self.assertEqual((cfg.eval.num_samples, cfg.eval.batch_size, cfg.eval.sampling_seed), (count, 512, 0))
            self.assertEqual((cfg.eval.begin_ckpt, cfg.eval.end_ckpt), (ckpt, ckpt))
            self.assertFalse(cfg.eval.enable_loss or cfg.eval.enable_bpd or cfg.eval.include_final_checkpoint)
            if arm == 'edm':
                self.assertEqual((cfg.training.sde, cfg.sampling.method, cfg.sampling.edm_num_steps), ('edm', 'edm', 18))
                self.assertEqual((cfg.sampling.edm_sigma_min, cfg.sampling.edm_sigma_max, cfg.sampling.edm_rho), (0.002, 80.0, 7.0))
                self.assertEqual(cfg.sampling.edm_s_churn, 0.0)
                self.assertFalse(cfg.allow_tf32)
                self.assertFalse(getattr(cfg.eval, 'endpoint_denoise_variants', False))
                continue
            self.assertEqual((cfg.sampling.method, cfg.sampling.time_grid), ('ode_edm_heun', 'karras'))
            self.assertEqual(cfg.eval.sampling_num_scales, 34)
            self.assertEqual((cfg.sampling.logsnr_min, cfg.sampling.logsnr_max), edm.WINDOWS[window])
            self.assertTrue(cfg.sampling.probability_flow and cfg.eval.endpoint_denoise_variants)
            self.assertEqual(tuple(cfg.eval.endpoint_denoise_outputs), edm.VARIANTS)
            self.assertTrue(cfg.allow_tf32)
            if arm in ('vp', 'fox'):
                self.assertEqual(cfg.training.sde.lower(), 'vpsde' if arm == 'vp' else 'foxvpsde')
            else:
                self.assertEqual(cfg.seed, int(arm[1:]))
                self.assertEqual(cfg.model.noise_conditioning, 'logsnr')
                self.assertEqual(cfg.training.noise_distribution, 'vp_time' if arm[0] == 'A' else 'normalized_fox_time')
                # The Karras window must not leave the trained A/B support.
                self.assertEqual((cfg.model.logsnr_min, cfg.model.logsnr_max), edm.WINDOWS['ab'])
        with self.assertRaises(ValueError):
            edm.get_config('edm_10000_2_edm')

    @patch.dict(sys.modules, tensorflow=SimpleNamespace())
    def test_report_intake(self):
        import numpy as np
        from artifact_utils import _canonicalize
        output = self.root / 'output'
        (output / 'logs').mkdir(parents=True)
        ids = {'{}_{}'.format(a, c): 'id-{}-{}'.format(a, c) for a, c in edm.weights()}
        (output / 'logs/preflight.json').write_text(json.dumps({'source_sha256': 'source', 'checkpoint_ids': ids}))
        with patch.object(edm, 'ROOT', self.root), patch.object(edm, 'OUTPUT', output):
            for count, ckpt, arm, window in edm.CELLS:
                directory = self.root / 'workdirs' / edm.model_name(arm) / edm.folder(count, ckpt, arm, window)
                (directory / 'ckpt_{}'.format(ckpt)).mkdir(parents=True)
                cfg = edm.get_config(edm.cell_name(count, ckpt, arm, window))
                config = {k: _canonicalize(cfg[k]) for k in ('model', 'data', 'sampling', 'eval')}
                config['eval'].pop('begin_ckpt', None)
                config['eval'].pop('end_ckpt', None)
                (directory / 'manifest.json').write_text(json.dumps({'config': config, 'source_sha256': 'source'}))
                values = dict(num_samples=count, checkpoint_step=edm.STEPS[ckpt],
                              checkpoint_id=ids['{}_{}'.format(arm, ckpt)], dataset_stats_id=edm.REFERENCE_SHA)
                if arm == 'edm':
                    values.update(IS=1.0, fid=2.0, kid=0.1)
                    np.savez(directory / 'ckpt_2/samples_0.npz', nfe=35)
                else:
                    values.update(nfe=35, **{m + '_' + v: 1.0 for m in ('IS', 'fid', 'kid') for v in edm.VARIANTS})
                np.savez(directory / 'report_{}.npz'.format(ckpt), **values)
                for i in range(math.ceil(count / 512)):
                    (directory / 'ckpt_{}'.format(ckpt) / ('statistics_%d.npz' % i)).touch()
                edm.report(count, ckpt, arm, window)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual((summary['status'], summary['completed_cells']), ('completed', 13))
            first = next(c for c in summary['cells'] if c['arm'] == 'edm')
            self.assertEqual((first['primary_fid'], first['primary_output'], first['nfe']), (2.0, 'edm_denoised', 35))
            directory = self.root / 'workdirs' / edm.EDM / edm.folder(*edm.CELLS[0])
            np.savez(directory / 'ckpt_2/samples_0.npz', nfe=34)
            with self.assertRaises(RuntimeError):
                edm.report(*edm.CELLS[0])
            np.savez(directory / 'ckpt_2/samples_0.npz', nfe=35)
            values = dict(np.load(directory / 'report_2.npz'))
            values['checkpoint_step'] = 50001
            np.savez(directory / 'report_2.npz', **values)
            with self.assertRaises(RuntimeError):
                edm.report(*edm.CELLS[0])

    def test_official_sources_ignore_git(self):
        source = self.root / 'edm'
        (source / 'training').mkdir(parents=True)
        (source / '.git').mkdir()
        (source / 'training/networks.py').write_text('x = 1\n')
        (source / '.git/hook.py').write_text('ignored\n')
        self.assertEqual(list(edm.official_sources(source)), ['training/networks.py'])


class Queue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_queue(self, fail=''):
        spec = importlib.util.spec_from_file_location('low_nfe_test_helpers', str(ROOT / 'tests/check_low_nfe_overnight.py'))
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        return helpers.QueueChecks.run_queue(self, fail=fail, profile='edm_protocol', cells=edm.CELLS)

    def test_queue_completes(self):
        result, calls, output = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([x for x in calls if x.startswith('eval ')],
                         ['eval {} {} {} {}'.format(*c) for c in edm.CELLS])
        self.assertEqual(sum(x.startswith('meta report ') for x in calls), 13)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')
        self.assertTrue((output / 'logs/edm_ckpt2_native_n10000.log').exists())

    def test_queue_stops_on_failure(self):
        result, calls, output = self.run_queue('10000:20:vp')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(x.startswith('eval ') for x in calls), 2)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_host_profile_mounts(self):
        source, data, binary, output = [self.root / n for n in ('code', 'data', 'bin', 'output')]
        for p in (source, data, binary):
            p.mkdir()
        shutil.copy2(str(ROOT / LAUNCHER), str(source))
        for name in ('main.py', 'e3b_final50k.py', 'run_e3b_splice_eval.sh', 'celeba_edm_protocol_eval.py',
                     'celeba_final_ode_eval.py', 'celeba_ode_eval.py', 'ode_sampling.py'):
            (source / name).write_text('fixture\n')
        (data / 'celeba/img_align_celeba').mkdir(parents=True)
        (data / 'celeba/list_eval_partition.txt').write_text('fixture\n')
        (data / 'assets/stats').mkdir(parents=True)
        (data / edm.REFERENCE).write_bytes(b'reference')
        (data / 'third_party/edm/.git').mkdir(parents=True)
        (data / 'third_party/edm/training').mkdir()
        (data / 'third_party/edm/training/networks.py').write_text('fixture\n')
        roots = {'42': data, '43': self.root / 'ab43', '44': self.root / 'ab44'}
        for arm, ckpt in edm.weights():
            origin = roots.get(arm[1:], data)
            work = origin / 'workdirs' / edm.model_name(arm)
            (work / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (work / 'checkpoints/checkpoint_{}.pth'.format(ckpt)).write_bytes(b'weights')
            (work / 'run_manifest.json').write_text('{}')
        docker = binary / 'docker'
        docker.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
a=sys.argv[1:]
with Path(os.environ['DOCKER_CALLS']).open('a') as f: f.write(json.dumps(a)+'\\n')
out=Path(os.environ['CELEBA_EDM_PROTOCOL_OUTPUT'])
if a[:2]==['image','inspect']: print('sha256:fixture')
elif a[0]=='inspect':
    if not out.exists(): sys.exit(1)
    print('{}')
elif a[0]=='run' and '--rm' in a:
    for name in ('runtime','logs','workdirs'): (out/name).mkdir()
elif a[0]=='run' and '-d' in a: print('fixture-container')
else: sys.exit('unexpected Docker call')
''')
        docker.chmod(0o755)
        for name, body in (('git', 'exit 0'), ('df', 'printf "Filesystem blocks used available capacity mount\\nfixture 1000000000000 0 1000000000000 0%% /\\n"')):
            p = binary / name
            p.write_text('#!/bin/bash\n' + body + '\n')
            p.chmod(0o755)
        calls = self.root / 'calls'
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
            CELEBA_EDM_PROTOCOL_INPUT_REPO=str(data), CELEBA_EDM_PROTOCOL_SEED43_ROOT=str(roots['43']),
            CELEBA_EDM_PROTOCOL_SEED44_ROOT=str(roots['44']), CELEBA_EDM_PROTOCOL_OUTPUT=str(output),
            DOCKER_CALLS=str(calls), TFDS_DATA_DIR='/scratch/tfds',
            TORCH_EXTENSIONS_DIR='/scratch/extensions', XDG_CACHE_HOME='/scratch/cache')
        result = subprocess.run(['bash', str(source / LAUNCHER), '--edm-protocol', '3'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        snap = next(c for c in commands if c[0] == 'run' and '--rm' in c)
        self.assertNotIn('--gpus', snap)
        self.assertIn('CELEBA_QUEUE_PROFILE=edm_protocol', snap)
        run = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertEqual(run[run.index('--gpus') + 1], 'device=3')
        self.assertIn('CELEBA_QUEUE_PROFILE=edm_protocol', run)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', run)
        self.assertIn('EDM_OFFICIAL_ROOT=/opt/edm', run)
        self.assertIn('type=bind,src={},dst=/opt/edm,readonly'.format(data / 'third_party/edm'), run)
        self.assertEqual(sum('/checkpoints,readonly' in x for x in run), 9)
        for suffix, target in (('.tfds', '/scratch/tfds'), ('.torch_extensions', '/scratch/extensions'), ('.cache', '/scratch/cache')):
            self.assertIn('type=bind,src={},dst={}'.format(data / suffix, target), run)
        previous = calls.read_text()
        again = subprocess.run(['bash', str(source / LAUNCHER), '--edm-protocol', '3'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(calls.read_text(), previous)
        shutil.rmtree(str(output))
        shutil.rmtree(str(data / 'third_party/edm/.git'))
        missing = subprocess.run(['bash', str(source / LAUNCHER), '--edm-protocol', '3'], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('Missing official EDM checkout', missing.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
