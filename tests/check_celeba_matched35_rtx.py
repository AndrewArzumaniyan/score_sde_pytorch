"""Docker CPU checks of the RTX matched35 profile: matrix, cross-GPU comparison and queue safety."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import celeba_matched35_eval as matched
import celeba_matched35_rtx_eval as rtx
import check_low_nfe_overnight as shared
LAUNCHER = shared.LAUNCHER


def write_round(directory, r, noise, seed_offset=0, perturb=0.0, samples=True):
    rng = np.random.default_rng(noise)
    features = rng.normal(size=(rtx.BATCH, 8)).astype(np.float32)
    images = rng.integers(0, 256, size=(rtx.BATCH, 4, 4, 3)).astype(np.uint8)
    seed = 1000 + r
    np.savez(directory / 'statistics_{}.npz'.format(r), pool_3_tweedie_xt=features + perturb,
             sampling_seed=np.asarray(seed + seed_offset))
    if samples:
        np.savez(directory / 'samples_{}.npz'.format(r), samples_tweedie_xt=images,
                 float_x_t=images.astype(np.float32) / 255. + perturb)


class Protocol(unittest.TestCase):
    def test_matrix_order_and_configs_match_h200_protocol(self):
        self.assertEqual(len(set(rtx.FULL)), 8)
        self.assertEqual(rtx.FULL[:2], [(34, 50000, 'vp', 'ode_edm_heun', 0), (34, 50000, 'fox', 'ode_edm_heun', 0)])
        self.assertEqual(set(rtx.FULL), set(matched.CELLS + matched.EXTERNAL_CELLS))
        self.assertEqual(rtx.SMOKE, [(34, 1024, 'vp', 'ode_edm_heun', 0)])
        for cell in rtx.FULL + rtx.SMOKE:
            cfg = rtx.config_for(cell)
            expected = matched.config_for((34, 50000) + cell[2:])
            expected.eval.num_samples = cell[1]
            self.assertEqual(cfg.to_dict(), expected.to_dict())
            self.assertEqual((cfg.allow_tf32, cfg.cudnn_benchmark), (True, True))
            self.assertEqual(rtx.get_config(matched.config_name(*cell)).to_dict(), cfg.to_dict())
        for name in ('ode_edm_heun_34_2048_vp_seed0', 'ode_dpmpp_2m_34_1024_vp_seed0', 'ode_dpmpp_2m_34_50000_vp_seed2'):
            with self.assertRaises(ValueError): rtx.get_config(name)

    def test_profile_is_required_and_pinned(self):
        with patch.dict(os.environ, CELEBA_QUEUE_PROFILE='matched35'):
            with self.assertRaisesRegex(RuntimeError, 'CELEBA_QUEUE_PROFILE'): rtx.profile()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            def fake_snapshot(source, out):
                matched.write_json(Path(out) / 'snapshot.json', dict(profile='matched35', external_cells=[1]))
            with patch.dict(os.environ, CELEBA_QUEUE_PROFILE='matched35_rtx_smoke'), \
                    patch.object(rtx.m35, 'snapshot', fake_snapshot):
                rtx.snapshot('source', output)
                saved = json.loads((output / 'snapshot.json').read_text())
                self.assertEqual((saved['profile'], saved['external_cells'], saved['anchors_reused']),
                                 ('matched35_rtx_smoke', [], False))
                with patch.object(rtx.m35, 'check', return_value=saved): rtx.check()
            with patch.dict(os.environ, CELEBA_QUEUE_PROFILE='matched35_rtx'), \
                    patch.object(rtx.m35, 'check', return_value=saved):
                with self.assertRaisesRegex(RuntimeError, 'profile differs'): rtx.check()

    def test_preflight_rejects_extension_arch_that_misses_the_gpu(self):
        import torch
        with patch.dict(os.environ, CELEBA_QUEUE_PROFILE='matched35_rtx', TORCH_CUDA_ARCH_LIST='9.0'), \
                patch.object(rtx, 'check'), patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(12, 0)), \
                patch.object(rtx.final, 'preflight') as final_preflight:
            with self.assertRaisesRegex(RuntimeError, 'sm 12.0'): rtx.preflight()
            final_preflight.assert_not_called()

    def test_cross_gpu_comparison_separates_same_and_different_noise(self):
        cell = rtx.SMOKE[0]
        with tempfile.TemporaryDirectory() as tmp:
            anchors, ours = Path(tmp) / 'anchors', Path(tmp) / 'ours'
            with patch.object(rtx.m35, 'ANCHORS', anchors):
                reference = rtx.m35.anchor_directory('vp') / 'ckpt_260'
                reference.mkdir(parents=True)
                (ours / 'ckpt_260').mkdir(parents=True)
                for r in range(2):
                    write_round(reference, r, noise=r)
                    write_round(ours / 'ckpt_260', r, noise=r, perturb=1e-3)
                same = rtx.compare_with_anchor(cell, ours)
                self.assertLess(same['pool_3_tweedie_xt']['paired_to_unpaired'], 0.01)
                self.assertEqual((same['pool_3_tweedie_xt']['rows'], same['image_rounds']), (1024, 2))
                self.assertLess(same['float_x_t']['paired_to_unpaired'], 0.01)
                for r in range(2):
                    for name in ('statistics_{}.npz', 'samples_{}.npz'): (ours / 'ckpt_260' / name.format(r)).unlink()
                    write_round(ours / 'ckpt_260', r, noise=100 + r, samples=False)
                other = rtx.compare_with_anchor(cell, ours)
                self.assertGreater(other['pool_3_tweedie_xt']['paired_to_unpaired'], 0.8)
                self.assertIsNone(other['samples_tweedie_xt_uint8'])
                (ours / 'ckpt_260/statistics_1.npz').unlink()
                write_round(ours / 'ckpt_260', 1, noise=1, seed_offset=1, samples=False)
                with self.assertRaisesRegex(RuntimeError, 'seeds differ'): rtx.compare_with_anchor(cell, ours)

    @patch.dict(sys.modules, tensorflow=SimpleNamespace())
    def test_reports_keep_rtx_contrasts_and_h200_reference_apart(self):
        from artifact_utils import _canonicalize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            (output / 'logs').mkdir(parents=True)
            (output / 'anchors').mkdir()
            pre = dict(source_sha256='source', checkpoint_ids={'vp': 'v', 'fox': 'f'},
                       environment=dict(visible_cuda_devices=['NVIDIA RTX PRO 6000']))
            (output / 'logs/preflight.json').write_text(json.dumps(pre))
            anchors = [dict(arm=arm, fid_tweedie_xt=2.0) for arm in ('vp', 'fox')]
            (output / 'anchors/validated.json').write_text(json.dumps(dict(anchors=anchors)))
            fids = {}
            with patch.dict(os.environ, CELEBA_QUEUE_PROFILE='matched35_rtx'), \
                    patch.object(rtx, 'ROOT', root), patch.object(rtx, 'OUTPUT', output), \
                    patch.object(rtx, 'compare_with_anchor', return_value=dict(pool_3_tweedie_xt=dict(paired_to_unpaired=0.001, rows=50000))):
                for i, cell in enumerate(rtx.FULL):
                    n, count, arm, method, seed = cell
                    directory = root / 'workdirs' / rtx.ARMS[arm] / rtx.folder(*cell)
                    (directory / 'ckpt_260').mkdir(parents=True)
                    config = _canonicalize(rtx.config_for(cell))
                    config['eval'].pop('begin_ckpt')
                    config['eval'].pop('end_ckpt')
                    (directory / 'manifest.json').write_text(json.dumps(dict(config=config, source_sha256='source')))
                    fids[cell] = 2.0 + 0.1 * i
                    values = dict(num_samples=count, nfe=35, checkpoint_step=1300002,
                                  checkpoint_id=pre['checkpoint_ids'][arm], dataset_stats_id=rtx.REFERENCE_SHA)
                    values.update({m + '_' + v: fids[cell] for m in ('IS', 'fid', 'kid') for v in rtx.VARIANTS})
                    np.savez(directory / 'report_260.npz', **values)
                    for r in range(98): (directory / 'ckpt_260/statistics_{}.npz'.format(r)).touch()
                    (output / 'logs' / (matched.config_name(*cell) + '_timing.json')).write_text(
                        json.dumps(dict(returncode=0, elapsed_seconds=60.)))
                    rtx.report(*cell)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual((summary['status'], summary['completed_cells'], len(summary['cells'])), ('completed', 8, 8))
                kinds = [c['kind'] for c in summary['point_contrasts']]
                self.assertEqual((kinds.count('dpmpp_minus_edm'), kinds.count('fox_minus_vp')), (4, 4))
                self.assertFalse(any(c.get('reused') for c in summary['cells']))
                self.assertEqual(sorted((c['arm'], round(c['rtx_minus_h200'], 6)) for c in summary['cross_gpu']),
                                 [('fox', 0.1), ('vp', 0.0)])
                self.assertFalse(summary['h200_anchors_mixed_into_contrasts'])
                cell = rtx.FULL[-1]
                directory = root / 'workdirs' / rtx.ARMS[cell[2]] / rtx.folder(*cell)
                (directory / 'ckpt_260/statistics_97.npz').unlink()
                with self.assertRaisesRegex(RuntimeError, 'shard count'):
                    rtx.validate_result(directory, cell, 'f', 'source')


class Queue(shared.QueueChecks):
    def test_rtx_plan_order_and_failure(self):
        result, calls, output = self.run_queue(profile='matched35_rtx', cells=rtx.FULL)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([c for c in calls if c.startswith('eval ')], ['eval {} {} {} {} {}'.format(*c) for c in rtx.FULL])
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')
        self.tearDown()
        self.setUp()
        result, calls, output = self.run_queue(fail='34:50000:fox', profile='matched35_rtx', cells=rtx.FULL)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len([c for c in calls if c.startswith('eval ')]), 2)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_rtx_smoke_runs_one_cell(self):
        result, calls, output = self.run_queue(profile='matched35_rtx_smoke', cells=rtx.SMOKE)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([c for c in calls if c.startswith('eval ')], ['eval 34 1024 vp ode_edm_heun 0'])

    def launch(self, flag, available_gib, extra_env=None):
        source, inputs, anchors, binary = [self.root / n for n in ('source', 'inputs', 'anchors', 'bin')]
        output = self.root / ('output' + flag)
        if not source.exists():
            for p in (source, inputs, anchors, binary): p.mkdir()
            shutil.copy2(str(LAUNCHER), str(source))
            for relative in ('run_e3b_splice_eval.sh', 'e3b_final50k.py', 'main.py', 'celeba_matched35_rtx_eval.py',
                             'celeba_matched35_eval.py', 'celeba_final_ode_eval.py', 'ode_sampling.py'):
                (source / relative).write_text('fixture\n')
            for relative in (matched.final.REFERENCE, 'celeba/list_eval_partition.txt',
                             'workdirs/' + matched.ARMS['vp'] + '/checkpoints/checkpoint_260.pth',
                             'workdirs/' + matched.ARMS['fox'] + '/checkpoints/checkpoint_260.pth'):
                path = inputs / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n')
            (inputs / 'celeba/img_align_celeba').mkdir()
            (anchors / 'logs').mkdir()
            for relative in ('snapshot.json', 'logs/preflight.json'): (anchors / relative).write_text('{}\n')
            docker = binary / 'docker'
            docker.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['DOCKER_CALLS']).open('a') as stream: stream.write(json.dumps(args) + '\\n')
output = Path(os.environ['CELEBA_MATCHED35_RTX_OUTPUT'])
if args[0] == 'inspect':
    if not output.exists(): sys.exit(1)
    print('{}')
elif args[:2] == ['image', 'inspect']: print('sha256:' + args[-1])
elif args[0] == 'run' and '--rm' in args:
    for name in ('runtime', 'logs', 'workdirs'): (output / name).mkdir()
elif args[0] == 'run' and '-d' in args: print('fake-container-id')
else: sys.exit('unexpected docker call')
''')
            docker.chmod(0o755)
            for name, script in (('git', 'exit 0'), ('df',
                'printf "Filesystem blocks used available capacity mount\\nfixture 999999999999 0 %s 0%% /\\n" "$AVAILABLE_BYTES"')):
                path = binary / name
                path.write_text('#!/bin/bash\n' + script + '\n')
                path.chmod(0o755)
        calls = self.root / ('calls' + flag)
        env = {k: v for k, v in os.environ.items()
               if k not in ('TORCH_CUDA_ARCH_LIST', 'TORCH_EXTENSIONS_DIR', 'XDG_CACHE_HOME', 'TFDS_DATA_DIR')}
        env.update(PATH=str(binary) + ':' + os.environ['PATH'], DOCKER_CALLS=str(calls),
                   CELEBA_FINAL_ODE_INPUT_REPO=str(inputs), CELEBA_MATCHED35_RTX_OUTPUT=str(output),
                   CELEBA_MATCHED35_RTX_CONTAINER='rtx-test' + flag.replace('-', '_'),
                   CELEBA_MATCHED35_ANCHORS=str(anchors), AVAILABLE_BYTES=str(available_gib * 1024 ** 3),
                   **(extra_env or {}))
        result = subprocess.run(['bash', str(source / LAUNCHER.name), flag, '1'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        commands = [json.loads(s) for s in calls.read_text().splitlines()] if calls.exists() else []
        return result, commands, inputs, anchors, output

    def test_rtx_host_uses_rtx_image_arch_and_separate_caches(self):
        result, commands, inputs, anchors, output = self.launch('--matched35-rtx', 79)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Need at least 80 GiB', result.stdout)
        result, commands, inputs, anchors, output = self.launch('--matched35-rtx', 80)
        self.assertEqual(result.returncode, 0, result.stdout)
        launch = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertEqual(launch[launch.index('--gpus') + 1], 'device=1')
        self.assertEqual(launch[-3], 'sha256:score-sde-rtx:latest')
        root = '/workspace/score_sde_pytorch'
        for setting in ('CUDA_VISIBLE_DEVICES=0', 'CELEBA_QUEUE_PROFILE=matched35_rtx', 'TORCH_CUDA_ARCH_LIST=12.0',
                        'TORCH_EXTENSIONS_DIR=' + root + '/.torch_extensions_rtx', 'XDG_CACHE_HOME=' + root + '/.cache_rtx'):
            self.assertIn(setting, launch)
        self.assertIn('type=bind,src={}/.torch_extensions_rtx,dst={}/.torch_extensions_rtx'.format(inputs, root), launch)
        self.assertIn('type=bind,src={}/.cache_rtx,dst={}/.cache_rtx'.format(inputs, root), launch)
        self.assertFalse(any(m.startswith('type=bind,src={}/.torch_extensions,'.format(inputs)) for m in launch))
        self.assertIn('type=bind,src=' + str(anchors) + ',dst=/inputs/edm35,readonly', launch)
        self.assertIn('TORCH_CUDA_ARCH_LIST=12.0', (output / 'docker.txt').read_text())

    def test_rtx_smoke_needs_little_space_and_uses_its_own_output(self):
        result, commands, inputs, anchors, output = self.launch('--matched35-rtx-smoke', 5)
        self.assertEqual(result.returncode, 0, result.stdout)
        launch = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertIn('CELEBA_QUEUE_PROFILE=matched35_rtx_smoke', launch)
        snapshot = next(c for c in commands if c[0] == 'run' and '--rm' in c)
        self.assertIn('CELEBA_QUEUE_PROFILE=matched35_rtx_smoke', snapshot)


if __name__ == '__main__': unittest.main(verbosity=2)
