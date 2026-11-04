"""Docker CPU checks of matched35 protocol, anchor rejection and queue safety."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import celeba_matched35_eval as matched
import check_low_nfe_overnight as shared
LAUNCHER = shared.LAUNCHER


class Protocol(unittest.TestCase):
    def test_snapshot_pins_anchors_and_rejects_changed_numerical_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, anchors, output = [root / n for n in ('source', 'anchors', 'output')]
            source.mkdir()
            (anchors / 'logs').mkdir(parents=True)
            output.mkdir()
            files = {}
            for relative in ('ode_sampling.py', 'sampling.py', 'sde_lib.py', 'main.py', 'run_lib.py',
                'evaluation.py', 'artifact_utils.py', 'reproducibility.py',
                'celeba_final_ode_eval.py', 'celeba_edm_protocol_eval.py', 'configs/example.py'):
                path = source / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text('fixture\n')
                files[relative] = matched.digest(path)
            (anchors / 'snapshot.json').write_text(json.dumps(dict(profile='edm_protocol', runtime_files=files)))
            (anchors / 'logs/preflight.json').write_text('{}')
            with patch.object(matched, 'ANCHORS', anchors):
                for arm in matched.ARMS:
                    directory = matched.anchor_directory(arm)
                    directory.mkdir(parents=True)
                    for name in ('manifest.json', 'report_260.npz'): (directory / name).write_text('fixture')
                def fake_snapshot(src, out):
                    matched.write_json(Path(out) / 'snapshot.json', dict(profile='final_ode', source_sha256='s'))
                with patch.object(matched.final, 'snapshot', fake_snapshot):
                    matched.snapshot(source, output)
                    saved = json.loads((output / 'snapshot.json').read_text())
                    self.assertEqual((saved['profile'], len(saved['cells']), len(saved['external_cells'])), ('matched35', 6, 2))
                    with patch.object(matched.final, 'check', return_value=saved):
                        matched.check()
                        (anchors / 'logs/preflight.json').write_text('changed')
                        with self.assertRaisesRegex(RuntimeError, 'Pinned anchor changed'): matched.check()
                    (source / 'ode_sampling.py').write_text('changed')
                    with self.assertRaisesRegex(RuntimeError, 'source differs'): matched.snapshot(source, output)

    def test_matrix_and_existing_protocols(self):
        import celeba_edm_protocol_eval as edm
        self.assertEqual(len(matched.CELLS), 6)
        self.assertEqual(len(set(matched.CELLS + matched.EXTERNAL_CELLS)), 8)
        self.assertEqual([c[4] for c in matched.CELLS], [0, 0, 1, 1, 1, 1])
        self.assertEqual([c[2] for c in matched.CELLS[:2]], ['vp', 'fox'])
        for cell in matched.CELLS + matched.EXTERNAL_CELLS:
            n, count, arm, method, seed = cell
            cfg = matched.config_for(cell)
            self.assertEqual((n, count, cfg.eval.num_samples, cfg.eval.batch_size), (34, 50000, 50000, 512))
            self.assertEqual(cfg.eval.sampling_num_scales + 1, 35)
            self.assertEqual(cfg.eval.sampling_seed, seed)
            self.assertEqual(cfg.sampling.logsnr_min, -10.)
            self.assertEqual(cfg.sampling.logsnr_max, 10.575749165617202)
            self.assertEqual(cfg.sampling.method, method)
            if method == 'ode_edm_heun':
                expected = edm.get_config('common_50000_260_' + arm)
                expected.eval.sampling_seed = seed
                self.assertEqual(cfg.to_dict(), expected.to_dict())
        for cell in matched.CELLS:
            self.assertEqual(matched.get_config(matched.config_name(*cell)).to_dict(), matched.config_for(cell).to_dict())
        for name in ('ode_edm_heun_34_50000_vp_seed0', 'ode_dpmpp_2m_32_50000_vp_seed0',
                     'ode_dpmpp_2m_34_10000_vp_seed0', 'ode_dpmpp_2m_34_50000_vp_seed2'):
            with self.assertRaises(ValueError): matched.get_config(name)

    @patch.dict(sys.modules, tensorflow=SimpleNamespace())
    def test_reports_reject_wrong_identity_precision_and_seed(self):
        import numpy as np
        from artifact_utils import _canonicalize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            (output / 'logs').mkdir(parents=True)
            (output / 'anchors').mkdir()
            pre = dict(source_sha256='source', checkpoint_ids={'vp': 'v', 'fox': 'f'})
            (output / 'logs/preflight.json').write_text(json.dumps(pre))
            anchors = []
            with patch.object(matched, 'ROOT', root), patch.object(matched, 'OUTPUT', output):
                for cell in matched.CELLS + matched.EXTERNAL_CELLS:
                    n, count, arm, method, seed = cell
                    directory = root / 'workdirs' / matched.ARMS[arm] / matched.folder(*cell)
                    (directory / 'ckpt_260').mkdir(parents=True)
                    cfg = matched.config_for(cell)
                    config = _canonicalize(cfg)
                    config['eval'].pop('begin_ckpt')
                    config['eval'].pop('end_ckpt')
                    manifest = dict(config=config, source_sha256='source')
                    (directory / 'manifest.json').write_text(json.dumps(manifest))
                    values = dict(num_samples=count, nfe=35, checkpoint_step=1300002,
                                  checkpoint_id=pre['checkpoint_ids'][arm], dataset_stats_id=matched.REFERENCE_SHA)
                    values.update({m + '_' + v: 1.0 for m in ('IS', 'fid', 'kid') for v in matched.VARIANTS})
                    np.savez(directory / 'report_260.npz', **values)
                    for i in range(98): (directory / 'ckpt_260/statistics_{}.npz'.format(i)).touch()
                    matched.validate_result(directory, cell, pre['checkpoint_ids'][arm], 'source')
                    if cell in matched.EXTERNAL_CELLS:
                        anchors.append(dict(arm=arm, sampling_seed=0, method=method, fid_tweedie_xt=1., reused=True))
                (output / 'anchors/validated.json').write_text(json.dumps(anchors))
                for cell in matched.CELLS:
                    name = matched.config_name(*cell)
                    (output / 'logs' / (name + '_timing.json')).write_text(json.dumps(dict(returncode=0, elapsed_seconds=60.)))
                    matched.report(*cell)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual((summary['status'], summary['completed_cells'], len(summary['cells'])), ('completed', 6, 8))
                self.assertEqual(len(summary['point_contrasts']), 4)
                self.assertFalse(summary['pairing_validated'])
                self.assertEqual(summary['evaluation_elapsed_seconds'], 360.)
                cell = matched.EXTERNAL_CELLS[-1]
                config['allow_tf32'] = False
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, 'precision'):
                    matched.validate_result(directory, cell, pre['checkpoint_ids']['fox'], 'source')
                config['allow_tf32'] = True
                config['eval']['sampling_seed'] = 1
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, 'config: eval'):
                    matched.validate_result(directory, cell, 'f', 'source')
                config['eval']['sampling_seed'] = 0
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                values['nfe'] = 33
                np.savez(directory / 'report_260.npz', **values)
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    matched.validate_result(directory, cell, 'f', 'source')

    def test_run_failure_records_status_and_routes_unique_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'logs').mkdir()
            cell = matched.CELLS[-1]
            with patch.object(matched, 'OUTPUT', output), patch.object(matched.subprocess, 'run') as execute:
                execute.return_value = subprocess.CompletedProcess([], 7)
                with self.assertRaises(subprocess.CalledProcessError): matched.run(*cell)
                self.assertIn('--config=celeba_matched35_eval.py:' + matched.config_name(*cell), execute.call_args.args[0])
                timing = json.loads(next((output / 'logs').glob('*_timing.json')).read_text())
                self.assertEqual(timing['returncode'], 7)

    def test_exact_35_calls_and_initial_states_match_edm_anchors(self):
        import torch
        import ode_sampling
        import sde_lib
        from check_celeba_ode import GaussianEps
        from reproducibility import derive_seed, isolated_torch_rng
        torch.set_num_threads(1)
        for arm in ('vp', 'fox'):
            initial = None
            for method in ('ode_dpmpp_2m', 'ode_edm_heun'):
                process = sde_lib.VPSDE(N=34) if arm == 'vp' else sde_lib.FoxVPSDE(
                    u=-5., kernel='matern32', matern_length_scale=346.4101615138,
                    target_terminal_variance=1., N=34)
                process.sampling_logsnr_min, process.sampling_logsnr_max = -10., 10.575749165617202
                model = GaussianEps(process)
                with isolated_torch_rng(derive_seed(0, 'sampling-round', 0)):
                    outputs, calls = ode_sampling.get_vp_ode_sampler(process, (2, 1, 2, 2), lambda x: x,
                        method, device='cpu')(model)
                self.assertEqual(calls, 35)
                self.assertTrue(all(torch.isfinite(x).all() for x in outputs.values()))
                if initial is None: initial = model.calls[0][0]
                else: torch.testing.assert_close(initial, model.calls[0][0], rtol=0, atol=0)


class Queue(shared.QueueChecks):
    def test_matched_plan_order_and_failure(self):
        result, calls, output = self.run_queue(profile='matched35', cells=matched.CELLS)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([c for c in calls if c.startswith('eval ')], ['eval {} {} {} {} {}'.format(*c) for c in matched.CELLS])
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')

    def test_matched_failure_stops_following_cells(self):
        result, calls, output = self.run_queue(fail='34:50000:fox', profile='matched35', cells=matched.CELLS)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len([c for c in calls if c.startswith('eval ')]), 2)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_matched_host_preserves_gpu_caches_and_readonly_anchors(self):
        source, inputs, anchors, binary, output = [self.root / n for n in ('source', 'inputs', 'anchors', 'bin', 'output')]
        for p in (source, inputs, anchors, binary): p.mkdir()
        shutil.copy2(str(LAUNCHER), str(source))
        for relative in ('run_e3b_splice_eval.sh', 'e3b_final50k.py', 'main.py',
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
output = Path(os.environ['CELEBA_MATCHED35_OUTPUT'])
if args[0] == 'inspect':
    if not output.exists(): sys.exit(1)
    print('{}')
elif args[:2] == ['image', 'inspect']: print('sha256:fixture')
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
        calls = self.root / 'docker_calls'
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'], DOCKER_CALLS=str(calls),
            CELEBA_FINAL_ODE_INPUT_REPO=str(inputs), CELEBA_MATCHED35_OUTPUT=str(output),
            CELEBA_MATCHED35_ANCHORS=str(anchors), TFDS_DATA_DIR='/scratch/tfds',
            TORCH_EXTENSIONS_DIR='/scratch/extensions', XDG_CACHE_HOME='/scratch/cache',
            AVAILABLE_BYTES=str(79 * 1024 ** 3))
        command = ['bash', str(source / LAUNCHER.name), '--matched35', '3']
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Need at least 80 GiB', result.stdout)
        self.assertFalse(output.exists())
        env['AVAILABLE_BYTES'] = str(80 * 1024 ** 3)
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [json.loads(s) for s in calls.read_text().splitlines()]
        launch = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertEqual(launch[launch.index('--gpus') + 1], 'device=3')
        for setting in ('CUDA_VISIBLE_DEVICES=0', 'CELEBA_QUEUE_PROFILE=matched35',
                        'TFDS_DATA_DIR=/scratch/tfds', 'TORCH_EXTENSIONS_DIR=/scratch/extensions', 'XDG_CACHE_HOME=/scratch/cache'):
            self.assertIn(setting, launch)
        self.assertIn('type=bind,src=' + str(anchors) + ',dst=/inputs/edm35,readonly', launch)
        self.assertIn('type=bind,src=' + str(output / 'runtime') + ',dst=/workspace/score_sde_pytorch,readonly', launch)
        old_calls = calls.read_text()
        again = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(calls.read_text(), old_calls)


if __name__ == '__main__': unittest.main(verbosity=2)
