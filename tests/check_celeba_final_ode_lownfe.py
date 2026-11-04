"""CPU-only checks of the two-method, two-seed final low-NFE profile."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import celeba_final_ode_eval as final
import celeba_final_ode_lownfe_eval as low
from check_low_nfe_overnight import QueueChecks, LAUNCHER


class LowProtocol(unittest.TestCase):
    def test_matrix_and_configs_differ_from_final_only_in_budget_and_seed(self):
        expected = {(n, 10000, arm, method, seed) for n in (16, 32, 64, 128, 256)
                    for seed in (0, 1) for method in ('ode_dpmpp_2m', 'ode_heun') for arm in ('vp', 'fox')}
        self.assertEqual(len(low.CELLS), 36)
        self.assertEqual(len(set(low.CELLS)), 36)
        self.assertFalse(set(low.CELLS) & set(low.EXTERNAL_CELLS))
        self.assertEqual(set(low.CELLS) | set(low.EXTERNAL_CELLS), expected)
        self.assertEqual([c[0] for c in low.CELLS], sorted(c[0] for c in low.CELLS))
        self.assertEqual(sum(c[0] for c in low.CELLS), 2944)
        for n, count, arm, method, seed in low.CELLS:
            cfg = low.get_config(low.config_name(n, count, arm, method, seed))
            reference = final.get_config('{}_{}_{}'.format(method, count, arm))
            self.assertEqual(cfg.eval.sampling_num_scales, n)
            reference.eval.sampling_num_scales = n
            reference.eval.sampling_seed = seed
            for section in ('model', 'data', 'sampling', 'eval', 'training'):
                self.assertEqual(cfg[section].to_dict(), reference[section].to_dict(), section)
            self.assertEqual((cfg.eval.sampling_seed, cfg.eval.num_samples, cfg.eval.batch_size), (seed, 10000, 512))
            self.assertEqual((cfg.sampling.logsnr_min, cfg.sampling.logsnr_max), (-10., 10.575749165617202))
        for name in ('ode_dpmpp_2m_256_10000_vp_seed0', 'ode_heun_16_10000_vp_seed2',
                     'ode_edm_heun_16_10000_vp_seed0', 'ode_heun_16_50000_vp_seed0'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                low.get_config(name)
        self.assertEqual(len({low.folder(*c) for c in low.CELLS} | {final.folder(c[1], c[2], c[3]) for c in final.CELLS}),
                         len(low.CELLS) + len(final.CELLS))

    @patch.dict(sys.modules, tensorflow=SimpleNamespace())
    def test_report_checks_budget_and_identity(self):
        import numpy as np
        from artifact_utils import _canonicalize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            (output / 'logs').mkdir(parents=True)
            (output / 'logs/preflight.json').write_text(json.dumps({'source_sha256': 'source', 'checkpoint_ids': {'vp': 'v', 'fox': 'f'}}))
            with patch.object(low, 'ROOT', root), patch.object(low, 'OUTPUT', output):
                for n, count, arm, method, seed in low.CELLS:
                    directory = root / 'workdirs' / low.ARMS[arm] / low.folder(n, count, arm, method, seed)
                    (directory / 'ckpt_260').mkdir(parents=True)
                    cfg = low.get_config(low.config_name(n, count, arm, method, seed))
                    config = {k: _canonicalize(cfg[k]) for k in ('model', 'data', 'sampling', 'eval')}
                    config['eval'].pop('begin_ckpt', None)
                    config['eval'].pop('end_ckpt', None)
                    (directory / 'manifest.json').write_text(json.dumps({'config': config, 'source_sha256': 'source'}))
                    values = dict(num_samples=count, nfe=n + 1, checkpoint_step=1300002,
                        checkpoint_id='v' if arm == 'vp' else 'f', dataset_stats_id=low.REFERENCE_SHA)
                    values.update({m + '_' + v: 1.0 for m in ('IS', 'fid', 'kid') for v in low.VARIANTS})
                    np.savez(directory / 'report_260.npz', **values)
                    for i in range(20):
                        (directory / 'ckpt_260' / ('statistics_%d.npz' % i)).touch()
                    (output / 'logs' / (low.config_name(n, count, arm, method, seed) + '_timing.json')).write_text(
                        json.dumps(dict(returncode=0, elapsed_seconds=60)))
                    low.report(n, count, arm, method, seed)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual((summary['status'], summary['completed_cells']), ('completed', 36))
                self.assertEqual(summary['evaluation_elapsed_seconds'], 2160)
                self.assertFalse(summary['external_cells_validated'])
                self.assertEqual({c['sampling_seed'] for c in summary['cells']}, {0, 1})
                self.assertEqual(sorted(c['nfe'] for c in summary['cells']), sorted(n + 1 for n, *_ in low.CELLS))
                values['nfe'] = 129  # a cell from the wrong budget must be rejected
                np.savez(directory / 'report_260.npz', **values)
                with self.assertRaises(RuntimeError):
                    low.report(n, count, arm, method, seed)
                values['nfe'] = n + 1
                np.savez(directory / 'report_260.npz', **values)
                config['eval']['sampling_seed'] = 0  # the other sampling seed must be rejected
                (directory / 'manifest.json').write_text(json.dumps({'config': config, 'source_sha256': 'source'}))
                with self.assertRaisesRegex(RuntimeError, 'Unexpected report config: eval'):
                    low.report(n, count, arm, method, seed)

    def test_run_routes_seed_and_records_timing_even_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'logs').mkdir()
            cell = (32, 10000, 'fox', 'ode_heun', 1)
            for rc in (0, 7):
                with patch.object(low, 'OUTPUT', output), patch.object(low.subprocess, 'run') as execute:
                    execute.return_value = subprocess.CompletedProcess([], rc)
                    if rc:
                        with self.assertRaises(subprocess.CalledProcessError):
                            low.run(*cell)
                    else:
                        low.run(*cell)
                    command = execute.call_args.args[0]
                    self.assertIn('--config=celeba_final_ode_lownfe_eval.py:ode_heun_32_10000_fox_seed1', command)
                    self.assertIn('--eval_folder=' + low.folder(*cell), command)
                    timing = json.loads((output / 'logs' / (low.config_name(*cell) + '_timing.json')).read_text())
                    self.assertEqual(timing['returncode'], rc)
                    self.assertGreaterEqual(timing['elapsed_seconds'], 0)

    def test_initial_states_are_shared_across_methods_and_budgets_within_seed(self):
        import torch
        import ode_sampling
        import sde_lib
        from check_celeba_ode import GaussianEps
        from reproducibility import derive_seed, isolated_torch_rng
        torch.set_num_threads(1)
        for arm in ('vp', 'fox'):
            first = {}
            for seed in (0, 1):
                for n in low.BUDGETS:
                    for method in low.METHODS:
                        process = sde_lib.VPSDE(N=n) if arm == 'vp' else sde_lib.FoxVPSDE(
                            u=-5.0, kernel='matern32', matern_length_scale=346.4101615138,
                            target_terminal_variance=1.0, N=n)
                        process.sampling_logsnr_min, process.sampling_logsnr_max = -10., 10.575749165617202
                        network = GaussianEps(process)
                        with isolated_torch_rng(derive_seed(seed, 'sampling-round', 0)):
                            outputs, nfe = ode_sampling.get_vp_ode_sampler(process, (2, 1, 2, 2),
                                lambda x: x, method, device='cpu')(network)
                        self.assertEqual(nfe, n + 1)
                        self.assertTrue(all(torch.isfinite(x).all() for x in outputs.values()))
                        initial = network.calls[0][0]
                        if seed in first:
                            torch.testing.assert_close(initial, first[seed], rtol=0, atol=0)
                        else:
                            first[seed] = initial
            self.assertFalse(torch.equal(first[0], first[1]))

    def test_snapshot_relabels_profile_and_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            def fake(source, out):
                final.write_json(Path(out) / 'snapshot.json', dict(profile='final_ode', cells=final.CELLS, source_sha256='s'))
            with patch.object(final, 'snapshot', fake):
                low.snapshot('src', output)
            saved = json.loads((output / 'snapshot.json').read_text())
            self.assertEqual(saved['profile'], 'final_ode_lownfe')
            self.assertEqual([tuple(c) for c in saved['cells']], low.CELLS)
            self.assertEqual(saved['source_sha256'], 's')
            self.assertEqual(saved['sampling_seeds'], [0, 1])
            self.assertEqual(saved['methods'], list(low.METHODS))
            self.assertEqual(len(saved['external_cells']), 4)


class LowQueue(QueueChecks):
    def test_queue_completes_in_plan_order(self):
        result, calls, output = self.run_queue(profile='final_ode_lownfe', cells=low.CELLS)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([x for x in calls if x.startswith('eval ')], ['eval {} {} {} {} {}'.format(*c) for c in low.CELLS])
        self.assertEqual(sum(x.startswith('meta report ') for x in calls), 36)
        for c in low.CELLS:
            n, count, arm, method, seed = c
            self.assertTrue((output / 'logs' / '{}_n{}_{}samples_{}_seed{}.log'.format(method, n, count, arm, seed)).exists())
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')

    def test_queue_stops_on_failure(self):
        result, calls, output = self.run_queue(fail='32:10000:fox', profile='final_ode_lownfe', cells=low.CELLS)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(x.startswith('eval ') for x in calls), 10)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_host_profile_wiring(self):
        text = LAUNCHER.read_text()
        for needle in ('--final-ode-lownfe', 'celeba_final_ode_lownfe_eval.py',
                       'tests/check_celeba_final_ode_lownfe.py', 'celeba_final_ode_lownfe_20261102',
                       'celeba-final-ode-lownfe-20261102'):
            self.assertIn(needle, text)

    def test_missing_plan_seed_stops_before_generation(self):
        result, calls, output = self.run_queue(profile='final_ode_lownfe', cells=[low.CELLS[0][:-1]])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(x.startswith('eval ') for x in calls))
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_low_profile_host_disk_guard_and_readonly_inputs(self):
        source, inputs, binary, output = [self.root / name for name in ('source', 'inputs', 'bin', 'output')]
        for directory in (source, inputs, binary):
            directory.mkdir()
        shutil.copy2(str(LAUNCHER), str(source))
        for relative in ('run_e3b_splice_eval.sh', 'e3b_final50k.py', 'main.py',
                         'celeba_final_ode_eval.py', 'celeba_final_ode_lownfe_eval.py', 'ode_sampling.py'):
            (source / relative).write_text('fixture\n')
        for relative in (low.final.REFERENCE, 'celeba/list_eval_partition.txt',
                         'workdirs/' + low.ARMS['vp'] + '/checkpoints/checkpoint_260.pth',
                         'workdirs/' + low.ARMS['fox'] + '/checkpoints/checkpoint_260.pth'):
            path = inputs / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n')
        (inputs / 'celeba/img_align_celeba').mkdir()
        docker = binary / 'docker'
        docker.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['DOCKER_CALLS']).open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
output = Path(os.environ['CELEBA_FINAL_ODE_LOWNFE_OUTPUT'])
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
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
                   DOCKER_CALLS=str(calls), CELEBA_QUEUE_PROFILE='final_ode_lownfe',
                   CELEBA_FINAL_ODE_INPUT_REPO=str(inputs), CELEBA_FINAL_ODE_LOWNFE_OUTPUT=str(output),
                   TFDS_DATA_DIR='/scratch/tfds', TORCH_EXTENSIONS_DIR='/scratch/extensions',
                   XDG_CACHE_HOME='/scratch/cache', CUDA_VISIBLE_DEVICES='3',
                   AVAILABLE_BYTES=str(119 * 1024 ** 3))
        command = ['bash', str(source / LAUNCHER.name), '--final-ode-lownfe', '3']
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Need at least 120 GiB', result.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(any(json.loads(line)[0] == 'run' for line in calls.read_text().splitlines()))
        env['AVAILABLE_BYTES'] = str(120 * 1024 ** 3)
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        launch = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertEqual(launch[launch.index('--gpus') + 1], 'device=3')
        self.assertEqual(launch[launch.index('--name') + 1], 'celeba-final-ode-lownfe-20261102')
        for setting in ('CUDA_VISIBLE_DEVICES=0', 'TFDS_DATA_DIR=/scratch/tfds',
                        'TORCH_EXTENSIONS_DIR=/scratch/extensions', 'XDG_CACHE_HOME=/scratch/cache',
                        'CELEBA_QUEUE_PROFILE=final_ode_lownfe'):
            self.assertIn(setting, launch)
        for arm in low.ARMS.values():
            self.assertIn('type=bind,src={}/workdirs/{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(
                inputs, arm, arm), launch)
        self.assertIn('type=bind,src=' + str(output / 'runtime') + ',dst=/workspace/score_sde_pytorch,readonly', launch)
        self.assertEqual(launch[-3:], ['sha256:fixture', LAUNCHER.name, '_container'])
        old_calls = calls.read_text()
        again = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(calls.read_text(), old_calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
