"""Docker-only regression checks for the overnight low-NFE queue."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LAUNCHER = REPO / 'run_celeba_low_nfe_overnight.sh'
code = LAUNCHER.read_text().split("<<'PY_LOW_NFE'\n", 1)[1].split('\nPY_LOW_NFE', 1)[0]
meta = {'__name__': 'low_nfe_test'}
exec(compile(code, str(LAUNCHER) + ':embedded-python', 'exec'), meta)


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_has_18_unique_cells_and_prioritizes_250_500(self):
        cells = meta['CELLS']
        self.assertEqual(len(cells), 18)
        self.assertEqual(len(set(cells)), 18)
        self.assertEqual(sum(n for n, _, _ in cells), 5100)
        self.assertEqual([n for n, _, _ in cells], [250] * 6 + [500] * 6 + [100] * 6)
        self.assertEqual(cells[:3], [(250, 0, 'none'), (250, 0, 'full'), (250, 0, 'head:-3.3')])

    def test_snapshot_is_isolated_and_detects_code_or_checkpoint_corruption(self):
        source, output = self.root / 'source', self.root / 'output'
        source.mkdir()
        output.mkdir()
        shutil.copy2(str(REPO / 'artifact_utils.py'), str(source))
        (source / 'probe.py').write_text('value = 1\n')
        (source / 'queue.sh').write_text('exit 0\n')
        (source / 'assets/stats').mkdir(parents=True)
        (source / meta['REFERENCE']).write_bytes(b'reference')
        for arm in (meta['VP'], meta['FOX']):
            folder = source / 'workdirs' / arm / 'checkpoints'
            folder.mkdir(parents=True)
            (folder / 'checkpoint_260.pth').write_bytes(arm.encode())
        ref_hash = meta['digest'](source / meta['REFERENCE'])
        with patch.dict(meta, REFERENCE_SHA=ref_hash):
            meta['snapshot'](source, output)
        runtime = output / 'runtime'
        for arm in (meta['VP'], meta['FOX']):
            target = runtime / 'workdirs' / arm / 'checkpoints'
            target.mkdir(parents=True)
            shutil.copy2(str(source / 'workdirs' / arm / 'checkpoints/checkpoint_260.pth'), str(target))
        (source / 'probe.py').write_text('value = 2\n')
        with patch.dict(meta, ROOT=runtime, PIN=output):
            meta['check']()
            self.assertEqual((runtime / 'probe.py').read_text(), 'value = 1\n')
            (runtime / 'queue.sh').write_text('exit 1\n')
            with self.assertRaisesRegex(RuntimeError, 'Pinned input changed'):
                meta['check']()
            (runtime / 'queue.sh').write_text('exit 0\n')
            (runtime / 'workdirs' / meta['VP'] / 'checkpoints/checkpoint_260.pth').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'Pinned input changed'):
                meta['check']()
        with self.assertRaises(FileExistsError):
            meta['snapshot'](source, output)

    def test_eval_launcher_changes_actual_steps_and_folder_together(self):
        binary = self.root / 'bin'
        binary.mkdir()
        shutil.copy2(str(REPO / 'run_e3b_splice_eval.sh'), str(self.root))
        for arm in ('vp', 'fox'):
            folder = self.root / arm / 'checkpoints'
            folder.mkdir(parents=True)
            (folder / 'checkpoint_260.pth').touch()
        executable = binary / 'python'
        executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS"\n')
        executable.chmod(0o755)
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
                   E3B_SPLICE_RUNS='head:-3.3', E3B_SPLICE_SEED='1',
                   E3B_SPLICE_VP_WORKDIR=str(self.root / 'vp'),
                   E3B_SPLICE_FOX_WORKDIR=str(self.root / 'fox'),
                   TFDS_DATA_DIR=str(self.root / 'tfds'),
                   TORCH_EXTENSIONS_DIR=str(self.root / 'extensions'),
                   XDG_CACHE_HOME=str(self.root / 'cache'), ARGS=str(self.root / 'args'))
        for steps in (100, 250, 500, None):
            custom = dict(env)
            if steps is not None:
                custom['E3B_SPLICE_STEPS'] = str(steps)
            result = subprocess.run(['bash', str(self.root / 'run_e3b_splice_eval.sh')],
                                    env=custom, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = (self.root / 'args').read_text().splitlines()
            n = steps or 1000
            self.assertIn('--config.eval.sampling_num_scales=' + str(n), args)
            self.assertTrue(any('_nfe{}_10000samples_batch512_seed1'.format(n) in a for a in args))
        invalid = subprocess.run(['bash', str(self.root / 'run_e3b_splice_eval.sh')],
                                 env=dict(env, E3B_SPLICE_STEPS='0'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(invalid.returncode, 0)

    def run_queue(self, fail='', profile='em', cells=None):
        binary, output = self.root / 'bin', self.root / 'output'
        binary.mkdir()
        (output / 'logs').mkdir(parents=True)
        script = LAUNCHER.read_text().replace('cd /workspace/score_sde_pytorch', 'cd ' + str(self.root))
        script = script.replace('/output/', str(output) + '/')
        (self.root / LAUNCHER.name).write_text(script)
        python = binary / 'python'
        python.write_text('''#!/bin/bash
set -eu
[[ "$2" != - ]] || cat >/dev/null
mode="$3"
echo "meta $mode ${*:4}" >> "$CALLS"
[[ "$FAIL_AT" != "$mode" ]] || exit 7
if [[ "$mode" == plan ]]; then cat "$PLAN"; fi
if [[ "$mode" == run ]]; then
  echo "eval ${*:4}" >> "$CALLS"
  [[ "$FAIL_AT" != "$4:$5:$6" ]]
fi
''')
        python.chmod(0o755)
        (self.root / 'run_e3b_splice_eval.sh').write_text('''#!/bin/bash
set -eu
echo "eval $E3B_SPLICE_STEPS $E3B_SPLICE_SEED $E3B_SPLICE_RUNS $E3B_SPLICE_NUM_SAMPLES" >> "$CALLS"
[[ "$FAIL_AT" != "$E3B_SPLICE_STEPS:$E3B_SPLICE_SEED:$E3B_SPLICE_RUNS" ]]
''')
        plan = self.root / 'plan'
        plan.write_text(''.join(' '.join(map(str, cell)) + '\n' for cell in (cells or meta['CELLS'])))
        calls = self.root / 'calls'
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
                   CALLS=str(calls), PLAN=str(plan), FAIL_AT=fail, CUDA_VISIBLE_DEVICES='0',
                   CELEBA_QUEUE_PROFILE=profile)
        result = subprocess.run(['bash', str(self.root / LAUNCHER.name), '_container'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return result, calls.read_text().splitlines(), output

    def test_queue_success(self):
        result, calls, output = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([line for line in calls if line.startswith('eval ')],
                         ['eval {} {} {} 10000'.format(*cell) for cell in meta['CELLS']])
        self.assertEqual(sum(line.startswith('meta report ') for line in calls), 18)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')
        self.assertEqual((output / 'logs/queue.exitcode').read_text(), '0\n')

    def test_preflight_failure_prevents_all_generation(self):
        result, calls, output = self.run_queue('preflight')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(line.startswith('eval ') for line in calls))
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_plan_failure_cannot_become_empty_success(self):
        result, calls, output = self.run_queue('plan')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(line.startswith('eval ') for line in calls))
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_failed_cell_stops_remaining_queue(self):
        result, calls, output = self.run_queue('250:0:full')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len([line for line in calls if line.startswith('eval ')]), 2)
        self.assertFalse(any('head:' in line for line in calls))
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_physical_gpu_caches_readonly_mounts_and_no_overwrite(self):
        source, binary, output = self.root / 'source', self.root / 'bin', self.root / 'output'
        source.mkdir()
        binary.mkdir()
        shutil.copy2(str(LAUNCHER), str(source))
        for relative in ('run_e3b_splice_eval.sh', 'e3b_final50k.py', 'main.py',
                         meta['REFERENCE'], 'celeba/list_eval_partition.txt',
                         'workdirs/' + meta['VP'] + '/checkpoints/checkpoint_260.pth',
                         'workdirs/' + meta['FOX'] + '/checkpoints/checkpoint_260.pth'):
            p = source / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture\n')
        (source / 'celeba/img_align_celeba').mkdir()
        docker = binary / 'docker'
        docker.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['DOCKER_CALLS']).open('a') as f: f.write(json.dumps(args) + '\\n')
if args[0] == 'inspect':
    if not Path(os.environ['CELEBA_LOW_NFE_OUTPUT']).exists(): sys.exit(1)
    print('{}')
elif args[:2] == ['image', 'inspect']: print('sha256:fixture')
elif args[0] == 'run' and '--rm' in args:
    p = Path(os.environ['CELEBA_LOW_NFE_OUTPUT'])
    for name in ('runtime', 'logs', 'workdirs'): (p / name).mkdir()
elif args[0] == 'run' and '-d' in args: print('fake-container-id')
else: sys.exit('unexpected docker call')
''')
        docker.chmod(0o755)
        for tool, body in (('git', 'exit 0'), ('df', 'printf "Filesystem blocks used available capacity mount\\nfixture 100000000000 0 100000000000 0%% /\\n"')):
            p = binary / tool
            p.write_text('#!/bin/bash\n' + body + '\n')
            p.chmod(0o755)
        calls = self.root / 'docker_calls'
        env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
                   DOCKER_CALLS=str(calls), CELEBA_LOW_NFE_OUTPUT=str(output),
                   TFDS_DATA_DIR='/scratch/tfds', TORCH_EXTENSIONS_DIR='/scratch/extensions',
                   XDG_CACHE_HOME='/scratch/cache', CUDA_VISIBLE_DEVICES='3')
        result = subprocess.run(['bash', str(source / LAUNCHER.name), '3'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        snapshot = next(c for c in commands if c[0] == 'run' and '--rm' in c)
        self.assertNotIn('--gpus', snapshot)
        launch = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertEqual(launch[launch.index('--gpus') + 1], 'device=3')
        for setting in ('CUDA_VISIBLE_DEVICES=0', 'TFDS_DATA_DIR=/scratch/tfds',
                        'TORCH_EXTENSIONS_DIR=/scratch/extensions', 'XDG_CACHE_HOME=/scratch/cache'):
            self.assertIn(setting, launch)
        self.assertIn('type=bind,src=' + str(output / 'runtime') + ',dst=/workspace/score_sde_pytorch,readonly', launch)
        self.assertEqual(launch[-3:], ['sha256:fixture', LAUNCHER.name, '_container'])
        old_calls = calls.read_text()
        again = subprocess.run(['bash', str(source / LAUNCHER.name), '3'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(old_calls, calls.read_text())

    def test_real_low_nfe_routes_and_none_identity(self):
        import torch
        torch.set_num_threads(1)
        import sampling
        import sde_lib
        from splice_model import SplicedEpsModel

        class Zero(torch.nn.Module):
            def forward(self, x, labels):
                return torch.zeros_like(x)

        for steps in (100, 250, 500):
            vp = sde_lib.VPSDE(N=steps)
            vp.sampling_logsnr_min = -10.0
            vp.sampling_logsnr_max = 10.575749165617202
            fox = sde_lib.FoxVPSDE(u=-5.0, kernel='matern_3_2',
                                  matern_length_scale=346.4101615138, target_terminal_variance=1.0)
            plain = Zero().eval()
            kwargs = dict(sde=vp, shape=(2, 3, 2, 2),
                          predictor=sampling.EulerMaruyamaPredictor, corrector=sampling.NoneCorrector,
                          inverse_scaler=lambda x: x, snr=0.17, continuous=True, denoise=True,
                          endpoint_variants=True, time_grid='uniform_logsnr', device='cpu')
            sampler = sampling.get_pc_sampler(**kwargs)
            torch.manual_seed(71)
            baseline, baseline_nfe = sampler(plain)
            self.assertEqual(baseline_nfe, steps + 1)
            for mode in ('none', 'full', 'head'):
                network = SplicedEpsModel(plain, plain, vp, fox, mode,
                                           threshold=-3.3 if mode == 'head' else None).eval()
                torch.manual_seed(71)
                outputs, nfe = sampler(network)
                self.assertEqual(nfe, steps + 1)
                network.checked_route_metadata()
                self.assertEqual(len(network.route_history), steps + 1)
                self.assertTrue(all(torch.isfinite(x).all().item() for x in outputs.values()))
                if mode == 'none':
                    for key in baseline:
                        self.assertTrue(torch.equal(outputs[key], baseline[key]))
                elif mode == 'head':
                    self.assertTrue(network.route_history[0])
                    self.assertFalse(network.route_history[-1])
                else:
                    self.assertTrue(all(network.route_history))

    def test_report_validates_metadata_and_builds_contrasts(self):
        import numpy as np
        runtime, output = self.root / 'runtime', self.root / 'output'
        (output / 'logs').mkdir(parents=True)
        identities = {meta['VP']: 'vp-id', meta['FOX']: 'fox-id'}
        (output / 'logs/preflight.json').write_text(json.dumps({'checkpoint_ids': identities}))
        with patch.dict(meta, ROOT=runtime, OUTPUT=output):
            for route, fid in (('none', 5.0), ('full', 4.5), ('head:-3.3', 4.6)):
                folder = meta['folder'](250, 0, route)
                (folder / 'ckpt_260').mkdir(parents=True)
                config = dict(eval=dict(num_samples=10000, batch_size=512, sampling_num_scales=250,
                    sampling_seed=0, endpoint_denoise_variants=True, splice_enabled=True,
                    splice_mode=route.split(':')[0], splice_lambda=-3.3),
                    sampling=dict(method='pc', predictor='euler_maruyama', corrector='none',
                    probability_flow=False, time_grid='uniform_logsnr', logsnr_min=-10.0,
                    logsnr_max=10.575749165617202, noise_removal=True))
                (folder / 'manifest.json').write_text(json.dumps({'config': config}))
                counts = {'none': (251, 0), 'full': (0, 251), 'head:-3.3': (160, 91)}[route]
                report = dict(num_samples=10000, nfe=251, checkpoint_step=1300002,
                    alt_checkpoint_step=1300002, dataset_stats_id=meta['REFERENCE_SHA'],
                    checkpoint_id='vp-id', alt_checkpoint_id='fox-id',
                    splice_vp_calls=counts[0], splice_fox_calls=counts[1])
                for variant in meta['VARIANTS']:
                    report.update({k + '_' + variant: v for k, v in (('fid', fid), ('IS', 2.0), ('kid', .01))})
                np.savez(str(folder / 'report_260.npz'), **report)
                for i in range(20):
                    (folder / 'ckpt_260' / ('statistics_{}.npz'.format(i))).touch()
                meta['report'](250, 0, route)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['completed_cells'], 3)
            self.assertEqual(summary['status'], 'running')
            self.assertAlmostEqual(summary['point_contrasts'][0]['fid_tweedie_xt'], -0.5)
            self.assertAlmostEqual(summary['point_contrasts'][1]['fid_tweedie_xt'], -0.4)
            folder = meta['folder'](250, 0, 'head:-3.3')
            config['sampling']['probability_flow'] = True
            (folder / 'manifest.json').write_text(json.dumps({'config': config}))
            with self.assertRaisesRegex(RuntimeError, 'Unexpected cell protocol'):
                meta['report'](250, 0, 'head:-3.3')


if __name__ == '__main__':
    if not Path('/.dockerenv').exists():
        raise SystemExit('Run inside Docker.')
    unittest.main(verbosity=2)
