"""Docker-only numerical, evaluator and queue checks; no checkpoints required."""
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
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import celeba_ode_eval as queue
import ode_sampling
import sampling
import sde_lib


def sde(budget):
    result = sde_lib.VPSDE(N=budget)
    result.sampling_logsnr_min = -10.0
    result.sampling_logsnr_max = queue.LOGSNR_MAX
    return result


def training_manifests(root):
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    from configs.factorial.celeba_b_vp_pfox import get_config as get_b
    (root / 'training_inputs').mkdir()
    for seed in (42, 43, 44):
        for arm, get in (('A', get_a), ('B', get_b)):
            cfg = get()
            cfg.seed = seed
            (root / 'training_inputs/{}_{}.json'.format(seed, arm)).write_text(
                json.dumps({'config': cfg.to_dict()}, default=str))


class GaussianEps(torch.nn.Module):
    def __init__(self, process, mean=.7, variance=.4):
        super().__init__()
        self.process, self.mean, self.variance = process, mean, variance
        self.calls = []

    def forward(self, x, labels):
        self.calls.append((x.clone(), labels.clone()))
        if getattr(self.process, 'noise_conditioning', 'time') == 'logsnr':
            lam = queue.LOGSNR_MAX - (labels / 999) * (queue.LOGSNR_MAX + 10)
            t = self.process.time_from_log_snr(lam)
        else:
            t = labels / 999
        return self.at_time(x, t)

    def at_time(self, x, t):
        a, sigma = self.process.marginal_prob(torch.ones_like(x[:, :1, :1, :1]), t)
        sigma = sigma[:, None, None, None]
        return sigma * (x - a * self.mean) / (a.square() * self.variance + sigma.square())


class NumericalChecks(unittest.TestCase):
    def test_second_order_convergence_to_exact_gaussian_flow(self):
        initial = torch.linspace(-1, 1, 8, dtype=torch.float64).reshape(2, 1, 2, 2)
        for method in ode_sampling.METHODS:
            errors = []
            for budget in (32, 64, 128):
                process = sde(budget)
                network = GaussianEps(process)
                grid = process.sampling_time_grid(1e-3, grid='uniform_logsnr', N=2, dtype=torch.float64)
                a, sig = process.marginal_prob(torch.ones((2, 1, 1, 1), dtype=torch.float64), grid)
                variance = a.flatten().square() * network.variance + sig.square()
                exact = a[-1] * network.mean + torch.sqrt(variance[-1] / variance[0]) * (initial - a[0] * network.mean)
                outputs, nfe = ode_sampling.get_vp_ode_sampler(process, initial.shape, lambda x: x, method, device='cpu')(network, z=initial)
                error = float((outputs['x_t'] - exact).abs().max())
                errors.append(error)
                self.assertEqual(nfe, budget + 1)
                self.assertEqual(len(network.calls), nfe)
                endpoint_score = -network.at_time(outputs['x_t'], grid[-1:].expand(2)) / sig[-1]
                torch.testing.assert_close(outputs['tweedie_xt'],
                    (outputs['x_t'] + sig[-1].square() * endpoint_score) / a[-1], rtol=1e-10, atol=1e-10)
            self.assertGreater(errors[0] / errors[1], 2.5, (method, errors))
            self.assertGreater(errors[1] / errors[2], 2.5, (method, errors))
            print(method, 'exact Gaussian errors:', errors)

    def test_same_prior_and_rng_no_hidden_stochastic_updates(self):
        first = []
        for method in ode_sampling.METHODS:
            process = sde(128)
            process.noise_conditioning = 'logsnr'
            process.conditioning_logsnr_min = -10.0
            process.conditioning_logsnr_max = queue.LOGSNR_MAX
            network = GaussianEps(process)
            torch.manual_seed(17)
            outputs, nfe = ode_sampling.get_vp_ode_sampler(process, (2, 1, 2, 2), lambda x: x, method, device='cpu')(network)
            first.append(network.calls[0][0])
            self.assertEqual(nfe, 129)
            self.assertTrue(all(torch.isfinite(x).all() for x in outputs.values()))
            self.assertAlmostEqual(float(network.calls[0][1][0]), 999., places=2)
            self.assertAlmostEqual(float(network.calls[-1][1][0]), 0., places=2)
            next_draw = torch.randn(2)
            torch.manual_seed(17)
            torch.randn(2, 1, 2, 2)
            torch.testing.assert_close(next_draw, torch.randn(2), rtol=0, atol=0)
        torch.testing.assert_close(first[0], first[1], rtol=0, atol=0)

    def test_guardrails_and_sampling_dispatch(self):
        cfg = NS(sampling=NS(method='ode_heun', probability_flow=True, noise_removal=True,
            corrector='none', time_grid='uniform_logsnr'), training=NS(continuous=True),
            eval=NS(endpoint_denoise_variants=True, endpoint_denoise_outputs=ode_sampling.VARIANTS), device='cpu')
        self.assertTrue(callable(sampling.get_sampling_fn(cfg, sde(128), (2, 1, 2, 2), lambda x: x, 1e-3)))
        cfg.sampling.probability_flow = False
        with self.assertRaises(ValueError):
            sampling.get_sampling_fn(cfg, sde(128), (2, 1, 2, 2), lambda x: x, 1e-3)
        with self.assertRaises(ValueError):
            ode_sampling.get_vp_ode_sampler(sde(3), (2, 1, 2, 2), lambda x: x, 'ode_heun')
        with self.assertRaises(ValueError):
            ode_sampling.get_vp_ode_sampler(sde_lib.FoxVPSDE(), (2, 1, 2, 2), lambda x: x, 'ode_heun')

    @unittest.skipUnless(os.environ.get('DPM_SOLVER_REFERENCE'), 'Optional pinned official reference not supplied')
    def test_matches_pinned_official_dpm_solver(self):
        spec = importlib.util.spec_from_file_location('official_dpm', os.environ['DPM_SOLVER_REFERENCE'])
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            process = sde(32)
            initial = torch.linspace(-1, 1, 8).reshape(2, 1, 2, 2)
            network = GaussianEps(process)
            ours, _ = ode_sampling.get_vp_ode_sampler(process, initial.shape, lambda x: x,
                                                     'ode_dpmpp_2m', device='cpu')(network, z=initial)
            schedule = official.NoiseScheduleVP('linear', continuous_beta_0=.1, continuous_beta_1=20.)
            solver = official.DPM_Solver(lambda x, t: network.at_time(x, t.expand(x.shape[0])),
                                         schedule, algorithm_type='dpmsolver++')
            grid = process.sampling_time_grid(1e-3, grid='uniform_logsnr', N=2, dtype=torch.float64)
            expected = solver.sample(initial, steps=32, t_start=float(grid[0]), t_end=float(grid[-1]),
                                     order=2, method='multistep', skip_type='logSNR', lower_order_final=False,
                                     denoise_to_zero=False, solver_type='dpmsolver')
            torch.testing.assert_close(ours['x_t'], expected, atol=1e-10, rtol=1e-10)
            print('Official DPM++ max difference:', float((ours['x_t'] - expected).abs().max()))
        finally:
            torch.set_default_dtype(previous_dtype)


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def shared_runner(self, fail=''):
        spec = importlib.util.spec_from_file_location('low_nfe_test_helpers', str(REPO / 'tests/check_low_nfe_overnight.py'))
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        return helpers.QueueChecks.run_queue(self, fail=fail, profile='ode', cells=queue.CELLS)

    def test_ode_queue_stops_on_failed_pair(self):
        result, calls, output = self.shared_runner('128:42:B')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([line for line in calls if line.startswith('eval ')],
                         ['eval 128 42 A ode_dpmpp_2m', 'eval 128 42 B ode_dpmpp_2m'])
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')

    def test_ode_queue_all_24_cells(self):
        result, calls, output = self.shared_runner()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(sum(line.startswith('meta report ') for line in calls), 24)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')

    def test_host_ode_profile_uses_six_readonly_weights_and_original_data_caches(self):
        source, data, binary, output = [self.root / n for n in ('code', 'data', 'bin', 'output')]
        for p in (source, data, binary):
            p.mkdir()
        launcher = 'run_celeba_low_nfe_overnight.sh'
        shutil.copy2(str(REPO / launcher), str(source))
        for name in ('main.py', 'e3b_final50k.py', 'run_e3b_splice_eval.sh', 'celeba_ode_eval.py', 'ode_sampling.py'):
            (source / name).write_text('fixture\n')
        (data / 'celeba/img_align_celeba').mkdir(parents=True)
        (data / 'celeba/list_eval_partition.txt').write_text('fixture\n')
        (data / 'assets/stats').mkdir(parents=True)
        (data / queue.REFERENCE).write_bytes(b'reference')
        roots = {42: data, 43: self.root / 'ab43', 44: self.root / 'ab44'}
        for seed, origin in roots.items():
            for arm in ('A', 'B'):
                work = origin / 'workdirs' / queue.model_name(seed, arm)
                (work / 'checkpoints').mkdir(parents=True)
                (work / 'checkpoints/checkpoint_{}.pth'.format(queue.checkpoint(seed))).write_bytes(b'weights')
                (work / 'run_manifest.json').write_text('{}')
        docker = binary / 'docker'
        docker.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
a=sys.argv[1:]
with Path(os.environ['DOCKER_CALLS']).open('a') as f: f.write(json.dumps(a)+'\\n')
out=Path(os.environ['CELEBA_ODE_OUTPUT'])
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
            CELEBA_ODE_INPUT_REPO=str(data), CELEBA_ODE_SEED43_ROOT=str(roots[43]),
            CELEBA_ODE_SEED44_ROOT=str(roots[44]), CELEBA_ODE_OUTPUT=str(output),
            DOCKER_CALLS=str(calls), TFDS_DATA_DIR='/scratch/tfds',
            TORCH_EXTENSIONS_DIR='/scratch/extensions', XDG_CACHE_HOME='/scratch/cache')
        result = subprocess.run(['bash', str(source / launcher), '--ode', '3'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        snap = next(c for c in commands if c[0] == 'run' and '--rm' in c)
        self.assertNotIn('--gpus', snap)
        run = next(c for c in commands if c[0] == 'run' and '-d' in c)
        self.assertEqual(run[run.index('--gpus') + 1], 'device=3')
        self.assertIn('CELEBA_QUEUE_PROFILE=ode', run)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', run)
        self.assertEqual(sum('/checkpoints,readonly' in x for x in run), 6)
        for suffix, target in (('.tfds', '/scratch/tfds'), ('.torch_extensions', '/scratch/extensions'), ('.cache', '/scratch/cache')):
            self.assertIn('type=bind,src={},dst={}'.format(data / suffix, target), run)
        self.assertFalse(any('checkpoint_260' in x for x in run))
        previous = calls.read_text()
        again = subprocess.run(['bash', str(source / launcher), '--ode', '3'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(calls.read_text(), previous)

    def test_all_training_pairs_and_actual_checkpoint_numbers(self):
        self.assertEqual(len(queue.CELLS), 24)
        self.assertEqual(len(set(queue.CELLS)), 24)
        self.assertEqual(sum(c[0] for c in queue.CELLS), 4608)
        self.assertEqual({c[1] for c in queue.CELLS}, {42, 43, 44})
        self.assertEqual(queue.checkpoint(42), 40)
        self.assertEqual(queue.checkpoint(43), 4)
        self.assertEqual(queue.checkpoint(44), 4)
        for a, b in zip(queue.CELLS[::2], queue.CELLS[1::2]):
            self.assertEqual(a[:2] + a[3:], b[:2] + b[3:])
            self.assertEqual((a[2], b[2]), ('A', 'B'))

    def test_configs_reuse_training_model_and_do_not_widen_support(self):
        from configs.factorial.celeba_a_vp_pvp import get_config as get_a
        from configs.factorial.celeba_b_vp_pfox import get_config as get_b
        training_manifests(self.root)
        with patch.object(queue, 'ROOT', self.root):
            for n, seed, arm, method in queue.CELLS:
                cfg = queue.get_config('{}_{}_{}_{}'.format(method, n, seed, arm))
                self.assertEqual(cfg.model.to_dict(), json.loads(json.dumps(
                    (get_a if arm == 'A' else get_b)().model.to_dict(), default=str)))
                self.assertEqual(cfg.sampling.logsnr_max, queue.LOGSNR_MAX)
                self.assertEqual(cfg.eval.sampling_num_scales, n)
                self.assertEqual(cfg.eval.sampling_seed, 0)
                self.assertEqual(tuple(cfg.eval.endpoint_denoise_outputs), ('x_t', 'tweedie_xt'))
                self.assertTrue(cfg.eval.save_endpoint_states)
                self.assertEqual(cfg.eval.end_ckpt, queue.checkpoint(seed))

    def test_report_intake_24_cells_and_12_contrasts(self):
        training_manifests(self.root)
        output = self.root / 'results'
        (output / 'logs').mkdir(parents=True)
        ids = {'{}_{}'.format(seed, arm): '{}_{}'.format(seed, arm) for seed in (42, 43, 44) for arm in ('A', 'B')}
        (output / 'logs/preflight.json').write_text(json.dumps(dict(source_sha256='sources', checkpoint_ids=ids)))
        with patch.object(queue, 'ROOT', self.root), patch.object(queue, 'OUTPUT', output), patch.dict(sys.modules, tensorflow=NS()):
            from artifact_utils import _canonicalize
            for n, seed, arm, method in queue.CELLS:
                cfg = queue.get_config('{}_{}_{}_{}'.format(method, n, seed, arm))
                config = _canonicalize(cfg)
                config['eval'].pop('begin_ckpt')
                config['eval'].pop('end_ckpt')
                path = self.root / 'workdirs' / queue.model_name(seed, arm) / queue.eval_folder(n, seed, arm, method)
                shard = path / 'ckpt_{}'.format(queue.checkpoint(seed))
                shard.mkdir(parents=True)
                (path / 'manifest.json').write_text(json.dumps(dict(config=config, source_sha256='sources')))
                report = dict(num_samples=10000, nfe=n + 1, checkpoint_step=200001,
                    checkpoint_id=ids['{}_{}'.format(seed, arm)], dataset_stats_id=queue.REFERENCE_SHA)
                for variant in queue.VARIANTS:
                    report.update({'IS_' + variant: 2., 'kid_' + variant: .01,
                                   'fid_' + variant: 4. if arm == 'A' else 3.7})
                np.savez(str(path / 'report_{}.npz'.format(queue.checkpoint(seed))), **report)
                for r in range(20):
                    (shard / 'statistics_{}.npz'.format(r)).touch()
                queue.report(n, seed, arm, method)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(len(summary['point_contrasts']), 12)
            self.assertTrue(all(abs(c['B_minus_A'] + .3) < 1e-12 for c in summary['point_contrasts']))
            config['sampling']['logsnr_max'] = 12.
            (path / 'manifest.json').write_text(json.dumps(dict(config=config, source_sha256='sources')))
            with self.assertRaisesRegex(RuntimeError, 'Unexpected ODE report config'):
                queue.report(n, seed, arm, method)

    def test_ode_snapshot_reuses_pinning_and_detects_manifest_corruption(self):
        source, output, inputs = self.root / 'code', self.root / 'results', self.root / 'data'
        source.mkdir()
        output.mkdir()
        inputs.mkdir()
        shutil.copy2(str(REPO / 'artifact_utils.py'), str(source))
        (inputs / 'assets/stats').mkdir(parents=True)
        (inputs / queue.REFERENCE).write_bytes(b'reference')
        for seed in (42, 43, 44):
            for arm in ('A', 'B'):
                work = inputs / 'workdirs' / queue.model_name(seed, arm)
                (work / 'checkpoints').mkdir(parents=True)
                (work / 'checkpoints/checkpoint_{}.pth'.format(queue.checkpoint(seed))).write_bytes(b'checkpoint')
                (work / 'run_manifest.json').write_text('{}')
        with patch.object(queue, 'INPUT_REPO', inputs), patch.object(queue, 'training_root', return_value=inputs), \
             patch.object(queue, 'REFERENCE_SHA', queue.digest(inputs / queue.REFERENCE)):
            queue.snapshot(source, output)
        runtime = output / 'runtime'
        for seed in (42, 43, 44):
            for arm in ('A', 'B'):
                relative = 'workdirs/{}/checkpoints/checkpoint_{}.pth'.format(queue.model_name(seed, arm), queue.checkpoint(seed))
                target = runtime / relative
                target.parent.mkdir(parents=True)
                shutil.copy2(str(inputs / relative), str(target))
        with patch.object(queue, 'ROOT', runtime), patch.object(queue, 'PIN', output):
            queue.check()
            (runtime / 'training_inputs/42_A.json').write_text('{"corrupt": true}')
            with self.assertRaisesRegex(RuntimeError, 'Pinned input changed'):
                queue.check()

    def test_shared_evaluator_two_outputs_float_state_and_cache_reuse(self):
        # Real artifact validation/writer and evaluator; replace only TF/Inception metrics.
        from ml_collections import ConfigDict
        tf = NS(io=NS(gfile=NS(exists=os.path.exists, makedirs=lambda p: Path(p).mkdir(parents=True, exist_ok=True),
                               GFile=open, rename=lambda a, b, overwrite=False: os.replace(a, b), remove=os.remove)),
                errors=NS(AlreadyExistsError=FileExistsError), convert_to_tensor=np.asarray)
        tensor = lambda x: NS(numpy=lambda: np.asarray(x))
        gan = NS(eval=NS(classifier_score_from_logits=lambda x: 2.,
            frechet_classifier_distance_from_activations=lambda a, b: float(np.mean(b)),
            kernel_classifier_distance_from_activations=lambda a, b: tensor(.01)))
        metrics = NS(run_inception_distributed=lambda x, model: dict(
            pool_3=tensor(np.ones((len(x), 2048))), logits=tensor(np.ones((len(x), 3)))),
            load_dataset_stats=lambda *a, **kw: {'pool_3': np.zeros((4, 2048))},
            dataset_stats_fingerprint=lambda cfg: 'reference')
        with patch.dict(sys.modules, tensorflow=tf, tensorflow_gan=gan, evaluation=metrics):
            spec = importlib.util.spec_from_file_location('endpoint_test', str(REPO / 'endpoint_denoise_eval.py'))
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            cfg = ConfigDict(dict(data=dict(dataset='CELEBA', image_size=64, num_channels=3),
                eval=dict(num_samples=3, batch_size=2, sampling_num_scales=4, sampling_seed=0,
                          endpoint_denoise_outputs=('x_t', 'tweedie_xt'), save_endpoint_states=True)))
            calls = []
            def sample(model):
                calls.append(1)
                return dict(x_t=torch.full((2, 3, 64, 64), 1.25), tweedie_xt=torch.ones((2, 3, 64, 64))), 5
            identity = {k: np.asarray(v) for k, v in dict(protocol_sha256='protocol', checkpoint_id='weights',
                checkpoint_step=200001, training_protocol_sha256='training').items()}
            for _ in range(2):
                evaluator.evaluate_checkpoint(cfg, str(self.root), 4, NS(), sample, NS(),
                    'protocol', 'weights', 200001, 'training', identity)
            self.assertEqual(len(calls), 2)  # Two rounds on first pass; second pass reuses cache.
            with np.load(str(self.root / 'ckpt_4/samples_0.npz')) as saved:
                self.assertEqual(float(saved['float_x_t'].max()), 1.25)
                self.assertEqual(saved['samples_x_t'].dtype, np.uint8)
                self.assertNotIn('samples_x_mean', saved.files)
            with np.load(str(self.root / 'report_4.npz')) as saved:
                self.assertEqual(int(saved['num_samples']), 3)
                self.assertEqual(int(saved['nfe']), 5)
                self.assertNotIn('fid_x_mean', saved.files)


if __name__ == '__main__':
    if not Path('/.dockerenv').exists():
        raise SystemExit('Run inside Docker.')
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
