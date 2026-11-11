"""CPU checks for the lower-window curve queue and its launcher; real SDEs and samplers, no checkpoints."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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

import celeba_followup_eval as followup
import celeba_gpu2_eval as gpu2
import celeba_lo_curve_eval as curve
import celeba_stage0_eval as stage0
import run_lib
from artifact_utils import _canonicalize
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from ode_sampling import get_vp_ode_sampler

LO = 9.115429865459795
EXPECTED = ['fox_c260_lo', 'A_c80_lo'] + ['fox_c{}_lo'.format(c) for c in (60, 80, 100, 120, 140, 160)] + \
    ['fox_c160_lo_n256', 'A_c80_lo_n256']


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def on_cpu(config):
    config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
    return config


def a_manifest():
    config = get_a()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.training.snapshot_freq = 5000
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, training_protocol_sha256='a-training')


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        (inputs / (followup.WORKDIRS['A'] + '.json')).write_text(json.dumps(a_manifest()))
        for module, target, value in ((curve, 'ROOT', self.root), (curve, 'LOGS', self.root / 'logs'),
                                      (followup, 'TRAINING_INPUTS', inputs), (stage0, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {'LOCURVE_STOP_AT': ''})
        environment.start()
        self.addCleanup(environment.stop)


def without_ckpt(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    resolved['eval'].pop('begin_ckpt')
    resolved['eval'].pop('end_ckpt')
    return resolved


class Plan(Base):
    def test_cells_in_order_controls_first(self):
        self.assertEqual(list(curve.CELLS), EXPECTED)
        self.assertEqual(list(curve.CELLS)[:len(curve.CONTROLS)], list(curve.CONTROLS))
        self.assertEqual([curve.CELLS[n]['calls'] for n in EXPECTED], [64] * 8 + [256] * 2)
        self.assertEqual(curve.WORKDIRS, {'fox': 'celeba_fox_matern32_kappa1000_50k',
                                          'A': 'celeba_factorial_A_vp_pvp_400k_rtx_seed42'})

    def test_controls_repeat_the_finished_cells(self):
        self.assertEqual(_canonicalize(curve.get_config('fox_c260_lo')), _canonicalize(stage0.get_config('fox_c260_lo')))
        self.assertEqual(_canonicalize(curve.get_config('A_c80_lo')), _canonicalize(gpu2.get_config('A_c80_lo')))

    def test_curve_shares_the_protocol_of_the_fox_control(self):
        control = without_ckpt(curve.get_config('fox_c260_lo'))
        for ckpt in curve.CURVE_CKPTS:
            config = curve.get_config(curve.cell_name('fox', ckpt))
            self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (ckpt, ckpt))
            self.assertEqual(without_ckpt(config), control)

    def test_dense_cells_differ_only_in_calls(self):
        for model, ckpt in curve.LIMIT_CELLS:
            dense = without_ckpt(curve.get_config(curve.cell_name(model, ckpt, 256)))
            plain = without_ckpt(curve.get_config(curve.cell_name(model, ckpt)))
            self.assertEqual(dense['eval'].pop('sampling_num_scales'), 256)
            self.assertEqual(plain['eval'].pop('sampling_num_scales'), 64)
            self.assertEqual(dense, plain)

    def test_explicit_protocol(self):
        for name in EXPECTED:
            config = curve.get_config(name)
            self.assertEqual((config.sampling.method, config.sampling.time_grid, config.sampling.logsnr_min,
                              config.sampling.logsnr_max), ('ode_dpmpp_2m', 'uniform_logsnr', -10.0, LO), name)
            self.assertEqual((config.eval.num_samples, config.eval.sampling_seed, config.eval.batch_size),
                             (10000, 0, 512), name)
            self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), ('x_t', 'tweedie_xt'), name)

    def test_unknown_cell(self):
        with self.assertRaises(ValueError):
            curve.get_config('fox_c40_lo')


class Grids(Base):
    def test_every_cell_spans_the_window_inside_its_support(self):
        for name in EXPECTED:
            spec = curve.CELLS[name]
            config = on_cpu(curve.get_config(name))
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            grid, levels = stage0.check_grid(config, sde, eps, spec['calls'])
            self.assertEqual(len(grid), spec['calls'] + 1, name)

    def test_samplers_make_the_expected_calls(self):
        for name in ('fox_c160_lo', 'fox_c160_lo_n256', 'A_c80_lo_n256'):
            spec = curve.CELLS[name]
            config = on_cpu(curve.get_config(name))
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            z = torch.zeros((1, 3, 64, 64))
            outputs, calls = get_vp_ode_sampler(sde, z.shape, lambda x: x, curve.METHOD, eps, 'cpu')(ZeroEps(), z=z)
            self.assertEqual(calls, spec['calls'] + 1, name)
            self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()), name)


class Report(Base):
    def prepare(self, name, nfe=None, checkpoint_id='id', source='src', shards=None):
        spec = curve.CELLS[name]
        directory = curve.cell_directory(name)
        (directory / 'ckpt_{}'.format(spec['ckpt'])).mkdir(parents=True)
        for index in range(20 if shards is None else shards):
            (directory / 'ckpt_{}'.format(spec['ckpt']) / 'statistics_{}.npz'.format(index)).write_text('x')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=without_ckpt(curve.get_config(name)),
                                                                 source_sha256=source)))
        checkpoints = {'{}_{}'.format(s['model'], s['ckpt']): dict(step=5000 * s['ckpt'] + 1, checkpoint_id='id')
                       for s in curve.CELLS.values()}
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(status='passed', checkpoints=checkpoints,
                                                                   source_sha256='src')))
        (curve.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=10000, nfe=nfe or spec['calls'] + 1, checkpoint_step=5000 * spec['ckpt'] + 1,
                      checkpoint_id=checkpoint_id, dataset_stats_id=curve.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in stage0.VARIANTS:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_reports_and_contrasts(self):
        for name in ('fox_c160_lo', 'fox_c160_lo_n256', 'A_c80_lo', 'A_c80_lo_n256'):
            self.prepare(name)
            curve.report(name)
        summary = json.loads((curve.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['completed_cells'], summary['total_cells']), (4, 10))
        self.assertEqual({c['contrast'] for c in summary['point_contrasts']},
                         {label for label, *_ in curve.CONTRASTS})
        result = json.loads(curve.result_path('fox_c160_lo_n256').read_text())
        self.assertEqual((result['nfe'], result['calls'], result['lambda_window']), (257, 256, [-10.0, LO]))
        self.assertAlmostEqual(summary['controls']['cells']['A_c80_lo']['difference'], 2.5 - 2.57734751701355)

    def test_refuses_wrong_reports(self):
        for name, changes, message in (('fox_c160_lo_n256', dict(nfe=65), 'report identity'),
                                       ('fox_c80_lo', dict(checkpoint_id='other'), 'report identity'),
                                       ('fox_c80_lo', dict(source='other'), 'different sources'),
                                       ('A_c80_lo', dict(shards=19), 'shards')):
            directory = self.prepare(name, **changes)
            with self.assertRaisesRegex(RuntimeError, message):
                curve.report(name)
            shutil.rmtree(str(directory))


class Queue(Base):
    def run_queue(self, controls=None, fail=(), stop=None):
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(status='passed')))
        values = dict({name: reference['value'] for name, reference in curve.CONTROLS.items()}, **(controls or {}))
        order = []

        def evaluate(name):
            order.append(name)
            if name in fail:
                return False
            curve.result_path(name).write_text(json.dumps(dict(fid_tweedie_xt=values.get(name, 3.0))))
            return True

        environment = {'LOCURVE_STOP_AT': '' if stop is None else str(stop)}
        with patch.object(curve, 'evaluate', evaluate), patch.object(curve, 'write_summary') as summary, \
                patch.dict(os.environ, environment):
            code = curve.queue()
        return code, order, summary

    def clear(self):
        for name in curve.CELLS:
            if curve.result_path(name).exists():
                curve.result_path(name).unlink()

    def test_all_cells_in_order(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, EXPECTED))

    def test_control_outside_the_tolerance_stops_everything(self):
        for name, value in (('fox_c260_lo', 2.62), ('A_c80_lo', 2.46)):
            self.clear()
            code, order, summary = self.run_queue(controls={name: value})
            self.assertEqual(code, curve.GATE_EXIT)
            self.assertEqual(order, EXPECTED[:EXPECTED.index(name) + 1])
            summary.assert_called_once_with(gate=False)

    def test_control_within_the_card_tolerance_goes_on(self):
        code, order, _ = self.run_queue(controls={'fox_c260_lo': 2.57, 'A_c80_lo': 2.50})
        self.assertEqual((code, order), (0, EXPECTED))

    def test_verdicts(self):
        for value, verdict in ((2.4948, 'matched'), (2.514, 'matched'), (2.55, 'matched_within_card_tolerance'),
                               (2.5947, 'matched_within_card_tolerance'), (2.5950, 'failed'), (2.39, 'failed')):
            self.assertEqual(curve.control_verdict('fox_c260_lo', value), verdict, value)

    def test_a_failed_control_stops_and_a_failed_cell_does_not(self):
        code, order, _ = self.run_queue(fail=('A_c80_lo',))
        self.assertEqual((code, order), (curve.GATE_EXIT, EXPECTED[:2]))
        self.clear()
        code, order, summary = self.run_queue(fail=('fox_c100_lo',))
        self.assertEqual((code, order), (1, EXPECTED))
        summary.assert_called_with(['fox_c100_lo'], skipped=[])

    def test_stop_time_skips_cells_that_may_not_end_before_it(self):
        now = time.time()
        with patch.object(curve.time, 'time', return_value=now):
            code, order, summary = self.run_queue(stop=int(now + 40 * 60))
        self.assertEqual((code, order), (1, EXPECTED[:8]))
        summary.assert_called_with([], skipped=EXPECTED[8:])

    def test_no_time_at_all_evaluates_nothing(self):
        code, order, _ = self.run_queue(stop=int(time.time()))
        self.assertEqual((code, order), (1, []))

    def test_failed_preflight_runs_nothing(self):
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(status='failed')))
        with patch.object(curve, 'evaluate') as evaluate:
            self.assertEqual(curve.queue(), 1)
        evaluate.assert_not_called()


DOCKER = r"""#!/usr/bin/env bash
# State in $STATE: NAME.running (true/false), NAME.gpus, NAME.started, NAME.exit; events in $STATE/events.
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.running; do [[ -e "$f" ]] || continue; [[ "$(cat "$f")" == true ]] && basename "$f" .running; done ;;
  inspect)
    [[ -e "$STATE/$last.running" ]] || { echo; exit 1; }
    case "$*" in
      *State.Running*) cat "$STATE/$last.running" ;;
      *ExitCode*) cat "$STATE/$last.exit" ;;
      *StartedAt*) cat "$STATE/$last.started" ;;
      *DeviceIDs*) cat "$STATE/$last.gpus" ;;
      *.Name*) echo "/$last" ;;
    esac ;;
  run)
    name=""; prev=""; for arg in "$@"; do [[ "$prev" == --name ]] && name="$arg"; prev="$arg"; done
    [[ -z "$name" ]] && name=check
    echo "run $name" >> "$STATE/events"; printf '%s\n' "$@" > "$STATE/$name.args" ;;
esac
exit 0
"""
STUBS = {
    'git': '#!/usr/bin/env bash\n[[ "$1" == rev-parse ]] && echo commit\nexit 0\n',
    'nvidia-smi': ('#!/usr/bin/env bash\nif [[ "$*" == *name* ]]; then echo "${GPU_NAME:-NVIDIA RTX PRO 6000 Blackwell}"; '
                   'else echo "${FREE_MIB:-97000}"; fi\n'),
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
}


class Launcher(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.bin, self.state, self.base = self.root / 'bin', self.root / 'state', self.root / 'base'
        for path in (self.bin, self.state, self.root / 'worktree'):
            path.mkdir()
        for name, text in dict(STUBS, docker=DOCKER).items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        shutil.copy(str(REPO / 'run_celeba_lo_curve_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        fox = main / 'workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints'
        a_work = self.base / 'celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / 'c4be6b55dc0338e82546082d22184ebdc77d76cf'
        for path in (fox, a_work / 'checkpoints', main / 'assets/stats', main / 'celeba', tfhub):
            path.mkdir(parents=True)
        files = [a_work / 'run_manifest.json', a_work / 'checkpoints/checkpoint_80.pth', tfhub / 'saved_model.pb',
                 main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt']
        files += [fox / 'checkpoint_{}.pth'.format(c) for c in (60, 80, 100, 120, 140, 160, 260)]
        for path in files:
            path.write_text('x')
        self.output = self.base / 'celeba_lo_curve_20261112'

    def container(self, name, running, gpus='1,', started='2030-01-01T00:00:00.000000000Z', exit_code=0):
        for suffix, value in (('running', running), ('gpus', gpus), ('started', started), ('exit', exit_code)):
            (self.state / '{}.{}'.format(name, suffix)).write_text(str(value))

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           LOCURVE_SERVER_BASE=str(self.base), LOCURVE_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_celeba_lo_curve_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_queue_mounts_and_environment(self):
        self.container('afhqv2-curve-rtx', 'true')
        result = self.launch('queue', '1', LOCURVE_ALLOWED_NEIGHBOURS='afhqv2-curve-rtx', LOCURVE_STOP_AT='2000000000')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run celeba-lo-curve-rtx'])
        args = (self.state / 'celeba-lo-curve-rtx.args').read_text().splitlines()
        for expected in ('device=1', 'TORCH_CUDA_ARCH_LIST=12.0', 'CUDA_VISIBLE_DEVICES=0', 'LOCURVE_STOP_AT=2000000000'):
            self.assertIn(expected, args)
        mounts = '\n'.join(args)
        self.assertIn('celeba_fox_matern32_kappa1000_50k/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/'
                      'celeba_fox_matern32_kappa1000_50k/checkpoints,readonly', mounts)
        self.assertIn('celeba_factorial_A_vp_pvp_400k_rtx_seed42/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/'
                      'celeba_factorial_A_vp_pvp_400k_rtx_seed42/checkpoints,readonly', mounts)
        self.assertTrue((self.output / 'workdirs/_training_inputs/celeba_factorial_A_vp_pvp_400k_rtx_seed42.json').exists())

    def test_queue_refuses_neighbours_memory_and_cards(self):
        self.container('afhqv2-curve-rtx', 'true')
        self.assertIn(b'runs containers: afhqv2-curve-rtx', self.launch('queue', '1').stdout)
        self.assertIn(b'need 60000', self.launch('queue', '1', LOCURVE_ALLOWED_NEIGHBOURS='afhqv2-curve-rtx',
                                                 FREE_MIB='47000').stdout)
        self.assertIn(b'not an RTX', self.launch('queue', '3', GPU_NAME='NVIDIA H200 NVL').stdout)
        self.assertIn(b'Unix time', self.launch('queue', '1', LOCURVE_STOP_AT='23:15').stdout)
        (self.base / 'score_sde_pytorch/workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_120.pth').unlink()
        self.assertIn(b'Missing input', self.launch('queue', '1', LOCURVE_ALLOWED_NEIGHBOURS='afhqv2-curve-rtx').stdout)
        self.assertEqual(self.events(), [])

    def test_wait_starts_after_a_new_run_ends_with_zero(self):
        self.container('celeba-pair43-lo-fox', 'false')
        result = self.launch('_wait', '1', str(int(time.time())))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run celeba-lo-curve-rtx'])

    def test_wait_ignores_an_old_run_and_a_missing_container(self):
        self.container('celeba-pair43-lo-fox', 'false', started='2026-11-09T14:19:13.000000000Z')
        result = self.launch('_wait', '1', str(int(time.time())), LOCURVE_WAIT_MAX_SECONDS='0')
        self.assertIn(b'nothing started', result.stdout)
        shutil.rmtree(str(self.state))
        self.state.mkdir()
        result = self.launch('_wait', '1', str(int(time.time())), LOCURVE_WAIT_MAX_SECONDS='0')
        self.assertIn(b'nothing started', result.stdout)
        self.assertEqual(self.events(), [])

    def test_wait_starts_nothing_after_a_failed_run(self):
        self.container('celeba-pair43-lo-fox', 'false', exit_code=1)
        result = self.launch('_wait', '1', str(int(time.time())))
        self.assertIn(b'did not exit with 0', result.stdout)
        self.assertEqual(self.events(), [])

    def test_after_returns_at_once_and_records_the_waiter(self):
        result = self.launch('after', '1', LOCURVE_WAIT_MAX_SECONDS='0')
        self.assertIn(b'Waiting for a new run of celeba-pair43-lo-fox', result.stdout)
        self.assertTrue((self.output / 'logs/wait.pid').exists())


if __name__ == '__main__':
    unittest.main()
