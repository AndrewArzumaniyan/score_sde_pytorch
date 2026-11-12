"""CPU checks for the AFHQv2 curve queue and its launcher; no GPU, data or checkpoints."""
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

import afhqv2_curve_eval as curve
import afhqv2_rtx_train as afhq
import celeba_stage0_eval as stage0
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
from artifact_utils import _canonicalize, build_model_manifest

LO = 9.115429865459795
STATS_ID = 'stats-sha256'


def run_manifest(arm):
    config = afhq.get_config(arm)
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256='protocol-' + arm)


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        (self.root / 'logs').mkdir()
        for arm in afhq.ARMS:
            (self.root / 'runs' / arm).mkdir(parents=True)
            (self.root / 'runs' / arm / 'run_manifest.json').write_text(json.dumps(run_manifest(arm)))
        for module, target, value in ((curve, 'ROOT', self.root), (curve, 'LOGS', self.root / 'logs'),
                                      (curve, 'RUNS', self.root / 'runs')):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class Plan(Base):
    def test_cells_order_and_steps(self):
        self.assertEqual(list(curve.CELLS), ['fox_c4_lo', 'a_c4_lo', 'vp_c4_lo', 'fox_c2_lo', 'a_c2_lo', 'vp_c2_lo',
                                             'fox_c8_lo', 'fox_c12_lo', 'fox_c4_lo_seed1', 'a_c4_lo_seed1',
                                             'vp_c4_lo_seed1', 'fox_c16_lo'])
        self.assertEqual([s['step'] for s in curve.CELLS.values()],
                         [200001, 200001, 200001, 100001, 100001, 100001, 400001, 600001, 200001, 200001, 200001,
                          800001])
        self.assertEqual([s['seed'] for s in curve.CELLS.values()], [0] * 8 + [1] * 3 + [0])

    def test_seed_1_cells_differ_from_seed_0_only_in_the_sampling_seed(self):
        for arm in afhq.ARMS:
            first = _canonicalize(curve.get_config(curve.cell_name(arm, 4)))
            second = _canonicalize(curve.get_config(curve.cell_name(arm, 4, 1)))
            self.assertEqual((first['eval'].pop('sampling_seed'), second['eval'].pop('sampling_seed')), (0, 1))
            self.assertEqual(first, second)
            self.assertNotEqual(curve.eval_folder(curve.cell_name(arm, 4)), curve.eval_folder(curve.cell_name(arm, 4, 1)))

    def test_rule_names_the_noise_cells(self):
        rule = (REPO / 'AFHQV2_STEP_RULE.md').read_text()
        self.assertIn('на выбор шага и на сравнения выше они не влияют', rule)
        self.assertEqual(curve.CELLS['vp_c4_lo']['work'], 'afhqv2_vp_rtx_seed42')
        self.assertEqual(curve.CELLS['fox_c16_lo']['work'], 'afhqv2_fox_matern32_kappa1000_rtx_seed42')

    def test_candidates_match_the_rule(self):
        rule = (REPO / 'AFHQV2_STEP_RULE.md').read_text()
        for line in ('| A | VP с метками log-SNR, распределение уровней VP | 100k, 200k |',
                     '| VP | нативный VP | 100k, 200k |', '| FOX | нативный FOX, κ = 1000, u = −5 | 100k, 200k, 400k, 600k, 800k |'):
            self.assertIn(line, rule)
        steps = {}
        for spec in curve.CELLS.values():
            steps.setdefault(spec['arm'], set()).add((spec['step'] - 1) // 1000)
        self.assertEqual(steps, dict(fox={100, 200, 400, 600, 800}, a={100, 200}, vp={100, 200}))

    def test_protocol_is_stage_0_on_the_common_window(self):
        for name, spec in curve.CELLS.items():
            config = curve.get_config(name)
            self.assertEqual((config.sampling.method, config.sampling.time_grid, config.sampling.probability_flow),
                             ('ode_dpmpp_2m', 'uniform_logsnr', True))
            self.assertEqual((config.sampling.logsnr_min, config.sampling.logsnr_max), (-10.0, LO))
            self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.sampling_seed,
                              config.eval.batch_size), (64, 10000, spec['seed'], 512))
            self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), ('x_t', 'tweedie_xt'))
            self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (spec['ckpt'], spec['ckpt']))
            self.assertEqual((config.data.dataset, config.data.centered), ('AFHQV2', True))

    def test_models_differ_only_in_training_and_model_sections(self):
        configs = {arm: _canonicalize(curve.get_config(curve.cell_name(arm, 4))) for arm in afhq.ARMS}
        for arm in ('a', 'vp'):
            for section in ('sampling', 'eval', 'data'):
                self.assertEqual(configs[arm][section], configs['fox'][section], (arm, section))

    def test_refuses_an_uncentered_or_foreign_run(self):
        for change in (lambda c: c['data'].update(centered=False), lambda c: c.update(seed=43)):
            saved = run_manifest('vp')
            change(saved['config'])
            (self.root / 'runs/vp/run_manifest.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'Unexpected run'):
                curve.get_config('vp_c4_lo')

    def test_identity(self):
        curve.check_identity('a', 200001, 200001, 'protocol-a')
        with self.assertRaisesRegex(RuntimeError, 'Expected step'):
            curve.check_identity('a', 150001, 200001, 'protocol-a')
        with self.assertRaisesRegex(RuntimeError, 'run manifest'):
            curve.check_identity('a', 200001, 200001, 'protocol-vp')


class Report(Base):
    def prepare(self, name, step=None, training=None, stats=STATS_ID):
        spec = curve.CELLS[name]
        directory = self.root / 'workdirs' / spec['work'] / curve.eval_folder(name)
        directory.mkdir(parents=True)
        resolved = json.loads(json.dumps(_canonicalize(curve.get_config(name))))
        resolved['eval'].pop('begin_ckpt')
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved)))
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(dataset_stats_id=STATS_ID, failures={})))
        (curve.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=10000, nfe=65, checkpoint_step=step or spec['step'], dataset_stats_id=stats,
                      training_protocol_sha256=training or 'protocol-' + spec['arm'], checkpoint_id='id')
        for metric in ('IS', 'fid', 'kid'):
            for variant in stage0.VARIANTS:
                values[metric + '_' + variant] = dict(fox=5.0, a=6.0, vp=7.0)[spec['arm']]
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_reports_and_equal_step_contrasts(self):
        for name in ('fox_c4_lo', 'a_c4_lo', 'vp_c4_lo'):
            self.prepare(name)
            curve.report(name)
        summary = json.loads((curve.LOGS / 'summary.json').read_text())
        values = {c['contrast']: c['value'] for c in summary['equal_step_contrasts']}
        self.assertEqual(values, {'FOX_minus_A_200k': -1.0, 'FOX_minus_VP_200k': -2.0, 'A_minus_VP_200k': -1.0})
        self.assertEqual(len((curve.LOGS / 'summary.csv').read_text().splitlines()), 4)
        self.assertEqual(summary['sampling_seed_noise'], {})

    def test_seed_noise_in_the_summary(self):
        for name, value in (('fox_c4_lo', 5.0), ('fox_c4_lo_seed1', 4.9), ('a_c4_lo', 6.0), ('a_c4_lo_seed1', 6.25)):
            curve.result_path(name).write_text(json.dumps(dict(fid_tweedie_xt=value, model='m', step=1, sampling_seed=0,
                lambda_window=[-10.0, LO], fid_x_t=1.0, kid_tweedie_xt=0.0, IS_tweedie_xt=1.0)))
        curve.write_summary()
        summary = json.loads((curve.LOGS / 'summary.json').read_text())
        noise = summary['sampling_seed_noise']
        self.assertEqual(sorted(noise), ['A_200k', 'FOX_200k'])
        self.assertAlmostEqual(noise['FOX_200k']['seed1_minus_seed0'], -0.1)
        self.assertAlmostEqual(noise['FOX_200k']['absolute'], 0.1)
        self.assertAlmostEqual(noise['A_200k']['absolute'], 0.25)
        self.assertEqual([c['contrast'] for c in summary['equal_step_contrasts']], ['FOX_minus_A_200k'])

    def test_seed_1_report(self):
        self.prepare('vp_c4_lo_seed1')
        curve.report('vp_c4_lo_seed1')
        result = json.loads(curve.result_path('vp_c4_lo_seed1').read_text())
        self.assertEqual((result['sampling_seed'], result['step']), (1, 200001))

    def test_refuses_wrong_identity(self):
        for changes in (dict(step=150001), dict(training='protocol-vp'), dict(stats='other')):
            directory = self.prepare('a_c4_lo', **changes)
            with self.assertRaisesRegex(RuntimeError, 'report identity'):
                curve.report('a_c4_lo')
            shutil.rmtree(str(directory))


class Selection(unittest.TestCase):
    def test_latest_within_the_tie_of_the_minimum(self):
        self.assertEqual(curve.select({400000: 3.00, 600000: 3.15, 800000: 3.30}), 600000)
        self.assertEqual(curve.select({400000: 3.30, 600000: 3.15, 800000: 3.00}), 800000)
        self.assertEqual(curve.select({100000: 4.0, 200000: 4.25}), 100000)
        self.assertEqual(curve.select({100000: 4.0, 200000: 4.2}), 200000)
        self.assertEqual(curve.select({100000: 8.1, 200000: 8.3}), 200000)

    def test_rule_example_is_the_tested_one(self):
        self.assertIn('400k = 3.00, 600k = 3.15, 800k = 3.30', (REPO / 'AFHQV2_STEP_RULE.md').read_text())

    def test_selection_and_fox_against_the_best_baseline(self):
        fids = dict(fox_c2_lo=6.0, fox_c4_lo=9.0, fox_c8_lo=3.00, fox_c12_lo=3.15, fox_c16_lo=3.30,
                    a_c2_lo=4.0, a_c4_lo=3.5, vp_c2_lo=3.6, vp_c4_lo=3.9)
        cells = {name: dict(fid_tweedie_xt=value) for name, value in fids.items()}
        result = curve.selection(cells)
        self.assertEqual(result['selected'], dict(FOX=dict(step=600000, fid=3.15), A=dict(step=200000, fid=3.5),
                                                  VP=dict(step=100000, fid=3.6)))
        self.assertEqual(result['baseline'], 'A')
        self.assertAlmostEqual(result['fox_minus_best_baseline'], 3.15 - 3.5)

    def test_seed_1_cells_do_not_move_the_selection(self):
        fids = dict(fox_c2_lo=6.0, fox_c4_lo=9.0, fox_c8_lo=3.00, fox_c12_lo=3.15, fox_c16_lo=3.30,
                    a_c2_lo=4.0, a_c4_lo=3.5, vp_c2_lo=3.6, vp_c4_lo=3.9,
                    fox_c4_lo_seed1=0.1, a_c4_lo_seed1=9.9, vp_c4_lo_seed1=0.1)
        result = curve.selection({name: dict(fid_tweedie_xt=value) for name, value in fids.items()})
        self.assertEqual({k: v['step'] for k, v in result['selected'].items()}, dict(FOX=600000, A=200000, VP=100000))

    def test_no_selection_until_every_candidate_is_in(self):
        cells = {name: dict(fid_tweedie_xt=3.0) for name in curve.CELLS if name != 'fox_c16_lo'}
        result = curve.selection(cells)
        self.assertEqual(sorted(result['selected']), ['A', 'VP'])
        self.assertNotIn('fox_minus_best_baseline', result)


class EndpointPath(Base):
    """The first launch failed in endpoint_denoise_eval on a dataset check that no test reached."""

    def test_endpoint_evaluation_accepts_afhqv2_64(self):
        import endpoint_denoise_eval
        for name in curve.CELLS:
            config = curve.get_config(name)
            config.eval.batch_size = 0  # Stops right after the dataset check, before any sampling.
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})

    def test_the_other_datasets_are_still_accepted_and_others_refused(self):
        import endpoint_denoise_eval
        config = curve.get_config('fox_c4_lo')
        config.eval.batch_size = 0
        for dataset, size, message in (('CELEBA', 64, 'must be positive'), ('CIFAR10', 32, 'must be positive'),
                                       ('AFHQV2', 32, 'supports'), ('LSUN', 64, 'supports')):
            config.data.dataset, config.data.image_size = dataset, size
            with self.assertRaisesRegex(ValueError, message):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})

    def test_endpoint_smoke_runs_one_batch_of_the_cell_protocol(self):
        import endpoint_denoise_eval
        import evaluation
        seen = {}

        def evaluate(config, directory, ckpt, model, sampling_fn, inception, *rest):
            seen.update(samples=config.eval.num_samples, batch=config.eval.batch_size, ckpt=ckpt,
                        window=config.sampling.logsnr_max, dataset=config.data.dataset, fn=callable(sampling_fn))
            Path(directory).mkdir(parents=True, exist_ok=True)
            np.savez(str(Path(directory) / 'report_{}.npz'.format(ckpt)), fid_tweedie_xt=np.asarray(40.0))

        spec = curve.CELLS['fox_c4_lo']
        config = curve.get_config('fox_c4_lo')
        config.device = 'cpu'
        with patch.object(endpoint_denoise_eval, 'evaluate_checkpoint', evaluate), \
                patch.object(evaluation, 'get_inception_model', lambda: 'inception'):
            value = curve.endpoint_smoke(config, 'model', spec, self.root / 'smoke')
        self.assertEqual(value, 40.0)
        self.assertEqual(seen, dict(samples=512, batch=512, ckpt=4, window=LO, dataset='AFHQV2', fn=True))
        self.assertEqual(curve.get_config('fox_c4_lo').eval.num_samples, 10000)  # The cell config is untouched.

    def run_preflight(self, smoke):
        def load(config, spec):
            return 'model', dict(checkpoint_id='id', step=spec['step'])

        original = curve.get_config

        def on_cpu(name):
            config = original(name)
            config.device = 'cpu'
            return config

        with patch('torch.cuda.is_available', lambda: True), patch('torch.cuda.device_count', lambda: 1), \
                patch('torch.cuda.get_device_capability', lambda i: (12, 0)), patch('torch.cuda.empty_cache'), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0'), patch.object(curve.lo, 'check_inception'), \
                patch.object(afhq, 'check_stats'), patch('evaluation.dataset_stats_fingerprint', lambda c: STATS_ID), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(curve, 'get_config', on_cpu), patch.object(curve, 'load_model', load), \
                patch.object(curve.followup, 'smoke'), patch.object(curve, 'endpoint_smoke', side_effect=smoke) as call:
            code = curve.preflight()
        return code, json.loads((curve.LOGS / 'preflight.json').read_text()), call

    def test_preflight_passes_the_endpoint_path_once(self):
        code, saved, call = self.run_preflight(lambda *a: 42.0)
        self.assertEqual((code, saved['status'], saved['endpoint_smoke_fid_512']), (0, 'passed', 42.0))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args[0][2], curve.CELLS['fox_c4_lo'])

    def test_preflight_fails_when_the_endpoint_path_fails(self):
        def smoke(*args):
            raise ValueError('Endpoint denoise evaluation supports CelebA 64 and CIFAR-10 32.')

        code, saved, call = self.run_preflight(smoke)
        self.assertEqual((code, saved['status']), (1, 'failed'))
        self.assertEqual(sorted(saved['failures']), ['a', 'endpoint', 'fox', 'vp'])
        self.assertEqual(call.call_count, 3)


class Queue(Base):
    def run_queue(self, missing=(), blocked=None):
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=blocked or {})))
        order = []

        def evaluate(name):
            order.append(name)
            curve.result_path(name).write_text('{}')
            return True

        with patch.object(curve, 'evaluate', evaluate), \
                patch.object(curve, 'wait_for', lambda spec, deadline: (spec['arm'], spec['ckpt']) not in missing), \
                patch.object(curve, 'write_summary') as summary:
            code = curve.queue()
        return code, order, summary

    def test_all_cells_in_order(self):
        self.assertEqual(self.run_queue()[:2], (0, list(curve.CELLS)))

    def test_fox_800k_missing_at_the_deadline(self):
        code, order, summary = self.run_queue(missing=(('fox', 16),))
        self.assertEqual((code, order), (1, list(curve.CELLS)[:-1]))
        summary.assert_called_once_with(['fox_c16_lo'])

    def test_a_model_blocked_in_preflight_does_not_stop_the_others(self):
        code, order, summary = self.run_queue(blocked=dict(a='bad'))
        self.assertEqual(code, 1)
        self.assertEqual(order, [n for n in curve.CELLS if not n.startswith('a_')])
        summary.assert_called_once_with(['a_c4_lo', 'a_c2_lo', 'a_c4_lo_seed1'])

    def test_seed_1_cells_run_before_fox_800k(self):
        order = self.run_queue()[1]
        self.assertEqual(order[-4:], ['fox_c4_lo_seed1', 'a_c4_lo_seed1', 'vp_c4_lo_seed1', 'fox_c16_lo'])

    def test_no_preflight_runs_nothing(self):
        with patch.object(curve, 'evaluate') as evaluate:
            self.assertEqual(curve.queue(), 1)
        evaluate.assert_not_called()

    def test_wait_for(self):
        spec = curve.CELLS['fox_c16_lo']
        with patch.object(curve, 'POLL_SECONDS', 0):
            self.assertFalse(curve.wait_for(spec, time.monotonic() - 1))
            curve.checkpoint_path(spec).parent.mkdir(parents=True)
            curve.checkpoint_path(spec).write_text('x')
            self.assertTrue(curve.wait_for(spec, time.monotonic() - 1))


DOCKER = r"""#!/usr/bin/env bash
# State in $STATE: NAME.polls (running while > 0), NAME.code, NAME.gpus; events in $STATE/events.
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.polls; do [[ -e "$f" ]] || continue; n="$(basename "$f" .polls)"; (( $(cat "$f") > 0 )) && echo "$n"; done ;;
  inspect)
    [[ -e "$STATE/$last.polls" ]] || { echo; exit 1; }
    case "$*" in
      *State.Running*) n="$(cat "$STATE/$last.polls")"; if (( n > 0 )); then echo $((n - 1)) > "$STATE/$last.polls"; echo true; else echo false; fi ;;
      *ExitCode*) cat "$STATE/$last.code" ;;
      *DeviceIDs*) cat "$STATE/$last.gpus" ;;
      *.Name*) echo "/$last" ;;
    esac ;;
  stop) echo "stop $last" >> "$STATE/events" ;;
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
                   'else echo "${FREE_MIB:-81000}"; fi\n'),
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
        shutil.copy(str(REPO / 'run_afhqv2_curve_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / curve.lo.INCEPTION_MODULE
        self.runs = self.base / 'afhqv2_rtx_20261108/workdirs'
        for path in (main / 'assets/stats', main / 'datasets/afhqv2-64x64', tfhub):
            path.mkdir(parents=True)
        for path in (main / afhq.STATS, tfhub / 'saved_model.pb'):
            path.write_text('x')
        for arm in afhq.ARMS:
            work = self.runs / Path(afhq.WORK[arm]).name
            (work / 'checkpoints').mkdir(parents=True)
            (work / 'run_manifest.json').write_text('x')
        for ckpt in (1, 2, 3, 4):
            (self.runs / 'afhqv2_vp_rtx_seed42/checkpoints/checkpoint_{}.pth'.format(ckpt)).write_text('x')
        self.container('afhqv2-rtx-a-vp', 2, 0, '1,')
        self.container('afhqv2-rtx-fox', 99, 0, '2,')

    def container(self, name, polls, code, gpus):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))
        (self.state / (name + '.gpus')).write_text(gpus)

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           AFHQ_CURVE_SERVER_BASE=str(self.base), AFHQ_CURVE_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_afhqv2_curve_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_waits_for_the_training_to_end_then_starts_without_stopping_anything(self):
        result = self.launch('_wait', '1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run afhqv2-curve-rtx'])
        args = (self.state / 'afhqv2-curve-rtx.args').read_text()
        self.assertIn('device=1', args)
        self.assertIn('TORCH_CUDA_ARCH_LIST=12.0', args)
        self.assertIn('/datasets/afhqv2-64x64,dst=/workspace/score_sde_pytorch/datasets/afhqv2-64x64,readonly', args)
        for arm in afhq.ARMS:
            work = Path(afhq.WORK[arm]).name
            self.assertIn('{},dst=/workspace/score_sde_pytorch/runs/{},readonly'.format(work, arm), args)
            self.assertIn('{0}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{0}/checkpoints,readonly'.format(work),
                          args)

    def test_failed_training_starts_nothing(self):
        self.container('afhqv2-rtx-a-vp', 0, 137, '1,')
        result = self.launch('_wait', '1')
        self.assertIn(b'did not exit with 0; nothing started', result.stdout)
        self.assertEqual(self.events(), [])

    def test_missing_snapshot_starts_nothing(self):
        (self.runs / 'afhqv2_vp_rtx_seed42/checkpoints/checkpoint_4.pth').unlink()
        self.assertIn(b'missing', self.launch('_wait', '1').stdout)
        self.assertEqual(self.events(), [])

    def test_timeout_starts_nothing(self):
        self.container('afhqv2-rtx-a-vp', 99, 0, '1,')
        self.assertIn(b'still runs; nothing started', self.launch('_wait', '1', AFHQ_CURVE_WAIT_MAX_SECONDS='0').stdout)
        self.assertEqual(self.events(), [])

    def test_after_avp_refusals(self):
        self.assertEqual(self.launch('after-avp', '1', GPU_NAME='NVIDIA H200 NVL').returncode, 1)
        self.assertEqual(self.launch('after-avp', '1', '2').returncode, 2)
        self.container('celeba-other', 99, 0, '1,')
        self.assertIn(b'runs containers: celeba-other', self.launch('after-avp', '1').stdout)
        (self.state / 'celeba-other.polls').write_text('0')
        self.container('afhqv2-rtx-a-vp', 0, 0, '1,')
        self.assertIn(b'if it has finished, use start', self.launch('after-avp', '1').stdout)

    def test_start_refuses_a_busy_card(self):
        self.container('afhqv2-rtx-a-vp', 99, 0, '1,')
        self.assertIn(b'runs containers: afhqv2-rtx-a-vp', self.launch('start', '1').stdout)
        self.container('afhqv2-rtx-a-vp', 0, 0, '1,')
        self.assertIn(b'need 60000', self.launch('start', '1', FREE_MIB='30000').stdout)

    def test_after_avp_returns_at_once(self):
        self.assertIn(b'Waiting for afhqv2-rtx-a-vp', self.launch('after-avp', '1').stdout)
        for _ in range(100):
            if self.events():
                break
            time.sleep(0.1)
        self.assertEqual(self.events(), ['run afhqv2-curve-rtx'])


if __name__ == '__main__':
    unittest.main()
