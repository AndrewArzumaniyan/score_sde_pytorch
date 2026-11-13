"""CPU checks for the GPU 2 queue and its launcher; no GPU, data or checkpoints."""
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
import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_stage0_eval as stage0
import gpu2_queue_eval as q
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

LO = 9.115429865459795
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'


def resolved_manifest(config, protocol):
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256=protocol)


def factorial(get, seed=42, freq=5000):
    config = get()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.seed = seed
    config.training.snapshot_freq = freq
    return config


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        for path in ('logs', 'previous', 'runs/a', 'runs/vp', 'runs/fox'):
            (self.root / path).mkdir(parents=True)
        for key, arm in (('afhq_a', 'a'), ('afhq_vp', 'vp'), ('afhq_fox', 'fox')):
            saved = resolved_manifest(afhq.get_config(arm), 'protocol-' + arm)
            (inputs / (key + '.json')).write_text(json.dumps(saved))
            (self.root / 'runs' / arm / 'run_manifest.json').write_text(json.dumps(saved))
        (inputs / 'celeba_b.json').write_text(json.dumps(resolved_manifest(factorial(get_b), 'protocol-b')))
        (inputs / (followup.WORKDIRS['A'] + '.json')).write_text(json.dumps(resolved_manifest(factorial(get_a), 'a')))
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (q, 'PREVIOUS', self.root / 'previous'), (curve, 'ROOT', self.root),
                                      (curve, 'RUNS', self.root / 'runs'), (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def result(self, directory, name, value):
        (directory / (name + '_result.json')).write_text(json.dumps(dict(fid_tweedie_xt=value)))


def without_range(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    resolved['eval'].pop('begin_ckpt')
    resolved['eval'].pop('end_ckpt')
    return resolved


class Plan(Base):
    def test_cells_in_the_requested_order(self):
        names = list(q.CELLS)
        self.assertEqual(names[:8], ['afhq_a_c5_lo', 'afhq_a_c6_lo', 'afhq_a_c7_lo', 'afhq_a_c8_lo', 'afhq_vp_c5_lo',
                                     'afhq_vp_c6_lo', 'celeba_fox_c140_lo', 'celeba_fox_c160_lo'])
        m35 = ['{}_m35_{}_seed{}'.format(prefix, method, seed) for prefix in ('celeba_b_c160', 'afhq_fox_c4', 'afhq_a_cbest')
               for seed in (0, 1) for method in (DPM, HEUN)]
        self.assertEqual(names[8:], m35)
        self.assertEqual(len(names), 20)

    def test_steps(self):
        steps = {name: q.resolve(name)['step'] for name in q.CELLS if q.CELLS[name]['ckpt'] is not None}
        self.assertEqual(steps['afhq_a_c5_lo'], 250001)
        self.assertEqual(steps['afhq_vp_c6_lo'], 300001)
        self.assertEqual(steps['celeba_fox_c140_lo'], 700001)
        self.assertEqual(steps['celeba_b_c160_m35_ode_dpmpp_2m_seed0'], 800001)
        self.assertEqual(steps['afhq_fox_c4_m35_ode_edm_heun_seed1'], 200001)

    def test_afhq_curve_cells_equal_the_first_curve_except_the_checkpoint(self):
        for name, theirs in (('afhq_a_c5_lo', 'a_c4_lo'), ('afhq_vp_c6_lo', 'vp_c4_lo')):
            self.assertEqual(without_range(q.get_config(name)), without_range(curve.get_config(theirs)), name)

    def test_celeba_fox_cells_equal_the_lower_window_curve_of_4744d55(self):
        for ckpt in (140, 160):
            mine = q.get_config('celeba_fox_c{}_lo'.format(ckpt))
            theirs = stage0.native_config('fox')  # celeba_lo_curve_eval.get_config for fox.
            stage0.set_protocol(theirs, 'lo', ckpt)
            theirs.eval.sampling_num_scales = stage0.BUDGET
            self.assertEqual(_canonicalize(mine), _canonicalize(theirs))
            self.assertEqual(without_range(mine), without_range(stage0.get_config('fox_c260_lo')))

    def test_celeba_b_matched35_equals_the_a_cells_of_the_50k_table(self):
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = without_range(q.get_config('celeba_b_c160_m35_{}_seed{}'.format(method, seed)))
                theirs = without_range(lo.get_config(lo.cell_name('A', method, seed)))
                for section in ('sampling', 'eval', 'data'):
                    self.assertEqual(mine[section], theirs[section], (method, seed, section))
                for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
                    self.assertEqual(mine[key], theirs[key], key)

    def test_afhq_matched35_protocol(self):
        for seed in (0, 1):
            for method in (DPM, HEUN):
                config = q.get_config('afhq_fox_c4_m35_{}_seed{}'.format(method, seed))
                celeba = q.get_config('celeba_b_c160_m35_{}_seed{}'.format(method, seed))
                self.assertEqual((config.data.dataset, config.data.centered), ('AFHQV2', True))
                for key in ('method', 'time_grid', 'logsnr_min', 'logsnr_max', 'probability_flow', 'noise_removal'):
                    self.assertEqual(config.sampling[key], celeba.sampling[key], key)
                for key in ('sampling_num_scales', 'num_samples', 'sampling_seed', 'batch_size', 'save_endpoint_states'):
                    self.assertEqual(config.eval[key], celeba.eval[key], key)
                self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples), (34, 50000))
                self.assertEqual(tuple(config.eval.endpoint_denoise_outputs), ('x_t', 'tweedie_xt'))

    def test_every_cell_passes_the_dataset_check_of_the_evaluation(self):
        import endpoint_denoise_eval
        self.result(q.PREVIOUS, 'a_c2_lo', 3.6)
        self.write_selection(8)
        for name in q.CELLS:
            config = q.get_config(name)
            config.eval.batch_size = 0
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})

    def write_selection(self, ckpt):
        (q.LOGS / q.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=ckpt)))

    def test_rule_addendum(self):
        rule = (REPO / 'AFHQV2_STEP_RULE.md').read_text()
        self.assertIn('## Дополнение от 14 ноября 2026: A и VP до 400k', rule)
        self.assertIn('**Кандидаты A и VP:** 100k, 200k, 250k, 300k, 350k, 400k', rule)
        self.assertIn('**150k не входит.**', rule)


class Selection(Base):
    def fill(self, values):
        for (origin, name, _), value in zip(q.A_CANDIDATES, values):
            if value is not None:
                self.result(q.PREVIOUS if origin == 'previous' else q.LOGS, name, value)

    def test_rule_over_100k_to_400k(self):
        self.fill([3.63, 3.49, 3.40, 3.20, 3.30, 3.35])
        result = q.a_selection()
        self.assertTrue(result['complete'])
        self.assertEqual((result['selected_step'], result['selected_ckpt'], result['minimum']), (400000, 8, 3.20))
        self.assertEqual([i['step'] for i in result['inputs']], [100000, 200000, 250000, 300000, 350000, 400000])

    def test_200k_resolves_to_the_continuation_and_100k_to_the_first_run(self):
        for ckpt, source in ((4, 'afhq_a'), (2, 'afhq_a_first'), (6, 'afhq_a')):
            (q.LOGS / q.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=ckpt)))
            spec = q.resolve('afhq_a_cbest_m35_ode_dpmpp_2m_seed0')
            self.assertEqual((spec['ckpt'], spec['source'], spec['work'], spec['step']),
                             (ckpt, source, source, 50000 * ckpt + 1))

    def test_incomplete_selection(self):
        self.fill([3.63, 3.49, 3.40, None, 3.30, 3.35])
        self.assertFalse(q.a_selection()['complete'])
        with self.assertRaisesRegex(ValueError, 'No step selected'):
            q.resolve('afhq_a_cbest_m35_ode_dpmpp_2m_seed0')

    def test_tie_boundary(self):
        self.assertEqual(q.select({100000: 8.1, 200000: 8.3}), 200000)
        self.assertEqual(q.select({100000: 3.0, 200000: 3.21}), 100000)


class Identity(Base):
    def test_checks(self):
        spec = q.resolve('afhq_a_c5_lo')
        q.check_identity(spec, 250001, 'protocol-a')
        with self.assertRaisesRegex(RuntimeError, 'Expected step'):
            q.check_identity(spec, 200001, 'protocol-a')
        with self.assertRaisesRegex(RuntimeError, 'does not belong'):
            q.check_identity(spec, 250001, 'protocol-vp')
        legacy = q.resolve('celeba_fox_c160_lo')
        q.check_identity(legacy, 800001, None, q.KNOWN_IDS[('celeba_fox', 160)])
        with self.assertRaisesRegex(RuntimeError, 'checkpoint id'):
            q.check_identity(legacy, 800001, None, 'other')
        with self.assertRaisesRegex(RuntimeError, 'does not belong'):
            q.check_identity(legacy, 800001, 'some-protocol')


class Report(Base):
    def prepare(self, name, step=None, training=None, stats=None, features=True):
        spec = q.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / q.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range((spec['count'] + 511) // 512):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{q.FEATURES if features else 'other': np.zeros((512, 2048))})
        (directory / 'manifest.json').write_text(json.dumps(dict(config=without_range(q.config_of(spec)))))
        (q.LOGS / 'preflight.json').write_text(json.dumps(dict(dataset_stats_id=dict(AFHQV2='afhq', CELEBA='celeba'))))
        (q.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=spec['count'], nfe=spec['calls'] + 1, checkpoint_step=step or spec['step'],
                      dataset_stats_id=stats or ('afhq' if spec['dataset'] == 'AFHQV2' else 'celeba'),
                      training_protocol_sha256=training or q.training_protocol(spec['source']), checkpoint_id='id')
        for metric in ('IS', 'fid', 'kid'):
            for variant in stage0.VARIANTS:
                values[metric + '_' + variant] = 2.0
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_reports_of_each_kind(self):
        for name in ('afhq_a_c5_lo', 'celeba_fox_c140_lo', 'celeba_b_c160_m35_ode_dpmpp_2m_seed0',
                     'afhq_fox_c4_m35_ode_edm_heun_seed1'):
            self.prepare(name)
            q.report(name)
        summary = json.loads((q.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['completed_cells'], summary['total_cells']), (4, 20))
        self.assertEqual(len((q.LOGS / 'summary.csv').read_text().splitlines()), 5)
        legacy = json.loads(q.result_path('celeba_fox_c140_lo').read_text())
        self.assertEqual(legacy['step'], 700001)

    def test_refuses_wrong_identity_and_missing_features(self):
        for name, changes, message in (('afhq_a_c5_lo', dict(step=200001), 'report identity'),
                                       ('afhq_fox_c4_m35_ode_dpmpp_2m_seed0', dict(stats='celeba'), 'report identity'),
                                       ('celeba_b_c160_m35_ode_dpmpp_2m_seed0', dict(training='other'), 'report identity'),
                                       ('celeba_b_c160_m35_ode_edm_heun_seed0', dict(features=False), 'bootstrap')):
            directory = self.prepare(name, **changes)
            with self.assertRaisesRegex(RuntimeError, message):
                q.report(name)
            shutil.rmtree(str(directory))


class Preflight(Base):
    def run_preflight(self, smoke):
        def load(config, spec):
            return 'model', dict(checkpoint_id='id', step=spec['step'])

        original = q.config_of

        def on_cpu(spec):
            config = original(spec)
            config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
            return config

        with patch('torch.cuda.is_available', lambda: True), patch('torch.cuda.device_count', lambda: 1), \
                patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0'), patch.object(lo, 'check_inception'), \
                patch.object(afhq, 'check_stats'), \
                patch('evaluation.dataset_stats_fingerprint',
                      lambda c: q.REFERENCE_SHA if c.data.dataset == 'CELEBA' else 'afhq'), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(q, 'config_of', on_cpu), patch.object(q, 'load_model', load), \
                patch.object(followup, 'smoke'), patch.object(q, 'endpoint_smoke', side_effect=smoke) as call:
            code = q.preflight()
        return code, json.loads((q.LOGS / 'preflight.json').read_text()), call

    def test_every_kind_passes_its_real_path(self):
        code, saved, call = self.run_preflight(lambda config, model, spec, directory: 3.0)
        self.assertEqual((code, saved['status']), (0, 'passed'))
        seen = [(args[2]['source'], args[2]['kind'], args[2]['method'], args[0].eval.sampling_num_scales,
                 args[0].data.dataset) for args, _ in call.call_args_list]
        self.assertEqual(seen, [('afhq_a', 'curve', DPM, 64, 'AFHQV2'), ('celeba_fox', 'curve', DPM, 64, 'CELEBA'),
                                ('celeba_b', 'm35', DPM, 34, 'CELEBA'), ('afhq_fox', 'm35', DPM, 34, 'AFHQV2'),
                                ('afhq_fox', 'm35', HEUN, 34, 'AFHQV2')])
        self.assertEqual(sorted(saved['endpoint_smoke_fid_512']), sorted(
            ['afhq_a_curve_' + DPM, 'celeba_fox_curve_' + DPM, 'celeba_b_m35_' + DPM, 'afhq_fox_m35_' + DPM,
             'afhq_fox_m35_' + HEUN]))
        self.assertEqual(saved['dataset_stats_id'], dict(AFHQV2='afhq', CELEBA=q.REFERENCE_SHA))
        heun = [args[0] for args, _ in call.call_args_list if args[2]['method'] == HEUN][0]
        self.assertEqual(heun.sampling.time_grid, 'karras')

    def test_a_failure_blocks_only_its_group(self):
        def smoke(config, model, spec, directory):
            if spec['source'] == 'afhq_fox':
                raise ValueError('Endpoint denoise evaluation supports CelebA 64 and CIFAR-10 32.')
            return 3.0

        code, saved, _ = self.run_preflight(smoke)
        self.assertEqual((code, sorted(saved['failures'])), (1, ['afhq_fox_m35']))


class Queue(Base):
    def run_queue(self, blocked=None, candidates=(3.63, 3.49)):
        (q.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=blocked or {})))
        for (origin, name, _), value in zip(q.A_CANDIDATES[:2], candidates):
            self.result(q.PREVIOUS, name, value)
        order = []

        def evaluate(name):
            order.append(name)
            values = dict(afhq_a_c5_lo=3.40, afhq_a_c6_lo=3.20, afhq_a_c7_lo=3.30, afhq_a_c8_lo=3.35)
            self.result(q.LOGS, name, values.get(name, 2.0))
            return True

        with patch.object(q, 'evaluate', evaluate), patch.object(q, 'write_summary') as summary:
            code = q.queue()
        return code, order, summary

    def test_all_cells_in_order_and_the_selection_before_the_best_a(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, list(q.CELLS)))
        selected = json.loads((q.LOGS / q.SELECTION).read_text())
        self.assertEqual((selected['selected_step'], selected['complete']), (400000, True))

    def test_a_blocked_group_does_not_stop_the_others(self):
        code, order, summary = self.run_queue(blocked=dict(celeba_b_m35='bad'))
        self.assertEqual(code, 1)
        self.assertEqual(order, [n for n in q.CELLS if not n.startswith('celeba_b_')])
        summary.assert_called_once_with([n for n in q.CELLS if n.startswith('celeba_b_')])

    def test_blocked_a_curve_leaves_the_best_a_without_a_step(self):
        code, order, summary = self.run_queue(blocked=dict(afhq_a_curve='bad'))
        self.assertEqual(code, 1)
        self.assertFalse(any(n.startswith('afhq_a_') for n in order))
        self.assertFalse(json.loads((q.LOGS / q.SELECTION).read_text())['complete'])

    def test_no_preflight_runs_nothing(self):
        with patch.object(q, 'evaluate') as evaluate:
            self.assertEqual(q.queue(), 1)
        evaluate.assert_not_called()


DOCKER = r"""#!/usr/bin/env bash
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
RUNS = {
    'afhq_a_first': 'afhqv2_rtx_20261108/workdirs/afhqv2_A_vp_pvp_rtx_seed42',
    'afhq_a': 'afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42',
    'afhq_vp': 'afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42',
    'afhq_fox': 'afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42',
    'celeba_fox': 'score_sde_pytorch/workdirs/celeba_fox_matern32_kappa1000_50k',
    'celeba_b': 'celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42',
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
        shutil.copy(str(REPO / 'run_gpu2_queue_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / lo.INCEPTION_MODULE
        previous = self.base / 'afhqv2_curve_20261112_r2/logs'
        for path in (main / 'assets/stats', main / 'celeba', main / 'datasets/afhqv2-64x64', tfhub, previous):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/afhqv2_64_cd017afab009_stats.npz',
                     main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt',
                     tfhub / 'saved_model.pb', previous / 'a_c2_lo_result.json', previous / 'a_c4_lo_result.json'):
            path.write_text('x')
        for work, run in RUNS.items():
            (self.base / run / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (self.base / run / 'run_manifest.json').write_text('manifest of ' + work)
        self.a8 = self.base / RUNS['afhq_a'] / 'checkpoints/checkpoint_8.pth'
        self.container('afhqv2-rtx-a400k', 2, 0, '2,')
        self.container('afhqv2-rtx-vp400k', 99, 0, '1,')
        self.container('celeba-a-800k-h200', 99, 0, '3,')

    def container(self, name, polls, code, gpus):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))
        (self.state / (name + '.gpus')).write_text(gpus)

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           GPU2Q_SERVER_BASE=str(self.base), GPU2Q_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_gpu2_queue_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_waits_for_a_then_starts_with_every_source_read_only(self):
        self.a8.write_text('x')
        result = self.launch('_wait', '2')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run afhqv2-gpu2-queue'])
        args = (self.state / 'afhqv2-gpu2-queue.args').read_text()
        self.assertIn('device=2', args)
        for work, run in RUNS.items():
            self.assertIn('{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(run, work),
                          args)
        self.assertIn('afhqv2_curve_20261112_r2/logs,dst=/workspace/score_sde_pytorch/previous,readonly', args)
        self.assertIn('/datasets/afhqv2-64x64,dst=/workspace/score_sde_pytorch/datasets/afhqv2-64x64,readonly', args)
        inputs = self.base / 'gpu2_queue_20261114/workdirs/_training_inputs'
        self.assertEqual(sorted(p.name for p in inputs.iterdir()),
                         ['afhq_a.json', 'afhq_fox.json', 'afhq_vp.json', 'celeba_b.json'])
        self.assertEqual((inputs / 'afhq_a.json').read_text(), 'manifest of afhq_a')

    def test_failed_or_unfinished_training_starts_nothing(self):
        self.container('afhqv2-rtx-a400k', 0, 1, '2,')
        self.a8.write_text('x')
        self.assertIn(b'did not exit with 0; nothing started', self.launch('_wait', '2').stdout)
        self.container('afhqv2-rtx-a400k', 0, 0, '2,')
        self.a8.unlink()
        self.assertIn(b'missing', self.launch('_wait', '2').stdout)
        self.container('afhqv2-rtx-a400k', 99, 0, '2,')
        self.assertIn(b'still runs; nothing started', self.launch('_wait', '2', GPU2Q_WAIT_MAX_SECONDS='0').stdout)
        self.assertEqual(self.events(), [])

    def test_refusals(self):
        self.assertEqual(self.launch('after-a400k', '3', GPU_NAME='NVIDIA H200 NVL').returncode, 1)
        self.assertEqual(self.launch('after-a400k', '2', '3').returncode, 2)
        self.assertIn(b'runs containers: afhqv2-rtx-vp400k', self.launch('start', '1').stdout)
        self.container('celeba-other', 99, 0, '2,')
        self.assertIn(b'runs containers: celeba-other', self.launch('after-a400k', '2').stdout)
        self.assertEqual(self.events(), [])

    def test_after_a400k_returns_at_once(self):
        self.a8.write_text('x')
        self.assertIn(b'Waiting for afhqv2-rtx-a400k', self.launch('after-a400k', '2').stdout)
        for _ in range(100):
            if self.events():
                break
            time.sleep(0.1)
        self.assertEqual(self.events(), ['run afhqv2-gpu2-queue'])


if __name__ == '__main__':
    unittest.main()
