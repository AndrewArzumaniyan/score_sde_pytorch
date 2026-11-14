"""CPU checks for the AFHQv2 VP tail on GPU 1 and its launcher; no GPU, data or checkpoints."""
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

import afhqv2_rtx_train as afhq
import celeba_followup_eval as followup
import gpu1_vp_tail_eval as tail
import gpu2_queue_eval as q
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
from artifact_utils import _canonicalize, build_model_manifest

LO = 9.115429865459795
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'


def manifest(arm):
    config = afhq.get_config(arm)
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256='protocol-' + arm)


def without_range(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    resolved['eval'].pop('begin_ckpt')
    resolved['eval'].pop('end_ckpt')
    return resolved


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        for path in ('logs', 'previous', 'gpu2'):
            (self.root / path).mkdir()
        for key, arm in (('afhq_vp', 'vp'), ('afhq_fox', 'fox'), ('afhq_a', 'a')):
            (inputs / (key + '.json')).write_text(json.dumps(manifest(arm)))
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (tail, 'ROOT', self.root), (tail, 'LOGS', self.root / 'logs'),
                                      (tail, 'PREVIOUS', self.root / 'previous'), (tail, 'GPU2', self.root / 'gpu2')):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def result(self, directory, name, value):
        (directory / (name + '_result.json')).write_text(json.dumps(dict(fid_tweedie_xt=value)))

    def fill(self, values):
        for (origin, name, _), value in zip(tail.VP_CANDIDATES, values):
            if value is not None:
                self.result(dict(previous=tail.PREVIOUS, gpu2=tail.GPU2, local=tail.LOGS)[origin], name, value)

    def select(self, ckpt):
        (tail.LOGS / tail.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=ckpt)))


class Plan(Base):
    def test_cells(self):
        self.assertEqual(list(tail.CELLS), ['afhq_vp_c7_lo', 'afhq_vp_c8_lo'] +
                         ['afhq_vp_cbest_m35_{}_seed{}'.format(m, s) for s in (0, 1) for m in (DPM, HEUN)])
        self.assertEqual([tail.resolve(n)['step'] for n in ('afhq_vp_c7_lo', 'afhq_vp_c8_lo')], [350001, 400001])

    def test_curve_equals_the_vp_cells_of_the_gpu2_queue_except_the_checkpoint(self):
        for name in ('afhq_vp_c7_lo', 'afhq_vp_c8_lo'):
            self.assertEqual(without_range(tail.get_config(name)), without_range(q.get_config('afhq_vp_c5_lo')))
            self.assertEqual(tail.get_config(name).eval.begin_ckpt, tail.resolve(name)['ckpt'])

    def test_matched35_equals_the_afhqv2_cells_of_the_gpu2_queue(self):
        self.select(6)
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = without_range(tail.get_config('afhq_vp_cbest_m35_{}_seed{}'.format(method, seed)))
                fox = without_range(q.get_config('afhq_fox_c4_m35_{}_seed{}'.format(method, seed)))
                for section in ('sampling', 'eval', 'data'):
                    self.assertEqual(mine[section], fox[section], (method, seed, section))
                self.assertEqual(mine['training']['sde'], 'vpsde')
                self.assertEqual((mine['eval']['sampling_num_scales'], mine['eval']['num_samples']), (34, 50000))

    def test_every_cell_passes_the_dataset_check_of_the_evaluation(self):
        import endpoint_denoise_eval
        self.select(8)
        for name in tail.CELLS:
            config = tail.get_config(name)
            config.eval.batch_size = 0
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})


class Selection(Base):
    def test_rule_over_100k_to_400k(self):
        self.fill([3.7148, 3.5060, 3.7049, 3.8868, 3.60, 3.75])
        result = tail.vp_selection()
        self.assertTrue(result['complete'])
        self.assertEqual((result['selected_step'], result['selected_ckpt']), (350000, 7))
        self.assertEqual([i['origin'] for i in result['inputs']], ['previous', 'previous', 'gpu2', 'gpu2', 'local', 'local'])
        self.fill([3.7148, 3.5060, 3.7049, 3.8868, 3.80, 3.90])
        self.assertEqual(tail.vp_selection()['selected_step'], 250000)
        self.fill([3.7148, 3.5060, 3.7049, 3.8868, 3.60, 3.65])
        self.assertEqual(tail.vp_selection()['selected_step'], 400000)  # 3.65 <= 3.506 + 0.2.

    def test_missing_candidate(self):
        self.fill([3.7148, 3.5060, 3.7049, 3.8868, None, 3.65])
        self.assertFalse(tail.vp_selection()['complete'])

    def test_checkpoint_sources(self):
        for ckpt, source in ((2, 'afhq_vp_first'), (4, 'afhq_vp'), (8, 'afhq_vp')):
            self.select(ckpt)
            spec = tail.resolve('afhq_vp_cbest_m35_ode_edm_heun_seed1')
            self.assertEqual((spec['ckpt'], spec['source'], spec['work'], spec['step']), (ckpt, source, source, 50000 * ckpt + 1))
            self.assertEqual(q.training_protocol(spec['source']), 'protocol-vp')


class Report(Base):
    def prepare(self, name, step=None, features=True):
        spec = tail.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / tail.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range((spec['count'] + 511) // 512):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{q.FEATURES if features else 'other': np.zeros((512, 2048))})
        (directory / 'manifest.json').write_text(json.dumps(dict(config=without_range(q.config_of(spec)))))
        (tail.LOGS / 'preflight.json').write_text(json.dumps(dict(dataset_stats_id=dict(AFHQV2='afhq'))))
        (tail.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=spec['count'], nfe=spec['calls'] + 1, checkpoint_step=step or spec['step'],
                      dataset_stats_id='afhq', training_protocol_sha256='protocol-vp', checkpoint_id='id')
        for metric in ('IS', 'fid', 'kid'):
            for variant in q.stage0.VARIANTS:
                values[metric + '_' + variant] = 3.0
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_reports(self):
        self.select(7)
        for name in ('afhq_vp_c7_lo', 'afhq_vp_cbest_m35_ode_edm_heun_seed0'):
            self.prepare(name)
            tail.report(name)
        summary = json.loads((tail.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['completed_cells'], summary['total_cells']), (2, 6))
        self.assertEqual(len((tail.LOGS / 'summary.csv').read_text().splitlines()), 3)

    def test_refusals(self):
        self.select(7)
        for name, changes, message in (('afhq_vp_c8_lo', dict(step=350001), 'report identity'),
                                       ('afhq_vp_cbest_m35_ode_dpmpp_2m_seed1', dict(features=False), 'bootstrap')):
            directory = self.prepare(name, **changes)
            with self.assertRaisesRegex(RuntimeError, message):
                tail.report(name)
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
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0'), patch.object(q.lo, 'check_inception'), \
                patch.object(afhq, 'check_stats'), patch('evaluation.dataset_stats_fingerprint', lambda c: 'afhq'), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(q, 'config_of', on_cpu), patch.object(q, 'load_model', load), \
                patch.object(followup, 'smoke'), patch.object(q, 'endpoint_smoke', side_effect=smoke) as call:
            code = tail.preflight()
        return code, json.loads((tail.LOGS / 'preflight.json').read_text()), call

    def test_curve_and_both_solvers_on_vp_weights(self):
        code, saved, call = self.run_preflight(lambda config, model, spec, directory: 3.0)
        self.assertEqual((code, saved['status']), (0, 'passed'))
        seen = [(args[2]['source'], args[2]['kind'], args[2]['ckpt'], args[2]['method'], args[0].eval.sampling_num_scales,
                 args[0].sampling.time_grid, args[0].training.sde) for args, _ in call.call_args_list]
        self.assertEqual(seen, [('afhq_vp', 'curve', 7, DPM, 64, 'uniform_logsnr', 'vpsde'),
                                ('afhq_vp', 'm35', 8, DPM, 34, 'uniform_logsnr', 'vpsde'),
                                ('afhq_vp', 'm35', 8, HEUN, 34, 'karras', 'vpsde')])

    def test_failure_blocks_its_group(self):
        def smoke(config, model, spec, directory):
            if spec['method'] == HEUN:
                raise ValueError('heun')
            return 3.0

        code, saved, _ = self.run_preflight(smoke)
        self.assertEqual((code, sorted(saved['failures'])), (1, ['afhq_vp_m35']))


class Queue(Base):
    def run_queue(self, blocked=None, local=True):
        (tail.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=blocked or {})))
        self.fill([3.7148, 3.5060, 3.7049, 3.8868, None, None])
        order = []

        def evaluate(name):
            order.append(name)
            if not local and name in ('afhq_vp_c7_lo', 'afhq_vp_c8_lo'):
                return False
            self.result(tail.LOGS, name, dict(afhq_vp_c7_lo=3.60, afhq_vp_c8_lo=3.75).get(name, 3.0))
            return True

        with patch.object(tail, 'evaluate', evaluate), patch.object(tail, 'write_summary') as summary:
            code = tail.queue()
        return code, order, summary

    def test_curve_then_selection_then_matched35(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, list(tail.CELLS)))
        self.assertEqual(json.loads((tail.LOGS / tail.SELECTION).read_text())['selected_step'], 350000)

    def test_failed_curve_cells_leave_no_50k(self):
        code, order, summary = self.run_queue(local=False)
        self.assertEqual((code, order), (1, ['afhq_vp_c7_lo', 'afhq_vp_c8_lo']))
        self.assertFalse(json.loads((tail.LOGS / tail.SELECTION).read_text())['complete'])
        summary.assert_called_once_with(list(tail.CELLS))

    def test_blocked_matched35(self):
        code, order, _ = self.run_queue(blocked=dict(afhq_vp_m35='bad'))
        self.assertEqual((code, order), (1, ['afhq_vp_c7_lo', 'afhq_vp_c8_lo']))


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
                   'else echo "${FREE_MIB:-64000}"; fi\n'),
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
}
RUNS = {'afhq_vp_first': 'afhqv2_rtx_20261108/workdirs/afhqv2_vp_rtx_seed42',
        'afhq_vp': 'afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42'}


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
        shutil.copy(str(REPO / 'run_gpu1_vp_tail_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / q.lo.INCEPTION_MODULE
        previous = self.base / 'afhqv2_curve_20261112_r2/logs'
        gpu2 = self.base / 'gpu2_queue_20261114/logs'
        for path in (main / 'assets/stats', main / 'datasets/afhqv2-64x64', tfhub, previous, gpu2):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/afhqv2_64_cd017afab009_stats.npz', tfhub / 'saved_model.pb',
                     previous / 'vp_c2_lo_result.json', previous / 'vp_c4_lo_result.json',
                     gpu2 / 'afhq_vp_c5_lo_result.json', gpu2 / 'afhq_vp_c6_lo_result.json'):
            path.write_text('x')
        for work, run in RUNS.items():
            (self.base / run / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (self.base / run / 'run_manifest.json').write_text('manifest of ' + work)
        self.vp8 = self.base / RUNS['afhq_vp'] / 'checkpoints/checkpoint_8.pth'
        self.container('afhqv2-rtx-vp400k', 2, 0, '1,')
        self.container('afhqv2-gpu2-queue', 99, 0, '2,')

    def container(self, name, polls, code, gpus):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))
        (self.state / (name + '.gpus')).write_text(gpus)

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           VPTAIL_SERVER_BASE=str(self.base), VPTAIL_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_gpu1_vp_tail_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_waits_for_vp_then_starts_with_read_only_sources(self):
        self.vp8.write_text('x')
        result = self.launch('_wait', '1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run afhqv2-gpu1-vp-tail'])
        args = (self.state / 'afhqv2-gpu1-vp-tail.args').read_text()
        self.assertIn('device=1', args)
        for work, run in RUNS.items():
            self.assertIn('{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(run, work),
                          args)
        self.assertIn('gpu2_queue_20261114/logs,dst=/workspace/score_sde_pytorch/gpu2,readonly', args)
        self.assertIn('afhqv2_curve_20261112_r2/logs,dst=/workspace/score_sde_pytorch/previous,readonly', args)
        self.assertIn('gpu1_vp_tail_eval.py preflight', args)
        inputs = self.base / 'gpu1_vp_tail_20261114/workdirs/_training_inputs'
        self.assertEqual([p.name for p in inputs.iterdir()], ['afhq_vp.json'])

    def test_failed_or_unfinished_vp_starts_nothing(self):
        self.container('afhqv2-rtx-vp400k', 0, 1, '1,')
        self.vp8.write_text('x')
        self.assertIn(b'did not exit with 0; nothing started', self.launch('_wait', '1').stdout)
        self.container('afhqv2-rtx-vp400k', 0, 0, '1,')
        self.vp8.unlink()
        self.assertIn(b'missing', self.launch('_wait', '1').stdout)
        self.container('afhqv2-rtx-vp400k', 99, 0, '1,')
        self.assertIn(b'still runs; nothing started', self.launch('_wait', '1', VPTAIL_WAIT_MAX_SECONDS='0').stdout)
        self.assertEqual(self.events(), [])

    def test_refusals(self):
        self.vp8.write_text('x')
        self.assertEqual(self.launch('after-vp', '3', GPU_NAME='NVIDIA H200 NVL').returncode, 1)
        self.assertEqual(self.launch('after-vp', '1', '2').returncode, 2)
        self.assertIn(b'runs containers: afhqv2-gpu2-queue', self.launch('start', '2').stdout)
        self.container('afhqv2-rtx-vp400k', 0, 0, '1,')
        self.assertIn(b'need 60000', self.launch('start', '1', FREE_MIB='50000').stdout)
        self.assertEqual(self.events(), [])

    def test_after_vp_returns_at_once(self):
        self.vp8.write_text('x')
        self.assertIn(b'Waiting for afhqv2-rtx-vp400k', self.launch('after-vp', '1').stdout)
        for _ in range(100):
            if self.events():
                break
            time.sleep(0.1)
        self.assertEqual(self.events(), ['run afhqv2-gpu1-vp-tail'])


if __name__ == '__main__':
    unittest.main()
