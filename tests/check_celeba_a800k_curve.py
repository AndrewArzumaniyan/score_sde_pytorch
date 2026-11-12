"""CPU checks for the A curve to 800k and its launcher; no GPU or checkpoints.

A800K_TEST_MANIFEST may point to a copy of the real run_manifest.json of the A 400k run.
"""
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

import celeba_a800k_curve_eval as curve
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
import celeba_stage0_eval as stage0
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_b

LO = 9.115429865459795


def b_manifest(protocol):
    real = os.environ.get('A800K_TEST_MANIFEST')
    if real:
        saved = json.loads(Path(real).read_text())
        saved['training_protocol_sha256'] = protocol
        return saved
    config = get_b()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.training.snapshot_freq = 5000
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256=protocol)


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        (self.root / 'continuation').mkdir()
        (inputs / (curve.SOURCE_WORK + '.json')).write_text(json.dumps(b_manifest('source-training')))
        (inputs / (stage0.WORKDIRS['A'] + '.json')).write_text(json.dumps(b_manifest('stage0-training')))
        (self.root / 'continuation/run_manifest.json').write_text(json.dumps(b_manifest('h200-training')))
        for module, target, value in ((curve, 'ROOT', self.root), (curve, 'LOGS', self.root / 'logs'),
                                      (curve, 'TRAINING_INPUTS', inputs), (curve, 'CONTINUATION', self.root / 'continuation'),
                                      (stage0, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class Plan(Base):
    def test_cells_in_order(self):
        self.assertEqual(list(curve.CELLS), ['A_c{}_lo'.format(c) for c in (80, 100, 120, 140, 160)])
        self.assertEqual(curve.CELLS['A_c80_lo']['work'], 'celeba_factorial_A_vp_pvp_400k_rtx_seed42')
        self.assertEqual({curve.CELLS[n]['work'] for n in list(curve.CELLS)[1:]}, {curve.WORK})
        self.assertEqual([s['step'] for s in curve.CELLS.values()], [400001, 500001, 600001, 700001, 800001])

    def test_control_has_the_stage_0_protocol(self):
        mine = _canonicalize(curve.get_config('A_c80_lo'))
        theirs = _canonicalize(stage0.get_config('A_c40_lo'))
        for config, ckpt in ((mine, 80), (theirs, 40)):
            self.assertEqual((config['eval'].pop('begin_ckpt'), config['eval'].pop('end_ckpt')), (ckpt, ckpt))
        for section in ('model', 'data', 'sampling', 'eval'):
            self.assertEqual(mine[section], theirs[section], section)

    def test_snapshots_share_the_protocol_of_the_control(self):
        control = _canonicalize(curve.get_config('A_c80_lo'))
        for name in list(curve.CELLS)[1:]:
            mine = _canonicalize(curve.get_config(name))
            ckpt = curve.CELLS[name]['ckpt']
            self.assertEqual((mine['eval'].pop('begin_ckpt'), mine['eval'].pop('end_ckpt')), (ckpt, ckpt))
            other = json.loads(json.dumps(control))
            other['eval'].pop('begin_ckpt')
            other['eval'].pop('end_ckpt')
            for section in ('model', 'data', 'sampling', 'eval', 'training'):
                self.assertEqual(mine[section], other[section], section)

    def test_explicit_protocol(self):
        config = curve.get_config('A_c160_lo')
        self.assertEqual((config.sampling.method, config.sampling.logsnr_min, config.sampling.logsnr_max),
                         ('ode_dpmpp_2m', -10.0, LO))
        self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.sampling_seed), (64, 10000, 0))
        self.assertEqual((config.model.noise_conditioning, config.training.factorial_arm), ('logsnr', 'A_vp_pvp'))

    def test_endpoint_evaluation_accepts_the_cells(self):
        import endpoint_denoise_eval
        config = curve.get_config('A_c160_lo')
        config.eval.batch_size = 0
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 160, None, None, None, 'p', 'c', 1, 't', {})

    def test_rule_lists_the_candidates(self):
        rule = (REPO / 'CELEBA_A800K_STEP_RULE.md').read_text()
        self.assertIn('| A (VP с метками log-SNR) | 400k, 500k, 600k, 700k, 800k |', rule)
        self.assertIn('600k = 2.50, 700k = 2.62, 800k = 2.75', rule)


class Selection(Base):
    def test_rule_example(self):
        self.assertEqual(curve.select({600000: 2.50, 700000: 2.62, 800000: 2.75}), 700000)
        self.assertEqual(curve.select({400000: 2.58, 800000: 2.78}), 800000)
        self.assertEqual(curve.select({400000: 2.58, 800000: 2.79}), 400000)

    def test_selection_waits_for_every_candidate(self):
        fids = dict(A_c80_lo=2.58, A_c100_lo=2.55, A_c120_lo=2.50, A_c140_lo=2.62, A_c160_lo=2.75)
        cells = {name: dict(fid_tweedie_xt=value) for name, value in fids.items()}
        self.assertEqual(curve.selection(cells)['selected'], dict(A=dict(step=700000, fid=2.62)))
        cells.pop('A_c160_lo')
        self.assertNotIn('selected', curve.selection(cells))


class Report(Base):
    def prepare(self, name, step=None, training=None, checkpoint_id=None):
        spec = curve.CELLS[name]
        directory = self.root / 'workdirs' / spec['work'] / curve.eval_folder(name)
        directory.mkdir(parents=True)
        resolved = json.loads(json.dumps(_canonicalize(curve.get_config(name))))
        resolved['eval'].pop('begin_ckpt')
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved)))
        (curve.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=10000, nfe=65, checkpoint_step=step or spec['step'], dataset_stats_id=curve.REFERENCE_SHA,
                      training_protocol_sha256=training or curve.manifest(spec)['training_protocol_sha256'],
                      checkpoint_id=checkpoint_id or curve.SOURCE_ID)
        for metric in ('IS', 'fid', 'kid'):
            for variant in stage0.VARIANTS:
                values[metric + '_' + variant] = 2.6
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_reports_and_summary(self):
        for name in ('A_c80_lo', 'A_c120_lo'):
            self.prepare(name)
            curve.report(name)
        summary = json.loads((curve.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['completed_cells'], summary['total_cells']), (2, 5))
        self.assertAlmostEqual(summary['control']['difference'], 2.6 - curve.CONTROL)
        self.assertEqual(summary['control']['verdict'], 'matched')
        self.assertEqual(summary['control']['earlier_reference'], 2.57734751701355)
        self.assertEqual(summary['step_selection']['candidates'], [400000, 500000, 600000, 700000, 800000])

    def test_refuses_wrong_identity(self):
        for name, changes in (('A_c120_lo', dict(step=595001)), ('A_c120_lo', dict(training='source-training')),
                              ('A_c80_lo', dict(checkpoint_id='other')), ('A_c80_lo', dict(training='h200-training'))):
            directory = self.prepare(name, **changes)
            with self.assertRaisesRegex(RuntimeError, 'report identity'):
                curve.report(name)
            shutil.rmtree(str(directory))


class Queue(Base):
    def run_queue(self, control=2.59, missing=()):
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(status='passed')))
        order = []

        def evaluate(name):
            order.append(name)
            value = control if name == 'A_c80_lo' else 2.5
            curve.result_path(name).write_text(json.dumps(dict(fid_tweedie_xt=value)))
            return True

        with patch.object(curve, 'evaluate', evaluate), \
                patch.object(curve, 'wait_for', lambda spec, deadline: spec['ckpt'] not in missing), \
                patch.object(curve, 'write_summary') as summary:
            code = curve.queue()
        return code, order, summary

    def clear(self):
        for name in curve.CELLS:
            if curve.result_path(name).exists():
                curve.result_path(name).unlink()

    def test_control_then_snapshots_in_order(self):
        self.assertEqual(self.run_queue()[:2], (0, list(curve.CELLS)))

    def test_control_tolerances(self):
        for value, code in ((2.60, 0), (2.68, 0), (2.49, 0), (2.69, curve.GATE_EXIT), (2.48, curve.GATE_EXIT)):
            self.clear()
            self.assertEqual(self.run_queue(control=value)[0], code, value)

    def test_a_missing_snapshot_stops_the_rest(self):
        code, order, summary = self.run_queue(missing=(140, 160))
        self.assertEqual((code, order), (1, ['A_c80_lo', 'A_c100_lo', 'A_c120_lo']))
        summary.assert_called_once_with(['A_c140_lo', 'A_c160_lo'])


class Preflight(Base):
    def run_preflight(self, smoke, identity=None):
        model_identity = identity or dict(checkpoint_id=curve.SOURCE_ID, step=400001)
        original = curve.get_config

        def on_cpu(name):
            config = original(name)
            config.device = 'cpu'
            return config

        with patch('torch.cuda.is_available', lambda: True), patch('torch.cuda.device_count', lambda: 1), \
                patch('torch.cuda.get_device_capability', lambda i: (9, 0)), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='9.0'), patch.object(curve.lo, 'check_inception'), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(curve, 'get_config', on_cpu), \
                patch.object(stage0, 'load_model', lambda config, work, ckpt: ('model', model_identity)), \
                patch.object(curve.followup, 'smoke'), patch.object(curve, 'endpoint_smoke', side_effect=smoke) as call:
            code = curve.preflight()
        return code, json.loads((curve.LOGS / 'preflight.json').read_text()), call

    def test_passes_the_endpoint_path(self):
        code, saved, call = self.run_preflight(lambda *a: 3.0)
        self.assertEqual((code, saved['status'], saved['endpoint_smoke_fid_512']), (0, 'passed', 3.0))
        self.assertEqual(call.call_args[0][2], curve.CELLS['A_c80_lo'])

    def test_fails_on_the_endpoint_path_or_a_wrong_control(self):
        def smoke(*args):
            raise ValueError('Endpoint denoise evaluation supports CelebA 64 and CIFAR-10 32.')

        code, saved, _ = self.run_preflight(smoke)
        self.assertEqual((code, sorted(saved['failures'])), (1, ['endpoint']))
        code, saved, call = self.run_preflight(lambda *a: 3.0, dict(checkpoint_id='other', step=400001))
        self.assertEqual((code, sorted(saved['failures'])), (1, ['A_c80_lo']))
        call.assert_not_called()


DOCKER = r"""#!/usr/bin/env bash
# State in $STATE: NAME.polls (running while > 0, one less per poll), NAME.code, NAME.gpus; events in $STATE/events.
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
    'nvidia-smi': ('#!/usr/bin/env bash\nif [[ "$*" == *name* ]]; then echo "${GPU_NAME:-NVIDIA H200 NVL}"; '
                   'else echo "${FREE_MIB:-140000}"; fi\n'),
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
        shutil.copy(str(REPO / 'run_celeba_a800k_h200.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        source = self.base / 'celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / curve.lo.INCEPTION_MODULE
        self.b = self.base / 'celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42/checkpoints'
        for path in (source / 'checkpoints', main / 'assets/stats', main / 'celeba', tfhub, self.b):
            path.mkdir(parents=True)
        for path in (source / 'checkpoints/checkpoint_80.pth', source / 'run_manifest.json', tfhub / 'saved_model.pb',
                     main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt'):
            path.write_text('x')
        self.container('celeba-b-800k-h200', 2, 0, '3,')
        self.container('celeba-b-curve-h200', 99, 0, '3,')

    def container(self, name, polls, code, gpus):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))
        (self.state / (name + '.gpus')).write_text(gpus)

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           A800K_SERVER_BASE=str(self.base), A800K_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_celeba_a800k_h200.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_waits_for_b_then_starts_both_beside_the_b_curve(self):
        (self.b / 'checkpoint_160.pth').write_text('x')
        result = self.launch('_wait', '3')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run celeba-a-800k-h200', 'run celeba-a-curve-h200'])
        train = (self.state / 'celeba-a-800k-h200.args').read_text()
        self.assertIn('celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42/checkpoints,'
                      'dst=/workspace/score_sde_pytorch/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42/checkpoints,'
                      'readonly', train)
        self.assertIn('main.py --mode=train --config=celeba_a800k_continue.py', train)
        self.assertIn('TORCH_CUDA_ARCH_LIST=9.0', train)
        self.assertIn('device=3', train)
        evaluation = (self.state / 'celeba-a-curve-h200.args').read_text()
        self.assertIn('celeba_a800k_curve_eval.py preflight', evaluation)

    def test_failed_b_starts_nothing(self):
        (self.b / 'checkpoint_160.pth').write_text('x')
        self.container('celeba-b-800k-h200', 0, 1, '3,')
        self.assertIn(b'B did not exit with 0; nothing started', self.launch('_wait', '3').stdout)
        self.assertEqual(self.events(), [])

    def test_missing_b_snapshot_starts_nothing(self):
        self.assertIn(b'missing', self.launch('_wait', '3').stdout)
        self.assertEqual(self.events(), [])

    def test_timeout_starts_nothing(self):
        self.container('celeba-b-800k-h200', 99, 0, '3,')
        self.assertIn(b'still runs; nothing started', self.launch('_wait', '3', A800K_WAIT_MAX_SECONDS='0').stdout)
        self.assertEqual(self.events(), [])

    def test_after_b_refusals(self):
        self.assertEqual(self.launch('after-b', '3', GPU_NAME='NVIDIA RTX PRO 6000').returncode, 1)
        self.assertEqual(self.launch('after-b', '3', '4').returncode, 2)
        self.container('celeba-other', 99, 0, '3,')
        self.assertIn(b'runs containers: celeba-other', self.launch('after-b', '3').stdout)

    def test_start_needs_memory_beside_the_b_curve(self):
        self.container('celeba-b-800k-h200', 0, 0, '3,')
        self.assertIn(b'need 90000', self.launch('start', '3', FREE_MIB='80000').stdout)
        self.assertEqual(self.launch('start', '3', FREE_MIB='106000').returncode, 0)

    def test_after_b_returns_at_once(self):
        (self.b / 'checkpoint_160.pth').write_text('x')
        self.assertIn(b'Waiting for celeba-b-800k-h200', self.launch('after-b', '3').stdout)
        for _ in range(100):
            if len(self.events()) == 2:
                break
            time.sleep(0.1)
        self.assertEqual(len(self.events()), 2)


if __name__ == '__main__':
    unittest.main()
