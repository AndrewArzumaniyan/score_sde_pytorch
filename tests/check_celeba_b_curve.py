"""CPU checks for the B curve queue and the launcher of the B continuation; no GPU or checkpoints.

B_TEST_MANIFEST may point to a copy of the real run_manifest.json of arm B.
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

import celeba_b_curve_eval as curve
import celeba_stage0_eval as stage0
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

LO = 9.115429865459795


def b_manifest(protocol):
    real = os.environ.get('B_TEST_MANIFEST')
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
        (self.root / 'continuation/run_manifest.json').write_text(json.dumps(b_manifest('h200-training')))
        for module, target, value in ((curve, 'ROOT', self.root), (curve, 'LOGS', self.root / 'logs'),
                                      (curve, 'TRAINING_INPUTS', inputs), (curve, 'CONTINUATION', self.root / 'continuation'),
                                      (stage0, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class Plan(Base):
    def test_cells_in_order(self):
        self.assertEqual(list(curve.CELLS), ['B_c{}_lo'.format(c) for c in (40, 60, 80, 100, 120, 140, 160)])
        self.assertEqual(curve.CELLS['B_c40_lo']['work'], 'celeba_factorial_B_vp_pfox_50k')
        self.assertEqual({curve.CELLS[n]['work'] for n in list(curve.CELLS)[1:]}, {curve.WORK})
        self.assertEqual([s['step'] for s in curve.CELLS.values()],
                         [200001, 300001, 400001, 500001, 600001, 700001, 800001])

    def test_control_is_the_stage_0_cell(self):
        mine = _canonicalize(curve.get_config('B_c40_lo'))
        theirs = _canonicalize(stage0.get_config('B_c40_lo'))
        self.assertEqual(mine, theirs)

    def test_snapshots_share_the_protocol_of_the_control(self):
        control = _canonicalize(curve.get_config('B_c40_lo'))
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
        config = curve.get_config('B_c160_lo')
        self.assertEqual((config.sampling.method, config.sampling.logsnr_min, config.sampling.logsnr_max),
                         ('ode_dpmpp_2m', -10.0, LO))
        self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.sampling_seed), (64, 10000, 0))
        self.assertEqual(config.training.noise_distribution, 'normalized_fox_time')
        self.assertEqual((config.model.noise_conditioning, config.training.factorial_arm), ('logsnr', 'B_vp_pfox'))


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
                values[metric + '_' + variant] = 8.3
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_reports(self):
        for name in ('B_c40_lo', 'B_c80_lo'):
            self.prepare(name)
            curve.report(name)
        summary = json.loads((curve.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['completed_cells'], summary['total_cells']), (2, 7))
        self.assertAlmostEqual(summary['control']['difference'], 8.3 - curve.CONTROL)

    def test_refuses_wrong_identity(self):
        for name, changes in (('B_c80_lo', dict(step=395001)), ('B_c80_lo', dict(training='source-training')),
                              ('B_c40_lo', dict(checkpoint_id='other')), ('B_c40_lo', dict(training='h200-training'))):
            directory = self.prepare(name, **changes)
            with self.assertRaisesRegex(RuntimeError, 'report identity'):
                curve.report(name)
            shutil.rmtree(str(directory))


class Queue(Base):
    def run_queue(self, control=8.33, missing=(), deadline='3600'):
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(status='passed')))
        order = []

        def evaluate(name):
            order.append(name)
            value = control if name == 'B_c40_lo' else 5.0
            curve.result_path(name).write_text(json.dumps(dict(fid_tweedie_xt=value)))
            return True

        def wait_for(spec, deadline):
            return spec['ckpt'] not in missing

        with patch.object(curve, 'evaluate', evaluate), patch.object(curve, 'wait_for', wait_for), \
                patch.object(curve, 'write_summary') as summary:
            code = curve.queue()
        return code, order, summary

    def test_control_then_snapshots_in_order(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, list(curve.CELLS)))

    def clear(self):
        for name in curve.CELLS:
            if curve.result_path(name).exists():
                curve.result_path(name).unlink()

    def test_control_outside_the_tolerance_stops_everything(self):
        for value in (8.43, 8.22):
            self.clear()
            code, order, summary = self.run_queue(control=value)
            self.assertEqual((code, order), (curve.GATE_EXIT, ['B_c40_lo']))
            summary.assert_called_once_with(gate=False)

    def test_control_within_the_card_tolerance_goes_on(self):
        for value in (8.35, 8.30, 8.42, 8.23):
            self.clear()
            code, order, _ = self.run_queue(control=value)
            self.assertEqual((code, order), (0, list(curve.CELLS)), value)

    def test_verdicts(self):
        cases = ((8.3244, 'matched'), (8.344, 'matched'), (8.305, 'matched'), (8.35, 'matched_within_card_tolerance'),
                 (8.24, 'matched_within_card_tolerance'), (8.4243, 'matched_within_card_tolerance'),
                 (8.4246, 'failed'), (8.2, 'failed'))
        for value, verdict in cases:
            self.assertEqual(curve.control_verdict(value), verdict, value)

    def test_summary_records_the_deviation_and_verdict(self):
        curve.result_path('B_c40_lo').write_text(json.dumps(dict(
            fid_tweedie_xt=8.38, model='B', step=200001, lambda_window=[-10.0, LO], fid_x_t=16.0, kid_tweedie_xt=0.007,
            IS_tweedie_xt=2.5)))
        curve.write_summary()
        control = json.loads((curve.LOGS / 'summary.json').read_text())['control']
        self.assertAlmostEqual(control['difference'], 8.38 - curve.CONTROL)
        self.assertEqual((control['verdict'], control['card_tolerance']), ('matched_within_card_tolerance', 0.1))

    def test_a_missing_snapshot_stops_the_rest(self):
        code, order, summary = self.run_queue(missing=(120, 140, 160))
        self.assertEqual((code, order), (1, ['B_c40_lo', 'B_c60_lo', 'B_c80_lo', 'B_c100_lo']))
        summary.assert_called_once_with(['B_c120_lo', 'B_c140_lo', 'B_c160_lo'])

    def test_failed_preflight_runs_nothing(self):
        (curve.LOGS / 'preflight.json').write_text(json.dumps(dict(status='failed')))
        with patch.object(curve, 'evaluate') as evaluate:
            self.assertEqual(curve.queue(), 1)
        evaluate.assert_not_called()

    def test_wait_for_returns_on_the_file_and_on_the_deadline(self):
        spec = curve.CELLS['B_c60_lo']
        with patch.object(curve, 'POLL_SECONDS', 0):
            self.assertFalse(curve.wait_for(spec, time.monotonic() - 1))
            curve.checkpoint_path(spec).parent.mkdir(parents=True)
            curve.checkpoint_path(spec).write_text('x')
            self.assertTrue(curve.wait_for(spec, time.monotonic() - 1))


DOCKER = r"""#!/usr/bin/env bash
# State in $STATE: NAME.polls (running while > 0), NAME.gpus; events in $STATE/events.
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.polls; do [[ -e "$f" ]] || continue; n="$(basename "$f" .polls)"; (( $(cat "$f") > 0 )) && echo "$n"; done ;;
  inspect)
    [[ -e "$STATE/$last.polls" ]] || { echo; exit 1; }
    case "$*" in
      *State.Running*) n="$(cat "$STATE/$last.polls")"; (( n > 0 )) && echo true || echo false ;;
      *ExitCode*) echo 137 ;;
      *DeviceIDs*) cat "$STATE/$last.gpus" ;;
      *.Name*) echo "/$last" ;;
    esac ;;
  stop) echo "stop $last" >> "$STATE/events"; echo 0 > "$STATE/$last.polls" ;;
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
        shutil.copy(str(REPO / 'run_celeba_b_continue_800k_h200.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        source = main / 'workdirs/celeba_factorial_B_vp_pfox_50k'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / curve.lo.INCEPTION_MODULE
        self.vp = self.base / 'celeba_pair_seed43_h200_20261106/workdirs/celeba_vp_continuous_h200_seed43/checkpoints'
        for path in (source / 'checkpoints', main / 'assets/stats', main / 'celeba', tfhub, self.vp):
            path.mkdir(parents=True)
        for path in (source / 'checkpoints/checkpoint_40.pth', source / 'run_manifest.json', tfhub / 'saved_model.pb',
                     main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt'):
            path.write_text('x')
        self.container('celeba-pair-seed43-h200', 99, '3,')

    def container(self, name, polls, gpus):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.gpus')).write_text(gpus)

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           B800K_SERVER_BASE=str(self.base), B800K_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_celeba_b_continue_800k_h200.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_waits_for_the_vp_snapshot_then_stops_the_pair_then_starts_both(self):
        (self.vp / 'checkpoint_80.pth').write_text('x' * 10)
        result = self.launch('_wait', '3')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['stop celeba-pair-seed43-h200', 'run celeba-b-800k-h200', 'run celeba-b-curve-h200'])
        for name in ('celeba-b-800k-h200', 'celeba-b-curve-h200'):
            args = (self.state / (name + '.args')).read_text().splitlines()
            self.assertIn('TORCH_CUDA_ARCH_LIST=9.0', args)
            self.assertIn('device=3', args)
            self.assertFalse(any('_rtx' in arg for arg in args), args)
        train = (self.state / 'celeba-b-800k-h200.args').read_text()
        self.assertIn('main.py --mode=train --config=celeba_b_continue.py', train)
        evaluation = (self.state / 'celeba-b-curve-h200.args').read_text()
        self.assertIn('celeba_factorial_B_vp_pfox_800k_h200_seed42,dst=/workspace/score_sde_pytorch/continuation,readonly',
                      evaluation)

    def test_no_snapshot_stops_nothing(self):
        result = self.launch('_wait', '3', B800K_WAIT_MAX_SECONDS='0')
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'nothing stopped or started', result.stdout)
        self.assertEqual(self.events(), [])

    def test_pair_that_dies_before_400k_stops_nothing(self):
        self.container('celeba-pair-seed43-h200', 0, '3,')
        result = self.launch('_wait', '3')
        self.assertIn(b'stopped before', result.stdout)
        self.assertEqual(self.events(), [])

    def test_after_vp_refuses_bad_input(self):
        self.assertEqual(self.launch('after-vp', '3', GPU_NAME='NVIDIA RTX PRO 6000').returncode, 1)
        self.assertEqual(self.launch('after-vp').returncode, 1)
        self.assertEqual(self.launch('after-vp', '3', '4').returncode, 2)
        self.container('celeba-other', 99, '3,')
        self.assertIn(b'runs containers: celeba-other', self.launch('after-vp', '3').stdout)

    def test_start_refuses_the_running_pair_unless_named(self):
        self.assertIn(b'runs containers: celeba-pair-seed43-h200', self.launch('start', '3').stdout)
        result = self.launch('start', '3', B800K_ALLOWED_NEIGHBOURS='celeba-pair-seed43-h200', FREE_MIB='96000')
        self.assertIn(b'need 100000', result.stdout)

    def test_after_vp_returns_at_once(self):
        (self.vp / 'checkpoint_80.pth').write_text('x')
        self.assertIn(b'Waiting for', self.launch('after-vp', '3').stdout)
        for _ in range(100):
            if len(self.events()) == 3:
                break
            time.sleep(0.1)
        self.assertEqual(self.events()[0], 'stop celeba-pair-seed43-h200')
        self.assertEqual(len(self.events()), 3)


if __name__ == '__main__':
    unittest.main()
