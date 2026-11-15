"""CPU checks for CelebA A at 800k (celeba_a800k_eval.py) and its launcher; no GPU, data or checkpoints."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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

import celeba_50k_lo_eval as lo
import celeba_a800k_eval as e
import celeba_followup_eval as followup
import gpu2_a50k_grids as g
import gpu2_queue_eval as q
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a

DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
CURVE_ID = 'sha-a800k'


def resolved_manifest(config, protocol):  # As in tests/check_gpu2_a50k_grids.py.
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


def without_range(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    resolved['eval'].pop('begin_ckpt')
    resolved['eval'].pop('end_ckpt')
    return resolved


class Base(unittest.TestCase):
    lane = 's0'

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        for path in ('logs', 'acurve'):
            (self.root / path).mkdir()
        a_manifest = json.dumps(resolved_manifest(factorial(get_a), 'protocol-a'))
        for key in ('celeba_a_first', 'celeba_a', followup.WORKDIRS['A']):
            (inputs / (key + '.json')).write_text(a_manifest)
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (g, 'ROOT', self.root), (g, 'LOGS', self.root / 'logs'),
                                      (g, 'ACURVE', self.root / 'acurve'), (e, 'ROOT', self.root),
                                      (e, 'LOGS', self.root / 'logs'), (e, 'ACURVE', self.root / 'acurve'),
                                      (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, A800K_LANE=self.lane)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.curve()

    def curve(self, ckpts=(80, 100, 120, 140, 160), checkpoint_id=CURVE_ID):
        cells = [dict(ckpt=c, cell='A_c{}_lo'.format(c), step=5000 * c + 1, fid_tweedie_xt=2.5 + c / 1000,
                      checkpoint_id=checkpoint_id if c == 160 else 'other-{}'.format(c)) for c in ckpts]
        (e.ACURVE / 'summary.json').write_text(json.dumps(dict(status='completed', cells=cells)))


class Plan(Base):
    def test_cells_lanes_and_steps(self):
        names = ['celeba_a_c160_m35_{}_seed{}'.format(m, s) for s in (0, 1) for m in (DPM, HEUN)]
        self.assertEqual(list(e.CELLS), names)
        self.assertEqual(e.lane_cells('s0'), names[:2])
        self.assertEqual(e.lane_cells('s1'), names[2:])
        for name in names:
            spec = e.resolve(name)
            self.assertEqual((spec['source'], spec['work'], spec['ckpt'], spec['step'], spec['count'], spec['calls']),
                             ('celeba_a', 'celeba_a', 160, 800001, 50000, 34))
            self.assertEqual(q.training_protocol(spec['source']), 'protocol-a')
        with patch.dict(os.environ, A800K_LANE='a'):
            with self.assertRaisesRegex(ValueError, 'must be s0 or s1'):
                e.lane()

    def test_configs_equal_the_a500k_cells_except_the_checkpoint(self):
        (g.LOGS / g.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=100)))
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = e.get_config('celeba_a_c160_m35_{}_seed{}'.format(method, seed))
                theirs = g.get_config('celeba_a_cbest_m35_{}_seed{}'.format(method, seed))
                self.assertEqual(without_range(mine), without_range(theirs), (method, seed))
                self.assertEqual((mine.eval.begin_ckpt, mine.eval.end_ckpt, theirs.eval.begin_ckpt), (160, 160, 100))
                self.assertEqual((mine.eval.sampling_seed, mine.eval.num_samples, mine.eval.batch_size,
                                  mine.eval.sampling_num_scales, mine.sampling.logsnr_max, mine.sampling.method),
                                 (seed, 50000, 512, 34, 9.115429865459795, method))
                self.assertEqual(mine.sampling.time_grid, 'karras' if method == HEUN else 'uniform_logsnr')
                self.assertEqual(mine.training.factorial_arm, 'A_vp_pvp')

    def test_equal_to_table_4_a_cells_except_the_checkpoint(self):
        mine = e.get_config('celeba_a_c160_m35_ode_edm_heun_seed1')
        self.assertEqual(without_range(mine), without_range(lo.get_config('A_c80_lo_ode_edm_heun_seed1')))

    def test_every_cell_passes_the_dataset_check_of_the_evaluation(self):
        import endpoint_denoise_eval
        for name in e.CELLS:
            config = e.get_config(name)
            config.eval.batch_size = 0
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})

    def test_curve_identity(self):
        self.assertEqual(e.curve_identity()['checkpoint_id'], CURVE_ID)
        self.curve(ckpts=(80, 100))
        with self.assertRaisesRegex(RuntimeError, 'no single 800k cell'):
            e.curve_identity()
        (e.ACURVE / 'summary.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'No A curve summary'):
            e.curve_identity()


class Preflight(Base):
    lane = 's1'

    def run_preflight(self, smoke, loaded_id=CURVE_ID):
        def load(config, spec):
            return 'model', dict(checkpoint_id=loaded_id, step=spec['step'])

        original = g.config_of

        def on_cpu(spec):
            config = original(spec)
            config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
            return config

        with patch('torch.cuda.is_available', lambda: True), patch('torch.cuda.device_count', lambda: 1), \
                patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0'), patch.object(lo, 'check_inception'), \
                patch('evaluation.dataset_stats_fingerprint', lambda c: q.REFERENCE_SHA), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(g, 'config_of', on_cpu), patch.object(q, 'load_model', load), \
                patch.object(followup, 'smoke'), patch.object(q, 'endpoint_smoke', side_effect=smoke) as call:
            code = e.preflight()
        return code, json.loads((e.LOGS / 'preflight_s1.json').read_text()), call

    def test_both_solvers_on_the_800k_weights(self):
        code, saved, call = self.run_preflight(lambda config, model, spec, directory: 3.0)
        self.assertEqual((code, saved['status'], saved['lane'], saved['step']), (0, 'passed', 's1', 800001))
        self.assertEqual(saved['checkpoint'], dict(checkpoint_id=CURVE_ID, step=800001))
        self.assertEqual(saved['curve']['cell'], 'A_c160_lo')
        seen = [(a[2]['source'], a[2]['ckpt'], a[2]['method'], a[2]['seed'], a[0].eval.sampling_num_scales,
                 a[0].sampling.time_grid, str(a[3]).endswith('preflight_endpoint_s1')) for a, _ in call.call_args_list]
        self.assertEqual(seen, [('celeba_a', 160, DPM, 1, 34, 'uniform_logsnr', True),
                                ('celeba_a', 160, HEUN, 1, 34, 'karras', True)])

    def test_other_checkpoint_than_the_curve_fails(self):
        code, saved, call = self.run_preflight(lambda config, model, spec, directory: 3.0, loaded_id='other')
        self.assertEqual((code, saved['status']), (1, 'failed'))
        self.assertIn('not the one of the A curve', saved['failures']['celeba_a_c160_m35'])
        self.assertEqual(call.call_count, 0)

    def test_smoke_failure_is_recorded(self):
        def smoke(config, model, spec, directory):
            raise ValueError('heun')

        code, saved, _ = self.run_preflight(smoke)
        self.assertEqual((code, sorted(saved['failures'])), (1, ['celeba_a_c160_m35']))


class Report(Base):
    def prepare(self, name, step=None, features=True, checkpoint_id=CURVE_ID):
        spec = e.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / e.eval_folder(name)
        rounds = directory / 'ckpt_160'
        rounds.mkdir(parents=True)
        for index in range(98):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{q.FEATURES if features else 'other': np.zeros((512, 2048))})
        (directory / 'manifest.json').write_text(json.dumps(dict(config=without_range(g.config_of(spec)))))
        (e.LOGS / 'preflight_{}.json'.format(spec['lane'])).write_text(json.dumps(dict(
            dataset_stats_id=dict(CELEBA=q.REFERENCE_SHA), checkpoint=dict(checkpoint_id=CURVE_ID, step=800001),
            curve=dict(cell='A_c160_lo', checkpoint_id=CURVE_ID))))
        (e.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=50000, nfe=35, checkpoint_step=step or 800001, dataset_stats_id=q.REFERENCE_SHA,
                      training_protocol_sha256='protocol-a', checkpoint_id=checkpoint_id)
        for metric in ('IS', 'fid', 'kid'):
            for variant in q.stage0.VARIANTS:
                values[metric + '_' + variant] = 2.0
        np.savez(str(directory / 'report_160.npz'), **values)
        return directory

    def test_report_and_both_summaries(self):
        name = 'celeba_a_c160_m35_ode_edm_heun_seed0'
        self.prepare(name)
        e.report(name)
        mine = json.loads((e.LOGS / 'summary_s0.json').read_text())
        self.assertEqual((mine['status'], mine['completed_cells'], mine['total_cells']), ('running', 1, 2))
        shared = json.loads((e.LOGS / 'summary.json').read_text())
        self.assertEqual((shared['completed_cells'], shared['total_cells'], shared['lanes']),
                         (1, 4, dict(s0='running', s1='not started')))
        self.assertEqual(shared['cells'][0]['step'], 800001)
        self.assertEqual(shared['curve_800k']['checkpoint_id'], CURVE_ID)
        self.assertEqual(len((e.LOGS / 'summary.csv').read_text().splitlines()), 2)
        self.assertEqual(oct((e.LOGS / 'summary.json').stat().st_mode & 0o777), '0o644')
        self.assertEqual([p.name for p in e.LOGS.iterdir() if p.name.endswith('.tmp')], [])

    def test_shared_summary_covers_both_lanes(self):
        for lane, name in (('s0', 'celeba_a_c160_m35_ode_dpmpp_2m_seed0'), ('s1', 'celeba_a_c160_m35_ode_dpmpp_2m_seed1')):
            with patch.dict(os.environ, A800K_LANE=lane):
                self.prepare(name)
                e.report(name)
        shared = json.loads((e.LOGS / 'summary.json').read_text())
        self.assertEqual(sorted(c['cell'] for c in shared['cells']),
                         ['celeba_a_c160_m35_ode_dpmpp_2m_seed0', 'celeba_a_c160_m35_ode_dpmpp_2m_seed1'])
        self.assertEqual(shared['lanes'], dict(s0='running', s1='running'))

    def test_refusals(self):
        for changes, message in ((dict(step=500001), 'report identity'), (dict(features=False), 'bootstrap'),
                                 (dict(checkpoint_id='other'), 'different checkpoints')):
            directory = self.prepare('celeba_a_c160_m35_ode_dpmpp_2m_seed0', **changes)
            with self.assertRaisesRegex(RuntimeError, message):
                e.report('celeba_a_c160_m35_ode_dpmpp_2m_seed0')
            shutil.rmtree(str(directory))


class Queue(Base):
    def run_queue(self, blocked=None, fail=(), preflight=True):
        if preflight:
            (e.LOGS / 'preflight_s0.json').write_text(json.dumps(dict(failures=blocked or {})))
        order = []

        def evaluate(name):
            order.append(name)
            return name not in fail

        with patch.object(e, 'evaluate', evaluate), patch.object(e, 'write_summary') as summary:
            code = e.queue()
        return code, order, summary

    def test_only_the_lane_cells_in_order(self):
        self.assertEqual(self.run_queue()[:2], (0, e.lane_cells('s0')))

    def test_failed_cell_does_not_stop_the_next(self):
        bad = e.lane_cells('s0')[0]
        code, order, summary = self.run_queue(fail=(bad,))
        self.assertEqual((code, order), (1, e.lane_cells('s0')))
        summary.assert_called_once_with([bad])

    def test_blocked_or_missing_preflight(self):
        self.assertEqual(self.run_queue(blocked=dict(celeba_a_c160_m35='bad'))[:2], (1, []))
        (e.LOGS / 'preflight_s0.json').unlink()
        self.assertEqual(self.run_queue(preflight=False)[:2], (1, []))

    def test_failed_lane_marks_the_shared_summary(self):
        (e.LOGS / 'summary_s1.json').write_text(json.dumps(dict(status='failed')))
        (e.LOGS / 'preflight_s0.json').write_text(json.dumps(dict(failures=dict(x='bad'))))
        self.assertEqual(e.queue(), 1)
        shared = json.loads((e.LOGS / 'summary.json').read_text())
        self.assertEqual((shared['status'], shared['lanes']), ('failed', dict(s0='failed', s1='failed')))


DOCKER = r"""#!/usr/bin/env bash
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.gpus; do [[ -e "$f" ]] || continue; basename "$f" .gpus; done ;;
  inspect)
    [[ -e "$STATE/$last.gpus" ]] || { echo; exit 1; }
    case "$*" in
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
RUN = 'celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42'
OUT = 'celeba_a800k_50k_20261115'


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
        shutil.copy(str(REPO / 'run_celeba_a800k_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / q.lo.INCEPTION_MODULE
        curve = self.base / 'celeba_a_800k_h200_20261113/eval/logs'
        for path in (main / 'assets/stats', main / 'celeba', tfhub, curve, self.base / RUN / 'checkpoints'):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt',
                     tfhub / 'saved_model.pb', curve / 'summary.json', self.base / RUN / 'checkpoints/checkpoint_160.pth'):
            path.write_text('x')
        (self.base / RUN / 'run_manifest.json').write_text('manifest')

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           A800K_SERVER_BASE=str(self.base), **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_celeba_a800k_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_two_lanes_share_one_output(self):
        for lane, gpu in (('s0', '1'), ('s1', '2')):
            result = self.launch('start', lane, gpu)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = (self.state / 'celeba-a800k-{}.args'.format(lane)).read_text()
            self.assertIn('device=' + gpu, args)
            self.assertIn('A800K_LANE=' + lane, args)
            self.assertIn('{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/celeba_a/checkpoints,readonly'.format(
                RUN), args)
            self.assertIn('celeba_a_800k_h200_20261113/eval/logs,dst=/workspace/score_sde_pytorch/acurve,readonly', args)
            self.assertIn('{}/logs,dst=/workspace/score_sde_pytorch/logs'.format(OUT), args)
            self.assertIn('logs/queue_{}_'.format(lane), args)
            self.assertNotIn('--rm', args)
        inputs = self.base / OUT / 'workdirs/_training_inputs'
        self.assertEqual([p.name for p in inputs.iterdir()], ['celeba_a.json'])
        self.assertEqual(self.events(), ['run celeba-a800k-s0', 'run celeba-a800k-s1'])

    def test_refusals(self):
        self.assertEqual(self.launch('start', 's0', '3', GPU_NAME='NVIDIA H200 NVL').returncode, 1)
        self.assertEqual(self.launch('start', 'a', '1').returncode, 2)
        self.assertEqual(self.launch('start', 's0').returncode, 2)
        self.assertIn(b'need 60000', self.launch('start', 's0', '1', FREE_MIB='50000').stdout)
        (self.state / 'celeba-other.gpus').write_text('1,')
        self.assertIn(b'runs containers: celeba-other', self.launch('start', 's0', '1').stdout)
        self.assertEqual(self.launch('start', 's1', '2').returncode, 0)
        (self.state / 'celeba-a800k-s1.gpus').write_text('2,')
        self.assertIn(b'Container exists', self.launch('start', 's1', '2').stdout)
        (self.base / RUN / 'run_manifest.json').write_text('changed')
        self.assertIn(b'differs from the one of the first lane', self.launch('start', 's0', '3').stdout)
        (self.base / RUN / 'checkpoints/checkpoint_160.pth').unlink()
        self.assertIn(b'Missing input', self.launch('start', 's0', '3').stdout)
        self.assertEqual(self.events(), ['run celeba-a800k-s1'])


if __name__ == '__main__':
    unittest.main()
