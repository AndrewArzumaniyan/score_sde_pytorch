"""CPU checks for the H200 EDM Heun curve and AFHQv2 DIAL 250k queues and their launcher; no GPU, data or checkpoints."""
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if importlib.util.find_spec('tensorflow') is None:  # Outside the Docker image.
    import torch._dynamo  # noqa: F401, its import scans module specs, which stubs lack.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())

import afhqv2_rtx_train as afhq
import celeba_fox800_lo_eval as fox800
import celeba_followup_eval as followup
import gpu2_a50k_grids as g
import gpu2_queue_eval as q
import h200_edm_check_eval as h
import run_lib  # noqa: F401
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'


def resolved_manifest(config, protocol):
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256=protocol)


def factorial(get):
    config = get()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.seed = 42
    config.training.snapshot_freq = 5000
    return config


def comparable(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    for key in ('begin_ckpt', 'end_ckpt', 'num_samples'):
        resolved['eval'].pop(key)
    return resolved


class Base(unittest.TestCase):
    lane = 'curve'

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        for key, arm in (('afhq_a', 'a'), ('afhq_vp', 'vp'), ('afhq_fox', 'fox')):
            (inputs / (key + '.json')).write_text(json.dumps(resolved_manifest(afhq.get_config(arm), 'protocol-' + arm)))
        (inputs / 'celeba_b.json').write_text(json.dumps(resolved_manifest(factorial(get_b), 'protocol-b')))
        for key in ('celeba_a_first', 'celeba_a', followup.WORKDIRS['A']):
            (inputs / (key + '.json')).write_text(json.dumps(resolved_manifest(factorial(get_a), 'protocol-a')))
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (g, 'ROOT', self.root), (g, 'LOGS', self.root / 'logs'),
                                      (h, 'ROOT', self.root), (h, 'LOGS', self.root / 'logs'),
                                      (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, H200_EDM_LANE=self.lane)
        patcher.start()
        self.addCleanup(patcher.stop)


class Plan(Base):
    def test_cells_order_and_lanes(self):
        curve = ['dial_c160_edm10k', 'b_c160_edm10k', 'dial_c140_edm10k', 'b_c140_edm10k', 'dial_c120_edm10k',
                 'b_c120_edm10k'] + ['a_c{}_edm10k'.format(c) for c in (80, 100, 120, 140, 160)]
        afhq_names = ['afhq_fox_c5_m35_{}_seed{}'.format(m, s) for s in (0, 1) for m in (DPM, HEUN)]
        self.assertEqual(h.lane_cells('curve'), curve)
        self.assertEqual(h.lane_cells('afhq'), afhq_names)
        self.assertEqual(list(h.CELLS), curve + afhq_names)
        for name in curve:
            spec = h.resolve(name)
            self.assertEqual((spec['method'], spec['seed'], spec['count'], spec['calls'], spec['dataset']),
                             (HEUN, 0, 10000, 34, 'CELEBA'))
            self.assertEqual(spec['step'], 5000 * spec['ckpt'] + 1)
        self.assertEqual(h.resolve('a_c80_edm10k')['source'], 'celeba_a_first')
        self.assertEqual(h.resolve('a_c100_edm10k')['source'], 'celeba_a')
        for name in afhq_names:
            spec = h.resolve(name)
            self.assertEqual((spec['source'], spec['ckpt'], spec['step'], spec['count']), ('afhq_fox', 5, 250001, 50000))

    def test_lane_is_required(self):
        with patch.dict(os.environ, H200_EDM_LANE='fox'):
            with self.assertRaisesRegex(ValueError, 'must be curve or afhq'):
                h.lane()

    def test_configs_equal_the_50k_cells_except_checkpoint_and_count(self):
        pairs = [(h.get_config('dial_c140_edm10k'), fox800.get_config(fox800.cell_name(HEUN, 0)), 140, 160),
                 (h.get_config('b_c120_edm10k'), q.get_config('celeba_b_c160_m35_{}_seed0'.format(HEUN)), 120, 160)]
        (g.LOGS / g.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=100)))
        pairs.append((h.get_config('a_c140_edm10k'), g.get_config('celeba_a_cbest_m35_{}_seed0'.format(HEUN)), 140, 100))
        (g.LOGS / g.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=80)))
        pairs.append((h.get_config('a_c80_edm10k'), g.get_config('celeba_a_cbest_m35_{}_seed0'.format(HEUN)), 80, 80))
        for mine, theirs, ckpt, other in pairs:
            self.assertEqual(comparable(mine), comparable(theirs))
            self.assertEqual((mine.eval.begin_ckpt, theirs.eval.begin_ckpt), (ckpt, other))
            self.assertEqual((mine.eval.num_samples, mine.sampling.time_grid, mine.eval.sampling_num_scales),
                             (10000, 'karras', 34))
            self.assertEqual(mine.sampling.logsnr_max, 9.115429865459795)
        for seed in (0, 1):
            for method in (DPM, HEUN):
                mine = h.get_config('afhq_fox_c5_m35_{}_seed{}'.format(method, seed))
                theirs = q.get_config('afhq_fox_c4_m35_{}_seed{}'.format(method, seed))
                self.assertEqual(comparable(mine), comparable(theirs))
                self.assertEqual((mine.eval.begin_ckpt, mine.eval.num_samples), (5, 50000))

    def test_edm_selection(self):
        def cell(name, fid):
            spec = h.resolve(name)
            return dict(spec, fid_tweedie_xt=fid)
        cells = {n: cell(n, f) for n, f in (('dial_c120_edm10k', 1.80), ('dial_c140_edm10k', 1.70),
                                            ('dial_c160_edm10k', 1.95), ('b_c120_edm10k', 1.9),
                                            ('b_c140_edm10k', 1.7))}
        out = h.edm_selection(cells)
        self.assertEqual((out['DIAL']['selected_step'], out['DIAL']['minimum']), (700000, 1.70))
        self.assertIsNone(out['B'])
        self.assertIsNone(out['A'])
        cells['dial_c160_edm10k'] = cell('dial_c160_edm10k', 1.90)
        self.assertEqual(h.edm_selection(cells)['DIAL']['selected_step'], 800000)


class Queue(Base):
    def run_queue(self, blocked=None, fail=()):
        (h.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=blocked or {})))
        order = []

        def evaluate(name):
            order.append(name)
            return name not in fail

        with patch.object(h, 'evaluate', evaluate), patch.object(h, 'write_summary') as summary:
            code = h.queue()
        return code, order, summary

    def test_order_failure_and_blocking(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, h.lane_cells('curve')))
        bad = h.lane_cells('curve')[1]
        code, order, summary = self.run_queue(fail=(bad,))
        self.assertEqual((code, order), (1, h.lane_cells('curve')))
        summary.assert_called_once_with([bad])
        code, order, _ = self.run_queue(blocked=dict(celeba_b_edm10k='bad'))
        self.assertEqual(code, 1)
        self.assertEqual(order, [n for n in h.lane_cells('curve') if not n.startswith('b_')])


class Report(Base):
    def test_report_and_summary(self):
        name = 'b_c140_edm10k'
        spec = h.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / h.eval_folder(name)
        rounds = directory / 'ckpt_140'
        rounds.mkdir(parents=True)
        for index in range(20):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=comparable_with_range(h.config_of(spec)))))
        (h.LOGS / 'preflight.json').write_text(json.dumps(dict(dataset_stats_id=dict(CELEBA='celeba'))))
        (h.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=10000, nfe=35, checkpoint_step=700001, dataset_stats_id='celeba',
                      training_protocol_sha256='protocol-b', checkpoint_id='id')
        for metric in ('IS', 'fid', 'kid'):
            for variant in q.stage0.VARIANTS:
                values[metric + '_' + variant] = 2.0
        np.savez(str(directory / 'report_140.npz'), **values)
        h.report(name)
        summary = json.loads((h.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['lane'], summary['completed_cells'], summary['total_cells']), ('curve', 1, 11))
        self.assertIsNone(summary['edm_selection']['B'])
        values['checkpoint_step'] = 800001
        np.savez(str(directory / 'report_140.npz'), **values)
        with self.assertRaisesRegex(RuntimeError, 'report identity'):
            h.report(name)


def comparable_with_range(config):
    resolved = json.loads(json.dumps(_canonicalize(config)))
    resolved['eval'].pop('begin_ckpt')
    resolved['eval'].pop('end_ckpt')
    return resolved


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
    'nvidia-smi': ('#!/usr/bin/env bash\nif [[ "$*" == *name* ]]; then echo "${GPU_NAME:-NVIDIA H200 NVL}"; '
                   'else echo "${FREE_MIB:-140000}"; fi\n'),
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
}
RUNS = {
    'celeba_fox': ('score_sde_pytorch/workdirs/celeba_fox_matern32_kappa1000_50k', (120, 140, 160), False),
    'celeba_b': ('celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42', (120, 140, 160), True),
    'celeba_a_first': ('celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42', (80,), True),
    'celeba_a': ('celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42', (100, 120, 140, 160), True),
    'afhq_fox': ('afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42', (5,), True),
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
        shutil.copy(str(REPO / 'run_h200_edm_check.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / q.lo.INCEPTION_MODULE
        for path in (main / 'assets/stats', main / 'datasets/afhqv2-64x64', main / 'celeba', tfhub):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/afhqv2_64_cd017afab009_stats.npz',
                     main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt',
                     tfhub / 'saved_model.pb'):
            path.write_text('x')
        for run, ckpts, has_manifest in RUNS.values():
            (self.base / run / 'checkpoints').mkdir(parents=True)
            for ckpt in ckpts:
                (self.base / run / 'checkpoints/checkpoint_{}.pth'.format(ckpt)).write_text('x')
            if has_manifest:
                (self.base / run / 'run_manifest.json').write_text('manifest')

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           H200EDM_SERVER_BASE=str(self.base), **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_h200_edm_check.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def test_two_lanes_on_two_gpus(self):
        for lane, gpu, name, works in (('curve', '3', 'celeba-h200-edm-curve',
                                        ('celeba_fox', 'celeba_b', 'celeba_a_first', 'celeba_a')),
                                       ('afhq', '0', 'afhqv2-h200-dial250', ('afhq_fox',))):
            result = self.launch('start', lane, gpu)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = (self.state / (name + '.args')).read_text()
            self.assertIn('device=' + gpu, args)
            self.assertIn('H200_EDM_LANE=' + lane, args)
            self.assertIn('TORCH_CUDA_ARCH_LIST=9.0', args)
            for work in works:
                self.assertIn('{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(
                    RUNS[work][0], work), args)
            inputs = self.base / 'h200_edm_check_20261115' / lane / 'workdirs/_training_inputs'
            expected = sorted(w + '.json' for w in works if RUNS[w][2])
            self.assertEqual(sorted(p.name for p in inputs.iterdir()), expected)

    def test_refusals(self):
        self.assertEqual(self.launch('start', 'curve', '1', GPU_NAME='NVIDIA RTX PRO 6000').returncode, 1)
        self.assertEqual(self.launch('start', 'fox', '3').returncode, 2)
        self.assertEqual(self.launch('start', 'curve').returncode, 2)
        self.assertIn(b'need 60000', self.launch('start', 'curve', '3', FREE_MIB='50000').stdout)
        (self.state / 'celeba-other.gpus').write_text('3,')
        self.assertIn(b'runs containers: celeba-other', self.launch('start', 'curve', '3').stdout)
        (self.base / RUNS['celeba_b'][0] / 'checkpoints/checkpoint_140.pth').unlink()
        self.assertIn(b'Missing input', self.launch('start', 'afhq', '0').stdout)


if __name__ == '__main__':
    unittest.main()
