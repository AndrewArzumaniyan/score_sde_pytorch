"""CPU checks for the extra NFE limit lanes (VP-lambda@500k, seed 43 pair) and their launcher."""
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if importlib.util.find_spec('tensorflow') is None:  # Outside the Docker image.
    import torch._dynamo  # noqa: F401, its import scans module specs, which stubs lack.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())

import torch

import celeba_followup_eval as followup
import celeba_pair43_lo_eval as p
import gpu2_a50k_grids as g
import gpu2_queue_eval as q
import nfe_limit_extra_eval as x
import nfe_limit_eval as n
import ode_sampling
import run_lib
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a

DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'


def resolved_manifest(config, protocol):
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256=protocol)


def factorial_a(seed, freq):
    config = get_a()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.seed = seed
    config.training.snapshot_freq = freq
    return config


def plain(config):
    return json.loads(json.dumps(_canonicalize(config)))


def cpu(config):
    config.device = 'cpu' if isinstance(config.device, str) else torch.device('cpu')
    return config


def names(model):
    return ['{}_nfe257_dpm_50k'.format(model), '{}_nfe257_dpm_10k'.format(model), '{}_nfe257_heun_10k'.format(model)]


class Base(unittest.TestCase):
    lane = 'a500'
    fox_seed = 43

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        (inputs / 'celeba_a.json').write_text(json.dumps(resolved_manifest(factorial_a(42, 5000), 'protocol-a')))
        (inputs / (x.A43_WORK + '.json')).write_text(json.dumps(resolved_manifest(factorial_a(43, 50000),
                                                                                  'protocol-a43')))
        (inputs / (x.FOX43_WORK + '.json')).write_text(json.dumps(dict(config=dict(seed=self.fox_seed),
                                                                       training_protocol_sha256='protocol-fox43')))
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (g, 'ROOT', self.root), (g, 'LOGS', self.root / 'logs'),
                                      (n, 'ROOT', self.root), (n, 'LOGS', self.root / 'logs'),
                                      (p, 'ROOT', self.root), (p, 'TRAINING_INPUTS', inputs),
                                      (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, NFE_LIMIT_LANE=self.lane)
        patcher.start()
        self.addCleanup(patcher.stop)


class Plan(Base):
    def test_cells_order_and_lanes(self):
        self.assertIs(n.CELLS, x.CELLS)
        self.assertEqual(list(n.CELLS), names('vpl500') + names('dial43') + names('vpl43'))
        self.assertEqual(n.lane_cells('a500'), names('vpl500'))
        self.assertEqual(n.lane_cells('fox43'), names('dial43'))
        self.assertEqual(n.lane_cells('a43'), names('vpl43'))
        expected = dict(VPL500=('celeba_a', 100, 500001), DIAL43=('celeba_fox43', 160, 800001),
                        VPL43=('celeba_a43', 8, 400001))
        for name in n.CELLS:
            spec = n.resolve(name)
            self.assertEqual((spec['source'], spec['ckpt'], spec['step']), expected[spec['model']])
            self.assertEqual((spec['seed'], spec['calls'], spec['dataset']), (0, 256, 'CELEBA'))
            self.assertEqual(spec['count'], 50000 if name.endswith('_50k') else 10000)
            self.assertEqual(spec['work'], spec['source'])

    def test_lane_is_required(self):
        with patch.dict(os.environ, NFE_LIMIT_LANE='a'):
            with self.assertRaisesRegex(ValueError, 'a500, fox43, a43'):
                n.lane()

    def existing_50k(self, model, method):
        if model == 'VPL500':
            (g.LOGS / g.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=100)))
            return g.get_config('celeba_a_cbest_m35_{}_seed0'.format(method))
        return p.get_config(p.cell_name(x.PAIR43[model], method, 0))

    def test_config_at_34_calls_equals_the_existing_50k_cell(self):
        for model in x.MODELS:
            for method in (DPM, HEUN):
                spec = n.resolve(n.cell(model, 256, method, 10000)[0])
                mine = n.config_of(dict(spec, calls=34, count=50000))
                self.assertEqual(plain(mine), plain(self.existing_50k(model, method)), (model, method))

    def test_cells_differ_only_in_calls_and_count(self):
        for name in n.CELLS:
            spec = n.resolve(name)
            mine, reference = plain(x.get_config(name)), plain(x.matched35_config(spec))
            self.assertEqual((mine['eval'].pop('sampling_num_scales'), mine['eval'].pop('num_samples')),
                             (256, spec['count']))
            for key in ('sampling_num_scales', 'num_samples'):
                reference['eval'].pop(key)
            self.assertEqual(mine, reference, name)
            config = x.get_config(name)
            self.assertEqual((config.sampling.method, config.eval.sampling_seed, config.eval.begin_ckpt),
                             (spec['method'], 0, spec['ckpt']))
            self.assertEqual(config.sampling.time_grid, 'karras' if spec['method'] == HEUN else 'uniform_logsnr')
            self.assertEqual((config.sampling.logsnr_min, config.sampling.logsnr_max), (-10.0, 9.115429865459795))

    def test_training_seeds(self):
        self.assertEqual(x.get_config('vpl500_nfe257_dpm_50k').seed, 42)
        self.assertEqual(x.get_config('vpl43_nfe257_dpm_50k').seed, 43)


class WrongFoxSeed(Base):
    fox_seed = 42

    def test_fox43_needs_seed_43(self):
        with self.assertRaisesRegex(ValueError, 'seed 43'):
            x.get_config('dial43_nfe257_dpm_50k')


class Grids(Base):
    def test_every_cell_grid_spans_the_window(self):
        for name in n.CELLS:
            spec = n.resolve(name)
            config = cpu(x.get_config(name))
            sde, eps = run_lib.get_sde(config, n_override=256)
            followup.check_cell_grid(config, sde, eps, spec)
            self.assertEqual(int(sde.N), 256)

    def test_sampler_runs_at_256_calls(self):
        def toy_score(sde, model, train, continuous):
            return lambda value, t: -value
        with patch.object(ode_sampling, 'get_score_fn', toy_score):
            for model in x.MODELS:
                for method in (DPM, HEUN):
                    config = cpu(x.get_config(n.cell(model, 256, method, 10000)[0]))
                    sde, eps = run_lib.get_sde(config, n_override=256)
                    torch.manual_seed(0)
                    outputs, used = ode_sampling.get_vp_ode_sampler(sde, (1, 3, 8, 8), lambda v: v, method, eps,
                                                                    'cpu')(None)
                    self.assertEqual(used, 257)
                    self.assertTrue(all(bool(torch.isfinite(v).all()) for v in outputs.values()))


class Queue(Base):
    def test_order_failure_blocking_and_command(self):
        (n.LOGS / 'preflight.json').write_text(json.dumps(dict(failures={})))
        order = []
        with patch.object(n, 'evaluate', lambda name: order.append(name) or True), patch.object(n, 'write_summary'):
            self.assertEqual(n.queue(), 0)
        self.assertEqual(order, names('vpl500'))
        (n.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=dict(celeba_a='bad'))))
        with patch.object(n, 'evaluate', lambda name: self.fail(name)), patch.object(n, 'write_summary'):
            self.assertEqual(n.queue(), 1)
        completed = MagicMock(returncode=0)
        with patch.object(x.subprocess, 'run', return_value=completed) as call:
            x.run('vpl500_nfe257_dpm_50k')
        command = call.call_args[0][0]
        self.assertIn('--config=nfe_limit_extra_eval.py:vpl500_nfe257_dpm_50k', command)
        self.assertIn('--workdir=workdirs/celeba_a', command)
        self.assertIn('--eval_folder=eval_nfelimit_vpl500_nfe257_dpm_50k', command)
        self.assertTrue((n.LOGS / 'vpl500_nfe257_dpm_50k_timing.json').exists())


def fake(name, fid):
    return dict(cell=name, fid_tweedie_xt=fid, gpu_name='card')


class Combine(Base):
    def write(self, lane, cells):
        directory = self.root / lane
        directory.mkdir()
        (directory / 'summary.json').write_text(json.dumps(dict(lane=lane, status='running',
            completed_cells=len(cells), total_cells=3, cells=[fake(k, v) for k, v in cells.items()])))
        return str(directory)

    def test_contrasts_of_the_rule(self):
        dirs = [self.write('a500', dict(vpl500_nfe257_dpm_50k=1.600, vpl500_nfe257_dpm_10k=2.53,
                                        vpl500_nfe257_heun_10k=2.52)),
                self.write('fox43', dict(dial43_nfe257_dpm_50k=1.45)),
                self.write('a43', dict(vpl43_nfe257_dpm_50k=1.55)),
                self.write('dial', dict(dial_nfe257_dpm_50k=1.4476)), self.write('b', dict(b_nfe257_dpm_50k=1.4729)),
                self.write('a', dict(vpl_nfe257_dpm_50k=1.6193)), str(self.root / 'missing')]
        stream = io.StringIO()
        with redirect_stdout(stream):
            x.combine(dirs)
        out = json.loads(stream.getvalue())
        self.assertAlmostEqual(out['vpl500_minus_vpl400']['difference'], -0.0193)
        self.assertIn('400k stays', out['vpl500_minus_vpl400']['status'])
        self.assertAlmostEqual(out['contrasts_with_vpl500'][0]['difference'], -0.1524)
        self.assertAlmostEqual(out['contrasts_with_vpl500'][1]['difference'], -0.1271)
        self.assertAlmostEqual(out['seed43']['difference'], -0.10)
        self.assertEqual((out['seed43']['status'], out['seed43']['replicates_seed42']), ('DIAL seed 43 better', True))
        self.assertTrue(out['solver_gap_257']['VPL500']['converged'])
        self.assertNotIn('DIAL43', out['solver_gap_257'])

    def test_stronger_500k_and_no_replication(self):
        dirs = [self.write('a500', dict(vpl500_nfe257_dpm_50k=1.55)), self.write('a', dict(vpl_nfe257_dpm_50k=1.6193)),
                self.write('fox43', dict(dial43_nfe257_dpm_50k=1.52)), self.write('a43', dict(vpl43_nfe257_dpm_50k=1.55))]
        stream = io.StringIO()
        with redirect_stdout(stream):
            x.combine(dirs)
        out = json.loads(stream.getvalue())
        self.assertIn('500k is the strongest', out['vpl500_minus_vpl400']['status'])
        self.assertEqual((out['seed43']['status'], out['seed43']['replicates_seed42']), ('indistinguishable', False))
        self.assertNotIn('difference', out['contrasts_with_vpl500'][0])


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
    printf '%s\n' "$@" > "$STATE/$name.args"; echo '{}' ;;
esac
exit 0
"""
STUBS = {
    'git': '#!/usr/bin/env bash\n[[ "$1" == rev-parse ]] && echo commit\nexit 0\n',
    'nvidia-smi': ('#!/usr/bin/env bash\nif [[ "$*" == *name* ]]; then gpu="${@: -3:1}"; v="GPU_NAME_$gpu"; '
                   'echo "${!v:-NVIDIA H200 NVL}"; else echo "${FREE_MIB:-90000}"; fi\n'),
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
}
RUNS = {
    'celeba_a': ('celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42', 100, 'celeba_a'),
    'celeba_fox43': ('celeba_pair_seed43_h200_20261106/workdirs/celeba_fox_matern32_kappa1000_h200_seed43', 160,
                     'celeba_fox_matern32_kappa1000_h200_seed43'),
    'celeba_a43': ('celeba_a43_400k_rtx_20261107/workdirs/celeba_ab_replication_A_vp_pvp_400k_rtx_seed43', 8,
                   'celeba_ab_replication_A_vp_pvp_400k_rtx_seed43'),
}
RTX = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'


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
        shutil.copy(str(REPO / 'run_nfe_limit_extra.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / q.lo.INCEPTION_MODULE
        for path in (main / 'assets/stats', main / 'celeba', tfhub):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt',
                     tfhub / 'saved_model.pb'):
            path.write_text('x')
        for run, ckpt, _ in RUNS.values():
            (self.base / run / 'checkpoints').mkdir(parents=True)
            (self.base / run / 'checkpoints/checkpoint_{}.pth'.format(ckpt)).write_text('x')
            (self.base / run / 'run_manifest.json').write_text('manifest')

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           NFE_LIMIT_SERVER_BASE=str(self.base), GPU_NAME_1=RTX, **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_nfe_limit_extra.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def test_lanes_on_either_card(self):
        for lane, gpu, card, arch, work in (('a500', '0', 'H200', '9.0', 'celeba_a'),
                                            ('fox43', '1', 'RTX', '12.0', 'celeba_fox43'),
                                            ('a43', '2', 'H200', '9.0', 'celeba_a43')):
            result = self.launch('start', lane, gpu)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = (self.state / ('celeba-nfe-' + lane + '.args')).read_text()
            for item in ('device=' + gpu, 'NFE_LIMIT_LANE=' + lane, 'NFE_LIMIT_CARD=' + card,
                         'TORCH_CUDA_ARCH_LIST=' + arch, 'CUDA_VISIBLE_DEVICES=0', 'nfe_limit_extra_eval.py queue',
                         '{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(
                             RUNS[work][0], work)):
                self.assertIn(item, args)
            self.assertEqual(args.count('/checkpoints,readonly'), 1)
            inputs = self.base / 'nfe_limit_extra_20261116' / lane / 'workdirs/_training_inputs'
            self.assertEqual([path.name for path in inputs.iterdir()], [RUNS[work][2] + '.json'])

    def test_refusals(self):
        self.assertIn(b'expected an H200', self.launch('start', 'a500', '0', GPU_NAME_0='NVIDIA A100').stdout)
        self.assertEqual(self.launch('start', 'a', '0').returncode, 2)
        self.assertEqual(self.launch('start', 'a500').returncode, 2)
        self.assertIn(b'need 60000', self.launch('start', 'a500', '0', FREE_MIB='50000').stdout)
        (self.state / 'celeba-other.gpus').write_text('0,')
        self.assertIn(b'runs containers: celeba-other', self.launch('start', 'a500', '0').stdout)
        (self.base / RUNS['celeba_a43'][0] / 'checkpoints/checkpoint_8.pth').unlink()
        self.assertIn(b'Missing input', self.launch('start', 'a500', '3').stdout)

    def test_combine_reads_both_queues(self):
        self.assertIn(b'No lane logs', self.launch('combine').stdout)
        for path in ('nfe_limit_extra_20261116/a500/logs', 'nfe_limit_20261116/a/logs'):
            (self.base / path).mkdir(parents=True)
        result = self.launch('combine')
        self.assertEqual(result.returncode, 0, result.stdout)
        args = (self.state / 'check.args').read_text()
        self.assertIn('dst=/nfe/a500,readonly', args)
        self.assertIn('dst=/prev/a,readonly', args)
        self.assertNotIn('/prev/dial', args)


if __name__ == '__main__':
    unittest.main()
