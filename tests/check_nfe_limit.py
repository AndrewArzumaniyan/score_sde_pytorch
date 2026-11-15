"""CPU checks for the NFE limit queues and their launcher; no GPU, data or checkpoints."""
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

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if importlib.util.find_spec('tensorflow') is None:  # Outside the Docker image.
    import torch._dynamo  # noqa: F401, its import scans module specs, which stubs lack.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())

import torch

import afhq200k_eval
import afhqv2_rtx_train as afhq
import celeba_fox800_lo_eval as fox800
import celeba_followup_eval as followup
import gpu2_a50k_grids as g
import gpu2_queue_eval as q
import nfe_limit_eval as n
import ode_sampling
import run_lib
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
CURVE = [(c, m) for c in (257, 129, 65, 15) for m in ('dpm', 'heun')]


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


def plain(config):
    return json.loads(json.dumps(_canonicalize(config)))


class Base(unittest.TestCase):
    lane = 'dial'

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
                                      (n, 'ROOT', self.root), (n, 'LOGS', self.root / 'logs'),
                                      (afhq200k_eval, 'ROOT', self.root), (afhq200k_eval, 'LOGS', self.root / 'logs'),
                                      (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, NFE_LIMIT_LANE=self.lane)
        patcher.start()
        self.addCleanup(patcher.stop)


def names(model, count_50k=True):
    out = ['{}_nfe{}_{}_10k'.format(model, nfe, m) for nfe, m in CURVE]
    return out + (['{}_nfe257_dpm_50k'.format(model)] if count_50k else [])


class Plan(Base):
    def test_cells_order_and_lanes(self):
        afhq_names = ['afhq_dial_nfe257_dpm_10k', 'afhq_vpl_nfe257_dpm_10k',
                      'afhq_dial_nfe257_heun_10k', 'afhq_vpl_nfe257_heun_10k']
        self.assertEqual(n.lane_cells('dial'), names('dial'))
        self.assertEqual(n.lane_cells('b'), names('b'))
        self.assertEqual(n.lane_cells('a'), names('vpl') + afhq_names)
        self.assertEqual(len(n.CELLS), 31)
        expected = dict(dial=('celeba_fox', 160, 800001), b=('celeba_b', 160, 800001),
                        vpl=('celeba_a_first', 80, 400001), afhq_dial=('afhq_fox', 4, 200001),
                        afhq_vpl=('afhq_a', 4, 200001))
        for name in n.CELLS:
            spec = n.resolve(name)
            self.assertEqual((spec['source'], spec['ckpt'], spec['step']), expected[spec['model'].lower()])
            self.assertEqual(spec['seed'], 0)
            self.assertIn(spec['calls'], (14, 64, 128, 256))
            if spec['method'] == HEUN:
                self.assertEqual(spec['calls'] % 2, 0)
            self.assertEqual(spec['count'], 50000 if name.endswith('_50k') else 10000)
            self.assertEqual(spec['dataset'], 'AFHQV2' if name.startswith('afhq') else 'CELEBA')

    def test_lane_is_required(self):
        with patch.dict(os.environ, NFE_LIMIT_LANE='curve'):
            with self.assertRaisesRegex(ValueError, 'must be dial, b or a'):
                n.lane()

    def existing_50k(self, model, method):
        if model == 'DIAL':
            return fox800.get_config(fox800.cell_name(method, 0))
        if model == 'B':
            return q.get_config('celeba_b_c160_m35_{}_seed0'.format(method))
        if model == 'VPL':
            (g.LOGS / g.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=80)))
            return g.get_config('celeba_a_cbest_m35_{}_seed0'.format(method))
        if model == 'AFHQ_DIAL':
            return q.get_config('afhq_fox_c4_m35_{}_seed0'.format(method))
        return afhq200k_eval.get_config('afhq_a_c4_m35_{}_seed0'.format(method))

    def test_config_at_34_calls_equals_the_existing_50k_cell(self):
        for model in n.MODELS:
            for method in (DPM, HEUN):
                spec = n.resolve(n.cell(model, 256, method, 10000)[0])
                mine = n.config_of(dict(spec, calls=34, count=50000))
                self.assertEqual(plain(mine), plain(self.existing_50k(model, method)), (model, method))

    def test_cells_differ_only_in_calls_and_count(self):
        for name in n.CELLS:
            spec = n.resolve(name)
            mine, reference = plain(n.config_of(spec)), plain(n.matched35_config(spec))
            self.assertEqual((mine['eval'].pop('sampling_num_scales'), mine['eval'].pop('num_samples')),
                             (spec['calls'], spec['count']))
            for key in ('sampling_num_scales', 'num_samples'):
                reference['eval'].pop(key)
            self.assertEqual(mine, reference, name)
            config = n.get_config(name)
            self.assertEqual((config.sampling.method, config.eval.sampling_seed, config.eval.begin_ckpt),
                             (spec['method'], 0, spec['ckpt']))
            self.assertEqual(config.sampling.time_grid, 'karras' if spec['method'] == HEUN else 'uniform_logsnr')
            self.assertEqual(config.sampling.logsnr_max, 9.115429865459795)
            self.assertEqual(config.sampling.logsnr_min, -10.0)


def cpu(config):
    config.device = 'cpu' if isinstance(config.device, str) else torch.device('cpu')
    return config


class Grids(Base):
    def test_every_cell_grid_spans_the_window(self):
        for name in n.CELLS:
            spec = n.resolve(name)
            config = cpu(n.config_of(spec))
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            self.assertEqual(int(sde.N), spec['calls'])

    def test_sampler_runs_every_budget(self):
        """The real sampler at 14 and 256 calls with a toy score: DPM++ 2M starts its multistep, Heun counts."""
        def toy_score(sde, model, train, continuous):
            return lambda x, t: -x
        with patch.object(ode_sampling, 'get_score_fn', toy_score):
            for model in n.MODELS:
                for calls, method in n.CURVE:
                    spec = dict(n.resolve(n.cell(model, 256, method, 10000)[0]), calls=calls)
                    config = cpu(n.config_of(spec))
                    sde, eps = run_lib.get_sde(config, n_override=calls)
                    torch.manual_seed(0)
                    outputs, used = ode_sampling.get_vp_ode_sampler(sde, (1, 3, 8, 8), lambda x: x, method, eps,
                                                                    'cpu')(None)
                    self.assertEqual(used, calls + 1)
                    self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs.values()))


class Queue(Base):
    def run_queue(self, blocked=None, fail=()):
        (n.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=blocked or {})))
        order = []

        def evaluate(name):
            order.append(name)
            return name not in fail

        with patch.object(n, 'evaluate', evaluate), patch.object(n, 'write_summary') as summary:
            code = n.queue()
        return code, order, summary

    def test_order_failure_and_blocking(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, names('dial')))
        bad = names('dial')[1]
        code, order, summary = self.run_queue(fail=(bad,))
        self.assertEqual((code, order), (1, names('dial')))
        summary.assert_called_once_with([bad])
        code, order, _ = self.run_queue(blocked=dict(celeba_fox='bad'))
        self.assertEqual((code, order), (1, []))

    def test_lane_a_blocks_one_source(self):
        with patch.dict(os.environ, NFE_LIMIT_LANE='a'):
            code, order, _ = self.run_queue(blocked=dict(afhq_a='bad'))
        self.assertEqual(code, 1)
        self.assertEqual(order, [x for x in n.lane_cells('a') if not x.startswith('afhq_vpl')])


def fake_result(name, fid):
    spec = n.resolve(name)
    return dict(spec, cell=name, fid_tweedie_xt=fid, gpu_name='card', label=n.MODELS[spec['model']]['label'])


class Report(Base):
    def test_report_and_summary(self):
        name = 'dial_nfe257_heun_10k'
        spec = n.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / n.eval_folder(name)
        rounds = directory / 'ckpt_160'
        rounds.mkdir(parents=True)
        for index in range(1, 20):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{q.FEATURES: np.zeros((q.BATCH, 2048))})
        saved = plain(n.config_of(spec))
        saved['eval'].pop('begin_ckpt')
        saved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=saved)))
        (n.LOGS / 'preflight.json').write_text(json.dumps(dict(dataset_stats_id=dict(CELEBA='celeba'),
                                                               gpu_name='NVIDIA RTX PRO 6000')))
        (n.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=10000, nfe=257, checkpoint_step=800001, dataset_stats_id='celeba',
                      training_protocol_sha256=q.LEGACY, checkpoint_id=q.KNOWN_IDS[('celeba_fox', 160)])
        for metric in ('IS', 'fid', 'kid'):
            for variant in q.stage0.VARIANTS:
                values[metric + '_' + variant] = 2.0
        np.savez(str(directory / 'report_160.npz'), **values)
        n.report(name)
        result = json.loads(n.result_path(name).read_text())
        self.assertEqual((result['gpu_name'], result['nfe'], result['label']), ('NVIDIA RTX PRO 6000', 257, 'DIAL@800k'))
        summary = json.loads((n.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['lane'], summary['completed_cells'], summary['total_cells']), ('dial', 1, 9))
        self.assertEqual(summary['solver_gap_257'], {})
        values['nfe'] = 35
        np.savez(str(directory / 'report_160.npz'), **values)
        with self.assertRaisesRegex(RuntimeError, 'report identity'):
            n.report(name)

    def test_solver_gap_and_combine(self):
        cells = {x: fake_result(x, f) for x, f in (('dial_nfe257_dpm_10k', 1.50), ('dial_nfe257_heun_10k', 1.54),
                                                   ('b_nfe257_dpm_10k', 1.60), ('b_nfe257_heun_10k', 1.70))}
        gaps = n.solver_gaps(cells)
        self.assertTrue(gaps['DIAL']['converged'])
        self.assertFalse(gaps['B']['converged'])
        self.assertAlmostEqual(gaps['B']['dpm_minus_heun'], -0.10)
        logs = {}
        lanes = dict(dial=dict(cells, dial_nfe257_dpm_50k=fake_result('dial_nfe257_dpm_50k', 1.40)),
                     b=dict(b_nfe257_dpm_50k=fake_result('b_nfe257_dpm_50k', 1.43)),
                     a=dict(afhq_dial_nfe257_dpm_10k=fake_result('afhq_dial_nfe257_dpm_10k', 3.0),
                            afhq_vpl_nfe257_dpm_10k=fake_result('afhq_vpl_nfe257_dpm_10k', 3.1)))
        for lane_name, lane_cells in lanes.items():
            logs[lane_name] = self.root / lane_name
            logs[lane_name].mkdir()
            (logs[lane_name] / 'summary.json').write_text(json.dumps(dict(
                lane=lane_name, status='running', completed_cells=len(lane_cells), total_cells=9,
                cells=list(lane_cells.values()))))
        stream = io.StringIO()
        with redirect_stdout(stream):
            n.combine([str(logs[x]) for x in ('dial', 'b', 'a')] + [str(self.root / 'missing')])
        out = json.loads(stream.getvalue())
        self.assertEqual(out['main_contrast']['status'], 'below threshold')
        self.assertAlmostEqual(out['main_contrast']['difference'], -0.03)
        self.assertEqual(out['second_contrast']['status'], 'missing')
        self.assertEqual(out['curves_10k'][0]['status'], 'below threshold')
        self.assertEqual(out['afhq_sign_257'][0]['status'], 'sign kept')
        self.assertEqual(out['afhq_sign_257'][1]['status'], 'missing')
        self.assertEqual(n.contrast(cells, 'B', 'DIAL', 256, HEUN, 10000, 0.05)['status'], 'DIAL better')


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
    echo "run $name" >> "$STATE/events"; printf '%s\n' "$@" > "$STATE/$name.args"; echo '{}' ;;
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
    'celeba_fox': ('score_sde_pytorch/workdirs/celeba_fox_matern32_kappa1000_50k', 160, False),
    'celeba_b': ('celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42', 160, True),
    'celeba_a_first': ('celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42', 80, True),
    'afhq_fox': ('afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42', 4, True),
    'afhq_a': ('afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42', 4, True),
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
        shutil.copy(str(REPO / 'run_nfe_limit.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / q.lo.INCEPTION_MODULE
        for path in (main / 'assets/stats', main / 'datasets/afhqv2-64x64', main / 'celeba', tfhub):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/afhqv2_64_cd017afab009_stats.npz',
                     main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt',
                     tfhub / 'saved_model.pb'):
            path.write_text('x')
        for run, ckpt, has_manifest in RUNS.values():
            (self.base / run / 'checkpoints').mkdir(parents=True)
            (self.base / run / 'checkpoints/checkpoint_{}.pth'.format(ckpt)).write_text('x')
            if has_manifest:
                (self.base / run / 'run_manifest.json').write_text('manifest')

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           NFE_LIMIT_SERVER_BASE=str(self.base), GPU_NAME_1=RTX, GPU_NAME_2=RTX, **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_nfe_limit.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def test_three_lanes_on_three_cards(self):
        for lane, gpu, arch, suffix, works in (('dial', '1', '12.0', 'rtx', ('celeba_fox',)),
                                               ('b', '2', '12.0', 'rtx', ('celeba_b',)),
                                               ('a', '3', '9.0', 'h200', ('celeba_a_first', 'afhq_fox', 'afhq_a'))):
            result = self.launch('start', lane, gpu)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = (self.state / ('celeba-nfe-' + lane + '.args')).read_text()
            self.assertIn('device=' + gpu, args)
            self.assertIn('NFE_LIMIT_LANE=' + lane, args)
            self.assertIn('TORCH_CUDA_ARCH_LIST=' + arch, args)
            self.assertIn('TORCH_EXTENSIONS_DIR=/workspace/score_sde_pytorch/.torch_extensions_' + suffix, args)
            self.assertIn('CUDA_VISIBLE_DEVICES=0', args)
            self.assertEqual(args.count('/checkpoints,readonly'), len(works))
            for work in works:
                self.assertIn('{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(
                    RUNS[work][0], work), args)
            inputs = self.base / 'nfe_limit_20261116' / lane / 'workdirs/_training_inputs'
            self.assertEqual(sorted(p.name for p in inputs.iterdir()), sorted(w + '.json' for w in works if RUNS[w][2]))

    def test_refusals(self):
        self.assertIn(b'lane dial needs RTX', self.launch('start', 'dial', '3').stdout)
        self.assertIn(b'lane a needs H200', self.launch('start', 'a', '1').stdout)
        self.assertEqual(self.launch('start', 'curve', '3').returncode, 2)
        self.assertEqual(self.launch('start', 'a').returncode, 2)
        self.assertIn(b'need 60000', self.launch('start', 'a', '3', FREE_MIB='50000').stdout)
        (self.state / 'celeba-other.gpus').write_text('3,')
        self.assertIn(b'runs containers: celeba-other', self.launch('start', 'a', '3').stdout)
        (self.base / RUNS['afhq_a'][0] / 'checkpoints/checkpoint_4.pth').unlink()
        self.assertIn(b'Missing input', self.launch('start', 'dial', '1').stdout)

    def test_combine_reads_only(self):
        self.assertIn(b'No lane logs', self.launch('combine').stdout)
        logs = self.base / 'nfe_limit_20261116/dial/logs'
        logs.mkdir(parents=True)
        result = self.launch('combine')
        self.assertEqual(result.returncode, 0, result.stdout)
        args = (self.state / 'check.args').read_text()
        self.assertIn('dst=/nfe/dial,readonly', args)
        self.assertNotIn('/nfe/b,', args)
        self.assertEqual(len(list((self.base / 'nfe_limit_20261116').glob('contrasts_*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
