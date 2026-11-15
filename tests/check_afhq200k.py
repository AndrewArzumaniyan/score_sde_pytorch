"""CPU checks for AFHQv2 A and VP at 200k and their launcher; no GPU, data or checkpoints."""
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

import afhqv2_rtx_train as afhq
import afhq200k_eval as k
import celeba_followup_eval as followup
import gpu1_vp_tail_eval as tail
import gpu2_queue_eval as q
import run_lib  # noqa: F401, imported before tests patch torch.cuda.is_available.
from artifact_utils import _canonicalize, build_model_manifest

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
    lane = 'a'

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        for path in ('logs', 'previous'):
            (self.root / path).mkdir()
        for key, arm in (('afhq_vp', 'vp'), ('afhq_fox', 'fox'), ('afhq_a', 'a')):
            (inputs / (key + '.json')).write_text(json.dumps(manifest(arm)))
        for module, target, value in ((q, 'ROOT', self.root), (q, 'LOGS', self.root / 'logs'), (q, 'INPUTS', inputs),
                                      (k, 'ROOT', self.root), (k, 'LOGS', self.root / 'logs'),
                                      (k, 'PREVIOUS', self.root / 'previous'), (tail, 'LOGS', self.root / 'logs')):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, AFHQ200K_LANE=self.lane)
        patcher.start()
        self.addCleanup(patcher.stop)


class Plan(Base):
    def test_cells_and_lanes(self):
        names = lambda source: ['{}_c4_m35_{}_seed{}'.format(source, m, s) for s in (0, 1) for m in (DPM, HEUN)]
        self.assertEqual(list(k.CELLS), names('afhq_a') + names('afhq_vp'))
        self.assertEqual(k.lane_cells('a'), names('afhq_a'))
        self.assertEqual(k.lane_cells('vp'), names('afhq_vp'))
        for name in k.CELLS:
            spec = k.resolve(name)
            self.assertEqual((spec['ckpt'], spec['step'], spec['work'], spec['count'], spec['calls']),
                             (4, 200001, spec['source'], 50000, 34))
            self.assertEqual(q.training_protocol(spec['source']), 'protocol-' + spec['lane'])

    def test_lane_is_required(self):
        with patch.dict(os.environ, AFHQ200K_LANE='fox'):
            with self.assertRaisesRegex(ValueError, 'must be a or vp'):
                k.lane()

    def test_configs_equal_the_250k_cells_except_the_checkpoint(self):
        (q.LOGS / q.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=5)))
        (tail.LOGS / tail.SELECTION).write_text(json.dumps(dict(complete=True, selected_ckpt=5)))
        for seed in (0, 1):
            for method in (DPM, HEUN):
                for mine, theirs in (
                        (k.get_config('afhq_a_c4_m35_{}_seed{}'.format(method, seed)),
                         q.get_config('afhq_a_cbest_m35_{}_seed{}'.format(method, seed))),
                        (k.get_config('afhq_vp_c4_m35_{}_seed{}'.format(method, seed)),
                         tail.get_config('afhq_vp_cbest_m35_{}_seed{}'.format(method, seed)))):
                    self.assertEqual(without_range(mine), without_range(theirs), (method, seed))
                    self.assertEqual((mine.eval.begin_ckpt, theirs.eval.begin_ckpt), (4, 5))
                    self.assertEqual(mine.sampling.logsnr_max, 9.115429865459795)
                    self.assertEqual(mine.sampling.time_grid, 'karras' if method == HEUN else 'uniform_logsnr')

    def test_every_cell_passes_the_dataset_check_of_the_evaluation(self):
        import endpoint_denoise_eval
        for name in k.CELLS:
            config = k.get_config(name)
            config.eval.batch_size = 0
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                endpoint_denoise_eval.evaluate_checkpoint(config, str(self.root), 4, None, None, None, 'p', 'c', 1, 't', {})


class Report(Base):
    def prepare(self, name, step=None, features=True, checkpoint_id='id'):
        spec = k.resolve(name)
        directory = self.root / 'workdirs' / spec['work'] / k.eval_folder(name)
        rounds = directory / 'ckpt_4'
        rounds.mkdir(parents=True)
        for index in range(98):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{q.FEATURES if features else 'other': np.zeros((512, 2048))})
        (directory / 'manifest.json').write_text(json.dumps(dict(config=without_range(q.config_of(spec)))))
        (k.LOGS / 'preflight.json').write_text(json.dumps(dict(
            dataset_stats_id=dict(AFHQV2='afhq'), checkpoints={spec['group']: dict(checkpoint_id='id', step=200001)})))
        (k.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=50000, nfe=35, checkpoint_step=step or 200001, dataset_stats_id='afhq',
                      training_protocol_sha256='protocol-a', checkpoint_id=checkpoint_id)
        for metric in ('IS', 'fid', 'kid'):
            for variant in q.stage0.VARIANTS:
                values[metric + '_' + variant] = 3.0
        np.savez(str(directory / 'report_4.npz'), **values)
        return directory

    def test_report_and_summary(self):
        (k.PREVIOUS / 'a_c4_lo_result.json').write_text(json.dumps(dict(fid_tweedie_xt=3.506)))
        self.prepare('afhq_a_c4_m35_ode_edm_heun_seed0')
        k.report('afhq_a_c4_m35_ode_edm_heun_seed0')
        summary = json.loads((k.LOGS / 'summary.json').read_text())
        self.assertEqual((summary['lane'], summary['completed_cells'], summary['total_cells']), ('a', 1, 4))
        self.assertEqual(summary['ten_k_200k'], dict(cell='a_c4_lo', fid_tweedie_xt=3.506))
        self.assertEqual(summary['cells'][0]['step'], 200001)
        self.assertEqual(len((k.LOGS / 'summary.csv').read_text().splitlines()), 2)

    def test_refusals(self):
        for changes, message in ((dict(step=250001), 'report identity'), (dict(features=False), 'bootstrap'),
                                 (dict(checkpoint_id='other'), 'different checkpoints')):
            directory = self.prepare('afhq_a_c4_m35_ode_dpmpp_2m_seed1', **changes)
            with self.assertRaisesRegex(RuntimeError, message):
                k.report('afhq_a_c4_m35_ode_dpmpp_2m_seed1')
            shutil.rmtree(str(directory))


class Preflight(Base):
    lane = 'vp'

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
            code = k.preflight()
        return code, json.loads((k.LOGS / 'preflight.json').read_text()), call

    def test_real_path_on_the_lane_weights(self):
        code, saved, call = self.run_preflight(lambda config, model, spec, directory: 3.0)
        self.assertEqual((code, saved['status'], saved['lane']), (0, 'passed', 'vp'))
        self.assertEqual(saved['checkpoints'], dict(afhq_vp_m35=dict(checkpoint_id='id', step=200001)))
        seen = [(a[2]['source'], a[2]['ckpt'], a[2]['method'], a[0].eval.sampling_num_scales, a[0].training.sde)
                for a, _ in call.call_args_list]
        self.assertEqual(seen, [('afhq_vp', 4, DPM, 34, 'vpsde')])

    def test_failure(self):
        def smoke(config, model, spec, directory):
            raise ValueError('bad')

        code, saved, _ = self.run_preflight(smoke)
        self.assertEqual((code, sorted(saved['failures'])), (1, ['afhq_vp_m35']))


class Queue(Base):
    def run_queue(self, blocked=None, fail=()):
        (k.LOGS / 'preflight.json').write_text(json.dumps(dict(failures=blocked or {})))
        order = []

        def evaluate(name):
            order.append(name)
            return name not in fail

        with patch.object(k, 'evaluate', evaluate), patch.object(k, 'write_summary') as summary:
            code = k.queue()
        return code, order, summary

    def test_only_the_lane_cells_in_order(self):
        code, order, _ = self.run_queue()
        self.assertEqual((code, order), (0, k.lane_cells('a')))

    def test_failed_cell_does_not_stop_the_next(self):
        bad = k.lane_cells('a')[1]
        code, order, summary = self.run_queue(fail=(bad,))
        self.assertEqual((code, order), (1, k.lane_cells('a')))
        summary.assert_called_once_with([bad])

    def test_blocked(self):
        code, order, _ = self.run_queue(blocked=dict(afhq_a_m35='bad'))
        self.assertEqual((code, order), (1, []))


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
RUNS = {'a': ('afhq_a', 'afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42'),
        'vp': ('afhq_vp', 'afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42')}


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
        shutil.copy(str(REPO / 'run_afhq200k_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / q.lo.INCEPTION_MODULE
        previous = self.base / 'afhqv2_curve_20261112_r2/logs'
        for path in (main / 'assets/stats', main / 'datasets/afhqv2-64x64', tfhub, previous):
            path.mkdir(parents=True, exist_ok=True)
        for path in (main / 'assets/stats/afhqv2_64_cd017afab009_stats.npz', tfhub / 'saved_model.pb',
                     previous / 'a_c4_lo_result.json', previous / 'vp_c4_lo_result.json'):
            path.write_text('x')
        for _, run in RUNS.values():
            (self.base / run / 'checkpoints').mkdir(parents=True)
            (self.base / run / 'checkpoints/checkpoint_4.pth').write_text('x')
            (self.base / run / 'run_manifest.json').write_text('manifest')

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           AFHQ200K_SERVER_BASE=str(self.base), **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_afhq200k_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_two_lanes_on_two_gpus(self):
        for lane, gpu in (('a', '1'), ('vp', '2')):
            result = self.launch('start', lane, gpu)
            self.assertEqual(result.returncode, 0, result.stdout)
            args = (self.state / 'afhqv2-200k-{}.args'.format(lane)).read_text()
            work, run = RUNS[lane]
            self.assertIn('device=' + gpu, args)
            self.assertIn('AFHQ200K_LANE=' + lane, args)
            self.assertIn('{}/checkpoints,dst=/workspace/score_sde_pytorch/workdirs/{}/checkpoints,readonly'.format(
                run, work), args)
            self.assertIn('afhqv2_curve_20261112_r2/logs,dst=/workspace/score_sde_pytorch/previous,readonly', args)
            self.assertIn('afhq200k_bootstrap_20261115/{}/logs,dst=/workspace/score_sde_pytorch/logs'.format(lane), args)
            self.assertNotIn('--rm', args)
            inputs = self.base / 'afhq200k_bootstrap_20261115' / lane / 'workdirs/_training_inputs'
            self.assertEqual([p.name for p in inputs.iterdir()], [work + '.json'])
        self.assertEqual(self.events(), ['run afhqv2-200k-a', 'run afhqv2-200k-vp'])

    def test_refusals(self):
        self.assertEqual(self.launch('start', 'a', '3', GPU_NAME='NVIDIA H200 NVL').returncode, 1)
        self.assertEqual(self.launch('start', 'fox', '1').returncode, 2)
        self.assertEqual(self.launch('start', 'a').returncode, 2)
        self.assertIn(b'need 60000', self.launch('start', 'a', '1', FREE_MIB='50000').stdout)
        (self.state / 'celeba-other.gpus').write_text('1,')
        self.assertIn(b'runs containers: celeba-other', self.launch('start', 'a', '1').stdout)
        self.assertEqual(self.launch('start', 'vp', '2').returncode, 0)
        (self.state / 'afhqv2-200k-vp.gpus').write_text('2,')
        self.assertIn(b'Container exists', self.launch('start', 'vp', '2').stdout)
        (self.base / RUNS['vp'][1] / 'checkpoints/checkpoint_4.pth').unlink()
        self.assertIn(b'Missing input', self.launch('start', 'a', '2').stdout)
        self.assertEqual(self.events(), ['run afhqv2-200k-vp'])


if __name__ == '__main__':
    unittest.main()
