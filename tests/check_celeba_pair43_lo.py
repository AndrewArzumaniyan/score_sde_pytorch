"""CPU checks for the seed 43 cells at 50k; real SDEs and samplers, no checkpoints.

PAIR43_TEST_MANIFESTS may name a directory with a-parent.json and fox.json, copies of the
real manifests of the A seed 43 parent and of FOX seed 43.
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
from types import SimpleNamespace
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
    import glob
    sys.modules['tensorflow'].io.gfile = SimpleNamespace(
        exists=os.path.exists, listdir=os.listdir, glob=glob.glob, GFile=open, remove=os.remove,
        makedirs=lambda path: os.makedirs(path, exist_ok=True),
        rename=lambda source, target, overwrite=False: os.replace(source, target))

import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_pair43_lo_eval as pair
import run_lib
import sampling
from artifact_utils import _canonicalize, build_model_manifest
from configs.factorial.celeba_a_vp_pvp import get_config as get_a

LO = 9.115429865459795
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
A0 = pair.cell_name('A', DPM, 0)
F0 = pair.cell_name('fox', DPM, 0)
CHECKPOINT = 'recorded-by-preflight'


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def a_manifest(seed, snapshot_freq):
    real = os.environ.get('PAIR43_TEST_MANIFESTS')
    if real and seed == 43:
        return json.loads((Path(real) / 'a-parent.json').read_text())
    config = get_a()
    config.seed = seed
    config.training.snapshot_freq = snapshot_freq
    resolved = _canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    return dict(config=resolved, model_manifest=build_model_manifest(config), training_protocol_sha256='a-training')


def fox_manifest():
    real = os.environ.get('PAIR43_TEST_MANIFESTS')
    if real:
        return json.loads((Path(real) / 'fox.json').read_text())
    return dict(config=dict(seed=43), training_protocol_sha256='fox-training')


def on_cpu(config):
    config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
    return config


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        inputs = self.root / 'workdirs/_training_inputs'
        inputs.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        (inputs / (pair.MODELS['A']['work'] + '.json')).write_text(json.dumps(a_manifest(43, 50000)))
        (inputs / (pair.MODELS['fox']['work'] + '.json')).write_text(json.dumps(fox_manifest()))
        (inputs / (followup.WORKDIRS['A'] + '.json')).write_text(json.dumps(a_manifest(42, 5000)))
        for module, target, value in ((pair, 'ROOT', self.root), (pair, 'LOGS', self.root / 'logs'),
                                      (pair, 'TRAINING_INPUTS', inputs), (followup, 'TRAINING_INPUTS', inputs)):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class Plan(Base):
    def test_lanes_split_the_eight_cells_in_order(self):
        expected = ['{}43_c{}_lo_{}_seed{}'.format(model, ckpt, method, seed)
                    for model, ckpt in (('A', 8), ('fox', 160)) for seed in (0, 1) for method in (DPM, HEUN)]
        self.assertEqual(list(pair.CELLS), expected)
        self.assertEqual(pair.lane_cells('a'), expected[:4])
        self.assertEqual(pair.lane_cells('fox'), expected[4:])

    def test_equal_to_the_seed_42_cells_except_checkpoint_and_training(self):
        for model, ckpt in (('A', 80), ('fox', 260)):
            for seed in (0, 1):
                for method in (DPM, HEUN):
                    mine = _canonicalize(pair.get_config(pair.cell_name(model, method, seed)))
                    theirs = _canonicalize(lo.get_config(lo.cell_name(model, method, seed)))
                    self.assertEqual(theirs['eval'].pop('begin_ckpt'), ckpt)
                    theirs['eval'].pop('end_ckpt')
                    mine['eval'].pop('begin_ckpt')
                    mine['eval'].pop('end_ckpt')
                    for section in ('eval', 'sampling', 'model', 'data'):
                        self.assertEqual(mine[section], theirs[section], section)
                    for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
                        self.assertEqual(mine[key], theirs[key], key)

    def test_explicit_values(self):
        for name, spec in pair.CELLS.items():
            config = pair.get_config(name)
            self.assertEqual((config.sampling.logsnr_min, config.sampling.logsnr_max), (-10.0, LO))
            self.assertEqual((config.eval.sampling_num_scales, config.eval.num_samples, config.eval.batch_size),
                             (34, 50000, 512))
            self.assertEqual((config.eval.begin_ckpt, config.eval.end_ckpt), (spec['ckpt'], spec['ckpt']))
            self.assertEqual(config.eval.sampling_seed, spec['seed'])
            self.assertEqual(config.sampling.time_grid, 'karras' if spec['method'] == HEUN else 'uniform_logsnr')

    def test_a_refuses_the_seed_42_run(self):
        pair.manifest_path('A').write_text(json.dumps(a_manifest(42, 5000)))
        with self.assertRaisesRegex(ValueError, 'Unexpected run'):
            pair.get_config(A0)

    def test_every_cell_grid_spans_the_window_and_its_sampler_builds(self):
        for name, spec in pair.CELLS.items():
            config = on_cpu(pair.get_config(name))
            sde, eps = run_lib.get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)

    def test_full_sampler_calls_and_outputs(self):
        for model in pair.MODELS:
            for method in (DPM, HEUN):
                config = on_cpu(pair.get_config(pair.cell_name(model, method, 0)))
                sde, eps = run_lib.get_sde(config, n_override=34)
                outputs, nfe = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), lambda x: x, eps)(ZeroEps())
                self.assertEqual(nfe, 35)
                self.assertEqual(set(outputs), {'x_t', 'tweedie_xt'})


class Loading(Base):
    def test_identity(self):
        for model, step in (('A', 400001), ('fox', 800001)):
            protocol = pair.read_manifest(model)['training_protocol_sha256']
            pair.check_identity(model, step, protocol)
            with self.assertRaisesRegex(RuntimeError, 'Expected step'):
                pair.check_identity(model, step - 1, protocol)
            with self.assertRaisesRegex(RuntimeError, 'run manifest'):
                pair.check_identity(model, step, 'other')

    def test_refuses_another_training_seed(self):
        for model, step in (('A', 400001), ('fox', 800001)):
            manifest = pair.read_manifest(model)
            manifest['config']['seed'] = 42
            pair.manifest_path(model).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, 'Expected training seed'):
                pair.check_identity(model, step, manifest['training_protocol_sha256'])

    def test_loader_reads_snapshots_of_50k(self):
        """A real save and load through a small model: checkpoint_8 holds step 400001."""
        import losses
        from models import utils as mutils
        from models.ema import ExponentialMovingAverage
        from utils import save_checkpoint
        config = on_cpu(pair.get_config(A0))
        small = lambda c: torch.nn.Linear(3, 2)
        model = small(config)
        protocol = build_model_manifest(config)['model_protocol_sha256']
        state = dict(model=model, optimizer=losses.get_optimizer(config, model.parameters()), step=400001,
                     ema=ExponentialMovingAverage(model.parameters(), config.model.ema_rate),
                     model_protocol_sha256=protocol, data_batches_consumed=0, eval_batches_consumed=0,
                     training_protocol_sha256=pair.read_manifest('A')['training_protocol_sha256'])
        path = self.root / 'workdirs' / pair.MODELS['A']['work'] / 'checkpoints/checkpoint_8.pth'
        path.parent.mkdir(parents=True)
        save_checkpoint(str(path), state)
        with patch.object(mutils, 'create_model', small):
            _, identity = pair.load_model(config, pair.CELLS[A0])
        self.assertEqual(identity['step'], 400001)
        self.assertEqual(len(identity['checkpoint_id']), 32)

    def test_missing_checkpoint_is_refused(self):
        from models import utils as mutils
        config = on_cpu(pair.get_config(A0))
        with patch.object(mutils, 'create_model', lambda c: torch.nn.Linear(3, 2)):
            with self.assertRaisesRegex(RuntimeError, 'Missing checkpoint'):
                pair.load_model(config, pair.CELLS[A0])


class Preflight(Base):
    def run_preflight(self, lane, load):
        module = self.root / 'cache' / lo.INCEPTION_MODULE
        (module / 'variables').mkdir(parents=True, exist_ok=True)
        (module / 'saved_model.pb').write_text('x')
        self.loaded = []

        def counted(config, spec):
            self.loaded.append(spec['model'])
            return load(config, spec)

        original = pair.get_config
        with patch.object(pair, 'load_model', counted), \
                patch.object(pair, 'get_config', lambda name: on_cpu(original(name))), \
                patch.object(followup, 'smoke') as self.smoke, patch('torch.cuda.is_available', lambda: True), \
                patch('torch.cuda.device_count', lambda: 1), patch('torch.cuda.get_device_capability', lambda i: (12, 0)), \
                patch.dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0', TFHUB_CACHE_DIR=str(self.root / 'cache')), \
                patch('artifact_utils._environment_manifest', lambda: {}), \
                patch('artifact_utils._source_fingerprint', lambda root: 'source'), \
                patch.object(sampling, 'get_sampling_fn'):
            return pair.preflight(lane)

    @staticmethod
    def good(config, spec):
        return ZeroEps(), dict(step=pair.MODELS[spec['model']]['step'], checkpoint_id=CHECKPOINT)

    def test_each_lane_touches_only_its_model_and_records_the_id(self):
        for lane, model in (('a', 'A'), ('fox', 'fox')):
            self.assertEqual(self.run_preflight(lane, self.good), 0)
            self.assertEqual((self.loaded, self.smoke.call_count), ([model], 4))
            saved = json.loads((pair.LOGS / 'preflight.json').read_text())
            self.assertEqual(saved['checkpoints'][model]['checkpoint_id'], CHECKPOINT)

    def test_a_failed_load_blocks_the_lane(self):
        def load(config, spec):
            raise RuntimeError('Expected step')

        self.assertEqual(self.run_preflight('a', load), 1)
        self.assertEqual(len(json.loads((pair.LOGS / 'preflight.json').read_text())['failures']), 4)


class Report(Base):
    def prepare(self, name, nfe=None, step=None, shards=98, checkpoint_id=CHECKPOINT):
        spec = pair.CELLS[name]
        expected = pair.MODELS[spec['model']]
        config = pair.get_config(name)
        directory = self.root / 'workdirs' / expected['work'] / pair.eval_folder(name)
        rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
        rounds.mkdir(parents=True)
        for index in range(shards):
            (rounds / 'statistics_{}.npz'.format(index)).write_text('x')
        np.savez(str(rounds / 'statistics_0.npz'), **{pair.FEATURES: np.zeros((512, 2048), np.float32)})
        resolved = json.loads(json.dumps(_canonicalize(config)))
        resolved['eval'].pop('begin_ckpt')  # Eval manifests exclude the checkpoint range.
        resolved['eval'].pop('end_ckpt')
        (directory / 'manifest.json').write_text(json.dumps(dict(config=resolved, source_sha256='source')))
        (pair.LOGS / 'preflight.json').write_text(json.dumps(dict(source_sha256='source', failures={}, checkpoints={
            spec['model']: dict(checkpoint_id=CHECKPOINT, step=expected['step'])})))
        (pair.LOGS / (name + '_timing.json')).write_text(json.dumps(dict(elapsed_seconds=60.0)))
        values = dict(num_samples=50000, nfe=nfe or 35, checkpoint_step=step or expected['step'],
                      checkpoint_id=checkpoint_id, dataset_stats_id=pair.REFERENCE_SHA)
        for metric in ('IS', 'fid', 'kid'):
            for variant in config.eval.endpoint_denoise_outputs:
                values[metric + '_' + variant] = 2.5
        np.savez(str(directory / 'report_{}.npz'.format(spec['ckpt'])), **values)
        return directory

    def test_report_of_each_model(self):
        for name, lane, step in ((A0, 'a', 400001), (F0, 'fox', 800001)):
            self.prepare(name)
            pair.report(name, lane)
            result = json.loads(pair.result_path(name).read_text())
            self.assertEqual((result['nfe'], result['num_samples'], result['checkpoint_step'], result['checkpoint_id'],
                              result['lambda_window']), (35, 50000, step, CHECKPOINT, [-10.0, LO]))
            summary = json.loads((pair.LOGS / 'summary.json').read_text())
            self.assertEqual((summary['status'], summary['completed_cells'], summary['total_cells']), ('running', 1, 4))

    def test_refuses_wrong_identity_and_shards(self):
        for arguments, message in ((dict(nfe=65), 'report identity'), (dict(step=395001), 'report identity'),
                                   (dict(checkpoint_id='other'), 'report identity'), (dict(shards=20), 'shards')):
            directory = self.prepare(A0, **arguments)
            with self.assertRaisesRegex(RuntimeError, message):
                pair.report(A0, 'a')
            shutil.rmtree(str(directory))


class Queue(Base):
    def test_a_failed_cell_gets_a_second_pass_and_the_others_run(self):
        (pair.LOGS / 'preflight.json').write_text(json.dumps(dict(failures={}, checkpoints={})))

        def fake(name):
            if name == A0:
                raise subprocess.CalledProcessError(1, 'main.py')
            pair.result_path(name).write_text('{}')

        with patch.object(pair, 'run', side_effect=fake) as run, patch.object(pair, 'report', lambda name, lane: None), \
                patch.object(pair, 'write_summary') as summary:
            code = pair.queue('a')
        self.assertEqual((code, [call[0][0] for call in run.call_args_list]), (1, pair.lane_cells('a') + [A0]))
        summary.assert_called_once_with('a', [A0])

    def test_blocked_lane_runs_nothing(self):
        (pair.LOGS / 'preflight.json').write_text(json.dumps(dict(
            failures={name: 'x' for name in pair.lane_cells('fox')}, checkpoints={})))
        with patch.object(pair, 'run') as run, patch.object(pair, 'write_summary'):
            self.assertEqual(pair.queue('fox'), 1)
        run.assert_not_called()


DOCKER = r"""#!/usr/bin/env bash
# State in $STATE: NAME.polls (running while > 0), NAME.code, NAME.gpus; events in $STATE/events.
last="${@: -1}"
polls() { local n; n="$(cat "$STATE/$1.polls")"; if (( n > 0 )); then echo $((n - 1)) > "$STATE/$1.polls"; echo true; else echo false; fi; }
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.polls; do [[ -e "$f" ]] || continue; n="$(basename "$f" .polls)"; (( $(cat "$f") > 0 )) && echo "$n"; done ;;
  inspect)
    [[ -e "$STATE/$last.polls" ]] || { echo; exit 1; }
    case "$*" in
      *State.Running*) polls "$last" ;;
      *ExitCode*) cat "$STATE/$last.code" ;;
      *DeviceIDs*) cat "$STATE/$last.gpus" ;;
      *.Name*) echo "/$last" ;;
    esac ;;
  run)
    name=""; prev=""; for arg in "$@"; do [[ "$prev" == --name ]] && name="$arg"; prev="$arg"; done
    echo "run $name" >> "$STATE/events"
    echo "${EVAL_POLLS:-2}" > "$STATE/$name.polls"; echo "${EVAL_CODE:-0}" > "$STATE/$name.code"; echo "1," > "$STATE/$name.gpus" ;;
esac
exit 0
"""
STUBS = {
    'git': '#!/usr/bin/env bash\n[[ "$1" == rev-parse ]] && echo commit\nexit 0\n',
    'nvidia-smi': '#!/usr/bin/env bash\necho "${FREE_MIB:-81000}"\n',
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
    'next': '#!/usr/bin/env bash\necho "afhqv2 $*" >> "$STATE/events"\n',
}


class AfterA(unittest.TestCase):
    """The host launcher on stub docker, git and nvidia-smi: arguments, order and stops."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.bin, self.state, self.base = self.root / 'bin', self.root / 'state', self.root / 'base'
        for path in (self.bin, self.state, self.root / 'worktree'):
            path.mkdir()
        for name, text in dict(STUBS, docker=DOCKER).items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        shutil.copy(str(REPO / 'run_celeba_pair43_lo_rtx.sh'), str(self.root / 'worktree'))
        work = self.base / 'celeba_a43_400k_rtx_20261107/workdirs/celeba_ab_replication_A_vp_pvp_400k_rtx_seed43'
        main = self.base / 'score_sde_pytorch'
        tfhub = self.base / 'celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules' / lo.INCEPTION_MODULE
        for path in (work / 'checkpoints', main / 'assets/stats', main / 'celeba', tfhub):
            path.mkdir(parents=True)
        for path in (work / 'checkpoints/checkpoint_8.pth', work / 'run_manifest.json', tfhub / 'saved_model.pb',
                     main / 'assets/stats/celeba_64_0c2d89311192_stats.npz', main / 'celeba/list_eval_partition.txt'):
            path.write_text('x')
        self.training('a43', polls=1, code=0)
        self.write_summary('completed', [])

    def training(self, name, polls, code, gpus='1,'):
        name = {'a43': 'celeba-a43-400k-rtx'}.get(name, name)
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))
        (self.state / (name + '.gpus')).write_text(gpus)

    def write_summary(self, status, failed):
        logs = self.base / 'celeba_pair43_lo_20261109/a/logs'
        logs.mkdir(parents=True, exist_ok=True)
        (logs / 'summary.json').write_text(json.dumps(dict(status=status, failed_cells=failed), indent=2) + '\n')

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           PAIR43_SERVER_BASE=str(self.base), PAIR43_NEXT=str(self.bin / 'next'),
                           PAIR43_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.root / 'worktree/run_celeba_pair43_lo_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_arguments(self):
        for arguments in (('after', 'fox', '1'), ('after', 'a'), ('after', 'a', '1', '2'), ('after', 'a', 'x')):
            self.assertEqual(self.launch(*arguments).returncode, 2, arguments)
        (self.state / 'celeba-a43-400k-rtx.polls').unlink()
        result = self.launch('after', 'a', '1')
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'No container celeba-a43-400k-rtx', result.stdout)

    def test_another_container_of_ours_on_the_gpu_is_refused(self):
        self.training('celeba-other', polls=99, code=0)
        self.assertIn(b'GPU 1 runs containers: celeba-other', self.launch('after', 'a', '1').stdout)
        self.training('a_ermilova_huggs', polls=99, code=0, gpus='0,1,2,3,4,')
        (self.state / 'celeba-other.polls').write_text('0')
        self.assertEqual(self.launch('_wait', '1').returncode, 0)

    def test_order_training_then_lane_then_afhqv2(self):
        result = self.launch('_wait', '1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run celeba-pair43-lo-a', 'afhqv2 queue a-vp 1'])

    def test_after_returns_at_once_and_runs_in_the_background(self):
        self.training('a43', polls=3, code=0)
        result = self.launch('after', 'a', '1')
        self.assertIn(b'Waiting for celeba-a43-400k-rtx', result.stdout)
        for _ in range(100):
            if len(self.events()) == 2:
                break
            time.sleep(0.1)
        self.assertEqual(self.events(), ['run celeba-pair43-lo-a', 'afhqv2 queue a-vp 1'])

    def assert_stops(self, message, events, **env):
        result = self.launch('_wait', '1', **env)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(message, result.stdout)
        self.assertEqual(self.events(), events)

    def test_failed_training_starts_nothing(self):
        self.training('a43', polls=1, code=137)
        self.assert_stops(b'did not exit with 0', [])

    def test_missing_checkpoint_starts_nothing(self):
        (self.base / 'celeba_a43_400k_rtx_20261107/workdirs/celeba_ab_replication_A_vp_pvp_400k_rtx_seed43'
         / 'checkpoints/checkpoint_8.pth').unlink()
        self.assert_stops(b'missing', [])

    def test_failed_lane_does_not_start_afhqv2(self):
        self.assert_stops(b'AFHQv2 not started', ['run celeba-pair43-lo-a'], EVAL_CODE='1')

    def test_failed_cells_do_not_start_afhqv2(self):
        self.write_summary('failed', ['A43_c8_lo_ode_edm_heun_seed1'])
        self.assert_stops(b'AFHQv2 not started', ['run celeba-pair43-lo-a'])

    def test_unfinished_summary_does_not_start_afhqv2(self):
        self.write_summary('running', [])
        self.assert_stops(b'AFHQv2 not started', ['run celeba-pair43-lo-a'])

    def test_timeouts(self):
        self.training('a43', polls=99, code=0)
        self.assert_stops(b'still runs; nothing started', [], PAIR43_WAIT_MAX_SECONDS='0')
        self.training('a43', polls=0, code=0)
        self.assert_stops(b'AFHQv2 not started', ['run celeba-pair43-lo-a'], PAIR43_EVAL_MAX_SECONDS='0',
                          EVAL_POLLS='99')

    def test_low_memory_starts_nothing(self):
        self.assert_stops(b'refused to start', [], FREE_MIB='1000')


if __name__ == '__main__':
    unittest.main()
