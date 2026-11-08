"""CelebA at 50k on the lower window: the second comparison, A seed 43 and FOX seed 43.

A seed 43 at 400k and FOX seed 43 at 800k, eight cells of the celeba_50k_lo_eval.py protocol:
matched35 (34 trajectory calls plus Tweedie, DPM++ and EDM Heun, sampling seeds 0 and 1),
window [-10; 9.115429865459795]. Shards and states are saved the same way, so the features
can go to a bootstrap. One training run per model.

Two independent lanes share no files: a and fox, each with its own workdirs and logs, on
whichever GPU is free. Contrasts between A and FOX therefore come from reading both
summaries. Compare them inside this pair only: the May pair ran under other code. A seed 43
went from 200k to 400k on an RTX PRO 6000 with TF32; FOX seed 43 trained on an H200.

The checkpoints did not exist when this was written, so their ids are not fixed in advance.
Preflight checks the step, the model protocol and the training protocol of the run manifest,
and records the id; reports must carry that id. A preflight failure blocks the cells of the
lane. Inception comes from a local copy of the tfhub cache.

The comparison rule, approved before the first evaluation of this pair:
FOX at 800k minus A at 400k, separately for training seeds 42 and 43; window
[-10; 9.115429865459795], 35 calls, 50k, DPM++ and EDM Heun. The sign is read from the mean
of the two sampling seeds. A gap below 0.05 is called indistinguishable. The result repeats
if the sign is the same in both pairs under both solvers. A negative result is reported.
Absolute FID of the two pairs is not mixed. A bootstrap interval of these cells shows the
spread of generated samples at fixed weights, not the spread between trainings.

Run through run_celeba_pair43_lo_rtx.sh from a separate worktree.
"""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_stage0_eval as stage0
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, set_protocol, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
TRAINING_INPUTS = ROOT / 'workdirs/_training_inputs'
BATCH = lo.BATCH
CALLS = lo.CALLS
COUNT = lo.COUNT
WINDOW = lo.WINDOW
ATTEMPTS = lo.ATTEMPTS
METHODS = lo.METHODS
FEATURES = lo.FEATURES
SEED = 43

MODELS = {
    'A': dict(work='celeba_ab_replication_A_vp_pvp_400k_rtx_seed43', ckpt=8, step=400001),  # Snapshots every 50k.
    'fox': dict(work='celeba_fox_matern32_kappa1000_h200_seed43', ckpt=160, step=800001),
}
LANES = {'a': 'A', 'fox': 'fox'}


def cell_name(model, method, seed):
    return '{}43_c{}_lo_{}_seed{}'.format(model, MODELS[model]['ckpt'], method, seed)


# The order is fixed in advance and does not depend on any observed FID.
CELLS = {cell_name(model, method, seed): dict(model=model, ckpt=MODELS[model]['ckpt'], window=WINDOW,
                                              protocol='matched35', method=method, seed=seed, count=COUNT,
                                              calls=CALLS)
         for model in MODELS for seed in (0, 1) for method in METHODS}


def lane_cells(lane):
    if lane not in LANES:
        raise ValueError('Unknown lane: ' + lane)
    return [name for name, spec in CELLS.items() if spec['model'] == LANES[lane]]


def contrasts(lane):
    """Left minus right, FID after Tweedie, inside one lane."""
    model = LANES[lane]
    result = []
    for seed in (0, 1):
        result.append(('{}_dpmpp_minus_heun_seed{}'.format(model, seed), cell_name(model, METHODS[0], seed),
                       cell_name(model, METHODS[1], seed), 'Solver, same seed'))
    for method in METHODS:
        result.append(('{}_{}_seed1_minus_seed0'.format(model, method), cell_name(model, method, 1),
                       cell_name(model, method, 0), 'Sampling seed, same solver'))
    return result


def manifest_path(model):
    return TRAINING_INPUTS / (MODELS[model]['work'] + '.json')


def read_manifest(model):
    return json.loads(manifest_path(model).read_text())


def a_config(spec):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    config = ConfigDict(read_manifest('A')['config'])
    if config.training.factorial_arm != 'A_vp_pvp' or config.seed != SEED:
        raise ValueError('Unexpected run behind A')
    config.eval = get_a().eval  # Manifests exclude evaluation settings.
    set_protocol(config, spec['window'], spec['ckpt'])
    config.sampling.method = spec['method']
    config.sampling.time_grid = 'karras' if spec['method'] == 'ode_edm_heun' else 'uniform_logsnr'
    config.eval.sampling_num_scales = spec['calls']
    config.eval.sampling_seed = spec['seed']
    config.eval.num_samples = spec['count']
    return config


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown seed 43 cell: ' + name)
    spec = CELLS[name]
    return a_config(spec) if spec['model'] == 'A' else lo.native_config(spec)


def load_model(config, spec):
    """stage0.load_model assumes 5k snapshots, which A seed 43 does not have."""
    from artifact_utils import build_model_manifest
    from losses import get_optimizer
    from models import utils as mutils, ncsnpp  # noqa: F401, registers the model.
    from models.ema import ExponentialMovingAverage
    from utils import restore_checkpoint
    expected = MODELS[spec['model']]
    model = mutils.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, optimizer=get_optimizer(config, model.parameters()), ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(config)['model_protocol_sha256'])
    path = ROOT / 'workdirs' / expected['work'] / 'checkpoints/checkpoint_{}.pth'.format(expected['ckpt'])
    if not path.is_file():
        raise RuntimeError('Missing checkpoint: ' + str(path))
    state = restore_checkpoint(str(path), state, config.device, restore_rng=False)
    check_identity(spec['model'], int(state['step']), state['training_protocol_sha256'])
    ema.copy_to(model.parameters())
    model.eval()
    return model, dict(checkpoint_id=state['checkpoint_id'], step=int(state['step']))


def check_identity(model, step, training_protocol):
    if step != MODELS[model]['step']:
        raise RuntimeError('Expected step {}, found {}'.format(MODELS[model]['step'], step))
    manifest = read_manifest(model)
    if manifest['config']['seed'] != SEED:
        raise RuntimeError('Expected training seed {}, found {}'.format(SEED, manifest['config']['seed']))
    if training_protocol != manifest['training_protocol_sha256']:
        raise RuntimeError('The checkpoint does not belong to the run manifest.')


def check_cell(name, spec, loaded):
    """Grid, sampler, real weights and a short real run; loaded keeps the model."""
    import sampling
    from run_lib import get_sde
    config = get_config(name)
    sde, eps = get_sde(config, n_override=spec['calls'])
    followup.check_cell_grid(config, sde, eps, spec)
    sampling.get_sampling_fn(config, sde, (BATCH, 3, 64, 64), lambda x: x, eps)
    if 'model' not in loaded:
        loaded['model'], loaded['identity'] = load_model(config, spec)
    followup.smoke(config, loaded['model'], spec['method'], eps)
    return loaded['identity']


def preflight(lane):
    """Returns 1 if a cell failed. Trouble shared by the lane raises and leaves no preflight.json."""
    import torch
    names = lane_cells(lane)
    path = LOGS / 'preflight.json'
    if path.exists():
        path.unlink()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    identities, failures, loaded = {}, {}, {}
    for name in names:
        spec = CELLS[name]
        try:
            identities[spec['model']] = check_cell(name, spec, loaded)
            print('Preflight', name, identities[spec['model']], flush=True)
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            failures[name] = repr(error)
            write_json(failure_path(name), dict(error=repr(error), attempt=0, time=time.time()))
            print('Preflight failed: {}: {!r}'.format(name, error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', lane=lane, checkpoints=identities,
        failures=failures, source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight finished: {} of {} cells passed.'.format(len(names) - len(failures), len(names)), flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_pair43_lo_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_pair43_lo_eval.py:' + name, '--workdir=workdirs/' + MODELS[CELLS[name]['model']]['work'],
        '--eval_folder=' + eval_folder(name)], cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name, lane):
    import numpy as np
    from artifact_utils import _canonicalize
    spec = CELLS[name]
    expected_model = MODELS[spec['model']]
    config = get_config(name)
    directory = ROOT / 'workdirs' / expected_model['work'] / eval_folder(name)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(config[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
        if manifest['config'].get(key) != config[key]:
            raise RuntimeError('Unexpected precision setting: ' + key)
    pre = json.loads((LOGS / 'preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Report comes from different sources.')
    checkpoint_id = pre['checkpoints'][spec['model']]['checkpoint_id']
    nfe = spec['calls'] + 1
    variants = tuple(config.eval.endpoint_denoise_outputs)
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=expected_model['step'],
                        checkpoint_id=checkpoint_id, dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in variants}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
    shards = len(list(rounds.glob('statistics_*.npz')))
    if shards != (spec['count'] + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    with np.load(str(rounds / 'statistics_0.npz'), allow_pickle=False) as data:
        if FEATURES not in data.files or data[FEATURES].shape != (BATCH, 2048):
            raise RuntimeError('Shards lack the bootstrap features.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=expected_model['step'],
        checkpoint_id=checkpoint_id, lambda_window=[LOGSNR_MIN, WINDOWS[spec['window']]], nfe=nfe,
        num_samples=spec['count'], sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary(lane)
    print('Seed 43 {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(lane, failed=()):
    names = lane_cells(lane)
    cells = {name: json.loads(result_path(name).read_text()) for name in names if result_path(name).exists()}
    point = [dict(contrast=label, left=left, right=right, question=question,
                  value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
             for label, left, right, question in contrasts(lane) if left in cells and right in cells]
    status = 'completed' if len(cells) == len(names) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, lane=lane, completed_cells=len(cells),
        total_cells=len(names), failed_cells=list(failed), cells=list(cells.values()), point_contrasts=point,
        confidence_intervals_computed=False, scope='One training run per model; compare inside this pair only'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'step', 'lambda_max', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['checkpoint_step'], value['lambda_window'][1],
                             value['method'], value['sampling_seed'], value['num_samples'], value['nfe'],
                             value['fid_tweedie_xt'], value['fid_x_t'], value['kid_tweedie_xt'],
                             value['IS_tweedie_xt']))


def queue(lane):
    """Runs the cells that passed preflight. A failed cell does not stop the next and gets a second pass."""
    names = lane_cells(lane)
    path = LOGS / 'preflight.json'
    if not path.exists():
        print('No preflight.json: preflight stopped the lane.', flush=True)
        return 1
    failures = json.loads(path.read_text())['failures']
    blocked = [name for name in names if name in failures and not result_path(name).exists()]
    failed = []
    for attempt in range(1, ATTEMPTS + 1):
        failed = []
        for name in names:
            if result_path(name).exists() or name in blocked:
                continue
            try:
                run(name)
                report(name, lane)
                if failure_path(name).exists():
                    failure_path(name).unlink()
            except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
                write_json(failure_path(name), dict(error=repr(error), attempt=attempt, time=time.time()))
                print('Seed 43 cell failed: {}: {!r}'.format(name, error), flush=True)
                failed.append(name)
        if not failed:
            break
    failed = blocked + failed
    write_summary(lane, failed)
    print('Seed 43 queue {} finished, failed cells: {}'.format(lane, failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight' and len(sys.argv) == 3:
        sys.exit(preflight(sys.argv[2]))
    elif mode == 'queue' and len(sys.argv) == 3:
        sys.exit(queue(sys.argv[2]))
    elif mode == 'plan':
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, cell_spec)
    else:
        raise SystemExit('Usage: celeba_pair43_lo_eval.py preflight LANE|queue LANE|plan')
