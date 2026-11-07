"""CelebA at 50k on the lower window: FOX, training seed 42, at 800k.

Four cells of the celeba_50k_lo_eval.py protocol: matched35 (34 trajectory calls plus
Tweedie, DPM++ and EDM Heun, seeds 0 and 1), window [-10; 9.115429865459795]. Only the
checkpoint differs from its fox_c260_lo_* cells. Shards and states are saved the same way,
so the features can go to a bootstrap.

Stage 0 has no cell for this checkpoint on the lower window. Its fox_c160_hi is another
window, 64 calls and 10k: REFERENCE is for reading only and stops nothing.

A preflight failure, including a wrong step or checkpoint id, blocks the cells. Inception
comes from a local copy of the tfhub cache; without the copy, preflight stops the queue.

Run through run_celeba_fox800_lo_rtx.sh from a separate worktree.
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
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
BATCH = lo.BATCH
CALLS = lo.CALLS
COUNT = lo.COUNT
WINDOW = lo.WINDOW
ATTEMPTS = lo.ATTEMPTS
METHODS = lo.METHODS
FEATURES = lo.FEATURES

MODEL = 'fox'
WORK = stage0.FOX
CKPT = 160
STEP = 800001  # Read from the checkpoint by the stage 0 preflight, entry fox_160.
CHECKPOINT_ID = 'legacy-sha256-7874545fa077e31812c4ca6cddf196d6102ec91e4ce88423a6ef8f43bc35b148'
REFERENCE = dict(value=2.4593145847320557,
                 source='stage 0, fox_c160_hi: upper window, DPM++ with 65 calls, 10k')


def cell_name(method, seed):
    return '{}_c{}_lo_{}_seed{}'.format(MODEL, CKPT, method, seed)


# The order is fixed in advance and does not depend on any observed FID.
CELLS = {cell_name(method, seed): dict(model=MODEL, ckpt=CKPT, window=WINDOW, protocol='matched35', method=method,
                                       seed=seed, count=COUNT, calls=CALLS)
         for seed in (0, 1) for method in METHODS}


def contrasts():
    """Left minus right, FID after Tweedie."""
    result = []
    for seed in (0, 1):
        result.append(('fox_dpmpp_minus_heun_seed{}'.format(seed), cell_name(METHODS[0], seed),
                       cell_name(METHODS[1], seed), 'Solver, same seed'))
    for method in METHODS:
        result.append(('fox_{}_seed1_minus_seed0'.format(method), cell_name(method, 1), cell_name(method, 0),
                       'Sampling seed, same solver'))
    return result


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown FOX 800k cell: ' + name)
    return lo.native_config(CELLS[name])


def load_model(config):
    model, identity = stage0.load_model(config, WORK, CKPT)
    if identity['step'] != STEP:
        raise RuntimeError('Expected step {}, found {}'.format(STEP, identity['step']))
    if identity['checkpoint_id'] != CHECKPOINT_ID:
        raise RuntimeError('Unexpected checkpoint id: ' + identity['checkpoint_id'])
    return model, identity


def check_cell(name, spec, loaded):
    """Grid, sampler, real weights and a short real run; loaded keeps the model."""
    import sampling
    from run_lib import get_sde
    config = get_config(name)
    sde, eps = get_sde(config, n_override=spec['calls'])
    followup.check_cell_grid(config, sde, eps, spec)
    sampling.get_sampling_fn(config, sde, (BATCH, 3, 64, 64), lambda x: x, eps)
    if 'model' not in loaded:
        loaded['model'], loaded['identity'] = load_model(config)
    followup.smoke(config, loaded['model'], spec['method'], eps)
    return loaded['identity']


def preflight():
    """Returns 1 if a cell failed. Shared trouble raises and leaves no preflight.json."""
    import torch
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
    for name, spec in CELLS.items():
        try:
            identities[MODEL] = check_cell(name, spec, loaded)
            print('Preflight', name, identities[MODEL]['step'], flush=True)
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            failures[name] = repr(error)
            write_json(failure_path(name), dict(error=repr(error), attempt=0, time=time.time()))
            print('Preflight failed: {}: {!r}'.format(name, error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', checkpoints=identities, failures=failures,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight finished: {} of {} cells passed.'.format(len(CELLS) - len(failures), len(CELLS)), flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_fox800_lo_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_fox800_lo_eval.py:' + name, '--workdir=workdirs/' + WORK,
        '--eval_folder=' + eval_folder(name)], cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    import numpy as np
    from artifact_utils import _canonicalize
    spec = CELLS[name]
    config = get_config(name)
    directory = ROOT / 'workdirs' / WORK / eval_folder(name)
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
    nfe = spec['calls'] + 1
    variants = tuple(config.eval.endpoint_denoise_outputs)
    with np.load(str(directory / 'report_{}.npz'.format(CKPT)), allow_pickle=False) as data:
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=STEP, checkpoint_id=CHECKPOINT_ID,
                        dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in variants}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    rounds = directory / 'ckpt_{}'.format(CKPT)
    shards = len(list(rounds.glob('statistics_*.npz')))
    if shards != (spec['count'] + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    with np.load(str(rounds / 'statistics_0.npz'), allow_pickle=False) as data:
        if FEATURES not in data.files or data[FEATURES].shape != (BATCH, 2048):
            raise RuntimeError('Shards lack the bootstrap features.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=STEP, checkpoint_id=CHECKPOINT_ID,
        lambda_window=[LOGSNR_MIN, WINDOWS[spec['window']]], nfe=nfe, num_samples=spec['count'],
        sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('FOX 800k {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(failed=()):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    point = [dict(contrast=label, left=left, right=right, question=question,
                  value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
             for label, left, right, question in contrasts() if left in cells and right in cells]
    references = [dict(cell=name, reference=REFERENCE['value'], source=REFERENCE['source'],
                       value=value['fid_tweedie_xt'] - REFERENCE['value']) for name, value in cells.items()]
    status = 'completed' if len(cells) == len(CELLS) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), cells=list(cells.values()), point_contrasts=point,
        reference_differences=references, confidence_intervals_computed=False,
        scope='One training run; the reference has another window, 64 calls and 10k'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'step', 'lambda_max', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['checkpoint_step'], value['lambda_window'][1],
                             value['method'], value['sampling_seed'], value['num_samples'], value['nfe'],
                             value['fid_tweedie_xt'], value['fid_x_t'], value['kid_tweedie_xt'],
                             value['IS_tweedie_xt']))


def queue():
    """Runs the cells that passed preflight. A failed cell does not stop the next and gets a second pass."""
    path = LOGS / 'preflight.json'
    if not path.exists():
        print('No preflight.json: preflight stopped the queue.', flush=True)
        return 1
    failures = json.loads(path.read_text())['failures']
    blocked = [name for name in CELLS if name in failures and not result_path(name).exists()]
    failed = []
    for attempt in range(1, ATTEMPTS + 1):
        failed = []
        for name in CELLS:
            if result_path(name).exists() or name in blocked:
                continue
            try:
                run(name)
                report(name)
                if failure_path(name).exists():
                    failure_path(name).unlink()
            except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
                write_json(failure_path(name), dict(error=repr(error), attempt=attempt, time=time.time()))
                print('FOX 800k cell failed: {}: {!r}'.format(name, error), flush=True)
                failed.append(name)
        if not failed:
            break
    failed = blocked + failed
    write_summary(failed)
    print('FOX 800k queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(preflight())
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'plan':
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, cell_spec)
    else:
        raise SystemExit('Usage: celeba_fox800_lo_eval.py preflight|queue|plan')
