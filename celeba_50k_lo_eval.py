"""CelebA at 50k on the lower window: A at 400k, VP at 200k, FOX at 1.3M.

Twelve cells: matched35 (34 trajectory calls plus Tweedie, DPM++ and EDM Heun, seeds 0
and 1), the protocol of vp_c40_hi_* in celeba_followup_eval.py except the window, which
is [-10; 9.115429865459795] for every model. Shards and states are saved as in matched35,
so the features can go to a bootstrap. One training run per model.

Two independent lanes share no files: gpu1 runs A, then VP; gpu2 runs FOX. Each lane has
its own workdirs and logs, so its summary covers its own cells only. Contrasts with FOX
therefore come from reading both summaries.

A preflight failure of a model, including a wrong step or checkpoint id, blocks its cells
and not the others. Inception comes from a local copy of the tfhub cache; without the copy,
preflight stops the lane.

Run through run_celeba_50k_lo_rtx.sh from a separate worktree.
"""
import csv
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import celeba_followup_eval as followup
import celeba_matched35_rtx_eval as rtx
import celeba_stage0_eval as stage0
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
BATCH = followup.BATCH
CALLS = followup.MATCHED_CALLS
COUNT = 50000
WINDOW = 'lo'
ATTEMPTS = 2
METHODS = ('ode_dpmpp_2m', 'ode_edm_heun')
INCEPTION_MODULE = 'c4be6b55dc0338e82546082d22184ebdc77d76cf'  # SHA-1 of evaluation.INCEPTION_TFHUB.
FEATURES = 'pool_3_tweedie_xt'  # What celeba_matched35_bootstrap.py reads from each shard.

MODELS = {
    'A': dict(work=followup.WORKDIRS['A'], ckpt=80, step=400001, checkpoint_id='7de461feeceb4a488583921d9f7f132d'),
    'vp': dict(work=stage0.VP, ckpt=40, step=200001,
               checkpoint_id='legacy-sha256-922cc63cb505177fe0d86ca919d39bc0b68a221706c80b75b14bd7910f352073'),
    'fox': dict(work=stage0.FOX, ckpt=260, step=1300002,
                checkpoint_id='legacy-sha256-86ab70f2ce55dde46453eca08073ade220693e866f722e0e64fe5d1ce0e7f7e2'),
}
LANES = {'gpu1': ('A', 'vp'), 'gpu2': ('fox',)}


def cell_name(model, method, seed):
    return '{}_c{}_lo_{}_seed{}'.format(model, MODELS[model]['ckpt'], method, seed)


# The order is fixed in advance and does not depend on any observed FID.
CELLS = {cell_name(model, method, seed): dict(model=model, ckpt=MODELS[model]['ckpt'], window=WINDOW,
                                              protocol='matched35', method=method, seed=seed, count=COUNT,
                                              calls=CALLS)
         for model in MODELS for seed in (0, 1) for method in METHODS}

# Finished 10k results with DPM++ and 65 calls, for reading only: another protocol, not recomputed here.
REFERENCES = {
    'A': dict(value=2.57734751701355, source='GPU 2 queue, A_c80_lo, DPM++ with 65 calls, 10k'),
    'vp': dict(value=3.3268163204193115, source='stage 0, vp_c40_lo, DPM++ with 65 calls, 10k'),
    'fox': dict(value=2.4948720932006836, source='stage 0, fox_c260_lo, DPM++ with 65 calls, 10k'),
}


def lane_cells(lane):
    if lane not in LANES:
        raise ValueError('Unknown lane: ' + lane)
    return [name for name, spec in CELLS.items() if spec['model'] in LANES[lane]]


def contrasts(lane):
    """Left minus right, FID after Tweedie, inside one lane."""
    result = []
    for model in LANES[lane]:
        for seed in (0, 1):
            result.append(('{}_dpmpp_minus_heun_seed{}'.format(model, seed), cell_name(model, METHODS[0], seed),
                           cell_name(model, METHODS[1], seed), 'Solver, same seed'))
        for method in METHODS:
            result.append(('{}_{}_seed1_minus_seed0'.format(model, method), cell_name(model, method, 1),
                           cell_name(model, method, 0), 'Sampling seed, same solver'))
    if lane == 'gpu1':
        for seed in (0, 1):
            for method in METHODS:
                result.append(('A_minus_vp_{}_seed{}'.format(method, seed), cell_name('A', method, seed),
                               cell_name('vp', method, seed), 'A at 400k against VP at 200k, unequal training'))
    return result


def a_config(spec):
    config = followup.a_config(spec)  # Stage 0 protocol on the trained A.
    config.sampling.method = spec['method']
    config.sampling.time_grid = 'karras' if spec['method'] == 'ode_edm_heun' else 'uniform_logsnr'
    config.eval.sampling_num_scales = spec['calls']
    config.eval.sampling_seed = spec['seed']
    config.eval.num_samples = spec['count']
    return config


def native_config(spec):
    config = rtx.config_for((spec['calls'], spec['count'], spec['model'], spec['method'], spec['seed']))
    config.eval.begin_ckpt = config.eval.end_ckpt = spec['ckpt']
    config.sampling.logsnr_max = WINDOWS[spec['window']]
    return config


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown 50k cell: ' + name)
    spec = CELLS[name]
    return a_config(spec) if spec['model'] == 'A' else native_config(spec)


def load_model(config, spec):
    expected = MODELS[spec['model']]
    model, identity = stage0.load_model(config, expected['work'], expected['ckpt'])
    if identity['step'] != expected['step']:
        raise RuntimeError('Expected step {}, found {}'.format(expected['step'], identity['step']))
    if identity['checkpoint_id'] != expected['checkpoint_id']:
        raise RuntimeError('Unexpected checkpoint id: ' + identity['checkpoint_id'])
    return model, identity


def check_inception():
    module = Path(os.environ.get('TFHUB_CACHE_DIR', '')) / INCEPTION_MODULE
    if not (module / 'saved_model.pb').is_file() or not (module / 'variables').is_dir():
        raise RuntimeError('No local Inception at ' + str(module))


def check_cell(name, spec, loaded):
    """Grid, sampler, real weights and a short real run; loaded keeps the last model."""
    import sampling
    import torch
    from run_lib import get_sde
    config = get_config(name)
    sde, eps = get_sde(config, n_override=spec['calls'])
    followup.check_cell_grid(config, sde, eps, spec)
    sampling.get_sampling_fn(config, sde, (BATCH, 3, 64, 64), lambda x: x, eps)
    if loaded.get('key') != spec['model']:
        loaded.clear()
        gc.collect()
        torch.cuda.empty_cache()
        loaded['model'], loaded['identity'] = load_model(config, spec)
        loaded['key'] = spec['model']
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
    check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    identities, failures, loaded = {}, {}, {}
    for name in names:
        spec = CELLS[name]
        try:
            identities[spec['model']] = check_cell(name, spec, loaded)
            print('Preflight', name, identities[spec['model']]['step'], flush=True)
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
    return 'eval_50k_lo_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_50k_lo_eval.py:' + name, '--workdir=workdirs/' + MODELS[CELLS[name]['model']]['work'],
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
    nfe = spec['calls'] + 1
    variants = tuple(config.eval.endpoint_denoise_outputs)
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=expected_model['step'],
                        checkpoint_id=expected_model['checkpoint_id'], dataset_stats_id=REFERENCE_SHA)
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
        checkpoint_id=expected_model['checkpoint_id'], lambda_window=[LOGSNR_MIN, WINDOWS[spec['window']]], nfe=nfe,
        num_samples=spec['count'], sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary(lane)
    print('50k lo {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(lane, failed=()):
    names = lane_cells(lane)
    cells = {name: json.loads(result_path(name).read_text()) for name in names if result_path(name).exists()}
    point = [dict(contrast=label, left=left, right=right, question=question,
                  value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
             for label, left, right, question in contrasts(lane) if left in cells and right in cells]
    references = [dict(cell=name, reference=REFERENCES[value['model']]['value'],
                       source=REFERENCES[value['model']]['source'],
                       value=value['fid_tweedie_xt'] - REFERENCES[value['model']]['value'])
                  for name, value in cells.items()]
    status = 'completed' if len(cells) == len(names) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, lane=lane, completed_cells=len(cells),
        total_cells=len(names), failed_cells=list(failed), cells=list(cells.values()), point_contrasts=point,
        reference_differences=references, confidence_intervals_computed=False,
        scope='One training run per model; 34 calls at 50k against 64 calls at 10k in the references'))
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
                print('50k lo cell failed: {}: {!r}'.format(name, error), flush=True)
                failed.append(name)
        if not failed:
            break
    failed = blocked + failed
    write_summary(lane, failed)
    print('50k lo queue {} finished, failed cells: {}'.format(lane, failed or 'none'), flush=True)
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
        raise SystemExit('Usage: celeba_50k_lo_eval.py preflight LANE|queue LANE|plan')
