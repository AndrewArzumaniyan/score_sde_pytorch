"""CelebA row of kappa: a0 (kappa=1000) and kappa=0.02 under the stage 0 protocol.

DPM++ 2M, 64 trajectory calls plus Tweedie, 10k images, sampling seed 0, on the
numeric windows of stage 0. One training run per cell, so this is a screen. a0 is
not native FOX (drift -5.025, 8192 nodes): compare it only inside this row.

The a0 cell at 100k comes first: if a0 behaves like native FOX, 200k sits in the
FOX dip, and the 200k cells show how deep it is, not how good the schedule is.

Configs come from the manifests of the runs, because kappa=0.02 was set by an
override. Run through run_celeba_kappa_eval_rtx.sh from a separate worktree.
"""
import csv
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from celeba_stage0_eval import (BATCH, BUDGET, COUNT, LOGSNR_MIN, METHOD, REFERENCE_SHA, SEED, VARIANTS,
                                WINDOWS, check_grid, set_protocol, write_json)

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
TRAINING_INPUTS = ROOT / 'workdirs/_training_inputs'
SNAPSHOT_FREQ = 25000  # Checkpoint 4 is 100k, 8 is 200k.
WORKDIRS = {'a0': 'celeba_kaffine_a0_200k_seed42', 'k002': 'celeba_kaffine_kappa0p02_200k_seed42'}
LENGTH_SCALES = {'a0': 346.4101615138, 'k002': 0.00692820323}


def cell(model, ckpt, window):
    return '{}_c{}_{}'.format(model, ckpt, window), dict(model=model, ckpt=ckpt, window=window)


# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict((cell('a0', 4, 'hi'), cell('a0', 8, 'hi'), cell('k002', 8, 'hi'),
              cell('a0', 8, 'lo'), cell('k002', 8, 'lo'), cell('k002', 4, 'hi')))

# Each contrast is left minus right, FID after Tweedie, negative favours left.
CONTRASTS = (
    ('a0_200k_vs_100k_hi', 'a0_c8_hi', 'a0_c4_hi', 'Does a0 fall into the FOX dip at 200k?'),
    ('k002_vs_a0_200k_hi', 'k002_c8_hi', 'a0_c8_hi', 'Kappa at 200k, wide window'),
    ('k002_vs_a0_200k_lo', 'k002_c8_lo', 'a0_c8_lo', 'Kappa at 200k, without the cleanest band'),
    ('k002_vs_a0_100k_hi', 'k002_c4_hi', 'a0_c4_hi', 'Kappa at 100k, before the dip'),
)


def manifest_path(model):
    return TRAINING_INPUTS / (WORKDIRS[model] + '.json')


def get_config(name):
    from ml_collections import ConfigDict
    from configs.fox.celeba_kaffine import get_config as get_kaffine
    if name not in CELLS:
        raise ValueError('Unknown kappa cell: ' + name)
    spec = CELLS[name]
    config = ConfigDict(json.loads(manifest_path(spec['model']).read_text())['config'])
    if config.model.fox_matern_length_scale != LENGTH_SCALES[spec['model']]:
        raise ValueError('Unexpected run behind ' + spec['model'])
    config.eval = get_kaffine('a0').eval  # Manifests exclude evaluation settings.
    set_protocol(config, spec['window'], spec['ckpt'])
    return config


def load_model(config, work, ckpt):
    from artifact_utils import build_model_manifest
    from losses import get_optimizer
    from models import utils as mutils, ncsnpp  # noqa: F401, registers the model.
    from models.ema import ExponentialMovingAverage
    from utils import restore_checkpoint
    model = mutils.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, optimizer=get_optimizer(config, model.parameters()), ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(config)['model_protocol_sha256'])
    path = ROOT / 'workdirs' / work / 'checkpoints/checkpoint_{}.pth'.format(ckpt)
    if not path.is_file():
        raise RuntimeError('Missing checkpoint: ' + str(path))
    state = restore_checkpoint(str(path), state, config.device, restore_rng=False)
    step = int(state['step'])
    if not SNAPSHOT_FREQ * ckpt <= step <= SNAPSHOT_FREQ * ckpt + 2:
        raise RuntimeError('Unexpected step {} in {}'.format(step, path))
    ema.copy_to(model.parameters())
    model.eval()
    return model, dict(checkpoint_id=state['checkpoint_id'], step=step)


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    from artifact_utils import _environment_manifest, _source_fingerprint
    from ode_sampling import get_vp_ode_sampler
    from run_lib import get_sde
    identities = {}
    for name, spec in CELLS.items():
        key = '{}_{}'.format(spec['model'], spec['ckpt'])
        config = get_config(name)
        sde, eps = get_sde(config, n_override=BUDGET)
        check_grid(config, sde, eps)
        if key in identities:
            continue
        model, identities[key] = load_model(config, WORKDIRS[spec['model']], spec['ckpt'])
        smoke_sde, _ = get_sde(config, n_override=4)
        z = torch.zeros((2, 3, 64, 64), device=config.device)
        outputs, calls = get_vp_ode_sampler(smoke_sde, z.shape, lambda x: x, METHOD, eps, config.device)(model, z=z)
        if calls != 5 or not all(bool(torch.isfinite(x).all()) for x in outputs.values()):
            raise RuntimeError('Failed real-weights smoke: ' + key)
        print('Preflight', key, identities[key]['step'], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(LOGS / 'preflight.json', dict(status='passed', checkpoints=identities,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight passed: {} checkpoints, {} grids.'.format(len(identities), len(CELLS)), flush=True)


def eval_folder(name):
    return 'eval_kappa_{}_n{}_{}_seed{}'.format(name, BUDGET, COUNT, SEED)


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_kappa_eval.py:' + name, '--workdir=workdirs/' + WORKDIRS[CELLS[name]['model']],
        '--eval_folder=' + eval_folder(name)], cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    import numpy as np
    from artifact_utils import _canonicalize
    spec = CELLS[name]
    config = get_config(name)
    directory = ROOT / 'workdirs' / WORKDIRS[spec['model']] / eval_folder(name)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(config[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads((LOGS / 'preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Report comes from different sources.')
    identity = pre['checkpoints']['{}_{}'.format(spec['model'], spec['ckpt'])]
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=COUNT, nfe=BUDGET + 1, checkpoint_step=identity['step'],
                        checkpoint_id=identity['checkpoint_id'], dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in VARIANTS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    shards = len(list((directory / 'ckpt_{}'.format(spec['ckpt'])).glob('statistics_*.npz')))
    if shards != (COUNT + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=identity['step'],
        lambda_window=[LOGSNR_MIN, WINDOWS[spec['window']]], nfe=BUDGET + 1, num_samples=COUNT,
        sampling_seed=SEED, elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('Kappa {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary():
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    contrasts = [dict(contrast=label, left=left, right=right, question=question,
                      value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
                 for label, left, right, question in CONTRASTS if left in cells and right in cells]
    write_json(LOGS / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=list(cells.values()),
        point_contrasts=contrasts, confidence_intervals_computed=False,
        scope='10k screen, sampling seed 0, one training run per cell'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'step', 'lambda_max', 'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['checkpoint_step'], value['lambda_window'][1],
                             value['fid_tweedie_xt'], value['fid_x_t'], value['kid_tweedie_xt'],
                             value['IS_tweedie_xt']))


def queue():
    failed = []
    for name in CELLS:
        if result_path(name).exists():
            continue
        try:
            run(name)
            report(name)
            if failure_path(name).exists():
                failure_path(name).unlink()
        except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
            write_json(failure_path(name), dict(error=repr(error), time=time.time()))
            print('Kappa cell failed: {}: {!r}'.format(name, error), flush=True)
            failed.append(name)
    print('Kappa queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        preflight()
    elif mode == 'queue':
        sys.exit(queue())
    else:
        raise SystemExit('Usage: celeba_kappa_eval.py preflight|queue')
