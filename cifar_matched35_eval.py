"""CIFAR-10 under the CelebA matched protocol: 34 calls plus Tweedie, 50k images.

DPM++ 2M and EDM Heun, sampling seeds 0 and 1, extended window, on NCSN++ VP@600k,
FOX@600k and VP@400k, the best checked VP, and on DDPM++ VP and FOX at 600k. VP@400k
also gets the DPM++ 64-call seed 2 cell, to pair with the existing FOX@600k one.

The first cell repeats that FOX@600k cell on the same weights. It must reproduce
the old report, or the queue stops: this driver is new, and the NCSN++ and DDPM++
weights live in two repositories. Precision flags stay as in the old CIFAR cells.

Configs come from the manifests of the runs. Run through run_cifar_matched35_rtx.sh.
"""
import csv
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from celeba_stage0_eval import check_grid, set_device, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
TRAINING_INPUTS = ROOT / 'workdirs/_training_inputs'
VARIANTS = ('x_t', 'tweedie_xt')
COUNT = 50000
BATCH = 512
LOGSNR_MIN = -10.0
LOGSNR_MAX = 10.575749165617202
REFERENCE_SHA = 'c464743bce16cb0f64408eb36bdc7a7ef0e7b4dfd618c38f8119d4e29ef0c9d5'
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'

NCSN_VP = 'cifar10_vp_200k_rtx_seed42'
NCSN_FOX = 'cifar10_fox_matern32_kappa1000_200k_rtx_seed42'
DDPM_VP = 'cifar10_vp_ddpmpp_deep_200k_rtx_seed42'
DDPM_FOX = 'cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx_seed42'
# name -> workdir, checkpoint, state.step
MODELS = {'ncsn_vp400': (NCSN_VP, 16, 400001), 'ncsn_vp600': (NCSN_VP, 24, 600001),
          'ncsn_fox600': (NCSN_FOX, 24, 600001),
          'ddpm_vp600': (DDPM_VP, 12, 600001), 'ddpm_fox600': (DDPM_FOX, 12, 600001)}

GATE_CELL = 'ncsn_fox600_{}_n64_seed2'.format(DPM)
GATE_FID = 3.4352831840515137  # Report of the old cell, after Tweedie.
GATE_TOLERANCE = 0.02
GATE_EXIT = 3


def cell(model, method, calls, seed):
    return '{}_{}_n{}_seed{}'.format(model, method, calls, seed), dict(
        model=model, method=method, calls=calls, seed=seed)


def matched(models, seed):
    return [cell(model, method, 34, seed) for method in (DPM, HEUN) for model in models]


# The order is fixed in advance and does not depend on any observed FID.
NCSN = ('ncsn_vp600', 'ncsn_fox600', 'ncsn_vp400')
DDPM = ('ddpm_vp600', 'ddpm_fox600')
CELLS = dict([cell('ncsn_fox600', DPM, 64, 2)] + matched(NCSN, 0) + matched(NCSN, 1) +
             [cell('ncsn_vp400', DPM, 64, 2)] + matched(DDPM, 0) + matched(DDPM, 1))


def contrast(left, right, method, calls, seed):
    a, b = (cell(model, method, calls, seed)[0] for model in (left, right))
    return '{}_vs_{}_{}_n{}_seed{}'.format(left, right, method, calls, seed), a, b


# Each contrast is left minus right, FID after Tweedie, negative favours left.
CONTRASTS = tuple(contrast(left, right, method, 34, seed)
                  for left, right in (('ncsn_fox600', 'ncsn_vp600'), ('ncsn_fox600', 'ncsn_vp400'),
                                      ('ddpm_fox600', 'ddpm_vp600'))
                  for method in (DPM, HEUN) for seed in (0, 1)) + (
    contrast('ncsn_fox600', 'ncsn_vp400', DPM, 64, 2),)


def get_config(name):
    from ml_collections import ConfigDict
    from configs.vp.cifar10_ncsnpp_continuous import get_config as get_base
    if name not in CELLS:
        raise ValueError('Unknown CIFAR cell: ' + name)
    spec = CELLS[name]
    work, ckpt, _ = MODELS[spec['model']]
    config = ConfigDict(json.loads((TRAINING_INPUTS / (work + '.json')).read_text())['config'])
    config.eval = get_base().eval  # Manifests exclude evaluation settings.
    set_device(config)
    config.seed = 42
    config.allow_tf32 = False
    config.cudnn_benchmark = False
    config.allow_fp16_reduced_precision_reduction = False
    config.sampling.method = spec['method']
    config.sampling.probability_flow = True
    config.sampling.time_grid = 'karras' if spec['method'] == HEUN else 'uniform_logsnr'
    config.sampling.logsnr_min = LOGSNR_MIN
    config.sampling.logsnr_max = LOGSNR_MAX
    config.sampling.noise_removal = True
    config.sampling.predictor = 'none'  # The ODE method owns its stages.
    config.sampling.corrector = 'none'
    config.eval.endpoint_denoise_variants = True
    config.eval.endpoint_denoise_outputs = VARIANTS
    config.eval.save_endpoint_states = True
    config.eval.sampling_num_scales = spec['calls']  # Trajectory calls.
    config.eval.sampling_seed = spec['seed']
    config.eval.num_samples = COUNT
    config.eval.batch_size = BATCH
    config.eval.enable_loss = False
    config.eval.enable_bpd = False
    config.eval.enable_sampling = True
    config.eval.begin_ckpt = config.eval.end_ckpt = ckpt
    config.eval.include_final_checkpoint = False
    return config


def load_model(config, model_name):
    from artifact_utils import build_model_manifest
    from losses import get_optimizer
    from models import utils as mutils, ddpm, ncsnpp  # noqa: F401, registers the models.
    from models.ema import ExponentialMovingAverage
    from utils import restore_checkpoint
    work, ckpt, expected = MODELS[model_name]
    model = mutils.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, optimizer=get_optimizer(config, model.parameters()), ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(config)['model_protocol_sha256'])
    path = ROOT / 'workdirs' / work / 'checkpoints/checkpoint_{}.pth'.format(ckpt)
    if not path.is_file():
        raise RuntimeError('Missing checkpoint: ' + str(path))
    state = restore_checkpoint(str(path), state, config.device, restore_rng=False)
    if int(state['step']) != expected:
        raise RuntimeError('Unexpected step {} in {}'.format(state['step'], path))
    ema.copy_to(model.parameters())
    model.eval()
    return model, dict(checkpoint_id=state['checkpoint_id'], step=expected)


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    from artifact_utils import _environment_manifest, _source_fingerprint
    from ode_sampling import get_vp_ode_sampler
    from run_lib import get_sde
    identities = {}
    for name, spec in CELLS.items():
        config = get_config(name)
        sde, eps = get_sde(config, n_override=spec['calls'])
        if spec['method'] == DPM:
            check_grid(config, sde, eps, spec['calls'])
        key = '{}_{}'.format(spec['model'], spec['method'])
        if key in identities:
            continue
        model, identities[key] = load_model(config, spec['model'])
        smoke_sde, _ = get_sde(config, n_override=4)
        z = torch.zeros((2, 3, 32, 32), device=config.device)
        outputs, calls = get_vp_ode_sampler(smoke_sde, z.shape, lambda x: x, spec['method'], eps, config.device)(model, z=z)
        if calls != 5 or not all(bool(torch.isfinite(x).all()) for x in outputs.values()):
            raise RuntimeError('Failed real-weights smoke: ' + key)
        print('Preflight', key, identities[key]['step'], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(LOGS / 'preflight.json', dict(status='passed', checkpoints=identities,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight passed: {} models, {} cells.'.format(len(MODELS), len(CELLS)), flush=True)


def eval_folder(name):
    return 'eval_matched35_{}_{}'.format(name, COUNT)


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    work = MODELS[CELLS[name]['model']][0]
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=cifar_matched35_eval.py:' + name, '--workdir=workdirs/' + work,
        '--eval_folder=' + eval_folder(name)], cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    import numpy as np
    from artifact_utils import _canonicalize
    spec = CELLS[name]
    config = get_config(name)
    work, ckpt, _ = MODELS[spec['model']]
    directory = ROOT / 'workdirs' / work / eval_folder(name)
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
    identity = pre['checkpoints']['{}_{}'.format(spec['model'], spec['method'])]
    with np.load(str(directory / 'report_{}.npz'.format(ckpt)), allow_pickle=False) as data:
        expected = dict(num_samples=COUNT, nfe=spec['calls'] + 1, checkpoint_step=identity['step'],
                        checkpoint_id=identity['checkpoint_id'], dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in VARIANTS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    shards = len(list((directory / 'ckpt_{}'.format(ckpt)).glob('statistics_*.npz')))
    if shards != (COUNT + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=identity['step'], nfe=spec['calls'] + 1,
        lambda_window=[LOGSNR_MIN, LOGSNR_MAX], num_samples=COUNT,
        elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('CIFAR {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary():
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    contrasts = [dict(contrast=label, left=left, right=right,
                      value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
                 for label, left, right in CONTRASTS if left in cells and right in cells]
    write_json(LOGS / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=list(cells.values()),
        point_contrasts=contrasts, confidence_intervals_computed=False,
        scope='50k, one training run per model'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'method', 'nfe', 'seed', 'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['method'], value['nfe'], value['seed'],
                             value['fid_tweedie_xt'], value['fid_x_t'], value['kid_tweedie_xt'],
                             value['IS_tweedie_xt']))


def check_gate():
    """Stops the queue when the driver does not reproduce the old FOX@600k cell."""
    actual = json.loads(result_path(GATE_CELL).read_text())['fid_tweedie_xt']
    if abs(actual - GATE_FID) > GATE_TOLERANCE:
        print('Gate failed: {:.6f}, expected {:.6f}.'.format(actual, GATE_FID), flush=True)
        raise SystemExit(GATE_EXIT)
    print('Gate passed: {:.6f}, expected {:.6f}.'.format(actual, GATE_FID), flush=True)


def queue():
    failed = []
    for name in CELLS:
        if not result_path(name).exists():
            try:
                run(name)
                report(name)
                if failure_path(name).exists():
                    failure_path(name).unlink()
            except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
                write_json(failure_path(name), dict(error=repr(error), time=time.time()))
                print('CIFAR cell failed: {}: {!r}'.format(name, error), flush=True)
                if name == GATE_CELL:
                    raise SystemExit(GATE_EXIT)
                failed.append(name)
                continue
        if name == GATE_CELL:
            check_gate()
    print('CIFAR queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        preflight()
    elif mode == 'queue':
        sys.exit(queue())
    else:
        raise SystemExit('Usage: cifar_matched35_eval.py preflight|queue')
