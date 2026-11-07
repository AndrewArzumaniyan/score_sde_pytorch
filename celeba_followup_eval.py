"""CelebA follow-up: A at 300k, native VP at 200k under matched35, a0 at 200k under EM.

Six cells in a fixed order. A at 300k repeats the stage 0 protocol and is compared
with A at 200k; the continuation restarted its RNG and data position at 200k, so
that step is not bitwise and the weights come from RTX, not H200. Native VP at 200k
repeats matched35 with checkpoint 40 in place of 260. a0 under EM asks whether the
FOX dip of the ODE at 200k is there under the sampler where FOX won.

EM returns x_mean, x_t and Tweedie and evaluate_checkpoint requires all three, so
that cell cannot be limited to two outputs. It reports 1001 network calls.

Run through run_celeba_followup_rtx.sh from a separate worktree.
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

import celeba_kappa_eval as kappa
import celeba_stage0_eval as stage0
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, check_grid, set_protocol, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
TRAINING_INPUTS = ROOT / 'workdirs/_training_inputs'
BATCH = 512
EM_CALLS = 1000
MATCHED_CALLS = 34
EM_OUTPUTS = ('x_mean', 'x_t', 'tweedie_xt')
WORKDIRS = {'A': 'celeba_factorial_A_vp_pvp_400k_rtx_seed42', 'vp': stage0.VP, 'a0': kappa.WORKDIRS['a0']}
ATTEMPTS = 2
WARM_ATTEMPTS = 6
WARM_TIMEOUT_SECONDS = 1200  # Above the 10 minutes tfhub waits for a stale download lock.
WARM_PAUSE_SECONDS = 60


def matched(method, seed):
    return ('vp_c40_hi_{}_seed{}'.format(method, seed),
            dict(model='vp', ckpt=40, window='hi', protocol='matched35', method=method, seed=seed,
                 count=50000, calls=MATCHED_CALLS))


# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict([('A_c60_lo', dict(model='A', ckpt=60, window='lo', protocol='stage0', method=stage0.METHOD,
                                seed=stage0.SEED, count=stage0.COUNT, calls=stage0.BUDGET))] +
             [matched(method, seed) for seed in (0, 1) for method in ('ode_dpmpp_2m', 'ode_edm_heun')] +
             [('a0_c8_hi_em', dict(model='a0', ckpt=8, window='hi', protocol='em', method='pc', seed=0,
                                   count=10000, calls=EM_CALLS))])

# Each contrast is left minus right, FID after Tweedie.
CONTRASTS = tuple(
    ('vp200k_dpmpp_minus_heun_seed{}'.format(seed), 'vp_c40_hi_ode_dpmpp_2m_seed{}'.format(seed),
     'vp_c40_hi_ode_edm_heun_seed{}'.format(seed), 'Solver at 200k, same seed') for seed in (0, 1)
) + tuple(
    ('vp200k_{}_seed1_minus_seed0'.format(method), 'vp_c40_hi_{}_seed1'.format(method),
     'vp_c40_hi_{}_seed0'.format(method), 'Sampling seed, same solver') for method in ('ode_dpmpp_2m', 'ode_edm_heun'))

# Finished results of other queues, for reading only: different cells, not recomputed here.
REFERENCES = {
    'A_c60_lo': dict(value=2.7945523262023926, source='stage 0, A_c40_lo, 200k on H200 weights'),
    'a0_c8_hi_em': dict(value=10.191018104553223, source='kappa row, a0_c8_hi, DPM++ with 65 calls'),
}


def manifest_path(model):
    return TRAINING_INPUTS / (WORKDIRS[model] + '.json')


def a_config(spec):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    config = ConfigDict(json.loads(manifest_path('A').read_text())['config'])
    if config.training.factorial_arm != 'A_vp_pvp' or config.seed != 42:
        raise ValueError('Unexpected run behind A')
    config.eval = get_a().eval  # Manifests exclude evaluation settings.
    set_protocol(config, spec['window'], spec['ckpt'])
    return config


def vp_config(spec):
    import celeba_matched35_rtx_eval as rtx
    config = rtx.config_for((spec['calls'], spec['count'], 'vp', spec['method'], spec['seed']))
    config.eval.begin_ckpt = config.eval.end_ckpt = spec['ckpt']
    return config


def em_config():
    config = kappa.get_config('a0_c8_hi')
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'
    sampling.noise_removal = True
    sampling.probability_flow = False
    config.eval.sampling_num_scales = EM_CALLS
    config.eval.endpoint_denoise_outputs = EM_OUTPUTS
    config.eval.save_endpoint_states = False
    return config


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown follow-up cell: ' + name)
    spec = CELLS[name]
    if spec['protocol'] == 'stage0':
        return a_config(spec)
    return vp_config(spec) if spec['protocol'] == 'matched35' else em_config()


def identity_key(spec):
    return '{}_{}'.format(spec['model'], spec['ckpt'])


def load_model(config, spec):
    # The two loaders differ only in the snapshot step they accept: 25k for a0, 5k otherwise.
    loader = kappa.load_model if spec['model'] == 'a0' else stage0.load_model
    return loader(config, WORKDIRS[spec['model']], spec['ckpt'])


def check_cell_grid(config, sde, eps, spec):
    import torch
    from ode_sampling import karras_time_grid
    if spec['method'] != 'ode_edm_heun':
        check_grid(config, sde, eps, spec['calls'])
        return
    grid = karras_time_grid(sde, eps, spec['calls'] // 2 + 1, config.device)
    levels = sde.log_snr(grid)
    if (not bool(torch.all(levels[1:] > levels[:-1])) or abs(float(levels[0]) - LOGSNR_MIN) > 1e-3 or
            abs(float(levels[-1]) - float(config.sampling.logsnr_max)) > 1e-3):
        raise RuntimeError('The Karras grid misses the requested window.')


def smoke(config, model, method, eps):
    import torch
    from ode_sampling import get_vp_ode_sampler
    from run_lib import get_sde
    sde, _ = get_sde(config, n_override=4)
    z = torch.zeros((2, 3, 64, 64), device=config.device)
    outputs, calls = get_vp_ode_sampler(sde, z.shape, lambda x: x, method, eps, config.device)(model, z=z)
    if calls != 5 or not all(bool(torch.isfinite(x).all()) for x in outputs.values()):
        raise RuntimeError('Failed real-weights smoke: ' + method)


def preflight():
    """Grids and samplers of all cells, real weights; ODE cells also take a short real run."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    import sampling
    from artifact_utils import _environment_manifest, _source_fingerprint
    from run_lib import get_sde
    identities, loaded_key, model = {}, None, None
    for name, spec in CELLS.items():
        config = get_config(name)
        sde, eps = get_sde(config, n_override=spec['calls'])
        check_cell_grid(config, sde, eps, spec)
        sampling.get_sampling_fn(config, sde, (BATCH, 3, 64, 64), lambda x: x, eps)
        key = identity_key(spec)
        if key != loaded_key:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model, identities[key] = load_model(config, spec)
            loaded_key = key
        if spec['protocol'] != 'em':
            smoke(config, model, spec['method'], eps)
        print('Preflight', name, identities[key]['step'], flush=True)
    write_json(LOGS / 'preflight.json', dict(status='passed', checkpoints=identities,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight passed: {} checkpoints, {} cells.'.format(len(identities), len(CELLS)), flush=True)


def warm_inception():
    """Fills the tfhub cache without a GPU: on 6 November a cell died on a download timeout."""
    command = [sys.executable, '-c', 'import evaluation; evaluation.get_inception_model()']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for attempt in range(1, WARM_ATTEMPTS + 1):
        try:
            if subprocess.run(command, cwd=str(ROOT), env=environment, timeout=WARM_TIMEOUT_SECONDS).returncode == 0:
                print('Inception cache is ready, attempt {}.'.format(attempt), flush=True)
                return True
        except subprocess.TimeoutExpired:
            print('Inception warm-up timed out, attempt {}.'.format(attempt), flush=True)
        if attempt < WARM_ATTEMPTS:
            time.sleep(WARM_PAUSE_SECONDS * attempt)
    print('Inception warm-up failed; the cells will try on their own.', flush=True)
    return False


def eval_folder(name):
    return 'eval_followup_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_followup_eval.py:' + name, '--workdir=workdirs/' + WORKDIRS[CELLS[name]['model']],
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
    for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
        if manifest['config'].get(key) != config[key]:
            raise RuntimeError('Unexpected precision setting: ' + key)
    pre = json.loads((LOGS / 'preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Report comes from different sources.')
    identity = pre['checkpoints'][identity_key(spec)]
    nfe = spec['calls'] + 1
    variants = tuple(config.eval.endpoint_denoise_outputs)
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=identity['step'],
                        checkpoint_id=identity['checkpoint_id'], dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in variants}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    shards = len(list((directory / 'ckpt_{}'.format(spec['ckpt'])).glob('statistics_*.npz')))
    if shards != (spec['count'] + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=identity['step'],
        checkpoint_id=identity['checkpoint_id'], lambda_window=[LOGSNR_MIN, WINDOWS[spec['window']]], nfe=nfe,
        num_samples=spec['count'], sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('Follow-up {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary():
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    contrasts = [dict(contrast=label, left=left, right=right, question=question,
                      value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
                 for label, left, right, question in CONTRASTS if left in cells and right in cells]
    references = [dict(cell=name, reference=value['value'], source=value['source'],
                       value=cells[name]['fid_tweedie_xt'] - value['value'])
                  for name, value in REFERENCES.items() if name in cells]
    write_json(LOGS / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=list(cells.values()), point_contrasts=contrasts,
        reference_differences=references, confidence_intervals_computed=False,
        scope='One training run per cell; 10k cells are a screen'))
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
    """A failed cell does not stop the others; the last pass retries it, reusing its finished rounds."""
    failed = []
    for attempt in range(1, ATTEMPTS + 1):
        failed = []
        pending = [name for name in CELLS if not result_path(name).exists()]
        if pending:
            warm_inception()
        for name in pending:
            try:
                run(name)
                report(name)
                if failure_path(name).exists():
                    failure_path(name).unlink()
            except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
                write_json(failure_path(name), dict(error=repr(error), attempt=attempt, time=time.time()))
                print('Follow-up cell failed: {}: {!r}'.format(name, error), flush=True)
                failed.append(name)
        if not failed:
            break
    print('Follow-up queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        preflight()
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'plan':
        for cell_name, cell_spec in CELLS.items():
            print(cell_name, cell_spec)
    else:
        raise SystemExit('Usage: celeba_followup_eval.py preflight|queue|plan')
