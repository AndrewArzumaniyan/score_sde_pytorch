"""CelebA on GPU 2: A at 400k under stage 0, then kappa=0.02 at 200k under EM.

Two cells in a fixed order, 10k images each. A at 400k continues A at 200k and 300k
of celeba_followup_eval.py. kappa=0.02 under EM repeats its a0_c8_hi_em cell on the
weights of kappa=0.02, so the two compare only as a 10k screen of two training runs.

EM returns x_mean, x_t and Tweedie and reports 1001 network calls.

A preflight failure of one cell, including a wrong step or a missing checkpoint, blocks
that cell and not the other. Inception comes from a local copy of the tfhub cache, so
this queue has no download step; without the copy, preflight stops both cells.

Run through run_celeba_gpu2_rtx.sh from a separate worktree.
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
import celeba_kappa_eval as kappa
import celeba_stage0_eval as stage0
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
BATCH = followup.BATCH
ATTEMPTS = 2
WORKDIRS = {'A': followup.WORKDIRS['A'], 'k002': kappa.WORKDIRS['k002']}
INCEPTION_MODULE = 'c4be6b55dc0338e82546082d22184ebdc77d76cf'  # SHA-1 of evaluation.INCEPTION_TFHUB.

# The order is fixed in advance and does not depend on any observed FID.
CELLS = {
    'A_c80_lo': dict(model='A', ckpt=80, step=400001, window='lo', protocol='stage0', method=stage0.METHOD,
                     seed=stage0.SEED, count=stage0.COUNT, calls=stage0.BUDGET),
    'k002_c8_hi_em': dict(model='k002', ckpt=8, step=200001, window='hi', protocol='em', method='pc', seed=0,
                          count=10000, calls=followup.EM_CALLS),
}

# Finished results of other queues, for reading only: different cells, not recomputed here.
REFERENCES = {
    'A_c80_lo': dict(value=2.6024169921875, source='follow-up, A_c60_lo, 300k'),
    'k002_c8_hi_em': dict(value=3.588902235031128, source='kappa row, k002_c8_hi, DPM++ with 65 calls'),
}


def em_config():
    config = kappa.get_config('k002_c8_hi')
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'
    sampling.noise_removal = True
    sampling.probability_flow = False
    config.eval.sampling_num_scales = followup.EM_CALLS
    config.eval.endpoint_denoise_outputs = followup.EM_OUTPUTS
    config.eval.save_endpoint_states = False
    return config


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown GPU 2 cell: ' + name)
    spec = CELLS[name]
    return followup.a_config(spec) if spec['protocol'] == 'stage0' else em_config()


def identity_key(spec):
    return '{}_{}'.format(spec['model'], spec['ckpt'])


def load_model(config, spec):
    # The two loaders differ only in the snapshot step they accept: 5k for A, 25k for kappa.
    loader = stage0.load_model if spec['model'] == 'A' else kappa.load_model
    model, identity = loader(config, WORKDIRS[spec['model']], spec['ckpt'])
    if identity['step'] != spec['step']:
        raise RuntimeError('Expected step {}, found {}'.format(spec['step'], identity['step']))
    return model, identity


def check_inception():
    module = Path(os.environ.get('TFHUB_CACHE_DIR', '')) / INCEPTION_MODULE
    if not (module / 'saved_model.pb').is_file() or not (module / 'variables').is_dir():
        raise RuntimeError('No local Inception at ' + str(module))


def check_cell(name, spec):
    """Grid, sampler and real weights of a cell; ODE cells also take a short real run."""
    import sampling
    from run_lib import get_sde
    config = get_config(name)
    sde, eps = get_sde(config, n_override=spec['calls'])
    followup.check_cell_grid(config, sde, eps, spec)
    sampling.get_sampling_fn(config, sde, (BATCH, 3, 64, 64), lambda x: x, eps)
    model, identity = load_model(config, spec)
    if spec['protocol'] != 'em':
        followup.smoke(config, model, spec['method'], eps)
    return identity


def preflight():
    """Returns 1 if a cell failed. Trouble shared by both cells raises and leaves no preflight.json."""
    import torch
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
    identities, failures = {}, {}
    for name, spec in CELLS.items():
        try:
            identities[identity_key(spec)] = check_cell(name, spec)
            print('Preflight', name, spec['step'], flush=True)
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            failures[name] = repr(error)
            write_json(failure_path(name), dict(error=repr(error), attempt=0, time=time.time()))
            print('Preflight failed: {}: {!r}'.format(name, error), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    write_json(path, dict(status='failed' if failures else 'passed', checkpoints=identities, failures=failures,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight finished: {} of {} cells passed.'.format(len(identities), len(CELLS)), flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_gpu2_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_gpu2_eval.py:' + name, '--workdir=workdirs/' + WORKDIRS[CELLS[name]['model']],
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
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=spec['step'],
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
    print('GPU 2 {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(failed=()):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    references = [dict(cell=name, reference=value['value'], source=value['source'],
                       value=cells[name]['fid_tweedie_xt'] - value['value'])
                  for name, value in REFERENCES.items() if name in cells]
    status = 'completed' if len(cells) == len(CELLS) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), cells=list(cells.values()), point_contrasts=[],
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
    """Runs the cells that passed preflight. A failed cell does not stop the other and gets a second pass."""
    path = LOGS / 'preflight.json'
    if not path.exists():
        print('No preflight.json: preflight stopped both cells.', flush=True)
        return 1
    blocked = [name for name in CELLS if name in json.loads(path.read_text())['failures']
               and not result_path(name).exists()]
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
                print('GPU 2 cell failed: {}: {!r}'.format(name, error), flush=True)
                failed.append(name)
        if not failed:
            break
    failed = blocked + failed
    write_summary(failed)
    print('GPU 2 queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(preflight())
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'plan':
        for cell_name, cell_spec in CELLS.items():
            print(cell_name, cell_spec)
    else:
        raise SystemExit('Usage: celeba_gpu2_eval.py preflight|queue|plan')
