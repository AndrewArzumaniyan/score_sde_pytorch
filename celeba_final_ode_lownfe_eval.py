"""Final VP/native FOX, Heun and DPM++ 2M, 10k, sampling seeds 0 and 1.

Same weights, extended endpoints, prior, Tweedie and reference as the completed
final ODE queue (celeba_final_ode_eval.py); only budget and sampling seed change.
The four existing 256-call 10k seed-0 points complete the two-seed curves.
They are external results, not counted as completed cells in this queue.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import celeba_final_ode_eval as final
from celeba_final_ode_eval import ARMS, OUTPUT, ROOT, VARIANTS, REFERENCE_SHA, write_json

METHODS = ('ode_dpmpp_2m', 'ode_heun')
BUDGETS = (16, 32, 64, 128, 256)
SEEDS = (0, 1)
COUNT = 10000
# Ascending budgets; finish both arms/methods/seeds before increasing NFE.
# This order and the omission of the completed seed-0 anchor are fixed in advance.
CELLS = [(n, COUNT, arm, method, seed) for n in BUDGETS for seed in SEEDS
         for method in METHODS for arm in ('vp', 'fox') if (n, seed) != (256, 0)]
EXTERNAL_CELLS = [(256, COUNT, arm, method, 0)
                  for method in METHODS for arm in ('vp', 'fox')]


def config_name(n, count, arm, method, seed):
    return '{}_{}_{}_{}_seed{}'.format(method, n, count, arm, seed)


def get_config(name):
    method, n, count, arm, seed_token = name.rsplit('_', 4)
    n, count = int(n), int(count)
    if seed_token not in ('seed0', 'seed1'):
        raise ValueError('Unknown sampling seed')
    seed = int(seed_token[4:])
    if (n, count, arm, method, seed) not in CELLS:
        raise ValueError('Unknown low-NFE final ODE cell')
    cfg = final.get_config('{}_{}_{}'.format(method, count, arm))
    cfg.eval.sampling_num_scales = n
    cfg.eval.sampling_seed = seed
    return cfg


def snapshot(source, output):
    final.snapshot(source, output)
    path = Path(output) / 'snapshot.json'
    saved = json.loads(path.read_text())
    saved.pop('sampling_seed', None)
    saved.update(profile='final_ode_lownfe', cells=CELLS, sampling_seeds=SEEDS,
                 methods=METHODS, budgets=BUDGETS, num_samples=COUNT,
                 external_cells=EXTERNAL_CELLS, external_cells_regenerated=False,
                 cell_fields=['trajectory_nfe', 'num_samples', 'arm', 'method', 'sampling_seed'],
                 estimated_gpu_hours=[8, 12], estimate_is_guarantee=False)
    write_json(path, saved)


def folder(n, count, arm, method, seed):
    return 'eval_final_ode_lownfe_{}_nfe{}_{}_{}_seed{}'.format(method, n, arm, count, seed)


def run(n, count, arm, method, seed):
    started, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_final_ode_lownfe_eval.py:' + config_name(n, count, arm, method, seed),
        '--workdir=workdirs/' + ARMS[arm], '--eval_folder=' + folder(n, count, arm, method, seed)])
    write_json(OUTPUT / 'logs' / (config_name(n, count, arm, method, seed) + '_timing.json'),
               dict(started_unix=started, finished_unix=time.time(),
                    elapsed_seconds=time.monotonic() - clock, returncode=result.returncode,
                    scope='main.py evaluation, including generation and metrics'))
    result.check_returncode()


def report(n, count, arm, method, seed):
    import numpy as np
    from artifact_utils import _canonicalize
    directory = ROOT / 'workdirs' / ARMS[arm] / folder(n, count, arm, method, seed)
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = get_config(config_name(n, count, arm, method, seed))
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(cfg[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Unexpected report source')
    with np.load(str(directory / 'report_260.npz')) as result:
        expected = dict(num_samples=count, nfe=n + 1, checkpoint_step=1300002,
                        checkpoint_id=pre['checkpoint_ids'][arm], dataset_stats_id=REFERENCE_SHA)
        if any(result[k].item() != v for k, v in expected.items()):
            raise RuntimeError('Unexpected report identity')
        metrics = {metric + '_' + variant: float(result[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in VARIANTS}
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError('Non-finite metric')
    if len(list((directory / 'ckpt_260').glob('statistics_*.npz'))) != (count + 511) // 512:
        raise RuntimeError('Unexpected number of statistics shards')
    timing = json.loads((OUTPUT / 'logs' / (config_name(n, count, arm, method, seed) + '_timing.json')).read_text())
    if timing['returncode'] != 0 or not math.isfinite(timing['elapsed_seconds']) or timing['elapsed_seconds'] < 0:
        raise RuntimeError('Invalid evaluation timing/status')
    write_json(OUTPUT / 'logs' / (config_name(n, count, arm, method, seed) + '_result.json'),
               dict(trajectory_nfe=n, nfe=n + 1, num_samples=count, arm=arm, method=method,
                    sampling_seed=seed, timing=timing, **metrics))
    cells = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    identities = [(c['trajectory_nfe'], c['num_samples'], c['arm'], c['method'], c['sampling_seed']) for c in cells]
    if len(set(identities)) != len(cells) or not set(identities).issubset(CELLS):
        raise RuntimeError('Unexpected or duplicate summary cell')
    write_json(OUTPUT / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=cells,
        external_cells=EXTERNAL_CELLS, external_cells_validated=False,
        evaluation_elapsed_seconds=sum(c['timing']['elapsed_seconds'] for c in cells)))
    print('Validated {}: FID Tweedie={:.6f}; {:.1f} min; {}/{} cells complete.'.format(
        config_name(n, count, arm, method, seed), metrics['fid_tweedie_xt'],
        timing['elapsed_seconds'] / 60, len(cells), len(CELLS)), flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot': snapshot(*args)
    elif mode == 'check': final.check()
    elif mode == 'preflight': final.preflight()
    elif mode == 'plan':
        for cell in CELLS: print(*cell)
    elif mode in ('run', 'report'):
        n, count, arm, method, seed = args
        cell = (int(n), int(count), arm, method, int(seed))
        if cell not in CELLS: raise ValueError('Unknown cell')
        (run if mode == 'run' else report)(*cell)
    else: raise ValueError(mode)
