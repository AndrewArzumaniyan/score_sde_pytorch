"""Six new native VP/FOX cells; reuse two validated EDM seed-0 anchors.

Extended window, 50k, 34 trajectory calls plus Tweedie. No sampler changes.
"""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import celeba_final_ode_eval as final
from celeba_final_ode_eval import ARMS, ROOT, OUTPUT, VARIANTS, REFERENCE_SHA, write_json
from e3b_final50k import digest

ANCHORS = Path('/inputs/edm35')
CELLS = [(34, 50000, arm, method, seed) for seed in (0, 1)
         for method in (('ode_dpmpp_2m',) if seed == 0 else ('ode_dpmpp_2m', 'ode_edm_heun'))
         for arm in ('vp', 'fox')]
EXTERNAL_CELLS = [(34, 50000, arm, 'ode_edm_heun', 0) for arm in ('vp', 'fox')]


def config_name(n, count, arm, method, seed):
    return '{}_{}_{}_{}_seed{}'.format(method, n, count, arm, seed)


def config_for(cell):
    n, count, arm, method, seed = cell
    if cell not in CELLS + EXTERNAL_CELLS:
        raise ValueError('Unknown matched35 cell')
    base = method if method == 'ode_dpmpp_2m' else 'ode_heun'
    cfg = final.get_config('{}_{}_{}'.format(base, count, arm))
    cfg.sampling.method = method
    cfg.sampling.time_grid = 'karras' if method == 'ode_edm_heun' else 'uniform_logsnr'
    cfg.eval.sampling_num_scales = n
    cfg.eval.sampling_seed = seed
    return cfg


def get_config(name):
    method, n, count, arm, seed = name.rsplit('_', 4)
    if seed not in ('seed0', 'seed1'):
        raise ValueError('Unknown sampling seed')
    cell = (int(n), int(count), arm, method, int(seed[4:]))
    if cell not in CELLS:
        raise ValueError('Not a new matched35 cell')
    return config_for(cell)


def anchor_directory(arm):
    return ANCHORS / 'workdirs' / ARMS[arm] / 'eval_edm_protocol_common_nfe35_50000samples_ckpt260_seed0'


def snapshot(source, output):
    # Pin old metadata before producing a new runtime; no old files are changed.
    legacy = json.loads((ANCHORS / 'snapshot.json').read_text())
    if legacy.get('profile') != 'edm_protocol':
        raise RuntimeError('Expected the completed EDM protocol snapshot')
    source = Path(source)
    essential = ('ode_sampling.py', 'sampling.py', 'sde_lib.py', 'main.py', 'run_lib.py',
                 'evaluation.py', 'artifact_utils.py', 'reproducibility.py',
                 'celeba_final_ode_eval.py', 'celeba_edm_protocol_eval.py')
    files = legacy['runtime_files']
    for relative in essential:
        if relative not in files or digest(source / relative) != files[relative]:
            raise RuntimeError('Anchor evaluation source differs: ' + relative)
    for relative, expected in files.items():
        if relative.startswith(('models/', 'configs/')) and relative.endswith('.py'):
            if digest(source / relative) != expected:
                raise RuntimeError('Anchor model/config source differs: ' + relative)
    final.snapshot(source, output)
    output = Path(output)
    target = output / 'anchors'
    target.mkdir()
    pinned = {}
    for relative in ('snapshot.json', 'logs/preflight.json'):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ANCHORS / relative), str(destination))
        pinned[relative] = digest(ANCHORS / relative)
    for arm in ARMS:
        for name in ('manifest.json', 'report_260.npz'):
            path = anchor_directory(arm) / name
            relative = str(path.relative_to(ANCHORS))
            pinned[relative] = digest(path)
            (target / arm).mkdir(exist_ok=True)
            shutil.copy2(str(path), str(target / arm / name))
    saved = json.loads((output / 'snapshot.json').read_text())
    saved.update(profile='matched35', cells=CELLS, external_cells=EXTERNAL_CELLS,
                 anchor_inputs=pinned, sampling_seeds=[0, 1], trajectory_nfe=34, nfe=35,
                 estimated_gpu_hours=[5, 7], estimate_is_guarantee=False)
    saved.pop('sampling_seed', None)
    write_json(output / 'snapshot.json', saved)


def check():
    saved = final.check()
    for relative, expected in saved['anchor_inputs'].items():
        if digest(ANCHORS / relative) != expected:
            raise RuntimeError('Pinned anchor changed: ' + relative)
    return saved


def validate_result(directory, cell, checkpoint_id, source_sha):
    import numpy as np
    from artifact_utils import _canonicalize
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = config_for(cell)
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(cfg[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected config: ' + section)
    # Precision settings affect all compared outputs, including reused anchors.
    for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
        if manifest['config'].get(key) != _canonicalize(cfg[key]):
            raise RuntimeError('Unexpected precision setting: ' + key)
    if manifest['source_sha256'] != source_sha:
        raise RuntimeError('Unexpected result source')
    with np.load(str(directory / 'report_260.npz'), allow_pickle=False) as data:
        expected = dict(num_samples=50000, nfe=35, checkpoint_step=1300002,
                        checkpoint_id=checkpoint_id, dataset_stats_id=REFERENCE_SHA)
        if any(data[k].item() != value for k, value in expected.items()):
            raise RuntimeError('Unexpected report identity')
        metrics = {m + '_' + v: float(data[m + '_' + v])
                   for m in ('IS', 'fid', 'kid') for v in VARIANTS}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError('Non-finite metrics')
    return metrics


def preflight():
    check()
    # Existing native-model restoration and real-weight DPM++/Heun smoke.
    final.preflight()
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    legacy = json.loads((OUTPUT / 'anchors/logs/preflight.json').read_text())
    anchors = []
    for cell in EXTERNAL_CELLS:
        arm = cell[2]
        if legacy['checkpoint_ids'][arm + '_260'] != pre['checkpoint_ids'][arm]:
            raise RuntimeError('Anchor checkpoint differs: ' + arm)
        metrics = validate_result(OUTPUT / 'anchors' / arm, cell,
                                  pre['checkpoint_ids'][arm], legacy['source_sha256'])
        directory = anchor_directory(arm) / 'ckpt_260'
        if len(list(directory.glob('statistics_*.npz'))) != 98:
            raise RuntimeError('Anchor feature shards missing: ' + arm)
        anchors.append(dict(trajectory_nfe=34, nfe=35, num_samples=50000, arm=arm,
                            method='ode_edm_heun', sampling_seed=0, reused=True,
                            features_directory=str(directory), **metrics))
    # The anchor environment must match the new evaluation environment.
    # The old canonical-EDM queue mounted /opt/edm; native score models never
    # use that path. Record paths, but compare the numerical environment here.
    current_env = {k: v for k, v in pre['environment'].items() if k != 'runtime_paths'}
    legacy_env = {k: v for k, v in legacy['environment'].items() if k != 'runtime_paths'}
    if current_env != legacy_env:
        raise RuntimeError('Anchor environment differs; inspect before reuse')
    write_json(OUTPUT / 'anchors/validated.json', anchors)
    print('Matched35 anchors validated; six new cells, two reused cells.', flush=True)


def folder(n, count, arm, method, seed):
    return 'eval_matched35_{}_nfe{}_{}_{}_seed{}'.format(method, n, arm, count, seed)


def run(n, count, arm, method, seed):
    began, clock = time.time(), time.monotonic()
    name = config_name(n, count, arm, method, seed)
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_matched35_eval.py:' + name, '--workdir=workdirs/' + ARMS[arm],
        '--eval_folder=' + folder(n, count, arm, method, seed)])
    write_json(OUTPUT / 'logs' / (name + '_timing.json'), dict(started_unix=began,
        finished_unix=time.time(), elapsed_seconds=time.monotonic() - clock,
        returncode=result.returncode, scope='main.py evaluation including generation and metrics'))
    result.check_returncode()


def report(n, count, arm, method, seed):
    cell = (n, count, arm, method, seed)
    directory = ROOT / 'workdirs' / ARMS[arm] / folder(*cell)
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    metrics = validate_result(directory, cell, pre['checkpoint_ids'][arm], pre['source_sha256'])
    if len(list((directory / 'ckpt_260').glob('statistics_*.npz'))) != 98:
        raise RuntimeError('Unexpected statistics shard count')
    name = config_name(*cell)
    timing = json.loads((OUTPUT / 'logs' / (name + '_timing.json')).read_text())
    if timing['returncode'] != 0 or not math.isfinite(timing['elapsed_seconds']) or timing['elapsed_seconds'] < 0:
        raise RuntimeError('Invalid timing/status')
    write_json(OUTPUT / 'logs' / (name + '_result.json'), dict(trajectory_nfe=n,
        nfe=n + 1, num_samples=count, arm=arm, method=method, sampling_seed=seed,
        reused=False, timing=timing, **metrics))
    cells = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    identities = [(c['trajectory_nfe'], c['num_samples'], c['arm'], c['method'], c['sampling_seed']) for c in cells]
    if len(set(identities)) != len(cells) or not set(identities).issubset(CELLS):
        raise RuntimeError('Unexpected or duplicate summary cell')
    anchors = json.loads((OUTPUT / 'anchors/validated.json').read_text())
    all_cells = cells + anchors
    contrasts = []
    for sampling_seed in (0, 1):
        for label in ARMS:
            selected = {c['method']: c for c in all_cells if c['sampling_seed'] == sampling_seed and c['arm'] == label}
            if len(selected) == 2:
                contrasts.append(dict(arm=label, sampling_seed=sampling_seed,
                    dpmpp_minus_edm=selected['ode_dpmpp_2m']['fid_tweedie_xt'] - selected['ode_edm_heun']['fid_tweedie_xt']))
    write_json(OUTPUT / 'summary.json', dict(status='completed' if len(cells) == 6 else 'running',
        completed_cells=len(cells), total_cells=6, cells=all_cells,
        external_cells_validated=True, point_contrasts=contrasts,
        pairing_validated=False, confidence_intervals_computed=False,
        evaluation_elapsed_seconds=sum(c['timing']['elapsed_seconds'] for c in cells)))
    print('Matched35 {}/6: {} FID={:.6f}, {:.1f} min'.format(
        len(cells), name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot': snapshot(*args)
    elif mode == 'check': check()
    elif mode == 'preflight': preflight()
    elif mode == 'plan':
        for cell in CELLS: print(*cell)
    elif mode in ('run', 'report'):
        n, count, arm, method, seed = args
        cell = (int(n), int(count), arm, method, int(seed))
        if cell not in CELLS: raise ValueError('Unknown cell')
        (run if mode == 'run' else report)(*cell)
    else: raise ValueError(mode)
