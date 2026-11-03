"""Matched35 on RTX PRO 6000: all eight native 50k cells, H200 EDM anchors as reference only.

New cells are compared only with each other, never mixed with H200 numbers.
EDM Heun seed 0 repeats the H200 anchors on the same weights and per-round seeds,
so the same cells measure how far RTX/torch 2.7 drifts from H200/torch 2.4.
The smoke profile is one 1024-image cell of that kind: pipeline, timing and the
row-by-row comparison in a few minutes.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import celeba_matched35_eval as m35
from celeba_matched35_eval import ARMS, ROOT, OUTPUT, VARIANTS, REFERENCE_SHA, final, write_json

METHODS = ('ode_edm_heun', 'ode_dpmpp_2m')
# Heun seed 0 first: the cross-GPU check against H200 is ready after two cells.
FULL = [(34, 50000, arm, method, seed) for seed in (0, 1)
        for method in METHODS for arm in ('vp', 'fox')]
SMOKE = [(34, 1024, 'vp', 'ode_edm_heun', 0)]
PROFILES = {'matched35_rtx': FULL, 'matched35_rtx_smoke': SMOKE}
ESTIMATED_GPU_HOURS = {'matched35_rtx': [4, 12], 'matched35_rtx_smoke': [0.1, 0.3]}
IMAGE_ROUNDS = 4  # image-level comparison; features use every common round
BATCH = 512


def profile():
    name = os.environ.get('CELEBA_QUEUE_PROFILE', '')
    if name not in PROFILES:
        raise RuntimeError('Expected CELEBA_QUEUE_PROFILE in ' + ', '.join(PROFILES))
    return name


def is_reference_cell(cell):
    return cell[3] == 'ode_edm_heun' and cell[4] == 0


def config_for(cell):
    if cell not in FULL + SMOKE:
        raise ValueError('Unknown matched35 RTX cell')
    n, count, arm, method, seed = cell
    # Same protocol as the H200 matched35 cells, including TF32/cudnn settings.
    cfg = m35.config_for((n, 50000, arm, method, seed))
    cfg.eval.num_samples = count
    return cfg


def get_config(name):
    method, n, count, arm, seed = name.rsplit('_', 4)
    if seed not in ('seed0', 'seed1'):
        raise ValueError('Unknown sampling seed')
    return config_for((int(n), int(count), arm, method, int(seed[4:])))


def snapshot(source, output):
    name = profile()
    # Pins the numerical source to the anchor runtime and copies anchor metadata.
    m35.snapshot(source, output)
    path = Path(output) / 'snapshot.json'
    saved = json.loads(path.read_text())
    saved.update(profile=name, cells=PROFILES[name], external_cells=[], anchors_reused=False,
                 anchor_role='H200 reference for EDM Heun seed 0; not mixed into RTX contrasts',
                 estimated_gpu_hours=ESTIMATED_GPU_HOURS[name], estimate_is_guarantee=False)
    write_json(path, saved)


def check():
    saved = m35.check()
    if saved.get('profile') != profile():
        raise RuntimeError('Snapshot profile differs from the queue profile')
    return saved


def preflight():
    import torch
    check()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU')
    # A JIT extension built for another architecture fails only at the first kernel.
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include the visible GPU, sm ' + arch)
    final.preflight()
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    legacy = json.loads((OUTPUT / 'anchors/logs/preflight.json').read_text())
    anchors = []
    for arm in ARMS:
        cell = (34, 50000, arm, 'ode_edm_heun', 0)
        if legacy['checkpoint_ids'][arm + '_260'] != pre['checkpoint_ids'][arm]:
            raise RuntimeError('Anchor checkpoint differs: ' + arm)
        metrics = m35.validate_result(OUTPUT / 'anchors' / arm, cell,
                                      pre['checkpoint_ids'][arm], legacy['source_sha256'])
        directory = m35.anchor_directory(arm) / 'ckpt_260'
        if len(list(directory.glob('statistics_*.npz'))) != 98:
            raise RuntimeError('Anchor feature shards missing: ' + arm)
        anchors.append(dict(trajectory_nfe=34, nfe=35, num_samples=50000, arm=arm,
                            method='ode_edm_heun', sampling_seed=0, reused=False,
                            role='h200_reference', features_directory=str(directory), **metrics))
    current_env = {k: v for k, v in pre['environment'].items() if k != 'runtime_paths'}
    legacy_env = {k: v for k, v in legacy['environment'].items() if k != 'runtime_paths'}
    write_json(OUTPUT / 'anchors/validated.json', dict(anchors=anchors,
        environment_differs=current_env != legacy_env,
        rtx_environment=current_env, h200_environment=legacy_env))
    print('H200 anchors validated as reference; RTX environment differs: {}'.format(
        current_env != legacy_env), flush=True)


def folder(n, count, arm, method, seed):
    return 'eval_matched35rtx_{}_nfe{}_{}_{}_seed{}'.format(method, n, arm, count, seed)


def run(n, count, arm, method, seed):
    began, clock = time.time(), time.monotonic()
    name = m35.config_name(n, count, arm, method, seed)
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_matched35_rtx_eval.py:' + name, '--workdir=workdirs/' + ARMS[arm],
        '--eval_folder=' + folder(n, count, arm, method, seed)])
    write_json(OUTPUT / 'logs' / (name + '_timing.json'), dict(started_unix=began,
        finished_unix=time.time(), elapsed_seconds=time.monotonic() - clock,
        returncode=result.returncode, scope='main.py evaluation including generation and metrics'))
    result.check_returncode()


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
    for key in ('allow_tf32', 'cudnn_benchmark', 'allow_fp16_reduced_precision_reduction'):
        if manifest['config'].get(key) != _canonicalize(cfg[key]):
            raise RuntimeError('Unexpected precision setting: ' + key)
    if manifest['source_sha256'] != source_sha:
        raise RuntimeError('Unexpected result source')
    with np.load(str(directory / 'report_260.npz'), allow_pickle=False) as data:
        expected = dict(num_samples=cell[1], nfe=35, checkpoint_step=1300002,
                        checkpoint_id=checkpoint_id, dataset_stats_id=REFERENCE_SHA)
        if any(data[k].item() != value for k, value in expected.items()):
            raise RuntimeError('Unexpected report identity')
        metrics = {m + '_' + v: float(data[m + '_' + v])
                   for m in ('IS', 'fid', 'kid') for v in VARIANTS}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError('Non-finite metrics')
    if len(list((directory / 'ckpt_260').glob('statistics_*.npz'))) != (cell[1] + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected statistics shard count')
    return metrics


def _relative_distances(mine, theirs):
    import numpy as np
    mine, theirs = mine.reshape(len(mine), -1).astype(np.float64), theirs.reshape(len(theirs), -1).astype(np.float64)
    scale = np.linalg.norm(theirs, axis=1)
    # Neighbouring rows come from independent noise: the "unrelated sample" scale.
    return (np.linalg.norm(mine - theirs, axis=1) / scale,
            np.linalg.norm(mine - np.roll(theirs, 1, axis=0), axis=1) / scale)


def compare_with_anchor(cell, directory):
    """Row-by-row RTX vs H200 comparison for the same weights and per-round seeds.

    Paired distances far below unpaired ones mean the same initial noise and close
    trajectories. Paired close to unpaired means the RNG streams differ; then only
    the 50k FID difference is a valid comparison.
    """
    import numpy as np
    arm = cell[2]
    reference = m35.anchor_directory(arm) / 'ckpt_260'
    ours = directory / 'ckpt_260'
    feature_pairs, image_pairs, float_pairs = [], [], []
    for r in range((cell[1] + BATCH - 1) // BATCH):
        name = 'statistics_{}.npz'.format(r)
        with np.load(str(ours / name)) as a, np.load(str(reference / name)) as b:
            if int(a['sampling_seed']) != int(b['sampling_seed']):
                raise RuntimeError('Per-round sampling seeds differ: round {}'.format(r))
            feature_pairs.append(_relative_distances(a['pool_3_tweedie_xt'], b['pool_3_tweedie_xt']))
        name = 'samples_{}.npz'.format(r)
        if r < IMAGE_ROUNDS and (ours / name).exists() and (reference / name).exists():
            with np.load(str(ours / name)) as a, np.load(str(reference / name)) as b:
                image_pairs.append(_relative_distances(a['samples_tweedie_xt'], b['samples_tweedie_xt']))
                if 'float_x_t' in a.files and 'float_x_t' in b.files:
                    float_pairs.append(_relative_distances(a['float_x_t'], b['float_x_t']))

    def summary(pairs):
        if not pairs:
            return None
        paired = np.concatenate([p for p, _ in pairs])
        unpaired = np.concatenate([u for _, u in pairs])
        return dict(rows=int(len(paired)), paired_median=float(np.median(paired)),
                    paired_p95=float(np.percentile(paired, 95)),
                    unpaired_median=float(np.median(unpaired)),
                    paired_to_unpaired=float(np.median(paired) / np.median(unpaired)))
    return dict(pool_3_tweedie_xt=summary(feature_pairs), samples_tweedie_xt_uint8=summary(image_pairs),
                float_x_t=summary(float_pairs), image_rounds=len(image_pairs))


def report(n, count, arm, method, seed):
    cell = (n, count, arm, method, seed)
    plan = PROFILES[profile()]
    directory = ROOT / 'workdirs' / ARMS[arm] / folder(*cell)
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    metrics = validate_result(directory, cell, pre['checkpoint_ids'][arm], pre['source_sha256'])
    name = m35.config_name(*cell)
    timing = json.loads((OUTPUT / 'logs' / (name + '_timing.json')).read_text())
    if timing['returncode'] != 0 or not math.isfinite(timing['elapsed_seconds']) or timing['elapsed_seconds'] < 0:
        raise RuntimeError('Invalid timing/status')
    result = dict(trajectory_nfe=n, nfe=n + 1, num_samples=count, arm=arm, method=method,
                  sampling_seed=seed, reused=False, gpu=pre['environment'].get('visible_cuda_devices'),
                  timing=timing, **metrics)
    if is_reference_cell(cell):
        result['cross_gpu'] = compare_with_anchor(cell, directory)
    write_json(OUTPUT / 'logs' / (name + '_result.json'), result)
    cells = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    identities = [(c['trajectory_nfe'], c['num_samples'], c['arm'], c['method'], c['sampling_seed']) for c in cells]
    if len(set(identities)) != len(cells) or not set(identities).issubset(plan):
        raise RuntimeError('Unexpected or duplicate summary cell')
    anchors = {a['arm']: a for a in json.loads((OUTPUT / 'anchors/validated.json').read_text())['anchors']}
    contrasts, cross_gpu = [], []
    others = {(o['arm'], o['method'], o['sampling_seed']): o for o in cells}
    for c in cells:
        if c['method'] == 'ode_dpmpp_2m' and (c['arm'], 'ode_edm_heun', c['sampling_seed']) in others:
            heun = others[(c['arm'], 'ode_edm_heun', c['sampling_seed'])]
            contrasts.append(dict(kind='dpmpp_minus_edm', arm=c['arm'], sampling_seed=c['sampling_seed'],
                                  value=c['fid_tweedie_xt'] - heun['fid_tweedie_xt']))
        if c['arm'] == 'fox' and ('vp', c['method'], c['sampling_seed']) in others:
            vp = others[('vp', c['method'], c['sampling_seed'])]
            contrasts.append(dict(kind='fox_minus_vp', method=c['method'], sampling_seed=c['sampling_seed'],
                                  value=c['fid_tweedie_xt'] - vp['fid_tweedie_xt']))
        if 'cross_gpu' in c:
            entry = dict(arm=c['arm'], num_samples=c['num_samples'], rows=c['cross_gpu'])
            if c['num_samples'] == 50000:
                entry.update(fid_rtx=c['fid_tweedie_xt'], fid_h200=anchors[c['arm']]['fid_tweedie_xt'],
                             rtx_minus_h200=c['fid_tweedie_xt'] - anchors[c['arm']]['fid_tweedie_xt'])
            cross_gpu.append(entry)
    elapsed = sum(c['timing']['elapsed_seconds'] for c in cells)
    write_json(OUTPUT / 'summary.json', dict(profile=profile(),
        status='completed' if len(cells) == len(plan) else 'running',
        completed_cells=len(cells), total_cells=len(plan), cells=cells,
        point_contrasts=contrasts, cross_gpu=cross_gpu, h200_anchors_mixed_into_contrasts=False,
        confidence_intervals_computed=False, evaluation_elapsed_seconds=elapsed))
    print('Matched35 RTX {}/{}: {} FID={:.6f}, {:.1f} min'.format(
        len(cells), len(plan), name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)
    if 'cross_gpu' in result:
        rows = result['cross_gpu']['pool_3_tweedie_xt']
        print('Cross-GPU features: paired/unpaired = {:.4f} on {} rows'.format(
            rows['paired_to_unpaired'], rows['rows']), flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot': snapshot(*args)
    elif mode == 'check': check()
    elif mode == 'preflight': preflight()
    elif mode == 'plan':
        for cell in PROFILES[profile()]: print(*cell)
    elif mode in ('run', 'report'):
        n, count, arm, method, seed = args
        cell = (int(n), int(count), arm, method, int(seed))
        if cell not in PROFILES[profile()]: raise ValueError('Unknown cell')
        (run if mode == 'run' else report)(*cell)
    else: raise ValueError(mode)
