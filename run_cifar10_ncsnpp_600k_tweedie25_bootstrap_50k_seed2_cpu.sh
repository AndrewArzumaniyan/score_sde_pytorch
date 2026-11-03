#!/usr/bin/env bash
# CPU only: FID(FOX)-FID(VP) at 25% final Tweedie, NCSN++ 600k, 50k, seed 2.
# Reads saved pool_3_w025 and manifests/summary; no model or trajectory execution.
# The default input is the SUCCESSFUL partial-Tweedie run supplied on November 3.
# One standalone file; all previous eval/bootstrap drivers stay unchanged.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${CIFAR_BOOTSTRAP_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export CIFAR_TWEEDIE25_BOOTSTRAP_SCRIPT="$PWD/$(basename "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" -u - "$@" <<'PY_BOOTSTRAP'
"""Conditional, pointwise FID-difference intervals for the fixed 25% operator.

NumPy only: no Torch, TensorFlow, Inception, checkpoint or float-state loading.
Paired resampling uses common image indices after checking round seeds/order.
The coefficient was explored on these same images. These intervals do NOT
adjust for coefficient selection, training variability or finite-sample FID bias.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

COUNT, BATCH, DIMENSION, STEP = 50000, 512, 2048, 600001
SEED, WEIGHT, ROUNDS = 2, 0.25, math.ceil(COUNT / BATCH)
REFERENCE_SHA = 'c464743bce16cb0f64408eb36bdc7a7ef0e7b4dfd618c38f8119d4e29ef0c9d5'
CHECKPOINT_IDS = {'vp': 'db841db3b8cd45949ef96d1d8915c1f9',
                  'fox': '8c08568db0d144b9a231763669af5319'}
IDENTITY = ('protocol_sha256', 'checkpoint_id', 'checkpoint_step',
            'training_protocol_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scalar(z, key):
    value = np.asarray(z[key])
    require(value.shape == (), 'Expected scalar metadata: ' + key)
    return value.item()


def manifest_hash(manifest, field):
    payload = {k: v for k, v in manifest.items() if k != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def round_seed(index):
    payload = '\0'.join((str(SEED), 'sampling-round', str(index)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2**31-1)


def check_source_config(cfg, arm):
    m, e, s = cfg['model'], cfg['eval'], cfg['sampling']
    for key, expected in dict(name='ncsnpp', nf=128, ch_mult=[1, 2, 2, 2],
                              num_res_blocks=4, fir=True, progressive_input='residual',
                              scale_by_sigma=False, num_scales=1000, ema_rate=0.9999).items():
        require(m.get(key) == expected, arm + ': wrong original model.' + key)
    require(cfg['data']['dataset'] == 'CIFAR10' and cfg['data']['image_size'] == 32
            and cfg['data']['num_channels'] == 3 and cfg['data']['centered'] is True
            and cfg['seed'] == 42, arm + ': wrong data/training seed')
    require(cfg['training']['sde'] == ('vpsde' if arm == 'vp' else 'foxvpsde')
            and cfg['training']['continuous'] is True
            and cfg['training'].get('noise_distribution', 'uniform_time') == 'uniform_time'
            and m.get('noise_conditioning', 'time') == 'time', arm + ': wrong native clock/SDE')
    for key, expected in dict(num_samples=COUNT, batch_size=BATCH, sampling_seed=SEED,
                              sampling_num_scales=64, endpoint_denoise_variants=True,
                              save_endpoint_states=True).items():
        require(e.get(key) == expected, arm + ': wrong original eval.' + key)
    # begin_ckpt/end_ckpt are intentionally absent from canonical eval manifests.
    require(tuple(e.get('endpoint_denoise_outputs', ())) == ('x_t', 'tweedie_xt'),
            arm + ': wrong endpoint output variants')
    for key, expected in dict(method='ode_dpmpp_2m', probability_flow=True,
                              time_grid='uniform_logsnr', logsnr_min=-10.0,
                              logsnr_max=10.575749165617202).items():
        require(s.get(key) == expected, arm + ': wrong original sampling.' + key)
    if arm == 'fox':
        require(m.get('fox_normalize_scale', False) is False, 'Expected unnormalized real FOX')
        for key, expected in dict(fox_u=-5.0, fox_drift_schedule='constant', fox_kernel='matern_3_2',
                                  fox_matern_length_scale=346.4101615138,
                                  fox_target_terminal_variance=1.0, fox_schedule_grid_size=4096).items():
            require(m.get(key) == expected, 'Wrong native FOX parameter: ' + key)


def read_arm(input_dir, protocol, summary, arm):
    """Check the exact source identities; load ONLY the new 25% feature rows."""
    pinned = protocol['input_sha256']
    suffix = (f'eval_cifar_ncsnpp600k_dpmpp64_extended50k_v1_ode_dpmpp_2m_{arm}_'
              'extended_ckpt24_nfe64_50000samples_batch512_seed2/manifest.json')
    candidates = [p for p in pinned if str(p).replace('\\', '/').endswith(suffix)]
    require(len(candidates) == 1, 'Missing or ambiguous original eval manifest: ' + arm)
    source_path = Path(candidates[0])
    source_dir = source_path.parent
    source_report_path = source_dir/'report_24.npz'
    require(file_hash(source_path) == pinned[str(source_path)], 'Original eval manifest changed: ' + arm)
    require(str(source_report_path) in pinned
            and file_hash(source_report_path) == pinned[str(source_report_path)],
            'Original eval report changed: ' + arm)
    source = json.loads(source_path.read_text())
    require(manifest_hash(source, 'protocol_sha256') == source['protocol_sha256'],
            'Corrupt original eval manifest: ' + arm)
    check_source_config(source['config'], arm)
    with np.load(source_report_path, allow_pickle=False) as z:
        identity = {k: scalar(z, k) for k in IDENTITY}
        require(identity['protocol_sha256'] == source['protocol_sha256']
                and identity['checkpoint_id'] == CHECKPOINT_IDS[arm]
                and identity['checkpoint_step'] == STEP and scalar(z, 'num_samples') == COUNT
                and scalar(z, 'nfe') == 65 and scalar(z, 'dataset_stats_id') == REFERENCE_SHA,
                'Wrong original checkpoint/reference/protocol: ' + arm)
    progress = summary['arms'][arm]
    require(progress.get('status') == 'complete' and progress.get('completed_rounds') == ROUNDS,
            'Partial-Tweedie evaluation has not finished: ' + arm)
    rows = [r for r in summary.get('rows', []) if r.get('arm') == arm and r.get('weight') == WEIGHT]
    require(len(rows) == 1, 'Missing or duplicate completed 25% metric row: ' + arm)
    row = rows[0]
    require(row.get('count') == COUNT and row.get('sampling_seed') == SEED
            and row.get('percent') == 25 and row.get('features') == 'new'
            and math.isfinite(row['fid']), 'Wrong 25% metric settings: ' + arm)
    arm_rows = [r for r in progress.get('rows', []) if r.get('weight') == WEIGHT]
    require(arm_rows == rows, 'Per-arm/global metric rows disagree: ' + arm)
    files = [input_dir/arm/f'statistics_{i}.npz' for i in range(ROUNDS)]
    require(set((input_dir/arm).glob('statistics_*.npz')) == set(files),
            'Missing or extra saved 25% feature batches: ' + arm)
    chunks, rounds = [], []
    hashes = {str(source_path): file_hash(source_path), str(source_report_path): file_hash(source_report_path)}
    for i, path in enumerate(files):
        hashes[str(path)] = file_hash(path)
        valid = min(BATCH, COUNT - i*BATCH)
        original_sample = source_dir/'ckpt_24'/f'samples_{i}.npz'
        original_stats = source_dir/'ckpt_24'/f'statistics_{i}.npz'
        # Compare to immutable source hashes already pinned by the diagnostic.
        # Large original samples/statistics need not be reread or copied for this bootstrap.
        require(str(original_sample) in pinned and str(original_stats) in pinned,
                'Source provenance incomplete for round ' + str(i))
        with np.load(path, allow_pickle=False) as z:
            for key, expected in dict(diagnostic_protocol_sha256=protocol['diagnostic_protocol_sha256'],
                                      checkpoint_id=CHECKPOINT_IDS[arm], checkpoint_step=STEP,
                                      source_protocol_sha256=identity['protocol_sha256'], round_id=i,
                                      sampling_seed=round_seed(i), num_samples=BATCH,
                                      num_samples_used=valid,
                                      source_sample_sha256=pinned[str(original_sample)],
                                      source_statistics_sha256=pinned[str(original_stats)]).items():
                require(scalar(z, key) == expected, 'Wrong %s: %s' % (key, path))
            require('pool_3_w025' in z.files, 'Missing saved 25% features: ' + str(path))
            pool = np.asarray(z['pool_3_w025'], dtype=np.float64)
            require(pool.shape == (BATCH, DIMENSION) and np.isfinite(pool).all(),
                    'Invalid 25% Inception features: ' + str(path))
            chunks.append(pool[:valid].copy())
            rounds.append((i, round_seed(i), valid))
    pools = np.concatenate(chunks, axis=0)
    require(pools.shape == (COUNT, DIMENSION), 'Expected exactly 50000 retained feature rows.')
    return dict(pools=pools, round_order=rounds, source_manifest=source,
                checkpoint_identity=identity, original_eval_dir=str(source_dir),
                recorded_fid=row['fid'], input_sha256=hashes)


def moments(x):
    mean = x.mean(axis=0)
    centered = x - mean
    return mean, centered.T @ centered / (len(x) - 1)


def reference_moments(x):
    mean, cov = moments(x)
    eig, vectors = np.linalg.eigh(cov)
    root = (vectors * np.sqrt(np.maximum(eig, 0))) @ vectors.T
    return mean, float(np.trace(cov)), root


def fid(x, real):
    mean, cov = moments(x)
    real_mean, real_trace, root = real
    cross = root @ cov @ root
    eig = np.linalg.eigvalsh((cross + cross.T) * 0.5)
    value = float(np.square(mean-real_mean).sum() + np.trace(cov) + real_trace
                  - 2*np.sqrt(np.maximum(eig, 0)).sum())
    require(math.isfinite(value) and value >= -1e-6, 'Non-finite or negative covariance FID.')
    return max(0.0, value)


def interval_summary(draws, point):
    low, median, high = (float(v) for v in np.quantile(draws, [.025, .5, .975]))
    return dict(point_delta_fox_minus_vp=point, percentile_95=[low, high],
                basic_95=[2*point-high, 2*point-low], bootstrap_median=median,
                bootstrap_mean=float(np.mean(draws)), estimated_bootstrap_bias=float(np.mean(draws)-point),
                bootstrap_sd=float(np.std(draws, ddof=1)) if len(draws)>1 else None)


def save_output(path, result):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='results/cifar10_ncsnpp_partial_tweedie/600k_seed2_50k_20261103T090756.153298Z')
    parser.add_argument('--reference', default='assets/stats/cifar10_stats.npz')
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261103)
    parser.add_argument('--resampling', choices=('paired', 'independent'), default='paired')
    parser.add_argument('--output', help='NEW output JSON; existing results are never overwritten.')
    args = parser.parse_args()
    require(args.reps > 0 and args.bootstrap_seed >= 0, 'Need positive reps and nonnegative bootstrap seed.')
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = Path(args.output or ('results/cifar10_ncsnpp_partial_tweedie_bootstrap/'
                                 f'vp600k_fox600k_w025_seed2_50000samples_{args.resampling}_{stamp}.json'))
    require(not output.exists(), 'Refusing to overwrite: ' + str(output))
    input_dir = Path(args.input_dir)
    manifest_path, summary_path = input_dir/'manifest.json', input_dir/'summary.json'
    protocol, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    require(manifest_hash(protocol, 'diagnostic_protocol_sha256') == protocol['diagnostic_protocol_sha256'],
            'Corrupt partial-Tweedie manifest.')
    require(summary.get('status') == 'complete'
            and summary.get('diagnostic_protocol_sha256') == protocol['diagnostic_protocol_sha256'],
            'Need a successfully COMPLETED matching partial-Tweedie result.')
    for key, expected in dict(schema_version=1, experiment='partial_native_final_tweedie', count=COUNT,
                              batch_size=BATCH, rounds=ROUNDS, sampling_seed=SEED, checkpoint_step=STEP,
                              lambda_max=10.575749165617202,
                              weights={'w000':0.0, 'w025':0.25, 'w050':0.5, 'w100':1.0}).items():
        require(protocol.get(key) == expected, 'Wrong partial-Tweedie protocol: ' + key)
    require(file_hash(args.reference) == REFERENCE_SHA, 'Reference differs from original 50k eval.')
    pair = {}
    for arm in ('vp', 'fox'):
        print('Checking saved 25% features:', input_dir/arm, flush=True)
        pair[arm] = read_arm(input_dir, protocol, summary, arm)
    va, fa = (pair[a]['source_manifest'] for a in ('vp', 'fox'))
    for key in ('environment', 'source_sha256'):
        require(va[key] == fa[key], 'Original VP/FOX runtime differs: ' + key)
    for key in ('data', 'sampling', 'eval'):
        require(va['config'][key] == fa['config'][key], 'Original VP/FOX config differs: ' + key)
    network_configs = [{k:v for k,v in m['config']['model'].items() if not k.startswith('fox_')}
                       for m in (va, fa)]
    require(network_configs[0] == network_configs[1], 'VP/FOX network/clock configurations differ.')
    require(pair['vp']['round_order'] == pair['fox']['round_order'], 'VP/FOX row/round alignment differs.')
    with np.load(args.reference, allow_pickle=False) as z:
        reference = np.asarray(z['pool_3'], dtype=np.float64)
    require(reference.shape == (COUNT, DIMENSION) and np.isfinite(reference).all(), 'Invalid real-reference features.')
    real = reference_moments(reference)
    del reference
    for arm, data in pair.items():
        data['recomputed_fid'] = fid(data['pools'], real)
        require(np.isclose(data['recomputed_fid'], data['recorded_fid'], atol=2e-3, rtol=1e-4),
                f'{arm}: saved 25% features do not reproduce reported FID: '
                f'{data["recomputed_fid"]:.8f} vs {data["recorded_fid"]:.8f}')
        print(f'{arm.upper()} 25% Tweedie: FID={data["recomputed_fid"]:.6f} '
              f'(report {data["recorded_fid"]:.6f})', flush=True)
    point = pair['fox']['recomputed_fid'] - pair['vp']['recomputed_fid']
    result = dict(schema_version=1, status='running', created_utc=stamp, config=vars(args),
                  contrast='FID(FOX)-FID(VP)', architecture='ncsnpp', checkpoint_step=STEP,
                  count=COUNT, sampling_seed=SEED, tweedie_weight=WEIGHT,
                  sampler='ode_dpmpp_2m', trajectory_nfe=64, total_nfe=65,
                  logsnr_min=-10.0, logsnr_max=10.575749165617202, reference_sha256=REFERENCE_SHA,
                  diagnostic_protocol_sha256=protocol['diagnostic_protocol_sha256'],
                  input_sha256={str(manifest_path):file_hash(manifest_path), str(summary_path):file_hash(summary_path)},
                  script_sha256=file_hash(os.environ['CIFAR_TWEEDIE25_BOOTSTRAP_SCRIPT']),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   threads=os.environ.get('OMP_NUM_THREADS')),
                  arms={arm:{k:v for k,v in data.items() if k not in ('pools','round_order')}
                        for arm,data in pair.items()}, completed_reps=0, bootstrap_deltas=[],
                  point_delta_fox_minus_vp=point,
                  interval_scope='Pointwise; fixed 25% coefficient, checkpoints, reference and sampling seed.',
                  limitations=['25% was explored on these images; no adjustment for coefficient selection.',
                               'No training-seed uncertainty or correction of finite-sample FID bias.',
                               'Paired using checked round seeds/order; no explicit per-image RNG tapes.',
                               'Percentile/basic intervals are approximate, not an equivalence test.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('Output:', output, flush=True)
    print('CPU only. No generation. 25% Tweedie; negative delta favours FOX.', flush=True)
    draws = []
    rng = np.random.default_rng(np.random.SeedSequence([args.bootstrap_seed, SEED]))
    loop_started = time.monotonic()
    try:
        for rep in range(args.reps):
            vp_ids = rng.integers(0, COUNT, size=COUNT)
            fox_ids = vp_ids if args.resampling == 'paired' else rng.integers(0, COUNT, size=COUNT)
            draws.append(fid(pair['fox']['pools'][fox_ids], real) - fid(pair['vp']['pools'][vp_ids], real))
            if rep == 0 or (rep+1)%25 == 0 or rep+1 == args.reps:
                result.update(completed_reps=rep+1, bootstrap_deltas=draws,
                              elapsed_seconds=time.monotonic()-started,
                              interval=interval_summary(draws, point))
                save_output(output, result)
                elapsed = time.monotonic()-loop_started
                print(f'seed={SEED} w=25% replicate={rep+1}/{args.reps} elapsed={elapsed:.1f}s '
                      f'ETA={elapsed/(rep+1)*(args.reps-rep-1):.1f}s', flush=True)
        result.update(status='complete', elapsed_seconds=time.monotonic()-started,
                      timing_only=args.reps<500)
        if args.reps<500:
            result['limitations'].append('Timing/smoke only: fewer than 500 draws; unstable tail quantiles.')
        save_output(output, result)
        ci = result['interval']
        print(f'25% Tweedie FOX-VP delta={point:+.6f} percentile95={ci["percentile_95"]} '
              f'basic95={ci["basic_95"]}', flush=True)
        print('Finished:', output, flush=True)
    except BaseException as error:
        result.update(status='failed', elapsed_seconds=time.monotonic()-started,
                      error=type(error).__name__+': '+str(error))
        save_output(output, result)
        raise


if __name__ == '__main__':
    main()
PY_BOOTSTRAP
