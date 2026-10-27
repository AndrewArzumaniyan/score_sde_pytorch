#!/usr/bin/env bash
# CPU-only, read-only analysis of existing CIFAR endpoint/Tweedie features.
# DDPM++ deep only: VP600 vs FOX600, checkpoint 12, 5000 images, seeds 0/1.
# bash run_cifar10_ddpmpp_deep_600k_bootstrap_intervals.sh --help
# Kept in one .sh so adding this analysis does not change runtime source hashes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${CIFAR_BOOTSTRAP_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export CIFAR_BOOTSTRAP_SCRIPT="$PWD/$(basename "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" -u - "$@" <<'PY_BOOTSTRAP'
"""Conditional FID bootstrap; no Torch, TensorFlow, Inception or generation.

Intervals are approximate, pointwise, fixed-checkpoint/fixed-reference intervals.
They do not correct finite-sample FID bias or include training/selection variance.
Paired mode checks seed/round/order metadata, not explicit per-image RNG tapes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

DIMENSION = 2048
CHECKPOINT = 12
EXPECTED_STEP = 600001
IDENTITY = ('protocol_sha256', 'checkpoint_id', 'checkpoint_step',
            'training_protocol_sha256')
VARIANTS = ('x_mean', 'tweedie_xt')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scalar(archive, key):
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError('Non-scalar metadata: ' + key)
    return value.item()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ddpmpp_protocol(config, arm, folder):
    """Reject the old NCSN++ pair and a different endpoint/sampler protocol."""
    model, training = config['model'], config['training']
    expected_model = dict(name='ncsnpp', nf=128, ch_mult=[1, 2, 2, 2],
                          num_res_blocks=8, fir=False, progressive_input='none',
                          progressive='none', scale_by_sigma=False,
                          embedding_type='positional', noise_conditioning='time',
                          ema_rate=0.9999)
    for key, expected in expected_model.items():
        require(model.get(key) == expected,
                '%s: expected DDPM++ deep model.%s=%r' % (folder, key, expected))
    require(training.get('sde') == ('vpsde' if arm == 'vp' else 'foxvpsde')
            and training.get('continuous') is True
            and training.get('noise_distribution') == 'uniform_time',
            'Wrong DDPM++ training SDE/conditioning protocol: ' + str(folder))
    require(config.get('seed') == 42 and config['data'].get('centered') is True,
            'Expected training seed 42 and centered data: ' + str(folder))
    for key in ('allow_tf32', 'cudnn_benchmark', 'deterministic',
                'allow_fp16_reduced_precision_reduction'):
        require(config.get(key) is False, 'Expected disabled flag %s: %s' % (key, folder))
    sampling = config['sampling']
    expected_sampling = dict(method='pc', predictor='euler_maruyama', corrector='none',
                             time_grid='uniform_logsnr', probability_flow=False,
                             noise_removal=True, logsnr_min=-10.0,
                             logsnr_max=9.115429865459795)
    for key, expected in expected_sampling.items():
        require(sampling.get(key) == expected,
                '%s: expected sampling.%s=%r' % (folder, key, expected))
    ev = config['eval']
    require(ev.get('sampling_num_scales') == 1000
            and ev.get('enable_sampling') is True
            and ev.get('enable_loss') is False and ev.get('enable_bpd') is False,
            'Expected sampling-only 1000 EM + 1 Tweedie eval: ' + str(folder))
    if arm == 'fox':
        for key, expected in dict(fox_u=-5.0, fox_drift_schedule='constant',
                                  fox_kernel='matern_3_2',
                                  fox_matern_length_scale=346.4101615138,
                                  fox_target_terminal_variance=1.0,
                                  fox_normalize_scale=False,
                                  fox_schedule_grid_size=4096).items():
            require(model.get(key) == expected,
                    '%s: expected real FOX model.%s=%r' % (folder, key, expected))


def round_seed(base, round_id):
    # Same stable derivation as reproducibility.derive_seed, without Torch.
    payload = '\0'.join((str(base), 'sampling-round', str(round_id)))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8],
                          byteorder='big') % (2 ** 31 - 1)


def read_arm(folder, checkpoint, count, batch, seed, variants, reference_id):
    folder = Path(folder)
    manifest_path = folder / 'manifest.json'
    report_path = folder / ('report_%d.npz' % checkpoint)
    manifest = json.loads(manifest_path.read_text())
    payload = {k: v for k, v in manifest.items() if k != 'protocol_sha256'}
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')).hexdigest()
    require(digest == manifest['protocol_sha256'],
            'Manifest hash mismatch: ' + str(manifest_path))
    config = manifest['config']
    require(config['data']['dataset'] == 'CIFAR10'
            and config['data']['image_size'] == 32,
            'Expected CIFAR10 32x32: ' + str(folder))
    ev = config['eval']
    for key, expected in (('num_samples', count), ('batch_size', batch),
                          ('sampling_seed', seed),
                          ('endpoint_denoise_variants', True)):
        require(ev.get(key) == expected,
                '%s: config.eval.%s must be %r' % (folder, key, expected))
    with np.load(report_path, allow_pickle=False) as z:
        report = {key: scalar(z, key) for key in IDENTITY}
        report.update({key: scalar(z, key)
                       for key in ('num_samples', 'nfe', 'dataset_stats_id')})
        report.update({'fid_' + v: float(scalar(z, 'fid_' + v))
                       for v in variants})
    require(report['protocol_sha256'] == digest,
            'Report/manifest protocol mismatch: ' + str(folder))
    require(report['num_samples'] == count,
            'Report sample count mismatch: ' + str(folder))
    require(report['dataset_stats_id'] == reference_id,
            'Exact reference SHA256 differs from report: ' + str(folder))
    require(isinstance(report['checkpoint_id'], str) and report['checkpoint_id']
            and isinstance(report['training_protocol_sha256'], str)
            and report['training_protocol_sha256'] not in ('', 'legacy-unknown'),
            'Missing checkpoint/training identity: ' + str(folder))
    require(report['nfe'] == ev['sampling_num_scales'] + 1,
            'NFE mismatch: ' + str(folder))
    # These continuation checkpoints store the next step (e.g. 600001).
    require(checkpoint == CHECKPOINT and report['checkpoint_step'] == EXPECTED_STEP,
            'Unexpected checkpoint step: ' + str(folder))
    rounds = int(math.ceil(count / float(batch)))
    sample_dir = folder / ('ckpt_%d' % checkpoint)
    expected_files = [sample_dir / ('statistics_%d.npz' % i)
                      for i in range(rounds)]
    require(set(sample_dir.glob('statistics_*.npz')) == set(expected_files),
            'Missing or extra statistics rounds: ' + str(sample_dir))
    chunks = {v: [] for v in variants}
    hashes = {str(manifest_path): file_hash(manifest_path),
              str(report_path): file_hash(report_path)}
    round_ids = []
    for i, path in enumerate(expected_files):
        hashes[str(path)] = file_hash(path)
        with np.load(path, allow_pickle=False) as z:
            for key in IDENTITY:
                require(scalar(z, key) == report[key],
                        'Statistics identity mismatch (%s): %s' % (key, path))
            require(scalar(z, 'round_id') == i,
                    'Wrong round ID: ' + str(path))
            actual_seed = scalar(z, 'sampling_seed')
            require(actual_seed == round_seed(seed, i),
                    'Wrong round sampling seed: ' + str(path))
            require(scalar(z, 'num_samples') == batch
                    and scalar(z, 'nfe') == report['nfe'],
                    'Wrong batch size/NFE: ' + str(path))
            round_ids.append((i, actual_seed, batch))
            for v in variants:
                x = np.asarray(z['pool_3_' + v], dtype=np.float64)
                require(x.shape == (batch, DIMENSION)
                        and np.isfinite(x).all(),
                        'Invalid Inception features: %s (%s)' % (path, v))
                chunks[v].append(x)
    pools = {v: np.concatenate(chunks[v], axis=0)[:count] for v in variants}
    return dict(folder=str(folder), checkpoint=checkpoint, report=report,
                manifest=manifest, pools=pools, rounds=round_ids,
                input_sha256=hashes)


def check_comparison(vp, fox, paired):
    a, b = vp['manifest'], fox['manifest']
    ca, cb = a['config'], b['config']
    require(ca['data'] == cb['data'], 'VP/FOX data settings differ.')
    require(ca['sampling'] == cb['sampling'],
            'VP/FOX sampling settings/endpoints differ.')
    vp_model = {k: v for k, v in ca['model'].items() if not k.startswith('fox_')}
    fox_model = {k: v for k, v in cb['model'].items() if not k.startswith('fox_')}
    require(vp_model == fox_model, 'VP/FOX network/conditioning settings differ.')
    require(a['source_sha256'] == b['source_sha256'],
            'VP/FOX evaluation runtime sources differ.')
    require(a['environment'] == b['environment'],
            'VP/FOX evaluation environments differ.')
    for key in ('allow_tf32', 'cudnn_benchmark', 'deterministic',
                'allow_fp16_reduced_precision_reduction'):
        require(ca.get(key) == cb.get(key), 'VP/FOX flag differs: ' + key)
    require(vp['report']['nfe'] == fox['report']['nfe'], 'VP/FOX NFE differ.')
    if paired:
        require(vp['rounds'] == fox['rounds'],
                'Paired mode needs identical round seeds and row order.')


def moments(x):
    mean = x.mean(axis=0)
    centered = x - mean
    return mean, centered.T @ centered / (len(x) - 1)


def reference_moments(x):
    mean, cov = moments(x)
    w, u = np.linalg.eigh(cov)
    root = (u * np.sqrt(np.maximum(w, 0))) @ u.T
    return mean, float(np.trace(cov)), root


def fid(x, real):
    mean, cov = moments(x)
    real_mean, real_trace, root = real
    cross = root @ cov @ root
    cross = (cross + cross.T) * 0.5
    eig = np.linalg.eigvalsh(cross)
    value = float(np.square(mean - real_mean).sum() + np.trace(cov)
                  + real_trace - 2 * np.sqrt(np.maximum(eig, 0)).sum())
    require(math.isfinite(value) and value >= -1e-6,
            'Invalid FID from numerical covariance calculation.')
    return max(0.0, value)


def interval_summary(draws, point):
    low, median, high = (float(x) for x in np.quantile(draws, [.025, .5, .975]))
    return dict(point_delta_fox_minus_vp=point,
                percentile_95=[low, high], bootstrap_median=median,
                bootstrap_mean=float(np.mean(draws)),
                estimated_bootstrap_bias=float(np.mean(draws) - point),
                basic_95=[2 * point - high, 2 * point - low],
                bootstrap_sd=float(np.std(draws, ddof=1)) if len(draws) > 1 else None)


def summarize_seed_mean(result, variants):
    """Fixed-seed average of separately computed FID contrasts, not pooled FID."""
    rows = result['seeds']
    for variant in variants:
        points = [row['variants'][variant]['point_delta_fox_minus_vp'] for row in rows]
        # Each sampling seed has its own independent bootstrap RNG stream.
        draws = np.mean([row['variants'][variant]['bootstrap_deltas'] for row in rows], axis=0)
        summary = interval_summary(draws, float(np.mean(points)))
        summary['bootstrap_deltas'] = draws.tolist()
        summary['sampling_seeds'] = [row['sampling_seed'] for row in rows]
        summary['definition'] = ('Arithmetic mean of separate per-seed FID(FOX)-FID(VP) '
                                 'contrasts. Not FID of pooled images; seeds are held fixed.')
        result['mean_of_seed_deltas'][variant] = summary
        print('mean-of-seed-deltas %s delta=%+.6f percentile95=%s basic95=%s' %
              (variant, summary['point_delta_fox_minus_vp'],
               summary['percentile_95'], summary['basic_95']), flush=True)


def save_output(path, result):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(
        prog='bash run_cifar10_ddpmpp_deep_600k_bootstrap_intervals.sh', description=__doc__)
    parser.add_argument('--vp-workdir', default='workdirs/cifar10_vp_ddpmpp_deep_200k_rtx_seed42')
    parser.add_argument('--fox-workdir',
                        default='workdirs/cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx_seed42')
    parser.add_argument('--vp-run-tag', help='Default: ddpmpp_deep600k_screen.')
    parser.add_argument('--fox-run-tag', help='Default: ddpmpp_deep600k_screen.')
    parser.add_argument('--sampling-seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--count', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--reference', default='assets/stats/cifar10_stats.npz')
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261027)
    parser.add_argument('--resampling', choices=('paired', 'independent'), default='paired')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--output', help='New JSON file; existing files are never overwritten.')
    args = parser.parse_args()
    require(args.count >= DIMENSION and args.batch_size > 0 and args.reps > 0,
            'Need count >= 2048, positive batch size and reps.')
    require(args.count == 5000 and args.batch_size == 512,
            'This 600k screening comparison requires 5000 samples and batch 512.')
    require(len(set(args.sampling_seeds)) == len(args.sampling_seeds)
            and all(0 <= s <= 2 ** 31 - 1 for s in args.sampling_seeds),
            'Sampling seeds must be unique and in [0, 2**31-1].')
    require(args.bootstrap_seed >= 0 and len(set(args.variants)) == len(args.variants),
            'Need nonnegative bootstrap seed and unique variants.')
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = Path(args.output or (
        'results/cifar10_ddpmpp_bootstrap/vp600k_fox600k_%dsamples_%s_%s.json' %
        (args.count, args.resampling, timestamp)))
    require(not output.exists(), 'Refusing to overwrite: ' + str(output))
    reference_id = file_hash(args.reference)
    arms = []
    for seed in args.sampling_seeds:
        pair = []
        for arm, workdir, tag in (
                ('vp', args.vp_workdir, args.vp_run_tag),
                ('fox', args.fox_workdir, args.fox_run_tag)):
            tag = tag or 'ddpmpp_deep600k_screen'
            checkpoint = CHECKPOINT
            folder = Path(workdir) / (
                'eval_cifar_endpoint_denoise_%s_%s_common_ckpt%d_nfe1000_%d'
                'samples_batch%d_seed%d' %
                (tag, arm, checkpoint, args.count, args.batch_size, seed))
            print('Checking:', folder, flush=True)
            arm_data = read_arm(folder, checkpoint, args.count, args.batch_size,
                                seed, args.variants, reference_id)
            check_ddpmpp_protocol(arm_data['manifest']['config'], arm, folder)
            pair.append(arm_data)
        check_comparison(*pair, paired=args.resampling == 'paired')
        arms.append((seed, *pair))
    # Every seed must evaluate the same fixed VP checkpoint and the same FOX checkpoint.
    for index in (1, 2):
        first = arms[0][index]
        for item in arms[1:]:
            other = item[index]
            for key in ('checkpoint_id', 'checkpoint_step', 'training_protocol_sha256'):
                require(first['report'][key] == other['report'][key],
                        'Checkpoint identity differs across sampling seeds: ' + key)
            for key in ('source_sha256', 'environment'):
                require(first['manifest'][key] == other['manifest'][key],
                        'Eval runtime differs across sampling seeds: ' + key)
    with np.load(args.reference, allow_pickle=False) as z:
        reference = np.asarray(z['pool_3'], dtype=np.float64)
    require(reference.ndim == 2 and reference.shape[1] == DIMENSION
            and reference.shape[0] > 1 and np.isfinite(reference).all(),
            'Invalid real-reference pool_3 features.')
    reference_count = len(reference)
    real = reference_moments(reference)
    del reference
    # Recompute all point FIDs before writing output or starting resampling.
    for seed, vp, fox in arms:
        for arm in (vp, fox):
            arm['recomputed_fid'] = {}
            for v in args.variants:
                value = fid(arm['pools'][v], real)
                recorded = arm['report']['fid_' + v]
                require(np.isfinite(recorded) and np.isclose(
                    value, recorded, rtol=1e-4, atol=2e-3),
                    'FID does not reproduce report: %s %s: %.8f vs %.8f' %
                    (arm['folder'], v, value, recorded))
                arm['recomputed_fid'][v] = value
                print('seed=%d %s %s: FID=%.6f (report %.6f)' %
                      (seed, 'VP' if arm is vp else 'FOX', v, value, recorded), flush=True)
    result = dict(schema_version=1, status='running', created_utc=timestamp,
                  contrast='FID(FOX) - FID(VP)', architecture='ddpmpp_deep',
                  checkpoint=CHECKPOINT, checkpoint_step=EXPECTED_STEP, config=vars(args),
                  reference_sha256=reference_id, reference_count=reference_count,
                  script_sha256=file_hash(os.environ['CIFAR_BOOTSTRAP_SCRIPT']),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   threads=os.environ.get('OMP_NUM_THREADS')),
                  interval_scope='Pointwise, fixed weights/reference; no training or selection uncertainty.',
                  limitations=['Bootstrap does not remove finite-sample FID bias.',
                               'Paired by checked round seeds/order, not explicit per-image RNG tapes.',
                               'Percentile/basic intervals are approximate; inspect bias and compare seeds.'],
                  seeds=[], mean_of_seed_deltas={})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects prior results. Later updates are atomic.
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('Output:', output, flush=True)
    print('CPU only. No generation. Negative delta favours FOX.', flush=True)
    for seed, vp, fox in arms:
        row = dict(sampling_seed=seed, completed_reps=0, variants={})
        for name, arm in (('vp', vp), ('fox', fox)):
            row[name] = {k: arm[k] for k in ('folder', 'checkpoint', 'report',
                                            'recomputed_fid', 'input_sha256', 'manifest')}
        result['seeds'].append(row)
        rng = np.random.default_rng(np.random.SeedSequence([args.bootstrap_seed, seed]))
        draws = {v: [] for v in args.variants}
        loop_started = time.monotonic()
        for rep in range(args.reps):
            vp_ids = rng.integers(0, args.count, size=args.count)
            fox_ids = (vp_ids if args.resampling == 'paired' else
                       rng.integers(0, args.count, size=args.count))
            for v in args.variants:
                draws[v].append(fid(fox['pools'][v][fox_ids], real)
                                - fid(vp['pools'][v][vp_ids], real))
            row['completed_reps'] = rep + 1
            if rep == 0 or (rep + 1) % 25 == 0 or rep + 1 == args.reps:
                for v in args.variants:
                    point = fox['recomputed_fid'][v] - vp['recomputed_fid'][v]
                    row['variants'][v] = interval_summary(draws[v], point)
                    row['variants'][v]['bootstrap_deltas'] = draws[v]
                result['elapsed_seconds'] = time.monotonic() - started
                save_output(output, result)
                elapsed = time.monotonic() - loop_started
                print('seed=%d replicate=%d/%d elapsed=%.1fs ETA=%.1fs' %
                      (seed, rep + 1, args.reps, elapsed,
                       elapsed / (rep + 1) * (args.reps - rep - 1)), flush=True)
        for v in args.variants:
            summary = row['variants'][v]
            print('seed=%d %s delta=%+.6f percentile95=%s basic95=%s' %
                  (seed, v, summary['point_delta_fox_minus_vp'],
                   summary['percentile_95'], summary['basic_95']), flush=True)
    summarize_seed_mean(result, args.variants)
    result['status'] = 'complete'
    result['elapsed_seconds'] = time.monotonic() - started
    result['timing_only'] = args.reps < 500
    if args.reps < 500:
        result['limitations'].append('Timing/smoke run: too few draws for stable 95% tail quantiles.')
        print('Timing/smoke only: fewer than 500 bootstrap replicates.', flush=True)
    save_output(output, result)
    print('Finished:', output, flush=True)


if __name__ == '__main__':
    main()
PY_BOOTSTRAP
