"""Paired fixed-checkpoint FID contrasts from E4/E3b saved pool_3 arrays.

The reference set is fixed. Resampling uses the same generated sample IDs for
baseline and splice. This measures finite-sample sensitivity, not training-seed
uncertainty. Exact 2048-dimensional covariance square roots are expensive;
use --reps 1 first to measure the runtime on the analysis machine.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg

VARIANTS = ('x_mean', 'tweedie_xt')


def load_pool(eval_dir, checkpoint, variant, count):
  sample_dir = Path(eval_dir) / f'ckpt_{checkpoint}'
  arrays = []
  for filename in sorted(sample_dir.glob('statistics_*.npz'),
                         key=lambda p: int(p.stem.split('_')[-1])):
    with np.load(filename) as z:
      arrays.append(np.asarray(z[f'pool_3_{variant}'], dtype=np.float64))
  if not arrays:
    raise FileNotFoundError(f'No statistics in {sample_dir}')
  x = np.concatenate(arrays, axis=0)[:count]
  if x.shape != (count, 2048):
    raise ValueError(f'Expected ({count}, 2048), got {x.shape}: {sample_dir}')
  return x


def moments(x):
  mean = x.mean(axis=0)
  centered = x - mean
  cov = centered.T @ centered / (len(x) - 1)
  return mean, cov


def fid(x, real_mean, real_trace, real_sqrt):
  mean, cov = moments(x)
  cross = real_sqrt @ cov @ real_sqrt
  eigenvalues = linalg.eigvalsh(cross, check_finite=False,
                                overwrite_a=True, driver='evr')
  return float(np.square(mean - real_mean).sum() + np.trace(cov) +
               real_trace - 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum())


def load_report(eval_dir, checkpoint):
  with np.load(Path(eval_dir) / f'report_{checkpoint}.npz') as z:
    return {key: np.asarray(z[key]).item() for key in z.files
            if np.asarray(z[key]).shape == ()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--reference', required=True,
                  help='The exact assets/stats/celeba_*_stats.npz used by E4/E3b')
  ap.add_argument('--baseline', required=True, help='E4 VP/extended eval folder')
  ap.add_argument('--splice', nargs='+', required=True, help='E3b eval folders')
  ap.add_argument('--checkpoint', default='260')
  ap.add_argument('--count', type=int, default=10000)
  ap.add_argument('--reps', type=int, default=100)
  ap.add_argument('--seed', type=int, default=20261001)
  ap.add_argument('--output', required=True)
  args = ap.parse_args()
  if args.count < 2048 or args.reps < 1:
    raise ValueError('Use at least 2048 samples and one bootstrap replicate.')
  with np.load(args.reference) as z:
    reference = np.asarray(z['pool_3'], dtype=np.float64)
  real_mean, real_cov = moments(reference)
  eigenvalues, eigenvectors = linalg.eigh(real_cov, check_finite=False)
  real_sqrt = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
  del reference, real_cov, eigenvalues, eigenvectors
  real_trace = float(np.square(real_sqrt).sum())
  base_report = load_report(args.baseline, args.checkpoint)
  splice_reports = [load_report(p, args.checkpoint) for p in args.splice]
  if any(r['dataset_stats_id'] != base_report['dataset_stats_id']
         for r in splice_reports):
    raise ValueError('Reference dataset fingerprint differs between reports.')
  if any(int(r['num_samples']) != args.count
         for r in [base_report, *splice_reports]):
    raise ValueError('Report sample count differs from --count.')
  rng = np.random.default_rng(args.seed)
  base = {v: load_pool(args.baseline, args.checkpoint, v, args.count)
          for v in VARIANTS}
  alternatives = [{v: load_pool(p, args.checkpoint, v, args.count)
                   for v in VARIANTS} for p in args.splice]
  contrasts = []
  for p, report, pools in zip(args.splice, splice_reports, alternatives):
    for variant in VARIANTS:
      contrasts.append((p, variant, report, pools[variant], base[variant]))
  draws = {f'{p}:{variant}': [] for p, variant, _, _, _ in contrasts}
  started = time.monotonic()
  for rep in range(args.reps):
    ids = rng.integers(0, args.count, size=args.count)
    baseline_fids = {v: fid(base[v][ids], real_mean, real_trace, real_sqrt)
                     for v in VARIANTS}
    for p, variant, _, alt, _ in contrasts:
      draws[f'{p}:{variant}'].append(
        fid(alt[ids], real_mean, real_trace, real_sqrt) -
        baseline_fids[variant])
    print(f'replicate {rep + 1}/{args.reps}, elapsed={time.monotonic()-started:.1f}s',
          flush=True)
  rows = []
  for p, variant, report, _, _ in contrasts:
    samples = np.asarray(draws[f'{p}:{variant}'])
    rows.append(dict(
      splice=p, variant=variant,
      point_delta=float(report[f'fid_{variant}'] - base_report[f'fid_{variant}']),
      bootstrap_p2_5=float(np.quantile(samples, .025)),
      bootstrap_p50=float(np.median(samples)),
      bootstrap_p97_5=float(np.quantile(samples, .975))))
  output = dict(count=args.count, reps=args.reps, seed=args.seed,
                reference=args.reference, baseline=args.baseline,
                reference_fingerprint=base_report['dataset_stats_id'],
                elapsed_seconds=time.monotonic()-started, contrasts=rows)
  target = Path(args.output)
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps(rows, indent=2))


if __name__ == '__main__':
  main()
