#!/usr/bin/env bash
# CelebA final50k: VP, head, full, tail and native FOX; sampling seeds 0/1.
# One .sh with inline Python: .py changes would change training source hashes.
# Host entry point starts CPU-only Docker; no model loading or image generation.
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash run_celeba_bootstrap_intervals.sh
Starts detached Docker container e3b-splice-bootstrap-celeba-cpu.
Defaults: score-sde-h200, 4 CPU / 4 GiB, 1000 repeats per seed (0 then 1).
Inputs: this repo/workdirs and ../e3b_final50k_splices (read-only).
Outputs: ../e3b_final50k_bootstrap/bootstrap_seed{0,1}.json.
Overrides: CELEBA_BOOTSTRAP_SPLICES, CELEBA_BOOTSTRAP_OUTPUT,
  CELEBA_BOOTSTRAP_NAME, CELEBA_BOOTSTRAP_IMAGE, CELEBA_BOOTSTRAP_CPUS,
  CELEBA_BOOTSTRAP_THREADS, CELEBA_BOOTSTRAP_MEMORY, CELEBA_BOOTSTRAP_REPS.
Resume an existing stopped container with docker start <name>.
HELP
  exit 0
fi

if [[ "${1:-}" != --inside-container ]]; then
  [[ "$#" == 0 ]] || { echo "Use --help for supported options." >&2; exit 2; }
  script="$(readlink -f "${BASH_SOURCE[0]}")"
  repo="$(dirname "$script")"
  parent="$(dirname "$repo")"
  splices="$(readlink -e "${CELEBA_BOOTSTRAP_SPLICES:-$parent/e3b_final50k_splices}")"
  out="${CELEBA_BOOTSTRAP_OUTPUT:-$parent/e3b_final50k_bootstrap}"
  name="${CELEBA_BOOTSTRAP_NAME:-e3b-splice-bootstrap-celeba-cpu}"
  cpus="${CELEBA_BOOTSTRAP_CPUS:-4}"
  threads="${CELEBA_BOOTSTRAP_THREADS:-4}"
  memory="${CELEBA_BOOTSTRAP_MEMORY:-4g}"
  image="${CELEBA_BOOTSTRAP_IMAGE:-score-sde-h200}"
  reference="$splices/runtime/assets/stats/celeba_64_0c2d89311192_stats.npz"
  [[ -f "$reference" && -d "$repo/workdirs" ]] || {
    echo "Missing reference or workdirs; check repo and splice paths." >&2; exit 1;
  }
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists. Inspect logs; resume with: docker start $name" >&2
    exit 1
  fi
  image_id="$(docker image inspect --format '{{.Id}}' "$image")"
  mkdir -p "$out/runtime"
  out="$(readlink -f "$out")"
  snapshot="$out/runtime/run_celeba_bootstrap_intervals.sh"
  cp -p "$script" "$snapshot"
  root=/workspace/score_sde_pytorch
  docker run -d --init --name "$name" --network none \
    --cpus "$cpus" --cpu-shares 128 --memory "$memory" --memory-swap "$memory" \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$repo,dst=$root,readonly" \
    --mount "type=bind,src=$splices,dst=/input/splices,readonly" \
    --mount "type=bind,src=$out,dst=/output" \
    --mount "type=bind,src=$snapshot,dst=/opt/run_celeba_bootstrap_intervals.sh,readonly" \
    --workdir "$root" \
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env CELEBA_BOOTSTRAP_SCRIPT=/opt/run_celeba_bootstrap_intervals.sh \
    --env CELEBA_BOOTSTRAP_REPS="${CELEBA_BOOTSTRAP_REPS:-1000}" \
    --env OMP_NUM_THREADS="$threads" --env OPENBLAS_NUM_THREADS="$threads" \
    --env MKL_NUM_THREADS="$threads" --env NUMEXPR_NUM_THREADS="$threads" \
    --env TFDS_DATA_DIR="${TFDS_DATA_DIR:-$root/.tfds}" \
    --env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/.cache}" \
    --entrypoint bash "$image_id" /opt/run_celeba_bootstrap_intervals.sh --inside-container
  docker inspect "$name" > "$out/docker_inspect_start.json"
  echo "Results: $out"
  echo "Logs: docker logs -f --tail 30 $name"
  exit 0
fi

# Only reached inside the dedicated Docker container.
cd /workspace/score_sde_pytorch
for seed in 0 1; do
  suffix="ckpt260_nfe1000_50000samples_batch512_seed$seed"
  vp="workdirs/celeba_vp_continuous_50k/eval_e4_endpoint_denoise_final50k_vp_extended_$suffix"
  fox="workdirs/celeba_fox_matern32_kappa1000_50k/eval_e4_endpoint_denoise_final50k_fox_extended_$suffix"
  splice=/input/splices/workdirs/celeba_vp_continuous_50k
  echo "CelebA bootstrap: sampling seed=$seed, repeats=$CELEBA_BOOTSTRAP_REPS"
  nice -n 19 python -u - \
    --reference /input/splices/runtime/assets/stats/celeba_64_0c2d89311192_stats.npz \
    --baseline "$vp" \
    --splice "$splice/eval_e3b_splice_final50k_head-3.3_$suffix" \
             "$splice/eval_e3b_splice_final50k_full_$suffix" \
             "$splice/eval_e3b_splice_final50k_tail-3.3_$suffix" "$fox" \
    --count 50000 --reps "$CELEBA_BOOTSTRAP_REPS" --seed "$((20261023+seed))" \
    --variants tweedie_xt --resampling independent --all-pairs \
    --resume --output "/output/bootstrap_seed$seed.json" <<'PY_BOOTSTRAP'
"""Fixed-checkpoint FID contrasts from E4/E3b saved pool_3 arrays.

The reference set is fixed. Paired resampling remains the default; independent
resampling is available when cross-model pairing is unproven. This measures
generated finite-sample sensitivity, not training-seed uncertainty.
Run inside Docker; no models, TensorFlow, Torch or image generation are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import scipy
from scipy import linalg

VARIANTS = ('x_mean', 'tweedie_xt')


def load_pool(eval_dir, checkpoint, variant, count):
  sample_dir = Path(eval_dir) / f'ckpt_{checkpoint}'
  x = np.empty((count, 2048), dtype=np.float32)
  filled = 0
  for filename in sorted(sample_dir.glob('statistics_*.npz'),
                         key=lambda p: int(p.stem.split('_')[-1])):
    with np.load(filename, allow_pickle=False) as z:
      block = np.asarray(z[f'pool_3_{variant}'])
      if block.ndim != 2 or block.shape[1] != 2048 or not np.isfinite(block).all():
        raise ValueError(f'Invalid Inception features: {filename}')
      take = min(len(block), count-filled)
      x[filled:filled+take] = block[:take]
      filled += take
    if filled == count:
      break
  if filled != count:
    raise ValueError(f'Expected {count} features, got {filled}: {sample_dir}')
  return x


def moments(x, weights=None):
  count = len(x) if weights is None else int(weights.sum())
  total = np.zeros(x.shape[1], dtype=np.float64)
  for start in range(0, len(x), 1024):
    block = np.asarray(x[start:start+1024], dtype=np.float64)
    total += block.sum(0) if weights is None else weights[start:start+len(block)] @ block
  mean = total / count
  cov = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
  for start in range(0, len(x), 1024):
    block = np.asarray(x[start:start+1024], dtype=np.float64) - mean
    if weights is not None:
      block *= np.sqrt(weights[start:start+len(block)])[:, None]
    cov += block.T @ block
  cov /= count - 1
  return mean, cov


def fid(x, real_mean, real_trace, real_sqrt, weights=None):
  mean, cov = moments(x, weights)
  cross = real_sqrt @ cov @ real_sqrt
  eigenvalues = linalg.eigvalsh((cross + cross.T) * 0.5, check_finite=False,
                                overwrite_a=True, driver='evr')
  return float(np.square(mean - real_mean).sum() + np.trace(cov) +
               real_trace - 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum())


def load_report(eval_dir, checkpoint):
  with np.load(Path(eval_dir) / f'report_{checkpoint}.npz') as z:
    return {key: np.asarray(z[key]).item() for key in z.files
            if np.asarray(z[key]).shape == ()}


def save(target, output):
  temporary = target.with_name(target.name + '.tmp')
  temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
  temporary.replace(target)


def file_hash(path):
  value = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      value.update(block)
  return value.hexdigest()


def intervals(draws, point, **labels):
  values = np.asarray(draws)
  return dict(point_delta=float(point),
              bootstrap_p2_5=float(np.quantile(values, .025)),
              bootstrap_p50=float(np.median(values)),
              bootstrap_p97_5=float(np.quantile(values, .975)), **labels)


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
  ap.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
  ap.add_argument('--resampling', choices=('paired', 'independent'), default='paired')
  ap.add_argument('--all-pairs', action='store_true', help='Also compare alternatives to each other.')
  ap.add_argument('--resume', action='store_true', help='Continue saved completed repeats in --output.')
  ap.add_argument('--output', required=True)
  args = ap.parse_args()
  if args.count < 2048 or args.reps < 1 or args.seed < 0:
    raise ValueError('Use at least 2048 samples and one bootstrap replicate.')
  if len(set(args.variants)) != len(args.variants):
    raise ValueError('Repeated variants are not allowed.')
  target = Path(args.output)
  if target.exists() and not args.resume:
    raise ValueError('Output exists; use --resume to continue it.')
  variants = args.variants
  with np.load(args.reference) as z:
    reference = np.asarray(z['pool_3'], dtype=np.float32)
  if reference.ndim != 2 or reference.shape[1] != 2048 or not np.isfinite(reference).all():
    raise ValueError('Invalid real-reference features.')
  real_mean, real_cov = moments(reference)
  eigenvalues, eigenvectors = linalg.eigh(real_cov, check_finite=False)
  real_sqrt = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
  del reference, real_cov, eigenvalues, eigenvectors
  real_trace = float(np.square(real_sqrt).sum())
  base_report = load_report(args.baseline, args.checkpoint)
  splice_reports = [load_report(p, args.checkpoint) for p in args.splice]
  reference_hash = file_hash(args.reference)
  if reference_hash != base_report['dataset_stats_id']:
    raise ValueError('Reference file fingerprint differs from report.')
  if any(r['dataset_stats_id'] != base_report['dataset_stats_id']
         for r in splice_reports):
    raise ValueError('Reference dataset fingerprint differs between reports.')
  if any(int(r['num_samples']) != args.count
         for r in [base_report, *splice_reports]):
    raise ValueError('Report sample count differs from --count.')
  rng = np.random.default_rng(args.seed)
  base = {v: load_pool(args.baseline, args.checkpoint, v, args.count)
          for v in variants}
  alternatives = [{v: load_pool(p, args.checkpoint, v, args.count)
                   for v in variants} for p in args.splice]
  pools = [base, *alternatives]
  reports = [base_report, *splice_reports]
  folders = [args.baseline, *args.splice]
  for folder, report, model in zip(folders, reports, pools):
    for variant in variants:
      value = fid(model[variant], real_mean, real_trace, real_sqrt)
      if not np.isfinite(value) or abs(value-report[f'fid_{variant}']) > 0.001:
        raise ValueError(f'FID does not reproduce report: {folder}, {variant}')
  signature = dict(reference_sha256=reference_hash, folders=folders,
                   checkpoint=args.checkpoint, count=args.count, seed=args.seed,
                   variants=variants, resampling=args.resampling, all_pairs=args.all_pairs,
                   script_sha256=file_hash(os.environ["CELEBA_BOOTSTRAP_SCRIPT"]), numpy_version=np.__version__,
                   scipy_version=scipy.__version__,
                   features=[{v: hashlib.sha256(memoryview(model[v])).hexdigest()
                              for v in variants} for model in pools],
                   reports_sha256=[file_hash(Path(p) / f'report_{args.checkpoint}.npz') for p in folders])
  fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
  output = dict(count=args.count, reps=0, requested_reps=args.reps, seed=args.seed,
                reference=args.reference, baseline=args.baseline, splice=args.splice,
                reference_fingerprint=reference_hash, input_signature=signature,
                input_fingerprint=fingerprint, status='running', completed_reps=0,
                fid_draws={v: [] for v in variants},
                scope='fixed weights/reference; generated finite-sample uncertainty')
  if target.exists():
    output = json.loads(target.read_text())
    if output.get('input_fingerprint') != fingerprint:
      raise ValueError('Inputs, options or script changed; cannot resume old draws.')
    rng.bit_generator.state = output['rng_state']
  completed = output['completed_reps']
  if completed > args.reps or any(len(output['fid_draws'][v]) != completed for v in variants):
    raise ValueError('Invalid saved draw count or requested repeats below completed count.')
  output.update(status='running', requested_reps=args.reps)
  target.parent.mkdir(parents=True, exist_ok=True)
  previous_elapsed = output.get('elapsed_seconds', 0.0)
  started = time.monotonic()
  for rep in range(completed, args.reps):
    ids = rng.integers(0, args.count, size=args.count)
    weights = [np.bincount(ids, minlength=args.count)]
    for _ in alternatives:
      weights.append(weights[0] if args.resampling == 'paired' else
                     np.bincount(rng.integers(0, args.count, size=args.count), minlength=args.count))
    for variant in variants:
      values = [fid(model[variant], real_mean, real_trace, real_sqrt, w)
                for model, w in zip(pools, weights)]
      if not np.isfinite(values).all():
        raise ValueError('Non-finite bootstrap FID.')
      output['fid_draws'][variant].append(values)
    output.update(completed_reps=rep+1, reps=rep+1, rng_state=rng.bit_generator.state,
                  elapsed_seconds=previous_elapsed+time.monotonic()-started)
    # Atomic update preserves the last complete repeat without a separate file.
    save(target, output)
    elapsed = time.monotonic()-started
    remaining = elapsed/(rep+1-completed) * (args.reps-rep-1)
    print(f'replicate {rep + 1}/{args.reps}, elapsed={elapsed:.1f}s, ETA={remaining/3600:.2f}h',
          flush=True)
  rows, pairs = [], []
  for variant in variants:
    values = np.asarray(output['fid_draws'][variant])
    for index, (p, report) in enumerate(zip(args.splice, splice_reports), start=1):
      rows.append(intervals(values[:, index]-values[:, 0],
                            report[f'fid_{variant}']-base_report[f'fid_{variant}'],
                            splice=p, variant=variant))
    if args.all_pairs:
      for left in range(1, len(pools)):
        for right in range(left+1, len(pools)):
          pairs.append(intervals(values[:, left]-values[:, right],
                                 reports[left][f'fid_{variant}']-reports[right][f'fid_{variant}'],
                                 left=folders[left], right=folders[right], variant=variant))
  output.update(status='completed', contrasts=rows, pairwise_contrasts=pairs)
  save(target, output)
  print(json.dumps(rows, indent=2))
  if pairs:
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
  main()
PY_BOOTSTRAP
done
echo 'CelebA bootstrap queue completed: sampling seeds 0/1.'
