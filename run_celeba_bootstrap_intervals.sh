#!/usr/bin/env bash
# CelebA final50k: VP, head, full, tail and native FOX; sampling seeds 0/1.
# One .sh with inline Python: .py changes would change training source hashes.
# Host entry point starts CPU-only Docker; no model loading or image generation.
set -euo pipefail

# Finalization uses the same FID implementation as the splice mode.
finalize_host() {
  [[ $# == 1 ]] || { echo 'Use --finalize --check|--run|--report.' >&2; exit 2; }
  local action="$1"
  case "$action" in --check|--run|--report) ;; *) echo 'Use --finalize --check|--run|--report.' >&2; exit 2 ;; esac
  local script repo parent root image image_id name out snapshot input common extended reference ab42 ab44 threads memory
  script="$(readlink -f "${BASH_SOURCE[0]}")"
  repo="$(dirname "$script")"
  parent="$(dirname "$repo")"
  root=/workspace/score_sde_pytorch
  image="${CELEBA_BOOTSTRAP_IMAGE:-score-sde-h200}"
  image_id="$(docker image inspect --format '{{.Id}}' "$image")"
  threads="${CELEBA_BOOTSTRAP_THREADS:-4}"
  memory="${CELEBA_BOOTSTRAP_MEMORY:-8g}"
  [[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid BLAS thread count.' >&2; exit 2; }
  local -a args=(run --init --network none --cpus "${CELEBA_BOOTSTRAP_CPUS:-4}"
    --cpu-shares 128 --memory "$memory" --memory-swap "$memory"
    --user "$(id -u):$(id -g)"
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=
    --env PYTHONDONTWRITEBYTECODE=1
    --env OMP_NUM_THREADS="$threads" --env OPENBLAS_NUM_THREADS="$threads"
    --env MKL_NUM_THREADS="$threads" --env NUMEXPR_NUM_THREADS="$threads"
    --env TFDS_DATA_DIR="${TFDS_DATA_DIR:-$root/.tfds}"
    --env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/.cache}"
    --env CELEBA_BOOTSTRAP_SCRIPT=/opt/bootstrap.sh
    --env CELEBA_FINALIZE_ACTION="${action#--}"
    --env CELEBA_FINALIZE_REPS="${CELEBA_BOOTSTRAP_REPS:-1000}"
    --mount "type=bind,src=$repo,dst=$root,readonly" --workdir "$root"
    --entrypoint bash)
  if [[ "$action" == --check ]]; then
    docker "${args[@]}" --rm --mount "type=bind,src=$script,dst=/opt/bootstrap.sh,readonly" \
      "$image_id" /opt/bootstrap.sh --inside-finalize
    return
  fi
  if [[ "$action" == --report ]]; then
    input="$(readlink -e "${CELEBA_FINALIZE_INPUT:-$parent/celeba_finalize_20261029/results}")"
    [[ -f "$input/completed.json" ]] || {
      echo 'Finalization has not completed: no completed.json. Leave the current calculation running.' >&2; exit 1; }
    out="${CELEBA_FINALIZE_OUTPUT:-$parent/celeba_interval_report}"
    name="${CELEBA_BOOTSTRAP_NAME:-celeba-interval-report}"
    args+=(--mount "type=bind,src=$input,dst=/input/run,readonly")
  else
    common="$(readlink -e "${CELEBA_FINALIZE_COMMON:-$parent/celeba_finalize_20261029/legacy/common}")"
    extended="$(readlink -e "${CELEBA_FINALIZE_EXTENDED:-$parent/celeba_finalize_20261029/legacy/extended}")"
    reference="$(readlink -e "${CELEBA_FINALIZE_REFERENCE:-$parent/e3b_final50k_splices/runtime/assets/stats/celeba_64_0c2d89311192_stats.npz}")"
    ab42="$(readlink -e "${CELEBA_FINALIZE_AB42:-$parent/ab_seed42_common10k}")"
    ab44="$(readlink -e "${CELEBA_FINALIZE_AB44:-$parent/ab_replication_seed44}")"
    for input in "$common" "$extended"; do
      [[ -f "$input/draws_seed0.jsonl" && -f "$input/draws_seed1.jsonl" ]] || {
        echo "Missing saved E4 draws: $input" >&2; exit 1; }
    done
    out="${CELEBA_FINALIZE_OUTPUT:-$parent/celeba_final_intervals}"
    name="${CELEBA_BOOTSTRAP_NAME:-celeba-final-intervals-cpu}"
    args+=(--mount "type=bind,src=$common,dst=/input/common,readonly"
      --mount "type=bind,src=$extended,dst=/input/extended,readonly"
      --mount "type=bind,src=$reference,dst=/input/reference.npz,readonly"
      --mount "type=bind,src=$ab42,dst=/input/ab42,readonly"
      --mount "type=bind,src=$ab44,dst=/input/ab44,readonly")
  fi
  out="$(realpath -m "$out")"
  for input in "$repo" "${input:-}" "${common:-}" "${extended:-}" "${ab42:-}" "${ab44:-}"; do
    [[ -n "$input" ]] || continue
    case "$out/" in "$input/"*) echo 'Output overlaps a read-only input.' >&2; exit 2 ;; esac
    case "$input/" in "$out/"*) echo 'Output contains a read-only input.' >&2; exit 2 ;; esac
  done
  if docker inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists. Inspect its status before restarting it." >&2
    exit 1
  fi
  mkdir -p "$out/runtime"
  snapshot="$out/runtime/run_celeba_bootstrap_intervals.sh"
  if [[ -e "$snapshot" ]]; then
    cmp -s "$script" "$snapshot" || { echo 'Frozen code differs; use a new output directory.' >&2; exit 1; }
  else
    cp -p -- "$script" "$snapshot"
  fi
  if [[ -s "$out/runtime/image_id.txt" ]]; then
    image_id="$(cat "$out/runtime/image_id.txt")"
    docker image inspect "$image_id" >/dev/null
  else
    printf '%s\n' "$image_id" > "$out/runtime/image_id.txt"
  fi
  docker "${args[@]}" -d --name "$name" \
    --mount "type=bind,src=$out,dst=/output" \
    --mount "type=bind,src=$snapshot,dst=/opt/bootstrap.sh,readonly" \
    "$image_id" /opt/bootstrap.sh --inside-finalize
  docker inspect "$name" > "$out/docker_inspect_start.json"
  echo "Results: $out"
  echo "Logs: docker logs -f --tail 20 $name"
}
if [[ "${1:-}" == --finalize ]]; then
  shift
  finalize_host "$@"
  exit
fi


if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash run_celeba_bootstrap_intervals.sh [--finalize --check|--run|--report]
Starts detached Docker container e3b-splice-bootstrap-celeba-cpu.
Defaults: score-sde-h200, 4 CPU / 4 GiB, 1000 repeats per seed (0 then 1).
Inputs: this repo/workdirs and ../e3b_final50k_splices (read-only).
Outputs: ../e3b_final50k_bootstrap/bootstrap_seed{0,1}.json.
Overrides: CELEBA_BOOTSTRAP_SPLICES, CELEBA_BOOTSTRAP_OUTPUT,
  CELEBA_BOOTSTRAP_NAME, CELEBA_BOOTSTRAP_IMAGE, CELEBA_BOOTSTRAP_CPUS,
  CELEBA_BOOTSTRAP_THREADS, CELEBA_BOOTSTRAP_MEMORY, CELEBA_BOOTSTRAP_REPS.
Resume an existing stopped container with docker start <name>.

--finalize --check: CPU-only checks without experiment data.
--finalize --run: common resume -> A/B seeds42/44 -> E4 endpoint/denoise.
--finalize --report: verify completed draws and export 32 contrasts; NO bootstrap.
Finalization defaults: 4 CPU / 8 GiB, no GPU/network.
Paths: CELEBA_FINALIZE_INPUT (completed results for --report),
  CELEBA_FINALIZE_OUTPUT, CELEBA_FINALIZE_COMMON, CELEBA_FINALIZE_EXTENDED,
  CELEBA_FINALIZE_REFERENCE, CELEBA_FINALIZE_AB42, CELEBA_FINALIZE_AB44.
--report defaults to ../celeba_finalize_20261029/results and writes a
separate ../celeba_interval_report. It refuses an unfinished calculation.
--run writes ../celeba_final_intervals; it does not migrate v1 runtime/cache.
Leave an already running finalization queue using its original frozen code.
HELP
  exit 0
fi

if [[ "${1:-}" != --inside-container && "${1:-}" != --inside-finalize ]]; then
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

bootstrap_python() {
  nice -n 19 python -u - "$@" <<'PY_BOOTSTRAP'
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



# Finalization shares the weighted moments and FID functions above.
EXPECTED_REFERENCE = '66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327'
OUT = Path('/output')
REPS = int(os.environ.get('CELEBA_FINALIZE_REPS', '1000'))
FINAL_SCRIPT_SHA = file_hash(os.environ['CELEBA_BOOTSTRAP_SCRIPT'])
digest = file_hash


def final_fid(x, real_mean, real_cov, real_sqrt, weights=None):
    return fid(x, real_mean, float(np.trace(real_cov)), real_sqrt, weights)

def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def scalar(value):
    a = np.asarray(value)
    return a.item() if a.shape == () else a.tolist()


def report(directory):
    paths = list(directory.glob('report_*.npz'))
    require(len(paths) == 1, 'Expected one report: ' + str(directory))
    with np.load(paths[0], allow_pickle=False) as z:
        return {k: scalar(z[k]) for k in z.files if np.asarray(z[k]).shape == ()}


def unique(base, pattern):
    paths = list(base.glob(pattern))
    require(len(paths) == 1, 'Expected one path: ' + str(base / pattern))
    return paths[0]


def e4(model, endpoint, seed):
    work = 'celeba_vp_continuous_50k' if model == 'vp' else 'celeba_fox_matern32_kappa1000_50k'
    return unique(Path('/workspace/score_sde_pytorch/workdirs')/work,
        'eval_e4_endpoint_denoise_*_%s_%s_ckpt260_nfe1000_50000samples_batch512_seed%d' % (model,endpoint,seed))


def ab(seed, arm):
    base = Path('/input/ab42/workdirs') if seed == 42 else Path('/input/ab44/eval_workdirs')
    return unique(base, '*/eval_ab_replication_seed%d_%s_common_ckpt%d_nfe1000_10000samples_batch512_seed0' %
                  (seed,arm,40 if seed == 42 else 4))


def percentiles(values):
    return np.quantile(np.asarray(values), [.025, .5, .975]).tolist()


def rows(path):
    result = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    require([r['rep'] for r in result] == list(range(len(result))), 'Noncontiguous draws: ' + str(path))
    require(len(result) <= REPS, 'Too many draws: ' + str(path))
    return result


def reference():
    source = Path('/input/reference.npz')
    require(digest(source) == EXPECTED_REFERENCE, 'Reference hash mismatch')
    target = OUT/'reference_moments.npz'
    meta = OUT/'reference.json'
    if not target.exists():
        with np.load(source, allow_pickle=False) as z:
            x = np.asarray(z['pool_3'])
        mean, cov = moments(x)
        del x
        vals, vecs = linalg.eigh(cov, check_finite=False)
        sqrt = (vecs*np.sqrt(np.maximum(vals, 0))) @ vecs.T
        np.savez(target, mean=mean, cov=cov, sqrt=sqrt)
        save(meta, dict(reference_sha256=EXPECTED_REFERENCE, moments_sha256=digest(target)))
    require(read(meta)['moments_sha256'] == digest(target), 'Changed reference moments')
    with np.load(target, allow_pickle=False) as z:
        return z['mean'], z['cov'], z['sqrt']


def features(directory, count, variant, label, ref):
    """Audit all shards and bind a float32 cache to their byte hashes."""
    cache = OUT/'cache'/(label+'__'+variant+'.npy')
    metadata = cache.with_suffix('.json')
    manifest = read(directory/'manifest.json')
    cfg = manifest['config']
    rep = report(directory)
    require(cfg['eval']['num_samples'] in (10000, 50000), 'Unexpected sample count')
    require(count <= cfg['eval']['num_samples'], 'Oversized subset')
    require(cfg['eval']['batch_size'] == 512, 'Unexpected batch size')
    require(rep['num_samples'] == cfg['eval']['num_samples'] and rep['nfe'] == 1001, 'Report count/NFE')
    require(rep['dataset_stats_id'] == EXPECTED_REFERENCE, 'Report reference')
    require(cfg['eval']['sampling_seed'] in (0, 1), 'Unexpected sampling seed')
    require(cfg['sampling']['time_grid'] == 'uniform_logsnr', 'Grid mismatch')
    require(cfg['sampling']['predictor'] == 'euler_maruyama' and cfg['sampling']['corrector'] == 'none', 'Sampler mismatch')
    require(abs(cfg['sampling']['logsnr_min']+10) < 1e-9, 'Noisy endpoint')
    if label.startswith('ab'):
        training_seed = int(label[2:4])
        arm = label[-1]
        require(cfg['seed'] == training_seed and cfg['training']['sde'] == 'vpsde', 'A/B training identity')
        require(rep['checkpoint_step'] == 200001, 'A/B checkpoint budget')
        require(cfg['training']['noise_distribution'] == ('vp_time' if arm == 'A' else 'normalized_fox_time'), 'A/B allocation')
        require(abs(cfg['sampling']['logsnr_max']-9.115429865459795) < 1e-9, 'A/B endpoint')
    folders = list(directory.glob('ckpt_*'))
    require(len(folders) == 1, 'Checkpoint folder count')
    shards = sorted(folders[0].glob('statistics_*.npz'), key=lambda p:int(p.stem.split('_')[-1]))
    n_shards = int(math.ceil(cfg['eval']['num_samples']/512))
    require([p.name for p in shards] == ['statistics_%d.npz'%i for i in range(n_shards)], 'Missing/reordered shards')
    identity = {k: rep[k] for k in ('checkpoint_id', 'checkpoint_step', 'protocol_sha256', 'source_sha256', 'training_protocol_sha256') if k in rep}
    signature = dict(directory=str(directory), count=count, variant=variant, reference=EXPECTED_REFERENCE,
                     identity=identity, manifest_sha256=digest(directory/'manifest.json'),
                     shards={p.name:digest(p) for p in shards}, fid_core_sha256=FINAL_SCRIPT_SHA)
    if metadata.exists():
        old = read(metadata)
        require(old['signature'] == signature and old['cache_sha256'] == digest(cache), 'Changed input/cache '+label)
        return np.load(cache, mmap_mode='r'), old
    require(not cache.exists(), 'Incomplete cache: '+str(cache))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix('.partial.npy')
    x = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(count, 2048))
    filled = 0
    for i, path in enumerate(shards):
        with np.load(path, allow_pickle=False) as z:
            for k, v in identity.items():
                require(k in z and scalar(z[k]) == v, 'Shard identity '+str(path)+' '+k)
            require(scalar(z['round_id']) == i and scalar(z['nfe']) == 1001 and scalar(z['num_samples']) == 512, 'Shard metadata')
            s = cfg['eval']['sampling_seed']
            expected = int.from_bytes(hashlib.sha256(('%d\0sampling-round\0%d'%(s,i)).encode()).digest()[:8], 'big') % (2**31-1)
            require(scalar(z['sampling_seed']) == expected, 'Shard RNG seed')
            block = z['pool_3_'+variant]
            require(block.shape == (512, 2048) and block.dtype == np.float32 and np.isfinite(block).all(), 'Invalid feature block')
            take = min(len(block), count-filled)
            if take:
                x[filled:filled+take] = block[:take]
                filled += take
    require(filled == count, 'Insufficient rows')
    x.flush()
    value = final_fid(x, *ref)
    target = rep['fid_'+variant] if count == rep['num_samples'] else None
    require(target is None or abs(value-target) < .001, 'FID did not reproduce report')
    del x
    temp.replace(cache)
    audit = dict(signature=signature, cache_sha256=digest(cache), point=value, report_point=target,
                 point_error=None if target is None else value-target)
    save(metadata, audit)
    print('AUDIT %s/%s n=%d FID=%.9f error=%s' % (label,variant,count,value,audit['point_error']), flush=True)
    return np.load(cache, mmap_mode='r'), audit


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
        f.flush()


def checkpoint_settings(path, config):
    config.update(numpy=np.__version__, scipy=scipy.__version__, reference=EXPECTED_REFERENCE,
                  implementation_sha256=FINAL_SCRIPT_SHA, core_sha256=FINAL_SCRIPT_SHA)
    if path.exists():
        require(read(path) == config, 'Resume settings changed: '+str(path))
    else:
        save(path, config)


def common_resume(ref):
    out = OUT/'common_resume'
    out.mkdir(exist_ok=True)
    legacy = Path('/input/common')
    checks = read(legacy/'point_checks.json')
    for seed in (0, 1):
        path = out/('draws_seed%d.jsonl'%seed)
        if not path.exists():
            path.write_bytes((legacy/path.name).read_bytes())
        completed = rows(path)
        arrays, audits = {}, {}
        for model in ('vp', 'fox'):
            arrays[model], audits[model] = features(e4(model,'common',seed),50000,'tweedie_xt','common_%s_s%d'%(model,seed),ref)
            old = next(c for c in checks if c['model']==model and c['seed']==seed)
            require(abs(old['recomputed_fid']-audits[model]['point']) < 1e-6, 'Legacy point changed')
        checkpoint_settings(out/('settings_seed%d.json'%seed), dict(seed=seed, rng_seed=20261009,
            resampling='independent VP/FOX, fixed reference; original E4 stream',
            caches={k:v['cache_sha256'] for k,v in audits.items()}, legacy_sha256=digest(legacy/path.name)))
        for r in completed:
            require(all(np.isfinite(r[k]) for k in ('vp','fox','delta')) and abs(r['delta']-(r['fox']-r['vp'])) < 1e-9,'Corrupt legacy row')
        for probe in sorted(set((0,len(completed)-1))):
            if probe < 0 or probe >= len(completed):
                continue
            rng=np.random.default_rng(np.random.SeedSequence([20261009,seed,probe]))
            for model in ('vp','fox'):
                weights=np.bincount(rng.integers(0,50000,50000),minlength=50000)
                value=final_fid(arrays[model],*ref,weights)
                require(abs(value-completed[probe][model])<1e-6,'Legacy RNG/FID replay mismatch')
        print('PASS legacy common RNG/FID replay seed%d'%seed,flush=True)
        for rep in range(len(completed), REPS):
            start=time.monotonic()
            rng=np.random.default_rng(np.random.SeedSequence([20261009,seed,rep]))
            values={}
            for model in ('vp','fox'):
                weights=np.bincount(rng.integers(0,50000,50000),minlength=50000)
                values[model]=final_fid(arrays[model],*ref,weights)
            r=dict(rep=rep,**values,delta=values['fox']-values['vp'],seconds=time.monotonic()-start)
            append(path,r); completed.append(r)
            print('COMMON seed%d %d/%d %.1fs'%(seed,len(completed),REPS,r['seconds']),flush=True)
        save(out/('summary_seed%d.json'%seed),dict(status='completed',reps=len(completed),seed=seed,
            point_delta=audits['fox']['point']-audits['vp']['point'],delta_percentiles=percentiles([r['delta'] for r in completed]),
            resampling='independent generated rows; fixed weights/reference'))


def ab_bootstrap(ref):
    for seed in (42,44):
        out=OUT/('ab_seed%d'%seed); out.mkdir(exist_ok=True)
        arrays, audits={},{}
        for arm in ('A','B'):
            arrays[arm],audits[arm]=features(ab(seed,arm),10000,'tweedie_xt','ab%d_%s'%(seed,arm),ref)
        checkpoint_settings(out/'settings.json',dict(training_seed=seed,count=10000,rng_seed=20261029,
            resampling='independent A/B generated rows; fixed reference',caches={k:v['cache_sha256'] for k,v in audits.items()}))
        path=out/'draws.jsonl'; completed=rows(path)
        for rep in range(len(completed),REPS):
            start=time.monotonic(); values={}
            for i,arm in enumerate(('A','B')):
                rng=np.random.default_rng(np.random.SeedSequence([20261029,seed,rep,i]))
                weights=np.bincount(rng.integers(0,10000,10000),minlength=10000)
                values[arm]=final_fid(arrays[arm],*ref,weights)
            r=dict(rep=rep,**values,delta=values['B']-values['A'],seconds=time.monotonic()-start)
            append(path,r); completed.append(r)
            print('AB%d %d/%d %.1fs'%(seed,len(completed),REPS,r['seconds']),flush=True)
        point=audits['B']['point']-audits['A']['point']
        qs=percentiles([r['delta'] for r in completed])
        save(out/'summary.json',dict(status='completed',reps=len(completed),training_seed=seed,count=10000,
            point_delta=point,percentile95=[qs[0],qs[2]],basic95=[2*point-qs[2],2*point-qs[0]],
            resampling='independent A/B generated rows; fixed weights/reference; not training-seed uncertainty'))


def raw_e4(ref):
    """Extend original draws with x_mean at exactly their original row counts.

    Cross-endpoint differences use the product of empirical marginal bootstrap
    laws, NOT rep-by-rep pairing: original streams are identical across endpoints
    and source evidence does not establish cross-endpoint trajectory pairing.
    """
    laws={}
    for endpoint in ('common','extended'):
        for seed in (0,1):
            out=OUT/('e4_%s_s%d'%(endpoint,seed));out.mkdir(exist_ok=True)
            oldpath=(OUT/'common_resume' if endpoint=='common' else Path('/input/extended'))/('draws_seed%d.jsonl'%seed)
            tweedie=rows(oldpath);require(len(tweedie)==REPS,'Incomplete Tweedie draws')
            arrays,audits={},{}
            replay_rng=np.random.default_rng(np.random.SeedSequence([20261009,seed,0]))
            for model in ('vp','fox'):
                arrays[model],audits[model]=features(e4(model,endpoint,seed),50000,'x_mean','%s_%s_s%d'%(endpoint,model,seed),ref)
                tweedie_array,_=features(e4(model,endpoint,seed),50000,'tweedie_xt','%s_%s_s%d'%(endpoint,model,seed),ref)
                weights=np.bincount(replay_rng.integers(0,50000,50000),minlength=50000)
                require(abs(final_fid(tweedie_array,*ref,weights)-tweedie[0][model])<1e-6,'Tweedie RNG replay mismatch')
                del tweedie_array
            checkpoint_settings(out/'settings.json',dict(endpoint=endpoint,seed=seed,rng_seed=20261009,
                tweedie_draws_sha256=digest(oldpath),caches={k:v['cache_sha256'] for k,v in audits.items()},
                resampling='paired outputs within trajectory, independent VP/FOX; original E4 counts'))
            path=out/'draws.jsonl';completed=rows(path)
            for rep in range(len(completed),REPS):
                start=time.monotonic();values={}
                rng=np.random.default_rng(np.random.SeedSequence([20261009,seed,rep]))
                for model in ('vp','fox'):
                    weights=np.bincount(rng.integers(0,50000,50000),minlength=50000)
                    values[model]=final_fid(arrays[model],*ref,weights)
                r=dict(rep=rep,vp_raw=values['vp'],fox_raw=values['fox'],
                       vp_tweedie=tweedie[rep]['vp'],fox_tweedie=tweedie[rep]['fox'],seconds=time.monotonic()-start)
                append(path,r);completed.append(r)
                print('E4 %s seed%d %d/%d %.1fs'%(endpoint,seed,len(completed),REPS,r['seconds']),flush=True)
            a=np.array([[r[k] for k in ('vp_raw','fox_raw','vp_tweedie','fox_tweedie')] for r in completed])
            laws[(endpoint,seed)]=a
            values=dict(fox_minus_vp_raw=a[:,1]-a[:,0],fox_minus_vp_tweedie=a[:,3]-a[:,2],
                        vp_denoise=a[:,2]-a[:,0],fox_denoise=a[:,3]-a[:,1],
                        denoise_gap_change=(a[:,3]-a[:,2])-(a[:,1]-a[:,0]))
            save(out/'summary.json',dict(status='completed',reps=len(completed),
                scope='fixed checkpoints/reference; nominal individual 95% intervals',
                percentiles={k:percentiles(v) for k,v in values.items()}))
    for seed in (0,1):
        common,extended=laws[('common',seed)],laws[('extended',seed)]
        result={}
        for i,k in enumerate(('vp_raw','fox_raw','vp_tweedie','fox_tweedie')):
            independent=extended[:,i,None]-common[None,:,i]
            result[k+'_extended_minus_common']=percentiles(independent)
        gap_ext=extended[:,3]-extended[:,2];gap_com=common[:,3]-common[:,2]
        result['tweedie_gap_endpoint_change']=percentiles(gap_ext[:,None]-gap_com[None,:])
        save(OUT/('endpoint_contrasts_seed%d.json'%seed),dict(status='completed',percentiles=result,
            resampling='product empirical marginal bootstrap laws, reps x reps; cross-endpoint independence',
            note='product of empirical marginal bootstrap laws, not additional bootstrap repetitions'))


def final_report(source, destination):
    """Validate completed v1/new draws and export; never evaluate feature FIDs."""
    import csv
    observed = {}

    def get(relative, jsonl=False):
        path = source / relative
        observed[relative] = digest(path)
        if jsonl:
            value = [json.loads(line) for line in path.read_text().splitlines()]
            require([r['rep'] for r in value] == list(range(REPS)), 'Incomplete/reordered draws: '+relative)
            return value
        return read(path)

    completion = get('completed.json')
    require(completion.get('status') == 'completed' and completion.get('reps') == REPS,
            'Input calculation is not complete at the requested repetition count')
    result = []
    laws, points = {}, {}

    def point(label, variant, count):
        m = get('cache/%s__%s.json' % (label, variant))
        require(m['signature']['count'] == count and m['signature']['reference'] == EXPECTED_REFERENCE,
                'Feature audit count/reference mismatch: '+label)
        require(np.isfinite(m['point']), 'Nonfinite point FID')
        return m['point']

    def add(family, seed, training_seed, name, count, value, samples, saved):
        samples = np.asarray(samples)
        require(np.isfinite(samples).all() and np.isfinite(value), 'Nonfinite contrast')
        q = percentiles(samples)
        require(np.max(np.abs(np.asarray(q)-saved)) < 1e-8, 'Saved quantiles differ: '+name)
        result.append(dict(family=family, sampling_seed=seed, training_seed=training_seed,
            contrast=name, count_per_cell=count, point=float(value), percentile_low=q[0],
            percentile_median=q[1], percentile_high=q[2],
            basic_low=2*value-q[2], basic_high=2*value-q[0]))

    for seed in (0,1):
        completed = get('common_resume/draws_seed%d.jsonl'%seed, True)
        summary = get('common_resume/summary_seed%d.json'%seed)
        require(summary['status']=='completed' and summary['reps']==REPS, 'Incomplete common summary')
        delta = np.array([r['fox']-r['vp'] for r in completed])
        require(np.isfinite(delta).all() and all(abs(r['delta']-d)<1e-9 for r,d in zip(completed,delta)), 'Common draw arithmetic')
        require(np.max(np.abs(np.asarray(percentiles(delta))-summary['delta_percentiles']))<1e-8, 'Common quantiles')
    for seed in (42,44):
        completed = get('ab_seed%d/draws.jsonl'%seed, True)
        summary = get('ab_seed%d/summary.json'%seed)
        require(summary['status']=='completed' and summary['reps']==REPS, 'Incomplete A/B summary')
        delta = np.array([r['B']-r['A'] for r in completed])
        require(all(abs(r['delta']-d)<1e-9 for r,d in zip(completed,delta)), 'A/B draw arithmetic')
        value = point('ab%d_B'%seed,'tweedie_xt',10000)-point('ab%d_A'%seed,'tweedie_xt',10000)
        require(abs(value-summary['point_delta'])<1e-9, 'A/B point contrast')
        add('controlled_ab',0,seed,'B_minus_A_200k',10000,value,delta,
            [summary['percentile95'][0],percentiles(delta)[1],summary['percentile95'][1]])
    for endpoint in ('common','extended'):
        for seed in (0,1):
            prefix='e4_%s_s%d'%(endpoint,seed)
            completed=get(prefix+'/draws.jsonl', True)
            summary=get(prefix+'/summary.json')
            require(summary['status']=='completed' and summary['reps']==REPS, 'Incomplete E4 summary')
            a=np.array([[r[k] for k in ('vp_raw','fox_raw','vp_tweedie','fox_tweedie')] for r in completed])
            require(np.isfinite(a).all(),'Nonfinite E4 draw')
            p=np.array([point('%s_%s_s%d'%(endpoint,m,seed),v,50000)
                for v in ('x_mean','tweedie_xt') for m in ('vp','fox')])
            laws[(endpoint,seed)],points[(endpoint,seed)]=a,p
            values=dict(fox_minus_vp_raw=(a[:,1]-a[:,0],p[1]-p[0]),
                fox_minus_vp_tweedie=(a[:,3]-a[:,2],p[3]-p[2]),
                vp_denoise=(a[:,2]-a[:,0],p[2]-p[0]),
                fox_denoise=(a[:,3]-a[:,1],p[3]-p[1]),
                denoise_gap_change=((a[:,3]-a[:,2])-(a[:,1]-a[:,0]),(p[3]-p[2])-(p[1]-p[0])))
            for name,(samples,value) in values.items():
                add('e4_'+endpoint,seed,None,name,50000,value,samples,summary['percentiles'][name])
    for seed in (0,1):
        summary=get('endpoint_contrasts_seed%d.json'%seed)
        require(summary['status']=='completed','Incomplete endpoint summary')
        common,extended=laws[('common',seed)],laws[('extended',seed)]
        pc,pe=points[('common',seed)],points[('extended',seed)]
        for i,label in enumerate(('vp_raw','fox_raw','vp_tweedie','fox_tweedie')):
            name=label+'_extended_minus_common'
            add('endpoint',seed,None,name,50000,pe[i]-pc[i],
                extended[:,i,None]-common[None,:,i],summary['percentiles'][name])
        name='tweedie_gap_endpoint_change'
        add('endpoint',seed,None,name,50000,(pe[3]-pe[2])-(pc[3]-pc[2]),
            (extended[:,3]-extended[:,2])[:,None]-(common[:,3]-common[:,2])[None,:],summary['percentiles'][name])
    require(len(result)==32,'Wrong final contrast count')
    destination.mkdir(parents=True,exist_ok=True)
    output=dict(status='completed',reps=REPS,contrasts=result,input_sha256=observed,
        report_script_sha256=FINAL_SCRIPT_SHA,
        scope='fixed weights and empirical reference; generated finite-sample uncertainty',
        intervals='nominal individual 95%; no multiplicity adjustment',
        endpoint_resampling='product empirical marginal laws; combinations are not new repetitions',
        note='common Tweedie FOX-VP is verified once more but not duplicated as an extra contrast')
    save(destination/'summary.json',output)
    target=destination/'intervals.csv'
    temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader();writer.writerows(result)
    temporary.replace(target)
    print('REPORT PASSED: 32 contrasts; summary.json and intervals.csv; no FID/bootstrap recomputation.',flush=True)
    return output


def final_self_test():
    import tempfile
    global REPS
    rng=np.random.default_rng(729)
    x=rng.normal(size=(20,8)).astype(np.float32)
    real=rng.normal(size=(70,8))
    mean,cov=moments(real)
    vals,vecs=linalg.eigh(cov)
    sqrt=(vecs*np.sqrt(np.maximum(vals,0)))@vecs.T
    weights=np.bincount(rng.integers(0,20,20),minlength=20)
    require(abs(final_fid(x,mean,cov,sqrt,weights)-final_fid(np.repeat(x,weights,axis=0),mean,cov,sqrt))<1e-10,
            'Weighted FID differs from explicitly repeated rows')
    old_reps=REPS;REPS=12
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'inputs';destination=base/'report'
            source.mkdir()
            try:
                final_report(source,destination)
            except (FileNotFoundError,ValueError):
                pass
            else:
                raise AssertionError('Incomplete input accepted')
            require(not destination.exists(),'Report wrote output for incomplete input')
            settings=base/'settings.json'
            checkpoint_settings(settings,dict(test=1))
            checkpoint_settings(settings,dict(test=1))
            try:
                checkpoint_settings(settings,dict(test=2))
            except ValueError:
                pass
            else:
                raise AssertionError('Changed resume settings accepted')
            # Build the same saved schema as the currently running queue.
            fixture_laws={}
            for endpoint in ('common','extended'):
                for seed in (0,1):
                    p=np.array([4.,3.7,3.,2.7])+(0.1 if endpoint=='common' else 0)+seed*.02
                    a=p+rng.normal(scale=.04,size=(REPS,4))
                    fixture_laws[(endpoint,seed)]=a
                    for i,(variant,model) in enumerate((('x_mean','vp'),('x_mean','fox'),('tweedie_xt','vp'),('tweedie_xt','fox'))):
                        save_fixture(source/'cache'/('%s_%s_s%d__%s.json'%(endpoint,model,seed,variant)),
                            dict(point=p[i],signature=dict(count=50000,reference=EXPECTED_REFERENCE)))
                    completed=[dict(rep=i,**dict(zip(('vp_raw','fox_raw','vp_tweedie','fox_tweedie'),a[i]))) for i in range(REPS)]
                    save_fixture_rows(source/('e4_%s_s%d/draws.jsonl'%(endpoint,seed)),completed)
                    values=dict(fox_minus_vp_raw=a[:,1]-a[:,0],fox_minus_vp_tweedie=a[:,3]-a[:,2],
                        vp_denoise=a[:,2]-a[:,0],fox_denoise=a[:,3]-a[:,1],
                        denoise_gap_change=(a[:,3]-a[:,2])-(a[:,1]-a[:,0]))
                    save_fixture(source/('e4_%s_s%d/summary.json'%(endpoint,seed)),dict(status='completed',reps=REPS,
                        percentiles={k:percentiles(v) for k,v in values.items()}))
            for seed in (0,1):
                a=fixture_laws[('common',seed)]
                save_fixture_rows(source/('common_resume/draws_seed%d.jsonl'%seed),
                    [dict(rep=i,vp=r[2],fox=r[3],delta=r[3]-r[2]) for i,r in enumerate(a)])
                save_fixture(source/('common_resume/summary_seed%d.json'%seed),dict(status='completed',reps=REPS,
                    delta_percentiles=percentiles(a[:,3]-a[:,2])))
                c,e=fixture_laws[('common',seed)],fixture_laws[('extended',seed)]
                values={k+'_extended_minus_common':percentiles(e[:,i,None]-c[None,:,i])
                    for i,k in enumerate(('vp_raw','fox_raw','vp_tweedie','fox_tweedie'))}
                values['tweedie_gap_endpoint_change']=percentiles((e[:,3]-e[:,2])[:,None]-(c[:,3]-c[:,2])[None,:])
                save_fixture(source/('endpoint_contrasts_seed%d.json'%seed),dict(status='completed',percentiles=values))
            for seed in (42,44):
                a=np.array([4.,3.7])+rng.normal(scale=.04,size=(REPS,2))
                for i,arm in enumerate(('A','B')):
                    save_fixture(source/('cache/ab%d_%s__tweedie_xt.json'%(seed,arm)),
                        dict(point=4.-.3*i,signature=dict(count=10000,reference=EXPECTED_REFERENCE)))
                save_fixture_rows(source/('ab_seed%d/draws.jsonl'%seed),
                    [dict(rep=i,A=r[0],B=r[1],delta=r[1]-r[0]) for i,r in enumerate(a)])
                q=percentiles(a[:,1]-a[:,0])
                save_fixture(source/('ab_seed%d/summary.json'%seed),dict(status='completed',reps=REPS,
                    point_delta=3.7-4.,percentile95=[q[0],q[2]]))
            save_fixture(source/'completed.json',dict(status='completed',reps=REPS))
            before={str(p):digest(p) for p in source.rglob('*') if p.is_file()}
            output=final_report(source,destination)
            require(len(output['contrasts'])==32,'Report count')
            require(before=={str(p):digest(p) for p in source.rglob('*') if p.is_file()},'Report modified input')
            common=fixture_laws[('common',0)][:,0]
            extended=fixture_laws[('extended',0)][:,0]
            require(not np.allclose(percentiles(extended-common),percentiles(extended[:,None]-common[None,:])),
                    'Endpoint report accidentally pairs repetition indices')
            # Tampering with saved quantiles must be rejected.
            p=source/'ab_seed42/summary.json';j=read(p);j['percentile95'][0]+=1;save(p,j)
            try:
                final_report(source,base/'bad_report')
            except ValueError:
                pass
            else:
                raise AssertionError('Corrupt summary accepted')
            require(not (base/'bad_report').exists(),'Invalid report wrote output')
    finally:
        REPS=old_reps
    print('PASS: weighted FID, resume guards, incomplete/corrupt reports, 32 contrasts, read-only inputs, independent endpoints.',flush=True)


def save_fixture(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    save(path,obj)


def save_fixture_rows(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(v)+'\n' for v in value))


def finalize_main():
    import fcntl
    require(REPS>=2,'At least two repetitions are required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void',
            'Finalization must run without a visible GPU')
    action=os.environ['CELEBA_FINALIZE_ACTION']
    if action=='check':
        final_self_test();return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action=='report':
            final_report(Path('/input/run'),OUT)
        elif action=='run':
            ref=reference()
            common_resume(ref);ab_bootstrap(ref);raw_e4(ref)
            save(OUT/'completed.json',dict(status='completed',reps=REPS,script_sha256=FINAL_SCRIPT_SHA))
        else:
            raise ValueError('Unknown finalization action')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--finalize']:
        finalize_main()
    else:
        main()

PY_BOOTSTRAP
}

if [[ "${1:-}" == --inside-finalize ]]; then
  bootstrap_python --finalize
  exit
fi

# Only reached inside the dedicated Docker container.
cd /workspace/score_sde_pytorch
for seed in 0 1; do
  suffix="ckpt260_nfe1000_50000samples_batch512_seed$seed"
  vp="workdirs/celeba_vp_continuous_50k/eval_e4_endpoint_denoise_final50k_vp_extended_$suffix"
  fox="workdirs/celeba_fox_matern32_kappa1000_50k/eval_e4_endpoint_denoise_final50k_fox_extended_$suffix"
  splice=/input/splices/workdirs/celeba_vp_continuous_50k
  echo "CelebA bootstrap: sampling seed=$seed, repeats=$CELEBA_BOOTSTRAP_REPS"
  bootstrap_python \
    --reference /input/splices/runtime/assets/stats/celeba_64_0c2d89311192_stats.npz \
    --baseline "$vp" \
    --splice "$splice/eval_e3b_splice_final50k_head-3.3_$suffix" \
             "$splice/eval_e3b_splice_final50k_full_$suffix" \
             "$splice/eval_e3b_splice_final50k_tail-3.3_$suffix" "$fox" \
    --count 50000 --reps "$CELEBA_BOOTSTRAP_REPS" --seed "$((20261023+seed))" \
    --variants tweedie_xt --resampling independent --all-pairs \
    --resume --output "/output/bootstrap_seed$seed.json"
done
echo 'CelebA bootstrap queue completed: sampling seeds 0/1.'
