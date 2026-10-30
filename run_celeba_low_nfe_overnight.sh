#!/usr/bin/env bash
# Host: bash run_celeba_low_nfe_overnight.sh GPU_ID
# CPU-only local checks: CELEBA_LOW_NFE_IMAGE=IMAGE bash ... --check
# Controlled A/B with Heun and DPM++: bash ... --ode GPU_ID|--check
set -euo pipefail

metadata() {
if [[ "${CELEBA_QUEUE_PROFILE:-em}" == ode ]]; then
  python -u celeba_ode_eval.py "$@"
  return
fi
python -u - "$@" <<'PY_LOW_NFE'
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from e3b_final50k import digest, copy_runtime, verify_runtime, REFERENCE, REFERENCE_SHA, VP, FOX

ROOT = Path('/workspace/score_sde_pytorch')
OUTPUT = Path('/output')
PIN = Path('/opt/low-nfe')
TAG = 'lownfe20261029'
VARIANTS = ('x_mean', 'x_t', 'tweedie_xt')
# Whole 250/500 comparisons precede the more aggressive 100-step screen.
CELLS = [(steps, seed, route) for steps in (250, 500, 100)
         for seed in (0, 1) for route in ('none', 'full', 'head:-3.3')]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def snapshot(source, output):
    source, output = Path(source), Path(output)
    runtime = output / 'runtime'
    before = copy_runtime(source, runtime)
    for directory in ('logs', 'workdirs', 'celeba', 'assets/stats'):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    if digest(source / REFERENCE) != REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA reference hash.')
    shutil.copy2(str(source / REFERENCE), str(runtime / REFERENCE))
    (output / 'logs').mkdir()
    (output / 'workdirs').mkdir()
    pinned_inputs = {REFERENCE: REFERENCE_SHA}
    for arm in (VP, FOX):
        relative = 'workdirs/{}/checkpoints/checkpoint_260.pth'.format(arm)
        pinned_inputs[relative] = digest(source / relative)
        (output / 'workdirs' / arm / 'checkpoints').mkdir(parents=True)
    write(output / 'snapshot.json', dict(
        source_sha256=before, inputs=pinned_inputs,
        runtime_files={str(p.relative_to(runtime)): digest(p)
                       for p in sorted(runtime.rglob('*')) if p.is_file()},
        cells=CELLS, num_samples=10000, batch_size=512, checkpoint=260,
        trajectory='VP reverse SDE', solver='Euler-Maruyama',
        logsnr_min=-10.0, logsnr_max=10.575749165617202,
        threshold=-3.3, outputs=VARIANTS, primary_output='tweedie_xt',
        estimated_gpu_hours=[12, 16], estimate_is_guarantee=False,
        cross_n_pairing_assumed=False, old_baselines_regenerated=False))
    print('Frozen runtime; 18 cells, 10k each, 5100 total trajectory calls per batch.', flush=True)


def check():
    saved = json.loads((PIN / 'snapshot.json').read_text())
    verify_runtime(ROOT, saved['source_sha256'], saved['runtime_files'], saved['inputs'])
    return saved


def preflight():
    import numpy as np
    import torch
    saved = check()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one Docker-routed CUDA GPU, CVD=0.')
    # Catch an incompatible CUDA image before loading the models or generating samples.
    probe = torch.ones((16, 16), device='cuda')
    if float((probe @ probe).mean().item()) != 16:
        raise RuntimeError('CUDA matmul probe failed.')
    identities = {}
    for arm in (VP, FOX):
        relative = 'workdirs/{}/checkpoints/checkpoint_260.pth'.format(arm)
        state = torch.load(str(ROOT / relative), map_location='cpu', weights_only=False)
        if int(state['step']) != 1300002:
            raise RuntimeError('Expected final 1.3M checkpoint: ' + arm)
        identities[arm] = state.get('checkpoint_id') or 'legacy-sha256-' + saved['inputs'][relative]
        del state
        gc.collect()
    with np.load(str(ROOT / REFERENCE)) as reference:
        if reference['pool_3'].ndim != 2 or reference['pool_3'].shape[1] != 2048 or not np.isfinite(reference['pool_3']).all():
            raise RuntimeError('Invalid reference features.')
    write(OUTPUT / 'logs/preflight.json', dict(
        status='passed', checkpoint_ids=identities,
        source_sha256=saved['source_sha256'], gpu=torch.cuda.get_device_name(0),
        started_unix=time.time()))
    print('Preflight passed: source, reference, final checkpoints, one CUDA GPU.', flush=True)


def folder(steps, seed, route):
    return ROOT / 'workdirs' / VP / (
        'eval_e3b_splice_{}_{}_ckpt260_nfe{}_10000samples_batch512_seed{}'.format(
            TAG, route.replace(':', ''), steps, seed))


def report(steps, seed, route):
    import numpy as np
    expected = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    directory = folder(steps, seed, route)
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = manifest['config']
    required_eval = dict(num_samples=10000, batch_size=512, sampling_num_scales=steps,
                         sampling_seed=seed, endpoint_denoise_variants=True,
                         splice_enabled=True, splice_mode=route.split(':')[0])
    required_sampling = dict(method='pc', predictor='euler_maruyama', corrector='none',
                             probability_flow=False, time_grid='uniform_logsnr',
                             logsnr_min=-10.0, logsnr_max=10.575749165617202,
                             noise_removal=True)
    for section, fields in (('eval', required_eval), ('sampling', required_sampling)):
        if any(cfg[section].get(k) != v for k, v in fields.items()):
            raise RuntimeError('Unexpected cell protocol: ' + str(directory))
    if route.startswith('head:') and cfg['eval']['splice_lambda'] != -3.3:
        raise RuntimeError('Wrong head threshold.')
    with np.load(str(directory / 'report_260.npz')) as result:
        required = dict(num_samples=10000, nfe=steps + 1, checkpoint_step=1300002,
                        alt_checkpoint_step=1300002, dataset_stats_id=REFERENCE_SHA,
                        checkpoint_id=expected['checkpoint_ids'][VP],
                        alt_checkpoint_id=expected['checkpoint_ids'][FOX])
        for key, value in required.items():
            if result[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        vp, fox = int(result['splice_vp_calls']), int(result['splice_fox_calls'])
        if vp + fox != steps + 1 or (route == 'none' and fox != 0) or (route == 'full' and vp != 0):
            raise RuntimeError('Incorrect network call counts.')
        if route.startswith('head:') and not (0 < vp < steps + 1 and 0 < fox < steps + 1):
            raise RuntimeError('Head did not switch networks.')
        if len(list((directory / 'ckpt_260').glob('statistics_*.npz'))) != math.ceil(10000 / 512):
            raise RuntimeError('Expected 20 statistics shards.')
        metrics = {metric + '_' + variant: float(result[metric + '_' + variant])
                   for variant in VARIANTS for metric in ('IS', 'fid', 'kid')}
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError('Non-finite metrics.')
    cell = dict(status='completed', steps=steps, nfe=steps + 1, seed=seed, route=route,
                folder=str(directory), vp_calls=vp, fox_calls=fox, **metrics)
    name = 'n{}_seed{}_{}'.format(steps, seed, route.replace(':', ''))
    write(OUTPUT / ('logs/' + name + '_result.json'), cell)
    completed = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    contrasts = []
    for n in (250, 500, 100):
        for s in (0, 1):
            cells = {c['route']: c for c in completed if (c['steps'], c['seed']) == (n, s)}
            if 'none' in cells:
                for mode in ('full', 'head:-3.3'):
                    if mode in cells:
                        contrasts.append(dict(steps=n, seed=s, contrast=mode + '-VP',
                            fid_tweedie_xt=cells[mode]['fid_tweedie_xt'] - cells['none']['fid_tweedie_xt']))
    write(OUTPUT / 'summary.json', dict(status='completed' if len(completed) == len(CELLS) else 'running',
          completed_cells=len(completed), total_cells=len(CELLS), cells=completed,
          point_contrasts=contrasts, confidence_intervals_computed=False,
          cross_n_pairing_assumed=False))
    print('Validated {}: FID Tweedie={:.6f}; {}/18 cells complete.'.format(
        name, metrics['fid_tweedie_xt'], len(completed)), flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot':
        snapshot(*args)
    elif mode == 'check':
        check()
    elif mode == 'preflight':
        preflight()
    elif mode == 'plan':
        for n, seed, route in CELLS:
            print(n, seed, route)
    elif mode == 'report':
        report(int(args[0]), int(args[1]), args[2])
    else:
        raise ValueError(mode)
PY_LOW_NFE
}

if [[ "${1:-}" == _meta ]]; then
  shift
  metadata "$@"
  exit
fi

if [[ "${1:-}" == _container ]]; then
  [[ "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || { echo 'Expected Docker-routed GPU, CVD=0.' >&2; exit 2; }
  cd /workspace/score_sde_pytorch
  exec > >(tee -a /output/logs/queue.log) 2>&1
  trap 'rc=$?; printf "%s\n" "$rc" > /output/logs/queue.exitcode; if (( rc != 0 )); then printf "failed\n" > /output/logs/queue.state; fi' EXIT
  printf 'preflight\n' > /output/logs/queue.state
  metadata preflight
  # Materialize the plan first: a failing generator must not look like an empty successful queue.
  metadata plan > /output/logs/plan.tsv
  export E3B_SPLICE_CKPT=260 E3B_SPLICE_ALT_CKPT=260
  export E3B_SPLICE_NUM_SAMPLES=10000 E3B_SPLICE_BATCH_SIZE=512
  export E3B_SPLICE_ALLOW_TF32=True E3B_SPLICE_CUDNN_BENCHMARK=True
  export E3B_SPLICE_VP_WORKDIR=workdirs/celeba_vp_continuous_50k
  export E3B_SPLICE_FOX_WORKDIR=workdirs/celeba_fox_matern32_kappa1000_50k
  while read -r steps seed route method; do
    metadata check
    cell="${method:+${method}_}n${steps}_seed${seed}_${route}"
    printf '%s\n' "$cell" > /output/logs/queue.state
    echo "=== Low NFE: $cell; $(date -u -Is) ==="
    if [[ "${CELEBA_QUEUE_PROFILE:-em}" == ode ]]; then
      metadata run "$steps" "$seed" "$route" "$method" \
        2>&1 | tee "/output/logs/${method}_n${steps}_train${seed}_${route}.log"
      metadata report "$steps" "$seed" "$route" "$method"
    else
      E3B_SPLICE_RUNS="$route" E3B_SPLICE_STEPS="$steps" E3B_SPLICE_SEED="$seed" \
        E3B_SPLICE_RUN_TAG=lownfe20261029 bash run_e3b_splice_eval.sh \
        2>&1 | tee "/output/logs/n${steps}_seed${seed}_${route/:/}.log"
      metadata report "$steps" "$seed" "$route"
    fi
  done < /output/logs/plan.tsv
  printf 'completed\n' > /output/logs/queue.state
  echo "Low-NFE queue completed at $(date -u -Is)"
  exit
fi

if [[ "${1:-}" == --ode ]]; then export CELEBA_QUEUE_PROFILE=ode; shift; fi
profile="${CELEBA_QUEUE_PROFILE:-em}"
[[ "$profile" == em || "$profile" == ode ]] || { echo 'Unknown queue profile.' >&2; exit 2; }
[[ $# == 1 ]] || { echo 'Usage: bash run_celeba_low_nfe_overnight.sh [--ode] GPU_ID|--check' >&2; exit 2; }
main="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$(basename "${BASH_SOURCE[0]}")"
root=/workspace/score_sde_pytorch
image="${CELEBA_LOW_NFE_IMAGE:-score-sde-h200}"
test_file=tests/check_low_nfe_overnight.py
if [[ "$profile" == ode ]]; then
  image="${CELEBA_ODE_IMAGE:-$image}"
  test_file=tests/check_celeba_ode.py
fi
if [[ "$1" == --check ]]; then
  exec docker run --rm --network none --cpus 2 --memory 4g \
    --mount "type=bind,src=$main,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint python "$image" "$test_file"
fi
gpu="$1"
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { echo 'Expected a physical GPU index or UUID.' >&2; exit 2; }
output="$(realpath -m "${CELEBA_LOW_NFE_OUTPUT:-$(dirname "$main")/celeba_low_nfe_overnight_20261029}")"
name="${CELEBA_LOW_NFE_CONTAINER:-celeba-low-nfe-overnight-20261029}"
data_repo="$main"
input_mounts=()
checkpoint_mounts=()
if [[ "$profile" == ode ]]; then
  data_repo="$(realpath -e "${CELEBA_ODE_INPUT_REPO:-$main}")"
  parent="$(dirname "$data_repo")"
  ab43="$(realpath -e "${CELEBA_ODE_SEED43_ROOT:-$parent/ab_replication_seed43}")"
  ab44="$(realpath -e "${CELEBA_ODE_SEED44_ROOT:-$parent/ab_replication_seed44}")"
  output="$(realpath -m "${CELEBA_ODE_OUTPUT:-$parent/celeba_controlled_ode_20261031}")"
  name="${CELEBA_ODE_CONTAINER:-celeba-controlled-ode-20261031}"
  input_mounts=(--mount "type=bind,src=$data_repo,dst=/inputs/repo,readonly"
    --mount "type=bind,src=$ab43,dst=/inputs/seed43,readonly"
    --mount "type=bind,src=$ab44,dst=/inputs/seed44,readonly")
  for seed in 42 43 44; do
    for label in A_vp_pvp B_vp_pfox; do
      if [[ "$seed" == 42 ]]; then
        origin="$data_repo"; work="celeba_factorial_${label}_50k"; ckpt=40
      else
        origin="$ab43"; [[ "$seed" != 44 ]] || origin="$ab44"
        work="celeba_ab_replication_${label}_seed${seed}_200k"; ckpt=4
      fi
      for input in "$origin/workdirs/$work/checkpoints/checkpoint_${ckpt}.pth" "$origin/workdirs/$work/run_manifest.json"; do
        [[ -s "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
      done
      checkpoint_mounts+=(--mount "type=bind,src=$origin/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
    done
  done
fi
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid container name.' >&2; exit 2; }
case "$output/" in "$main/"*|/) echo 'Output must be outside the source repository.' >&2; exit 2 ;; esac
case "$output/" in "$data_repo/"*) echo 'Output must be outside the input repository.' >&2; exit 2 ;; esac
[[ "$output" != *','* && "$main" != *','* ]] || { echo 'Bind source paths must not contain commas.' >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Output exists: $output; inspect the original job, do not overwrite." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it before restarting." >&2; exit 1
fi
tfds="${TFDS_DATA_DIR:-$root/.tfds}"
extensions="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
cache="${XDG_CACHE_HOME:-$root/.cache}"
for path in "$tfds" "$extensions" "$cache"; do
  [[ "$path" == /* && "$path" != *','* && "$path" != *'..'* ]] || { echo "Invalid cache path: $path" >&2; exit 2; }
  case "$path/" in
    "$root/.tfds/"|"$root/.torch_extensions/"|"$root/.cache/") ;;
    "$root/"*|/|/workspace/|/output/|/output/*|/opt/|/opt/*|/tmp/|/usr/*|/etc/*|/proc/*|/dev/*|/sys/*|/bin/*|/sbin/*|/lib/*|/lib64/*)
      echo "Cache overlaps runtime/system directories: $path" >&2; exit 2 ;;
  esac
done
for pair in "$tfds|$extensions" "$tfds|$cache" "$extensions|$cache"; do
  left="${pair%%|*}/"; right="${pair#*|}/"
  [[ "$left" != "$right"* && "$right" != "$left"* ]] || { echo 'Cache mounts must not overlap.' >&2; exit 2; }
done
cd "$main"
required=("$launcher" run_e3b_splice_eval.sh e3b_final50k.py main.py
  "$data_repo/assets/stats/celeba_64_0c2d89311192_stats.npz" "$data_repo/celeba/list_eval_partition.txt")
if [[ "$profile" == em ]]; then
  required+=(workdirs/celeba_vp_continuous_50k/checkpoints/checkpoint_260.pth
    workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_260.pth)
else
  required+=(celeba_ode_eval.py ode_sampling.py)
  git ls-files --error-unmatch celeba_ode_eval.py ode_sampling.py >/dev/null
fi
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
[[ -d "$data_repo/celeba/img_align_celeba" ]] || { echo 'Missing CelebA images.' >&2; exit 1; }
git ls-files --error-unmatch "$launcher" run_e3b_splice_eval.sh >/dev/null
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
# About 14 GB of uncompressed images/features, plus runtime and filesystem overhead.
parent="$(dirname "$output")"
[[ -d "$parent" ]] || { echo "Output parent must exist: $parent" >&2; exit 2; }
available="$(df -PB1 "$parent" | awk 'NR==2 {print $4}')"
minimum_gib=25; [[ "$profile" != ode ]] || minimum_gib=40
(( available >= minimum_gib * 1024 * 1024 * 1024 )) || { echo "Need at least $minimum_gib GiB free for this queue." >&2; exit 1; }
mkdir "$output"
mkdir -p "$data_repo/.tfds" "$data_repo/.torch_extensions" "$data_repo/.cache"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$main,dst=/source,readonly" \
  --mount "type=bind,src=$output,dst=/output" --workdir /source \
  "${input_mounts[@]}" --env CELEBA_QUEUE_PROFILE="$profile" \
  --env PYTHONDONTWRITEBYTECODE=1 --entrypoint bash "$image_id" \
  "/source/$launcher" _meta snapshot /source /output
git rev-parse HEAD > "$output/commit.txt"
git status --porcelain > "$output/source_status.txt"
printf 'image=%s\nphysical_gpu=%s\nTFDS_DATA_DIR=%s\nTORCH_EXTENSIONS_DIR=%s\nXDG_CACHE_HOME=%s\n' \
  "$image_id" "$gpu" "$tfds" "$extensions" "$cache" > "$output/docker.txt"
for path in "$tfds" "$extensions" "$cache"; do
  case "$path" in "$root"/*) mkdir -p "$output/runtime/${path#"$root"/}" ;; esac
done
args=(run -d --name "$name" --gpus "device=$gpu" --ipc=host
  --mount "type=bind,src=$output/runtime,dst=$root,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$root/logs"
  --mount "type=bind,src=$output,dst=/opt/low-nfe,readonly"
  --mount "type=bind,src=$output,dst=/output"
  --mount "type=bind,src=$data_repo/celeba,dst=$root/celeba,readonly"
  --mount "type=bind,src=$data_repo/.tfds,dst=$tfds"
  --mount "type=bind,src=$data_repo/.torch_extensions,dst=$extensions"
  --mount "type=bind,src=$data_repo/.cache,dst=$cache"
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env CELEBA_QUEUE_PROFILE="$profile"
  --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache"
  --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}"
  --entrypoint bash)
if [[ "$profile" == em ]]; then
  for arm in celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k; do
    checkpoint_mounts+=(--mount "type=bind,src=$main/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
  done
fi
args+=("${checkpoint_mounts[@]}")
docker "${args[@]}" "$image_id" "$launcher" _container
docker inspect "$name" > "$output/docker_inspect_start.json"
printf 'Started %s (profile=%s) on physical GPU %s. Night-sized queue; refine ETA from cell logs.\n' "$name" "$profile" "$gpu"
printf 'Results: %s\nFollow: docker logs -f --tail 60 %s\n' "$output" "$name"
