#!/usr/bin/env bash
# Host launcher: CelebA A at 800k, 50k matched35 (celeba_a800k_eval.py), in case a strictly equal step
# with DIAL 800k and B 800k is asked for. Two lanes, one container and one RTX each, one shared output:
# lane s0 (sampling seed 0: DPM++ 2M, EDM Heun) and lane s1 (seed 1). Run from a clean, separate Git
# worktree, by default ~/kovshov-arzumanyan/arzumand/score_sde_celeba_a800k_20261115.
#
#   bash run_celeba_a800k_rtx.sh check              CPU tests and input paths; no GPU
#   bash run_celeba_a800k_rtx.sh start s0 GPU_ID    detached: preflight, then the 2 seed 0 cells
#   bash run_celeba_a800k_rtx.sh start s1 GPU_ID    detached: preflight, then the 2 seed 1 cells
#
# RTX only: every other 50k cell of the paper was sampled on RTX. Results: $output/logs/summary.{json,csv}
# (all four cells), logs/summary_LANE.json, logs/queue_LANE_*.log. Containers are not --rm, so exit
# codes stay inspectable. A rerun with the same output skips finished cells; remove the stopped
# container first. Bootstrap intervals: run_celeba_a800k_bootstrap.sh, started by hand.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_a800k_rtx.sh check|start s0|s1 GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${A800K_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${A800K_MAIN:-$base/score_sde_pytorch}"
output="${A800K_OUTPUT:-$base/celeba_a800k_50k_20261115}"
run="${A800K_RUN:-$base/celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42}"
curve_logs="${A800K_CURVE_LOGS:-$base/celeba_a_800k_h200_20261113/eval/logs}"
tfhub="${A800K_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${A800K_IMAGE:-score-sde-rtx:latest}"
allowed="${A800K_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${A800K_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
work=celeba_a  # The eval workdir name: the source celeba_a of gpu2_a50k_grids.py.

# Our running containers bound to this GPU. Other people's containers are left out: several
# hold every GPU without using it, and the memory threshold covers the ones that do use it.
gpu_containers() {
  local id ids container
  for id in $(docker ps -q); do
    container="$(docker inspect --format '{{.Name}}' "$id" | tr -d /)"
    [[ "$container" =~ ^(celeba|cifar|afhqv2)- ]] || continue
    ids="$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}},{{end}}{{end}}' "$id")"
    [[ ",$ids" != *",$1,"* ]] || echo "$container"
  done
}
not_allowed() {
  local container
  while read -r container; do
    [[ -z "$container" || ",$allowed," == *",$container,"* ]] || echo "$container"
  done < <(gpu_containers "$1")
}

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_a800k_eval.py gpu2_a50k_grids.py gpu2_queue_eval.py afhqv2_rtx_train.py celeba_pair_seed.py \
    celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py \
    celeba_stage0_eval.py endpoint_denoise_eval.py ode_sampling.py CELEBA_A800K_STEP_RULE.md \
    run_celeba_a800k_rtx.sh tests/check_celeba_a800k.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$tfhub/$inception/saved_model.pb" "$run/run_manifest.json" "$run/checkpoints/checkpoint_160.pth" \
    "$curve_logs/summary.json"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_celeba_a800k.py
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ $# -eq 2 ]] || { echo 'Expected a lane (s0 or s1) and one physical GPU index.' >&2; exit 2; }
lane="$1"
gpu="$2"
[[ "$lane" == s0 || "$lane" == s1 ]] || { echo "Unknown lane: $lane" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
name="celeba-a800k-$lane"
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || { echo "GPU $gpu is not an RTX." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
busy="$(not_allowed "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in A800K_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
mkdir -p "$output/logs" "$output/workdirs/_training_inputs" "$output/workdirs/$work/checkpoints"
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }

[[ -e "$output/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/workdirs/_tfhub_modules"
if [[ -e "$output/workdirs/_training_inputs/$work.json" ]]; then
  cmp -s "$run/run_manifest.json" "$output/workdirs/_training_inputs/$work.json" || {
    echo "The run manifest differs from the one of the first lane." >&2; exit 1; }
else
  cp "$run/run_manifest.json" "$output/workdirs/_training_inputs/$work.json"
fi
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba assets/stats workdirs logs acurve .tfds .torch_extensions_rtx .cache_rtx
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nlane=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$lane" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_${lane}_$run_id.txt"

docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$run/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly" \
  --mount "type=bind,src=$curve_logs,dst=$root/acurve,readonly" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 --env A800K_LANE="$lane" \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_a800k_eval.py preflight && python -u celeba_a800k_eval.py queue
    } 2>&1 | tee -a logs/queue_${lane}_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
