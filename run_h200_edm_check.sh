#!/usr/bin/env bash
# Host launcher: h200_edm_check_eval.py (rule of 15 November in its docstring). One lane per
# container and per H200: lane curve (CelebA 10k curves under EDM Heun for DIAL, B and A) and
# lane afhq (AFHQv2 DIAL at 250k, 50k matched35). Run from a clean, separate Git worktree.
#
#   bash run_h200_edm_check.sh check                CPU tests and input paths; no GPU
#   bash run_h200_edm_check.sh start curve GPU_ID   detached: preflight, then the 11 curve cells
#   bash run_h200_edm_check.sh start afhq GPU_ID    detached: preflight, then the 4 AFHQv2 cells
#
# Results: $output/LANE/logs/summary.{json,csv}; logs/queue_*.log. A rerun with the same output
# skips finished cells; remove the stopped container first.
set -euo pipefail

mode="${1:?Usage: bash run_h200_edm_check.sh check|start curve|afhq GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${H200EDM_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${H200EDM_MAIN:-$base/score_sde_pytorch}"
output="${H200EDM_OUTPUT:-$base/h200_edm_check_20261115}"
tfhub="${H200EDM_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${H200EDM_IMAGE:-score-sde-rtx:latest}"
allowed="${H200EDM_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${H200EDM_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
# Lane: eval workdir name | source run workdir (holds checkpoints/ and run_manifest.json) | manifest key.
sources_of() {
  if [[ "$1" == curve ]]; then
    echo "celeba_fox|$main/workdirs/celeba_fox_matern32_kappa1000_50k|"
    echo "celeba_b|$base/celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42|celeba_b"
    echo "celeba_a_first|$base/celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42|celeba_a_first"
    echo "celeba_a|$base/celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42|celeba_a"
  else
    echo "afhq_fox|$base/afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42|afhq_fox"
  fi
}
checkpoints_of() {
  if [[ "$1" == curve ]]; then
    printf '%s\n' "$main/workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_"{120,140,160}.pth \
      "$base/celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42/checkpoints/checkpoint_"{120,140,160}.pth \
      "$base/celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42/checkpoints/checkpoint_80.pth" \
      "$base/celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42/checkpoints/checkpoint_"{100,120,140,160}.pth
  else
    echo "$base/afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42/checkpoints/checkpoint_5.pth"
  fi
}

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
for path in h200_edm_check_eval.py gpu2_queue_eval.py gpu2_a50k_grids.py afhqv2_rtx_train.py celeba_pair_seed.py \
    celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py \
    celeba_stage0_eval.py endpoint_denoise_eval.py run_h200_edm_check.sh \
    tests/check_h200_edm_check.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
[[ -d "$main/datasets/afhqv2-64x64" ]] || { echo "Missing input: $main/datasets/afhqv2-64x64" >&2; exit 1; }
for path in "$main/assets/stats/afhqv2_64_cd017afab009_stats.npz" "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" \
    "$main/celeba/list_eval_partition.txt" "$tfhub/$inception/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
for lane in curve afhq; do
  while IFS='|' read -r work run key; do
    [[ -z "$key" || -s "$run/run_manifest.json" ]] || { echo "Missing input: $run/run_manifest.json" >&2; exit 1; }
  done < <(sources_of "$lane")
  while read -r path; do
    [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  done < <(checkpoints_of "$lane")
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_h200_edm_check.py
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ $# -eq 2 ]] || { echo 'Expected a lane (curve or afhq) and one physical GPU index.' >&2; exit 2; }
lane="$1"
gpu="$2"
[[ "$lane" == curve || "$lane" == afhq ]] || { echo "Unknown lane: $lane" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
if [[ "$lane" == curve ]]; then name=celeba-h200-edm-curve; else name=afhqv2-h200-dial250; fi
out="$output/$lane"
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *H200* ]] || { echo "GPU $gpu is not an H200." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
busy="$(not_allowed "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in H200EDM_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
mkdir -p "$out/logs" "$out/workdirs/_training_inputs" "$out/.torch_extensions_h200" "$out/.cache_h200"
available="$(df -PB1 "$out" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }

[[ -e "$out/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$out/workdirs/_tfhub_modules"
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba datasets/afhqv2-64x64 assets/stats workdirs logs .tfds .torch_extensions_h200 .cache_h200
mounts=()
while IFS='|' read -r work run key; do
  mkdir -p "$out/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$run/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
  [[ -z "$key" ]] || cp "$run/run_manifest.json" "$out/workdirs/_training_inputs/$key.json"
done < <(sources_of "$lane")
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nlane=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$lane" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$out/logs/launch_$run_id.txt"

# The container path of the AFHQv2 data is part of the name of its statistics file.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$out/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$out/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/datasets/afhqv2-64x64,dst=$root/datasets/afhqv2-64x64,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$out/.torch_extensions_h200,dst=$root/.torch_extensions_h200" \
  --mount "type=bind,src=$out/.cache_h200,dst=$root/.cache_h200" \
  "${mounts[@]}" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 --env H200_EDM_LANE="$lane" \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_h200" \
  --env XDG_CACHE_HOME="$root/.cache_h200" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u h200_edm_check_eval.py preflight && python -u h200_edm_check_eval.py queue
    } 2>&1 | tee -a logs/queue_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
