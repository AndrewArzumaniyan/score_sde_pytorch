#!/usr/bin/env bash
# Host launcher: CelebA row of kappa on an RTX PRO 6000 (celeba_kappa_eval.py).
# Run from a clean, separate Git worktree.
#
#   bash run_celeba_kappa_eval_rtx.sh check           CPU tests on the real manifests, no GPU
#   bash run_celeba_kappa_eval_rtx.sh queue GPU_ID    detached queue: preflight, six cells
#
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: $KAPPA_OUTPUT/logs/summary.{json,csv}
# After any change to the code, use a new KAPPA_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_kappa_eval_rtx.sh check|queue GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${KAPPA_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${KAPPA_MAIN:-$base/score_sde_pytorch}"
output="${KAPPA_OUTPUT:-$base/celeba_kappa_eval_20261106}"
image="${KAPPA_IMAGE:-score-sde-rtx:latest}"
name="${KAPPA_CONTAINER:-celeba-kappa-eval-rtx}"
min_free_mib="${KAPPA_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
workdirs=(celeba_kaffine_a0_200k_seed42 celeba_kaffine_kappa0p02_200k_seed42)

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_kappa_eval.py celeba_stage0_eval.py run_celeba_kappa_eval_rtx.sh tests/check_celeba_kappa_eval.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

mkdir -p "$output/logs" "$output/workdirs/_training_inputs"
for work in "${workdirs[@]}"; do
  cp "$main/workdirs/$work/run_manifest.json" "$output/workdirs/_training_inputs/$work.json"
done

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs/_training_inputs,dst=/inputs,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env KAPPA_TEST_INPUTS=/inputs --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_celeba_kappa_eval.py
fi
[[ "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
# Free memory, not an empty card: other people may hold part of it, and a running
# training leaves too little. Their load only slows the cells.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 20 * 1024 * 1024 * 1024 )) || { echo 'Need at least 20 GiB free.' >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mounts=()
for work in "${workdirs[@]}"; do
  for ckpt in 4 8; do
    [[ -s "$main/workdirs/$work/checkpoints/checkpoint_$ckpt.pth" ]] || {
      echo "Missing weights: $work checkpoint_$ckpt" >&2; exit 1; }
  done
  mkdir -p "$output/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$main/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
done
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  "${mounts[@]}" --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_kappa_eval.py preflight && python -u celeba_kappa_eval.py queue
    } 2>&1 | tee -a logs/kappa_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
