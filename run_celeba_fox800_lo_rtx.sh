#!/usr/bin/env bash
# Host launcher: CelebA, FOX seed 42 at 800k, 4 cells at 50k on the lower window
# (celeba_fox800_lo_eval.py). Run from a clean, separate Git worktree, by default
# ~/kovshov-arzumanyan/score_sde_fox800_20261107.
#
#   bash run_celeba_fox800_lo_rtx.sh check          CPU tests, no GPU
#   bash run_celeba_fox800_lo_rtx.sh queue GPU_ID   detached queue: preflight, then the cells
#
# Inception is copied once from the 50k queue; this queue does not download it.
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: logs/summary.{json,csv}
# After any change to the code, use a new FOX800_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_fox800_lo_rtx.sh check|queue GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${FOX800_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${FOX800_MAIN:-$base/score_sde_pytorch}"
output="${FOX800_OUTPUT:-$base/celeba_fox800_lo_20261107}"
tfhub="${FOX800_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${FOX800_IMAGE:-score-sde-rtx:latest}"
name="${FOX800_CONTAINER:-celeba-fox800-lo}"
min_free_mib="${FOX800_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
work=celeba_fox_matern32_kappa1000_50k
ckpt=160
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_fox800_lo_eval.py celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py \
    celeba_matched35_rtx_eval.py celeba_stage0_eval.py run_celeba_fox800_lo_rtx.sh tests/check_celeba_fox800_lo.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --entrypoint python "$image_id" tests/check_celeba_fox800_lo.py
fi
[[ "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
mkdir -p "$output/logs" "$output/workdirs/$work/checkpoints"
[[ -e "$output/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/workdirs/_tfhub_modules"
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$main/workdirs/$work/checkpoints/checkpoint_$ckpt.pth" \
    "$output/workdirs/_tfhub_modules/$inception/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

# Preflight exits 1 when a cell failed; the queue then runs the others and exits 1 itself.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --mount "type=bind,src=$main/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_fox800_lo_eval.py preflight; python -u celeba_fox800_lo_eval.py queue
    } 2>&1 | tee -a logs/fox800_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
