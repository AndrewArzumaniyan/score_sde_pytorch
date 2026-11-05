#!/usr/bin/env bash
# Host launcher for CelebA stage 0 on an RTX PRO 6000 (celeba_stage0_eval.py).
# Run from a clean, separate Git worktree. The checkout used by running trainings
# must stay as it is: a new source file there would block their resume.
#
#   bash run_celeba_stage0_rtx.sh check           CPU tests inside the RTX image
#   bash run_celeba_stage0_rtx.sh queue GPU_ID    detached queue: preflight, cells, energy
#
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: $STAGE0_OUTPUT/logs/summary.{json,csv}
# After any change to the code, use a new STAGE0_OUTPUT and recompute the cells:
# finished cells are skipped unchecked, so one output would mix two code versions.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_stage0_rtx.sh check|queue GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${STAGE0_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${STAGE0_MAIN:-$base/score_sde_pytorch}"
output="${STAGE0_OUTPUT:-$base/celeba_stage0_20261105}"
image="${STAGE0_IMAGE:-score-sde-rtx:latest}"
name="${STAGE0_CONTAINER:-celeba-stage0-rtx}"
root=/workspace/score_sde_pytorch
workdirs=(celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k
  celeba_factorial_A_vp_pvp_50k celeba_factorial_B_vp_pfox_50k
  celeba_factorial_C_normfox_pvp_50k celeba_factorial_D_normfox_pfox_50k
  celeba_factorial_E_realfox_pfox_50k celeba_wave2_F_vp_pvp_vptime_common_100k
  celeba_wave2_G_normfox_pfox_foxtime_common_100k)

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_stage0_eval.py run_celeba_stage0_rtx.sh sampling.py tests/check_celeba_stage0.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_celeba_stage0.py
fi
[[ "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
(( used < 1000 )) || { echo "GPU $gpu is busy: $used MiB." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for path in "$main/celeba/list_eval_partition.txt" "$main/assets/stats/celeba_64_0c2d89311192_stats.npz"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 60 * 1024 * 1024 * 1024 )) || { echo 'Need at least 60 GiB free.' >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/logs" "$output/workdirs/_training_inputs"
mounts=()
for work in "${workdirs[@]}"; do
  [[ -d "$main/workdirs/$work/checkpoints" ]] || { echo "Missing weights: $work" >&2; exit 1; }
  mkdir -p "$output/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$main/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
  if [[ "$work" == celeba_factorial_* || "$work" == celeba_wave2_* ]]; then
    cp "$main/workdirs/$work/run_manifest.json" "$output/workdirs/_training_inputs/$work.json"
  fi
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
    { python -u celeba_stage0_eval.py preflight &&
      { python -u celeba_stage0_eval.py queue; code=\$?
        [[ \$code == 3 ]] || python -u celeba_stage0_eval.py energy; exit \$code; }
    } 2>&1 | tee -a logs/stage0_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
