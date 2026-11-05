#!/usr/bin/env bash
# Host launcher: CelebA arm A from 200k to 400k on an RTX PRO 6000 (celeba_a_continue.py).
# Run from a clean, separate Git worktree. The checkout used by running trainings
# must stay as it is: a new source file there would block their resume.
#
#   bash run_celeba_a_continue_400k_rtx.sh check           CPU tests and real weights, no GPU, no training
#   bash run_celeba_a_continue_400k_rtx.sh queue GPU_ID    detached: import, training, final check
#
# Rerunning the queue with the same output resumes it. Remove the stopped container
# first. The run's manifest pins the source fingerprint: keep the same worktree.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_a_continue_400k_rtx.sh check|queue GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${A400K_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${A400K_MAIN:-$base/score_sde_pytorch}"
output="${A400K_OUTPUT:-$base/celeba_a_400k_rtx_20261106}"
image="${A400K_IMAGE:-score-sde-rtx:latest}"
name="${A400K_CONTAINER:-celeba-a-400k-rtx}"
root=/workspace/score_sde_pytorch
parent=celeba_factorial_A_vp_pvp_50k
work=workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_a_continue.py run_celeba_a_continue_400k_rtx.sh tests/check_celeba_a_continue.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for path in "$main/workdirs/$parent/checkpoints/checkpoint_40.pth" "$main/workdirs/$parent/run_manifest.json"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/logs" "$output/workdirs/_training_inputs" "$output/workdirs/$parent/checkpoints"
cp "$main/workdirs/$parent/run_manifest.json" "$output/workdirs/_training_inputs/$parent.json"
weights="type=bind,src=$main/workdirs/$parent/checkpoints,dst=$root/workdirs/$parent/checkpoints,readonly"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 16g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs,dst=$root/workdirs,readonly" --mount "$weights" \
    --workdir "$root" --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint bash "$image_id" -c \
    'python tests/check_celeba_a_continue.py && python celeba_a_continue.py check'
fi
[[ "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
(( used < 1000 )) || { echo "GPU $gpu is busy: $used MiB." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 80 * 1024 * 1024 * 1024 )) || { echo 'Need at least 80 GiB free.' >&2; exit 1; }

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --mount "$weights" --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_a_continue.py import &&
      python -u main.py --mode=train --config=celeba_a_continue.py --workdir=$work &&
      python -u celeba_a_continue.py verify
    } 2>&1 | tee -a logs/a400k_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
