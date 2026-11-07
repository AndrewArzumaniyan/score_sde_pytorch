#!/usr/bin/env bash
# Host launcher: CelebA arm A, seed 43, from 200k to 400k on an RTX PRO 6000 (celeba_a43_continue.py).
# Run from a clean, separate Git worktree. The checkout used by running trainings
# must stay as it is: a new source file there would block their resume.
#
#   bash run_celeba_a43_continue_400k_rtx.sh check           CPU tests and real weights, no GPU, no training
#   bash run_celeba_a43_continue_400k_rtx.sh queue GPU_ID    detached: import, training, final check
#   bash run_celeba_a43_continue_400k_rtx.sh after GPU_ID    in the background: wait for $A43_AFTER to stop,
#                                                            whatever its exit code, then start the queue
#
# GPU 1 holds someone else's Jupyter, so the check is free memory, not an empty card.
# The H200 training of the same network and batch holds about 47 GB; 60000 MiB leaves a margin.
# The wait gives up after A43_WAIT_MAX_SECONDS without starting anything: see logs/wait_*.log.
# Rerunning the queue with the same output resumes it. Remove the stopped container
# first. The run's manifest pins the source fingerprint: keep the same worktree.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_a43_continue_400k_rtx.sh check|queue GPU_ID|after GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${A43_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${A43_MAIN:-$base/score_sde_pytorch}"  # Data and caches only; the weights are in $parent_root.
parent_root="${A43_PARENT_ROOT:-$base/ab_replication_seed43}"
output="${A43_OUTPUT:-$base/celeba_a43_400k_rtx_20261107}"
image="${A43_IMAGE:-score-sde-rtx:latest}"
name="${A43_CONTAINER:-celeba-a43-400k-rtx}"
after="${A43_AFTER:-celeba-50k-lo-gpu1}"
wait_seconds="${A43_WAIT_SECONDS:-60}"
wait_max_seconds="${A43_WAIT_MAX_SECONDS:-28800}"
min_free_mib="${A43_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
parent=celeba_ab_replication_A_vp_pvp_seed43_200k
work=workdirs/celeba_ab_replication_A_vp_pvp_400k_rtx_seed43

free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
state() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$after" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }
give_up() { (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $1; nothing started"; exit 1; }; }

if [[ "$mode" == _wait ]]; then
  gpu="${1:?Expected a physical GPU index.}"
  echo "$(stamp) waiting for $after, then GPU $gpu, at most $wait_max_seconds s"
  while [[ "$(state)" == true ]]; do
    give_up "$after still runs"
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null)" || code=unknown
  echo "$(stamp) $after stopped, exit code $code; the queue starts regardless"
  until (( "$(free_mib "$gpu")" >= min_free_mib )); do
    give_up "GPU $gpu has $(free_mib "$gpu") MiB free"
    echo "$(stamp) GPU $gpu has $(free_mib "$gpu") MiB free, need $min_free_mib"
    sleep "$wait_seconds"
  done
  bash "$source_dir/run_celeba_a43_continue_400k_rtx.sh" queue "$gpu"
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_a43_continue.py run_celeba_a43_continue_400k_rtx.sh tests/check_celeba_a43_continue.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*|"$parent_root/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for path in "$parent_root/workdirs/$parent/checkpoints/checkpoint_4.pth" "$parent_root/workdirs/$parent/run_manifest.json"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/logs" "$output/workdirs/_training_inputs" "$output/workdirs/$parent/checkpoints"
cp "$parent_root/workdirs/$parent/run_manifest.json" "$output/workdirs/_training_inputs/$parent.json"
weights="type=bind,src=$parent_root/workdirs/$parent/checkpoints,dst=$root/workdirs/$parent/checkpoints,readonly"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 16g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs,dst=$root/workdirs,readonly" --mount "$weights" \
    --workdir "$root" --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint bash "$image_id" -c \
    'python tests/check_celeba_a43_continue.py && python celeba_a43_continue.py check'
fi
[[ "$mode" == queue || "$mode" == after ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 80 * 1024 * 1024 * 1024 )) || { echo 'Need at least 80 GiB free.' >&2; exit 1; }

if [[ "$mode" == after ]]; then
  # A misspelt name would otherwise count as stopped and start the queue on a busy GPU.
  [[ "$(state)" != gone ]] || { echo "No container $after; if it was removed, use queue." >&2; exit 1; }
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_celeba_a43_continue_400k_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

[[ "$(state)" != true ]] || { echo "$after is still running; use after." >&2; exit 1; }
free="$(free_mib "$gpu")"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

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
    { python -u celeba_a43_continue.py import &&
      python -u main.py --mode=train --config=celeba_a43_continue.py --workdir=$work &&
      python -u celeba_a43_continue.py verify
    } 2>&1 | tee -a logs/a43_400k_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
