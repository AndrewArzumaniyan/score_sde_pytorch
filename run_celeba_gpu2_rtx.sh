#!/usr/bin/env bash
# Host launcher: two CelebA cells on an RTX PRO 6000 (celeba_gpu2_eval.py).
# Run from a clean, separate Git worktree, by default ~/kovshov-arzumanyan/score_sde_gpu2_20261107.
#
#   bash run_celeba_gpu2_rtx.sh check           CPU tests on the real manifests, no GPU
#   bash run_celeba_gpu2_rtx.sh queue GPU_ID    detached queue: preflight, two cells
#   bash run_celeba_gpu2_rtx.sh after GPU_ID    in the background: wait for the container
#                                               $GPU2_AFTER to stop, whatever its exit code,
#                                               then start the queue. Log: logs/wait_*.log
#
# Before queue or after, copy Inception once; this queue does not download it:
#
#   mkdir -p ~/kovshov-arzumanyan/celeba_gpu2_20261107/workdirs
#   cp -a ~/kovshov-arzumanyan/celeba_followup_20261107/workdirs/_tfhub_modules \
#     ~/kovshov-arzumanyan/celeba_gpu2_20261107/workdirs/
#
# Checkpoint 80 appears only at the end of training, so nothing here requires it before
# the wait: preflight inside the container checks it and its step, cell by cell.
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: $GPU2_OUTPUT/logs/summary.{json,csv}
# After any change to the code, use a new GPU2_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_gpu2_rtx.sh check|queue GPU_ID|after GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${GPU2_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${GPU2_MAIN:-$base/score_sde_pytorch}"
a_run="${GPU2_A_RUN:-$base/celeba_a_400k_rtx_20261106}"
output="${GPU2_OUTPUT:-$base/celeba_gpu2_20261107}"
image="${GPU2_IMAGE:-score-sde-rtx:latest}"
name="${GPU2_CONTAINER:-celeba-gpu2-rtx}"
after="${GPU2_AFTER:-celeba-a-400k-rtx}"
wait_seconds="${GPU2_WAIT_SECONDS:-60}"
min_free_mib="${GPU2_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
a_work=celeba_factorial_A_vp_pvp_400k_rtx_seed42
a0_work=celeba_kaffine_a0_200k_seed42
k002_work=celeba_kaffine_kappa0p02_200k_seed42
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf

free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
state() { docker inspect --format '{{.State.Running}}' "$after" 2>/dev/null || echo gone; }

if [[ "$mode" == _wait ]]; then
  gpu="$1"
  echo "$(stamp) waiting for $after, then GPU $gpu"
  while [[ "$(state)" == true ]]; do
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null || echo unknown)"
  echo "$(stamp) $after stopped, exit code $code; the queue starts regardless"
  until (( "$(free_mib "$gpu")" >= min_free_mib )); do
    echo "$(stamp) GPU $gpu has $(free_mib "$gpu") MiB free, need $min_free_mib"
    sleep "$wait_seconds"
  done
  bash "$source_dir/run_celeba_gpu2_rtx.sh" queue "$gpu"
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_gpu2_eval.py celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py \
    celeba_stage0_eval.py run_celeba_gpu2_rtx.sh tests/check_celeba_gpu2.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

mkdir -p "$output/logs" "$output/workdirs/_training_inputs"
cp "$a_run/workdirs/$a_work/run_manifest.json" "$output/workdirs/_training_inputs/$a_work.json"
cp "$main/workdirs/$a0_work/run_manifest.json" "$output/workdirs/_training_inputs/$a0_work.json"
cp "$main/workdirs/$k002_work/run_manifest.json" "$output/workdirs/_training_inputs/$k002_work.json"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs/_training_inputs,dst=/inputs,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env GPU2_TEST_INPUTS=/inputs --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --entrypoint python "$image_id" tests/check_celeba_gpu2.py
fi
[[ "$mode" == queue || "$mode" == after ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$output/workdirs/_tfhub_modules/$inception/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path. For Inception, see the copy command in this script." >&2; exit 1; }
done
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 40 * 1024 * 1024 * 1024 )) || { echo 'Need at least 40 GiB free.' >&2; exit 1; }

if [[ "$mode" == after ]]; then
  # A misspelt name would otherwise count as stopped and start the queue during training.
  [[ "$(state)" != gone ]] || { echo "No container $after; if it was removed, use queue." >&2; exit 1; }
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_celeba_gpu2_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

[[ "$(state)" != true ]] || { echo "$after is still running; use after." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it, and a running
# training leaves too little. Their load only slows the cells.
free="$(free_mib "$gpu")"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
# A missing checkpoint blocks only its cell, in preflight; here it is a warning.
for path in "$a_run/workdirs/$a_work/checkpoints/checkpoint_80.pth" "$main/workdirs/$k002_work/checkpoints/checkpoint_8.pth"; do
  [[ -s "$path" ]] || echo "Warning: missing checkpoint $path; its cell will fail." >&2
done

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/workdirs/$a_work/checkpoints" "$output/workdirs/$k002_work/checkpoints"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

# Preflight exits 1 when one cell failed; the queue then runs the other and exits 1 itself.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --mount "type=bind,src=$a_run/workdirs/$a_work/checkpoints,dst=$root/workdirs/$a_work/checkpoints,readonly" \
  --mount "type=bind,src=$main/workdirs/$k002_work/checkpoints,dst=$root/workdirs/$k002_work/checkpoints,readonly" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_gpu2_eval.py preflight; python -u celeba_gpu2_eval.py queue
    } 2>&1 | tee -a logs/gpu2_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
