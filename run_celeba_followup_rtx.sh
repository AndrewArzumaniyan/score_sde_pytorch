#!/usr/bin/env bash
# Host launcher: CelebA follow-up cells on an RTX PRO 6000 (celeba_followup_eval.py).
# Run from a clean, separate Git worktree.
#
#   bash run_celeba_followup_rtx.sh check           CPU tests on the real manifests, no GPU
#   bash run_celeba_followup_rtx.sh queue GPU_ID    detached queue: preflight, six cells
#   bash run_celeba_followup_rtx.sh after GPU_ID    in the background: wait for the container
#                                                   $FOLLOWUP_AFTER to stop, whatever its exit
#                                                   code, then start the queue. Log: logs/wait_*.log
#
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: $FOLLOWUP_OUTPUT/logs/summary.{json,csv}
# After any change to the code, use a new FOLLOWUP_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_followup_rtx.sh check|queue GPU_ID|after GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${FOLLOWUP_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${FOLLOWUP_MAIN:-$base/score_sde_pytorch}"
a_run="${FOLLOWUP_A_RUN:-$base/celeba_a_400k_rtx_20261106}"
output="${FOLLOWUP_OUTPUT:-$base/celeba_followup_20261107}"
image="${FOLLOWUP_IMAGE:-score-sde-rtx:latest}"
name="${FOLLOWUP_CONTAINER:-celeba-followup-rtx}"
after="${FOLLOWUP_AFTER:-cifar-matched35-rtx}"
wait_seconds="${FOLLOWUP_WAIT_SECONDS:-60}"
min_free_mib="${FOLLOWUP_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
a_work=celeba_factorial_A_vp_pvp_400k_rtx_seed42
vp_work=celeba_vp_continuous_50k
a0_work=celeba_kaffine_a0_200k_seed42

free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

if [[ "$mode" == _wait ]]; then
  gpu="$1"
  echo "$(stamp) waiting for $after, then GPU $gpu"
  while [[ "$(docker inspect --format '{{.State.Running}}' "$after" 2>/dev/null || echo gone)" == true ]]; do
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null || echo unknown)"
  echo "$(stamp) $after stopped, exit code $code; the queue starts regardless"
  until (( "$(free_mib "$gpu")" >= min_free_mib )); do
    echo "$(stamp) GPU $gpu has $(free_mib "$gpu") MiB free, need $min_free_mib"
    sleep "$wait_seconds"
  done
  bash "$source_dir/run_celeba_followup_rtx.sh" queue "$gpu"
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py celeba_stage0_eval.py \
    run_celeba_followup_rtx.sh tests/check_celeba_followup.py; do
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

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs/_training_inputs,dst=/inputs,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env FOLLOWUP_TEST_INPUTS=/inputs --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --entrypoint python "$image_id" tests/check_celeba_followup.py
fi
[[ "$mode" == queue || "$mode" == after ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$a_run/workdirs/$a_work/checkpoints/checkpoint_60.pth" "$main/workdirs/$vp_work/checkpoints/checkpoint_40.pth" \
    "$main/workdirs/$a0_work/checkpoints/checkpoint_8.pth"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 40 * 1024 * 1024 * 1024 )) || { echo 'Need at least 40 GiB free.' >&2; exit 1; }

if [[ "$mode" == after ]]; then
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_celeba_followup_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

# Free memory, not an empty card: other people may hold part of it, and a running
# training leaves too little. Their load only slows the cells.
free="$(free_mib "$gpu")"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/workdirs/$a_work/checkpoints" "$output/workdirs/$vp_work/checkpoints" \
  "$output/workdirs/$a0_work/checkpoints"
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
  --mount "type=bind,src=$a_run/workdirs/$a_work/checkpoints,dst=$root/workdirs/$a_work/checkpoints,readonly" \
  --mount "type=bind,src=$main/workdirs/$vp_work/checkpoints,dst=$root/workdirs/$vp_work/checkpoints,readonly" \
  --mount "type=bind,src=$main/workdirs/$a0_work/checkpoints,dst=$root/workdirs/$a0_work/checkpoints,readonly" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_followup_eval.py preflight && python -u celeba_followup_eval.py queue
    } 2>&1 | tee -a logs/followup_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
