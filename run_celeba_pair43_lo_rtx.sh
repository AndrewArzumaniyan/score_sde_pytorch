#!/usr/bin/env bash
# Host launcher: CelebA, A seed 43 at 400k and FOX seed 43 at 800k, 8 cells at 50k on the
# lower window (celeba_pair43_lo_eval.py). Run from a clean, separate Git worktree, by default
# ~/kovshov-arzumanyan/score_sde_pair43_eval_20261109.
#
#   bash run_celeba_pair43_lo_rtx.sh check                CPU tests, no GPU
#   bash run_celeba_pair43_lo_rtx.sh queue a|fox GPU_ID   detached queue: preflight, then the cells of that lane
#   bash run_celeba_pair43_lo_rtx.sh after a GPU_ID       in the background: wait for $PAIR43_AFTER to stop,
#                                                         then lane a, then AFHQv2 A and VP on the same GPU
#
# after a: the training must exit with code 0 and leave checkpoint_8 and its manifest; the
# lane a must exit with code 0 and a summary without failed cells. Otherwise nothing more
# starts: AFHQv2 would hold the card for about 60 h, and a rerun of this lane comes first.
# Logs: a/logs/wait_*.log. The wait for the training gives up after PAIR43_WAIT_MAX_SECONDS,
# the wait for the lane after PAIR43_EVAL_MAX_SECONDS.
#
# The lanes are independent and share no output: $PAIR43_OUTPUT/a and $PAIR43_OUTPUT/fox.
# Start each one once its checkpoint exists. Inception is copied once from the 50k queue.
# Rerunning a queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: LANE/logs/summary.{json,csv}
# After any change to the code, use a new PAIR43_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_pair43_lo_rtx.sh check|queue a|fox GPU_ID|after a GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${PAIR43_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${PAIR43_MAIN:-$base/score_sde_pytorch}"
a_run="${PAIR43_A_RUN:-$base/celeba_a43_400k_rtx_20261107}"
fox_run="${PAIR43_FOX_RUN:-$base/celeba_pair_seed43_h200_20261106}"
output_root="${PAIR43_OUTPUT:-$base/celeba_pair43_lo_20261109}"
tfhub="${PAIR43_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${PAIR43_IMAGE:-score-sde-rtx:latest}"
min_free_mib="${PAIR43_MIN_FREE_MIB:-60000}"
after="${PAIR43_AFTER:-celeba-a43-400k-rtx}"
next_launcher="${PAIR43_NEXT:-$base/score_sde_afhqv2_20261108/run_afhqv2_rtx.sh}"
wait_seconds="${PAIR43_WAIT_SECONDS:-60}"
wait_max_seconds="${PAIR43_WAIT_MAX_SECONDS:-57600}"
eval_max_seconds="${PAIR43_EVAL_MAX_SECONDS:-28800}"
root=/workspace/score_sde_pytorch
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
a_work=celeba_ab_replication_A_vp_pvp_400k_rtx_seed43

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
running() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }
exit_code() { docker inspect --format '{{.State.ExitCode}}' "$1" 2>/dev/null || echo unknown; }
# Our running containers bound to this GPU. Other people's containers are left out: several
# hold every GPU without using it, and the memory threshold covers the ones that do use it.
gpu_containers() {
  local id ids name
  for id in $(docker ps -q); do
    name="$(docker inspect --format '{{.Name}}' "$id" | tr -d /)"
    [[ "$name" =~ ^(celeba|cifar|afhqv2)- ]] || continue
    ids="$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}},{{end}}{{end}}' "$id")"
    [[ ",$ids" != *",$1,"* ]] || echo "$name"
  done
}

if [[ "$mode" == _wait ]]; then
  gpu="${1:?Expected a physical GPU index.}"
  echo "$(stamp) waiting for $after, then lane a and AFHQv2 A and VP on GPU $gpu, at most $wait_max_seconds s"
  while [[ "$(running "$after")" == true ]]; do
    (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $after still runs; nothing started"; exit 1; }
    sleep "$wait_seconds"
  done
  code="$(exit_code "$after")"
  echo "$(stamp) $after stopped, exit code $code"
  [[ "$code" == 0 ]] || { echo "$(stamp) the training did not exit with 0; nothing started"; exit 1; }
  for path in "$a_run/workdirs/$a_work/checkpoints/checkpoint_8.pth" "$a_run/workdirs/$a_work/run_manifest.json"; do
    [[ -s "$path" ]] || { echo "$(stamp) missing $path; nothing started"; exit 1; }
  done
  if ! bash "$source_dir/run_celeba_pair43_lo_rtx.sh" queue a "$gpu"; then
    echo "$(stamp) lane a refused to start; nothing started"; exit 1
  fi
  deadline=$((SECONDS + eval_max_seconds))
  while [[ "$(running celeba-pair43-lo-a)" == true ]]; do
    (( SECONDS < deadline )) || { echo "$(stamp) lane a still runs after $eval_max_seconds s; AFHQv2 not started"; exit 1; }
    sleep "$wait_seconds"
  done
  code="$(exit_code celeba-pair43-lo-a)"
  summary="$output_root/a/logs/summary.json"
  echo "$(stamp) lane a stopped, exit code $code"
  if [[ "$code" != 0 ]] || ! grep -q '"status": "completed"' "$summary" 2>/dev/null ||
      ! grep -q '"failed_cells": \[\]' "$summary"; then
    echo "$(stamp) lane a did not complete every cell; AFHQv2 not started. See $summary"; exit 1
  fi
  echo "$(stamp) lane a completed; starting AFHQv2 A and VP: $next_launcher queue a-vp $gpu"
  if ! bash "$next_launcher" queue a-vp "$gpu"; then
    echo "$(stamp) the AFHQv2 launcher refused to start"; exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_pair43_lo_eval.py celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py \
    celeba_matched35_rtx_eval.py celeba_stage0_eval.py run_celeba_pair43_lo_rtx.sh tests/check_celeba_pair43_lo.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output_root/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --entrypoint python "$image_id" tests/check_celeba_pair43_lo.py
fi

if [[ "$mode" == after ]]; then
  [[ "${1:-}" == a && $# -eq 2 ]] || { echo 'Usage: after a GPU_ID' >&2; exit 2; }
  gpu="$2"
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
  # A misspelt name would otherwise count as stopped and start on a busy GPU.
  [[ "$(running "$after")" != gone ]] || { echo "No container $after; if it was removed, use queue a." >&2; exit 1; }
  for name in celeba-pair43-lo-a afhqv2-rtx-a-vp; do
    if docker inspect "$name" >/dev/null 2>&1; then
      echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
    fi
  done
  [[ -s "$next_launcher" ]] || { echo "Missing AFHQv2 launcher: $next_launcher" >&2; exit 1; }
  busy="$(gpu_containers "$gpu" | grep -v -x -e "$after" || true)"
  [[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
  mkdir -p "$output_root/a/logs"
  pid_file="$output_root/a/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output_root/a/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_celeba_pair43_lo_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

[[ "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
lane="${1:?Expected a lane: a or fox.}"
gpu="${2:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
case "$lane" in
  a) run_dir="$a_run"; work=celeba_ab_replication_A_vp_pvp_400k_rtx_seed43; ckpt=8 ;;
  fox) run_dir="$fox_run"; work=celeba_fox_matern32_kappa1000_h200_seed43; ckpt=160 ;;
  *) echo 'Expected a lane: a or fox.' >&2; exit 2 ;;
esac
name="celeba-pair43-lo-$lane"
output="$output_root/$lane"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
for path in "$run_dir/workdirs/$work/checkpoints/checkpoint_$ckpt.pth" "$run_dir/workdirs/$work/run_manifest.json"; do
  [[ -s "$path" ]] || { echo "Missing input: $path. Has training reached this checkpoint?" >&2; exit 1; }
done
mkdir -p "$output/logs" "$output/workdirs/_training_inputs" "$output/workdirs/$work/checkpoints"
cp "$run_dir/workdirs/$work/run_manifest.json" "$output/workdirs/_training_inputs/$work.json"
[[ -e "$output/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/workdirs/_tfhub_modules"
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$output/workdirs/_tfhub_modules/$inception/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }
busy="$(gpu_containers "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

# Preflight exits 1 when a cell failed; the queue then exits 1 itself.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --mount "type=bind,src=$run_dir/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_pair43_lo_eval.py preflight $lane; python -u celeba_pair43_lo_eval.py queue $lane
    } 2>&1 | tee -a logs/${lane}_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
