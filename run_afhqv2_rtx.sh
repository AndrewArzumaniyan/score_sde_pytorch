#!/usr/bin/env bash
# Host launcher: AFHQv2-64 on an RTX PRO 6000 (afhqv2_rtx_train.py).
# Run from a clean, separate Git worktree, by default ~/kovshov-arzumanyan/score_sde_afhqv2_20261108.
#
#   bash run_afhqv2_rtx.sh check GPU_ID             tests, data, statistics and a real 300-update smoke
#                                                   of every arm, about 15 min; prints s/step
#   bash run_afhqv2_rtx.sh queue fox GPU_ID [ITERS] detached: FOX to ITERS, 800000 by default
#   bash run_afhqv2_rtx.sh queue a-vp GPU_ID        detached: A to 200k, then VP to 200k
#   bash run_afhqv2_rtx.sh after fox GPU_ID         in the background: wait for $AFHQ_AFTER to stop, whatever
#                                                   its exit code, run the check, and if it passes start
#                                                   FOX to 800k. Logs: logs/wait_*.log, logs/check_*.log
#
# The wait gives up after AFHQ_WAIT_MAX_SECONDS. A failed check starts nothing. The speed only
# goes to the log: to stop FOX at 600k, stop the container once checkpoint_12 exists.
#
# ITERS is a multiple of 50000 and may change between runs: it is outside the training protocol.
# Stop with docker stop, remove the container, and rerun the queue to resume from the
# last 5k: the run replays its data position first. The manifests pin the source
# fingerprint: keep the same worktree.
set -euo pipefail

mode="${1:?Usage: bash run_afhqv2_rtx.sh check GPU_ID|queue fox GPU_ID [ITERS]|queue a-vp GPU_ID|after fox GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AFHQ_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${AFHQ_MAIN:-$base/score_sde_pytorch}"
output="${AFHQ_OUTPUT:-$base/afhqv2_rtx_20261108}"
image="${AFHQ_IMAGE:-score-sde-rtx:latest}"
min_free_mib="${AFHQ_MIN_FREE_MIB:-60000}"
after="${AFHQ_AFTER:-celeba-fox800-lo}"
wait_seconds="${AFHQ_WAIT_SECONDS:-60}"
wait_max_seconds="${AFHQ_WAIT_MAX_SECONDS:-28800}"
root=/workspace/score_sde_pytorch
stats=assets/stats/afhqv2_64_cd017afab009_stats.npz

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
state() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$after" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }
# Our running containers bound to this GPU: a memory threshold alone passes beside a running
# queue. Other people's containers are left out: several hold every GPU without using it, and
# the memory threshold covers the ones that do use it.
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
  echo "$(stamp) waiting for $after, then GPU $gpu, at most $wait_max_seconds s"
  while [[ "$(state)" == true ]]; do
    (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $after still runs; nothing started"; exit 1; }
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null)" || code=unknown
  check_log="$output/logs/check_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "$(stamp) $after stopped, exit code $code; the check starts, output in $check_log"
  if ! bash "$source_dir/run_afhqv2_rtx.sh" check "$gpu" > "$check_log" 2>&1; then
    echo "$(stamp) the check failed; nothing started. Its last lines:"
    tail -n 20 "$check_log"
    exit 1
  fi
  echo "$(stamp) the check passed"
  grep -E 's/step|^Statistics|^Pixels' "$check_log" || true
  if ! bash "$source_dir/run_afhqv2_rtx.sh" queue fox "$gpu" 800000; then
    echo "$(stamp) the queue refused to start; FOX is not running"
    exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

case "$mode" in
  check) gpu="${1:?Expected a physical GPU index.}" ;;
  queue) lane="${1:?Expected fox or a-vp.}"; gpu="${2:?Expected a physical GPU index.}"; iters="${3:-800000}"
    case "$lane" in
      fox) [[ "$iters" =~ ^[1-9][0-9]*$ ]] && (( iters % 50000 == 0 )) || {
             echo 'ITERS must be a multiple of 50000.' >&2; exit 2; }
           commands="python -u afhqv2_rtx_train.py train fox $iters" ;;
      a-vp) [[ $# -le 2 ]] || { echo 'a-vp takes no ITERS.' >&2; exit 2; }
            commands='python -u afhqv2_rtx_train.py train a 200000 && python -u afhqv2_rtx_train.py train vp 200000' ;;
      *) echo 'Expected fox or a-vp.' >&2; exit 2 ;;
    esac ;;
  after) lane="${1:?Expected fox.}"; gpu="${2:?Expected a physical GPU index.}"
    [[ "$lane" == fox && $# -eq 2 ]] || { echo 'Usage: after fox GPU_ID' >&2; exit 2; } ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in afhqv2_rtx_train.py celeba_pair_seed.py celeba_stage0_eval.py run_afhqv2_rtx.sh \
    tests/check_afhqv2_rtx_train.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || {
  echo "GPU $gpu is not an RTX." >&2; exit 1; }
busy="$(gpu_containers "$gpu" | grep -v -x -e "$after" || true)"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
if [[ "$mode" != after ]]; then
  [[ "$(state)" != true ]] || { echo "$after is still running; use after." >&2; exit 1; }
  # Free memory, not an empty card: other people may hold part of it.
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
  (( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
fi
# The images are closed to this user, so only the folder can be checked here; smoke reads them.
[[ -d "$main/datasets/afhqv2-64x64" && -s "$main/$stats" ]] || { echo 'Missing AFHQv2 data or statistics.' >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
commit="$(git rev-parse HEAD)"

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p datasets/afhqv2-64x64 assets/stats workdirs logs .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/logs" "$output/workdirs"

if [[ "$mode" == after ]]; then
  # A misspelt name would otherwise count as stopped and start on a busy GPU.
  [[ "$(state)" != gone ]] || { echo "No container $after; if it was removed, use check and queue." >&2; exit 1; }
  if docker inspect "afhqv2-rtx-$lane" >/dev/null 2>&1; then
    echo "Container exists: afhqv2-rtx-$lane. Inspect it and remove it before restarting." >&2; exit 1
  fi
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_afhqv2_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

# The container path of the data is part of the name of the statistics file.
common=(--gpus "device=$gpu" --ipc=host --shm-size=16g
  --mount "type=bind,src=$source_dir,dst=$root,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$root/logs"
  --mount "type=bind,src=$main/datasets/afhqv2-64x64,dst=$root/datasets/afhqv2-64x64,readonly"
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly"
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds"
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx"
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx"
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx"
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8
  --entrypoint bash)

if [[ "$mode" == check ]]; then
  docker run --rm --name "afhqv2-rtx-check" "${common[@]}" "$image_id" -c \
    'python tests/check_afhqv2_rtx_train.py && python -u afhqv2_rtx_train.py smoke'
  printf 'image_id=%s\nphysical_gpu=%s\n' "$image_id" "$gpu" > "$output/logs/check_passed_$commit.txt"
  echo "Check passed for $commit."
  exit 0
fi

[[ -s "$output/logs/check_passed_$commit.txt" ]] || {
  echo "Run the check for this commit first: $commit" >&2; exit 1; }
name="afhqv2-rtx-$lane"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 100 * 1024 * 1024 * 1024 )) || { echo 'Need at least 100 GiB free.' >&2; exit 1; }

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\ncommands=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$commit" "$commands" > "$output/logs/launch_$run_id.txt"
docker run -d --name "$name" "${common[@]}" "$image_id" -c "
  { $commands
  } 2>&1 | tee -a logs/${lane}_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
