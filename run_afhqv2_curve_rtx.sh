#!/usr/bin/env bash
# Host launcher: AFHQv2-64, FOX, A and VP over training steps (afhqv2_curve_eval.py); the step
# rule is AFHQV2_STEP_RULE.md. Run from a clean, separate Git worktree, by default
# ~/kovshov-arzumanyan/arzumand/score_sde_afhqv2_curve_20261111.
#
#   bash run_afhqv2_curve_rtx.sh check              CPU tests and input paths; no GPU
#   bash run_afhqv2_curve_rtx.sh start GPU_ID       detached: preflight, then nine cells, each as its snapshot appears
#   bash run_afhqv2_curve_rtx.sh after-avp GPU_ID   in the background: wait until $AFHQ_CURVE_AFTER ends by
#                                                   itself with code 0 and leaves VP checkpoint_4, then start
#
# after-avp stops nothing. If the training ends with another code or without the snapshot,
# or the wait passes AFHQ_CURVE_WAIT_MAX_SECONDS, nothing starts. Logs: logs/wait_*.log.
# FOX at 800k comes about a day later; the queue waits for it on the same GPU.
# Results: logs/summary.{json,csv}. A rerun with the same output skips finished cells.
set -euo pipefail

mode="${1:?Usage: bash run_afhqv2_curve_rtx.sh check|start GPU_ID|after-avp GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AFHQ_CURVE_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${AFHQ_CURVE_MAIN:-$base/score_sde_pytorch}"
runs="${AFHQ_CURVE_RUNS:-$base/afhqv2_rtx_20261108/workdirs}"
output="${AFHQ_CURVE_OUTPUT:-$base/afhqv2_curve_20261112}"
tfhub="${AFHQ_CURVE_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${AFHQ_CURVE_IMAGE:-score-sde-rtx:latest}"
after="${AFHQ_CURVE_AFTER:-afhqv2-rtx-a-vp}"
allowed="${AFHQ_CURVE_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${AFHQ_CURVE_MIN_FREE_MIB:-60000}"
wait_seconds="${AFHQ_CURVE_WAIT_SECONDS:-60}"
wait_max_seconds="${AFHQ_CURVE_WAIT_MAX_SECONDS:-57600}"
root=/workspace/score_sde_pytorch
stats=assets/stats/afhqv2_64_cd017afab009_stats.npz
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
name=afhqv2-curve-rtx
work_of() {
  case "$1" in fox) echo afhqv2_fox_matern32_kappa1000_rtx_seed42 ;; a) echo afhqv2_A_vp_pvp_rtx_seed42 ;;
    vp) echo afhqv2_vp_rtx_seed42 ;; esac
}
last_snapshot="$runs/$(work_of vp)/checkpoints/checkpoint_4.pth"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
running() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }
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
# Those not named in AFHQ_CURVE_ALLOWED_NEIGHBOURS or in the second argument.
not_allowed() {
  local container
  while read -r container; do
    [[ -z "$container" || ",$allowed,$2," == *",$container,"* ]] || echo "$container"
  done < <(gpu_containers "$1")
}

if [[ "$mode" == _wait ]]; then
  gpu="${1:?Expected a physical GPU index.}"
  echo "$(stamp) waiting until $after ends by itself with $last_snapshot, then GPU $gpu, at most $wait_max_seconds s"
  while [[ "$(running "$after")" == true ]]; do
    (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $after still runs; nothing started"; exit 1; }
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null || echo unknown)"
  echo "$(stamp) $after stopped, exit code $code"
  [[ "$code" == 0 ]] || { echo "$(stamp) the training did not exit with 0; nothing started"; exit 1; }
  [[ -s "$last_snapshot" ]] || { echo "$(stamp) missing $last_snapshot; nothing started"; exit 1; }
  if ! bash "$source_dir/run_afhqv2_curve_rtx.sh" start "$gpu"; then
    echo "$(stamp) the start refused; nothing started"; exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in afhqv2_curve_eval.py afhqv2_rtx_train.py celeba_pair_seed.py celeba_50k_lo_eval.py celeba_followup_eval.py \
    celeba_kappa_eval.py celeba_matched35_rtx_eval.py celeba_stage0_eval.py AFHQV2_STEP_RULE.md \
    run_afhqv2_curve_rtx.sh tests/check_afhqv2_curve.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*|"$runs/"*) echo 'Output must be outside the repositories and runs.' >&2; exit 2 ;; esac
# The images are closed to this user, so only the folder can be checked here; preflight reads them.
[[ -d "$main/datasets/afhqv2-64x64" ]] || { echo "Missing input: $main/datasets/afhqv2-64x64" >&2; exit 1; }
for path in "$main/$stats" "$tfhub/$inception/saved_model.pb" "$runs/$(work_of fox)/run_manifest.json" \
    "$runs/$(work_of a)/run_manifest.json" "$runs/$(work_of vp)/run_manifest.json"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_afhqv2_curve.py
fi

[[ "$mode" == start || "$mode" == after-avp ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ $# -eq 1 && "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || {
  echo "GPU $gpu is not an RTX." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
mkdir -p "$output/logs" "$output/workdirs"
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 20 * 1024 * 1024 * 1024 )) || { echo 'Need at least 20 GiB free.' >&2; exit 1; }

if [[ "$mode" == after-avp ]]; then
  [[ "$(running "$after")" == true ]] || { echo "$after is not running; if it has finished, use start." >&2; exit 1; }
  busy="$(not_allowed "$gpu" "$after")"
  [[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_afhqv2_curve_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

busy="$(not_allowed "$gpu" "")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in AFHQ_CURVE_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

[[ -e "$output/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/workdirs/_tfhub_modules"
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p datasets/afhqv2-64x64 assets/stats workdirs logs runs/fox runs/a runs/vp .tfds .torch_extensions_rtx .cache_rtx
mounts=()
for arm in fox a vp; do
  work="$(work_of "$arm")"
  mkdir -p "$output/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$runs/$work,dst=$root/runs/$arm,readonly"
           --mount "type=bind,src=$runs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
done
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

# The container path of the data is part of the name of the statistics file.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/datasets/afhqv2-64x64,dst=$root/datasets/afhqv2-64x64,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  "${mounts[@]}" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u afhqv2_curve_eval.py preflight; python -u afhqv2_curve_eval.py queue
    } 2>&1 | tee -a logs/curve_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
