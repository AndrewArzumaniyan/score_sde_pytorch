#!/usr/bin/env bash
# Host launcher: CelebA on the lower window, native FOX at 300k-800k and the 256-call pair
# (celeba_lo_curve_eval.py). Run from a clean, separate Git worktree, by default
# ~/kovshov-arzumanyan/arzumand/score_sde_lo_curve_20261112.
#
#   bash run_celeba_lo_curve_rtx.sh check            CPU tests, no GPU
#   bash run_celeba_lo_curve_rtx.sh queue GPU_ID     detached queue: preflight, then the cells
#   bash run_celeba_lo_curve_rtx.sh after GPU_ID     in the background: wait until $LOCURVE_AFTER
#                                                    starts after this call and ends by itself with
#                                                    code 0, then queue on the same GPU
#
# after stops nothing. If the awaited container ends with another code, never starts, or the
# wait passes LOCURVE_WAIT_MAX_SECONDS, nothing starts. Logs: logs/wait_*.log under the output.
# LOCURVE_STOP_AT (Unix time) goes into the container: no cell starts that may not end before it.
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# After any change to the code, use a new LOCURVE_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_lo_curve_rtx.sh check|queue GPU_ID|after GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${LOCURVE_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${LOCURVE_MAIN:-$base/score_sde_pytorch}"
a_run="${LOCURVE_A_RUN:-$base/celeba_a_400k_rtx_20261106}"
output="${LOCURVE_OUTPUT:-$base/celeba_lo_curve_20261112}"
tfhub="${LOCURVE_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${LOCURVE_IMAGE:-score-sde-rtx:latest}"
name="${LOCURVE_CONTAINER:-celeba-lo-curve-rtx}"
after="${LOCURVE_AFTER:-celeba-pair43-lo-fox}"
allowed="${LOCURVE_ALLOWED_NEIGHBOURS:-}"
stop_at="${LOCURVE_STOP_AT:-}"
min_free_mib="${LOCURVE_MIN_FREE_MIB:-60000}"
wait_seconds="${LOCURVE_WAIT_SECONDS:-60}"
wait_max_seconds="${LOCURVE_WAIT_MAX_SECONDS:-57600}"
root=/workspace/score_sde_pytorch
fox_work=celeba_fox_matern32_kappa1000_50k
a_work=celeba_factorial_A_vp_pvp_400k_rtx_seed42
fox_ckpts=(60 80 100 120 140 160 260)
a_ckpt=80
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
running() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }
started_unix() {
  local value
  value="$(docker inspect --format '{{.State.StartedAt}}' "$1" 2>/dev/null)" || { echo 0; return; }
  date -d "$value" +%s 2>/dev/null || echo 0
}
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
not_allowed() {
  local container
  while read -r container; do
    [[ -z "$container" || ",$allowed," == *",$container,"* ]] || echo "$container"
  done < <(gpu_containers "$1")
}

if [[ "$mode" == _wait ]]; then
  gpu="${1:?Expected a physical GPU index.}"
  since="${2:?Expected the Unix time of the after call.}"
  echo "$(stamp) waiting until $after starts after $since and ends by itself with code 0, then GPU $gpu, at most $wait_max_seconds s"
  while :; do
    state="$(running "$after")"
    if [[ "$state" == false ]] && (( $(started_unix "$after") >= since )); then
      break
    fi
    (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $after did not finish a new run; nothing started"; exit 1; }
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null || echo unknown)"
  echo "$(stamp) $after stopped, exit code $code"
  [[ "$code" == 0 ]] || { echo "$(stamp) $after did not exit with 0; nothing started"; exit 1; }
  if ! bash "$source_dir/run_celeba_lo_curve_rtx.sh" queue "$gpu"; then
    echo "$(stamp) the start refused; nothing started"; exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_lo_curve_eval.py celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py \
    celeba_matched35_rtx_eval.py celeba_stage0_eval.py e3b_final50k.py run_celeba_lo_curve_rtx.sh \
    tests/check_celeba_lo_curve.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*|"$a_run/"*) echo 'Output must be outside the repositories and runs.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --entrypoint python "$image_id" tests/check_celeba_lo_curve.py
fi

[[ "$mode" == queue || "$mode" == after ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ $# -eq 1 && "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
[[ -z "$stop_at" || "$stop_at" =~ ^[0-9]+$ ]] || { echo 'LOCURVE_STOP_AT must be a Unix time.' >&2; exit 2; }
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || {
  echo "GPU $gpu is not an RTX." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
inputs=("$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt"
        "$tfhub/$inception/saved_model.pb" "$a_run/workdirs/$a_work/run_manifest.json"
        "$a_run/workdirs/$a_work/checkpoints/checkpoint_$a_ckpt.pth")
for ckpt in "${fox_ckpts[@]}"; do inputs+=("$main/workdirs/$fox_work/checkpoints/checkpoint_$ckpt.pth"); done
for path in "${inputs[@]}"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$output/logs"
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }

if [[ "$mode" == after ]]; then
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup setsid bash "$source_dir/run_celeba_lo_curve_rtx.sh" _wait "$gpu" "$(date +%s)" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for a new run of $after in process $!. Log: $log"
  exit 0
fi

busy="$(not_allowed "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in LOCURVE_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

mkdir -p "$output/workdirs/_training_inputs" "$output/workdirs/$fox_work/checkpoints" "$output/workdirs/$a_work/checkpoints"
cp "$a_run/workdirs/$a_work/run_manifest.json" "$output/workdirs/_training_inputs/$a_work.json"
[[ -e "$output/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/workdirs/_tfhub_modules"
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\nstop_at=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" "${stop_at:-none}" > "$output/logs/launch_$run_id.txt"

# Preflight exits 1 when a cell failed; then the queue does not start.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --mount "type=bind,src=$main/workdirs/$fox_work/checkpoints,dst=$root/workdirs/$fox_work/checkpoints,readonly" \
  --mount "type=bind,src=$a_run/workdirs/$a_work/checkpoints,dst=$root/workdirs/$a_work/checkpoints,readonly" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 --env LOCURVE_STOP_AT="$stop_at" \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_lo_curve_eval.py preflight && python -u celeba_lo_curve_eval.py queue
    } 2>&1 | tee -a logs/locurve_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
