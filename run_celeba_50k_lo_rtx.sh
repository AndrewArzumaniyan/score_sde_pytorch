#!/usr/bin/env bash
# Host launcher: CelebA, 12 cells at 50k on the lower window (celeba_50k_lo_eval.py).
# Run from a clean, separate Git worktree, by default ~/kovshov-arzumanyan/score_sde_50k_lo_20261107.
#
#   bash run_celeba_50k_lo_rtx.sh check       CPU tests on the real manifest, no GPU
#   bash run_celeba_50k_lo_rtx.sh queue 1|2   detached queue: preflight, then the cells of that GPU
#   bash run_celeba_50k_lo_rtx.sh after 1|2   in the background: wait for the container that
#                                             holds the GPU to stop, whatever its exit code,
#                                             then start the queue. Log: gpuN/logs/wait_*.log
#
# GPU 1 runs four A cells, then four VP cells, after celeba-followup-rtx.
# GPU 2 runs four FOX cells, after celeba-gpu2-rtx.
# The two queues are independent and share no output: $LO50K_OUTPUT/gpu1 and $LO50K_OUTPUT/gpu2.
#
# Before queue or after, copy Inception once per GPU; these queues do not download it:
#
#   for lane in gpu1 gpu2; do
#     mkdir -p ~/kovshov-arzumanyan/celeba_50k_lo_20261107/$lane/workdirs
#     cp -a ~/kovshov-arzumanyan/celeba_followup_20261107/workdirs/_tfhub_modules \
#       ~/kovshov-arzumanyan/celeba_50k_lo_20261107/$lane/workdirs/
#   done
#
# Rerunning a queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: gpuN/logs/summary.{json,csv}
# After any change to the code, use a new LO50K_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_50k_lo_rtx.sh check|queue 1|2|after 1|2}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${LO50K_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${LO50K_MAIN:-$base/score_sde_pytorch}"
a_run="${LO50K_A_RUN:-$base/celeba_a_400k_rtx_20261106}"
output_root="${LO50K_OUTPUT:-$base/celeba_50k_lo_20261107}"
image="${LO50K_IMAGE:-score-sde-rtx:latest}"
wait_seconds="${LO50K_WAIT_SECONDS:-60}"
min_free_mib="${LO50K_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
a_work=celeba_factorial_A_vp_pvp_400k_rtx_seed42
vp_work=celeba_vp_continuous_50k
fox_work=celeba_fox_matern32_kappa1000_50k
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf

gpu="${1:-1}"
case "$gpu" in
  1) after="${LO50K_AFTER:-celeba-followup-rtx}"
     checkpoints=("$a_run/workdirs/$a_work/checkpoints:$a_work:80" "$main/workdirs/$vp_work/checkpoints:$vp_work:40") ;;
  2) after="${LO50K_AFTER:-celeba-gpu2-rtx}"
     checkpoints=("$main/workdirs/$fox_work/checkpoints:$fox_work:260") ;;
  *) echo 'Expected GPU 1 or 2.' >&2; exit 2 ;;
esac
lane="gpu$gpu"
name="celeba-50k-lo-$lane"
output="$output_root/$lane"

free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
state() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$after" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }

if [[ "$mode" == _wait ]]; then
  echo "$(stamp) waiting for $after, then GPU $gpu"
  while [[ "$(state)" == true ]]; do
    sleep "$wait_seconds"
  done
  code="$(docker inspect --format '{{.State.ExitCode}}' "$after" 2>/dev/null)" || code=unknown
  echo "$(stamp) $after stopped, exit code $code; the queue starts regardless"
  until (( "$(free_mib "$gpu")" >= min_free_mib )); do
    echo "$(stamp) GPU $gpu has $(free_mib "$gpu") MiB free, need $min_free_mib"
    sleep "$wait_seconds"
  done
  bash "$source_dir/run_celeba_50k_lo_rtx.sh" queue "$gpu"
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py \
    celeba_stage0_eval.py run_celeba_50k_lo_rtx.sh tests/check_celeba_50k_lo.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output_root/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

mkdir -p "$output/logs" "$output/workdirs/_training_inputs"
cp "$a_run/workdirs/$a_work/run_manifest.json" "$output/workdirs/_training_inputs/$a_work.json"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs/_training_inputs,dst=/inputs,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env LO50K_TEST_INPUTS=/inputs --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --entrypoint python "$image_id" tests/check_celeba_50k_lo.py
fi
[[ "$mode" == queue || "$mode" == after ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ $# -ge 1 ]] || { echo 'Expected GPU 1 or 2.' >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$output/workdirs/_tfhub_modules/$inception/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path. For Inception, see the copy command in this script." >&2; exit 1; }
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 60 * 1024 * 1024 * 1024 )) || { echo 'Need at least 60 GiB free.' >&2; exit 1; }

if [[ "$mode" == after ]]; then
  # A misspelt name would otherwise count as stopped and start the queue on a busy GPU.
  [[ "$(state)" != gone ]] || { echo "No container $after; if it was removed, use queue." >&2; exit 1; }
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_celeba_50k_lo_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $after in process $!. Log: $log"
  exit 0
fi

[[ "$(state)" != true ]] || { echo "$after is still running; use after." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it, and a running
# job leaves too little. Their load only slows the cells.
free="$(free_mib "$gpu")"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mounts=()
for entry in "${checkpoints[@]}"; do
  IFS=: read -r path work ckpt <<< "$entry"
  # A missing checkpoint blocks only its cells, in preflight; here it is a warning.
  [[ -s "$path/checkpoint_$ckpt.pth" ]] || echo "Warning: missing $path/checkpoint_$ckpt.pth; its cells will fail." >&2
  mkdir -p "$output/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$path,dst=$root/workdirs/$work/checkpoints,readonly")
done
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
  "${mounts[@]}" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u celeba_50k_lo_eval.py preflight $lane; python -u celeba_50k_lo_eval.py queue $lane
    } 2>&1 | tee -a logs/${lane}_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
