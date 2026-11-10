#!/usr/bin/env bash
# Host launcher: CelebA arm B, seed 42, from 200k to 800k on the H200 (celeba_b_continue.py),
# with its curve evaluated alongside on the same card (celeba_b_curve_eval.py).
# Run from a clean, separate Git worktree, by default ~/kovshov-arzumanyan/arzumand/score_sde_b800k_20261110.
#
#   bash run_celeba_b_continue_800k_h200.sh check            CPU tests, real B weights on CPU, input paths; no GPU
#   bash run_celeba_b_continue_800k_h200.sh start GPU_ID     detached: training and evaluation queue now
#   bash run_celeba_b_continue_800k_h200.sh after-vp GPU_ID  in the background: wait for VP seed 43 checkpoint_80
#                                                            (400k), stop $B800K_PAIR, then start
#
# after-vp stops the pair container only once checkpoint_80 exists and its size has not
# changed over one poll; the snapshot is renamed into place, so it is complete. VP seed 43
# can resume later from its last 5k by rerunning its own queue. Logs: logs/wait_*.log.
# The wait gives up after B800K_WAIT_MAX_SECONDS and then starts nothing.
#
# The image is the RTX one, as for lane fox of the pair on this card: sm 9.0 and caches of
# its own under the output, so the RTX caches of running jobs stay as they are.
# Training: logs/b800k_*.log. Curve: eval/logs/summary.csv. A rerun resumes both.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_b_continue_800k_h200.sh check|start GPU_ID|after-vp GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${B800K_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${B800K_MAIN:-$base/score_sde_pytorch}"
output="${B800K_OUTPUT:-$base/celeba_b_800k_h200_20261111}"
tfhub="${B800K_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${B800K_IMAGE:-score-sde-rtx:latest}"
pair="${B800K_PAIR:-celeba-pair-seed43-h200}"
vp_checkpoint="${B800K_VP_CHECKPOINT:-$base/celeba_pair_seed43_h200_20261106/workdirs/celeba_vp_continuous_h200_seed43/checkpoints/checkpoint_80.pth}"
allowed="${B800K_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${B800K_MIN_FREE_MIB:-100000}"
wait_seconds="${B800K_WAIT_SECONDS:-60}"
wait_max_seconds="${B800K_WAIT_MAX_SECONDS:-108000}"
root=/workspace/score_sde_pytorch
source_work=celeba_factorial_B_vp_pfox_50k
work=celeba_factorial_B_vp_pfox_800k_h200_seed42
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
train_name=celeba-b-800k-h200
eval_name=celeba-b-curve-h200

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# Docker 29 prints an empty line for an unknown name, so only the exit code tells.
running() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }
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
# Those not named in B800K_ALLOWED_NEIGHBOURS or in the second argument.
not_allowed() {
  local name
  while read -r name; do
    [[ -z "$name" || ",$allowed,$2," == *",$name,"* ]] || echo "$name"
  done < <(gpu_containers "$1")
}
size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

if [[ "$mode" == _wait ]]; then
  gpu="${1:?Expected a physical GPU index.}"
  echo "$(stamp) waiting for $vp_checkpoint, then stopping $pair and starting B on GPU $gpu, at most $wait_max_seconds s"
  previous=""
  while :; do
    if [[ -s "$vp_checkpoint" ]]; then
      current="$(size "$vp_checkpoint")"
      [[ "$current" != "$previous" ]] || break
      previous="$current"
    elif [[ "$(running "$pair")" != true ]]; then
      echo "$(stamp) $pair stopped before $vp_checkpoint appeared; nothing stopped or started"; exit 1
    fi
    (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s for $vp_checkpoint; nothing stopped or started"; exit 1; }
    sleep "$wait_seconds"
  done
  echo "$(stamp) $vp_checkpoint is in place, $previous bytes"
  if [[ "$(running "$pair")" == true ]]; then
    docker stop -t 120 "$pair" >/dev/null || true
  fi
  [[ "$(running "$pair")" != true ]] || { echo "$(stamp) $pair did not stop; B not started"; exit 1; }
  echo "$(stamp) $pair stopped, exit code $(docker inspect --format '{{.State.ExitCode}}' "$pair" 2>/dev/null || echo unknown)"
  if ! bash "$source_dir/run_celeba_b_continue_800k_h200.sh" start "$gpu"; then
    echo "$(stamp) the start refused; B is not running"; exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_b_continue.py celeba_b_curve_eval.py celeba_50k_lo_eval.py celeba_followup_eval.py \
    celeba_kappa_eval.py celeba_matched35_rtx_eval.py celeba_stage0_eval.py run_celeba_b_continue_800k_h200.sh \
    tests/check_celeba_b_continue.py tests/check_celeba_b_curve.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for path in "$main/workdirs/$source_work/checkpoints/checkpoint_40.pth" "$main/workdirs/$source_work/run_manifest.json" \
    "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$tfhub/$inception/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats continuation .tfds .torch_extensions_h200 .cache_h200
mkdir -p "$output/logs" "$output/workdirs/_training_inputs" "$output/workdirs/$source_work/checkpoints" \
  "$output/workdirs/$work/checkpoints" "$output/eval/logs" "$output/eval/workdirs/_training_inputs" \
  "$output/eval/workdirs/$source_work/checkpoints" "$output/eval/workdirs/$work/checkpoints" \
  "$output/.torch_extensions_h200" "$output/.cache_h200" "$output/eval/.torch_extensions_h200" "$output/eval/.cache_h200"
cp "$main/workdirs/$source_work/run_manifest.json" "$output/workdirs/_training_inputs/$source_work.json"
cp "$main/workdirs/$source_work/run_manifest.json" "$output/eval/workdirs/_training_inputs/$source_work.json"
[[ -e "$output/eval/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/eval/workdirs/_tfhub_modules"
weights="type=bind,src=$main/workdirs/$source_work/checkpoints,dst=$root/workdirs/$source_work/checkpoints,readonly"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 16g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs,dst=$root/workdirs,readonly" --mount "$weights" \
    --workdir "$root" --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=4 \
    --entrypoint bash "$image_id" -c \
    'python tests/check_celeba_b_continue.py && python tests/check_celeba_b_curve.py && python celeba_b_continue.py check'
fi

[[ "$mode" == start || "$mode" == after-vp ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ $# -eq 1 && "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *H200* ]] || {
  echo "GPU $gpu is not an H200." >&2; exit 1; }
for name in "$train_name" "$eval_name"; do
  if docker inspect "$name" >/dev/null 2>&1; then
    echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
  fi
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 200 * 1024 * 1024 * 1024 )) || { echo 'Need at least 200 GiB free: 120 snapshots of 1 GB.' >&2; exit 1; }

if [[ "$mode" == after-vp ]]; then
  [[ "$(running "$pair")" == true ]] || { echo "$pair is not running; if VP already stopped, use start." >&2; exit 1; }
  busy="$(not_allowed "$gpu" "$pair")"
  [[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_celeba_b_continue_800k_h200.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $vp_checkpoint in process $!. Log: $log"
  exit 0
fi

busy="$(not_allowed "$gpu" "")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in B800K_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it. Training and evaluation share the card.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'containers=%s %s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$train_name" "$eval_name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"
common=(--gpus "device=$gpu" --ipc=host --shm-size=16g
  --mount "type=bind,src=$source_dir,dst=$root,readonly"
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly"
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds"
  --mount "$weights" --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_h200"
  --env XDG_CACHE_HOME="$root/.cache_h200" --env TORCH_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=8
  --entrypoint bash)

docker run -d --name "$train_name" "${common[@]}" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$output/.torch_extensions_h200,dst=$root/.torch_extensions_h200" \
  --mount "type=bind,src=$output/.cache_h200,dst=$root/.cache_h200" \
  "$image_id" -c "
    { python -u celeba_b_continue.py import &&
      python -u main.py --mode=train --config=celeba_b_continue.py --workdir=workdirs/$work &&
      python -u celeba_b_continue.py verify
    } 2>&1 | tee -a logs/b800k_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $train_name on physical GPU $gpu. Follow: docker logs -f --tail 50 $train_name"

# Preflight exits 1 on failure; the queue then stops itself. Exit 3: the control failed.
docker run -d --name "$eval_name" "${common[@]}" \
  --mount "type=bind,src=$output/eval/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/eval/logs,dst=$root/logs" \
  --mount "type=bind,src=$output/workdirs/$work,dst=$root/continuation,readonly" \
  --mount "type=bind,src=$output/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$output/eval/.torch_extensions_h200,dst=$root/.torch_extensions_h200" \
  --mount "type=bind,src=$output/eval/.cache_h200,dst=$root/.cache_h200" \
  --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  "$image_id" -c "
    { python -u celeba_b_curve_eval.py preflight && python -u celeba_b_curve_eval.py queue
    } 2>&1 | tee -a logs/bcurve_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $eval_name on physical GPU $gpu. Follow: docker logs -f --tail 50 $eval_name"
