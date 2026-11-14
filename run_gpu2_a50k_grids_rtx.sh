#!/usr/bin/env bash
# Host launcher on RTX GPU 2 for gpu2_a50k_grids.py, two containers one after the other:
#   celeba-gpu2-grids  preflight on A weights (500k of the 800k run), then the sample grids of the
#                      known models: CelebA VP 200k, B 800k, DIAL 800k; AFHQv2 VP 250k, A 250k, DIAL 200k
#   celeba-gpu2-a50k   after $AGRID_CURVE ends by itself with code 0: the A step from its summary
#                      (CELEBA_A800K_STEP_RULE.md), the A grid, then 50k matched35 for A
# Run from a clean, separate Git worktree, by default
# ~/kovshov-arzumanyan/arzumand/score_sde_gpu2_a50k_grids_20261114.
#
#   bash run_gpu2_a50k_grids_rtx.sh check            CPU tests and input paths; no GPU
#   bash run_gpu2_a50k_grids_rtx.sh start GPU_ID     detached: preflight, then the grids
#   bash run_gpu2_a50k_grids_rtx.sh after-a GPU_ID   in the background: wait until $AGRID_CURVE ends by itself
#                                                    and the grids container stops, then start-a
#   bash run_gpu2_a50k_grids_rtx.sh start-a GPU_ID   detached: the A step, the A grid, 50k for A
#
# Nothing here stops or touches a container of another card. start-a refuses unless the curve container
# exists, has stopped and exited with 0. If its summary holds no clean selection, the A container writes
# logs/celeba_a_selection.json with the reason and evaluates nothing. If the wait passes AGRID_WAIT_MAX_SECONDS,
# nothing starts. Logs: logs/wait_*.log, logs/grids_*.log, logs/a50k_*.log. Results: grids/, logs/summary.json.
set -euo pipefail

mode="${1:?Usage: bash run_gpu2_a50k_grids_rtx.sh check|start GPU_ID|after-a GPU_ID|start-a GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AGRID_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${AGRID_MAIN:-$base/score_sde_pytorch}"
output="${AGRID_OUTPUT:-$base/gpu2_a50k_grids_20261114}"
curve_logs="${AGRID_CURVE_LOGS:-$base/celeba_a_800k_h200_20261113/eval/logs}"
table4="${AGRID_TABLE4:-$base/celeba_50k_lo_20261107/gpu1/logs}"
tfhub="${AGRID_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${AGRID_IMAGE:-score-sde-rtx:latest}"
curve="${AGRID_CURVE:-celeba-a-curve-h200}"
allowed="${AGRID_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${AGRID_MIN_FREE_MIB:-60000}"
wait_seconds="${AGRID_WAIT_SECONDS:-60}"
wait_max_seconds="${AGRID_WAIT_MAX_SECONDS:-86400}"
root=/workspace/score_sde_pytorch
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
grids_name=celeba-gpu2-grids
a_name=celeba-gpu2-a50k
# Eval workdir name, source run workdir (holds checkpoints/ and run_manifest.json), manifest key.
sources=(
  "celeba_vp|$main/workdirs/celeba_vp_continuous_50k|"
  "celeba_fox|$main/workdirs/celeba_fox_matern32_kappa1000_50k|"
  "celeba_b|$base/celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42|celeba_b"
  "celeba_a_first|$base/celeba_a_400k_rtx_20261106/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42|celeba_a_first"
  "celeba_a|$base/celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42|celeba_a"
  "afhq_vp|$base/afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42|afhq_vp"
  "afhq_a|$base/afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42|afhq_a"
  "afhq_fox|$base/afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42|afhq_fox"
)
# Round 0 of the finished seed 0 DPM++ 50k cells, compared with the grids: refs/NAME/samples_0.npz.
references=(
  "celeba_vp|$base/celeba_50k_lo_20261107/gpu1/workdirs/celeba_vp_continuous_50k/eval_50k_lo_vp_c40_lo_ode_dpmpp_2m_seed0/ckpt_40"
  "celeba_a_400k|$base/celeba_50k_lo_20261107/gpu1/workdirs/celeba_factorial_A_vp_pvp_400k_rtx_seed42/eval_50k_lo_A_c80_lo_ode_dpmpp_2m_seed0/ckpt_80"
  "celeba_b|$base/gpu2_queue_20261114/workdirs/celeba_b/eval_gpu2q_celeba_b_c160_m35_ode_dpmpp_2m_seed0/ckpt_160"
  "afhq_dial|$base/gpu2_queue_20261114/workdirs/afhq_fox/eval_gpu2q_afhq_fox_c4_m35_ode_dpmpp_2m_seed0/ckpt_4"
  "afhq_a|$base/gpu2_queue_20261114/workdirs/afhq_a/eval_gpu2q_afhq_a_cbest_m35_ode_dpmpp_2m_seed0/ckpt_5"
  "afhq_vp|$base/gpu1_vp_tail_20261114/workdirs/afhq_vp/eval_gpu1vp_afhq_vp_cbest_m35_ode_dpmpp_2m_seed0/ckpt_5"
)
checkpoints=(
  "$main/workdirs/celeba_vp_continuous_50k/checkpoints/checkpoint_40.pth"
  "$main/workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_160.pth"
  "$base/celeba_b_800k_h200_20261111/workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42/checkpoints/checkpoint_160.pth"
  "$base/celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42/checkpoints/checkpoint_100.pth"
  "$base/afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42/checkpoints/checkpoint_5.pth"
  "$base/afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42/checkpoints/checkpoint_5.pth"
  "$base/afhqv2_rtx_20261108/workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42/checkpoints/checkpoint_4.pth"
)

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
not_allowed() {
  local container
  while read -r container; do
    [[ -z "$container" || ",$allowed,$2," == *",$container,"* ]] || echo "$container"
  done < <(gpu_containers "$1")
}

if [[ "$mode" == _wait ]]; then
  gpu="${1:?Expected a physical GPU index.}"
  echo "$(stamp) waiting until $curve ends by itself and $grids_name stops, then GPU $gpu, at most $wait_max_seconds s"
  for container in "$curve" "$grids_name"; do
    while [[ "$(running "$container")" == true ]]; do
      (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $container still runs; nothing started"; exit 1; }
      sleep "$wait_seconds"
    done
    echo "$(stamp) $container is not running"
  done
  if ! bash "$source_dir/run_gpu2_a50k_grids_rtx.sh" start-a "$gpu"; then
    echo "$(stamp) the start refused; nothing started"; exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in gpu2_a50k_grids.py gpu2_queue_eval.py afhqv2_rtx_train.py celeba_pair_seed.py celeba_50k_lo_eval.py \
    celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py celeba_stage0_eval.py \
    endpoint_denoise_eval.py ode_sampling.py AFHQV2_STEP_RULE.md CELEBA_A800K_STEP_RULE.md \
    run_gpu2_a50k_grids_rtx.sh tests/check_gpu2_a50k_grids.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$tfhub/$inception/saved_model.pb" "${checkpoints[@]}"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
for entry in "${sources[@]}"; do
  IFS='|' read -r work run key <<< "$entry"
  [[ -d "$run/checkpoints" ]] || { echo "Missing input: $run/checkpoints" >&2; exit 1; }
  [[ -z "$key" || -s "$run/run_manifest.json" ]] || { echo "Missing input: $run/run_manifest.json" >&2; exit 1; }
done
for entry in "${references[@]}"; do
  IFS='|' read -r ref path <<< "$entry"
  [[ -s "$path/samples_0.npz" ]] || { echo "Missing input: $path/samples_0.npz" >&2; exit 1; }
done
[[ -d "$curve_logs" && -s "$table4/A_c80_lo_ode_dpmpp_2m_seed0_result.json" ]] || {
  echo "Missing input: $curve_logs or the table 4 results in $table4" >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_gpu2_a50k_grids.py
fi

[[ "$mode" == start || "$mode" == after-a || "$mode" == start-a ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ $# -eq 1 && "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || { echo "GPU $gpu is not an RTX." >&2; exit 1; }
name="$grids_name"
[[ "$mode" == start ]] || name="$a_name"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
mkdir -p "$output/logs" "$output/grids" "$output/workdirs/_training_inputs"
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 60 * 1024 * 1024 * 1024 )) || { echo 'Need at least 60 GiB free.' >&2; exit 1; }

if [[ "$mode" == after-a ]]; then
  [[ "$(running "$curve")" == true ]] || { echo "$curve is not running; if it has finished, use start-a." >&2; exit 1; }
  busy="$(not_allowed "$gpu" "$grids_name")"
  [[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
  pid_file="$output/logs/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/logs/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_gpu2_a50k_grids_rtx.sh" _wait "$gpu" >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for $curve in process $!. Log: $log"
  exit 0
fi

if [[ "$mode" == start-a ]]; then
  state="$(running "$curve")"
  [[ "$state" == false ]] || { echo "$curve is $state; the A step needs its finished curve." >&2; exit 1; }
  code="$(docker inspect --format '{{.State.ExitCode}}' "$curve")"
  [[ "$code" == 0 ]] || { echo "$curve exited with $code; nothing started." >&2; exit 1; }
  [[ -s "$curve_logs/summary.json" ]] || { echo "Missing input: $curve_logs/summary.json" >&2; exit 1; }
fi
busy="$(not_allowed "$gpu" "")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in AGRID_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }

[[ -e "$output/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$output/workdirs/_tfhub_modules"
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba assets/stats workdirs logs grids acurve table4 .tfds .torch_extensions_rtx .cache_rtx
mounts=()
for entry in "${sources[@]}"; do
  IFS='|' read -r work run key <<< "$entry"
  mkdir -p "$output/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$run/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
  [[ -z "$key" ]] || cp "$run/run_manifest.json" "$output/workdirs/_training_inputs/$key.json"
done
for entry in "${references[@]}"; do
  IFS='|' read -r ref path <<< "$entry"
  mkdir -p "refs/$ref"
  mounts+=(--mount "type=bind,src=$path,dst=$root/refs/$ref,readonly")
done
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"
if [[ "$mode" == start ]]; then
  command="{ python -u gpu2_a50k_grids.py preflight; python -u gpu2_a50k_grids.py grids; } 2>&1 | tee -a logs/grids_$run_id.log"
else
  command="{ python -u gpu2_a50k_grids.py select && { python -u gpu2_a50k_grids.py preflight-a;
    python -u gpu2_a50k_grids.py a; }; } 2>&1 | tee -a logs/a50k_$run_id.log"
fi

docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$output/grids,dst=$root/grids" \
  --mount "type=bind,src=$curve_logs,dst=$root/acurve,readonly" \
  --mount "type=bind,src=$table4,dst=$root/table4,readonly" \
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
  --entrypoint bash "$image_id" -c "$command; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
