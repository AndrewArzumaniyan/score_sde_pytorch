#!/usr/bin/env bash
# Host launcher: nfe_limit_extra_eval.py (rule of 16 November in its docstring). One lane per container and
# per card; any card type (H200: sm 9.0, RTX PRO 6000: sm 12.0). Run from a clean, separate Git worktree.
#
#   bash run_nfe_limit_extra.sh check                CPU tests and input paths; no GPU
#   bash run_nfe_limit_extra.sh start a500 GPU_ID    detached: VP-lambda@500k, 50k then 10k x 2, 257 NFE
#   bash run_nfe_limit_extra.sh start fox43 GPU_ID   detached: DIAL seed 43 @800k, the same cells
#   bash run_nfe_limit_extra.sh start a43 GPU_ID     detached: VP-lambda seed 43 @400k, the same cells
#   bash run_nfe_limit_extra.sh combine              contrasts of the rule; CPU, read-only
#
# Results: $output/LANE/logs/summary.{json,csv}; logs/queue_*.log. A rerun with the same output skips
# finished cells and reuses finished sample rounds; remove the stopped container first.
set -euo pipefail

mode="${1:?Usage: bash run_nfe_limit_extra.sh check|start a500|fox43|a43 GPU_ID|combine}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${NFE_LIMIT_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${NFE_LIMIT_MAIN:-$base/score_sde_pytorch}"
output="${NFE_LIMIT_OUTPUT:-$base/nfe_limit_extra_20261116}"
previous="${NFE_LIMIT_PREVIOUS:-$base/nfe_limit_20261116}"  # Lanes dial, b, a of the first queue, read-only.
tfhub="${NFE_LIMIT_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${NFE_LIMIT_IMAGE:-score-sde-rtx:latest}"  # The same image on both card types, as in the first queue.
allowed="${NFE_LIMIT_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${NFE_LIMIT_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
lanes=(a500 fox43 a43)
# Lane: eval workdir name | source run workdir (holds checkpoints/ and run_manifest.json) | manifest key | checkpoint.
sources_of() {
  case "$1" in
    a500) echo "celeba_a|$base/celeba_a_800k_h200_20261113/workdirs/celeba_factorial_A_vp_pvp_800k_h200_seed42|celeba_a|100" ;;
    fox43) echo "celeba_fox43|$base/celeba_pair_seed43_h200_20261106/workdirs/celeba_fox_matern32_kappa1000_h200_seed43|celeba_fox_matern32_kappa1000_h200_seed43|160" ;;
    a43) echo "celeba_a43|$base/celeba_a43_400k_rtx_20261107/workdirs/celeba_ab_replication_A_vp_pvp_400k_rtx_seed43|celeba_ab_replication_A_vp_pvp_400k_rtx_seed43|8" ;;
  esac
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

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == combine ]]; then
  mounts=() dirs=()
  for lane in "${lanes[@]}"; do
    [[ -d "$output/$lane/logs" ]] || continue
    mounts+=(--mount "type=bind,src=$output/$lane/logs,dst=/nfe/$lane,readonly"); dirs+=("/nfe/$lane")
  done
  for lane in dial b a; do
    [[ -d "$previous/$lane/logs" ]] || continue
    mounts+=(--mount "type=bind,src=$previous/$lane/logs,dst=/prev/$lane,readonly"); dirs+=("/prev/$lane")
  done
  (( ${#mounts[@]} )) || { echo "No lane logs under $output or $previous." >&2; exit 1; }
  mkdir -p "$output"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  docker run --rm --network none --cpus 2 --memory 4g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" "${mounts[@]}" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint python "$image_id" nfe_limit_extra_eval.py combine "${dirs[@]}" \
    | tee "$output/contrasts_$stamp.json"
  exit "${PIPESTATUS[0]}"
fi

for path in nfe_limit_extra_eval.py nfe_limit_eval.py celeba_pair43_lo_eval.py gpu2_queue_eval.py gpu2_a50k_grids.py \
    afhqv2_rtx_train.py celeba_pair_seed.py celeba_50k_lo_eval.py celeba_followup_eval.py celeba_kappa_eval.py \
    celeba_matched35_rtx_eval.py celeba_stage0_eval.py endpoint_denoise_eval.py ode_sampling.py \
    run_nfe_limit_extra.sh tests/check_nfe_limit_extra.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
for path in "$main/assets/stats/celeba_64_0c2d89311192_stats.npz" "$main/celeba/list_eval_partition.txt" \
    "$tfhub/c4be6b55dc0338e82546082d22184ebdc77d76cf/saved_model.pb"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
for lane in "${lanes[@]}"; do
  while IFS='|' read -r work run key ckpt; do
    [[ -s "$run/run_manifest.json" ]] || { echo "Missing input: $run/run_manifest.json" >&2; exit 1; }
    [[ -s "$run/checkpoints/checkpoint_$ckpt.pth" ]] || { echo "Missing input: $run/checkpoints/checkpoint_$ckpt.pth" >&2; exit 1; }
  done < <(sources_of "$lane")
done

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_nfe_limit_extra.py
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ $# -eq 2 ]] || { echo 'Expected a lane (a500, fox43 or a43) and one physical GPU index.' >&2; exit 2; }
lane="$1"
gpu="$2"
[[ "$lane" == a500 || "$lane" == fox43 || "$lane" == a43 ]] || { echo "Unknown lane: $lane" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
name="celeba-nfe-$lane"
out="$output/$lane"
gpu_name="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)"
case "$gpu_name" in
  *H200*) card=H200; arch=9.0 ;;
  *"RTX PRO 6000"*) card=RTX; arch=12.0 ;;
  *) echo "GPU $gpu is $gpu_name; expected an H200 or an RTX PRO 6000." >&2; exit 1 ;;
esac
suffix="$(echo "$card" | tr '[:upper:]' '[:lower:]')"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
busy="$(not_allowed "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in NFE_LIMIT_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
mkdir -p "$out/logs" "$out/workdirs/_training_inputs" "$out/.torch_extensions_$suffix" "$out/.cache_$suffix"
available="$(df -PB1 "$out" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }

[[ -e "$out/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$out/workdirs/_tfhub_modules"
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba assets/stats workdirs logs .tfds ".torch_extensions_$suffix" ".cache_$suffix"
mounts=()
while IFS='|' read -r work run key ckpt; do
  mkdir -p "$out/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$run/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
  cp "$run/run_manifest.json" "$out/workdirs/_training_inputs/$key.json"
done < <(sources_of "$lane")
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nlane=%s\nphysical_gpu=%s\ngpu_name=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$lane" "$gpu" "$gpu_name" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$out/logs/launch_$run_id.txt"

docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$out/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$out/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$out/.torch_extensions_$suffix,dst=$root/.torch_extensions_$suffix" \
  --mount "type=bind,src=$out/.cache_$suffix,dst=$root/.cache_$suffix" \
  "${mounts[@]}" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 --env NFE_LIMIT_LANE="$lane" \
  --env NFE_LIMIT_CARD="$card" \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_$suffix" \
  --env XDG_CACHE_HOME="$root/.cache_$suffix" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST="$arch" --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u nfe_limit_extra_eval.py preflight && python -u nfe_limit_extra_eval.py queue
    } 2>&1 | tee -a logs/queue_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu ($gpu_name). Follow: docker logs -f --tail 50 $name"
