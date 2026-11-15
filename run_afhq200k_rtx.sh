#!/usr/bin/env bash
# Host launcher: AFHQv2 A and VP at 200k, 50k matched35 (afhq200k_eval.py), a robustness check of
# the +0.2 tie rule (AFHQV2_STEP_RULE.md, 15 November). One lane per container and per RTX GPU:
# lane a (A at 200k) and lane vp (VP at 200k). Run from a clean, separate Git worktree, by default
# ~/kovshov-arzumanyan/arzumand/score_sde_afhq200k_20261115.
#
#   bash run_afhq200k_rtx.sh check              CPU tests and input paths; no GPU
#   bash run_afhq200k_rtx.sh start a GPU_ID     detached: preflight, then the 4 cells of A at 200k
#   bash run_afhq200k_rtx.sh start vp GPU_ID    detached: preflight, then the 4 cells of VP at 200k
#
# Results: $output/LANE/logs/summary.{json,csv}; logs/queue_*.log. Containers are not --rm, so
# run_paper_diff_bootstrap.sh can wait for their exit codes. A rerun with the same output skips
# finished cells; remove the stopped container first.
set -euo pipefail

mode="${1:?Usage: bash run_afhq200k_rtx.sh check|start a|vp GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AFHQ200K_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${AFHQ200K_MAIN:-$base/score_sde_pytorch}"
output="${AFHQ200K_OUTPUT:-$base/afhq200k_bootstrap_20261115}"
previous="${AFHQ200K_PREVIOUS:-$base/afhqv2_curve_20261112_r2/logs}"
tfhub="${AFHQ200K_TFHUB:-$base/celeba_50k_lo_20261107/gpu2/workdirs/_tfhub_modules}"
image="${AFHQ200K_IMAGE:-score-sde-rtx:latest}"
allowed="${AFHQ200K_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${AFHQ200K_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
inception=c4be6b55dc0338e82546082d22184ebdc77d76cf
# Lane: eval workdir name, source run workdir (holds checkpoints/ and run_manifest.json). 200k is
# checkpoint_4 of the continuation, the copy gpu2_queue_eval and gpu1_vp_tail_eval use for 200k.
work_of() { [[ "$1" == a ]] && echo afhq_a || echo afhq_vp; }
run_of() {
  if [[ "$1" == a ]]; then echo "$base/afhqv2_a400k_20261113/workdirs/afhqv2_A_vp_pvp_rtx_seed42"
  else echo "$base/afhqv2_vp400k_20261113/workdirs/afhqv2_vp_rtx_seed42"; fi
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
for path in afhq200k_eval.py gpu2_queue_eval.py afhqv2_rtx_train.py celeba_pair_seed.py celeba_50k_lo_eval.py \
    celeba_followup_eval.py celeba_kappa_eval.py celeba_matched35_rtx_eval.py celeba_stage0_eval.py \
    endpoint_denoise_eval.py AFHQV2_STEP_RULE.md run_afhq200k_rtx.sh tests/check_afhq200k.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
# The AFHQv2 images are closed to this user, so only the folder can be checked here; preflight reads them.
[[ -d "$main/datasets/afhqv2-64x64" ]] || { echo "Missing input: $main/datasets/afhqv2-64x64" >&2; exit 1; }
for path in "$main/assets/stats/afhqv2_64_cd017afab009_stats.npz" "$tfhub/$inception/saved_model.pb" \
    "$previous/a_c4_lo_result.json" "$previous/vp_c4_lo_result.json"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
for lane in a vp; do
  for path in "$(run_of "$lane")/run_manifest.json" "$(run_of "$lane")/checkpoints/checkpoint_4.pth"; do
    [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  done
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_afhq200k.py
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ $# -eq 2 ]] || { echo 'Expected a lane (a or vp) and one physical GPU index.' >&2; exit 2; }
lane="$1"
gpu="$2"
[[ "$lane" == a || "$lane" == vp ]] || { echo "Unknown lane: $lane" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
name="afhqv2-200k-$lane"
work="$(work_of "$lane")"
run="$(run_of "$lane")"
out="$output/$lane"
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || { echo "GPU $gpu is not an RTX." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
busy="$(not_allowed "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy. Name allowed ones in AFHQ200K_ALLOWED_NEIGHBOURS." >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
mkdir -p "$out/logs" "$out/workdirs/_training_inputs" "$out/workdirs/$work/checkpoints"
available="$(df -PB1 "$out" | awk 'NR==2 {print $4}')"
(( available >= 30 * 1024 * 1024 * 1024 )) || { echo 'Need at least 30 GiB free.' >&2; exit 1; }

[[ -e "$out/workdirs/_tfhub_modules" ]] || cp -a "$tfhub" "$out/workdirs/_tfhub_modules"
cp "$run/run_manifest.json" "$out/workdirs/_training_inputs/$work.json"
# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba datasets/afhqv2-64x64 assets/stats workdirs logs previous .tfds .torch_extensions_rtx .cache_rtx
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nlane=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$lane" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$out/logs/launch_$run_id.txt"

# The container path of the AFHQv2 data is part of the name of its statistics file. CelebA is not used.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$out/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$out/logs,dst=$root/logs" \
  --mount "type=bind,src=$run/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly" \
  --mount "type=bind,src=$previous,dst=$root/previous,readonly" \
  --mount "type=bind,src=$main/datasets/afhqv2-64x64,dst=$root/datasets/afhqv2-64x64,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 --env AFHQ200K_LANE="$lane" \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TFHUB_CACHE_DIR="$root/workdirs/_tfhub_modules" \
  --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u afhq200k_eval.py preflight && python -u afhq200k_eval.py queue
    } 2>&1 | tee -a logs/queue_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
