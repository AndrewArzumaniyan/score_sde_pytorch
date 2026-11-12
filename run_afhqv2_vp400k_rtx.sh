#!/usr/bin/env bash
# Host launcher: AFHQv2-64 arm VP from 200k to 400k on an RTX, snapshots every 50k (250k-400k).
# Run from a clean, separate Git worktree, by default ~/kovshov-arzumanyan/arzumand/score_sde_afhqv2_vp400k_20261113.
#
#   bash run_afhqv2_vp400k_rtx.sh check          CPU tests, paths, the training worktree; no GPU
#   bash run_afhqv2_vp400k_rtx.sh start GPU_ID   copy VP at 200k into a new output and train it to 400k, detached
#
# The training runs the code it was trained with: the worktree of 95de22d, mounted read-only.
# n_iters lies outside its training protocol, so the run resumes from its own 200k state:
# optimizer, EMA, RNG and data position. The source workdir is only read: its manifest,
# checkpoints-meta and checkpoint_4 are copied into the new output first. No evaluation here.
# Log: logs/vp400k_*.log. Rerunning start with the same output resumes; remove the stopped container first.
set -euo pipefail

mode="${1:?Usage: bash run_afhqv2_vp400k_rtx.sh check|start GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AFHQ_VP400K_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${AFHQ_VP400K_MAIN:-$base/score_sde_pytorch}"
train_source="${AFHQ_VP400K_TRAIN_SOURCE:-$base/score_sde_afhqv2_20261108}"
train_commit="${AFHQ_VP400K_TRAIN_COMMIT:-95de22d247b06b8b4c932a1de3a61a10046aa0a7}"
source_run="${AFHQ_VP400K_SOURCE_RUN:-$base/afhqv2_rtx_20261108/workdirs/afhqv2_vp_rtx_seed42}"
output="${AFHQ_VP400K_OUTPUT:-$base/afhqv2_vp400k_20261113}"
image="${AFHQ_VP400K_IMAGE:-score-sde-rtx:latest}"
allowed="${AFHQ_VP400K_ALLOWED_NEIGHBOURS:-}"
min_free_mib="${AFHQ_VP400K_MIN_FREE_MIB:-60000}"
root=/workspace/score_sde_pytorch
work=afhqv2_vp_rtx_seed42
stats=assets/stats/afhqv2_64_cd017afab009_stats.npz
name=afhqv2-rtx-vp400k

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
for path in run_afhqv2_vp400k_rtx.sh tests/check_afhqv2_vp400k.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
# The training must see exactly the code of its first 200k, or its resume refuses.
[[ "$(git -C "$train_source" rev-parse HEAD)" == "$train_commit" ]] || {
  echo "The training worktree $train_source is not at $train_commit." >&2; exit 1; }
[[ -z "$(git -C "$train_source" status --porcelain --untracked-files=no)" ]] || {
  echo "The training worktree $train_source has changed tracked files." >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*|"$train_source/"*|"$source_run/"*)
  echo 'Output must be outside the repositories and the source run.' >&2; exit 2 ;; esac
# The images are closed to this user, so only the folder can be checked here.
[[ -d "$main/datasets/afhqv2-64x64" ]] || { echo "Missing input: $main/datasets/afhqv2-64x64" >&2; exit 1; }
for path in "$main/$stats" "$source_run/run_manifest.json" "$source_run/checkpoints-meta/checkpoint.pth" \
    "$source_run/checkpoints/checkpoint_4.pth"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 2 --memory 4g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint python "$image_id" tests/check_afhqv2_vp400k.py
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ $# -eq 1 && "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected one physical GPU index.' >&2; exit 2; }
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *RTX* ]] || { echo "GPU $gpu is not an RTX." >&2; exit 1; }
busy="$(not_allowed "$gpu")"
[[ -z "$busy" ]] || { echo "GPU $gpu runs containers: $busy" >&2; exit 1; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 20 * 1024 * 1024 * 1024 )) || { echo 'Need at least 20 GiB free: four more snapshots of 1 GB.' >&2; exit 1; }

# The copy is the starting point; a rerun keeps whatever the training has written since.
target="$output/workdirs/$work"
mkdir -p "$output/logs" "$target/checkpoints" "$target/checkpoints-meta"
[[ -s "$target/run_manifest.json" ]] || cp "$source_run/run_manifest.json" "$target/run_manifest.json"
[[ -s "$target/checkpoints/checkpoint_4.pth" ]] || cp "$source_run/checkpoints/checkpoint_4.pth" "$target/checkpoints/"
[[ -s "$target/checkpoints-meta/checkpoint.pth" ]] || cp "$source_run/checkpoints-meta/checkpoint.pth" "$target/checkpoints-meta/"

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\ntraining_source=%s\ntraining_commit=%s\nlauncher_commit=%s\n' \
  "$name" "$gpu" "$image_id" "$train_source" "$train_commit" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"
# The container path of the data is part of the name of the statistics file.
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$train_source,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/datasets/afhqv2-64x64,dst=$root/datasets/afhqv2-64x64,readonly" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    python -u afhqv2_rtx_train.py train vp 400000 2>&1 | tee -a logs/vp400k_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
