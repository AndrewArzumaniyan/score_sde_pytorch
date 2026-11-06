#!/usr/bin/env bash
# Host launcher: seed 43 of the native VP/FOX pair on one H200 (celeba_pair_seed.py).
# FOX trains to 800k first, then VP to 800k, in one container on one GPU.
# Run from a clean, separate Git worktree. The checkout used by running trainings
# must stay as it is: a new source file there would block their resume.
#
#   bash run_celeba_pair_seed43_h200.sh check GPU_ID   tests and a real 300-update smoke, about 10 min
#   bash run_celeba_pair_seed43_h200.sh queue GPU_ID   detached: FOX, then VP
#
# Stop with docker stop, remove the container, and rerun the queue to resume from the
# last 5k: the run replays its data position first. The manifests pin the source
# fingerprint: keep the same worktree. To end after FOX, stop while VP is training.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_pair_seed43_h200.sh check|queue GPU_ID}"
gpu="${2:?Expected a physical GPU index.}"
[[ "$mode" == check || "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${PAIR_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${PAIR_MAIN:-$base/score_sde_pytorch}"
output="${PAIR_OUTPUT:-$base/celeba_pair_seed43_h200_20261106}"
image="${PAIR_IMAGE:-score-sde-h200:latest}"
name="${PAIR_CONTAINER:-celeba-pair-seed43-h200}"
root=/workspace/score_sde_pytorch

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the training checkout.' >&2; exit 1; }
for path in celeba_pair_seed.py run_celeba_pair_seed43_h200.sh tests/check_celeba_pair_seed.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
[[ "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" == *H200* ]] || {
  echo "GPU $gpu is not an H200." >&2; exit 1; }
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
(( used < 1000 )) || { echo "GPU $gpu is busy: $used MiB." >&2; exit 1; }
for path in "$main/celeba/img_align_celeba" "$main/.torch_extensions"; do
  [[ -d "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
commit="$(git rev-parse HEAD)"

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs .tfds .torch_extensions .cache
mkdir -p "$output/logs" "$output/workdirs"
common=(--gpus "device=$gpu" --ipc=host --shm-size=16g
  --mount "type=bind,src=$source_dir,dst=$root,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$root/logs"
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly"
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds"
  --mount "type=bind,src=$main/.torch_extensions,dst=$root/.torch_extensions"
  --mount "type=bind,src=$main/.cache,dst=$root/.cache"
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions"
  --env XDG_CACHE_HOME="$root/.cache" --env TORCH_CUDA_ARCH_LIST=9.0 --env MAX_JOBS=8
  --entrypoint bash)

if [[ "$mode" == check ]]; then
  docker run --rm --name "$name-check" "${common[@]}" "$image_id" -c \
    'python tests/check_celeba_pair_seed.py && python -u celeba_pair_seed.py smoke'
  printf 'image_id=%s\nphysical_gpu=%s\n' "$image_id" "$gpu" > "$output/logs/check_passed_$commit.txt"
  echo "Check passed for $commit."
  exit 0
fi

[[ -s "$output/logs/check_passed_$commit.txt" ]] || {
  echo "Run the check for this commit first: $commit" >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 400 * 1024 * 1024 * 1024 )) || { echo 'Need at least 400 GiB free.' >&2; exit 1; }

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$commit" > "$output/logs/launch_$run_id.txt"
docker run -d --name "$name" "${common[@]}" "$image_id" -c "
  { python -u celeba_pair_seed.py train fox &&
    python -u celeba_pair_seed.py train vp
  } 2>&1 | tee -a logs/pair_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
