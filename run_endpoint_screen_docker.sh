#!/usr/bin/env bash
# Server: bash run_endpoint_screen_docker.sh 3
# Freeze current tracked sources and start one detached sequential eval queue.
set -euo pipefail
gpu="${1:?Usage: bash run_endpoint_screen_docker.sh GPU_ID}"
[[ $# == 1 && ( "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ) ]] || {
  echo 'Expected one physical GPU index or UUID.' >&2; exit 2; }
main="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="$(dirname "$main")/endpoint_screen_20261023"
name=celeba-endpoint-screen-20261023
root=/workspace/score_sde_pytorch
tfds="${TFDS_DATA_DIR:-$root/.tfds}"
extensions="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
cache="${XDG_CACHE_HOME:-$root/.cache}"
reference=assets/stats/celeba_64_0c2d89311192_stats.npz

[[ ! -e "$output" ]] || { echo "Output exists: $output; inspect/resume the existing container." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect/resume it." >&2; exit 1
fi
for path in "$tfds" "$extensions" "$cache"; do
  [[ "$path" == /* && "$path" != *','* && "$path" != *'..'* ]] || {
    echo "Invalid container cache path: $path" >&2; exit 2; }
  case "$path/" in
    "$root/.tfds/"|"$root/.torch_extensions/"|"$root/.cache/") ;;
    "$root/"*|/|/workspace/|/opt/|/opt/*|/tmp/|/usr/*|/etc/*|/proc/*|/dev/*|/sys/*|/bin/*|/lib/*)
      echo "Cache overlaps runtime/system directories: $path" >&2; exit 2 ;;
  esac
done
for pair in "$tfds|$extensions" "$tfds|$cache" "$extensions|$cache"; do
  left="${pair%%|*}/"; right="${pair#*|}/"
  [[ "$left" != "$right"* && "$right" != "$left"* ]] || {
    echo 'Cache mounts must not overlap.' >&2; exit 2; }
done
cd "$main"
for path in main.py endpoint_screen_config.py run_endpoint_screen.sh "$reference" \
  workdirs/celeba_vp_continuous_50k/checkpoints/checkpoint_260.pth \
  workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_260.pth \
  celeba/list_eval_partition.txt; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
[[ -d celeba/img_align_celeba ]] || { echo 'Missing CelebA images.' >&2; exit 1; }
# New launch files must have arrived through Git before making the snapshot.
git ls-files --error-unmatch endpoint_screen_config.py run_endpoint_screen.sh \
  run_endpoint_screen_docker.sh >/dev/null
image_id="$(docker image inspect --format '{{.Id}}' score-sde-h200)"
mkdir -p "$output/runtime" "$output/logs" "$output/workdirs" \
  "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
# Copy working-tree contents of tracked code, not files from HEAD.
git ls-files -z -- '*.py' '*.sh' '*.c' '*.cc' '*.cpp' '*.cu' '*.cuh' '*.h' requirements.txt \
  | tar --null -T - -cf - | tar -xf - -C "$output/runtime"
mkdir -p "$output/runtime/assets/stats" "$output/runtime/workdirs" \
  "$output/runtime/logs" "$output/runtime/celeba"
cp "$reference" "$output/runtime/$reference"
printf '%s  %s\n' \
  66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327 "$reference" \
  | sha256sum -c -
for path in "$tfds" "$extensions" "$cache"; do
  case "$path" in "$root"/*) mkdir -p "$output/runtime/${path#"$root"/}" ;; esac
done
(
  cd "$output/runtime"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$output/runtime.sha256"
sha256sum "$reference" \
  workdirs/celeba_vp_continuous_50k/checkpoints/checkpoint_260.pth \
  workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_260.pth \
  > "$output/inputs.sha256"
git rev-parse HEAD > "$output/commit.txt"
git status --porcelain > "$output/source_status.txt"
printf 'image=%s\nphysical_gpu=%s\nTFDS_DATA_DIR=%s\nTORCH_EXTENSIONS_DIR=%s\nXDG_CACHE_HOME=%s\n' \
  "$image_id" "$gpu" "$tfds" "$extensions" "$cache" > "$output/docker.txt"

args=(run -d --name "$name" --gpus "device=$gpu" --ipc=host
  --mount "type=bind,src=$output/runtime,dst=$root,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$root/logs"
  --mount "type=bind,src=$output,dst=/opt/endpoint-screen,readonly"
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly"
  --mount "type=bind,src=$main/.tfds,dst=$tfds"
  --mount "type=bind,src=$main/.torch_extensions,dst=$extensions"
  --mount "type=bind,src=$main/.cache,dst=$cache"
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache"
  --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}"
  --entrypoint bash)
for arm in celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k; do
  mkdir -p "$output/workdirs/$arm/checkpoints"
  args+=(--mount "type=bind,src=$main/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
done
docker "${args[@]}" "$image_id" run_endpoint_screen.sh
docker inspect "$name" > "$output/docker_inspect_start.json"
printf 'Started %s on physical GPU %s. Results: %s\n' "$name" "$gpu" "$output"
