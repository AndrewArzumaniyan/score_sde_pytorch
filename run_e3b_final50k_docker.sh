#!/usr/bin/env bash
# Server: bash run_e3b_final50k_docker.sh GPU_ID
# Creates a frozen eval runtime, then starts one detached sequential GPU job.
set -euo pipefail
gpu="${1:?Usage: bash run_e3b_final50k_docker.sh GPU_ID}"
[[ $# == 1 && ( "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ) ]] || {
  echo 'Expected one physical GPU index or UUID.' >&2; exit 2; }
main="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="${E3B_FINAL_OUTPUT:-$(dirname "$main")/e3b_final50k_splices}"
output="$(realpath -m "$output")"
case "$output/" in "$main/"*|/) echo 'Output must be outside the source repository.' >&2; exit 2 ;; esac
[[ ! -e "$output" ]] || { echo "Output already exists: $output; inspect/resume the existing container." >&2; exit 1; }
name=e3b-final50k-splices-seeds01
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; inspect it before restarting." >&2; exit 1
fi
root=/workspace/score_sde_pytorch
tfds="${TFDS_DATA_DIR:-$root/.tfds}"
extensions="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
cache="${XDG_CACHE_HOME:-$root/.cache}"
for path in "$tfds" "$extensions" "$cache"; do
  [[ "$path" == /* && "$path" != *','* && "$path" != *'..'* ]] || {
    echo "Invalid container cache path: $path" >&2; exit 2; }
  case "$path/" in "$root/".tfds/|"$root/".torch_extensions/|"$root/".cache/) ;;
    "$root/"*|/|/workspace/|/opt/|/opt/e3b/*|/tmp/|/usr/*|/etc/*|/proc/*|/dev/*|/sys/*|/bin/*|/lib/*)
      echo "Cache path overlaps the runtime or system directories: $path" >&2; exit 2 ;;
  esac
done
for pair in "$tfds|$extensions" "$tfds|$cache" "$extensions|$cache"; do
  left="${pair%%|*}/"; right="${pair#*|}/"
  [[ "$left" != "$right"* && "$right" != "$left"* ]] || {
    echo 'Container cache mounts must not overlap.' >&2; exit 2; }
done
for path in "$main/main.py" "$main/e3b_final50k.py" "$main/run_e3b_final50k.sh" \
  "$main/run_e3b_splice_eval.sh" "$main/celeba/list_eval_partition.txt"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done
[[ -d "$main/celeba/img_align_celeba" ]] || { echo 'Missing local CelebA images.' >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' score-sde-h200)"
mkdir -p "$output" "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
# Snapshot preparation is CPU-only. A failure leaves the directory for inspection.
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$main,dst=/source,readonly" \
  --mount "type=bind,src=$output,dst=/output" \
  --entrypoint python "$image_id" /source/e3b_final50k.py snapshot /source /output
printf 'image=%s\nphysical_gpu=%s\nTFDS_DATA_DIR=%s\nTORCH_EXTENSIONS_DIR=%s\nXDG_CACHE_HOME=%s\n' \
  "$image_id" "$gpu" "$tfds" "$extensions" "$cache" > "$output/identity/docker.txt"
# runc needs nested destinations to exist before making the parent read-only.
for path in "$tfds" "$extensions" "$cache"; do
  case "$path" in "$root"/*) mkdir -p "$output/runtime/${path#"$root"/}" ;; esac
done
args=(run -d --name "$name" --gpus "device=$gpu" --ipc=host
  --mount "type=bind,src=$output/runtime,dst=$root,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$root/logs"
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly"
  --mount "type=bind,src=$output/identity,dst=/opt/e3b/identity,readonly"
  --mount "type=bind,src=$output/baselines,dst=/opt/e3b/baselines,readonly"
  --mount "type=bind,src=$main/.tfds,dst=$tfds"
  --mount "type=bind,src=$main/.torch_extensions,dst=$extensions"
  --mount "type=bind,src=$main/.cache,dst=$cache"
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache"
  --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}"
  --entrypoint bash)
for arm in celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k; do
  args+=(--mount "type=bind,src=$main/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
done
docker "${args[@]}" "$image_id" run_e3b_final50k.sh
docker inspect "$name" > "$output/identity/docker_inspect_start.json"
printf 'Started %s on physical GPU %s. Results: %s\n' "$name" "$gpu" "$output"
printf 'Follow: docker logs -f --tail 60 %s\n' "$name"
