#!/usr/bin/env bash
# Server wrapper. Usage: bash run_ab_replication_seed43_docker.sh preflight GPU_ID
# Explicit train mode starts a detached container and repeats the full preflight.
set -euo pipefail

mode="${1:-preflight}"
gpu="${2:?Usage: bash run_ab_replication_seed43_docker.sh preflight|train GPU_ID}"
case "$mode" in preflight|train) ;; *) echo 'Expected preflight or train.' >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || {
  echo 'GPU_ID must be a physical GPU index or GPU UUID.' >&2; exit 2; }

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AB_REPLICATION_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="$base/score_sde_pytorch"
frozen="$base/score_sde_pytorch_ab_frozen"
output="$base/ab_replication_seed43"
launcher="$bundle_dir/run_ab_replication_train.sh"
container_root=/workspace/score_sde_pytorch
tfds="${TFDS_DATA_DIR:-$container_root/.tfds}"
extensions="${TORCH_EXTENSIONS_DIR:-$container_root/.torch_extensions}"
cache="${XDG_CACHE_HOME:-$container_root/.cache}"

for path in "$frozen/main.py" "$frozen/artifact_utils.py" "$launcher" \
  "$main/celeba/list_eval_partition.txt" "$main/celeba/list_attr_celeba.txt" \
  "$main/celeba/list_landmarks_align_celeba.txt" \
  "$main/workdirs/celeba_factorial_A_vp_pvp_50k/run_manifest.json" \
  "$main/workdirs/celeba_factorial_B_vp_pfox_50k/run_manifest.json"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$frozen/third_party" "$main/celeba/img_align_celeba"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 1; }
done
for arm in A_vp_pvp B_vp_pfox; do
  target="$output/workdirs/celeba_ab_replication_${arm}_seed43_200k"
  [[ ! -e "$target" && ! -L "$target" ]] || {
    echo "Fresh replication requires an absent workdir: $target" >&2; exit 1; }
done
for path in "$tfds" "$extensions" "$cache"; do
  [[ "$path" == /* && "$path" != / && "$path" != "$container_root" ]] || {
    echo "Expected an absolute container cache directory: $path" >&2; exit 2; }
done

# Pin the installed image by ID for this invocation. Do not pull or rebuild it.
image_id="$(docker image inspect --format '{{.Id}}' score-sde-h200)"
mkdir -p "$output/logs" "$output/workdirs" \
  "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
printf '%s\n' "$image_id" > "$output/logs/docker_${run_id}_${mode}_image_id.txt"
printf '%s\n' "$gpu" > "$output/logs/docker_${run_id}_${mode}_physical_gpu.txt"
sha256sum "$launcher" "$bundle_dir/run_ab_replication_seed43_docker.sh" \
  > "$output/logs/docker_${run_id}_${mode}_launchers_sha256.txt"
if launcher_commit="$(git -C "$bundle_dir" rev-parse HEAD 2>/dev/null)"; then
  printf '%s\n' "$launcher_commit" > "$output/logs/docker_${run_id}_${mode}_launcher_commit.txt"
  git -C "$bundle_dir" status --porcelain -- \
    run_ab_replication_train.sh run_ab_replication_seed43_docker.sh \
    > "$output/logs/docker_${run_id}_${mode}_launcher_git_status.txt"
else
  printf 'Launchers supplied without Git metadata; identify by SHA-256.\n' \
    > "$output/logs/docker_${run_id}_${mode}_launcher_commit.txt"
fi

args=(run --name "ab-replication-seed43-${mode}" --gpus "device=$gpu"
  --ipc=host
  --mount "type=bind,src=$frozen,dst=$container_root,readonly"
  --mount "type=bind,src=$main/celeba,dst=$container_root/celeba,readonly"
  --mount "type=bind,src=$main/workdirs,dst=/workspace/seed42_workdirs,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$container_root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$container_root/logs"
  --mount "type=bind,src=$main/.tfds,dst=$tfds"
  --mount "type=bind,src=$main/.torch_extensions,dst=$extensions"
  --mount "type=bind,src=$main/.cache,dst=$cache"
  --mount "type=bind,src=$launcher,dst=/opt/ab_replication/run_ab_replication_train.sh,readonly"
  --workdir "$container_root"
  --env CUDA_VISIBLE_DEVICES=0
  --env PYTHONDONTWRITEBYTECODE=1
  --env AB_REPLICATION_SOURCE_ROOT="$container_root"
  --env AB_REPLICATION_REFERENCE_ROOT=/workspace/seed42_workdirs
  --env AB_REPLICATION_MODE="$mode"
  --env AB_REPLICATION_SEED=43
  --env TFDS_DATA_DIR="$tfds"
  --env TORCH_EXTENSIONS_DIR="$extensions"
  --env XDG_CACHE_HOME="$cache"
  --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  --env MAX_JOBS="${MAX_JOBS:-8}"
  --entrypoint bash)
if [[ "$mode" == preflight ]]; then
  args+=(--rm)
else
  args+=(-d)
fi
printf 'Mode=%s; physical GPU=%s; image=%s; results=%s\n' \
  "$mode" "$gpu" "$image_id" "$output"
docker "${args[@]}" "$image_id" /opt/ab_replication/run_ab_replication_train.sh
