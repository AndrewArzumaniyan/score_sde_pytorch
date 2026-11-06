#!/usr/bin/env bash
# Host launcher: CIFAR-10 matched protocol on an RTX PRO 6000 (cifar_matched35_eval.py).
# Run from a clean, separate Git worktree. NCSN++ weights come from the main checkout,
# DDPM++ weights from the old repository; both are mounted read-only.
#
#   bash run_cifar_matched35_rtx.sh check           CPU tests on the real manifests, no GPU
#   bash run_cifar_matched35_rtx.sh queue GPU_ID    detached queue: preflight, then the cells
#
# Rerunning the queue with the same output resumes it: finished cells are skipped.
# Remove the stopped container first. Results: $CIFAR35_OUTPUT/logs/summary.{json,csv}
# After any change to the code, use a new CIFAR35_OUTPUT and recompute the cells.
set -euo pipefail

mode="${1:?Usage: bash run_cifar_matched35_rtx.sh check|queue GPU_ID}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${CIFAR35_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${CIFAR35_MAIN:-$base/score_sde_pytorch}"
old="${CIFAR35_OLD:-$base/arzumand/score_sde_pytorch}"
output="${CIFAR35_OUTPUT:-$base/cifar_matched35_20261106}"
image="${CIFAR35_IMAGE:-score-sde-rtx:latest}"
name="${CIFAR35_CONTAINER:-cifar-matched35-rtx}"
min_free_mib="${CIFAR35_MIN_FREE_MIB:-40000}"
root=/workspace/score_sde_pytorch
ncsn=(cifar10_vp_200k_rtx_seed42 cifar10_fox_matern32_kappa1000_200k_rtx_seed42)
ddpm=(cifar10_vp_ddpmpp_deep_200k_rtx_seed42 cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx_seed42)

cd "$source_dir"
[[ "$(realpath "$source_dir")" != "$(realpath -m "$main")" ]] || {
  echo 'Run from a separate worktree, not from the main checkout.' >&2; exit 1; }
for path in cifar_matched35_eval.py run_cifar_matched35_rtx.sh tests/check_cifar_matched35.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
case "$output/" in "$source_dir/"*|"$main/"*|"$old/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

mkdir -p "$output/logs" "$output/workdirs/_training_inputs"
for work in "${ncsn[@]}"; do cp "$main/workdirs/$work/run_manifest.json" "$output/workdirs/_training_inputs/$work.json"; done
for work in "${ddpm[@]}"; do cp "$old/workdirs/$work/run_manifest.json" "$output/workdirs/_training_inputs/$work.json"; done

if [[ "$mode" == check ]]; then
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs/_training_inputs,dst=/inputs,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env CIFAR35_TEST_INPUTS=/inputs --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" tests/check_cifar_matched35.py
fi
[[ "$mode" == queue ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
gpu="${1:?Expected a physical GPU index.}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Expected a physical GPU index.' >&2; exit 2; }
# Free memory, not an empty card: other people may hold part of it.
free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= min_free_mib )) || { echo "GPU $gpu has $free MiB free, need $min_free_mib." >&2; exit 1; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it and remove it before restarting." >&2; exit 1
fi
[[ -s "$main/assets/stats/cifar10_stats.npz" ]] || { echo 'Missing CIFAR reference.' >&2; exit 1; }
[[ -d "$main/.tfds/cifar10" ]] || { echo 'Missing prepared CIFAR-10.' >&2; exit 1; }
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 40 * 1024 * 1024 * 1024 )) || { echo 'Need at least 40 GiB free.' >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mounts=()
add_weights() {
  local repo="$1" work="$2"; shift 2
  for ckpt in "$@"; do
    [[ -s "$repo/workdirs/$work/checkpoints/checkpoint_$ckpt.pth" ]] || {
      echo "Missing weights: $work checkpoint_$ckpt" >&2; exit 1; }
  done
  mkdir -p "$output/workdirs/$work/checkpoints"
  mounts+=(--mount "type=bind,src=$repo/workdirs/$work/checkpoints,dst=$root/workdirs/$work/checkpoints,readonly")
}
add_weights "$main" "${ncsn[0]}" 16 24
add_weights "$main" "${ncsn[1]}" 24
add_weights "$old" "${ddpm[0]}" 12
add_weights "$old" "${ddpm[1]}" 12
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'container=%s\nphysical_gpu=%s\nimage_id=%s\nsource=%s\ncommit=%s\n' \
  "$name" "$gpu" "$image_id" "$source_dir" "$(git rev-parse HEAD)" > "$output/logs/launch_$run_id.txt"

docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g \
  --mount "type=bind,src=$source_dir,dst=$root,readonly" \
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs" \
  --mount "type=bind,src=$output/logs,dst=$root/logs" \
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly" \
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds" \
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx" \
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx" \
  "${mounts[@]}" --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx" \
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=8 \
  --entrypoint bash "$image_id" -c "
    { python -u cifar_matched35_eval.py preflight && python -u cifar_matched35_eval.py queue
    } 2>&1 | tee -a logs/cifar35_$run_id.log; exit \${PIPESTATUS[0]}"
echo "Started $name on physical GPU $gpu. Follow: docker logs -f --tail 50 $name"
