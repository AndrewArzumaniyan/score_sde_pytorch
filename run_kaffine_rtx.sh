#!/usr/bin/env bash
# Host launcher for the k_a(t) shape experiment on RTX PRO 6000 (sm_120, image score-sde-rtx).
# Run from a clean, dedicated Git worktree; it is mounted read-only as the runtime.
#
#   bash run_kaffine_rtx.sh check                CPU-only tests inside the RTX image
#   bash run_kaffine_rtx.sh smoke GPU_ID         preflight + 400 steps + one eval round, foreground
#   bash run_kaffine_rtx.sh queue GPU_ID [STAGE...]
#                                                detached queue; default: three arms, then evals
#
# All arms of one comparison must run on the same GPU type with the same image.
# Results: $KAFFINE_OUTPUT/{workdirs,logs}.  Rerunning the same queue resumes it:
# finished stages are skipped and an interrupted training continues from checkpoints-meta.
set -euo pipefail

mode="${1:?Usage: bash run_kaffine_rtx.sh check|smoke GPU_ID|queue GPU_ID [STAGE...]}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${KAFFINE_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${KAFFINE_MAIN:-$base/score_sde_pytorch}"
output="${KAFFINE_OUTPUT:-$base/kaffine_rtx_20261104}"
image="${KAFFINE_IMAGE:-score-sde-rtx:latest}"
root=/workspace/score_sde_pytorch
default_queue=(preflight
  train:a0 train:ap5 eval:a0:4:10000 eval:ap5:4:10000
  train:vplin_norm eval:vplin_norm:4:10000
  eval:a0:2:5000 eval:ap5:2:5000 eval:vplin_norm:2:5000)

cd "$source_dir"
for path in configs/fox/celeba_kaffine.py run_kaffine_queue.sh main.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || {
    echo "Not committed: $path. Commit and use the committed revision." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD in the runtime worktree; use a clean worktree.' >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' "$image")"

if [[ "$mode" == check ]]; then
  # Importing run_lib JIT-builds the NCSN++ ops; a private /tmp cache keeps the shared one untouched.
  exec docker run --rm --network none --cpus 4 --memory 8g \
    --mount "type=bind,src=$source_dir,dst=$root,readonly" --workdir "$root" \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void \
    --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions --env TORCH_CUDA_ARCH_LIST=12.0 --env MAX_JOBS=4 \
    --entrypoint python "$image_id" -m pytest -q -p no:cacheprovider \
    tests/test_fox_sde.py tests/test_kaffine_config.py
fi

gpu="${1:?Expected a physical GPU index or UUID.}"
shift
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { echo 'Expected a physical GPU index or UUID.' >&2; exit 2; }
case "$mode" in
  smoke) stages=(preflight smoke); name="kaffine-rtx-smoke" ;;
  queue) stages=("$@"); (( ${#stages[@]} )) || stages=("${default_queue[@]}")
         name="${KAFFINE_CONTAINER:-kaffine-rtx-queue}" ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid container name.' >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name. Inspect it (docker logs $name) and remove it before restarting." >&2; exit 1
fi
case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac

for path in "$main/celeba/list_eval_partition.txt" "$main/assets/stats/celeba_64_0c2d89311192_stats.npz"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
[[ -d "$main/celeba/img_align_celeba" ]] || { echo 'Missing CelebA images.' >&2; exit 1; }
available="$(df -PB1 "$(dirname "$output")" | awk 'NR==2 {print $4}')"
(( available >= 40 * 1024 * 1024 * 1024 )) || { echo 'Need at least 40 GiB free for the output.' >&2; exit 1; }

# Bind targets inside a read-only source mount must already exist as directories.
mkdir -p celeba workdirs logs assets/stats .tfds .torch_extensions_rtx .cache_rtx
mkdir -p "$output/workdirs" "$output/logs" "$main/.tfds" "$main/.torch_extensions_rtx" "$main/.cache_rtx"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
{
  printf 'mode=%s\ncontainer=%s\nphysical_gpu=%s\nimage=%s\nimage_id=%s\n' "$mode" "$name" "$gpu" "$image" "$image_id"
  printf 'source=%s\ncommit=%s\nstages=%s\n' "$source_dir" "$(git rev-parse HEAD)" "${stages[*]}"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv -i "$gpu" || true
} > "$output/logs/launch_${run_id}_${mode}.txt"

args=(run --name "$name" --gpus "device=$gpu" --ipc=host --shm-size=16g
  --mount "type=bind,src=$source_dir,dst=$root,readonly"
  --mount "type=bind,src=$output/workdirs,dst=$root/workdirs"
  --mount "type=bind,src=$output/logs,dst=$root/logs"
  --mount "type=bind,src=$main/celeba,dst=$root/celeba,readonly"
  --mount "type=bind,src=$main/assets/stats,dst=$root/assets/stats,readonly"
  --mount "type=bind,src=$main/.tfds,dst=$root/.tfds"
  --mount "type=bind,src=$main/.torch_extensions_rtx,dst=$root/.torch_extensions_rtx"
  --mount "type=bind,src=$main/.cache_rtx,dst=$root/.cache_rtx"
  --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
  --env TFDS_DATA_DIR="$root/.tfds" --env TORCH_EXTENSIONS_DIR="$root/.torch_extensions_rtx"
  --env XDG_CACHE_HOME="$root/.cache_rtx" --env TORCH_CUDA_ARCH_LIST=12.0
  --env MAX_JOBS="${MAX_JOBS:-8}" --entrypoint bash)
log="logs/${mode}_${run_id}.log"
if [[ "$mode" == smoke ]]; then
  docker "${args[@]}" --rm "$image_id" -c "bash run_kaffine_queue.sh ${stages[*]} 2>&1 | tee $log; exit \${PIPESTATUS[0]}"
  echo "Smoke finished; summary: $output/logs/smoke_done.txt, full log: $output/$log"
else
  docker "${args[@]}" -d "$image_id" -c "bash run_kaffine_queue.sh ${stages[*]} 2>&1 | tee $log; exit \${PIPESTATUS[0]}"
  docker inspect "$name" > "$output/logs/docker_inspect_${run_id}.json"
  printf 'Started %s on physical GPU %s.\nFollow: docker logs -f --tail 50 %s\nResults: %s\n' \
    "$name" "$gpu" "$name" "$output"
fi
