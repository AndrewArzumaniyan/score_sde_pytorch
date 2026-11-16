#!/usr/bin/env bash
# CPU Docker: bootstrap intervals for the 50k differences at 257 NFE (nfe_limit_bootstrap.py).
# Run from a clean, separate Git worktree.
#
#   bash run_nfe_limit_bootstrap.sh check       synthetic checks and CPU tests
#   bash run_nfe_limit_bootstrap.sh start       detached: audit of 3 cells, then the repeats
#   bash run_nfe_limit_bootstrap.sh summarize   rebuild summary.{json,md} from the draws so far
#
# Output: $output/summary.{json,md}, every 25 repeats and at the end. A rerun continues the draws.
set -euo pipefail

mode="${1:?Usage: bash run_nfe_limit_bootstrap.sh check|start|summarize}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${NLB_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${NLB_MAIN:-$base/score_sde_pytorch}"
inputs="${NLB_INPUTS:-$base/nfe_limit_20261116}"
output="${NLB_OUTPUT:-$base/nfe_limit_20261116/bootstrap}"
image="${NLB_IMAGE:-score-sde-h200}"
cpus="${NLB_CPUS:-8}"
mem="${NLB_MEMORY:-16g}"
reps="${NLB_REPS:-1000}"
name=celeba-nfe-boot
files=(nfe_limit_bootstrap.py paper_diff_bootstrap.py celeba_final_ode_bootstrap.py celeba_50k_lo_bootstrap.py)
reference="$main/assets/stats/celeba_64_0c2d89311192_stats.npz"

cd "$source_dir"
[[ "$cpus" =~ ^[1-9][0-9]*$ && "$reps" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPUs or repeats' >&2; exit 2; }
for path in "${files[@]}" run_nfe_limit_bootstrap.sh tests/check_nfe_limit_bootstrap.py; do
  git ls-files --error-unmatch "$path" >/dev/null 2>&1 || { echo "Not committed: $path." >&2; exit 1; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo 'Tracked files differ from HEAD; use a clean worktree.' >&2; exit 1; }
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
common=(--network none --cpus "$cpus" --memory "$mem" --memory-swap "$mem"
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1
  -e OPENBLAS_NUM_THREADS="$cpus" -e OMP_NUM_THREADS="$cpus" -e MKL_NUM_THREADS="$cpus")

if [[ "$mode" == check ]]; then
  exec docker run --rm "${common[@]}" --mount "type=bind,src=$source_dir,dst=/code,readonly" \
    --workdir /code --entrypoint bash "$image_id" -c \
    'python nfe_limit_bootstrap.py --check && python tests/check_nfe_limit_bootstrap.py'
fi

case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
for lane in dial b a; do
  case "$output/" in "$inputs/$lane/"*) echo 'Output must be separate from the lane folders.' >&2; exit 2 ;; esac
done
mkdir -p "$output/runtime"
for file in "${files[@]}"; do  # The code of the first run runs every later one.
  if [[ -e "$output/runtime/$file" ]]; then
    cmp -s "$file" "$output/runtime/$file" || { echo "Script differs from the first run: $file" >&2; exit 1; }
  else
    cp "$file" "$output/runtime/$file"
  fi
done
[[ -s "$output/runtime/image_id.txt" ]] || printf '%s\n' "$image_id" > "$output/runtime/image_id.txt"
image_id="$(cat "$output/runtime/image_id.txt")"
[[ -s "$output/runtime/git_head.txt" ]] || git rev-parse HEAD > "$output/runtime/git_head.txt"

if [[ "$mode" == summarize ]]; then
  exec docker run --rm "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output/runtime,dst=/code,readonly" --mount "type=bind,src=$output,dst=/output" \
    --workdir /output --entrypoint python "$image_id" -u /code/nfe_limit_bootstrap.py --summarize --out /output
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it before a restart" >&2; exit 1
fi
[[ -s "$reference" ]] || { echo "Missing input: $reference" >&2; exit 1; }
mounts=(--mount "type=bind,src=$reference,dst=/input/reference_celeba.npz,readonly")
for lane in dial b a; do
  [[ -d "$inputs/$lane/workdirs" && -d "$inputs/$lane/logs" ]] || { echo "Missing input: $inputs/$lane" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$inputs/$lane,dst=/input/$lane,readonly")
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 5 * 1024 * 1024 * 1024 )) || { echo 'Need at least 5 GiB free.' >&2; exit 1; }
if ! docker run --rm "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${mounts[@]}" \
    --entrypoint python "$image_id" -u /code/nfe_limit_bootstrap.py --inputs /input --missing; then
  echo 'Inputs are not ready; nothing started.' >&2; exit 1
fi
docker run -d --init --name "$name" "${common[@]}" --cpu-shares 128 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${mounts[@]}" \
  --mount "type=bind,src=$output,dst=/output" --workdir /output \
  --entrypoint nice "$image_id" -n 19 python -u /code/nfe_limit_bootstrap.py --inputs /input --out /output --reps "$reps"
docker inspect "$name" > "$output/docker_inspect.json"
echo "Started $name; docker logs -f --tail 30 $name; output: $output"
