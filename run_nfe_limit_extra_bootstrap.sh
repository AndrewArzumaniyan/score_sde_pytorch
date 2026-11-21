#!/usr/bin/env bash
# CPU Docker: bootstrap intervals for the 257-NFE cells of nfe_limit_extra_eval.py (nfe_limit_extra_bootstrap.py),
# combined with the finished draws of nfe_limit_20261116/bootstrap. Run from a clean, separate Git worktree.
#
#   bash run_nfe_limit_extra_bootstrap.sh check       synthetic checks and CPU tests
#   bash run_nfe_limit_extra_bootstrap.sh start       detached: audit of 3 cells, then the repeats
#   bash run_nfe_limit_extra_bootstrap.sh summarize   rebuild summary.* and combined.* from the draws so far
#   bash run_nfe_limit_extra_bootstrap.sh combine     rebuild combined.* only
#
# Output: $output/summary.{json,md} and $output/combined.{json,md}, every 25 repeats and at the end.
# A rerun continues the draws. NEB_CPUS sets the CPU limit (default 16).
set -euo pipefail

mode="${1:?Usage: bash run_nfe_limit_extra_bootstrap.sh check|start|summarize|combine}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${NEB_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${NEB_MAIN:-$base/score_sde_pytorch}"
inputs="${NEB_INPUTS:-$base/nfe_limit_extra_20261116}"
previous="${NEB_PREVIOUS:-$base/nfe_limit_20261116/bootstrap}"
output="${NEB_OUTPUT:-$base/nfe_limit_extra_20261116/bootstrap}"
image="${NEB_IMAGE:-score-sde-h200}"
cpus="${NEB_CPUS:-16}"
mem="${NEB_MEMORY:-16g}"
reps="${NEB_REPS:-1000}"
name=celeba-nfe-extra-boot
files=(nfe_limit_extra_bootstrap.py nfe_limit_bootstrap.py paper_diff_bootstrap.py celeba_final_ode_bootstrap.py
       celeba_50k_lo_bootstrap.py)
reference="$main/assets/stats/celeba_64_0c2d89311192_stats.npz"
lanes=(a500 fox43 a43)

cd "$source_dir"
[[ "$cpus" =~ ^[1-9][0-9]*$ && "$reps" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPUs or repeats' >&2; exit 2; }
for path in "${files[@]}" run_nfe_limit_extra_bootstrap.sh tests/check_nfe_limit_extra_bootstrap.py; do
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
    'python nfe_limit_extra_bootstrap.py --check && python tests/check_nfe_limit_extra_bootstrap.py'
fi

case "$output/" in "$source_dir/"*|"$main/"*|"$previous/"*) echo 'Output must be outside the repositories and the previous run.' >&2; exit 2 ;; esac
for lane in "${lanes[@]}"; do
  case "$output/" in "$inputs/$lane/"*) echo 'Output must be separate from the lane folders.' >&2; exit 2 ;; esac
done
[[ -s "$previous/summary.json" && -s "$previous/draws.jsonl" ]] || { echo "Missing previous run: $previous" >&2; exit 1; }
prev_mount=(--mount "type=bind,src=$previous,dst=/previous,readonly")
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

if [[ "$mode" == summarize || "$mode" == combine ]]; then
  exec docker run --rm "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${prev_mount[@]}" \
    --mount "type=bind,src=$output,dst=/output" --workdir /output --entrypoint python "$image_id" \
    -u /code/nfe_limit_extra_bootstrap.py "--$mode" --out /output --previous /previous
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it before a restart" >&2; exit 1
fi
[[ -s "$reference" ]] || { echo "Missing input: $reference" >&2; exit 1; }
mounts=(--mount "type=bind,src=$reference,dst=/input/reference_celeba.npz,readonly")
for lane in "${lanes[@]}"; do
  [[ -d "$inputs/$lane/workdirs" && -d "$inputs/$lane/logs" ]] || { echo "Missing input: $inputs/$lane" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$inputs/$lane,dst=/input/$lane,readonly")
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 5 * 1024 * 1024 * 1024 )) || { echo 'Need at least 5 GiB free.' >&2; exit 1; }
if ! docker run --rm "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${mounts[@]}" "${prev_mount[@]}" \
    --entrypoint bash "$image_id" -c \
    'python -u /code/nfe_limit_extra_bootstrap.py --inputs /input --missing &&
     python -u /code/nfe_limit_extra_bootstrap.py --previous /previous --check-previous'; then
  echo 'Inputs or the previous run are not ready; nothing started.' >&2; exit 1
fi
docker run -d --init --name "$name" "${common[@]}" --cpu-shares 128 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${mounts[@]}" "${prev_mount[@]}" \
  --mount "type=bind,src=$output,dst=/output" --workdir /output \
  --entrypoint nice "$image_id" -n 19 python -u /code/nfe_limit_extra_bootstrap.py --inputs /input --out /output \
  --previous /previous --reps "$reps"
docker inspect "$name" > "$output/docker_inspect.json"
echo "Started $name; docker logs -f --tail 30 $name; output: $output"
