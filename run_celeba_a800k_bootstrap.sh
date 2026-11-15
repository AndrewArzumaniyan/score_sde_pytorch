#!/usr/bin/env bash
# CPU Docker, optional, started by hand: bootstrap intervals for DIAL 800k - A 800k and B 800k - A 800k
# on CelebA (celeba_a800k_bootstrap.py, the code of paper_diff_bootstrap.py). Run from a clean, separate
# Git worktree, by default ~/kovshov-arzumanyan/arzumand/score_sde_celeba_a800k_20261115, after both
# lanes of run_celeba_a800k_rtx.sh have finished.
#
#   bash run_celeba_a800k_bootstrap.sh check       synthetic checks and CPU tests
#   bash run_celeba_a800k_bootstrap.sh start       audit DIAL 800k, B 800k, A 800k; then the repeats
#   bash run_celeba_a800k_bootstrap.sh summarize   rebuild summary.{json,csv,md}
#
# Writes $output/celeba_a800k/ and, every 25 repeats and at the end, $output/summary.{json,csv,md}.
# A rerun continues the draws. Separate from paper_diff_bootstrap_20261115.
set -euo pipefail

mode="${1:?Usage: bash run_celeba_a800k_bootstrap.sh check|start|summarize}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${A800KB_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${A800KB_MAIN:-$base/score_sde_pytorch}"
output="${A800KB_OUTPUT:-$base/celeba_a800k_bootstrap_20261115}"
image="${A800KB_IMAGE:-score-sde-h200}"
cpus="${A800KB_CPUS:-8}"
mem="${A800KB_MEMORY:-16g}"
reps="${A800KB_REPS:-1000}"
part=celeba_a800k
name=paper-boot-a800k
lanes=(celeba-a800k-s0 celeba-a800k-s1)
files=(celeba_a800k_bootstrap.py paper_diff_bootstrap.py celeba_final_ode_bootstrap.py celeba_50k_lo_bootstrap.py)
reference="$main/assets/stats/celeba_64_0c2d89311192_stats.npz"
# Root key under /input and its folder; the same keys as ROOTS in celeba_a800k_bootstrap.py.
root_path() {
  case "$1" in
    fox800) echo "$base/celeba_fox800_lo_20261107" ;;
    gpu2q) echo "$base/gpu2_queue_20261114" ;;
    a800k) echo "$base/celeba_a800k_50k_20261115" ;;
  esac
}
running() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }

cd "$source_dir"
[[ $# -eq 0 ]] || { echo "$mode takes no arguments." >&2; exit 2; }
[[ "$cpus" =~ ^[1-9][0-9]*$ && "$reps" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPUs or repeats' >&2; exit 2; }
for path in "${files[@]}" run_celeba_a800k_bootstrap.sh tests/check_celeba_a800k_bootstrap.py; do
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
    'python celeba_a800k_bootstrap.py --check && python tests/check_celeba_a800k_bootstrap.py'
fi

case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
[[ "$output" != *','* ]] || { echo 'Bind paths may not contain commas' >&2; exit 2; }

# The code that ran first runs every later step.
mkdir -p "$output/runtime"
for file in "${files[@]}"; do
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
    --workdir /output --entrypoint python "$image_id" -u /code/celeba_a800k_bootstrap.py --summarize --out /output
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it before attempting a restart" >&2; exit 1
fi
for lane in "${lanes[@]}"; do
  [[ "$(running "$lane")" != true ]] || { echo "$lane still runs; start after both lanes finish." >&2; exit 1; }
done
[[ -s "$reference" ]] || { echo "Missing input: $reference" >&2; exit 1; }
mounts=(--mount "type=bind,src=$reference,dst=/input/reference_celeba.npz,readonly")
for key in fox800 gpu2q a800k; do
  path="$(root_path "$key")"
  [[ -d "$path/workdirs" && -d "$path/logs" ]] || { echo "Missing input: $path" >&2; exit 1; }
  case "$output/" in "$path/"*) echo 'Output must be separate from inputs' >&2; exit 2 ;; esac
  [[ "$path" != *','* ]] || { echo 'Bind paths may not contain commas' >&2; exit 2; }
  mounts+=(--mount "type=bind,src=$path,dst=/input/$key,readonly")
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 20 * 1024 * 1024 * 1024 )) || { echo 'Need at least 20 GiB free.' >&2; exit 1; }

# Every cell must be finished: a cell missing at audit time would fail its differences.
if ! docker run --rm "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${mounts[@]}" \
    --entrypoint python "$image_id" -u /code/celeba_a800k_bootstrap.py --part "$part" --inputs /input --out /tmp --missing; then
  echo "Inputs of part $part are not ready; nothing started." >&2; exit 1
fi
docker run -d --init --name "$name" "${common[@]}" --cpu-shares 128 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$output/runtime,dst=/code,readonly" \
  "${mounts[@]}" \
  --mount "type=bind,src=$output,dst=/output" --workdir /output \
  --entrypoint nice "$image_id" -n 19 python -u /code/celeba_a800k_bootstrap.py \
    --part "$part" --inputs /input --out /output --reps "$reps"
docker inspect "$name" > "$output/docker_inspect_$part.json"
echo "Started $name; docker logs -f --tail 30 $name; output: $output"
