#!/usr/bin/env bash
# CPU Docker: conditional bootstrap intervals for the new 50k differences (paper_diff_bootstrap.py).
# Run from a clean, separate Git worktree, by default ~/kovshov-arzumanyan/arzumand/score_sde_afhq200k_20261115.
#
#   bash run_paper_diff_bootstrap.sh check               synthetic checks and CPU tests
#   bash run_paper_diff_bootstrap.sh start celeba        now: DIAL 800k, B 800k, A 500k, A 400k
#   bash run_paper_diff_bootstrap.sh start afhq          now: DIAL 200k, A 250k, VP 250k
#   bash run_paper_diff_bootstrap.sh after-part1         in the background: wait until afhqv2-200k-a and
#                                                        afhqv2-200k-vp end by themselves with code 0, then
#                                                        start afhq200k (A 200k, VP 200k)
#   bash run_paper_diff_bootstrap.sh start afhq200k      the same at once, if part 1 has already finished
#   bash run_paper_diff_bootstrap.sh summarize           rebuild summary.{json,csv,md} from all parts
#
# Each part writes $output/PART/ and, every 25 repeats and at the end, summary.{json,csv,md} over all
# parts present. A failed cell fails only its differences. A rerun continues the draws of a part.
set -euo pipefail

mode="${1:?Usage: bash run_paper_diff_bootstrap.sh check|start PART|after-part1|summarize}"
shift
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${PDB_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="${PDB_MAIN:-$base/score_sde_pytorch}"
output="${PDB_OUTPUT:-$base/paper_diff_bootstrap_20261115}"
image="${PDB_IMAGE:-score-sde-h200}"
cpus="${PDB_CPUS:-8}"
mem="${PDB_MEMORY:-16g}"
reps="${PDB_REPS:-1000}"
wait_seconds="${PDB_WAIT_SECONDS:-60}"
wait_max_seconds="${PDB_WAIT_MAX_SECONDS:-86400}"
after=(afhqv2-200k-a afhqv2-200k-vp)
files=(paper_diff_bootstrap.py celeba_final_ode_bootstrap.py celeba_50k_lo_bootstrap.py)
celeba_ref="$main/assets/stats/celeba_64_0c2d89311192_stats.npz"
afhq_ref="$main/assets/stats/afhqv2_64_cd017afab009_stats.npz"
# Root key under /input and its folder; the same keys as ROOTS in paper_diff_bootstrap.py.
root_path() {
  case "$1" in
    fox800) echo "$base/celeba_fox800_lo_20261107" ;;
    lo_gpu1) echo "$base/celeba_50k_lo_20261107/gpu1" ;;
    gpu2q) echo "$base/gpu2_queue_20261114" ;;
    gpu2a) echo "$base/gpu2_a50k_grids_20261114" ;;
    gpu1vp) echo "$base/gpu1_vp_tail_20261114" ;;
    afhq200k_a) echo "$base/afhq200k_bootstrap_20261115/a" ;;
    afhq200k_vp) echo "$base/afhq200k_bootstrap_20261115/vp" ;;
  esac
}
part_roots() {
  case "$1" in
    celeba) echo fox800 lo_gpu1 gpu2q gpu2a ;;
    afhq) echo gpu2q gpu1vp ;;
    afhq200k) echo afhq200k_a afhq200k_vp ;;
  esac
}
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
running() { local value; if value="$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)"; then echo "$value"; else echo gone; fi; }

if [[ "$mode" == _wait ]]; then
  echo "$(stamp) waiting until ${after[*]} end by themselves with code 0, at most $wait_max_seconds s"
  for name in "${after[@]}"; do
    while [[ "$(running "$name")" == true ]]; do
      (( SECONDS < wait_max_seconds )) || { echo "$(stamp) waited $SECONDS s: $name still runs; nothing started"; exit 1; }
      sleep "$wait_seconds"
    done
    code="$(docker inspect --format '{{.State.ExitCode}}' "$name" 2>/dev/null || echo unknown)"
    echo "$(stamp) $name stopped, exit code $code"
    [[ "$code" == 0 ]] || { echo "$(stamp) $name did not exit with 0; nothing started"; exit 1; }
  done
  if ! bash "$source_dir/run_paper_diff_bootstrap.sh" start afhq200k; then
    echo "$(stamp) the start refused; nothing started"; exit 1
  fi
  echo "$(stamp) launcher finished"
  exit 0
fi

cd "$source_dir"
[[ "$cpus" =~ ^[1-9][0-9]*$ && "$reps" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPUs or repeats' >&2; exit 2; }
for path in "${files[@]}" run_paper_diff_bootstrap.sh tests/check_paper_diff_bootstrap.py; do
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
    'python paper_diff_bootstrap.py --check && python tests/check_paper_diff_bootstrap.py'
fi

case "$output/" in "$source_dir/"*|"$main/"*) echo 'Output must be outside the repositories.' >&2; exit 2 ;; esac
[[ "$output" != *','* ]] || { echo 'Bind paths may not contain commas' >&2; exit 2; }

if [[ "$mode" == after-part1 ]]; then
  [[ $# -eq 0 ]] || { echo 'after-part1 takes no arguments.' >&2; exit 2; }
  for name in "${after[@]}"; do
    docker inspect "$name" >/dev/null 2>&1 || { echo "No container $name: start part 1 first." >&2; exit 1; }
  done
  docker inspect paper-boot-afhq200k >/dev/null 2>&1 && { echo 'Container exists: paper-boot-afhq200k.' >&2; exit 1; }
  mkdir -p "$output"
  pid_file="$output/wait.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Already waiting: process $(cat "$pid_file")." >&2; exit 1
  fi
  log="$output/wait_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash "$source_dir/run_paper_diff_bootstrap.sh" _wait >> "$log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "Waiting for ${after[*]} in process $!. Log: $log"
  exit 0
fi

# The code that ran the first part runs every later one.
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
    --workdir /output --entrypoint python "$image_id" -u /code/paper_diff_bootstrap.py --summarize --out /output
fi

[[ "$mode" == start ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ $# -eq 1 ]] || { echo 'Expected one part: celeba, afhq or afhq200k.' >&2; exit 2; }
part="$1"
[[ "$part" == celeba || "$part" == afhq || "$part" == afhq200k ]] || { echo "Unknown part: $part" >&2; exit 2; }
name="paper-boot-$part"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it before attempting a restart" >&2; exit 1
fi
if [[ "$part" == afhq200k ]]; then
  for container in "${after[@]}"; do
    [[ "$(running "$container")" != true ]] || { echo "$container still runs; use after-part1." >&2; exit 1; }
  done
fi
reference="$afhq_ref"
[[ "$part" != celeba ]] || reference="$celeba_ref"
[[ -s "$reference" ]] || { echo "Missing input: $reference" >&2; exit 1; }
mounts=(--mount "type=bind,src=$reference,dst=/input/reference_$([[ $part == celeba ]] && echo celeba || echo afhqv2).npz,readonly")
for key in $(part_roots "$part"); do
  path="$(root_path "$key")"
  [[ -d "$path/workdirs" && -d "$path/logs" ]] || { echo "Missing input: $path" >&2; exit 1; }
  case "$output/" in "$path/"*) echo 'Output must be separate from inputs' >&2; exit 2 ;; esac
  [[ "$path" != *','* ]] || { echo 'Bind paths may not contain commas' >&2; exit 2; }
  mounts+=(--mount "type=bind,src=$path,dst=/input/$key,readonly")
done
available="$(df -PB1 "$output" | awk 'NR==2 {print $4}')"
(( available >= 20 * 1024 * 1024 * 1024 )) || { echo 'Need at least 20 GiB free.' >&2; exit 1; }

# Every cell of the part must be finished: a cell missing at audit time would fail its differences.
if ! docker run --rm "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output/runtime,dst=/code,readonly" "${mounts[@]}" \
    --entrypoint python "$image_id" -u /code/paper_diff_bootstrap.py --part "$part" --inputs /input --out /tmp --missing; then
  echo "Inputs of part $part are not ready; nothing started." >&2; exit 1
fi
docker run -d --init --name "$name" "${common[@]}" --cpu-shares 128 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$output/runtime,dst=/code,readonly" \
  "${mounts[@]}" \
  --mount "type=bind,src=$output,dst=/output" --workdir /output \
  --entrypoint nice "$image_id" -n 19 python -u /code/paper_diff_bootstrap.py \
    --part "$part" --inputs /input --out /output --reps "$reps"
docker inspect "$name" > "$output/docker_inspect_$part.json"
echo "Started $name; docker logs -f --tail 30 $name; output: $output"
