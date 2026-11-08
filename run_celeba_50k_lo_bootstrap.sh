#!/usr/bin/env bash
# CPU Docker, 16 CelebA cells at 50k on the lower window: --check, --pilot (audit + 3 repeats), --full (1000 total).
set -euo pipefail
mode="${1:-}"
[[ $# == 1 && ( "$mode" == --check || "$mode" == --pilot || "$mode" == --full ) ]] || {
  echo "Usage: bash $0 --check|--pilot|--full" >&2; exit 2;
}
repo="$(dirname "$(readlink -f "$0")")"
parent="$(dirname "$repo")"
image="${LO50K_BOOT_IMAGE:-score-sde-h200}"
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
cpus="${LO50K_BOOT_CPUS:-8}"
mem="${LO50K_BOOT_MEMORY:-8g}"
[[ "$cpus" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid CPUs' >&2; exit 2; }
for file in celeba_50k_lo_bootstrap.py celeba_final_ode_bootstrap.py; do
  git -C "$repo" ls-files --error-unmatch "$file" >/dev/null
  git -C "$repo" diff --quiet HEAD -- "$file" || { echo "Commit changes to $file first" >&2; exit 2; }
done
common=(--network none --cpus "$cpus" --memory "$mem" --memory-swap "$mem"
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1
  -e OPENBLAS_NUM_THREADS="$cpus" -e OMP_NUM_THREADS="$cpus" -e MKL_NUM_THREADS="$cpus")
if [[ "$mode" == --check ]]; then
  exec docker run --rm "${common[@]}" --mount "type=bind,src=$repo,dst=/code,readonly" \
    --workdir /code --entrypoint python "$image_id" /code/celeba_50k_lo_bootstrap.py --check
fi
input="$(readlink -e "${LO50K_BOOT_INPUT:-$parent/celeba_50k_lo_20261107}")"
fox800="$(readlink -e "${LO50K_BOOT_FOX800:-$parent/celeba_fox800_lo_20261107}")"
reference="$(readlink -e "${LO50K_BOOT_REFERENCE:-$parent/score_sde_pytorch/assets/stats/celeba_64_0c2d89311192_stats.npz}")"
out="$(realpath -m "${LO50K_BOOT_OUTPUT:-$parent/celeba_50k_lo_bootstrap_20261107}")"
for summary in "$input/gpu1/logs/summary.json" "$input/gpu2/logs/summary.json" "$fox800/logs/summary.json"; do
  grep -q '"status": "completed"' "$summary" || { echo "Queue is not completed: $summary" >&2; exit 1; }
done
for path in "$repo" "$input" "$fox800" "$reference"; do
  case "$out/" in "$path/"*) echo 'Output must be separate from inputs and code' >&2; exit 2;; esac
done
[[ "$out$input$fox800$reference" != *','* ]] || { echo 'Bind paths may not contain commas' >&2; exit 2; }
name="celeba-50k-lo-bootstrap-${mode#--}"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it before attempting a restart" >&2; exit 1
fi
if [[ "$mode" == --pilot ]]; then
  [[ ! -e "$out" ]] || { echo "Output already exists: $out" >&2; exit 1; }
  mkdir -p "$out/runtime"
  cp "$repo/celeba_50k_lo_bootstrap.py" "$repo/celeba_final_ode_bootstrap.py" "$out/runtime/"
  printf '%s\n' "$image_id" > "$out/runtime/image_id.txt"
  git -C "$repo" rev-parse HEAD > "$out/runtime/git_head.txt"
  reps=3
  prepare=(--prepare)
else
  [[ -s "$out/summary.json" && -s "$out/point_checks.json" ]] || { echo 'Complete pilot first' >&2; exit 1; }
  for file in celeba_50k_lo_bootstrap.py celeba_final_ode_bootstrap.py; do
    cmp -s "$repo/$file" "$out/runtime/$file" || { echo "Script differs from pilot: $file" >&2; exit 1; }
  done
  image_id="$(cat "$out/runtime/image_id.txt")"
  reps="${LO50K_BOOT_REPS:-1000}"
  [[ "$reps" =~ ^[1-9][0-9]*$ && "$reps" -ge 3 ]] || { echo 'Invalid repeat count' >&2; exit 2; }
  prepare=()
fi
docker run -d --init --name "$name" "${common[@]}" --cpu-shares 128 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$out/runtime,dst=/code,readonly" \
  --mount "type=bind,src=$input,dst=/input/lo,readonly" \
  --mount "type=bind,src=$fox800,dst=/input/fox800,readonly" \
  --mount "type=bind,src=$reference,dst=/input/reference.npz,readonly" \
  --mount "type=bind,src=$out,dst=/output" --workdir /output \
  --entrypoint nice "$image_id" -n 19 python -u /code/celeba_50k_lo_bootstrap.py \
    --root /input/lo --root-fox800 /input/fox800 --reference /input/reference.npz --out /output --reps "$reps" "${prepare[@]}"
docker inspect "$name" > "$out/docker_inspect_${mode#--}.json"
echo "Started $name; docker logs -f --tail 30 $name; output: $out"
