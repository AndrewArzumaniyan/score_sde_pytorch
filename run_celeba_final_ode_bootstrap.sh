#!/usr/bin/env bash
# Docker-only CPU bootstrap of FOX-VP for the final ODE queue (+ historical EM comparison).
# Usage: bash run_celeba_final_ode_bootstrap.sh --pilot|--full
#   --pilot: audit inputs, reproduce 10 point FIDs, 3 repeats.
#   --full:  continue the same output to CELEBA_FINAL_ODE_BOOT_REPS repeats (default 1000, total, not extra).
# Overrides CELEBA_FINAL_ODE_BOOT_{ODE,EM,OUTPUT,IMAGE,CPUS,MEMORY,REPS}.
# A stopped container is resumed with docker start <name>; draws.jsonl is appended, not rewritten.
set -euo pipefail
setting() { printenv "$1" || printf '%s\n' "$2"; }
mode="${1:-}"
case "$mode" in --pilot) reps=3 ;; --full) reps="$(setting CELEBA_FINAL_ODE_BOOT_REPS 1000)" ;;
  *) sed -n '2,7p' "$0"; exit 2 ;; esac
[[ "$reps" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid repeat count.' >&2; exit 2; }
repo="$(dirname "$(readlink -f "$0")")"
parent="$(dirname "$repo")"
root=/workspace/score_sde_pytorch
ode="$(readlink -e "$(setting CELEBA_FINAL_ODE_BOOT_ODE "$parent/celeba_final_ode_20261031")")"
em="$(readlink -e "$(setting CELEBA_FINAL_ODE_BOOT_EM "$repo")")"
out="$(realpath -m "$(setting CELEBA_FINAL_ODE_BOOT_OUTPUT "$parent/celeba_final_ode_bootstrap_20261101")")"
cpus="$(setting CELEBA_FINAL_ODE_BOOT_CPUS 8)"
memory="$(setting CELEBA_FINAL_ODE_BOOT_MEMORY 8g)"
image_id="$(docker image inspect --format '{{.Id}}' "$(setting CELEBA_FINAL_ODE_BOOT_IMAGE score-sde-h200)")"
[[ "$(cat "$ode/logs/queue.state")" == completed ]] || { echo 'ODE queue is not completed.' >&2; exit 1; }
reference="$em/assets/stats/celeba_64_0c2d89311192_stats.npz"
[[ -s "$reference" ]] || { echo "Missing reference: $reference" >&2; exit 1; }
work_mounts=()
for work in celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k; do
  [[ -d "$em/workdirs/$work" ]] || { echo "Missing EM workdir: $work" >&2; exit 1; }
  work_mounts+=(--mount "type=bind,src=$em/workdirs/$work,dst=/input/em/workdirs/$work,readonly")
done
for input in "$repo" "$ode" "$em"; do
  case "$out/" in "$input/"*) echo 'Output must be outside the inputs.' >&2; exit 2 ;; esac
done
[[ "$out" != *','* && "$ode" != *','* && "$em" != *','* ]] || { echo 'Bind paths must not contain commas.' >&2; exit 2; }
name="celeba-final-ode-bootstrap-${mode#--}"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container exists: $name; inspect it, then resume with docker start if appropriate." >&2; exit 1
fi
mkdir -p "$out/runtime"
snapshot="$out/runtime/celeba_final_ode_bootstrap.py"
if [[ "$mode" == --pilot ]]; then
  [[ ! -e "$out/audit.json" ]] || { echo "Output already audited: $out; use --full or a new output." >&2; exit 1; }
  cp -- "$repo/celeba_final_ode_bootstrap.py" "$snapshot"
  printf '%s\n' "$image_id" > "$out/runtime/image_id.txt"
  git -C "$repo" rev-parse HEAD > "$out/runtime/git_head.txt"
  prepare=--prepare
else
  [[ -s "$out/point_checks.json" ]] || { echo 'Run the audited --pilot first.' >&2; exit 1; }
  cmp -s "$repo/celeba_final_ode_bootstrap.py" "$snapshot" || { echo 'Script changed since pilot; use a new output.' >&2; exit 1; }
  image_id="$(cat "$out/runtime/image_id.txt")"
  prepare=
fi
docker run -d --init --name "$name" --network none \
  --cpus "$cpus" --cpu-shares 128 --memory "$memory" --memory-swap "$memory" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$snapshot,dst=/opt/celeba_final_ode_bootstrap.py,readonly" \
  --mount "type=bind,src=$ode,dst=/input/ode,readonly" \
  --mount "type=bind,src=$reference,dst=/input/reference.npz,readonly" \
  "${work_mounts[@]}" \
  --mount "type=bind,src=$out,dst=/output" --workdir /output \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -e OPENBLAS_NUM_THREADS="$cpus" -e OMP_NUM_THREADS="$cpus" -e MKL_NUM_THREADS="$cpus" \
  -e TFDS_DATA_DIR="$(setting TFDS_DATA_DIR "$root/.tfds")" \
  -e TORCH_EXTENSIONS_DIR="$(setting TORCH_EXTENSIONS_DIR "$root/.torch_extensions")" \
  -e XDG_CACHE_HOME="$(setting XDG_CACHE_HOME "$root/.cache")" \
  --entrypoint nice "$image_id" -n 19 python -I -u /opt/celeba_final_ode_bootstrap.py \
    --ode /input/ode --em /input/em --reference /input/reference.npz --out /output --reps "$reps" $prepare
docker inspect "$name" > "$out/docker_inspect_${mode#--}.json"
echo "Started $name; progress: docker logs -f --tail 20 $name; results: $out"
