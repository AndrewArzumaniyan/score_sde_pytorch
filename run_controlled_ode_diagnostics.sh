#!/usr/bin/env bash
# One-GPU diagnostics of controlled A/B under PF-ODE (controlled_ode_diagnostics.py).
# Usage: bash run_controlled_ode_diagnostics.sh --pilot|--run GPU_ID [TRAINING_SEED ...]
#   --pilot: tiny sizes (64 images/samples) on seed 42, a few minutes.
#   --run:   default seed 42; extra seeds 43/44 run sequentially in the same container.
# Overrides CONTROLLED_ODE_DIAG_{OUTPUT,IMAGE}; cache paths TFDS_DATA_DIR/TORCH_EXTENSIONS_DIR/XDG_CACHE_HOME.
set -euo pipefail
mode="${1:-}"; gpu="${2:-}"
[[ "$mode" == --pilot || "$mode" == --run ]] && [[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { sed -n '2,6p' "$0"; exit 2; }
shift 2
seeds=("$@"); [[ ${#seeds[@]} -gt 0 ]] || seeds=(42)
[[ "$mode" == --run ]] || seeds=(42)
repo="$(dirname "$(readlink -f "$0")")"
parent="$(dirname "$repo")"
root=/workspace/score_sde_pytorch
stage="${mode#--}"
output="$(realpath -m "${CONTROLLED_ODE_DIAG_OUTPUT:-$parent/controlled_ode_diagnostics_20261101_$stage}")"
name="controlled-ode-diagnostics-$stage"
image_id="$(docker image inspect --format '{{.Id}}' "${CONTROLLED_ODE_DIAG_IMAGE:-score-sde-h200}")"
git -C "$repo" ls-files --error-unmatch controlled_ode_diagnostics.py run_controlled_ode_diagnostics.sh >/dev/null
[[ -z "$(git -C "$repo" status --porcelain -- controlled_ode_diagnostics.py ode_sampling.py e2_noise_level_error.py)" ]] ||
  { echo 'Diagnostic sources differ from the committed revision.' >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Output exists: $output; inspect it, do not overwrite." >&2; exit 1; }
! docker inspect "$name" >/dev/null 2>&1 || { echo "Container exists: $name; inspect it first." >&2; exit 1; }
[[ -s "$repo/celeba/list_eval_partition.txt" ]] || { echo 'Missing local CelebA.' >&2; exit 1; }
input_mounts=(); commands=()
for seed in "${seeds[@]}"; do
  case "$seed" in
    42) src="$repo"; dst="$root"; works=(celeba_factorial_A_vp_pvp_50k celeba_factorial_B_vp_pfox_50k); ckpt=40 ;;
    43|44) src="$parent/ab_replication_seed$seed"; dst="/inputs/seed$seed"
      works=("celeba_ab_replication_A_vp_pvp_seed${seed}_200k" "celeba_ab_replication_B_vp_pfox_seed${seed}_200k"); ckpt=4 ;;
    *) echo "Unknown training seed: $seed" >&2; exit 2 ;;
  esac
  for work in "${works[@]}"; do
    [[ -s "$src/workdirs/$work/checkpoints/checkpoint_$ckpt.pth" && -s "$src/workdirs/$work/run_manifest.json" ]] ||
      { echo "Missing inputs: $src/workdirs/$work" >&2; exit 1; }
    [[ "$seed" == 42 ]] || input_mounts+=(--mount "type=bind,src=$src/workdirs/$work,dst=$dst/workdirs/$work,readonly")
  done
  extra=(); [[ "$mode" == --run ]] || extra=(--pilot)
  commands+=("python -u controlled_ode_diagnostics.py --root $dst --seed $seed --out /output/seed$seed ${extra[*]}")
done
tfds="${TFDS_DATA_DIR:-$root/.tfds}"
extensions="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
cache="${XDG_CACHE_HOME:-$root/.cache}"
mkdir -p "$repo/.tfds" "$repo/.torch_extensions" "$repo/.cache" "$output"
git -C "$repo" rev-parse HEAD > "$output/commit.txt"
printf 'image=%s\nphysical_gpu=%s\nseeds=%s\n' "$image_id" "$gpu" "${seeds[*]}" > "$output/docker.txt"
script="set -e; $(IFS=';'; echo "${commands[*]}")"
docker run -d --name "$name" --gpus "device=$gpu" --ipc=host --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$repo,dst=$root,readonly" \
  --mount "type=bind,src=$repo/.tfds,dst=$tfds" \
  --mount "type=bind,src=$repo/.torch_extensions,dst=$extensions" \
  --mount "type=bind,src=$repo/.cache,dst=$cache" \
  --mount "type=bind,src=$output,dst=/output" \
  "${input_mounts[@]}" --workdir "$root" \
  --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache" \
  --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}" \
  --entrypoint bash "$image_id" -c "$script"
docker inspect "$name" > "$output/docker_inspect_start.json"
echo "Started $name on physical GPU $gpu; follow: docker logs -f --tail 40 $name; results: $output"
