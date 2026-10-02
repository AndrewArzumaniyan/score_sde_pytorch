#!/usr/bin/env bash
# G1: continue controlled CelebA arms A and B from 100k to 200k updates.
# Run ONLY from the frozen tree (commit d476c8f + third_party + the three
# historical untracked scripts, source_sha256 ddf4c6ac...), mounted at
# /workspace/score_sde_pytorch, with the existing A/B workdirs mounted in place.
# Resumes from checkpoints-meta/checkpoint.pth; immutable checkpoint_N.pth files
# (1..20) are never overwritten, new snapshots are checkpoint_21..checkpoint_40.
#   G1_ARMS="A B"    arms to continue, sequentially (default both)
#   G1_DRY_RUN=1     run all preflight checks and print the plan, no training
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/g1_continue_ab_200k.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"

TRAIN_ITERS="${G1_TRAIN_ITERS:-200000}"
SNAPSHOT_FREQ="${FACTORIAL_SNAPSHOT_FREQ:-5000}"
SOURCE_CKPT="${G1_SOURCE_CKPT:-20}"
SEED="${FACTORIAL_SEED:-42}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"
ARMS_REQUESTED="${G1_ARMS:-A B}"
DRY_RUN="${G1_DRY_RUN:-0}"

declare -A CONFIGS=(
  [A]=configs/factorial/celeba_a_vp_pvp.py
  [B]=configs/factorial/celeba_b_vp_pfox.py
)
declare -A WORKDIRS=(
  [A]=workdirs/celeba_factorial_A_vp_pvp_50k
  [B]=workdirs/celeba_factorial_B_vp_pfox_50k
)

# The training manifest hashes these environment values: keep them exactly as in
# the original A/B runs (TFDS_DATA_DIR set, the other two unset).
[[ "${TFDS_DATA_DIR}" == /workspace/score_sde_pytorch/.tfds ]] || {
  echo "TFDS_DATA_DIR must be /workspace/score_sde_pytorch/.tfds (it is part of the manifest)." >&2; exit 2; }
[[ -z "${CELEBA_DIR:-}" && -z "${EDM_OFFICIAL_ROOT:-}" ]] || {
  echo "CELEBA_DIR/EDM_OFFICIAL_ROOT must be unset (they are part of the manifest)." >&2; exit 2; }
[[ "${TRAIN_ITERS}" =~ ^[0-9]+$ && "${TRAIN_ITERS}" -gt 100000 ]] || {
  echo "G1_TRAIN_ITERS must be an integer above 100000." >&2; exit 2; }
for arm in ${ARMS_REQUESTED}; do
  [[ "${arm}" == A || "${arm}" == B ]] || { echo "Invalid arm: ${arm}" >&2; exit 2; }
done
[[ -d celeba/img_align_celeba ]] || {
  echo "Missing celeba/img_align_celeba: mount the main tree's celeba directory." >&2; exit 1; }
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

for arm in ${ARMS_REQUESTED}; do
  workdir="${WORKDIRS[${arm}]}"
  for required in "${workdir}/run_manifest.json" \
                  "${workdir}/checkpoints-meta/checkpoint.pth" \
                  "${workdir}/checkpoints/checkpoint_${SOURCE_CKPT}.pth"; do
    [[ -f "${required}" ]] || {
      echo "Refusing to start arm ${arm} from scratch: missing ${required}" >&2; exit 1; }
  done
  latest="$(ls "${workdir}/checkpoints" | sed -n 's/^checkpoint_\([0-9]\+\)\.pth$/\1/p' | sort -n | tail -n 1)"
  [[ "${latest}" == "${SOURCE_CKPT}" ]] || {
    echo "Arm ${arm}: latest numbered checkpoint is ${latest}, expected ${SOURCE_CKPT}." >&2
    echo "The workdir was already advanced; decide explicitly before continuing." >&2; exit 1; }
done

echo "=== G1 A/B continuation $(date --iso-8601=seconds) ==="
echo "Arms: ${ARMS_REQUESTED}; target ${TRAIN_ITERS} updates; snapshot_freq=${SNAPSHOT_FREQ}; seed=${SEED}"
for arm in ${ARMS_REQUESTED}; do
  ls -la --time-style=full-iso "${WORKDIRS[${arm}]}/checkpoints-meta/checkpoint.pth"
done
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Dry run: preflight checks passed, no training started."
  exit 0
fi

for arm in ${ARMS_REQUESTED}; do
  echo "=== Continuing arm ${arm} -> ${TRAIN_ITERS} updates ($(date --iso-8601=seconds)) ==="
  python -u main.py \
    --mode=train \
    --config="${CONFIGS[${arm}]}" \
    --workdir="${WORKDIRS[${arm}]}" \
    --config.training.batch_size=128 \
    --config.training.n_iters="${TRAIN_ITERS}" \
    --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
    --config.training.snapshot_freq_for_preemption="${SNAPSHOT_FREQ}" \
    --config.training.snapshot_sampling=False \
    --config.training.log_freq=50 \
    --config.training.eval_freq=500 \
    --config.seed="${SEED}" \
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
done
echo "=== G1 A/B continuation finished: $(date --iso-8601=seconds) ==="
