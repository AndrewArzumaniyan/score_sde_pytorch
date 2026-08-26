#!/usr/bin/env bash
# Two optimizer steps per AFHQv2 variant using the real prepared dataset.
# This verifies data -> model -> loss/backward -> EMA -> checkpoint only;
# it does not produce or evaluate quality metrics.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-${ROOT_DIR}/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${ROOT_DIR}/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
AFHQV2_DIR="${AFHQV2_DIR:-${ROOT_DIR}/datasets/afhqv2-64x64}"

if [[ ! -f "${AFHQV2_DIR}/dataset.json" ]]; then
  echo "Prepared AFHQv2-64 not found at ${AFHQV2_DIR}." >&2
  echo "Run tools/prepare_afhqv2_64.sh first or set AFHQV2_DIR." >&2
  exit 1
fi
EXPECTED_IMAGES="${AFHQV2_EXPECTED_IMAGES:-15803}"
IMAGE_COUNT="$(find "${AFHQV2_DIR}" -type f -iname '*.png' | wc -l | tr -d ' ')"
if [[ "${IMAGE_COUNT}" != "${EXPECTED_IMAGES}" ]]; then
  echo "Expected ${EXPECTED_IMAGES} prepared AFHQv2 images, found ${IMAGE_COUNT}." >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    vp ve cosine_vp edm edm_canonical fox_gaussian fox_matern32
    fox_matern32_vpdrift
  )
fi

tools/install_official_edm.sh
SMOKE_ROOT="$(mktemp -d /tmp/afhqv2-smoke.XXXXXX)"
cleanup() {
  if [[ "${SMOKE_ROOT}" == /tmp/afhqv2-smoke.* && -d "${SMOKE_ROOT}" ]]; then
    rm -rf "${SMOKE_ROOT}"
  fi
}
trap cleanup EXIT

config_for_model() {
  case "$1" in
    vp) echo configs/vp/afhqv2_ncsnpp_continuous.py ;;
    ve) echo configs/ve/afhqv2_ncsnpp_continuous.py ;;
    cosine_vp) echo configs/vp/afhqv2_ncsnpp_cosine_continuous.py ;;
    edm) echo configs/edm/afhqv2_ncsnpp.py ;;
    edm_canonical) echo configs/edm/afhqv2_canonical.py ;;
    fox_gaussian) echo configs/fox/afhqv2_ncsnpp_continuous.py ;;
    fox_matern32) echo configs/fox/afhqv2_ncsnpp_continuous_matern_3_2.py ;;
    fox_matern32_vpdrift) echo configs/fox/afhqv2_ncsnpp_continuous_matern_3_2_vp_drift.py ;;
    *) echo "Unknown model: $1" >&2; return 1 ;;
  esac
}

for model in "${MODELS[@]}"; do
  config="$(config_for_model "${model}")"
  workdir="${SMOKE_ROOT}/${model}"
  echo "=== AFHQv2 smoke: ${model} ==="
  smoke_command=(python -u main.py \
    --mode=train \
    --config="${config}" \
    --workdir="${workdir}" \
    --config.data.afhqv2_dir="${AFHQV2_DIR}" \
    --config.data.afhqv2_train_take=16 \
    --config.data.afhqv2_eval_take=16 \
    --config.training.batch_size=2 \
    --config.training.n_iters=1 \
    --config.training.snapshot_freq=1 \
    --config.training.snapshot_freq_for_preemption=1 \
    --config.training.snapshot_sampling=False \
    --config.training.log_freq=1 \
    --config.training.eval_freq=1000000 \
    --config.seed=42)
  if [[ "${model}" == edm_canonical ]]; then
    # Accumulation is already tested independently; keep this GPU smoke small.
    smoke_command+=(
      --config.training.gradient_accumulation_steps=1
      --config.training.effective_batch_size=2
    )
  fi
  "${smoke_command[@]}"
  if [[ ! -f "${workdir}/checkpoints/checkpoint_1.pth" ]]; then
    echo "Smoke run did not publish checkpoint_1.pth for ${model}." >&2
    exit 1
  fi
  rm -rf "${workdir}"
done

echo "All requested AFHQv2 model smoke runs passed."
