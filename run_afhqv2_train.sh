#!/usr/bin/env bash
# Equal-image-budget AFHQv2-64 training entry point for all paper variants.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {vp|ve|cosine_vp|edm|edm_canonical|fox_gaussian|fox_matern32|fox_matern32_vpdrift} [config overrides...]" >&2
  exit 1
fi

MODEL="$1"
shift
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-${ROOT_DIR}/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${ROOT_DIR}/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}" workdirs

case "${MODEL}" in
  vp)            CONFIG=configs/vp/afhqv2_ncsnpp_continuous.py; EFFECTIVE_BATCH=128 ;;
  ve)            CONFIG=configs/ve/afhqv2_ncsnpp_continuous.py; EFFECTIVE_BATCH=128 ;;
  cosine_vp)     CONFIG=configs/vp/afhqv2_ncsnpp_cosine_continuous.py; EFFECTIVE_BATCH=128 ;;
  edm)           CONFIG=configs/edm/afhqv2_ncsnpp.py; EFFECTIVE_BATCH=128 ;;
  edm_canonical) CONFIG=configs/edm/afhqv2_canonical.py; EFFECTIVE_BATCH=256 ;;
  fox_gaussian)  CONFIG=configs/fox/afhqv2_ncsnpp_continuous.py; EFFECTIVE_BATCH=128 ;;
  fox_matern32)  CONFIG=configs/fox/afhqv2_ncsnpp_continuous_matern_3_2.py; EFFECTIVE_BATCH=128 ;;
  fox_matern32_vpdrift) CONFIG=configs/fox/afhqv2_ncsnpp_continuous_matern_3_2_vp_drift.py; EFFECTIVE_BATCH=128 ;;
  *) echo "Unknown model: ${MODEL}" >&2; exit 1 ;;
esac

AFHQV2_DIR="${AFHQV2_DIR:-${ROOT_DIR}/datasets/afhqv2-64x64}"
TARGET_KIMG="${TARGET_KIMG:-6400}"
SEED="${SEED:-42}"
if (( TARGET_KIMG * 1000 % EFFECTIVE_BATCH != 0 )); then
  echo "TARGET_KIMG=${TARGET_KIMG} is not divisible into batch ${EFFECTIVE_BATCH}." >&2
  exit 1
fi
UPDATES=$((TARGET_KIMG * 1000 / EFFECTIVE_BATCH))
if (( UPDATES < 4 || UPDATES % 4 != 0 )); then
  echo "The image budget must yield at least four updates and divide into four snapshots." >&2
  exit 1
fi
SNAPSHOT_FREQ=$((UPDATES / 4))
WORKDIR="${WORKDIR:-workdirs/afhqv2_${MODEL}_${TARGET_KIMG}kimg_seed${SEED}}"

if [[ "${MODEL}" == "edm_canonical" ]]; then
  tools/install_official_edm.sh
fi

echo "Training ${MODEL}: approximately ${TARGET_KIMG} kimg, ${UPDATES} updates, seed ${SEED}"
python -u main.py \
  --mode=train \
  --config="${CONFIG}" \
  --workdir="${WORKDIR}" \
  --config.data.afhqv2_dir="${AFHQV2_DIR}" \
  --config.training.n_iters="${UPDATES}" \
  --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
  --config.training.snapshot_freq_for_preemption="${SNAPSHOT_FREQ}" \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=50 \
  --config.training.eval_freq=500 \
  --config.seed="${SEED}" \
  "$@"

echo "Finished ${WORKDIR}; checkpoint_4.pth is the matched-budget endpoint."
