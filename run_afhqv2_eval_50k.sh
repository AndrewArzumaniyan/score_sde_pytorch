#!/usr/bin/env bash
# Common 50k-sample FID/IS/KID evaluation for an AFHQv2 training run.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {vp|cosine_vp|edm|edm_canonical|fox_gaussian|fox_matern32|fox_matern32_vpdrift} [checkpoint] [config overrides...]" >&2
  exit 1
fi

MODEL="$1"
shift
CKPT=4
if [[ $# -gt 0 && "$1" != --* ]]; then
  CKPT="$1"
  shift
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-${ROOT_DIR}/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${ROOT_DIR}/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}" workdirs

case "${MODEL}" in
  vp)            CONFIG=configs/vp/afhqv2_ncsnpp_continuous.py ;;
  cosine_vp)     CONFIG=configs/vp/afhqv2_ncsnpp_cosine_continuous.py ;;
  edm)           CONFIG=configs/edm/afhqv2_ncsnpp.py ;;
  edm_canonical) CONFIG=configs/edm/afhqv2_canonical.py ;;
  fox_gaussian)  CONFIG=configs/fox/afhqv2_ncsnpp_continuous.py ;;
  fox_matern32)  CONFIG=configs/fox/afhqv2_ncsnpp_continuous_matern_3_2.py ;;
  fox_matern32_vpdrift) CONFIG=configs/fox/afhqv2_ncsnpp_continuous_matern_3_2_vp_drift.py ;;
  *) echo "Unknown model: ${MODEL}" >&2; exit 1 ;;
esac
if [[ ! "${CKPT}" =~ ^[1-4]$ ]]; then
  echo "Checkpoint must be 1, 2, 3, or 4; got ${CKPT}." >&2
  exit 1
fi

AFHQV2_DIR="${AFHQV2_DIR:-${ROOT_DIR}/datasets/afhqv2-64x64}"
TARGET_KIMG="${TARGET_KIMG:-6400}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
WORKDIR="${WORKDIR:-workdirs/afhqv2_${MODEL}_${TARGET_KIMG}kimg_seed${SEED}}"

if [[ "${MODEL}" == "edm_canonical" ]]; then
  tools/install_official_edm.sh
fi

# This intentionally uses each method's declared sampler: EDM uses the
# paper's deterministic 40-step Heun grid (79 NFE), while VP/FOX retain their
# 1000-step PC sampler. Report NFE beside FID; use sampling_num_scales in an
# additional ablation when an NFE-matched comparison is required.
python -u main.py \
  --mode=eval \
  --config="${CONFIG}" \
  --workdir="${WORKDIR}" \
  --eval_folder="eval_ckpt${CKPT}_${NUM_SAMPLES}" \
  --config.data.afhqv2_dir="${AFHQV2_DIR}" \
  --config.eval.begin_ckpt="${CKPT}" \
  --config.eval.end_ckpt="${CKPT}" \
  --config.eval.batch_size=64 \
  --config.eval.num_samples="${NUM_SAMPLES}" \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.enable_bpd=False \
  --config.seed="${SEED}" \
  "$@"

echo "Finished common AFHQv2 metrics for ${WORKDIR}, checkpoint ${CKPT}."
