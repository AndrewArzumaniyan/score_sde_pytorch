#!/usr/bin/env bash
# E4 -- evaluate all three arms under ONE identical protocol and decompose the
# CelebA FOX gain into "log-SNR shape" vs "overall-scale path r(t)".
#
#   arm            log-SNR      scale r(t)     checkpoint
#   ------------   ----------   -----------    -------------------------------
#   VP             lambda_VP    r == 1         ${VP_WORKDIR} ckpt-${E4_CKPT}
#   normalized-FOX lambda_FOX   r == 1         ${NORMFOX_WORKDIR} ckpt-${E4_CKPT}
#   real-FOX k1000 lambda_FOX   r_F(t) (sags)  ${REALFOX_WORKDIR} ckpt-${E4_CKPT}
#
#   normalized-FOX > VP           -> the log-SNR SHAPE carries the gain
#   real-FOX > normalized-FOX     -> the scale path r(t) matters
#   all three ~equal              -> schedule does not explain the gain (pivot)
#
# Usage:  run_e4_eval.sh <num_samples> <folder_suffix>
#   fast signal :  run_e4_eval.sh 5000  5k
#   publication :  run_e4_eval.sh 50000 50k
#
# Protocol (forced identically on every arm): PC sampler, euler_maruyama
# predictor, no corrector, NFE = num_scales = 1000, uniform_time grid, seed 0.
# H200 wall-clock per arm: ~15-25 min @ 5k,  ~2.5-4 h @ 50k.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/run_e4_common.sh"

NUM_SAMPLES="${1:?usage: run_e4_eval.sh <num_samples> <folder_suffix>}"
SUFFIX="${2:?usage: run_e4_eval.sh <num_samples> <folder_suffix>}"
FOLDER="eval_e4_${SUFFIX}"

COMMON_ARGS=(
  --mode=eval
  --eval_folder="${FOLDER}"
  --config.eval.begin_ckpt="${E4_CKPT}"
  --config.eval.end_ckpt="${E4_CKPT}"
  --config.eval.batch_size=512
  --config.eval.num_samples="${NUM_SAMPLES}"
  --config.eval.enable_loss=False
  --config.eval.enable_sampling=True
  --config.eval.enable_bpd=False
  --config.eval.sampling_seed=0
  --config.sampling.method=pc
  --config.sampling.predictor=euler_maruyama
  --config.sampling.corrector=none
  --config.sampling.time_grid=uniform_time
)

echo "=== [E4 eval ${SUFFIX}] arm 1/3: VP ==="
python -u main.py --config="${VP_CONFIG}" --workdir="${VP_WORKDIR}" "${COMMON_ARGS[@]}"

echo "=== [E4 eval ${SUFFIX}] arm 2/3: normalized-FOX ==="
python -u main.py --config="${NORMFOX_CONFIG}" --workdir="${NORMFOX_WORKDIR}" "${COMMON_ARGS[@]}"

echo "=== [E4 eval ${SUFFIX}] arm 3/3: real-FOX kappa=1000 ==="
python -u main.py --config="${REALFOX_CONFIG}" --workdir="${REALFOX_WORKDIR}" "${COMMON_ARGS[@]}" \
  --config.model.fox_u=-5.0 \
  --config.model.fox_matern_length_scale="${KAPPA1000_ELL}" \
  --config.model.fox_target_terminal_variance=1.0

echo "=== [E4 eval ${SUFFIX}] done -- FID/KID/IS in <workdir>/${FOLDER}/ ==="
