#!/usr/bin/env bash
# E4 arm 2/3 -- train "normalized-FOX": the CelebA kappa=1000 winner's log-SNR
# path lambda_FOX(t), rescaled to alpha^2 + q = 1 (VP-type overall scale r == 1).
#
# batch 128, seed 42, snapshot_freq 5000.  Length = E4_TRAIN_ITERS (default
# 100000 -> ckpt-20; set -e E4_TRAIN_ITERS=50000 for ckpt-10).  Resumable: rerun
# the same command to continue from checkpoints-meta (keep the same flags/env).
# Extra `--config.*` overrides passed to this script are forwarded to main.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/run_e4_common.sh"

echo "=== [E4] train normalized-FOX -> ${NORMFOX_WORKDIR} (${E4_TRAIN_ITERS} steps, cudnn_benchmark=${E4_CUDNN_BENCHMARK}, allow_tf32=${E4_ALLOW_TF32}) ==="
python -u main.py \
  --mode=train \
  --config="${NORMFOX_CONFIG}" \
  --workdir="${NORMFOX_WORKDIR}" \
  --config.training.batch_size=128 \
  --config.training.n_iters="${E4_TRAIN_ITERS}" \
  --config.training.snapshot_freq=5000 \
  --config.training.snapshot_freq_for_preemption=5000 \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=50 \
  --config.training.eval_freq=500 \
  --config.seed=42 \
  --config.cudnn_benchmark="${E4_CUDNN_BENCHMARK}" \
  --config.allow_tf32="${E4_ALLOW_TF32}" \
  "$@"
echo "=== [E4] normalized-FOX training done (ckpt-${E4_CKPT}) ==="
