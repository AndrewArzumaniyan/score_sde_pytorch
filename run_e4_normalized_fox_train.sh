#!/usr/bin/env bash
# E4 arm 2/3 -- train "normalized-FOX": the CelebA kappa=1000 winner's log-SNR
# path lambda_FOX(t), rescaled to alpha^2 + q = 1 (VP-type overall scale r == 1).
#
# 100k steps, batch 128, seed 42, snapshot_freq 5000  ->  checkpoint-20.
# H200 wall-clock estimate: ~5 h (range 4-8 h depending on the GPU).
# Resumable: rerun the same command to continue from checkpoints-meta.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/run_e4_common.sh"

echo "=== [E4] train normalized-FOX -> ${NORMFOX_WORKDIR} ==="
python -u main.py \
  --mode=train \
  --config="${NORMFOX_CONFIG}" \
  --workdir="${NORMFOX_WORKDIR}" \
  --config.training.batch_size=128 \
  --config.training.n_iters=100000 \
  --config.training.snapshot_freq=5000 \
  --config.training.snapshot_freq_for_preemption=5000 \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=50 \
  --config.training.eval_freq=500 \
  --config.seed=42
echo "=== [E4] normalized-FOX training done (ckpt-${E4_CKPT}) ==="
