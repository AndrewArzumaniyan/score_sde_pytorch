#!/usr/bin/env bash
# Evaluate one checkpoint from the canonical 50k-step EDM-on-CelebA run.
#
# Usage: run_celeba_edm_canonical_eval_50k.sh <ckpt>
#   ckpt=1 -> step 12500 (effective batch 512 => 6.4M images seen).
#             Matched-images point: run_celeba_compare_50k.sh's existing
#             VE/VP/FOX baselines used 50000 steps at batch 128 (no
#             accumulation), i.e. also 6.4M images (see EXPERIMENTS_REPORT.md,
#             "Базовое сравнение VE, VP, Gaussian FOX, Matern 3/2 FOX на 50k",
#             ckpt-10 there). This is the checkpoint to compare against those
#             numbers (VP FID=9.326062, Gaussian FOX kappa=1 FID=8.189067,
#             Matern 3/2 FOX kappa=1 FID=8.195414).
#   ckpt=4 -> step 50000 (25.6M images). NOT equal-budget -- report as a
#             separate, longer data point.
#   (ckpt=2/3 exist too, at 25000/37500 steps, if you need intermediate
#   points.)
#
# The old baseline's eval used 5000 samples (see run_celeba_compare_eval_5k.sh);
# canonical default here is 50000. For a provisional apples-to-apples read
# against that specific old table, rerun with --config.eval.num_samples=5000,
# then still redo the uniform 50k pass before finalizing numbers -- same
# caveat as the CIFAR-10 eval script.
#
# begin_ckpt/end_ckpt are pinned to a single checkpoint on purpose:
# run_lib.evaluate() blocks in a `while not exists: sleep(60)` loop waiting
# for checkpoints that never arrive, so the config defaults would hang past
# whatever checkpoints actually exist.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[1-4]$ ]]; then
  echo "Usage: $0 <ckpt>  where ckpt is 1 (12500), 2 (25000), 3 (37500), or 4 (50000)" >&2
  exit 1
fi
CKPT="$1"

CONFIG=configs/edm/celeba_canonical.py
WORKDIR=workdirs/celeba_edm_canonical_50k

tools/install_official_edm.sh

# eval.batch_size=512 for throughput (unlike training's effective batch, eval
# batch size doesn't affect the reported numbers, only how fast they arrive).
# 18-step Heun sampler => 2*18-1 = 35 NFE.
python main.py --mode=eval \
  --config="${CONFIG}" \
  --workdir="${WORKDIR}" \
  --eval_folder=eval_ckpt${CKPT}_50k \
  --config.eval.begin_ckpt=${CKPT} \
  --config.eval.end_ckpt=${CKPT} \
  --config.eval.batch_size=512 \
  --config.eval.num_samples=50000 \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.enable_bpd=False \
  --config.seed=42

echo "eval finished; final line has the form:"
echo "  ckpt-${CKPT} --- inception_score: ..., FID: ..., KID: ..."
