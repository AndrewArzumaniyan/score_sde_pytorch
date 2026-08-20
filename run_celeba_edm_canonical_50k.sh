#!/usr/bin/env bash
# Canonical NVLabs EDM recipe applied to local CelebA, trained to 50k
# optimizer steps.
#
# IMPORTANT: this is NOT an official NVLabs recipe. configs/edm/celeba_canonical.py
# takes the CIFAR canonical hyperparameters/architecture (edm_channel_mult,
# edm_attn_resolutions, etc.) and applies them to CelebA data -- NVLabs never
# published a plain 64x64 CelebA config. Label results "canonical recipe
# applied to CelebA", not "reproduction of an official CelebA result".
#
# Budget note: canonical effective batch is 512 (128 x 4 accumulation) --
# unchanged from the official recipe on purpose, same reasoning as the
# CIFAR-10 canonical run (LR/EMA-halflife/warmup are tuned assuming it).
# snapshot_freq=12500 => checkpoints at 12500/25000/37500/50000 => ckpt-1..4.
# ckpt-1 (12500 steps x 512 = 6.4M images) matches the existing
# run_celeba_compare_50k.sh baseline's budget (50000 steps x 128, no
# accumulation = 6.4M images) -- that's the checkpoint to compare against
# EXPERIMENTS_REPORT.md's VE/VP/FOX numbers (ckpt-10 there, same 6.4M).
# ckpt-4 (25.6M images) is a separate, longer, NOT equal-budget point.
set -euo pipefail

CONFIG=configs/edm/celeba_canonical.py
WORKDIR=workdirs/celeba_edm_canonical_50k
N_ITERS=50000

# The official EDM source is CC BY-NC-SA 4.0 and is gitignored, so `git pull`
# does not deliver it. Idempotent; verifies the pinned commit.
tools/install_official_edm.sh

# snapshot_sampling is disabled: costs sampling time every snapshot and the
# preview grids aren't used for any reported metric.
# No celeba_train_take/celeba_validation_take override: full local split.
python main.py --mode=train \
  --config="${CONFIG}" \
  --workdir="${WORKDIR}" \
  --config.training.n_iters=${N_ITERS} \
  --config.training.snapshot_freq=12500 \
  --config.training.snapshot_freq_for_preemption=2500 \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=50 \
  --config.training.eval_freq=500 \
  --config.seed=42

echo "training finished: ${WORKDIR} (final checkpoint should be ckpt-4)"
