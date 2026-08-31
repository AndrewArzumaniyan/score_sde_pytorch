#!/usr/bin/env bash
# EDM CelebA -- the honest-baseline gate for the "FOX is the strongest schedule
# on CelebA" headline.  Already-trained canonical 50k-step run; sampling-only.
#
# 18-step Heun => 35 NFE, so this is fast: ~25-45 min for 50k samples on H200.
# NOT an equal-budget comparison (EDM 50k steps x effective-batch 512 = 25.6M
# images vs the FOX/VP 100k x 128 = 12.8M); report as a separate data point.
#
# begin_ckpt == end_ckpt is mandatory: run_lib.evaluate() otherwise blocks
# forever waiting for checkpoints that will never be written.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/run_e4_common.sh"

# third_party/edm is gitignored (CC BY-NC-SA).  This is idempotent -- it prints
# "already installed" and exits 0 if the pinned checkout is present.  If it is
# NOT present yet, run it on the HOST once before starting the container
# (RUNNING_EXPERIMENTS.md: a root container would leave root-owned files).
tools/install_official_edm.sh

NUM_SAMPLES="${1:-50000}"

echo "=== [E4] EDM CelebA eval -> ${EDM_WORKDIR} ckpt-${EDM_CKPT}, ${NUM_SAMPLES} samples ==="
python -u main.py \
  --mode=eval \
  --config="${EDM_CONFIG}" \
  --workdir="${EDM_WORKDIR}" \
  --eval_folder="eval_e4_ckpt${EDM_CKPT}_${NUM_SAMPLES}" \
  --config.eval.begin_ckpt="${EDM_CKPT}" \
  --config.eval.end_ckpt="${EDM_CKPT}" \
  --config.eval.batch_size=512 \
  --config.eval.num_samples="${NUM_SAMPLES}" \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.enable_bpd=False \
  --config.seed=42
echo "=== [E4] EDM CelebA eval done ==="
