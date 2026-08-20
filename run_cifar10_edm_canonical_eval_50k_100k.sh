#!/usr/bin/env bash
# Evaluate one checkpoint from the canonical 100k-step EDM CIFAR-10 run: 50k
# samples each.
#
# Usage: run_cifar10_edm_canonical_eval_50k_100k.sh <ckpt>
#   ckpt=1 -> step 25000  (effective batch 512 => ~12.8M images seen).
#             Matched-images point: the existing VP/FOX runs used 100k steps
#             at batch 128 (no accumulation), i.e. also ~12.8M images. This
#             is the checkpoint to compare against those numbers directly.
#   ckpt=4 -> step 100000 (~51.2M images seen). NOT equal-budget to the VP/FOX
#             100k x 128 rows -- report it as a separate, longer data point,
#             don't put it in the same table row as a matched comparison.
#   (ckpt=2/3 exist too, at 50k/75k steps, if you need intermediate points.)
#
# The former dropout-during-sampling bug (get_edm_sampler()/eval loss never
# called model.eval(), ~22% mean sample corruption) is fixed: the sampler and
# eval loss now save/restore train-vs-eval mode around inference. Samples and
# FID/KID/IS produced by an eval_dir written before that fix are stale and
# must not be reused -- the eval manifest's code hash will refuse to reuse
# such a cache automatically, but do not manually copy old numbers into a
# report either.
#
# begin_ckpt/end_ckpt are pinned to a single checkpoint on purpose:
# run_lib.evaluate() still blocks in a `while not exists: sleep(60)` loop
# waiting for checkpoints that never arrive, so the config defaults (1..15)
# would hang forever past whatever checkpoints actually exist. (What changed:
# once a checkpoint file exists, the read is no longer retried on transient
# errors -- a corrupt/short read now fails fast instead of silently sleeping
# and retrying twice.)
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[1-4]$ ]]; then
  echo "Usage: $0 <ckpt>  where ckpt is 1 (25k), 2 (50k), 3 (75k), or 4 (100k)" >&2
  exit 1
fi
CKPT="$1"

CONFIG=configs/edm/cifar10_canonical.py
WORKDIR=workdirs/cifar10_edm_canonical_100k

tools/install_official_edm.sh

# eval.batch_size=512 matches the prior CIFAR-10 eval protocol in
# EXPERIMENTS_REPORT.md (the canonical config default of 64 is far slower).
# 18-step Heun sampler => 2*18-1 = 35 NFE, vs ~1000 NFE for the PC/Euler-Maruyama
# runs used for VP/FOX. Keep that in mind for any quality-vs-NFE statement.
# num_samples=50000 matches the canonical default; the old VP/FOX table used
# 25k -- for a provisional apples-to-apples read against that specific old
# table, rerun with --config.eval.num_samples=25000, then still redo the
# uniform 50k pass before finalizing numbers.
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
