#!/usr/bin/env bash
# Evaluate the 100k-step canonical EDM CIFAR-10 run: 50k samples on ckpt-4.
#
# The former dropout-during-sampling bug (get_edm_sampler()/eval loss never
# called model.eval(), ~22% mean sample corruption) is fixed: the sampler and
# eval loss now save/restore train-vs-eval mode around inference. Samples and
# FID/KID/IS produced by an eval_dir written before that fix are stale and
# must not be reused -- the eval manifest's code hash will refuse to reuse
# such a cache automatically, but do not manually copy old numbers into a
# report either.
#
# begin_ckpt/end_ckpt are pinned to 4 on purpose: run_lib.evaluate() still
# blocks in a `while not exists: sleep(60)` loop waiting for checkpoints that
# never arrive, so the config defaults (1..15) would hang forever after a
# 100k-step run. (What changed: once a checkpoint file exists, the read is no
# longer retried on transient errors -- a corrupt/short read now fails fast
# instead of silently sleeping and retrying twice.)
set -euo pipefail

CONFIG=configs/edm/cifar10_canonical.py
WORKDIR=workdirs/cifar10_edm_canonical_100k
CKPT=4

tools/install_official_edm.sh

# eval.batch_size=512 matches the prior CIFAR-10 eval protocol in
# EXPERIMENTS_REPORT.md (the canonical config default of 64 is far slower).
# 18-step Heun sampler => 2*18-1 = 35 NFE, vs ~1000 NFE for the PC/Euler-Maruyama
# runs used for VP/FOX. Keep that in mind for any quality-vs-NFE statement.
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
