#!/usr/bin/env bash
# Canonical NVLabs EDM baseline on CIFAR-10, trained to 100k optimizer steps.
#
# Budget note: the canonical recipe uses an effective batch of 512
# (128 x 4 gradient accumulation), so 100k steps = 51.2M images.  The official
# EDM budget is 200M images (390625 steps).  This run is therefore ~1/4 of the
# paper budget and will NOT reproduce the published CIFAR-10 FID of 1.97.
# Treat it as a controlled-budget baseline, not an EDM reproduction.
#
# Comparability note: prior VP/FOX CIFAR-10 runs in EXPERIMENTS_REPORT.md used
# batch 128 with no accumulation, i.e. 100k steps = 12.8M images.  Matching
# *steps* here does not match *images seen*.  Decide explicitly which axis you
# are holding fixed before putting these numbers in the same table.
set -euo pipefail

CONFIG=configs/edm/cifar10_canonical.py
WORKDIR=workdirs/cifar10_edm_canonical_100k
N_ITERS=100000

# The official EDM source is CC BY-NC-SA 4.0 and is gitignored, so `git pull`
# does not deliver it.  Idempotent; verifies the pinned commit.
tools/install_official_edm.sh

# snapshot_freq=25000 => checkpoints at 25k/50k/75k/100k => ckpt-1..ckpt-4.
# snapshot_sampling is disabled: it costs sampling time every snapshot and the
# preview grids are not used for any reported metric.
python main.py --mode=train \
  --config="${CONFIG}" \
  --workdir="${WORKDIR}" \
  --config.training.n_iters=${N_ITERS} \
  --config.training.snapshot_freq=25000 \
  --config.training.snapshot_freq_for_preemption=5000 \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=50 \
  --config.training.eval_freq=500 \
  --config.seed=42

echo "training finished: ${WORKDIR} (final checkpoint should be ckpt-4)"
