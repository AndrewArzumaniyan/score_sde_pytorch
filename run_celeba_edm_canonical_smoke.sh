#!/usr/bin/env bash
# Smoke test for the canonical NVLabs EDM recipe transferred to local CelebA.
#
# IMPORTANT: unlike CIFAR-10, this is NOT an official NVLabs recipe.
# configs/edm/celeba_canonical.py takes the CIFAR canonical hyperparameters
# and architecture knobs (edm_channel_mult, edm_attn_resolutions, etc.) and
# applies them to CelebA data -- NVLabs' own EDM paper never published a
# plain 64x64 CelebA config (theirs is FFHQ-64/AFHQ-64/CIFAR-10/ImageNet).
# Label any result from this path "canonical recipe applied to CelebA", not
# "reproduction of an official CelebA result".
#
# No --mode=eval stage here on purpose. evaluation.py's CelebA reference-stats
# computation always reads the FULL local train split
# (get_local_celeba_split_paths('train', ...) -- it does not know about
# celeba_train_take), so the first sampling/FID call would eat roughly as
# long as running the entire ~163k-image train split through Inception once:
# not a quick check. That sampling+Inception path is already proven end to
# end by the CIFAR-10 smoke test (same code, dataset-agnostic past the stats
# step). This script instead exercises what's actually new here: the local
# CelebA file-list data loader (datasets.py's from_tensor_slices path, fixed
# for a NoneTensor bug just before this script was written) and resume on
# that data source specifically.
#
# Stage 1: 30 steps on a small deterministic CelebA subset (proves the local
#          loader + training loop actually run on this data path).
# Stage 2: overfit a 10-image subset (loss must fall; see the NOTE in the
#          CIFAR-10 smoke script about the EDM loss floor -- same caveat).
# Stage 3: kill-and-resume check, same design as the CIFAR-10 smoke script.
set -euo pipefail

CONFIG=configs/edm/celeba_canonical.py
WORKDIR_SMOKE=workdirs/smoke_celeba_edm_canonical
WORKDIR_OVERFIT=workdirs/smoke_celeba_edm_canonical_overfit
WORKDIR_RESUME=workdirs/smoke_celeba_edm_canonical_resume

# The official EDM source is CC BY-NC-SA 4.0 and therefore gitignored; a plain
# `git pull` will NOT bring it. This installer is idempotent.
tools/install_official_edm.sh

echo "=== dependency-light regression tests ==="
python baseline_validation.py
python canonical_edm_validation.py

echo
echo "=== stage 1: 30 training steps on a small CelebA subset ==="
rm -rf "${WORKDIR_SMOKE}"
python main.py --mode=train \
  --config="${CONFIG}" \
  --workdir="${WORKDIR_SMOKE}" \
  --config.training.n_iters=30 \
  --config.training.batch_size=8 \
  --config.training.gradient_accumulation_steps=2 \
  --config.training.effective_batch_size=16 \
  --config.training.snapshot_freq=30 \
  --config.training.snapshot_freq_for_preemption=30 \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=5 \
  --config.training.eval_freq=1000000 \
  --config.data.celeba_train_take=200 \
  --config.data.celeba_validation_take=200

echo
echo "=== stage 2: overfit a 10-image subset (loss must fall) ==="
rm -rf "${WORKDIR_OVERFIT}"
python main.py --mode=train \
  --config="${CONFIG}" \
  --workdir="${WORKDIR_OVERFIT}" \
  --config.training.n_iters=400 \
  --config.training.batch_size=8 \
  --config.training.gradient_accumulation_steps=1 \
  --config.training.effective_batch_size=8 \
  --config.training.snapshot_freq=400 \
  --config.training.snapshot_freq_for_preemption=400 \
  --config.training.snapshot_sampling=False \
  --config.training.log_freq=20 \
  --config.training.eval_freq=1000000 \
  --config.optim.warmup_kimg=0.0 \
  --config.data.celeba_train_take=10 \
  --config.data.celeba_validation_take=10

echo
echo "=== stage 3: kill-and-resume check ==="
# snapshot_freq=20 so the *second* invocation's final step (20) lands exactly
# on a periodic boundary; snapshot_freq_for_preemption=5 exercises the
# periodic (not just final) meta-checkpoint save path too.
rm -rf "${WORKDIR_RESUME}"
RESUME_COMMON_FLAGS=(
  --config="${CONFIG}"
  --workdir="${WORKDIR_RESUME}"
  --config.training.batch_size=8
  --config.training.gradient_accumulation_steps=2
  --config.training.effective_batch_size=16
  --config.training.snapshot_freq=20
  --config.training.snapshot_freq_for_preemption=5
  --config.training.snapshot_sampling=False
  --config.training.log_freq=1
  --config.training.eval_freq=1000000
  --config.data.celeba_train_take=200
  --config.data.celeba_validation_take=200
)
# Flags must be byte-identical between the two invocations except n_iters:
# run_manifest.json only allows extending n_iters, not changing recipe/seed.

echo "--- resume stage, part A: train to step 10, then exit ---"
python main.py --mode=train "${RESUME_COMMON_FLAGS[@]}" \
  --config.training.n_iters=10 \
  2>&1 | tee "${WORKDIR_RESUME}.part_a.log"

echo "--- resume stage, part B: relaunch with n_iters=20 on the same workdir ---"
python main.py --mode=train "${RESUME_COMMON_FLAGS[@]}" \
  --config.training.n_iters=20 \
  2>&1 | tee "${WORKDIR_RESUME}.part_b.log"

echo "--- resume stage, verification ---"
resume_start_step=$(grep -o "Starting training loop at step [0-9]*" \
  "${WORKDIR_RESUME}.part_b.log" | tail -1 | grep -o "[0-9]*$")
if [ -z "${resume_start_step}" ] || [ "${resume_start_step}" -eq 0 ]; then
  echo "FAIL: part B did not resume from a positive step" \
    "(parsed '${resume_start_step:-<none>}'); it likely restarted from scratch." >&2
  exit 1
fi
echo "OK: part B resumed at step ${resume_start_step} (not 0)."

echo
echo "=== smoke test finished ==="
echo "Check: stage 2 training_loss trends down across steps 0 -> 400."
echo "Check: stage 3 printed 'OK: part B resumed at step ... (not 0)'."
echo "No sampling/FID stage here on purpose (see header comment) -- that path"
echo "is already covered by the CIFAR-10 smoke test."
