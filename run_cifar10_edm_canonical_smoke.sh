#!/usr/bin/env bash
# Smoke test for the canonical NVLabs EDM baseline.
#
# Purpose: prove the pipeline runs end to end (data -> loss -> backward ->
# checkpoint -> resume -> EDM Heun sampler -> IS/FID/KID) on a tiny budget,
# before committing a GPU to the long run.  Nothing here is a quality result.
#
# Stage 1: 30 steps on a small balanced subset, then eval on few samples.
# Stage 2: overfit a 10-image subset (1 per class); the loss must fall clearly.
# Stage 3: kill-and-resume check -- train to step 10, relaunch the *same*
#          command with a larger n_iters, and confirm the second process
#          resumes from checkpoints-meta/checkpoint.pth instead of
#          restarting at step 0. Then eval the resulting boundary checkpoint.
#          This is the coarse decision-gate "restart/resume check"; the
#          deeper continuous-vs-resumed byte/RNG comparison (batch hashes,
#          weights, optimizer, EMA, next RNG draws) is a separate, later
#          validation and is NOT what this stage proves.
#
# NOTE: the EDM loss draws a fresh log-normal sigma per sample per step, so even
# a perfectly overfit model keeps a nonzero loss floor.  Look for a clear
# downward trend, not loss -> 0.
set -euo pipefail

CONFIG=configs/edm/cifar10_canonical.py
WORKDIR_SMOKE=workdirs/smoke_cifar10_edm_canonical
WORKDIR_OVERFIT=workdirs/smoke_cifar10_edm_canonical_overfit
WORKDIR_RESUME=workdirs/smoke_cifar10_edm_canonical_resume

# The official EDM source is CC BY-NC-SA 4.0 and therefore gitignored; a plain
# `git pull` will NOT bring it. This installer is idempotent.
tools/install_official_edm.sh

echo "=== dependency-light regression tests ==="
python baseline_validation.py
python canonical_edm_validation.py

echo
echo "=== stage 1: 30 training steps on a small balanced subset ==="
# Safe to wipe unconditionally: WORKDIR_SMOKE is a dedicated, disposable
# smoke path, and this stage does not test resume (stage 3 does, and does
# NOT wipe between its two invocations).
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
  --config.data.cifar10_train_per_class=20 \
  --config.data.cifar10_test_per_class=20

echo
echo "=== stage 1b: eval the resulting ckpt-1 on a few samples ==="
# begin_ckpt/end_ckpt MUST bracket only checkpoints that exist: run_lib.evaluate
# blocks in a `while not exists: sleep(60)` loop waiting for missing ones.
python main.py --mode=eval \
  --config="${CONFIG}" \
  --workdir="${WORKDIR_SMOKE}" \
  --eval_folder=eval_smoke \
  --config.eval.begin_ckpt=1 \
  --config.eval.end_ckpt=1 \
  --config.eval.batch_size=32 \
  --config.eval.num_samples=64 \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.enable_bpd=False \
  --config.data.cifar10_train_per_class=20 \
  --config.data.cifar10_test_per_class=20

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
  --config.data.cifar10_train_per_class=1 \
  --config.data.cifar10_test_per_class=1

echo
echo "=== stage 3: kill-and-resume check ==="
# snapshot_freq=20 so the *second* invocation's final step (20) lands exactly
# on a periodic boundary and is published as checkpoint_1.pth (see
# utils.checkpoint_filename); an off-boundary final would only produce a
# checkpoint_final_step_N.pth, which --eval.begin_ckpt/end_ckpt cannot address.
# snapshot_freq_for_preemption=5 also exercises the periodic (not just final)
# meta-checkpoint save path.
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
  --config.data.cifar10_train_per_class=20
  --config.data.cifar10_test_per_class=20
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
echo "=== stage 3b: eval the resumed run's boundary checkpoint-1 ==="
python main.py --mode=eval \
  --config="${CONFIG}" \
  --workdir="${WORKDIR_RESUME}" \
  --eval_folder=eval_smoke_resume \
  --config.eval.begin_ckpt=1 \
  --config.eval.end_ckpt=1 \
  --config.eval.batch_size=32 \
  --config.eval.num_samples=64 \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.enable_bpd=False \
  --config.data.cifar10_train_per_class=20 \
  --config.data.cifar10_test_per_class=20

echo
echo "=== smoke test finished ==="
echo "Check: stage 2 training_loss trends down across steps 0 -> 400."
echo "Check: stage 3 printed 'OK: part B resumed at step ... (not 0)'."
echo "Stage 1b/3b IS/FID/KID on 64 samples are meaningless numbers; they only"
echo "prove the sampling + Inception path executes."
