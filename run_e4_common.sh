#!/usr/bin/env bash
# Shared environment for the E4 (normalized-FOX / log-SNR vs scale-path) cycle.
# Sourced by the other run_e4_*.sh scripts; not run directly.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# These env vars are normally already set by the docker run invocation
# (see RUNBOOK / RUNNING_EXPERIMENTS.md); the defaults here are only a fallback.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"        # H200 = Hopper = 9.0
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
# GPU pinning is done at `docker run` with --gpus '"device=0"' (the free H200;
# NOT GPU 2 -- RTX PRO 6000 Blackwell sm_120 needs a rebuilt image).  The
# container then sees exactly one GPU as cuda:0.

mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" workdirs

# ---- existing checkpoints for the two reference arms (VERIFY on the server) ---
# Both were trained to 100k steps at snapshot_freq=5000  =>  checkpoint-20.
VP_WORKDIR="${VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
REALFOX_WORKDIR="${REALFOX_WORKDIR:-workdirs/celeba_fox_matern32_kappa1000_50k}"
# new training produced by run_e4_normalized_fox_train.sh
NORMFOX_WORKDIR="${NORMFOX_WORKDIR:-workdirs/celeba_fox_normalized_kappa1000_100k}"
# canonical EDM CelebA (already trained, 50k steps; snapshot_freq 12500 => ckpt-4).
# VERIFY name + highest ckpt on the server.
EDM_WORKDIR="${EDM_WORKDIR:-workdirs/celeba_edm_canonical_50k}"
EDM_CKPT="${EDM_CKPT:-4}"

# Training length for the new normalized-FOX arm.  snapshot_freq is fixed at
# 5000, so E4_CKPT (the checkpoint every arm is evaluated at) derives from it:
#   E4_TRAIN_ITERS=100000 -> ckpt-20   (100k-matched with the existing VP / real-FOX numbers)
#   E4_TRAIN_ITERS=50000  -> ckpt-10   (~half the wall-clock; VP / real-FOX have ckpt-10 too)
E4_TRAIN_ITERS="${E4_TRAIN_ITERS:-100000}"
E4_CKPT="${E4_CKPT:-$((E4_TRAIN_ITERS / 5000))}"

# cuDNN conv autotuning: ~1.5x on H200 for this model; only picks among
# mathematically-equivalent conv algorithms.  Override `-e E4_CUDNN_BENCHMARK=False`.
E4_CUDNN_BENCHMARK="${E4_CUDNN_BENCHMARK:-True}"

# TF32 conv/matmul.  Default True because the reference VP / real-FOX checkpoints
# were trained BEFORE commit afb5297 (2026-08-14) added reproducibility.py, i.e.
# with torch's default cuDNN TF32 ON.  So allow_tf32=True *matches* them (and is
# ~3x faster on Hopper).  Set `-e E4_ALLOW_TF32=False` only if you have verified
# the reference checkpoints are post-afb5297 (true FP32).
E4_ALLOW_TF32="${E4_ALLOW_TF32:-True}"

KAPPA1000_ELL="346.4101615138"          # |u|=5, kappa=1000, matern_3_2

VP_CONFIG="configs/vp/celeba_ncsnpp_continuous.py"
REALFOX_CONFIG="configs/fox/celeba_ncsnpp_continuous_matern_3_2.py"
NORMFOX_CONFIG="configs/fox/celeba_ncsnpp_continuous_normalized.py"
EDM_CONFIG="configs/edm/celeba_canonical.py"
