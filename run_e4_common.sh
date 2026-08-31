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

E4_CKPT="${E4_CKPT:-20}"                 # 100k / snapshot_freq 5000
KAPPA1000_ELL="346.4101615138"          # |u|=5, kappa=1000, matern_3_2

VP_CONFIG="configs/vp/celeba_ncsnpp_continuous.py"
REALFOX_CONFIG="configs/fox/celeba_ncsnpp_continuous_matern_3_2.py"
NORMFOX_CONFIG="configs/fox/celeba_ncsnpp_continuous_normalized.py"
EDM_CONFIG="configs/edm/celeba_canonical.py"
