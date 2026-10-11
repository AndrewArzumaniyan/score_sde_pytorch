#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-score-sde-rtx}"
GPU_DEVICE="${2:-GPU-25e966e6-f936-4c1d-370b-fbe8e47c0095}"

docker run --gpus "device=${GPU_DEVICE}" --rm -it \
  --ipc=host \
  --shm-size=16g \
  -e TFDS_DATA_DIR=/workspace/score_sde_pytorch/.tfds \
  -e TORCH_EXTENSIONS_DIR=/workspace/score_sde_pytorch/.torch_extensions_rtx \
  -e XDG_CACHE_HOME=/workspace/score_sde_pytorch/.cache_rtx \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e MAX_JOBS=8 \
  -v "$(pwd)":/workspace/score_sde_pytorch \
  -w /workspace/score_sde_pytorch \
  "${IMAGE_NAME}" \
  bash
