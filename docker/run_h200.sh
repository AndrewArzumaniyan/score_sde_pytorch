#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-score-sde-h200}"

docker run --gpus all --rm -it \
  --ipc=host \
  --shm-size=16g \
  -e TFDS_DATA_DIR=/workspace/score_sde_pytorch/.tfds \
  -e TORCH_EXTENSIONS_DIR=/workspace/score_sde_pytorch/.torch_extensions \
  -e XDG_CACHE_HOME=/workspace/score_sde_pytorch/.cache \
  -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e MAX_JOBS=8 \
  -v "$(pwd)":/workspace/score_sde_pytorch \
  -w /workspace/score_sde_pytorch \
  "${IMAGE_NAME}" \
  bash
