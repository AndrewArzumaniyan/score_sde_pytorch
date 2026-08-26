#!/usr/bin/env bash
# AFHQv2-64 pilot: equal 6.4M-image training budget for all six methods.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

TARGET_KIMG=6400
SEED=42

# VP, VE, Cosine-VP, FOX and controlled EDM: 50k updates x 128 images.
# Canonical EDM: 25k updates x effective batch 256 images.
MODELS=(vp ve cosine_vp fox_matern32_vpdrift edm_canonical edm)

for model in "${MODELS[@]}"; do
  echo "=== Training ${model}; ${TARGET_KIMG} kimg ==="
  TARGET_KIMG="${TARGET_KIMG}" SEED="${SEED}" \
    ./run_afhqv2_train.sh "${model}"
done

echo "Training complete."
