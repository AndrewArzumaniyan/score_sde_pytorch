#!/usr/bin/env bash
# Fast common AFHQv2-64 pilot evaluation: 5,000 generated samples per method.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

TARGET_KIMG=6400
SEED=42
NUM_SAMPLES=5000

MODELS=(vp ve cosine_vp fox_matern32_vpdrift edm_canonical edm)

for model in "${MODELS[@]}"; do
  echo "=== Evaluating ${model}; checkpoint 4; ${NUM_SAMPLES} samples ==="
  TARGET_KIMG="${TARGET_KIMG}" SEED="${SEED}" NUM_SAMPLES="${NUM_SAMPLES}" \
    ./run_afhqv2_eval_50k.sh "${model}" 4
done

echo "Evaluation complete."
