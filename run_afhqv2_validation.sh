#!/usr/bin/env bash
# Run all AFHQv2 dependency, config, loader, and regression checks in-container.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

tools/install_official_edm.sh

python -m py_compile \
  datasets.py evaluation.py afhqv2_validation.py \
  configs/default_afhqv2_configs.py \
  configs/vp/afhqv2_ncsnpp_continuous.py \
  configs/vp/afhqv2_ncsnpp_cosine_continuous.py \
  configs/edm/afhqv2_ncsnpp.py \
  configs/edm/afhqv2_canonical.py \
  configs/fox/afhqv2_ncsnpp_continuous.py \
  configs/fox/afhqv2_ncsnpp_continuous_matern_3_2.py \
  configs/fox/afhqv2_ncsnpp_continuous_matern_3_2_vp_drift.py

python afhqv2_validation.py
python baseline_validation.py
python canonical_edm_validation.py
python -m unittest tests.test_fox_sde
python -m unittest tests.test_afhqv2_dataset

echo "All AFHQv2 static, loader, stats, FOX, Cosine-VP, and EDM checks passed."
