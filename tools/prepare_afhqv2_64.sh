#!/usr/bin/env bash
# Convert the downloaded AFHQv2 tree with the pinned official EDM converter.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 SOURCE_DIR [DEST_DIR]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$(cd "$1" && pwd)"
DEST_DIR="${2:-${ROOT_DIR}/datasets/afhqv2-64x64}"
EXPECTED_IMAGES="${AFHQV2_EXPECTED_IMAGES:-15803}"

"${ROOT_DIR}/tools/install_official_edm.sh"

validate_dataset() {
  local dataset_dir="$1"
  if [[ ! -f "${dataset_dir}/dataset.json" ]]; then
    echo "Missing official dataset.json in ${dataset_dir}" >&2
    return 1
  fi
  local count
  count="$(find "${dataset_dir}" -type f -iname '*.png' | wc -l | tr -d ' ')"
  if [[ "${count}" != "${EXPECTED_IMAGES}" ]]; then
    echo "Expected ${EXPECTED_IMAGES} AFHQv2 PNGs, found ${count} in ${dataset_dir}." >&2
    echo "Use the updated afhq-v2-dataset, including train and test images." >&2
    return 1
  fi
  echo "Validated AFHQv2-64: ${count} images in ${dataset_dir}"
}

if [[ -e "${DEST_DIR}" ]]; then
  validate_dataset "${DEST_DIR}"
  echo "Dataset is already prepared; leaving it unchanged."
  exit 0
fi

mkdir -p "$(dirname "${DEST_DIR}")"
STAGING_DIR="$(mktemp -d "$(dirname "${DEST_DIR}")/.afhqv2-64x64.XXXXXX")"
cleanup() {
  if [[ -d "${STAGING_DIR}" ]]; then
    rm -rf "${STAGING_DIR}"
  fi
}
trap cleanup EXIT

python "${ROOT_DIR}/third_party/edm/dataset_tool.py" \
  --source="${SOURCE_DIR}" \
  --dest="${STAGING_DIR}" \
  --resolution=64x64
validate_dataset "${STAGING_DIR}"
mv "${STAGING_DIR}" "${DEST_DIR}"
trap - EXIT
echo "AFHQV2_DIR=${DEST_DIR}"
