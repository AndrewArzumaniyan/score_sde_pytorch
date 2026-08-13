#!/usr/bin/env bash
set -euo pipefail

EDM_COMMIT="008a4e5316c8e3bfe61a62f874bddba254295afb"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${EDM_OFFICIAL_ROOT:-${REPO_ROOT}/third_party/edm}"

if [[ -e "${TARGET}" ]]; then
  if [[ ! -d "${TARGET}/.git" ]]; then
    echo "Target exists but is not a git checkout: ${TARGET}" >&2
    exit 1
  fi
  CURRENT="$(git -C "${TARGET}" rev-parse HEAD)"
  if [[ "${CURRENT}" != "${EDM_COMMIT}" ]]; then
    echo "Official EDM checkout is at ${CURRENT}; expected ${EDM_COMMIT}." >&2
    exit 1
  fi
  echo "Official EDM is already installed at the pinned commit: ${TARGET}"
  exit 0
fi

mkdir -p "$(dirname "${TARGET}")"
git clone https://github.com/NVlabs/edm.git "${TARGET}"
git -C "${TARGET}" checkout --detach "${EDM_COMMIT}"
echo "Installed official NVLabs EDM ${EDM_COMMIT} at ${TARGET}"
echo "Its source remains governed by CC BY-NC-SA 4.0 (see ${TARGET}/LICENSE.txt)."
