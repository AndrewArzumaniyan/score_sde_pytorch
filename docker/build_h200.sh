#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-score-sde-h200}"

docker build -t "${IMAGE_NAME}" -f Dockerfile .
