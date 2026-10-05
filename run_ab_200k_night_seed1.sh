#!/usr/bin/env bash
# Run inside Docker with bash; keeps tmux's interactive shell intact.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
# Do not inherit alternate workdirs or numerical flags from earlier queues.
export AB_ENDPOINT_DENOISE_A_WORKDIR=workdirs/celeba_factorial_A_vp_pvp_50k
export AB_ENDPOINT_DENOISE_B_WORKDIR=workdirs/celeba_factorial_B_vp_pfox_50k
export AB_SPLICE_A_WORKDIR="$AB_ENDPOINT_DENOISE_A_WORKDIR"
export AB_SPLICE_B_WORKDIR="$AB_ENDPOINT_DENOISE_B_WORKDIR"
export AB_ENDPOINT_DENOISE_ALLOW_TF32=True AB_ENDPOINT_DENOISE_CUDNN_BENCHMARK=True
export AB_SPLICE_ALLOW_TF32=True AB_SPLICE_CUDNN_BENCHMARK=True
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" logs results
exec > >(tee -a logs/ab_200k_night_seed1.log) 2>&1
START=$SECONDS
LIMIT=50400
# Reserve the last four hours for CPU bootstrap.
RESERVE=14400
REFERENCE=assets/stats/celeba_64_0c2d89311192_stats.npz
for path in "$REFERENCE" "$AB_SPLICE_A_WORKDIR/checkpoints/checkpoint_40.pth" "$AB_SPLICE_B_WORKDIR/checkpoints/checkpoint_40.pth"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
run() {
  local budget="$1"; shift
  local remaining=$((budget - (SECONDS - START)))
  if (( remaining <= 0 )); then
    echo "Stage time budget exhausted; stopping queue." >&2
    return 124
  fi
  timeout --signal=INT --kill-after=90s "${remaining}s" "$@"
}
endpoint() {
  local tag="$1" samples="$2" nfe="$3" runs="$4"
  run "$((LIMIT - RESERVE))" env AB_ENDPOINT_DENOISE_RUN_TAG="$tag" \
    AB_ENDPOINT_DENOISE_CKPT=40 AB_ENDPOINT_DENOISE_NUM_SAMPLES="$samples" \
    AB_ENDPOINT_DENOISE_BATCH_SIZE=512 AB_ENDPOINT_DENOISE_SEED=1 \
    AB_ENDPOINT_DENOISE_NFE="$nfe" AB_ENDPOINT_DENOISE_RUNS="$runs" \
    bash ./run_ab_endpoint_denoise_eval.sh
}
splice() {
  local tag="$1" samples="$2" runs="$3"
  run "$((LIMIT - RESERVE))" env AB_SPLICE_RUN_TAG="$tag" AB_SPLICE_CKPT=40 \
    AB_SPLICE_ALT_CKPT=40 AB_SPLICE_NUM_SAMPLES="$samples" AB_SPLICE_NFE=1000 \
    AB_SPLICE_BATCH_SIZE=512 AB_SPLICE_SEED=1 AB_SPLICE_RUNS="$runs" \
    bash ./run_ab_logsnr_splice_eval.sh
}
A="$AB_SPLICE_A_WORKDIR"
B="$AB_SPLICE_B_WORKDIR"
echo "Queue started with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES at $(date -Is)"
endpoint night_smoke 512 1000 'A:common B:common'
splice night_smoke 512 'none full'
run "$((LIMIT - RESERVE))" python validate_ab_splice_identity.py \
  --direct-a "$A/eval_ab_endpoint_denoise_night_smoke_A_common_ckpt40_nfe1000_512samples_batch512_seed1" \
  --direct-b "$B/eval_ab_endpoint_denoise_night_smoke_B_common_ckpt40_nfe1000_512samples_batch512_seed1" \
  --route-a "$A/eval_ab_splice_night_smoke_none_ckpt40_nfe1000_512samples_batch512_seed1" \
  --route-b "$A/eval_ab_splice_night_smoke_full_ckpt40_nfe1000_512samples_batch512_seed1" \
  --checkpoint 40
endpoint night_seed1 10000 1000 'A:common'
endpoint night_seed1 10000 1000 'B:common'
splice night_seed1 10000 'head:-3.3'
splice night_seed1 10000 'tail:-3.3'
run "$LIMIT" python -u e3b_paired_bootstrap.py \
  --reference "$REFERENCE" \
  --baseline "$A/eval_ab_endpoint_denoise_night_seed1_A_common_ckpt40_nfe1000_10000samples_batch512_seed1" \
  --splice \
    "$B/eval_ab_endpoint_denoise_night_seed1_B_common_ckpt40_nfe1000_10000samples_batch512_seed1" \
    "$A/eval_ab_splice_night_seed1_head-3.3_ckpt40_nfe1000_10000samples_batch512_seed1" \
    "$A/eval_ab_splice_night_seed1_tail-3.3_ckpt40_nfe1000_10000samples_batch512_seed1" \
  --checkpoint 40 --count 10000 --reps 500 --seed 1 \
  --output results/ab_200k_seed1_bootstrap.json
echo "Queue completed at $(date -Is)"
