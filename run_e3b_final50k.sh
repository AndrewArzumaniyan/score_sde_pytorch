#!/usr/bin/env bash
# Container entrypoint. Frozen native VP/FOX@1.3M, extended, 50k, seeds 0/1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || { echo 'Expected exactly one Docker-routed GPU (CUDA_VISIBLE_DEVICES=0).' >&2; exit 2; }
mkdir -p logs
trap 'rc=$?; printf "%s\n" "$rc" > logs/final50k.exitcode; if (( rc != 0 )); then printf "failed\n" > logs/final50k.state; fi' EXIT
export E3B_SPLICE_CKPT=260 E3B_SPLICE_ALT_CKPT=260
export E3B_SPLICE_BATCH_SIZE=512 E3B_SPLICE_ALLOW_TF32=True E3B_SPLICE_CUDNN_BENCHMARK=True
export E3B_SPLICE_VP_WORKDIR=workdirs/celeba_vp_continuous_50k
export E3B_SPLICE_FOX_WORKDIR=workdirs/celeba_fox_matern32_kappa1000_50k
printf 'preflight\n' > logs/final50k.state
python -u e3b_final50k.py preflight
# Reuse old 50k VP only if its first full batch matches all three outputs.
# Both checks finish before any of the long cells starts.
for seed in 0 1; do
  printf 'identity_seed%s\n' "$seed" > logs/final50k.state
  E3B_SPLICE_RUNS=none E3B_SPLICE_SEED="$seed" E3B_SPLICE_RUN_TAG=final50k_identity \
    E3B_SPLICE_NUM_SAMPLES=512 bash run_e3b_splice_eval.sh 2>&1 | tee "logs/identity_seed${seed}.log"
  python -u e3b_final50k.py identity "$seed"
done
# Primary head/full contrasts on both seeds precede the complementary tail.
for cell in '0 head:-3.3' '0 full' '1 head:-3.3' '1 full' '0 tail:-3.3' '1 tail:-3.3'; do
  read -r seed route <<< "$cell"
  python -u e3b_final50k.py check
  printf 'evaluating_seed%s_%s\n' "$seed" "$route" > logs/final50k.state
  echo "=== Final50k seed=$seed route=$route at $(date -u -Is) ==="
  E3B_SPLICE_RUNS="$route" E3B_SPLICE_SEED="$seed" E3B_SPLICE_RUN_TAG=final50k \
    E3B_SPLICE_NUM_SAMPLES=50000 bash run_e3b_splice_eval.sh 2>&1 | tee "logs/seed${seed}_${route/:/}.log"
  python -u e3b_final50k.py report "$seed" "$route"
done
printf 'completed\n' > logs/final50k.state
echo "Final 50k splice queue completed at $(date -u -Is)"
