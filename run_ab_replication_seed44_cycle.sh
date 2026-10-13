#!/usr/bin/env bash
# Server: bash run_ab_replication_seed44_cycle.sh GPU_ID [--with-seed42]
# Fresh A/B training, followed automatically by common/10k eval at 200k/100k.
set -euo pipefail
gpu="${1:?Usage: bash run_ab_replication_seed44_cycle.sh GPU_ID [--with-seed42]}"
[[ $# == 1 || ( $# == 2 && "$2" == --with-seed42 ) ]] || { echo 'Expected GPU_ID [--with-seed42].' >&2; exit 2; }
export AB_REPLICATION_INCLUDE_SEED42=0
[[ $# == 1 ]] || export AB_REPLICATION_INCLUDE_SEED42=1
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AB_REPLICATION_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
mkdir -p "$base/ab_replication_seed44/logs"
exec 8> "$base/ab_replication_seed44/logs/cycle.lock"
flock -n 8 || { echo 'Seed44 cycle setup is already active.' >&2; exit 1; }
for file in run_ab_replication_train.sh run_ab_replication_seed43_docker.sh \
  run_ab_replication_seed44_eval.sh run_ab_replication_seed44_eval_queue.sh; do
  [[ -f "$bundle_dir/$file" ]] || { echo "Missing launcher: $file" >&2; exit 1; }
done
[[ -f "$base/score_sde_pytorch/assets/stats/celeba_64_0c2d89311192_stats.npz" ]] || {
  echo 'Missing CelebA eval reference.' >&2; exit 1; }
for container in ab-replication-seed44-train ab-replication-seed44-eval-seed0; do
  if docker inspect "$container" >/dev/null 2>&1; then
    echo "Container already exists: $container; inspect before restarting." >&2; exit 1
  fi
done
for arm in A_vp_pvp B_vp_pfox; do
  target="$base/ab_replication_seed44/eval_workdirs/celeba_ab_replication_${arm}_seed44_200k"
  [[ ! -e "$target" ]] || { echo "Eval workdir exists: $target" >&2; exit 1; }
done
if [[ "$AB_REPLICATION_INCLUDE_SEED42" == 1 ]]; then
  for arm in A_vp_pvp B_vp_pfox; do
    for file in run_manifest.json checkpoints/checkpoint_40.pth checkpoints/checkpoint_20.pth; do
      [[ -f "$base/score_sde_pytorch/workdirs/celeba_factorial_${arm}_50k/$file" ]] || {
        echo "Missing seed42 artifact: $arm/$file" >&2; exit 1; }
    done
    [[ ! -e "$base/ab_seed42_common10k/workdirs/celeba_factorial_${arm}_50k" ]] || {
      echo "Seed42 eval workdir already exists: $arm" >&2; exit 1; }
  done
  if docker inspect ab-seed42-common10k-eval-seed0 >/dev/null 2>&1; then
    echo 'Seed42 eval container already exists.' >&2; exit 1
  fi
fi
export AB_REPLICATION_SEED=44
bash "$bundle_dir/run_ab_replication_seed43_docker.sh" train "$gpu"
if ! bash "$bundle_dir/run_ab_replication_seed44_eval_queue.sh" "$gpu" ab-replication-seed44-train; then
  echo 'Training was started, but eval queue setup failed. Do not restart training.' >&2
  echo "After fixing the error, run: AB_REPLICATION_INCLUDE_SEED42=$AB_REPLICATION_INCLUDE_SEED42 bash run_ab_replication_seed44_eval_queue.sh $gpu ab-replication-seed44-train" >&2
  exit 1
fi
