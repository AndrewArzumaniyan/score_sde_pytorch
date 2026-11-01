#!/usr/bin/env bash
# Inside Docker only: 12 new native VP/FOX trajectory cells; no anchor reruns.
set -euo pipefail
cd /workspace/score_sde_pytorch
exec > >(tee -a logs/endpoint_screen.log) 2>&1
trap 'code=$?; printf "failed exit=%s\n" "$code" > logs/endpoint_screen.state' ERR

check_inputs() {
  sha256sum -c /opt/endpoint-screen/runtime.sha256
  sha256sum -c /opt/endpoint-screen/inputs.sha256
}

check_inputs
printf 'running\n' > logs/endpoint_screen.state
for endpoint in 5 6 7 8 10 12; do
  for model in vp fox; do
    check_inputs
    if [[ "$model" == vp ]]; then
      arm=celeba_vp_continuous_50k
    else
      arm=celeba_fox_matern32_kappa1000_50k
    fi
    echo "=== Endpoint screen: $model lambda_max=$endpoint ckpt=260 samples=10000 seed=0; $(date -Is) ==="
    python -u main.py \
      --mode=eval --config="endpoint_screen_config.py:${model}_${endpoint}" \
      --workdir="workdirs/$arm" \
      --eval_folder="eval_endpoint_screen_v1_${model}_lambda${endpoint}_ckpt260_nfe1000_10000samples_batch512_seed0" \
      --config.eval.begin_ckpt=260 --config.eval.end_ckpt=260 \
      --config.eval.batch_size=512 --config.eval.num_samples=10000 \
      --config.eval.enable_loss=False --config.eval.enable_sampling=True \
      --config.eval.enable_bpd=False --config.eval.sampling_num_scales=1000 \
      --config.eval.sampling_seed=0 --config.sampling.method=pc \
      --config.sampling.predictor=euler_maruyama --config.sampling.corrector=none \
      --config.seed=42 --config.allow_tf32=True --config.cudnn_benchmark=True \
      --config.allow_fp16_reduced_precision_reduction=False
  done
done
printf 'completed\n' > logs/endpoint_screen.state
echo "Endpoint screen completed at $(date -Is)"
