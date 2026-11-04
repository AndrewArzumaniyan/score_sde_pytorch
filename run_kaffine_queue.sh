#!/usr/bin/env bash
# In-container queue for the k_a(t) shape experiment (configs/fox/celeba_kaffine.py).
# Started by run_kaffine_rtx.sh; do not run on the host.
#
# Stages run in the given order, each one is skipped when its result exists:
#   preflight                 environment, arm configs, shared initial weights
#   smoke                     400 training steps of a0 + one 512-image eval round (timing, memory)
#   train:ARM                 100k steps, resumes from checkpoints-meta after a crash
#   eval:ARM:CKPT:SAMPLES     common window [-10, 10.5757], EM 1000 + Tweedie, seed 0
set -euo pipefail

[[ -f /.dockerenv && "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || {
  echo 'Run inside Docker with one physical GPU and CUDA_VISIBLE_DEVICES=0.' >&2; exit 2; }
cd /workspace/score_sde_pytorch
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" logs workdirs

CONFIG=configs/fox/celeba_kaffine.py
ARMS='a0 ap5 vplin_norm vplin_raw ap9p95'
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

preflight() {
  python -u - <<'PY'
import gc, hashlib, json, os
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Exactly one visible CUDA GPU is required.')
torch.cuda.init()
import run_lib
from artifact_utils import _canonicalize, _environment_manifest
from configs.fox.celeba_kaffine import ARMS, get_config
from reproducibility import configure_reproducibility

capability = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
stats = Path('assets/stats/celeba_64_0c2d89311192_stats.npz')
if not stats.is_file():
    raise RuntimeError(f'Missing FID reference: {stats}')

def scientific(config):
    result = json.loads(json.dumps(_canonicalize(config)))
    result['model'].pop('fox_drift_a'); result['model'].pop('fox_normalize_scale')
    result['training'].pop('kaffine_arm')
    return result

base = scientific(get_config('a0'))
audit = dict(gpu=name, capability=list(capability), environment=_environment_manifest(),
             fid_reference_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(), arms={})
for arm in ARMS:
    config = get_config(arm)
    if scientific(config) != base:
        raise RuntimeError(f'Arm {arm} differs from a0 outside drift_a / normalize_scale.')
    sde, _ = run_lib.get_sde(config)
    lam = [float(sde.log_snr(torch.tensor([t])).item()) for t in (1e-5, 1e-3, 1.0)]
    if not (lam[0] > config.sampling.logsnr_max and lam[2] < config.sampling.logsnr_min):
        raise RuntimeError(f'Arm {arm}: common window outside training support {lam}.')
    configure_reproducibility(config, tensorflow_module=run_lib.tf)
    model = run_lib.mutils.create_model(config)
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode() + b'\0' + tensor.detach().cpu().contiguous().numpy().tobytes())
    audit['arms'][arm] = dict(drift_a=config.model.fox_drift_a,
                              normalize_scale=config.model.fox_normalize_scale,
                              logsnr_at_1e-5_1e-3_1=lam, initial_state_sha256=digest.hexdigest())
    del model; gc.collect(); torch.cuda.empty_cache()
if len({run['initial_state_sha256'] for run in audit['arms'].values()}) != 1:
    raise RuntimeError('Arms do not share the initial weights.')
audit['status'] = 'passed'
import time
Path(time.strftime('logs/preflight_%Y%m%dT%H%M%SZ.json', time.gmtime())).write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'environment'}, indent=2), flush=True)
PY
}

train_arm() {
  local arm="$1" workdir="workdirs/celeba_kaffine_${1}_100k_seed42"
  if [[ -s "$workdir/checkpoints/checkpoint_4.pth" ]]; then
    log "train:$arm already has checkpoint_4, skipped"; return; fi
  log "train:$arm -> $workdir (100k steps, ckpt every 25k, resumable)"
  python -u main.py --mode=train --config="$CONFIG:$arm" --workdir="$workdir"
  log "train:$arm finished"
}

eval_arm() {
  local arm="$1" ckpt="$2" samples="$3" seed=0
  local workdir="workdirs/celeba_kaffine_${arm}_100k_seed42"
  local folder="eval_kaffine_common_ckpt${ckpt}_nfe1000_${samples}samples_seed${seed}"
  [[ -s "$workdir/checkpoints/checkpoint_${ckpt}.pth" ]] || {
    echo "Missing checkpoint: $workdir/checkpoints/checkpoint_${ckpt}.pth" >&2; exit 1; }
  if [[ -s "$workdir/$folder/report_${ckpt}.npz" ]]; then
    log "eval:$arm:$ckpt:$samples already reported, skipped"; return; fi
  log "eval:$arm ckpt=$ckpt samples=$samples seed=$seed"
  python -u main.py --mode=eval --config="$CONFIG:$arm" --workdir="$workdir" \
    --eval_folder="$folder" \
    --config.eval.begin_ckpt="$ckpt" --config.eval.end_ckpt="$ckpt" \
    --config.eval.num_samples="$samples" --config.eval.sampling_seed="$seed"
  log "eval:$arm:$ckpt:$samples finished"
}

smoke() {
  local workdir=workdirs/smoke_a0
  if [[ -s logs/smoke_done.txt ]]; then log 'smoke already done, skipped'; return; fi
  rm -rf "$workdir"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 \
    > logs/smoke_gpu.csv & local monitor=$!
  local start; start="$(date +%s)"
  python -u main.py --mode=train --config="$CONFIG:a0" --workdir="$workdir" \
    --config.training.n_iters=400 --config.training.snapshot_freq=200 \
    --config.training.snapshot_freq_for_preemption=200
  local trained; trained="$(date +%s)"
  python -u main.py --mode=eval --config="$CONFIG:a0" --workdir="$workdir" \
    --eval_folder=eval_smoke --config.training.n_iters=400 \
    --config.training.snapshot_freq=200 \
    --config.eval.begin_ckpt=2 --config.eval.end_ckpt=2 \
    --config.eval.num_samples=512
  kill "$monitor" 2>/dev/null || true
  local done_at; done_at="$(date +%s)"
  # A 10k eval is 20 rounds of 512 images, a 5k eval is 10 rounds.
  printf 'train_400_steps_s=%s\neval_one_512_round_nfe1000_s=%s\npeak_memory_mib=%s\n' \
    "$((trained - start))" "$((done_at - trained))" \
    "$(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' logs/smoke_gpu.csv)" \
    | tee logs/smoke_done.txt
  log 'smoke: s/step from consecutive "step:" lines of the training log is the precise number'
}

(( $# > 0 )) || { echo 'Usage: run_kaffine_queue.sh STAGE...' >&2; exit 2; }
for stage in "$@"; do
  IFS=: read -r kind arm ckpt samples <<< "$stage"
  if [[ -n "${arm:-}" && " $ARMS " != *" $arm "* ]]; then
    echo "Unknown arm in stage $stage" >&2; exit 2; fi
  case "$kind" in
    preflight) preflight ;;
    smoke) smoke ;;
    train) [[ -n "${arm:-}" ]] || { echo "Bad stage: $stage" >&2; exit 2; }; train_arm "$arm" ;;
    eval)
      [[ "${ckpt:-}" =~ ^[0-9]+$ && "${samples:-}" =~ ^[0-9]+$ ]] || {
        echo "Bad stage: $stage (eval:ARM:CKPT:SAMPLES)" >&2; exit 2; }
      eval_arm "$arm" "$ckpt" "$samples" ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
done
log "queue finished: $*"
