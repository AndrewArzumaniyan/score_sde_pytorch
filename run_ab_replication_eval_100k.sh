#!/usr/bin/env bash
# Server: bash run_ab_replication_eval_100k.sh GPU_ID [TRAIN_CONTAINER]
set -euo pipefail

eval_container() {
[[ -f /.dockerenv && "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || {
  echo 'Use Docker with one physical GPU and CUDA_VISIBLE_DEVICES=0.' >&2; exit 2; }
cd /workspace/score_sde_pytorch
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

python -u - <<'PY'
import gc
import hashlib
import json
import os
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Eval requires exactly one visible CUDA GPU.')
torch.cuda.init()
from ml_collections import ConfigDict
from artifact_utils import _source_fingerprint, _environment_manifest, build_model_manifest
import evaluation

source = _source_fingerprint(os.getcwd())
if source != os.environ['AB_REPLICATION_EVAL_SOURCE_SHA256']:
    raise RuntimeError('Eval Python sources changed after queue creation.')
environment = _environment_manifest()
reference = Path('assets/stats/celeba_64_0c2d89311192_stats.npz')
reference_digest = hashlib.sha256()
with reference.open('rb') as reference_file:
    for block in iter(lambda: reference_file.read(1024 * 1024), b''):
        reference_digest.update(block)
reference_hash = reference_digest.hexdigest()
if reference_hash != os.environ['AB_REPLICATION_EVAL_REFERENCE_SHA256']:
    raise RuntimeError('CelebA reference changed after queue creation.')
audit = dict(source_sha256=source, reference_sha256=reference_hash,
             environment=environment, sampling_seed=0, num_samples=10000,
             checkpoint=2, expected_checkpoint_step=100001,
             endpoint='common', em_steps=1000, nfe=1001, runs={})
for arm, name, allocation in (
    ('A', 'A_vp_pvp', 'vp_time'), ('B', 'B_vp_pfox', 'normalized_fox_time')):
    root = Path('/workspace/training_workdirs') / f'celeba_ab_replication_{name}_seed43_200k'
    manifest = json.loads((root / 'run_manifest.json').read_text())
    config = ConfigDict(manifest['config'])
    if (manifest['source_sha256'] != 'ddf4c6acbc51b67da7a3465f2b90678ff559a628f6925d47e991350b02ff54f7'
            or config.seed != 43
            or config.training.snapshot_freq != 50000
            or config.training.batch_size != 128
            or config.training.sde != 'vpsde'
            or config.training.factorial_arm != name
            or config.training.noise_distribution != allocation):
        raise RuntimeError(f'Unexpected training manifest for {arm}.')
    for key in ('python', 'numpy', 'tensorflow', 'torch', 'torch_cuda', 'cudnn', 'packages'):
        if environment[key] != manifest['environment'][key]:
            raise RuntimeError(f'Training/eval environment differs in {key}.')
    if evaluation._get_stats_filename(config) != str(reference):
        raise RuntimeError('Eval would select a different CelebA reference.')
    checkpoint = root / 'checkpoints/checkpoint_2.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (int(state['step']) != 100001
            or state['training_protocol_sha256'] != manifest['training_protocol_sha256']
            or state['model_protocol_sha256'] != build_model_manifest(config)['model_protocol_sha256']):
        raise RuntimeError(f'Checkpoint identity/step mismatch for {arm}.')
    audit['runs'][arm] = dict(checkpoint_id=state['checkpoint_id'], step=int(state['step']),
                             training_protocol_sha256=manifest['training_protocol_sha256'])
    del state
    gc.collect()
audit['status'] = 'passed'
Path('logs/ab_replication_eval_100k_seed0_preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2), flush=True)
PY

tmp_config="$(mktemp /tmp/ab_replication_eval.XXXXXX.py)"
trap 'rm -f "$tmp_config"' EXIT
cat > "$tmp_config" <<'PY'
import json
from pathlib import Path
from ml_collections import ConfigDict
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

def get_config(arm):
    name = {'A': 'A_vp_pvp', 'B': 'B_vp_pfox'}[arm]
    root = Path('/workspace/training_workdirs') / f'celeba_ab_replication_{name}_seed43_200k'
    config = ConfigDict(json.loads((root / 'run_manifest.json').read_text())['config'])
    # Training manifests deliberately omit eval and orchestration fields.
    config.eval = (get_a if arm == 'A' else get_b)().eval
    config.training.n_iters = 200000
    config.sampling.time_grid = 'uniform_logsnr'
    config.sampling.logsnr_min = -10.0
    config.sampling.logsnr_max = 9.115429865459795
    config.sampling.noise_removal = True
    config.eval.endpoint_denoise_variants = True
    config.eval.include_final_checkpoint = False
    return config
PY
for arm in A B; do
  if [[ "$arm" == A ]]; then name=A_vp_pvp; else name=B_vp_pfox; fi
  workdir="workdirs/celeba_ab_replication_${name}_seed43_200k"
  folder="eval_ab_replication_seed43_${arm}_common_ckpt2_nfe1000_10000samples_batch512_seed0"
  echo "=== A/B replication eval $arm/common: ckpt=2, 10k, sampling_seed=0, NFE=1001 ==="
  python -u main.py --mode=eval --config="$tmp_config:$arm" \
    --workdir="$workdir" --eval_folder="$folder" \
    --config.eval.begin_ckpt=2 --config.eval.end_ckpt=2 \
    --config.eval.batch_size=512 --config.eval.num_samples=10000 \
    --config.eval.enable_loss=False --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False --config.eval.sampling_num_scales=1000 \
    --config.eval.sampling_seed=0 --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama --config.sampling.corrector=none \
    --config.allow_tf32=True --config.cudnn_benchmark=True
done
echo 'A/B replication primary eval completed; reports and features saved.'
}

if [[ "${1:-}" == --container ]]; then
  eval_container
  exit 0
fi


worker() {
  trap 'rc=$?; printf "%s\n" "$rc" > "$output/logs/ab_eval_100k_seed0.exitcode"; if (( rc != 0 )); then printf "failed\n" > "$state_file"; fi' EXIT
  echo "Starting 100k evaluation at $(date -u -Is)"
  [[ "$(sha256sum "$eval_launcher" | cut -d ' ' -f 1)" == "$launcher_hash" ]] || {
    echo 'Eval launcher changed after queue creation.' >&2; exit 1; }
  [[ "$(sha256sum "$reference" | cut -d ' ' -f 1)" == "$reference_hash" ]] || {
    echo 'CelebA reference changed after queue creation.' >&2; exit 1; }
  for name in A_vp_pvp B_vp_pfox; do
    local arm="celeba_ab_replication_${name}_seed43_200k"
    [[ -f "$output/workdirs/$arm/checkpoints/checkpoint_2.pth" ]] || {
      echo "Missing 100k checkpoint for $arm" >&2; exit 1; }
    [[ -f "$output/workdirs/$arm/run_manifest.json" ]] || {
      echo "Missing training manifest for $arm" >&2; exit 1; }
    [[ ! -e "$output/eval_workdirs_100k/$arm" ]] || {
      echo "Eval workdir already exists: $arm" >&2; exit 1; }
  done
  mkdir -p "$main/logs" "$main/workdirs" "$output/eval_workdirs_100k"
  local mounts=()
  for name in A_vp_pvp B_vp_pfox; do
    local arm="celeba_ab_replication_${name}_seed43_200k"
    mkdir -p "$output/eval_workdirs_100k/$arm/checkpoints"
    cp "$output/workdirs/$arm/run_manifest.json" "$output/eval_workdirs_100k/$arm/run_manifest.json"
    mounts+=(--mount "type=bind,src=$output/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
  done
  for path in "$tfds" "$extensions" "$cache"; do
    case "$path" in "$root"/*) mkdir -p "$main/${path#"$root"/}" ;; esac
  done
  printf 'evaluating\n' > "$state_file"
  docker run --name ab-replication-seed43-eval-100k-seed0 --gpus "device=$gpu" --ipc=host \
    --mount "type=bind,src=$main,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs,dst=/workspace/training_workdirs,readonly" \
    --mount "type=bind,src=$output/eval_workdirs_100k,dst=$root/workdirs" \
    --mount "type=bind,src=$output/logs,dst=$root/logs" \
    "${mounts[@]}" \
    --mount "type=bind,src=$main/.tfds,dst=$tfds" \
    --mount "type=bind,src=$main/.torch_extensions,dst=$extensions" \
    --mount "type=bind,src=$main/.cache,dst=$cache" \
    --mount "type=bind,src=$eval_launcher,dst=/opt/ab_replication/run_ab_replication_eval_100k.sh,readonly" \
    --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
    --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache" \
    --env AB_REPLICATION_EVAL_SOURCE_SHA256="$source_hash" \
    --env AB_REPLICATION_EVAL_REFERENCE_SHA256="$reference_hash" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}" \
    --entrypoint bash "$image_id" /opt/ab_replication/run_ab_replication_eval_100k.sh --container
  printf 'completed\n' > "$state_file"
  echo "Primary eval completed at $(date -u -Is)"
}

if [[ "${1:-}" == --worker ]]; then
  worker
  exit 0
fi

gpu="${1:?Usage: bash run_ab_replication_eval_100k.sh GPU_ID [TRAIN_CONTAINER]}"
for existing in ab-replication-seed43-eval-seed0 ab-replication-seed43-eval-seed1; do
  if [[ "$(docker inspect --format '{{.State.Running}}' "$existing" 2>/dev/null || true)" == true ]]; then
    echo "Eval container $existing is still running; wait before starting 100k." >&2
    exit 1
  fi
done
train_container="${2:-ab-replication-seed43-train}"
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { echo 'Invalid physical GPU.' >&2; exit 2; }
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AB_REPLICATION_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="$base/score_sde_pytorch"
output="$base/ab_replication_seed43"
root=/workspace/score_sde_pytorch
eval_launcher="$bundle_dir/run_ab_replication_eval_100k.sh"
reference="$main/assets/stats/celeba_64_0c2d89311192_stats.npz"
tfds="${TFDS_DATA_DIR:-$root/.tfds}"
extensions="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
cache="${XDG_CACHE_HOME:-$root/.cache}"
for path in "$eval_launcher" "$reference" "$main/artifact_utils.py" "$main/endpoint_denoise_eval.py"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$tfds" "$extensions" "$cache"; do
  [[ "$path" == /* && "$path" != / && "$path" != "$root" ]] || {
    echo "Invalid cache directory: $path" >&2; exit 2; }
done
train_id="$(docker inspect --format '{{.Id}}' "$train_container")"
image_id="$(docker inspect --format '{{.Image}}' "$train_id")"
train_state="$(docker inspect --format '{{.State.Status}}' "$train_id")"
[[ "$train_state" == exited && "$(docker inspect --format '{{.State.ExitCode}}' "$train_id")" == 0 ]] || {
  echo 'Training must already have completed successfully.' >&2; exit 1; }
physical_gpu="$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{println .}}{{end}}{{end}}' "$train_id")"
[[ "$physical_gpu" == "$gpu" ]] || { echo "Training physical GPU is $physical_gpu, requested $gpu" >&2; exit 1; }
training_mount="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/score_sde_pytorch/workdirs"}}{{.Source}}{{end}}{{end}}' "$train_id")"
[[ "$(readlink -f "$training_mount")" == "$(readlink -f "$output/workdirs")" ]] || {
  echo 'Training container uses a different workdir mount.' >&2; exit 1; }
if docker inspect ab-replication-seed43-eval-100k-seed0 >/dev/null 2>&1; then
  echo 'The seed0 eval container already exists; inspect it before requeuing.' >&2; exit 1
fi
mkdir -p "$output/logs" "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
state_file="$output/logs/ab_eval_100k_seed0.state"
exec 9> "$output/logs/ab_eval_100k_seed0.lock"
flock -n 9 || { echo 'A primary eval queue is already active.' >&2; exit 1; }
for name in A_vp_pvp B_vp_pfox; do
  [[ ! -e "$output/eval_workdirs_100k/celeba_ab_replication_${name}_seed43_200k" ]] || {
    echo 'Primary eval workdirs already exist; inspect before requeuing.' >&2; exit 1; }
done
launcher_hash="$(sha256sum "$eval_launcher" | cut -d ' ' -f 1)"
reference_hash="$(sha256sum "$reference" | cut -d ' ' -f 1)"
# CPU-only snapshot of the actual source fingerprint, including untracked runtime
# files. No TensorFlow imports, no GPU request, no checkpoint reads while waiting.
source_hash="$(docker run --rm --network none --mount "type=bind,src=$main,dst=$root,readonly" \
  --workdir "$root" --entrypoint python "$image_id" -c 'import ast, hashlib, os; from pathlib import Path; tree=ast.parse(Path("artifact_utils.py").read_text()); fn=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="_source_fingerprint"); exec(compile(ast.Module(body=[fn], type_ignores=[]), "fingerprint", "exec")); print(_source_fingerprint(os.getcwd()))')"
[[ "$source_hash" =~ ^[a-f0-9]{64}$ ]] || { echo 'Could not fingerprint eval source.' >&2; exit 1; }
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$output/logs/ab_eval_100k_seed0_${run_id}.log"
printf 'train_id=%s\nimage_id=%s\ngpu=%s\nsource_sha256=%s\nreference_sha256=%s\nlauncher_sha256=%s\n' \
  "$train_id" "$image_id" "$gpu" "$source_hash" "$reference_hash" "$launcher_hash" \
  > "$output/logs/ab_eval_100k_seed0_identity.txt"
git -C "$bundle_dir" rev-parse HEAD > "$output/logs/ab_eval_100k_seed0_launcher_commit.txt"
export gpu train_id image_id main output root eval_launcher source_hash reference reference_hash launcher_hash
export tfds extensions cache state_file
printf 'queued\n' > "$state_file"
nohup bash "$bundle_dir/run_ab_replication_eval_100k.sh" --worker \
  > "$log_file" 2>&1 < /dev/null 9>&9 &
queue_pid=$!
printf '%s\n' "$queue_pid" > "$output/logs/ab_eval_100k_seed0.pid"
printf 'Queued seed0 primary eval: PID=%s, training=%s, GPU=%s\nLog: %s\nState: %s\n' \
  "$queue_pid" "$train_container" "$gpu" "$log_file" "$state_file"
