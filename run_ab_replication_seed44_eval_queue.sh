#!/usr/bin/env bash
# Server: bash run_ab_replication_seed44_eval_queue.sh GPU_ID [TRAIN_CONTAINER]
# Waits without reserving a GPU, then evaluates 200k and 100k on sampling seed0.
# AB_REPLICATION_INCLUDE_SEED42=1 appends matched 10k eval of the old pair.
set -euo pipefail

seed42_eval() {
  sha256sum --check --status "$seed42_hashfile" || {
    echo 'Seed42 manifests/checkpoints changed while waiting.' >&2; exit 1; }
  local mounts=()
  for name in A_vp_pvp B_vp_pfox; do
    local arm="celeba_factorial_${name}_50k"
    [[ ! -e "$seed42_output/workdirs/$arm" ]] || {
      echo "Seed42 output already exists: $arm" >&2; exit 1; }
    mkdir -p "$seed42_output/workdirs/$arm/checkpoints"
    cp "$main/workdirs/$arm/run_manifest.json" "$seed42_output/workdirs/$arm/run_manifest.json"
    mounts+=(--mount "type=bind,src=$main/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
  done
  printf 'evaluating_seed42\n' > "$state_file"
  printf 'evaluating\n' > "$seed42_state"
  docker run --name ab-seed42-common10k-eval-seed0 --gpus "device=$gpu" --ipc=host \
    --mount "type=bind,src=$main,dst=$root,readonly" \
    --mount "type=bind,src=$main/workdirs,dst=/workspace/training_workdirs,readonly" \
    --mount "type=bind,src=$seed42_output/workdirs,dst=$root/workdirs" \
    --mount "type=bind,src=$seed42_output/logs,dst=$root/logs" \
    "${mounts[@]}" \
    --mount "type=bind,src=$main/.tfds,dst=$tfds" \
    --mount "type=bind,src=$main/.torch_extensions,dst=$extensions" \
    --mount "type=bind,src=$main/.cache,dst=$cache" \
    --mount "type=bind,src=$eval_launcher,dst=/opt/ab_replication/run_ab_replication_seed44_eval.sh,readonly" \
    --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
    --env AB_EVAL_TRAINING_SEED=42 \
    --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache" \
    --env AB_REPLICATION_EVAL_SOURCE_SHA256="$source_hash" \
    --env AB_REPLICATION_EVAL_REFERENCE_SHA256="$reference_hash" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}" \
    --entrypoint bash "$image_id" /opt/ab_replication/run_ab_replication_seed44_eval.sh
  printf 'completed\n' > "$seed42_state"
}

worker() {
  local exit_code
  trap 'rc=$?; printf "%s\n" "$rc" > "$output/logs/ab_eval_seed0_queue.exitcode"; if (( rc != 0 )); then printf "failed\n" > "$state_file"; if [[ "$include_seed42" == 1 ]]; then printf "failed\n" > "$seed42_state"; fi; fi' EXIT
  printf 'waiting\n' > "$state_file"
  echo "Waiting for training container $train_id at $(date -u -Is)"
  exit_code="$(docker wait "$train_id")"
  [[ "$exit_code" == 0 ]] || {
    echo "Training exit code $exit_code; evaluation will not start." >&2; exit 1; }
  echo "Training finished successfully at $(date -u -Is)"
  [[ "$(sha256sum "$eval_launcher" | cut -d ' ' -f 1)" == "$launcher_hash" ]] || {
    echo 'Eval launcher changed after queue creation.' >&2; exit 1; }
  [[ "$(sha256sum "$reference" | cut -d ' ' -f 1)" == "$reference_hash" ]] || {
    echo 'CelebA reference changed after queue creation.' >&2; exit 1; }
  for name in A_vp_pvp B_vp_pfox; do
    local arm="celeba_ab_replication_${name}_seed44_200k"
    for ckpt in 4 2; do
      [[ -f "$output/workdirs/$arm/checkpoints/checkpoint_${ckpt}.pth" ]] || {
        echo "Missing checkpoint $ckpt for $arm" >&2; exit 1; }
    done
    [[ -f "$output/workdirs/$arm/run_manifest.json" ]] || {
      echo "Missing training manifest for $arm" >&2; exit 1; }
    [[ ! -e "$output/eval_workdirs/$arm" ]] || {
      echo "Eval workdir already exists: $arm" >&2; exit 1; }
  done
  mkdir -p "$main/logs" "$main/workdirs" "$output/eval_workdirs"
  local mounts=()
  for name in A_vp_pvp B_vp_pfox; do
    local arm="celeba_ab_replication_${name}_seed44_200k"
    mkdir -p "$output/eval_workdirs/$arm/checkpoints"
    cp "$output/workdirs/$arm/run_manifest.json" "$output/eval_workdirs/$arm/run_manifest.json"
    mounts+=(--mount "type=bind,src=$output/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
  done
  for path in "$tfds" "$extensions" "$cache"; do
    case "$path" in "$root"/*) mkdir -p "$main/${path#"$root"/}" ;; esac
  done
  printf 'evaluating\n' > "$state_file"
  docker run --name ab-replication-seed44-eval-seed0 --gpus "device=$gpu" --ipc=host \
    --mount "type=bind,src=$main,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs,dst=/workspace/training_workdirs,readonly" \
    --mount "type=bind,src=$output/eval_workdirs,dst=$root/workdirs" \
    --mount "type=bind,src=$output/logs,dst=$root/logs" \
    "${mounts[@]}" \
    --mount "type=bind,src=$main/.tfds,dst=$tfds" \
    --mount "type=bind,src=$main/.torch_extensions,dst=$extensions" \
    --mount "type=bind,src=$main/.cache,dst=$cache" \
    --mount "type=bind,src=$eval_launcher,dst=/opt/ab_replication/run_ab_replication_seed44_eval.sh,readonly" \
    --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
    --env AB_EVAL_TRAINING_SEED=44 \
    --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache" \
    --env AB_REPLICATION_EVAL_SOURCE_SHA256="$source_hash" \
    --env AB_REPLICATION_EVAL_REFERENCE_SHA256="$reference_hash" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}" \
    --entrypoint bash "$image_id" /opt/ab_replication/run_ab_replication_seed44_eval.sh
  printf '0\n' > "$output/logs/seed44_eval.exitcode"
  if [[ "$include_seed42" == 1 ]]; then seed42_eval; fi
  printf 'completed\n' > "$state_file"
  echo "Primary eval completed at $(date -u -Is)"
}

if [[ "${1:-}" == --worker ]]; then
  worker
  exit 0
fi

gpu="${1:?Usage: bash run_ab_replication_seed44_eval_queue.sh GPU_ID [TRAIN_CONTAINER]}"
train_container="${2:-ab-replication-seed44-train}"
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { echo 'Invalid physical GPU.' >&2; exit 2; }
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AB_REPLICATION_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="$base/score_sde_pytorch"
output="$base/ab_replication_seed44"
include_seed42="${AB_REPLICATION_INCLUDE_SEED42:-0}"
case "$include_seed42" in 0|1) ;; *) echo 'AB_REPLICATION_INCLUDE_SEED42 must be 0 or 1.' >&2; exit 2 ;; esac
seed42_output="$base/ab_seed42_common10k"
seed42_state="$seed42_output/logs/eval.state"
seed42_hashfile="$output/logs/seed42_input_sha256.txt"
root=/workspace/score_sde_pytorch
eval_launcher="$bundle_dir/run_ab_replication_seed44_eval.sh"
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
case "$train_state" in running|exited) ;; *) echo "Training state is $train_state" >&2; exit 1 ;; esac
physical_gpu="$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{println .}}{{end}}{{end}}' "$train_id")"
[[ "$physical_gpu" == "$gpu" ]] || { echo "Training physical GPU is $physical_gpu, requested $gpu" >&2; exit 1; }
training_mount="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/score_sde_pytorch/workdirs"}}{{.Source}}{{end}}{{end}}' "$train_id")"
[[ "$(readlink -f "$training_mount")" == "$(readlink -f "$output/workdirs")" ]] || {
  echo 'Training container uses a different workdir mount.' >&2; exit 1; }
if docker inspect ab-replication-seed44-eval-seed0 >/dev/null 2>&1; then
  echo 'The seed0 eval container already exists; inspect it before requeuing.' >&2; exit 1
fi
mkdir -p "$output/logs" "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
state_file="$output/logs/ab_eval_seed0_queue.state"
exec 9> "$output/logs/ab_eval_seed0_queue.lock"
flock -n 9 || { echo 'A primary eval queue is already active.' >&2; exit 1; }
if [[ "$include_seed42" == 1 ]]; then
  if docker inspect ab-seed42-common10k-eval-seed0 >/dev/null 2>&1; then
    echo 'Seed42 eval container already exists.' >&2; exit 1
  fi
  mkdir -p "$seed42_output/logs" "$seed42_output/workdirs"
  exec 7> "$seed42_output/logs/eval.lock"
  flock -n 7 || { echo 'Seed42 eval is already queued.' >&2; exit 1; }
  inputs=()
  for name in A_vp_pvp B_vp_pfox; do
    arm="celeba_factorial_${name}_50k"
    [[ ! -e "$seed42_output/workdirs/$arm" ]] || { echo "Seed42 eval output exists: $arm" >&2; exit 1; }
    for file in run_manifest.json checkpoints/checkpoint_40.pth checkpoints/checkpoint_20.pth; do
      [[ -f "$main/workdirs/$arm/$file" ]] || { echo "Missing seed42 input: $arm/$file" >&2; exit 1; }
      inputs+=("$main/workdirs/$arm/$file")
    done
  done
  sha256sum "${inputs[@]}" > "$seed42_hashfile"
fi
for name in A_vp_pvp B_vp_pfox; do
  [[ ! -e "$output/eval_workdirs/celeba_ab_replication_${name}_seed44_200k" ]] || {
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
log_file="$output/logs/ab_eval_seed0_queue_${run_id}.log"
printf 'train_id=%s\nimage_id=%s\ngpu=%s\nsource_sha256=%s\nreference_sha256=%s\nlauncher_sha256=%s\n' \
  "$train_id" "$image_id" "$gpu" "$source_hash" "$reference_hash" "$launcher_hash" \
  > "$output/logs/ab_eval_seed0_queue_identity.txt"
git -C "$bundle_dir" rev-parse HEAD > "$output/logs/ab_eval_seed0_queue_launcher_commit.txt"
export gpu train_id image_id main output root eval_launcher source_hash reference reference_hash launcher_hash
export tfds extensions cache state_file
export include_seed42 seed42_output seed42_state seed42_hashfile
if [[ "$include_seed42" == 1 ]]; then printf 'waiting_seed44\n' > "$seed42_state"; fi
printf 'queued\n' > "$state_file"
nohup bash "$bundle_dir/run_ab_replication_seed44_eval_queue.sh" --worker \
  > "$log_file" 2>&1 < /dev/null 9>&9 &
queue_pid=$!
printf '%s\n' "$queue_pid" > "$output/logs/ab_eval_seed0_queue.pid"
printf 'Queued seed0 primary eval: PID=%s, training=%s, GPU=%s\nLog: %s\nState: %s\n' \
  "$queue_pid" "$train_container" "$gpu" "$log_file" "$state_file"
printf 'Seed42 follow-up enabled: %s\n' "$include_seed42"
