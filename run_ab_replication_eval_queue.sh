#!/usr/bin/env bash
# Server: bash run_ab_replication_eval_queue.sh GPU_ID [TRAIN_CONTAINER]
# Waits on the host without a GPU reservation. Launches the fixed seed0 eval.
set -euo pipefail

worker() {
  local exit_code
  trap 'rc=$?; printf "%s\n" "$rc" > "$output/logs/ab_eval_seed0_queue.exitcode"; if (( rc != 0 )); then printf "failed\n" > "$state_file"; fi' EXIT
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
    local arm="celeba_ab_replication_${name}_seed43_200k"
    [[ -f "$output/workdirs/$arm/checkpoints/checkpoint_4.pth" ]] || {
      echo "Missing final checkpoint for $arm" >&2; exit 1; }
    [[ -f "$output/workdirs/$arm/run_manifest.json" ]] || {
      echo "Missing training manifest for $arm" >&2; exit 1; }
    [[ ! -e "$output/eval_workdirs/$arm" ]] || {
      echo "Eval workdir already exists: $arm" >&2; exit 1; }
  done
  mkdir -p "$main/logs" "$main/workdirs" "$output/eval_workdirs"
  local mounts=()
  for name in A_vp_pvp B_vp_pfox; do
    local arm="celeba_ab_replication_${name}_seed43_200k"
    mkdir -p "$output/eval_workdirs/$arm/checkpoints"
    cp "$output/workdirs/$arm/run_manifest.json" "$output/eval_workdirs/$arm/run_manifest.json"
    mounts+=(--mount "type=bind,src=$output/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
  done
  for path in "$tfds" "$extensions" "$cache"; do
    case "$path" in "$root"/*) mkdir -p "$main/${path#"$root"/}" ;; esac
  done
  printf 'evaluating\n' > "$state_file"
  docker run --name ab-replication-seed43-eval-seed0 --gpus "device=$gpu" --ipc=host \
    --mount "type=bind,src=$main,dst=$root,readonly" \
    --mount "type=bind,src=$output/workdirs,dst=/workspace/training_workdirs,readonly" \
    --mount "type=bind,src=$output/eval_workdirs,dst=$root/workdirs" \
    --mount "type=bind,src=$output/logs,dst=$root/logs" \
    "${mounts[@]}" \
    --mount "type=bind,src=$main/.tfds,dst=$tfds" \
    --mount "type=bind,src=$main/.torch_extensions,dst=$extensions" \
    --mount "type=bind,src=$main/.cache,dst=$cache" \
    --mount "type=bind,src=$eval_launcher,dst=/opt/ab_replication/run_ab_replication_eval.sh,readonly" \
    --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1 \
    --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache" \
    --env AB_REPLICATION_EVAL_SOURCE_SHA256="$source_hash" \
    --env AB_REPLICATION_EVAL_REFERENCE_SHA256="$reference_hash" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}" \
    --entrypoint bash "$image_id" /opt/ab_replication/run_ab_replication_eval.sh
  printf 'completed\n' > "$state_file"
  echo "Primary eval completed at $(date -u -Is)"
}

if [[ "${1:-}" == --worker ]]; then
  worker
  exit 0
fi

gpu="${1:?Usage: bash run_ab_replication_eval_queue.sh GPU_ID [TRAIN_CONTAINER]}"
train_container="${2:-ab-replication-seed43-train}"
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { echo 'Invalid physical GPU.' >&2; exit 2; }
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${AB_REPLICATION_SERVER_BASE:-/home/e.nuzhin/kovshov-arzumanyan}"
main="$base/score_sde_pytorch"
output="$base/ab_replication_seed43"
root=/workspace/score_sde_pytorch
eval_launcher="$bundle_dir/run_ab_replication_eval.sh"
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
if docker inspect ab-replication-seed43-eval-seed0 >/dev/null 2>&1; then
  echo 'The seed0 eval container already exists; inspect it before requeuing.' >&2; exit 1
fi
mkdir -p "$output/logs" "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
state_file="$output/logs/ab_eval_seed0_queue.state"
exec 9> "$output/logs/ab_eval_seed0_queue.lock"
flock -n 9 || { echo 'A primary eval queue is already active.' >&2; exit 1; }
for name in A_vp_pvp B_vp_pfox; do
  [[ ! -e "$output/eval_workdirs/celeba_ab_replication_${name}_seed43_200k" ]] || {
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
printf 'queued\n' > "$state_file"
nohup bash "$bundle_dir/run_ab_replication_eval_queue.sh" --worker \
  > "$log_file" 2>&1 < /dev/null 9>&9 &
queue_pid=$!
printf '%s\n' "$queue_pid" > "$output/logs/ab_eval_seed0_queue.pid"
printf 'Queued seed0 primary eval: PID=%s, training=%s, GPU=%s\nLog: %s\nState: %s\n' \
  "$queue_pid" "$train_container" "$gpu" "$log_file" "$state_file"
