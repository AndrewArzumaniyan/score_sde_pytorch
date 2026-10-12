#!/usr/bin/env bash
set -euo pipefail
repo=/source/score_sde_pytorch_ours
bash -n "$repo/run_ab_replication_eval_queue.sh"
bash -n "$repo/run_ab_replication_eval.sh"
python - <<'PY'
import ast
from pathlib import Path
for name in ('run_ab_replication_eval.sh', 'run_ab_replication_eval_queue.sh'):
    text=Path('/source/score_sde_pytorch_ours',name).read_text()
    if name.endswith('eval.sh'):
        for block in text.split("<<'PY'\n")[1:]:
            ast.parse(block.split('\nPY\n',1)[0])
    else:
        script=text.split("-c '",1)[1].split("')",1)[0]
        ast.parse(script)
print('PASS: shell and all embedded Python syntax')
PY
task_tmp=$(mktemp -d)
mkdir -p "$task_tmp/bin" "$task_tmp/bundle"
cp "$repo/run_ab_replication_eval_queue.sh" "$repo/run_ab_replication_eval.sh" "$task_tmp/bundle/"
cat > "$task_tmp/bin/docker" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == inspect ]]; then
  if [[ "$*" == *'ab-replication-seed43-eval-seed0'* ]]; then exit 1; fi
  case "$3" in
    '{{.Id}}') echo mocked-training-id ;;
    '{{.Image}}') echo mocked-image-id ;;
    '{{.State.Status}}') echo running ;;
    *DeviceRequests*) echo 3 ;;
    *Mounts*) echo "$AB_REPLICATION_SERVER_BASE/ab_replication_seed43/workdirs" ;;
    *) exit 4 ;;
  esac
elif [[ "$1" == wait ]]; then
  read -r rc < "$AB_TEST_GATE"
  echo "$rc"
elif [[ "$1" == run ]]; then
  if [[ "$*" == *'--entrypoint python'* ]]; then
    cd "$AB_REPLICATION_SERVER_BASE/score_sde_pytorch"
    /usr/bin/python -c "${@: -1}"
  else
    printf '%s\n' "$@" > "$AB_TEST_GPU_RUN"
    exit "${AB_TEST_EVAL_RC:-0}"
  fi
else exit 5
fi
MOCK
cat > "$task_tmp/bin/git" <<'MOCK'
#!/usr/bin/env bash
printf 'mocked-launcher-commit\n'
MOCK
chmod +x "$task_tmp/bin/"*
export PATH="$task_tmp/bin:$PATH"
export TFDS_DATA_DIR=/cache/tfds
export TORCH_EXTENSIONS_DIR=/cache/extensions
export XDG_CACHE_HOME=/cache/xdg
new_case() {
  local id="$1"
  export AB_REPLICATION_SERVER_BASE="$task_tmp/$id"
  export AB_TEST_GATE="$task_tmp/$id.gate"
  export AB_TEST_GPU_RUN="$task_tmp/$id.gpu"
  mkfifo "$AB_TEST_GATE"
  local main="$AB_REPLICATION_SERVER_BASE/score_sde_pytorch"
  local output="$AB_REPLICATION_SERVER_BASE/ab_replication_seed43"
  mkdir -p "$main/assets/stats"
  cp "$repo/artifact_utils.py" "$main/artifact_utils.py"
  touch "$main/endpoint_denoise_eval.py"
  printf 'reference-fixture\n' > "$main/assets/stats/celeba_64_0c2d89311192_stats.npz"
  for name in A_vp_pvp B_vp_pfox; do
    mkdir -p "$output/workdirs/celeba_ab_replication_${name}_seed43_200k/checkpoints"
    printf '{}\n' > "$output/workdirs/celeba_ab_replication_${name}_seed43_200k/run_manifest.json"
    touch "$output/workdirs/celeba_ab_replication_${name}_seed43_200k/checkpoints/checkpoint_4.pth"
  done
  state="$output/logs/ab_eval_seed0_queue.state"
}
wait_state() {
  for i in $(seq 1 100); do
    if [[ -f "$state" && "$(cat "$state")" == "$1" ]]; then return 0; fi
    sleep 0.02
  done
  cat "$AB_REPLICATION_SERVER_BASE/ab_replication_seed43/logs/"*.log >&2
  return 1
}
new_case success
bash "$task_tmp/bundle/run_ab_replication_eval_queue.sh" 3
wait_state waiting
[[ ! -e "$AB_TEST_GPU_RUN" ]]
if bash "$task_tmp/bundle/run_ab_replication_eval_queue.sh" 3 > "$task_tmp/duplicate.out" 2>&1; then
  echo 'FAIL: duplicate queue accepted'; exit 1
fi
grep -q 'already active' "$task_tmp/duplicate.out"
printf '0\n' > "$AB_TEST_GATE"
wait_state completed
grep -Fxq -- 'device=3' "$AB_TEST_GPU_RUN"
grep -Fxq -- 'CUDA_VISIBLE_DEVICES=0' "$AB_TEST_GPU_RUN"
grep -Fxq -- 'TFDS_DATA_DIR=/cache/tfds' "$AB_TEST_GPU_RUN"
grep -Fxq -- 'TORCH_EXTENSIONS_DIR=/cache/extensions' "$AB_TEST_GPU_RUN"
grep -Fxq -- 'XDG_CACHE_HOME=/cache/xdg' "$AB_TEST_GPU_RUN"
grep -q 'checkpoints,readonly' "$AB_TEST_GPU_RUN"
grep -q 'training_workdirs,readonly' "$AB_TEST_GPU_RUN"
[[ "$(cat "$AB_REPLICATION_SERVER_BASE/ab_replication_seed43/logs/ab_eval_seed0_queue.exitcode")" == 0 ]]
echo 'PASS: queue waits without GPU, rejects duplicates, and starts eval only after exit0 with read-only checkpoints'
new_case failed_training
bash "$task_tmp/bundle/run_ab_replication_eval_queue.sh" 3
wait_state waiting
printf '2\n' > "$AB_TEST_GATE"
wait_state failed
[[ ! -e "$AB_TEST_GPU_RUN" && ! -d "$AB_REPLICATION_SERVER_BASE/ab_replication_seed43/eval_workdirs" ]]
echo 'PASS: failed training starts no eval and creates no eval workdirs'
new_case missing_checkpoint
rm "$AB_REPLICATION_SERVER_BASE/ab_replication_seed43/workdirs/celeba_ab_replication_B_vp_pfox_seed43_200k/checkpoints/checkpoint_4.pth"
bash "$task_tmp/bundle/run_ab_replication_eval_queue.sh" 3
wait_state waiting
printf '0\n' > "$AB_TEST_GATE"
wait_state failed
[[ ! -e "$AB_TEST_GPU_RUN" ]]
echo 'PASS: missing final checkpoint blocks eval'
new_case changed_launcher
bash "$task_tmp/bundle/run_ab_replication_eval_queue.sh" 3
wait_state waiting
printf '\n# changed after queue creation\n' >> "$task_tmp/bundle/run_ab_replication_eval.sh"
printf '0\n' > "$AB_TEST_GATE"
wait_state failed
[[ ! -e "$AB_TEST_GPU_RUN" ]]
echo 'PASS: changed eval launcher blocks eval'
new_case changed_reference
bash "$task_tmp/bundle/run_ab_replication_eval_queue.sh" 3
wait_state waiting
printf 'changed reference\n' >> "$AB_REPLICATION_SERVER_BASE/score_sde_pytorch/assets/stats/celeba_64_0c2d89311192_stats.npz"
printf '0\n' > "$AB_TEST_GATE"
wait_state failed
[[ ! -e "$AB_TEST_GPU_RUN" ]]
echo 'PASS: changed reference blocks eval before GPU launch'
echo 'Queue tests mock Docker wait/GPU execution; they do not run scientific evaluation.'
