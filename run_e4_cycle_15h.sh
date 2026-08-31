#!/usr/bin/env bash
# ===========================================================================
# E4 cycle -- single queued job, budgeted for ~15-16 h on one H200 (GPU 0).
# ===========================================================================
# Question: is the CelebA FOX gain from the log-SNR SHAPE, the overall-scale
# path r(t), or neither?  This is the experiment that decides whether the
# paper has a mechanism (option A) or needs a pivot.
#
# Phase 1 (core, ~7-13 h)   -- everything you need for the decomposition:
#   1. train normalized-FOX to 100k                     ~5 h
#   2. EDM CelebA eval, 50k samples (baseline gate)     ~0.5 h
#   3. eval all 3 arms @ 5k samples (fast signal)       ~1 h
#      => read <workdir>/eval_e4_5k/  now.  If normalized-FOX clearly beats VP
#         (or clearly matches real-FOX), Phase 2 is worth it.
#
# Phase 2 (publication, ~8-12 h)  -- 50k-sample eval of the same 3 arms.
#   Kept as a separate trailing step so you can stop after Phase 1, inspect,
#   and only spend the big eval if the 5k signal is real.  Safe to Ctrl-C
#   between arms; each arm's numbers are written when it finishes.
#
# Total if run straight through: ~15-25 h.  Set RUN_PHASE2=0 to stop after
# Phase 1.  Everything is resumable (rerun this script).
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/run_e4_common.sh"

RUN_PHASE2="${RUN_PHASE2:-1}"
LOG_DIR="${HERE}/run_e4_logs"
mkdir -p "$LOG_DIR"
ts() { date +%Y-%m-%dT%H:%M:%S; }

echo "[$(ts)] E4 cycle start   GPU=${CUDA_VISIBLE_DEVICES:-pinned by docker --gpus}   phase2=${RUN_PHASE2}"
echo "[$(ts)] arms: VP=${VP_WORKDIR}  realFOX=${REALFOX_WORKDIR}  normFOX=${NORMFOX_WORKDIR}"

# --- Phase 1 --------------------------------------------------------------
echo "[$(ts)] 1/4  train normalized-FOX"
bash "${HERE}/run_e4_normalized_fox_train.sh" 2>&1 | tee "${LOG_DIR}/1_train_normfox.log"

echo "[$(ts)] 2/4  EDM CelebA eval (50k)"
bash "${HERE}/run_e4_edm_celeba_eval.sh" 50000 2>&1 | tee "${LOG_DIR}/2_edm_eval.log"

echo "[$(ts)] 3/4  E4 eval @ 5k (all 3 arms)"
bash "${HERE}/run_e4_eval.sh" 5000 5k 2>&1 | tee "${LOG_DIR}/3_eval_5k.log"

echo "[$(ts)] ---- Phase 1 done.  Inspect eval_e4_5k/ before committing to Phase 2. ----"
grep -hE "ckpt-.*(FID|inception)" "${LOG_DIR}"/*.log || true

# --- Phase 2 -------------------------------------------------------------
if [[ "${RUN_PHASE2}" == "1" ]]; then
  echo "[$(ts)] 4/4  E4 eval @ 50k (all 3 arms) -- ~8-12 h, Ctrl-C-safe between arms"
  bash "${HERE}/run_e4_eval.sh" 50000 50k 2>&1 | tee "${LOG_DIR}/4_eval_50k.log"
else
  echo "[$(ts)] RUN_PHASE2=0 -> stopping after Phase 1.  Later:  ${HERE}/run_e4_eval.sh 50000 50k"
fi

echo "[$(ts)] E4 cycle done"
grep -hE "ckpt-.*(FID|inception)" "${LOG_DIR}"/*.log || true
