#!/usr/bin/env bash
# run_studyA.sh — the 13-run Study A ledger (spatial preservation vs parameter count).
# Mirrors code/configs/studyA_configs.py::LEDGER — keep the two in sync.
#
# Run under tmux (the H100 instance has NO volume — a dropped ssh loses the shell):
#   tmux new -s studyA 'bash code/training/run_studyA.sh 2>&1 | tee studyA.log'
#
# Each run is resume-safe: train_studyA.py skips a config+seed whose summary already
# exists. After every run we sync results+weights to HuggingFace (set STUDYA_HF_REPO),
# because an instance without a volume loses everything on destroy.
set -u

# Cap CPU threads: utils.localization_error is a per-sample Python/numpy loop; on a
# many-core box (the H100 had 192 cores) an uncapped numpy thread-pool spin-waits and
# stalls the eval. (See the figure-pipeline gotcha.)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/train_studyA.py"
RESULTS_DIR="${RESULTS_DIR:-./results_studyA}"
OUT_DIR="${OUT_DIR:-./runs_studyA}"
EPOCHS="${EPOCHS:-120}"
# STUDYA_HF_REPO: e.g. JustAGeek/dloc-code  (results pushed under studyA/). Leave empty to skip.
HF_REPO="${STUDYA_HF_REPO:-}"

# use_skips  decoder_width  width_mult  bottleneck  seed
RUNS=(
  "0 0.5 1.0 none 42"
  "0 1.0 1.0 none 42"
  "0 1.0 1.0 none 123"
  "0 1.0 1.0 none 777"
  "0 2.0 1.0 none 42"
  "1 0.5 1.0 none 42"
  "1 1.0 1.0 none 42"
  "1 1.0 1.0 none 123"
  "1 1.0 1.0 none 777"
  "1 2.0 1.0 none 42"
  "1 1.0 1.0 aspp 42"
  "1 1.0 0.5 none 42"
  "1 1.0 1.4 none 42"
)

sync_hf() {
  [ -z "$HF_REPO" ] && return 0
  echo ">> syncing results to HF: $HF_REPO (studyA/)"
  # Additive upload only — NEVER touches existing weights on the repo.
  hf upload "$HF_REPO" "$RESULTS_DIR" studyA/results_studyA --repo-type model >/dev/null 2>&1 \
    && echo "   results synced" || echo "   !! results sync failed (check: hf auth whoami)"
  hf upload "$HF_REPO" "$OUT_DIR" studyA/runs_studyA --repo-type model >/dev/null 2>&1 \
    && echo "   weights synced" || echo "   !! weights sync failed"
}

echo "=== Study A: ${#RUNS[@]} runs, ${EPOCHS} epochs each ==="
[ -z "$HF_REPO" ] && echo "!! STUDYA_HF_REPO unset — results stay LOCAL ONLY (sync before destroy!)"

i=0
for r in "${RUNS[@]}"; do
  i=$((i + 1))
  set -- $r
  echo ""
  echo "--- [$i/${#RUNS[@]}] use_skips=$1 dw=$2 wm=$3 bn=$4 seed=$5 ---"
  "$PY" "$TRAIN" \
    --use_skips "$1" --decoder_width "$2" --width_mult "$3" --bottleneck "$4" --seed "$5" \
    --epochs "$EPOCHS" --results_dir "$RESULTS_DIR" --out_dir "$OUT_DIR"
  status=$?
  if [ $status -ne 0 ]; then
    echo "!! run $i failed (exit $status) — continuing to next"
  fi
  sync_hf
done

echo ""
echo "=== Study A ledger complete. Aggregate with: $PY code/evaluation/aggregate_studyA.py ==="
sync_hf
