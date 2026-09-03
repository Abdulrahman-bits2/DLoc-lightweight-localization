#!/usr/bin/env bash
# run_studyC.sh — the Study C ledger (distillation-mechanism study) + robustness eval.
# Mirrors code/configs/studyC_configs.py::VARIANTS (lean seeds: 3 on the anchors
# gt_only / gt_kd_out / gt_at_multi, 1 on the screening variants) => 12 runs.
#
# Pre-flight (confirm teacher/student feature shapes):
#   python code/training/studyC_layer_probe.py --teacher_dir ./runs/2026-05-21-06-51-03
#
# Run under tmux (no volume on the instance):
#   tmux new -s studyC 'bash code/training/run_studyC.sh 2>&1 | tee studyC.log'
set -u

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TRAIN="$HERE/train_studyC.py"
EVAL="$ROOT/evaluation/eval_studyC.py"
RESULTS_DIR="${RESULTS_DIR:-./results_studyC}"
OUT_DIR="${OUT_DIR:-./runs_studyC}"
EPOCHS="${EPOCHS:-50}"
TEACHER_DIR="${TEACHER_DIR:-./runs/2026-05-21-06-51-03}"
DATA_DIR="${DATA_DIR:-./data}"
HF_REPO="${STUDYC_HF_REPO:-}"

# variant + its seeds (mirrors studyC_configs.py)
RUNS=(
  "gt_only 42" "gt_only 123" "gt_only 777"
  "kd_out_only 42"
  "gt_kd_out 42" "gt_kd_out 123" "gt_kd_out 777"
  "gt_at_single 42"
  "gt_at_multi 42" "gt_at_multi 123" "gt_at_multi 777"
  "gt_fitnet 42"
)

sync_hf() {
  [ -z "$HF_REPO" ] && return 0
  hf upload "$HF_REPO" "$RESULTS_DIR" studyC/results_studyC --repo-type model >/dev/null 2>&1 \
    && echo "   results synced" || echo "   !! results sync failed"
  hf upload "$HF_REPO" "$OUT_DIR" studyC/runs_studyC --repo-type model >/dev/null 2>&1 \
    && echo "   weights synced" || echo "   !! weights sync failed"
}

# --- one-time teacher pre-compute (offline KD; ~2 min) ---
if [ ! -f "$DATA_DIR/teacher_train_outputs.pt" ]; then
  echo "=== precompute teacher output heatmaps ==="
  "$PY" "$HERE/precompute_teacher.py"
fi
if [ ! -f "$DATA_DIR/teacher_train_bottleneck.pt" ]; then
  echo "=== precompute teacher encoder bottleneck ==="
  "$PY" "$HERE/precompute_teacher_bottleneck.py" --teacher_dir "$TEACHER_DIR" --data_dir "$DATA_DIR"
fi

echo "=== Study C: ${#RUNS[@]} runs, ${EPOCHS} epochs each ==="
[ -z "$HF_REPO" ] && echo "!! STUDYC_HF_REPO unset — results stay LOCAL ONLY (sync before destroy!)"

i=0
for r in "${RUNS[@]}"; do
  i=$((i + 1))
  set -- $r
  echo ""
  echo "--- [$i/${#RUNS[@]}] variant=$1 seed=$2 ---"
  "$PY" "$TRAIN" --variant "$1" --seed "$2" --epochs "$EPOCHS" \
    --data_dir "$DATA_DIR" --results_dir "$RESULTS_DIR" --out_dir "$OUT_DIR" \
    || echo "!! run $i ($1 seed $2) failed — continuing"
  sync_hf
done

echo ""
echo "=== Training done. Robustness eval on every best checkpoint ==="
"$PY" "$EVAL" --results_dir "$RESULTS_DIR" --data_dir "$DATA_DIR"
sync_hf

echo ""
echo "=== Aggregate: $PY code/evaluation/aggregate_studyC.py ==="
