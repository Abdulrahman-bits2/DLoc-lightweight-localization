# Study C Runbook — Distillation-Mechanism Study (Vast.ai H100)

Which *type* of teacher knowledge helps a compressed CSI-localization student.
**12 runs** (lean seeds: 3 on the anchors `gt_only` / `gt_kd_out` / `gt_at_multi`, 1 on the
screening variants), MobileNet, 50 epochs, **fig10b data only** (same 5 `.mat` files as Study A —
no Aug16). **Offline** — teacher signals are pre-computed once, so every variant trains at
student speed. ~6 instance-hours, **~$14 at $2.40/hr**.
Design: `docs/superpowers/specs/2026-09-03-studyC-distillation-mechanism-design.md`.

## 0. Files
- `code/models/kd_losses.py`, `code/configs/studyC_configs.py`,
  `code/training/{studyC_layer_probe.py, precompute_teacher.py, precompute_teacher_bottleneck.py,
  train_studyC.py, run_studyC.sh}`, `code/evaluation/{eval_studyC.py, aggregate_studyC.py}`.
- Needs the **DLoc teacher weights** (`runs/2026-05-21-06-51-03/`, on HF `dloc-code`) — used ONCE by
  the pre-compute step; training itself never loads the teacher.

## 1. Setup (same data as Study A; skip the download if already present)
```bash
pip install huggingface_hub
hf download JustAGeek/dloc-code --local-dir ./dloc-code --repo-type model   # code + teacher weights
cd dloc-code
# fig10b data (5 files) — see STUDYA_RUNBOOK.md step 1 if not already under ./data
```

## 2. Pre-flight — confirm the teacher/student feature mapping (the one judgment call)
```bash
python code/training/studyC_layer_probe.py --teacher_dir ./runs/2026-05-21-06-51-03
```
Expect student `deep` ~[1,320,6,12], `mid` ~[1,96,11,23]; teacher `enc_model.output` (bottleneck)
and `decoder.output` (heatmap). The FitNets regressor auto-adapts to the probed bottleneck channel
count. If `enc_model.output` is NOT the encoder feature on this checkpoint, tell me and I'll rewire it.

## 3. Run the ledger (~6 h) — under tmux (no volume!)
```bash
export STUDYC_HF_REPO=JustAGeek/dloc-code     # additive sync under studyC/ after each run
tmux new -s studyC 'bash code/training/run_studyC.sh 2>&1 | tee studyC.log'
```
`run_studyC.sh` first pre-computes the teacher signals **once** (~2 min each) if missing —
`teacher_train_outputs.pt` (heatmaps) + `teacher_train_bottleneck.pt` (encoder feature, ~4.5 GB) —
then trains the 12 runs and finally calls `eval_studyC.py` (corruption suite on every best
checkpoint; corruptions seeded identically across variants). Resume-safe (skips existing summaries).

## 4. Aggregate
```bash
python code/evaluation/aggregate_studyC.py     # variant x corruption table (mean+/-std) +
                                               # winner per corruption -> results_studyC/studyC_summary.json
python code/figures/plot_studyC.py             # robustness bar chart -> Figures/generated/
```

## 5. Before destroy
```bash
hf upload JustAGeek/dloc-code ./results_studyC studyC/results_studyC --repo-type model
hf upload JustAGeek/dloc-code ./runs_studyC    studyC/runs_studyC    --repo-type model
# also scp results_studyC/ down locally.
```

## Variants + seeds (mirrors code/configs/studyC_configs.py)
| variant | loss | seeds | professor item |
|---|---|---|---|
| `gt_only` | GT MSE only | **3** (anchor) | GT only |
| `kd_out_only` | teacher-output MSE only | 1 (screen) | KD only |
| `gt_kd_out` | 0.7·GT + output-KD | **3** (anchor) | GT+KD / output-heatmap KD |
| `gt_at_single` | GT + attention transfer (deepest) | 1 (screen) | attention/spatial KD |
| `gt_at_multi` | GT + attention transfer (multi-stage) | **3** (anchor) | multi-scale KD |
| `gt_fitnet` | GT + FitNets intermediate-feature | 1 (screen) | intermediate-feature KD |

## Notes / honesty guards
- Teacher signals **pre-computed once**; training is fully offline (student-speed). If a screening
  variant unexpectedly wins, add its 2 extra seeds afterward (~$3) before claiming it.
- KD-term weight is **auto-balanced on the first batch** (`beta·L_aux = kd_ratio·gt_weight·L_GT`),
  so AT / FitNets / output compare on equal footing — not hand-tuned.
- Best checkpoint by **in-dist mean**; corruption metrics computed on that checkpoint (no selection
  on the corruption test → no leakage).
- **Expected:** in-dist ~flat across variants (KD doesn't help in-dist here); the finding is in
  robustness. A "nothing beats output-KD" result is legitimate and reported honestly.
