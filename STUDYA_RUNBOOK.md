# Study A Runbook — Spatial Preservation vs. Parameter Count (Vast.ai H100)

Controlled ablation for the journal paper. 13 runs, 120 epochs each, ~14 instance-hours,
**~$34 at $2.40/hr**. Design: `docs/superpowers/specs/2026-09-03-studyA-spatial-preservation-design.md`.

## 0. What runs where
- Code: `code/models/student_configurable.py`, `code/configs/studyA_configs.py`,
  `code/training/train_studyA.py`, `code/training/run_studyA.sh`,
  `code/evaluation/{studyA_param_table.py, aggregate_studyA.py}`, `code/figures/plot_studyA.py`.
- Data: **fig10b in-distribution only** (no Aug16 / cross-session download). 5 `.mat` files.

## 1. Provision + setup
```bash
# H100 instance (torch env). Then:
pip install huggingface_hub scipy            # scipy = paired Wilcoxon (optional; bootstrap fallback exists)
hf download JustAGeek/dloc-code --local-dir ./dloc-code --repo-type model   # or scp your local code/
cd dloc-code

# fig10b data subset (~34 GB) — ONLY these 5 files:
hf download JustAGeek/dloc-wild-fig10b --repo-type dataset --local-dir ./data_dl \
  --include "data/dataset_jacobs_July28.mat" \
            "data/dataset_non_fov_train_jacobs_July28_2.mat" \
            "data/dataset_fov_train_jacobs_July28_2.mat" \
            "data/dataset_fov_test_jacobs_July28_2.mat" \
            "data/dataset_non_fov_test_jacobs_July28_2.mat"
mkdir -p data && mv ./data_dl/data/*.mat ./data/
```

## 2. Pre-flight (seconds, no GPU) — confirm the design holds on this box
```bash
python code/configs/studyA_configs.py        # prints the 13-run ledger
python code/evaluation/studyA_param_table.py # params/FLOPs; confirms no-skip family brackets skip curve
```
Expect: no-skip dw1.0 = 2.27M, skip dw1.0 = 2.41M (matches shipped nets), 2/3 skip points bracketed.

## 3. Run the ledger (~13 h) — ALWAYS under tmux (no volume on the instance!)
```bash
export STUDYA_HF_REPO=JustAGeek/dloc-code    # results/weights auto-synced under studyA/ after each run
tmux new -s studyA 'bash code/training/run_studyA.sh 2>&1 | tee studyA.log'
# detach: Ctrl-b d   |   reattach: tmux attach -t studyA
```
- Resume-safe: a config+seed whose `results_studyA/summary_*.json` exists is skipped. Re-run the
  same command after any interruption.
- ⚠️ **No volume**: if `STUDYA_HF_REPO` is unset, results live only on the instance and are lost on
  destroy. The sync is **additive** (uploads under `studyA/`) and never touches existing weights.

## 4. Aggregate + figures
```bash
python code/evaluation/aggregate_studyA.py   # table + 3-seed mean+/-std + paired Wilcoxon + "equal-params" gap
                                             # -> results_studyA/studyA_summary.json + LaTeX table
OMP_NUM_THREADS=1 python code/figures/plot_studyA.py   # -> Figures/generated/studyA_scatter.*, studyA_cdf.*
```

## 5. Before you destroy the instance
```bash
hf upload JustAGeek/dloc-code ./results_studyA studyA/results_studyA --repo-type model
hf upload JustAGeek/dloc-code ./runs_studyA    studyA/runs_studyA    --repo-type model
# pull results_studyA/ + Figures/generated/studyA_* down locally too (scp) before stopping.
```

## Ledger (mirrors code/configs/studyA_configs.py)
| # | use_skips | decoder_width | width_mult | bottleneck | seeds |
|---|---|---|---|---|---|
| 1–3 | 0 | 0.5 / 1.0 / 2.0 | 1.0 | none | 42 (dw1.0: +123,777) |
| 4–6 | 1 | 0.5 / 1.0 / 2.0 | 1.0 | none | 42 (dw1.0: +123,777) |
| 7 | 1 | 1.0 | 1.0 | aspp | 42 |
| 8–9 | 1 | 1.0 | 0.5 / 1.4 | none | 42 |

13 runs total. Headline = the decoder_width=1.0 pair (3 seeds each) + the two capacity curves.

## Notes / honesty guards
- Best checkpoint selected by **mean** error (primary metric; median is pixel-quantized).
- `width_mult != 1.0` points are **from-scratch** (no ImageNet pretraining) — labelled as such;
  never merged into the pretrained decoder-width curve.
- Single-seed points carry trend/CDF only; the significance claim rests on the 3-seed dw1.0 pair.
