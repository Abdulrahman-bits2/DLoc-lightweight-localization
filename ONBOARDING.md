# Onboarding — DLoc Lightweight Localization (team setup)

This guide gets a new collaborator from zero to running experiments on a shared
GPU workstation. The project is split across three places:

| Where | What it holds | How to get it |
|-------|---------------|---------------|
| **GitHub** (this repo) | all source code, data splits, final result JSONs, figures, paper/thesis/presentation sources | `git clone` |
| **HuggingFace `JustAGeek/dloc-wild-fig10b`** (dataset) | the WILD `.mat` dataset (~66 GB) + split indices | `hf download` |
| **HuggingFace `JustAGeek/dloc-code`** (model) | trained weights, per-epoch logs, result files, LOSO run dirs | `hf download` |

A clone alone does **not** include the dataset or the trained weights — pull those from
HuggingFace (steps 3–4). Nothing here requires the dataset to just read code or results.

---

## 1. Clone the repo
```bash
git clone https://github.com/Abdulrahman-bits2/DLoc-lightweight-localization.git
cd DLoc-lightweight-localization
```

## 2. Python environment
Trained on Python 3.13 / PyTorch 2.12 + CUDA 13.0 (H100), but any recent 2.x works.
```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r code/requirements.txt
```
If `torch`/`torchvision` mismatch on the workstation's CUDA, reinstall torchvision
against the same build (see the note at the bottom of `code/requirements.txt`).
The **Mamba** student additionally needs `mamba-ssm` + `causal-conv1d` (CUDA-only);
not required for DLoc / MobileNetV2 / TinyCNN / U-Net.

## 3. Download the dataset (~66 GB) from HuggingFace
```bash
pip install huggingface_hub           # if not already from step 2
hf download JustAGeek/dloc-wild-fig10b --repo-type dataset \
    --local-dir code/data
```
The data splits (`data_split_idx/`) are already in this repo — no download needed.

## 4. Download trained weights / logs / results
```bash
hf download JustAGeek/dloc-code --repo-type model --local-dir ./dloc-code-hub
# weights/  logs/  results/  multiseed/  crosssession_official/
```
Point the eval scripts at `./dloc-code-hub/weights/...`, or copy into `code/weights/`.

## 5. (Only if re-running the raw data-prep notebooks)
The WILD dataset download needs the UCSD SharePoint password, read from an env var
(not stored in the repo). Request it from the team, then:
```bash
export WILD_PASSWORD='...'        # Windows PowerShell:  $env:WILD_PASSWORD='...'
```
You normally do **not** need this — the processed `.mat` dataset is already on
HuggingFace (step 3).

---

## Where things live
- `code/core/` — DLoc engine (Generators, modelADT, joint_model, trainer, data_loader, utils)
- `code/models/` — student_mobilenet, student_tinycnn, student_unet, mamba_model;
  `student_configurable.py` + `kd_losses.py` (Study A / Study C)
- `code/training/` — train/eval entry points (see runbooks below)
- `code/evaluation/` — test_robustness, eval_cross_session, benchmark_models, aggregators
- `code/configs/` — `params.py` + `params_storage/` + `studyA_configs.py` / `studyC_configs.py`
- `code/results_crosssession/` — final leave-one-session-out result JSONs + summary
- `Figures/` — generated figures + `figure_data/` (per-sample arrays)
- `reports/` — LaTeX result reports (phase2_results.tex is the canonical numbers table)
- `NilesConf/`, `thesis/`, `paper/`, `presentation/`, `guide*/` — paper / thesis / talk sources

## Running experiments
See the runbooks at the repo root:
- `REPRODUCIBILITY.md` — end-to-end reproduction (GitHub code + HF dataset + HF weights)
- `RUN_PLAN.md` — the overall experiment plan
- `STUDYA_RUNBOOK.md` — Study A (spatial preservation vs. parameter count) — BUILT, not yet run
- `STUDYC_RUNBOOK.md` — Study C (distillation-mechanism study) — BUILT, not yet run

Switch configs via the `CONFIG`/`MODE` blocks at the top of each `train_*.py`
(e.g. `train_distillation.py`, `train_unet_advanced.py`, `train_unet_crosssession.py`).

## Collaboration rules
- **Branch + PR via GitHub.** Commit and push to `github` as normal.
- ⚠️ **Never force-push to the HuggingFace remote (`origin`).** It holds the canonical
  weights and run artifacts; a force-push from a local clone can destroy them. Treat
  HuggingFace as download-only unless you deliberately mean to publish new weights.
- Large artifacts (datasets, `.pth`, `.pt`, logs, run dirs) are git-ignored on purpose —
  they live on HuggingFace, not in git.
