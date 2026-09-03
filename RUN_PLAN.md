# RUN PLAN — Study A + Study C (Vast.ai H100, ~$48)

Execution plan, pre-flight checklist, and academic-value mapping for the two journal
experiments built 2026-09-03. Read alongside `STUDYA_RUNBOOK.md`, `STUDYC_RUNBOOK.md`, and the
specs in `docs/superpowers/specs/2026-09-03-study{A,C}-*.md`.

## 0. TL;DR
One H100 instance, both studies, **~$48 + ~$3 pilots**. Study A (~$34, 13 runs, no teacher) is the
high-confidence headline. Study C (~$14, 12 runs, needs the teacher) is the KD taxonomy. Do the
pilots + the teacher probe as go/no-go gates BEFORE the full ledgers.

## 1. Pre-flight checklist — verify ALL before launching (this is where money gets wasted)

- [ ] **Instance**: H100, torch env with CUDA. `python -c "import torch;print(torch.cuda.is_available())"` → True.
- [ ] **Python deps**: `torchvision` (Study A model + blur), `huggingface_hub` (`hf` CLI), `scipy`
      (paired Wilcoxon; optional — bootstrap fallback exists), `easydict` (teacher params).
- [ ] **Code present**: `code/` from HF `JustAGeek/dloc-code` (or scp local). Confirm the new files:
      `student_configurable.py`, `kd_losses.py`, `studyA_configs.py`, `studyC_configs.py`,
      `train_study{A,C}.py`, `run_study{A,C}.sh`, `precompute_teacher_bottleneck.py`.
- [ ] **Data (5 fig10b files) under `./data/`** — shared by BOTH studies, no Aug16:
      `dataset_jacobs_July28.mat`, `dataset_{fov,non_fov}_train_jacobs_July28_2.mat`,
      `dataset_{fov,non_fov}_test_jacobs_July28_2.mat`.
- [ ] **⚠️ TEACHER WEIGHTS (Study C only) — the loader needs an EXACT layout.**
      `load_networks("best", load_dir=D)` reads `D/{name}/best_net_{name}.pth` for
      name ∈ {encoder, decoder, offset_decoder}. The HF weights are flat/renamed, so remap them:
      ```bash
      D=./runs/2026-05-21-06-51-03
      mkdir -p $D/encoder $D/decoder $D/offset_decoder
      cp weights/baseline_encoder_best.pth        $D/encoder/best_net_encoder.pth
      cp weights/baseline_decoder_best.pth         $D/decoder/best_net_decoder.pth
      cp weights/baseline_offset_decoder_best.pth  $D/offset_decoder/best_net_offset_decoder.pth
      ```
      (Your memory notes this run-dir was logged "weights deleted" — the real weights live under
      HF `dloc-code/weights/`; the copy above puts them where the loader looks. Study A needs NONE
      of this.)
- [ ] **HF auth for sync**: `hf auth whoami` works (results are pushed under `studyA/` `studyC/`).
      ⚠️ Instance has **no volume** — unsynced results are lost on destroy. Additive upload only;
      never force-push (it would delete the weights on `dloc-code`).

## 2. Go/no-go gates (cheap, run BEFORE the full ledgers)

1. **Param table (Study A, seconds, no GPU/data):** `python code/evaluation/studyA_param_table.py`
   → confirms no-skip 2.27M / skip 2.41M and that the families overlap in params. If numbers differ
   wildly, stop.
2. **Teacher probe (Study C):** `python code/training/studyC_layer_probe.py`
   → if the teacher loads and prints `enc_model.output` + `decoder.output` shapes, the whole Study C
   teacher path works. **If this fails, Study C is blocked** — fix the weight layout (checklist above)
   before spending anything.
3. **Pilots (~$1-2 each):** one Study A config at 3 epochs, one Study C `gt_at_multi` at 3 epochs.
   Confirms the train loop, data, and (Study C) the pre-computed teacher tensors before the ~6-13 h ledgers.

## 3. Execution sequence

```bash
tmux new -s dloc
export STUDYA_HF_REPO=JustAGeek/dloc-code STUDYC_HF_REPO=JustAGeek/dloc-code

# Study A (~13 h) — no teacher
bash code/training/run_studyA.sh 2>&1 | tee studyA.log
python code/evaluation/aggregate_studyA.py
OMP_NUM_THREADS=1 python code/figures/plot_studyA.py

# Study C (~6 h) — auto pre-computes teacher outputs + bottleneck first
bash code/training/run_studyC.sh 2>&1 | tee studyC.log
python code/evaluation/aggregate_studyC.py
python code/figures/plot_studyC.py
```
Both ledgers are resume-safe (skip completed `summary_*.json`). Total wall-clock ~20 h on one
instance; or run the two studies on two instances in parallel to finish in ~13 h. **Sync + scp
`results_study{A,C}/` and `Figures/generated/study*` down before destroying the instance.**

## 4. Academic value — every run maps to a concrete contribution

**Study A → capacity/architecture story (high confidence, low redundancy).**
- *Journal:* Sec 5 "Capacity analysis" — the controlled capacity-vs-inductive-bias scatter +
  the "at equal parameters, skips win" result answers *is it capacity or spatial structure?* and
  **hardens the Mamba/2D-flatten argument** with a controlled experiment (not a hand-wave).
- *Thesis:* Results / model-architecture chapter — a new controlled-ablation subsection + the
  compression Pareto; upgrades the existing "DLoc is over-parameterized / skips help 9.5% / Mamba
  diverged" claims from observations to a controlled finding.
- *Artifacts:* `studyA_scatter`, `studyA_cdf`, `aggregate_studyA` LaTeX table.
- *Why not redundant with existing runs:* the shipped MobileNet/U-Net used different decoder
  designs; Study A retrains both from ONE parameterized model so the only difference is the ablated
  knob — required for a clean claim.

**Study C → distillation-knowledge taxonomy (completes professor's Option C; payoff conditional).**
- *Journal:* Sec 6 "Robustness analysis / distillation" — *which* teacher knowledge (output /
  attention / multi-scale / intermediate-feature) buys robustness; extends the existing "KD → 3.4x
  noise robustness" result into a mechanism study.
- *Thesis:* distillation chapter — deepens "distillation improves robustness" into "which knowledge
  type, and why."
- *Artifacts:* `studyC_robustness`, `aggregate_studyC` LaTeX table.
- *Honest caveat:* KD gives this task no in-dist gain, so the result may be muted/null. A "nothing
  beats output-KD" outcome is still a legitimate, publishable answer to Option C — reported as-is.

**Decision rule:** Study A is worth running regardless. If budget is tight, run Study A first; run
Study C only if the professor wants Option C answered even at the risk of a null result (he asked for
it, so this is likely yes) — its ~$14 is small.

## 5. After the runs (feeding the write-up)
- Journal (`NilesConf/journal_paper/`): drop the two tables + three figures into Sec 5 / Sec 6;
  update the 6-question framing answers with the controlled evidence.
- Thesis (`thesis/`): add the ablation subsections; note Final_Thesis.pdf then needs an Overleaf
  recompile (still on the June-20 build).
- Keep the honesty guards from the specs (published-vs-repro; LOSO-standalone vs zero-shot-distilled;
  best-by-mean; from-scratch bottleneck caveat).
