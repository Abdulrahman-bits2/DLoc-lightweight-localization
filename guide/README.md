# Concept Guide to the DLoc Compression Thesis

A standalone, concepts-first **teaching & defense guide** for the thesis *"Lightweight
Knowledge-Distilled Deep Learning for WiFi Indoor Localization."* It teaches every concept the thesis
rests on (from WiFi CSI to error CDFs), dissects each model architecture and **why it behaved the way
it did**, and ends with a defense Q&A bank. It complements the system-reference set in `../docs/`.

## What's here

```
guide/
├── main.tex                 # compile this
├── preamble.tex             # packages, macros, callout boxes (matches thesis style)
├── chapters/
│   ├── 00_how_to_use.tex    # orientation: thesis-in-one-page, the argument, concept→chapter map
│   ├── 01_foundations.tex   # I    localization, WiFi CSI, AoA/ToF, clock offset, image-translation framing
│   ├── 02_building_blocks.tex# II  conv, encoder–decoder, residual/skip, depthwise-separable, U-Net skips, Mamba
│   ├── 03_distillation.tex  # III  knowledge distillation → our spatial-label-smoothing version
│   ├── 04_evaluation.tex    # IV   error metric, median/P90/P99, CDF, quantization, corruptions, LOSO vs zero-shot
│   ├── 05_architectures.tex # V    MODEL-BY-MODEL deep dive (the core): arch → concepts → results → why
│   ├── 06_findings.tex      # VI   the six cross-cutting findings
│   ├── 07_defense_qa.tex    # VII  ~35 anticipated committee questions + answers
│   └── 08_glossary.tex      # glossary, symbols, key-numbers cheat-sheet, model scorecard
└── figures/                 # reused thesis figures + the U-Net TikZ; new diagrams are inline TikZ
```

## How to compile

**Overleaf (recommended):** upload the `guide/` folder, set `main.tex` as the main document,
compiler `pdfLaTeX`. No bibliography step is needed — the guide cites the thesis itself rather than a
`.bib`. Run twice so the table of contents and cross-references resolve.

**Locally:**

```bash
cd guide
latexmk -pdf main.tex        # or: pdflatex main && pdflatex main
```

Requires a full TeX distribution (TeX Live / MiKTeX): the only non-trivial package is `tcolorbox`
(for the callout boxes), which ships with both.

## Notes

- The guide is **self-contained**: the figures it reuses are copied into `figures/`, and every other
  diagram is inline TikZ, so it compiles without the rest of the repo. (As a convenience,
  `graphicspath` also falls back to `../thesis/figures/` and `../Figures/`.)
- Every number is sourced from the thesis tables; see the cheat-sheet in `08_glossary.tex`.
- Pairs with `../thesis/` (the thesis itself) and `../docs/` (the four system-reference documents).
