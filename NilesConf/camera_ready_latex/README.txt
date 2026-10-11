NILES 2026 — Camera-Ready LaTeX Project (self-contained)
========================================================
Paper: "Edge-Efficient WiFi Indoor Localization via Lightweight Deep Learning
        and Knowledge Distillation"
Output: main.pdf  (4 pages, A4, no Type-3 fonts — EDAS-compliant)

FILES
-----
  main.tex            - the paper (\documentclass[conference,a4paper]{IEEEtran})
  references.bib      - bibliography (22 cited refs)
  IEEEtran.cls        - IEEE conference class (bundled so it compiles anywhere)
  IEEEtran.bst        - IEEE bib style
  main.pdf            - the compiled camera-ready
  figures/
    unet_arch_body.tex   - Fig. 1, MobileNetV2-UNet architecture (TikZ, \input into main.tex)
    fig2_framework.pdf   - Fig. 2, distillation framework (vector PDF)
    cdf_fig10b.png       - Fig. 3, in-distribution error CDF (300-DPI raster; PNG is used
                           instead of the PDF so there are no Type-3 fonts)
    cdf_fig10b.pdf       - vector version of Fig. 3 (NOT used by main.tex; kept for reference)

Note: there are no .jpg figures — the paper's figures are TikZ (Fig. 1), a vector PDF
(Fig. 2), and a PNG (Fig. 3).

HOW TO COMPILE
--------------
Overleaf (easiest): create a new project, upload every file (keep the figures/ folder),
set the main document to main.tex, and compile. IEEEtran is also built into Overleaf, so
the bundled .cls/.bst are optional there.

Command line:
  Tectonic:  tectonic -X compile main.tex
  TeX Live:  pdflatex main  ->  bibtex main  ->  pdflatex main  ->  pdflatex main

Keep it at 4 pages, A4. After compiling, verify the page size is ~210x297 mm and that no
Type-3 fonts are embedded (both required by EDAS).
