# NILES 2026 camera-ready — Edge-Efficient WiFi Indoor Localization

Accepted (minor revision) at **NILES 2026**. This folder holds the **camera-ready** paper,
rebuilt from the accepted EDAS version (`1571324247 paper.pdf`, 6 pp) and condensed to
**exactly 4 pages** with both reviewers' revisions merged.

Title: *Edge-Efficient WiFi Indoor Localization via Lightweight Deep Learning and Knowledge
Distillation.*

## Deliverables
- `main.tex` — the paper. Anonymity toggle at the top: `\anonymousfalse` = camera-ready
  (authors + E-JUST block shown, **current setting**); `\anonymoustrue` = anonymous review copy
  (no authors, generic future-work wording).
- `main_CAMERAREADY.pdf` / `main.pdf` — the compiled camera-ready, **4 pages** (page count
  verified). Both are the same current build (rebuilt 2026-09-10).
- `reviewer_response.pdf` (+ `.tex`) — Response-to-Reviewers document (5 pp).
- `1571324247 paper.pdf` — the accepted 6-page EDAS version; source of truth for content.
- `references.bib` — **22 cited** references (incl. self-citations `[3]–[5]`). Note: an uncited
  `hendrycks2019robustness` entry remains in the file; IEEEtran only prints cited refs, so it
  does not appear in the PDF.
- `figures/` — `cdf_fig10b.pdf`, `unet_arch_body.tex` (Fig. 1, TikZ, `\input` into a `figure`),
  `fig2_framework.pdf` (Fig. 2).

## Reviewer revisions merged (all four asks)
- **Rev A**: English/format polish; distillation methodology + loss equations expanded (§III-D).
- **Rev B**: TinyCNN + Mamba added to Table II (`tab:indist`) as a labelled under-capacity
  sub-block; §V-C explains that a fold-specific DLoc teacher would only *lower* LOSO error, so the
  standalone U-Net is a conservative baseline.

## Compile (Overleaf or Tectonic)
pdfLaTeX → BibTeX → pdfLaTeX ×2. No packages beyond IEEEtran/booktabs/tikz/subcaption.
⚠️ If `main.pdf` looks stale after a compile, close the PDF viewer first — an open viewer locks
the file (tectonic then fails with "os error 32" and the old PDF stays on disk). The freshest
compile also lands in `build/main.pdf`.

## Edits 2026-09-14 (Prof. Ehab, via WhatsApp) — compiled + verified **4 pages** (local Tectonic)
- Ref [17] (visual Mamba) corrected: now *Zhang et al., "A survey on visual Mamba," Applied
  Sciences 14(13):5683, 2024* (added Ziyang Wang; arXiv → journal). Fixed in both
  `submission/references.bib` and `journal_paper/references.bib`.
- Author order changed to **Abdulrahman, Ahmed Emadeldin, Ehab El-Shazly, Fadwa** (Ehab third)
  in `main.tex` and `reviewer_response.tex`; emails reordered to match.
- Fig. 1 (U-Net) enlarged for legibility by folding it into a **two-row U-shape** (`figures/unet_arch_body.tex`):
  encoder left→right on top, bottleneck turns down, decoder right→left on bottom, skip
  connections vertical. Kept single-column. A two-column `figure*` was tried first but pushed 8
  refs onto a 5th page (a full-width float costs ~2×+packing, and only ~96 pt was free), so the
  taller single-column redraw is the correct fit. Now fills the page-4 whitespace (slack 96→46 pt).
  `main.pdf` / `main_CAMERAREADY.pdf` rebuilt (4 pp, from `build/main.pdf`).

## Still open — author actions before submitting to EDAS/IEEE
- [ ] Confirm the NILES camera-ready deadline (CFP said **Sep 7, 2026**; likely extended since
      revisions were due ~Sep 9) and upload the PDF.
- [ ] Verify the two constructed co-author emails: `ahmed.emadeldin@` / `fadwa.khalil@ejust.edu.eg`.
- [x] Surname spelling resolved: author block + reviewer response now use "Ehab **El-Shazly**"
      (matches refs `[3]–[5]`); only the email localpart is necessarily `ehab.elshazly`.
- [ ] Confirm 4 pages is acceptable (the accepted paper was 6 pp) — or decide to restore the full
      "School of ECCE / Dept. of CSE" affiliation and/or content that was trimmed for the 4-page fit.

## Content notes (audited, unchanged)
- Honest comparison: U-Net 0.640 m **ties DLoc's published 0.64 m** and improves 9.5% over **our
  reproduction** (0.707 m); Table II shows both DLoc rows. Efficiency: 6.8× fewer params, 56× fewer
  FLOPs, ~10× lower CPU time.
- Honesty guard: the LOSO-vs-DLoc comparison (Table IV, `tab:loso`) uses the *standalone* U-Net;
  the "distilled is best" claim belongs to the zero-shot protocol (Table V, `tab:zeroshot`) only.
