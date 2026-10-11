# Defense Presentation — Lightweight WiFi Localization

A self-contained **HTML/CSS/JS** slide deck for the 20-minute graduation defense, plus the speaker
script and a per-section Q&A/concepts prep guide. No internet, no build step, no libraries — it runs
by opening one file.

## Run it

Double-click **`index.html`** (or open it in Chrome/Edge/Firefox). Press **F** for fullscreen.

| Key | Action |
|---|---|
| → / Space / PageDown | next slide |
| ← / PageUp | previous slide |
| **S** | toggle speaker notes (the script for the current slide) |
| **F** | fullscreen |
| **Home / End** | first / last slide |
| **type a number + Enter** | jump to that slide (e.g. `24` + Enter → backup slide) — handy in Q&A |
| click | right 75% = next, left 25% = back |

The current slide is also in the URL (`…/index.html#17`), so you can bookmark or deep-link any slide.

## Present from the notes

Every slide carries its speaker lines. Press **S** to open the notes panel and present from it, or use
`SCRIPT.md` for the full timed script. The deck has **29 main slides + 3 backup slides** (32 total).
The flow is: Overview → Part 1 (DLoc & method) → Part 2 (the four model architectures) → Part 3 (data,
experimental setup, then the Experiments & Results block, slides 21–26). Backups (30–32) are pulled up
to answer questions — see the slide map at the end of `PREP_GUIDE.md`.

## Export to PDF (for a handout or a backup copy)

Open in Chrome → **Print** → *Save as PDF* → **Landscape**, margins **None**, enable *Background
graphics*. The print stylesheet lays every slide out one-per-page.

## Files

```
presentation/
├── index.html        # the deck (open this)
├── css/style.css     # theme + layout + print styles
├── js/deck.js        # navigation engine (no dependencies)
├── figures/          # charts (PNG) used in the slides
│   ├── _gen_charts.py   # regenerates the themed charts from thesis data/numbers
│   ├── pareto.png · crosssession.png · efficiency.png · robustness.png · cdf_indist.png
│   └── heatmaps_unet_fig10b.png · heatmaps_unet_cross_env4.png
├── SCRIPT.md         # timed speaker script, 3-presenter split, delivery rules
├── PREP_GUIDE.md     # concepts + likely Q&A + deflection lines, per section
└── README.md
```

## Regenerating the charts

The data charts are built from the thesis numbers and the authentic per-sample error arrays
(`../Figures/figure_data/`). To rebuild them:

```bash
../.venv/Scripts/python.exe figures/_gen_charts.py
```

Requires `matplotlib` (already in the repo `.venv`). The qualitative heatmap PNGs are copied from
`../Figures/generated/`.

## Companion material

- `SCRIPT.md` — what to say, timed, with the suggested 3-presenter handoff points.
- `PREP_GUIDE.md` — for each section: concepts to master, likely questions with answers, and
  deflection lines for cross-section questions.
- Deeper conceptual backup lives in `../guide/` (the concept guide) and `../thesis/` (the thesis).
