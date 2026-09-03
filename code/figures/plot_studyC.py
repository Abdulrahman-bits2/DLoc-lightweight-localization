"""
plot_studyC.py
Study C figure: robustness by distillation-knowledge type. Grouped bars of mean error
under noise / random-AP-dropout for each variant (in-dist is ~flat, so the story is here),
with 3-seed std error bars. Run aggregate_studyC.py first.

Usage:
  python plot_studyC.py [--results_dir ./results_studyC] [--outdir ../../Figures/generated]
"""

# --- path bootstrap ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# corruptions to show (where KD demonstrably matters) + pretty labels
SHOW = [("noise_0.1", "noise σ=0.1"), ("noise_0.2", "noise σ=0.2"), ("rand_1ap", "random AP drop")]
COLORS = ["#0353a4", "#2a9d8f", "#e07a1f"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_studyC")
    ap.add_argument("--outdir", default=os.path.join(_CODE_ROOT, "..", "Figures", "generated"))
    args = ap.parse_args()

    summ_fp = os.path.join(args.results_dir, "studyC_summary.json")
    if not os.path.exists(summ_fp):
        print(f"Missing {summ_fp} — run aggregate_studyC.py first.")
        return
    rows = json.load(open(summ_fp))["rows"]
    rows = [r for r in rows if "noise_0.1" in r]
    if not rows:
        print("No corruption data in summary — run eval_studyC.py first.")
        return
    rows.sort(key=lambda r: r["noise_0.1"])          # most robust first
    variants = [r["variant"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(variants))
    w = 0.8 / len(SHOW)
    for j, (key, lbl) in enumerate(SHOW):
        y = [r.get(key, np.nan) for r in rows]
        yerr = [r.get(key + "_std", 0.0) for r in rows]
        ax.bar(x + j * w, y, w, yerr=yerr, capsize=2, color=COLORS[j], label=lbl)

    ax.set_xticks(x + w * (len(SHOW) - 1) / 2)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
    ax.set_ylabel("Mean localization error (m)")
    ax.set_title("Robustness by distillation-knowledge type (MobileNet, Fig. 10b)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"studyC_robustness.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote studyC_robustness.pdf / .png to {args.outdir}")


if __name__ == "__main__":
    main()
