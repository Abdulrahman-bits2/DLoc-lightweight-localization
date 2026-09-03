"""
plot_studyA.py
Study A figures:
  1. studyA_scatter  -- mean localization error vs parameters, no-skip vs skip
     families (the money figure: the skip curve sits below the no-skip curve across
     the parameter range -> spatial preservation helps independent of capacity),
     with multi-scale (ASPP) and bottleneck-sweep points marked.
  2. studyA_cdf      -- CDF of per-sample errors, skip vs no-skip at decoder_width 1.0.

Run aggregate_studyA.py first (produces studyA_summary.json). Reads per-sample errors
from results_studyA/errors_*.npy.

Usage:
  python plot_studyA.py [--results_dir ./results_studyA] [--outdir ../../Figures/generated]
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
from studyA_configs import config_tag

C_NOSKIP = "#c1121f"   # red
C_SKIP = "#0353a4"     # blue
C_MULTI = "#2a9d8f"    # teal
C_BOTTLE = "#8338ec"   # purple


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def scatter(summary, outdir):
    cfgs = summary["configs"]

    def fam(use_skips):
        pts = [c for c in cfgs if c["use_skips"] == use_skips and c["bottleneck"] == "none"
               and abs(c["width_mult"] - 1.0) < 1e-6]
        pts.sort(key=lambda c: c["params"])
        return pts

    ns, sk = fam(False), fam(True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    def plot_fam(pts, color, label, marker):
        if not pts:
            return
        x = [c["params"] / 1e6 for c in pts]
        y = [c["mean"] for c in pts]
        yerr = [c["mean_std"] for c in pts]
        ax.plot(x, y, "-", color=color, lw=1.8, zorder=2)
        ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, ms=8, capsize=3,
                    label=label, zorder=3, mec="white", mew=0.8)

    plot_fam(ns, C_NOSKIP, "no skips (plain decoder)", "o")
    plot_fam(sk, C_SKIP, "skips (U-Net)", "s")

    # multi-scale + bottleneck-sweep points
    for c in cfgs:
        if c["bottleneck"] == "aspp":
            ax.scatter(c["params"] / 1e6, c["mean"], marker="*", s=220, color=C_MULTI,
                       edgecolor="white", linewidth=0.8, zorder=4, label="skips + multi-scale (ASPP)")
        elif c["use_skips"] and abs(c["width_mult"] - 1.0) > 1e-6:
            lbl = "skips, bottleneck sweep (from scratch)" if c["width_mult"] < 1.0 else None
            ax.scatter(c["params"] / 1e6, c["mean"], marker="^", s=90, color=C_BOTTLE,
                       edgecolor="white", linewidth=0.8, zorder=4, label=lbl)

    # "equal params" annotation at the headline pair
    for eq in summary.get("equal_params", []):
        if abs(eq["params"] - 2.409e6) < 5e4:  # the decoder_width=1.0 skip point
            xp = eq["params"] / 1e6
            ax.annotate("", xy=(xp, eq["skip_mean"]), xytext=(xp, eq["noskip_interp_mean"]),
                        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
            ax.text(xp + 0.03, (eq["skip_mean"] + eq["noskip_interp_mean"]) / 2,
                    f'{eq["gap"]*100:.0f} cm\nat equal params', fontsize=8, color="gray", va="center")

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Mean localization error (m)")
    ax.set_title("Spatial preservation vs. parameter count (Fig. 10b)")
    ax.grid(True, alpha=0.3)
    # de-duplicate legend labels
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    _save(fig, outdir, "studyA_scatter")


def cdf(results_dir, outdir):
    def load(use_skips):
        rl = f"{config_tag(use_skips, 1.0)}_seed42"
        fp = os.path.join(results_dir, f"errors_{rl}.npy")
        return np.load(fp) if os.path.exists(fp) else None

    e_ns, e_sk = load(False), load(True)
    if e_ns is None or e_sk is None:
        print("  [cdf] pair errors not found — skipping CDF")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for e, color, label in [(e_ns, C_NOSKIP, "no skips"), (e_sk, C_SKIP, "skips (U-Net)")]:
        xs = np.sort(e)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, lw=2, label=f"{label} (mean {e.mean():.3f} m)")
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, np.percentile(np.concatenate([e_ns, e_sk]), 99))
    ax.set_title("Error CDF at decoder\\_width = 1.0")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, outdir, "studyA_cdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_studyA")
    ap.add_argument("--outdir", default=os.path.join(_CODE_ROOT, "..", "Figures", "generated"))
    args = ap.parse_args()

    summ_fp = os.path.join(args.results_dir, "studyA_summary.json")
    if not os.path.exists(summ_fp):
        print(f"Missing {summ_fp} — run aggregate_studyA.py first.")
        return
    with open(summ_fp) as f:
        summary = json.load(f)
    print("Study A figures:")
    scatter(summary, args.outdir)
    cdf(args.results_dir, args.outdir)


if __name__ == "__main__":
    main()
