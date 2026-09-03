"""
aggregate_studyC.py
Collect Study C runs -> variant x corruption robustness table + "which knowledge wins".

Reads results_studyC/summary_*.json (in-dist) and corrupt_*.json (robustness), groups by
variant, and reports mean +/- std over seeds for in-distribution and each corruption, plus
the noise/dropout degradation and the winning variant per corruption.

Usage:
  python aggregate_studyC.py [--results_dir ./results_studyC]
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
import glob
import json
import argparse
import numpy as np
from studyC_configs import VARIANTS, CORRUPTIONS

# summary columns for the printed table
COLS = ["clean", "noise_0.1", "noise_0.2", "drop_avg", "rand_1ap", "blur_k5"]


def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_studyC")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.results_dir, "studyC_summary.json")

    # index runs by variant
    by_variant = {v["name"]: [] for v in VARIANTS}
    for fp in sorted(glob.glob(os.path.join(args.results_dir, "summary_*.json"))):
        with open(fp) as f:
            s = json.load(f)
        rl = s["run_label"]
        cfp = os.path.join(args.results_dir, f"corrupt_{rl}.json")
        s["_corrupt"] = json.load(open(cfp)) if os.path.exists(cfp) else None
        if s["variant"] in by_variant:
            by_variant[s["variant"]].append(s)

    def corruption_mean(run, key):
        c = run.get("_corrupt")
        if not c:
            return None
        if key == "drop_avg":
            vals = [c[f"drop_ap{k}"]["mean"] for k in range(4) if f"drop_ap{k}" in c]
            return float(np.mean(vals)) if vals else None
        return c.get(key, {}).get("mean")

    rows = []
    for v in VARIANTS:
        runs = by_variant.get(v["name"], [])
        if not runs:
            continue
        row = dict(variant=v["name"], method=v["method"], n_seeds=len(runs))
        indist = [r["best"]["mean"] for r in runs]
        row["indist_mean"], row["indist_std"] = agg(indist)
        for col in COLS:
            vals = [corruption_mean(r, col) for r in runs]
            vals = [x for x in vals if x is not None]
            if vals:
                m, sd = agg(vals)
                row[col], row[col + "_std"] = m, sd
        rows.append(row)

    if not rows:
        print(f"No runs found in {args.results_dir}. Run the ledger + eval_studyC first.")
        return

    print(f"Loaded {sum(r['n_seeds'] for r in rows)} runs across {len(rows)} variants.\n")
    hdr = f'{"variant":<14}{"seeds":>6}{"in-dist":>10}' + "".join(f"{c:>11}" for c in COLS)
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        line = f'{r["variant"]:<14}{r["n_seeds"]:>6}{r["indist_mean"]:>10.3f}'
        for c in COLS:
            line += f'{r.get(c, float("nan")):>11.3f}' if c in r else f'{"-":>11}'
        print(line)
    print("=" * len(hdr))

    # Winners per corruption (lowest mean error).
    print("\n--- Most robust variant per corruption (lower = better) ---")
    winners = {}
    for c in COLS:
        cand = [(r["variant"], r[c]) for r in rows if c in r]
        if cand:
            w = min(cand, key=lambda t: t[1])
            winners[c] = dict(variant=w[0], mean=w[1])
            print(f"  {c:<12}: {w[0]:<14} ({w[1]:.3f} m)")

    # Noise-robustness story vs the plain GT baseline and the output-KD recipe.
    def get(var):
        return next((r for r in rows if r["variant"] == var), None)
    gt, out = get("gt_only"), get("gt_kd_out")
    if gt and out and "noise_0.1" in gt and "noise_0.1" in out:
        print(f"\n--- Noise sigma=0.1 (in-dist ~ {gt['indist_mean']:.2f} m for all) ---")
        for r in sorted(rows, key=lambda r: r.get("noise_0.1", 1e9)):
            if "noise_0.1" in r:
                vs_gt = gt["noise_0.1"] / r["noise_0.1"] if r["noise_0.1"] else float("nan")
                print(f"  {r['variant']:<14} noise0.1 {r['noise_0.1']:.3f} m  ({vs_gt:.2f}x vs gt_only)")

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows, winners=winners), f, indent=2)
    print(f"\nWrote {args.out}")

    # LaTeX
    print("\n--- LaTeX (robustness) ---")
    print(r"\begin{tabular}{l" + "r" * (1 + len(COLS)) + r"}\toprule")
    print(r"Variant & In-dist & " + " & ".join(c.replace("_", " ") for c in COLS) + r"\\\midrule")
    for r in rows:
        cells = " & ".join(f'{r[c]:.3f}' if c in r else "--" for c in COLS)
        print(f'{r["variant"].replace("_", " ")} & {r["indist_mean"]:.3f} & {cells}\\\\')
    print(r"\bottomrule\end{tabular}")


if __name__ == "__main__":
    main()
