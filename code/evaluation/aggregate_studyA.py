"""
aggregate_studyA.py
Collect Study A runs -> summary table + statistics for the paper.

Reads results_studyA/summary_*.json (+ errors_*.npy) and reports:
  - per-config params/FLOPs and best mean/median/P90/P99 (mean +/- std over seeds),
  - the headline decoder_width=1.0 pair (skip vs no-skip): 3-seed mean +/- std,
    paired Wilcoxon signed-rank on the per-sample errors, and the "at equal
    parameters" gap (no-skip curve interpolated to the skip point's param count),
  - a LaTeX table and studyA_summary.json for plot_studyA.py.

Usage:
  python aggregate_studyA.py [--results_dir ./results_studyA]
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
from studyA_configs import config_tag


def load_summaries(results_dir):
    runs = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "summary_*.json"))):
        with open(fp) as f:
            s = json.load(f)
        err_fp = os.path.join(results_dir, s.get("errors_file", f"errors_{s['run_label']}.npy"))
        s["_errors_path"] = err_fp if os.path.exists(err_fp) else None
        runs.append(s)
    return runs


def group_by_tag(runs):
    groups = {}
    for s in runs:
        groups.setdefault(s["tag"], []).append(s)
    return groups


def agg(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)


def paired_test(errs_a, errs_b):
    """Paired comparison of per-sample errors (same test order). Returns (name, stat, p)."""
    d = np.asarray(errs_a) - np.asarray(errs_b)
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(errs_a, errs_b)
        return "wilcoxon", float(stat), float(p)
    except Exception:
        # numpy-only bootstrap of the mean difference (95% CI + two-sided p ~ CI crossing 0)
        rng = np.random.default_rng(0)
        n = len(d)
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot > 0).mean(), (boot < 0).mean())
        return f"bootstrap(mean_diff={d.mean():.4f}, 95%CI[{lo:.4f},{hi:.4f}])", float(d.mean()), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_studyA")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <results_dir>/studyA_summary.json)")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.results_dir, "studyA_summary.json")

    runs = load_summaries(args.results_dir)
    if not runs:
        print(f"No summaries in {args.results_dir}. Run the ledger first.")
        return
    groups = group_by_tag(runs)
    print(f"Loaded {len(runs)} runs across {len(groups)} configs "
          f"(expected 13 runs / 9 configs when complete).\n")

    # Per-config aggregate.
    table = []
    for tag, gs in groups.items():
        g0 = gs[0]
        mean_m, mean_s = agg([x["best"]["mean"] for x in gs])
        med_m, med_s = agg([x["best"]["median"] for x in gs])
        p90_m, p90_s = agg([x["best"]["p90"] for x in gs])
        table.append(dict(
            tag=tag, use_skips=g0["use_skips"], decoder_width=g0["decoder_width"],
            width_mult=g0["width_mult"], bottleneck=g0["bottleneck"],
            pretrained=g0.get("pretrained_encoder", True),
            params=g0["params"], flops=g0["flops"], n_seeds=len(gs),
            mean=mean_m, mean_std=mean_s, median=med_m, median_std=med_s,
            p90=p90_m, p90_std=p90_s,
        ))
    table.sort(key=lambda r: r["params"])

    print("=" * 96)
    print(f'{"config":<24}{"params":>9}{"FLOPs":>8}{"seeds":>6}{"mean(m)":>16}{"median":>9}{"P90":>8}')
    print("-" * 96)
    for r in table:
        mstr = f'{r["mean"]:.3f}+/-{r["mean_std"]:.3f}' if r["n_seeds"] > 1 else f'{r["mean"]:.3f}'
        print(f'{r["tag"]:<24}{r["params"]/1e6:>8.2f}M{r["flops"]/1e9:>7.2f}G{r["n_seeds"]:>6}'
              f'{mstr:>16}{r["median"]:>9.3f}{r["p90"]:>8.3f}')
    print("=" * 96)

    # Headline: decoder_width=1.0 pair.
    report = dict(configs=table)
    tag_sk = config_tag(True, 1.0)
    tag_ns = config_tag(False, 1.0)
    if tag_sk in groups and tag_ns in groups:
        sk = next(r for r in table if r["tag"] == tag_sk)
        ns = next(r for r in table if r["tag"] == tag_ns)
        print(f"\n--- Headline pair (decoder_width=1.0) ---")
        print(f"  no-skip: {ns['params']/1e6:.3f}M  mean {ns['mean']:.3f}+/-{ns['mean_std']:.3f} m  ({ns['n_seeds']} seeds)")
        print(f"  skip   : {sk['params']/1e6:.3f}M  mean {sk['mean']:.3f}+/-{sk['mean_std']:.3f} m  ({sk['n_seeds']} seeds)")
        print(f"  skip improves mean by {100*(ns['mean']-sk['mean'])/ns['mean']:.1f}% "
              f"for +{100*(sk['params']-ns['params'])/ns['params']:.1f}% params.")

        # paired test on seed-42 per-sample errors
        e_sk = next((x["_errors_path"] for x in groups[tag_sk] if x["seed"] == 42), None)
        e_ns = next((x["_errors_path"] for x in groups[tag_ns] if x["seed"] == 42), None)
        if e_sk and e_ns:
            a, b = np.load(e_ns), np.load(e_sk)  # no-skip vs skip
            if len(a) == len(b):
                name, stat, p = paired_test(a, b)
                print(f"  paired {name}: stat={stat:.4g}, p={p:.2e} (seed 42, n={len(a)})")
                report["paired_test"] = dict(method=name, stat=stat, p=p, n=int(len(a)))

    # "At equal parameters" read-off: interpolate the no-skip curve to each skip point.
    ns_curve = sorted([r for r in table if not r["use_skips"]], key=lambda r: r["params"])
    if len(ns_curve) >= 2:
        xs = np.array([r["params"] for r in ns_curve])
        ys = np.array([r["mean"] for r in ns_curve])
        print(f"\n--- 'At equal parameters' (no-skip curve interpolated to each skip point) ---")
        eq = []
        for r in sorted([r for r in table if r["use_skips"] and r["bottleneck"] == "none"
                         and abs(r["width_mult"] - 1.0) < 1e-6], key=lambda r: r["params"]):
            if xs.min() <= r["params"] <= xs.max():
                ns_at = float(np.interp(r["params"], xs, ys))
                print(f"  {r['tag']:<20} {r['params']/1e6:.2f}M: skip {r['mean']:.3f} m vs "
                      f"no-skip@equal-params {ns_at:.3f} m  -> skip better by {ns_at-r['mean']:.3f} m")
                eq.append(dict(tag=r["tag"], params=r["params"], skip_mean=r["mean"],
                               noskip_interp_mean=ns_at, gap=ns_at - r["mean"]))
            else:
                print(f"  {r['tag']:<20} {r['params']/1e6:.2f}M: outside no-skip param range (not bracketed)")
        report["equal_params"] = eq

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")

    # LaTeX table.
    print("\n--- LaTeX (main results) ---")
    print(r"\begin{tabular}{lrrrrr}\toprule")
    print(r"Config & Skips & Params & FLOPs & Mean (m) & P90 (m)\\\midrule")
    for r in table:
        sk = "yes" if r["use_skips"] else "no"
        mstr = f'{r["mean"]:.3f}$\\pm${r["mean_std"]:.3f}' if r["n_seeds"] > 1 else f'{r["mean"]:.3f}'
        print(f'{r["tag"].replace("_", " ")} & {sk} & {r["params"]/1e6:.2f}M & '
              f'{r["flops"]/1e9:.2f}G & {mstr} & {r["p90"]:.3f}\\\\')
    print(r"\bottomrule\end{tabular}")


if __name__ == "__main__":
    main()
