"""
studyC_configs.py
Single source of truth for the Study C run ledger (distillation-mechanism study).

Imported by train_studyC.py (variant lookup), run_studyC.sh mirrors it, and
aggregate_studyC.py uses it to know the expected variants.

Seed policy (same principle as Study A): 3 seeds only on the contrasts the headline
claim rests on -- the no-KD baseline (gt_only), the incumbent output-KD recipe
(gt_kd_out), and the primary hypothesis (gt_at_multi). The other three variants are
single-seed screening points; add seeds later only if one unexpectedly wins.
=> 3+1+3+1+3+1 = 12 runs (was 18 when every variant had 3 seeds).

Each variant differs ONLY by the loss term (clean isolation). The KD-term weight is
auto-balanced at train time (see train_studyC.py) so AT/FitNets/output are compared on
equal footing; the ledger just names the method + the GT/KD split.

  method : 'none'      -> GT MSE only
           'output'    -> teacher output-heatmap MSE
           'at_single' -> attention transfer, deepest stage
           'at_multi'  -> attention transfer, multiple stages (multi-scale)
           'fitnet'    -> FitNets intermediate-feature (hint) MSE
  gt_weight : weight on the GT MSE term (0.0 => pure KD)
  kd_ratio  : target KD-term strength relative to the GT term at init (auto-balance)
"""

ANCHOR_SEEDS = [42, 123, 777]   # error-bar anchors
SCREEN_SEED = [42]              # single-seed screening

VARIANTS = [
    dict(name="gt_only",      method="none",      gt_weight=1.0, kd_ratio=0.0, seeds=ANCHOR_SEEDS),
    dict(name="kd_out_only",  method="output",    gt_weight=0.0, kd_ratio=1.0, seeds=SCREEN_SEED),
    dict(name="gt_kd_out",    method="output",    gt_weight=0.7, kd_ratio=0.3, seeds=ANCHOR_SEEDS),
    dict(name="gt_at_single", method="at_single", gt_weight=0.7, kd_ratio=0.3, seeds=SCREEN_SEED),
    dict(name="gt_at_multi",  method="at_multi",  gt_weight=0.7, kd_ratio=0.3, seeds=ANCHOR_SEEDS),
    dict(name="gt_fitnet",    method="fitnet",    gt_weight=0.7, kd_ratio=0.3, seeds=SCREEN_SEED),
]

# corruption suite for eval_studyC.py (matches the paper's existing robustness protocol)
CORRUPTIONS = [
    dict(name="clean"),
    dict(name="noise_0.1", noise=0.1),
    dict(name="noise_0.2", noise=0.2),
    dict(name="drop_ap0", drop_ap=0),
    dict(name="drop_ap1", drop_ap=1),
    dict(name="drop_ap2", drop_ap=2),
    dict(name="drop_ap3", drop_ap=3),
    dict(name="rand_1ap", rand_ap=1),
    dict(name="blur_k5", blur=5),
]


def variant_by_name(name):
    for v in VARIANTS:
        if v["name"] == name:
            return v
    raise KeyError(f"unknown variant: {name}")


def total_runs():
    return sum(len(v["seeds"]) for v in VARIANTS)


if __name__ == "__main__":
    print(f"Study C ledger: {len(VARIANTS)} variants, {total_runs()} runs")
    for v in VARIANTS:
        tag = "anchor(3)" if len(v["seeds"]) > 1 else "screen(1)"
        print(f"  {v['name']:<14} method={v['method']:<9} gt={v['gt_weight']} "
              f"kd_ratio={v['kd_ratio']} seeds={v['seeds']} [{tag}]")
    print(f"Corruptions: {[c['name'] for c in CORRUPTIONS]}")
