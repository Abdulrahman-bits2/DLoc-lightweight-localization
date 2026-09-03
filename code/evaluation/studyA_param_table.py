"""
studyA_param_table.py
Params + FLOPs for every Study A config. Run FIRST (seconds, CPU-only, no data) to:
  1. confirm the parameterized model reproduces the shipped nets, and
  2. confirm the no-skip and skip families overlap in parameter-count, so the
     "at equal parameters, skips win" claim can be read straight off the scatter.

Usage:
  python studyA_param_table.py
"""

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import torch
from student_configurable import build_student
from studyA_configs import LEDGER, config_tag


def count_flops(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(x)
    return fc.get_total_flops()


def main():
    dummy = torch.randn(1, 4, 161, 361)
    rows = []
    for c in LEDGER:
        tag = config_tag(c["use_skips"], c["decoder_width"], c["width_mult"], c["bottleneck"])
        m = build_student(use_skips=c["use_skips"], decoder_width=c["decoder_width"],
                          width_mult=c["width_mult"], bottleneck=c["bottleneck"]).eval()
        n = sum(p.numel() for p in m.parameters())
        try:
            flops = count_flops(m, dummy)
        except Exception as e:
            print(f"  FLOPs failed for {tag}: {e}")
            flops = 0
        rows.append(dict(tag=tag, use_skips=c["use_skips"], decoder_width=c["decoder_width"],
                         width_mult=c["width_mult"], bottleneck=c["bottleneck"],
                         params=n, flops=flops, seeds=c["seeds"]))

    print("=" * 78)
    print(f'{"config":<28}{"params":>12}{"FLOPs":>12}{"seeds":>14}')
    print("-" * 78)
    for r in rows:
        pstr = f'{r["params"]/1e6:.3f}M'
        fstr = f'{r["flops"]/1e9:.2f}G' if r["flops"] >= 1e9 else f'{r["flops"]/1e6:.0f}M'
        print(f'{r["tag"]:<28}{pstr:>12}{fstr:>12}{str(r["seeds"]):>14}')
    print("=" * 78)

    # Sanity vs shipped nets.
    def find(sk, dw, wm=1.0, bn="none"):
        t = config_tag(sk, dw, wm, bn)
        return next((r for r in rows if r["tag"] == t), None)
    ns10, sk10 = find(False, 1.0), find(True, 1.0)
    if ns10 and sk10:
        print(f'\nShipped-net check: no-skip dw1.0 = {ns10["params"]/1e6:.3f}M (expect ~2.27M / 0.707 m), '
              f'skip dw1.0 = {sk10["params"]/1e6:.3f}M (expect ~2.41M / 0.640 m)')
        print(f'Skip adds {(sk10["params"]-ns10["params"])/1e3:.0f}K params '
              f'(+{100*(sk10["params"]-ns10["params"])/ns10["params"]:.1f}%) at equal decoder width.')

    # Param-overlap: does the no-skip family bracket the skip curve points?
    ns = sorted([r for r in rows if not r["use_skips"]], key=lambda r: r["params"])
    sk = sorted([r for r in rows if r["use_skips"] and r["bottleneck"] == "none"
                 and abs(r["width_mult"] - 1.0) < 1e-6], key=lambda r: r["params"])
    if ns and sk:
        ns_lo, ns_hi = ns[0]["params"], ns[-1]["params"]
        covered = [r for r in sk if ns_lo <= r["params"] <= ns_hi]
        print(f'\nParam overlap: no-skip family spans {ns_lo/1e6:.2f}-{ns_hi/1e6:.2f}M; '
              f'{len(covered)}/{len(sk)} skip points fall inside that range '
              f'(=> "equal-params" read-off is valid where covered).')


if __name__ == "__main__":
    main()
