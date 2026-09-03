#!/usr/bin/python
"""
eval_studyC.py
Robustness evaluation for Study C -- the headline metric (KD gives no in-dist gain here,
only robustness). Runs each trained variant's BEST checkpoint through a corruption suite on
the fig10b test set (no extra data). Corruptions are seeded per-type so every variant sees
the identical realization (fair comparison).

Writes results_studyC/corrupt_<run_label>.json = {corruption: {mean, median, p90}} per run.

Usage:
  python eval_studyC.py                 # evaluate every summary in results_studyC/
  python eval_studyC.py --variant gt_at_multi
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
import torch
from torchvision.transforms.functional import gaussian_blur

from utils import localization_error
from data_loader import load_data
from student_mobilenet import MobileNetStudent
from studyC_configs import CORRUPTIONS


def load_test(data_dir):
    ws, lbls = [], []
    for fp in [f"{data_dir}/dataset_fov_test_jacobs_July28_2.mat",
               f"{data_dir}/dataset_non_fov_test_jacobs_July28_2.mat"]:
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        ws.append(w); lbls.append(lbl)
    return torch.cat(ws, 0), torch.cat(lbls, 0)


def apply_corruption(x, c):
    """x: [B,4,H,W] on device. Deterministic (caller seeds before the batch loop)."""
    x = x.clone()
    if "noise" in c:
        x = x + torch.randn_like(x) * c["noise"]
    if "drop_ap" in c:
        x[:, c["drop_ap"]] = 0.0
    if "rand_ap" in c:
        idx = torch.randint(0, x.shape[1], (x.shape[0],), device=x.device)
        x[torch.arange(x.shape[0], device=x.device), idx] = 0.0
    if "blur" in c:
        x = gaussian_blur(x, kernel_size=c["blur"], sigma=1.0)
    return x


def eval_run(summary, test_w, test_lbl, device, batch_size=32):
    model = MobileNetStudent(input_nc=4).to(device)
    wpath = summary.get("best_weights")
    if not wpath or not os.path.exists(wpath):
        print(f"  !! weights missing for {summary['run_label']} ({wpath}) -- skip")
        return None
    model.load_state_dict(torch.load(wpath, map_location=device))
    model.eval()

    out = {}
    for c in CORRUPTIONS:
        torch.manual_seed(1234)  # identical corruption realization across all variants
        errs = []
        with torch.no_grad():
            for i in range(0, test_w.shape[0], batch_size):
                xb = test_w[i:i + batch_size].to(device)
                lb = test_lbl[i:i + batch_size]
                xb = apply_corruption(xb, c)
                pred = model(xb)
                errs.extend(localization_error(pred.cpu().numpy(), lb.numpy(), scale=0.1))
        errs = np.asarray(errs)
        out[c["name"]] = dict(mean=float(errs.mean()), median=float(np.median(errs)),
                              p90=float(np.percentile(errs, 90)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_studyC")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--variant", default=None, help="only this variant (prefix match)")
    args = ap.parse_args()

    summaries = []
    for fp in sorted(glob.glob(os.path.join(args.results_dir, "summary_*.json"))):
        with open(fp) as f:
            s = json.load(f)
        if args.variant and not s["variant"].startswith(args.variant):
            continue
        summaries.append(s)
    if not summaries:
        print(f"No summaries in {args.results_dir}.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading fig10b test set...")
    test_w, test_lbl = load_test(args.data_dir)
    print(f"Test samples: {test_w.shape[0]} | evaluating {len(summaries)} runs")

    for s in summaries:
        rl = s["run_label"]
        out_fp = os.path.join(args.results_dir, f"corrupt_{rl}.json")
        if os.path.exists(out_fp):
            print(f"  [skip] {rl} (corrupt_{rl}.json exists)")
            continue
        res = eval_run(s, test_w, test_lbl, device)
        if res is None:
            continue
        with open(out_fp, "w") as f:
            json.dump(res, f, indent=2)
        clean_m = res["clean"]["mean"]
        n01 = res.get("noise_0.1", {}).get("mean", float("nan"))
        print(f"  {rl:<22} clean {clean_m:.3f}m | noise0.1 {n01:.3f}m -> corrupt_{rl}.json")


if __name__ == "__main__":
    main()
