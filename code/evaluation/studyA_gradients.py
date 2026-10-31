#!/usr/bin/python
"""
studyA_gradients.py
----------------------------------------------------------------------
Gradient-flow + training-dynamics probe for the Study A spatial-preservation
ablation.  Answers the professor's question:

  "Why do skip connections increase accuracy?  Is it because the gradient to the
   early layers is very small (vanishing), so the skips carry over part of what
   was learned in the initial layers?"

It trains the TWO anchor configs -- no-skip (plain MobileNetV2) and skip
(MobileNetV2-UNet), both decoder_width=1.0, seed 42 -- with EXACTLY the same
setup as train_studyA.py, and additionally records, every epoch:

  * train_loss, test mean / median / P90  (the "gradient descent snapshot")
  * the L2 norm of the gradient reaching each module group
    (inc, enc1..enc4, bottleneck, dec1..dec3, final), averaged over the epoch.
    inc/enc1 are the EARLIEST layers -> this is the vanishing-gradient test.

Outputs (under --out):
  grad_noskip.json, grad_skip.json          raw per-epoch curves + grad norms
  fig_training_curves.png                    test error vs epoch (skip vs no-skip)
  fig_grad_profile.png                       grad norm vs layer depth (the key plot)
  fig_grad_early_vs_epoch.png                early-layer grad norm vs epoch

Runtime: ~epoch time (train_studyA ~67 s/epoch on the RTX 5000) x epochs x 2.
  --epochs 60  ~2.2 h total.  --epochs 40 ~1.5 h.  30 already shows the pattern.
JSON is rewritten every epoch and a finished model is skipped on restart, so an
interruption only loses the current model's in-progress epochs.

Usage:
  python code/evaluation/studyA_gradients.py --epochs 60 --data_dir ./data --out ./results_studyA/grads
"""

# --- path bootstrap (mirror train_studyA.py) ---
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

from utils import localization_error
from data_loader import load_data
from student_configurable import build_student

# module groups ordered input -> output (for the gradient-vs-depth profile)
GROUP_ORDER = ["inc", "enc1", "enc2", "enc3", "enc4",
               "bottleneck", "dec1", "dec2", "dec3", "final"]


def parse_args():
    p = argparse.ArgumentParser(description="Gradient-flow probe for Study A skip vs no-skip.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="./results_studyA/grads")
    p.add_argument("--no_plots", action="store_true", help="skip PNG generation (JSON only)")
    return p.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(paths):
    ws, lbls = [], []
    for fp in paths:
        _, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        ws.append(w)
        lbls.append(lbl)
    return torch.cat(ws, dim=0), torch.cat(lbls, dim=0)


def group_of(param_name):
    """Map 'enc1.0.conv.0.weight' -> 'enc1', 'dec1.0.weight' -> 'dec1', etc."""
    return param_name.split(".")[0]


def build_groups(model):
    """{group_name: [params]} using the top-level attribute as the group."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups.setdefault(group_of(name), []).append(p)
    return groups


def depth_inventory(model):
    """#weight layers (Conv2d / ConvTranspose2d / Linear) per group and total."""
    WL = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    per_group, total = {}, 0
    for name, m in model.named_modules():
        if isinstance(m, WL):
            g = group_of(name) if name else "root"
            per_group[g] = per_group.get(g, 0) + 1
            total += 1
    return total, per_group


def run_one(use_skips, args, device, train_loader, test_loader, n_batches):
    tag = "skip" if use_skips else "noskip"
    out_json = os.path.join(args.out, f"grad_{tag}.json")
    if os.path.exists(out_json):
        with open(out_json) as f:
            done = json.load(f)
        if done.get("complete"):
            print(f"[skip] {tag} already complete ({out_json})")
            return done

    seed_everything(args.seed)
    model = build_student(use_skips=use_skips, decoder_width=1.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    total_depth, depth_by_group = depth_inventory(model)
    groups = build_groups(model)
    params_by_group = {g: sum(p.numel() for p in ps) for g, ps in groups.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n=== {tag}: params={n_params/1e6:.3f}M | depth={total_depth} weight layers ===")
    print(f"    depth by group: {depth_by_group}")

    record = dict(tag=tag, use_skips=bool(use_skips), params=n_params,
                  depth_total=total_depth, depth_by_group=depth_by_group,
                  params_by_group=params_by_group, group_order=GROUP_ORDER,
                  epochs=[], complete=False)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        g_accum = {g: 0.0 for g in groups}
        loss_accum = 0.0
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            # --- read per-group gradient L2 norm BEFORE the step ---
            for g, ps in groups.items():
                s = 0.0
                for p in ps:
                    if p.grad is not None:
                        s += float(p.grad.detach().pow(2).sum().item())
                g_accum[g] += s ** 0.5
            optimizer.step()
            loss_accum += float(loss.item())

        grad_norms = {g: g_accum[g] / n_batches for g in groups}
        train_loss = loss_accum / n_batches

        # --- test ---
        model.eval()
        errs = []
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                labels = data[1].to(device)
                out = model(inputs)
                errs.extend(localization_error(out.cpu().numpy(),
                                               labels.cpu().numpy(), scale=0.1))
        errs = np.asarray(errs, dtype=np.float64)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ep = dict(epoch=epoch, train_loss=train_loss, lr=lr_now,
                  test_mean=float(errs.mean()), test_median=float(np.median(errs)),
                  test_p90=float(np.percentile(errs, 90)), grad_norms=grad_norms)
        record["epochs"].append(ep)
        with open(out_json, "w") as f:               # rewrite every epoch (crash-safe)
            json.dump(record, f, indent=2)

        print(f"[{tag}] ep {epoch}/{args.epochs} | {time.time()-t0:.0f}s | "
              f"loss={train_loss:.4e} mean={ep['test_mean']:.3f}m "
              f"|g(inc)|={grad_norms.get('inc', 0):.2e} "
              f"|g(dec1)|={grad_norms.get('dec1', 0):.2e}")

    record["complete"] = True
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[{tag}] done -> {out_json}")
    return record


def make_plots(rec_no, rec_sk, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable ({e}); JSON saved, plot later.")
        return

    def curve(rec, key):
        return [e["epoch"] for e in rec["epochs"]], [e[key] for e in rec["epochs"]]

    # 1) training-dynamics: test mean error vs epoch
    plt.figure(figsize=(6, 4))
    ex, ey = curve(rec_no, "test_mean"); plt.plot(ex, ey, label="no-skip (plain)", color="#c0392b")
    sx, sy = curve(rec_sk, "test_mean"); plt.plot(sx, sy, label="skip (U-Net)", color="#2471a3")
    plt.xlabel("epoch"); plt.ylabel("test mean error (m)")
    plt.title("Training dynamics (gradient descent): skip vs no-skip")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig-training-curves.png"), dpi=150); plt.close()

    # 2) gradient-vs-depth profile at an early epoch (vanishing-gradient picture)
    idx = min(1, len(rec_no["epochs"]) - 1)   # epoch 2 if available, else first
    order = [g for g in GROUP_ORDER if g in rec_sk["epochs"][idx]["grad_norms"]
             or g in rec_no["epochs"][idx]["grad_norms"]]
    gn_no = [rec_no["epochs"][idx]["grad_norms"].get(g, np.nan) for g in order]
    gn_sk = [rec_sk["epochs"][idx]["grad_norms"].get(g, np.nan) for g in order]
    x = np.arange(len(order))
    plt.figure(figsize=(7, 4))
    plt.semilogy(x, gn_no, "o-", label="no-skip (plain)", color="#c0392b")
    plt.semilogy(x, gn_sk, "s-", label="skip (U-Net)", color="#2471a3")
    plt.xticks(x, order, rotation=45, ha="right")
    plt.xlabel("layer group  (input  ->  output)"); plt.ylabel("gradient L2 norm (log)")
    plt.title(f"Gradient reaching each layer (epoch {rec_no['epochs'][idx]['epoch']})")
    plt.axvspan(-0.5, 1.5, color="gold", alpha=.15)   # highlight the early layers
    plt.legend(); plt.grid(alpha=.3, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig-grad-profile.png"), dpi=150); plt.close()

    # 3) early-layer (inc) gradient norm vs epoch
    plt.figure(figsize=(6, 4))
    for rec, lab, col in [(rec_no, "no-skip", "#c0392b"), (rec_sk, "skip", "#2471a3")]:
        ep = [e["epoch"] for e in rec["epochs"]]
        gi = [e["grad_norms"].get("inc", np.nan) for e in rec["epochs"]]
        plt.semilogy(ep, gi, label=f"{lab}: inc (first conv)", color=col)
    plt.xlabel("epoch"); plt.ylabel("gradient L2 norm at first layer (log)")
    plt.title("Gradient reaching the earliest layer over training")
    plt.legend(); plt.grid(alpha=.3, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig-grad-early.png"), dpi=150); plt.close()
    print(f"[plots] wrote 3 PNGs to {out_dir}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | epochs {args.epochs} | batch {args.batch_size} | lr {args.lr}")

    d = args.data_dir
    trainpath = [f"{d}/dataset_jacobs_July28.mat",
                 f"{d}/dataset_non_fov_train_jacobs_July28_2.mat",
                 f"{d}/dataset_fov_train_jacobs_July28_2.mat"]
    testpath = [f"{d}/dataset_fov_test_jacobs_July28_2.mat",
                f"{d}/dataset_non_fov_test_jacobs_July28_2.mat"]
    print("Loading data...")
    train_w, train_lbl = load_split(trainpath)
    test_w, test_lbl = load_split(testpath)
    print(f"Train {train_w.shape[0]} | Test {test_w.shape[0]}")

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_w, train_lbl),
        batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_w, test_lbl),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    n_batches = len(train_loader)

    rec_no = run_one(False, args, device, train_loader, test_loader, n_batches)
    rec_sk = run_one(True,  args, device, train_loader, test_loader, n_batches)

    if not args.no_plots:
        make_plots(rec_no, rec_sk, args.out)

    # compact console summary
    print("\n================ GRADIENT-FLOW SUMMARY ================")
    for rec in (rec_no, rec_sk):
        e0 = rec["epochs"][0]["grad_norms"]
        print(f"{rec['tag']:7s} depth={rec['depth_total']} params={rec['params']/1e6:.3f}M "
              f"| g(inc)={e0.get('inc', 0):.2e} g(enc1)={e0.get('enc1', 0):.2e} "
              f"g(dec1)={e0.get('dec1', 0):.2e}  (epoch 1)")
    print("=======================================================")


if __name__ == "__main__":
    main()
