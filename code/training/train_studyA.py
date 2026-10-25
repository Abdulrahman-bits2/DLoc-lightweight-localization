#!/usr/bin/python
"""
train_studyA.py
Train ONE Study A config (spatial-preservation ablation) from CLI args.

Every run is identical except the ablated knob, so any two runs are a controlled
comparison. Selects the best checkpoint by MEAN error (the study's primary metric,
which -- unlike the pixel-quantized median -- resolves sub-pixel differences) and
saves the best epoch's per-sample test errors for CDF / paired-test analysis.

Usage (one config):
  python train_studyA.py --use_skips 1 --decoder_width 1.0 --seed 42
  python train_studyA.py --use_skips 0 --decoder_width 2.0 --seed 42
  python train_studyA.py --use_skips 1 --decoder_width 1.0 --bottleneck aspp --seed 42
  python train_studyA.py --use_skips 1 --width_mult 0.5 --seed 42   # from-scratch encoder

Outputs (under --results_dir):
  errors_<tag>_seed<seed>.npy    per-sample test errors at the best epoch (metres)
  summary_<tag>_seed<seed>.json  config + params + FLOPs + best-epoch metrics
Weights + per-epoch logs go under --out_dir/<tag>_seed<seed>_<timestamp>/.
"""

# --- path bootstrap (code reorganized into core/ models/ configs/) ---
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
import warnings

import numpy as np
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from utils import localization_error, write_log
from data_loader import load_data
from student_configurable import build_student
from studyA_configs import config_tag


def parse_args():
    p = argparse.ArgumentParser(description="Train one Study A config.")
    p.add_argument("--use_skips", type=int, default=1, choices=[0, 1])
    p.add_argument("--decoder_width", type=float, default=1.0)
    p.add_argument("--width_mult", type=float, default=1.0)
    p.add_argument("--bottleneck", type=str, default="none", choices=["none", "aspp"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)  # 0 = no RAM-heavy worker forks (safe on <256GB boxes)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs_studyA")
    p.add_argument("--results_dir", type=str, default="./results_studyA")
    p.add_argument("--force", action="store_true",
                   help="re-run even if a summary for this config+seed already exists")
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


def count_flops(model, device):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 4, 161, 361, device=device)
    fc = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), fc:
            model(x)
        return fc.get_total_flops()
    except Exception:
        return 0


def main():
    args = parse_args()
    use_skips = bool(args.use_skips)
    tag = config_tag(use_skips, args.decoder_width, args.width_mult, args.bottleneck)
    run_label = f"{tag}_seed{args.seed}"

    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, f"summary_{run_label}.json")
    if os.path.exists(summary_path) and not args.force:
        print(f"[skip] {run_label} already done ({summary_path}). Use --force to re-run.")
        return

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join(args.out_dir, f"{run_label}_{save_name}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Study A run: {run_label} ===")
    print(f"Device: {device} | epochs {args.epochs} | batch {args.batch_size} | lr {args.lr}")
    print(f"Config: use_skips={use_skips} decoder_width={args.decoder_width} "
          f"width_mult={args.width_mult} bottleneck={args.bottleneck}")
    print(f"Save dir: {save_dir}")

    # --- data (fig10b Jacobs Hall) ---
    d = args.data_dir
    trainpath = [
        f"{d}/dataset_jacobs_July28.mat",
        f"{d}/dataset_non_fov_train_jacobs_July28_2.mat",
        f"{d}/dataset_fov_train_jacobs_July28_2.mat",
    ]
    testpath = [
        f"{d}/dataset_fov_test_jacobs_July28_2.mat",
        f"{d}/dataset_non_fov_test_jacobs_July28_2.mat",
    ]
    print("Loading data...")
    train_w, train_lbl = load_split(trainpath)
    test_w, test_lbl = load_split(testpath)
    print(f"Train {train_w.shape[0]} | Test {test_w.shape[0]}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_w, train_lbl),
        batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_w, test_lbl),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=(args.num_workers > 0),
    )

    # --- model ---
    model = build_student(use_skips=use_skips, decoder_width=args.decoder_width,
                          width_mult=args.width_mult, bottleneck=args.bottleneck).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    flops = count_flops(model, device)
    print(f"Params: {n_params:,} ({n_params/1e6:.3f}M) | FLOPs: {flops/1e9:.2f}G")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = dict(mean=float("inf"), median=None, p90=None, p99=None, epoch=-1)
    best_errors = None

    # --- mid-run resume: if a checkpoint for this config+seed exists, continue from it ---
    ckpt_path = os.path.join(args.out_dir, f"resume_{run_label}.pt")
    start_epoch = 1
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        best, best_errors, start_epoch = ck["best"], ck["best_errors"], ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"[resume] {run_label}: continuing from epoch {start_epoch} "
              f"(best mean={best['mean']:.4f}m @ epoch {best['epoch']})")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        for data in train_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- test ---
        model.eval()
        test_errors = []
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                labels = data[1].to(device)
                out = model(inputs)
                test_errors.extend(localization_error(
                    out.cpu().numpy(), labels.cpu().numpy(), scale=0.1))
        test_errors = np.asarray(test_errors, dtype=np.float64)

        mean_e = float(test_errors.mean())
        median_e = float(np.median(test_errors))
        p90 = float(np.percentile(test_errors, 90))
        p99 = float(np.percentile(test_errors, 99))
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        write_log([str(mean_e)], run_label, log_dir=save_dir, log_type="test_mean_error")
        write_log([str(median_e)], run_label, log_dir=save_dir, log_type="test_median_error")
        write_log([str(p90)], run_label, log_dir=save_dir, log_type="test_90_error")
        write_log([str(p99)], run_label, log_dir=save_dir, log_type="test_99_error")

        # best-by-MEAN (study's primary metric)
        is_best = mean_e < best["mean"]
        marker = "  <-- NEW BEST" if is_best else ""
        print(f"Epoch {epoch}/{args.epochs} | {time.time()-t0:.0f}s | LR={lr_now:.6f} | "
              f"mean={mean_e:.3f}m median={median_e:.3f}m P90={p90:.3f}m P99={p99:.3f}m{marker}")
        if is_best:
            best = dict(mean=mean_e, median=median_e, p90=p90, p99=p99, epoch=epoch)
            best_errors = test_errors.copy()
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{tag}.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"latest_{tag}.pth"))
        # resume checkpoint (atomic write) — full state to continue after an interruption
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optim": optimizer.state_dict(), "sched": scheduler.state_dict(),
            "best": best, "best_errors": best_errors,
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    # --- persist results for aggregation ---
    err_path = os.path.join(args.results_dir, f"errors_{run_label}.npy")
    np.save(err_path, best_errors)
    summary = dict(
        tag=tag, run_label=run_label, seed=args.seed,
        use_skips=use_skips, decoder_width=args.decoder_width,
        width_mult=args.width_mult, bottleneck=args.bottleneck,
        pretrained_encoder=abs(args.width_mult - 1.0) < 1e-6,
        params=n_params, flops=flops, epochs=args.epochs,
        best=best, save_dir=save_dir, errors_file=os.path.basename(err_path),
    )
    with open(os.path.join(args.results_dir, f"summary_{run_label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(ckpt_path):      # config complete -> drop its resume checkpoint
        os.remove(ckpt_path)

    print(f"\nDone {run_label}: best mean={best['mean']:.4f}m median={best['median']:.4f}m "
          f"P90={best['p90']:.3f}m (epoch {best['epoch']})")
    print(f"Saved: {err_path}")


if __name__ == "__main__":
    main()
