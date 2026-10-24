#!/usr/bin/python
"""
train_studyC.py
Train ONE Study C distillation variant (MobileNet student, fig10b, 50 epochs), OFFLINE.

Every variant is identical except the KD loss term, so any two runs are a controlled
comparison of *which kind of teacher knowledge helps*. Teacher signals are PRE-COMPUTED
(no live teacher forward -> student-speed, one clean path, reuses the offline-KD pattern
of train_distillation.py):
  - teacher_train_outputs.pt     : output heatmap  (output-KD; shallow target for at_multi)
  - teacher_train_bottleneck.pt  : encoder bottleneck feature (AT / FitNets deep target)
Run precompute_teacher.py + precompute_teacher_bottleneck.py first.

The KD-term weight is auto-balanced on the first batch so AT / FitNets / output start at
the same relative strength (kd_ratio of the GT term) -- removing hand-tuning as a confound.
Student signals (forward hooks): encoder bottleneck ('deep') + encoder[13] ('mid').

Usage:
  python train_studyC.py --variant gt_kd_out --seed 42
  python train_studyC.py --variant gt_at_multi --seed 123
  python train_studyC.py --variant gt_only --seed 777   # no teacher tensors loaded
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
from student_mobilenet import MobileNetStudent
from kd_losses import attention_transfer_loss, FitNetRegressor, fitnet_loss
from studyC_configs import variant_by_name

NEED_OUT = {"output", "at_multi"}                 # methods that use the teacher output heatmap
NEED_BOT = {"at_single", "at_multi", "fitnet"}    # methods that use the teacher bottleneck feature


def parse_args():
    p = argparse.ArgumentParser(description="Train one Study C distillation variant (offline).")
    p.add_argument("--variant", required=True, help="name from studyC_configs.VARIANTS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)  # 0 = no RAM-heavy worker forks (safe on <256GB boxes)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--teacher_out", type=str, default="./data/teacher_train_outputs.pt")
    p.add_argument("--teacher_bottleneck", type=str, default="./data/teacher_train_bottleneck.pt")
    p.add_argument("--out_dir", type=str, default="./runs_studyC")
    p.add_argument("--results_dir", type=str, default="./results_studyC")
    p.add_argument("--force", action="store_true")
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
        ws.append(w); lbls.append(lbl)
    return torch.cat(ws, 0), torch.cat(lbls, 0)


def main():
    args = parse_args()
    v = variant_by_name(args.variant)
    method, gt_w, kd_ratio = v["method"], float(v["gt_weight"]), float(v["kd_ratio"])
    run_label = f"{args.variant}_seed{args.seed}"
    need_out, need_bot = method in NEED_OUT, method in NEED_BOT

    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, f"summary_{run_label}.json")
    if os.path.exists(summary_path) and not args.force:
        print(f"[skip] {run_label} already done. Use --force to re-run.")
        return

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join(args.out_dir, f"{run_label}_{save_name}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Study C: {run_label} | method={method} gt_w={gt_w} kd_ratio={kd_ratio} ===")
    print(f"Device: {device} | epochs {args.epochs} | need_out={need_out} need_bot={need_bot}")

    # --- data ---
    d = args.data_dir
    trainpath = [f"{d}/dataset_jacobs_July28.mat",
                 f"{d}/dataset_non_fov_train_jacobs_July28_2.mat",
                 f"{d}/dataset_fov_train_jacobs_July28_2.mat"]
    testpath = [f"{d}/dataset_fov_test_jacobs_July28_2.mat",
                f"{d}/dataset_non_fov_test_jacobs_July28_2.mat"]
    print("Loading data...")
    tr_w, tr_lbl = load_split(trainpath)
    te_w, te_lbl = load_split(testpath)
    print(f"Train {tr_w.shape[0]} | Test {te_w.shape[0]}")

    # --- precomputed teacher tensors (aligned to trainpath order) ---
    tensors = [tr_w, tr_lbl]
    i_out = i_bot = None
    t_bot_channels = None
    if need_out:
        t_out = torch.load(args.teacher_out, weights_only=True)
        assert t_out.shape[0] == tr_w.shape[0], f"teacher_out {t_out.shape[0]} != {tr_w.shape[0]}"
        i_out = len(tensors); tensors.append(t_out)
        print(f"teacher outputs: {tuple(t_out.shape)}")
    if need_bot:
        t_bot = torch.load(args.teacher_bottleneck, weights_only=True)
        assert t_bot.shape[0] == tr_w.shape[0], f"teacher_bot {t_bot.shape[0]} != {tr_w.shape[0]}"
        t_bot_channels = t_bot.shape[1]
        i_bot = len(tensors); tensors.append(t_bot)
        print(f"teacher bottleneck: {tuple(t_bot.shape)}")

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors), batch_size=args.batch_size,
        shuffle=True, generator=g, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(te_w, te_lbl), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- student + feature hooks ---
    student = MobileNetStudent(input_nc=4).to(device)
    sfeat = {}
    student.encoder.register_forward_hook(lambda m, i, o: sfeat.__setitem__("deep", o))
    student.encoder[13].register_forward_hook(lambda m, i, o: sfeat.__setitem__("mid", o))

    criterion = nn.MSELoss()
    params = list(student.parameters())
    regressor = None
    if method == "fitnet":
        regressor = FitNetRegressor(320, t_bot_channels).to(device)
        params += list(regressor.parameters())
        print(f"FitNets regressor: student 320ch -> teacher {t_bot_channels}ch")

    def aux_loss(student_out, t_out_b, t_bot_b):
        if method == "output":
            return criterion(student_out, t_out_b)
        if method == "at_single":
            return attention_transfer_loss(sfeat["deep"], t_bot_b)
        if method == "at_multi":
            return attention_transfer_loss([sfeat["deep"], sfeat["mid"]], [t_bot_b, t_out_b])
        if method == "fitnet":
            return fitnet_loss(sfeat["deep"], t_bot_b, regressor)
        raise ValueError(method)

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    beta = None
    best = dict(mean=float("inf"), median=None, p90=None, p99=None, epoch=-1)
    best_errors = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        for data in train_loader:
            w = data[0].to(device)
            lbl = data[1].to(device)
            t_out_b = data[i_out].to(device) if i_out is not None else None
            t_bot_b = data[i_bot].to(device) if i_bot is not None else None

            out = student(w)
            L_GT = criterion(out, lbl)
            if method == "none":
                loss = L_GT
            else:
                L_aux = aux_loss(out, t_out_b, t_bot_b)
                if beta is None:
                    beta = 1.0 if gt_w == 0.0 else float(kd_ratio * (gt_w * L_GT.detach())
                                                          / (L_aux.detach() + 1e-12))
                    print(f"  auto-balanced beta = {beta:.4g}")
                loss = gt_w * L_GT + beta * L_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- test (student only) ---
        student.eval()
        errs = []
        with torch.no_grad():
            for w, lbl in test_loader:
                out = student(w.to(device))
                errs.extend(localization_error(out.cpu().numpy(), lbl.numpy(), scale=0.1))
        errs = np.asarray(errs, dtype=np.float64)
        mean_e, median_e = float(errs.mean()), float(np.median(errs))
        p90, p99 = float(np.percentile(errs, 90)), float(np.percentile(errs, 99))
        scheduler.step()

        write_log([str(mean_e)], run_label, log_dir=save_dir, log_type="test_mean_error")
        write_log([str(median_e)], run_label, log_dir=save_dir, log_type="test_median_error")
        write_log([str(p90)], run_label, log_dir=save_dir, log_type="test_90_error")

        is_best = mean_e < best["mean"]
        print(f"Epoch {epoch}/{args.epochs} | {time.time()-t0:.0f}s | "
              f"mean={mean_e:.3f}m median={median_e:.3f}m P90={p90:.3f}m"
              f"{'  <-- BEST' if is_best else ''}")
        if is_best:
            best = dict(mean=mean_e, median=median_e, p90=p90, p99=p99, epoch=epoch)
            best_errors = errs.copy()
            torch.save(student.state_dict(), os.path.join(save_dir, f"best_{args.variant}.pth"))
        torch.save(student.state_dict(), os.path.join(save_dir, f"latest_{args.variant}.pth"))

    # --- persist ---
    err_path = os.path.join(args.results_dir, f"errors_{run_label}.npy")
    np.save(err_path, best_errors)
    summary = dict(variant=args.variant, run_label=run_label, seed=args.seed, method=method,
                   gt_weight=gt_w, kd_ratio=kd_ratio, beta=beta, epochs=args.epochs,
                   best=best, best_weights=os.path.join(save_dir, f"best_{args.variant}.pth"),
                   errors_file=os.path.basename(err_path))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone {run_label}: best mean={best['mean']:.4f}m (epoch {best['epoch']})")


if __name__ == "__main__":
    main()
