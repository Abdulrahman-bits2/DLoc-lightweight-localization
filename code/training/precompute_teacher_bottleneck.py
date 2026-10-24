#!/usr/bin/python
"""
precompute_teacher_bottleneck.py
Pre-compute the DLoc teacher's ENCODER BOTTLENECK feature (enc_model.output) for every
TRAIN sample and save it, so Study C's feature-based KD (AT / FitNets) trains fully
offline at student speed -- no live teacher forward in train_studyC.py.

Companion to precompute_teacher.py (which saves the output heatmaps). Run both once:
  python precompute_teacher.py                 # -> data/teacher_train_outputs.pt
  python precompute_teacher_bottleneck.py      # -> data/teacher_train_bottleneck.pt

Output: ./data/teacher_train_bottleneck.pt  (~4.5 GB for 17.5k samples at [256,H',W'])
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
import time
import argparse
import warnings
import torch
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from utils import *
from data_loader import load_data
from modelADT import ModelADT
from joint_model import Enc_2Dec_Network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default="./runs/2026-05-21-06-51-03")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="./data/teacher_train_bottleneck.pt")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = args.data_dir
    trainpath = [f"{d}/dataset_jacobs_July28.mat",
                 f"{d}/dataset_non_fov_train_jacobs_July28_2.mat",
                 f"{d}/dataset_fov_train_jacobs_July28_2.mat"]

    print("Loading training data (same order as precompute_teacher.py)...")
    wos, ws, lbls = [], [], []
    for fp in trainpath:
        wo, w, lbl = load_data(fp)
        if lbl.dim() == 3:
            lbl = lbl.unsqueeze(1)
        wos.append(wo); ws.append(w); lbls.append(lbl)
    train_wo = torch.cat(wos, 0); train_w = torch.cat(ws, 0); train_lbl = torch.cat(lbls, 0)
    print(f"Training samples: {train_w.shape[0]}")

    # Load teacher (mirror precompute_teacher.py).
    from params_storage.params_fig10b_single_gpu import (
        opt_exp, opt_encoder, opt_decoder, opt_offset_decoder)
    for opt in (opt_encoder, opt_decoder, opt_offset_decoder):
        opt.checkpoints_load_dir = args.teacher_dir
    enc = ModelADT(); enc.initialize(opt_encoder); enc.setup(opt_encoder)
    dec = ModelADT(); dec.initialize(opt_decoder); dec.setup(opt_decoder)
    off = ModelADT(); off.initialize(opt_offset_decoder); off.setup(opt_offset_decoder)
    teacher = Enc_2Dec_Network()
    teacher.initialize(opt_exp, enc, dec, off, gpu_ids=opt_exp.gpu_ids)
    for m in (enc, dec, off):
        m.load_networks("best", load_dir=args.teacher_dir)
    teacher.eval()
    print("Teacher loaded. Extracting encoder bottleneck (enc_model.output)...")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_wo, train_w, train_lbl),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i, (wo, w, lbl) in enumerate(loader):
            teacher.set_input(w, lbl, wo, shuffle_channel=False)
            teacher.test()
            feats.append(enc.output.detach().cpu())
            if (i + 1) % 50 == 0:
                print(f"  {(i+1)*args.batch_size}/{train_w.shape[0]}")
    result = torch.cat(feats, 0)
    torch.save(result, args.out)
    print(f"Saved {args.out}: {tuple(result.shape)} in {time.time()-t0:.1f}s "
          f"({os.path.getsize(args.out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
