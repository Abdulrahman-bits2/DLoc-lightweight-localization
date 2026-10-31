#!/usr/bin/python
"""
model_depths.py
----------------------------------------------------------------------
Print the DEPTH (number of learnable weight layers along the network) and the
parameter count of every model used in the project, so the report's depth table
is computed, not guessed.  CPU-only, runs in seconds, no data needed.

"Depth" = count of weight layers (Conv1d / Conv2d / ConvTranspose2d / Linear).
For Study A note: all 13 runs share the SAME depth -- the ablation varies WIDTH
(channels) and SKIP-CONNECTIVITY, never depth.

Usage:  python code/evaluation/model_depths.py
"""
import os as _os, sys as _sys
_CODE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "models", "configs"):
    _p = _os.path.join(_CODE_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import torch.nn as nn

WL = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


def stats(model):
    depth = sum(isinstance(m, WL) for m in model.modules())
    params = sum(p.numel() for p in model.parameters())
    return depth, params


def try_build(name, fn, note=""):
    try:
        depth, params = stats(fn())
        print(f"  {name:34s} {params/1e6:8.3f}M   depth={depth:3d}   {note}")
    except Exception as e:
        print(f"  {name:34s}  [could not build: {type(e).__name__}: {e}]")


def main():
    print("MODEL DEPTHS (weight layers along the network) and parameter counts")
    print("=" * 78)

    # --- OLD architecture: DLoc baseline (ResNet encoder-decoder) ---
    try:
        from Generators import ResnetEncoder, ResnetDecoder

        class DLocBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = ResnetEncoder(input_nc=4, output_nc=256, ngf=64, n_blocks=6)
                self.dec = ResnetDecoder(input_nc=256, output_nc=1, ngf=64,
                                         n_blocks=9, encoder_blocks=6)
        try_build("DLoc baseline (ResNet enc-dec)", DLocBaseline,
                  "OLD arch; 6 enc + 9 dec ResNet blocks")
    except Exception as e:
        print(f"  DLoc baseline                       [import failed: {e}]")

    # --- NEW lightweight students ---
    from student_configurable import build_student
    from student_tinycnn import TinyCNNStudent
    try_build("MobileNetV2 (no-skip / plain)", lambda: build_student(use_skips=False, decoder_width=1.0),
              "NEW; no spatial preservation")
    try_build("MobileNetV2-UNet (skip)", lambda: build_student(use_skips=True, decoder_width=1.0),
              "NEW; +3 skip connections (same depth)")
    try_build("MobileNetV2-UNet + ASPP", lambda: build_student(use_skips=True, decoder_width=1.0, bottleneck='aspp'),
              "NEW; + multi-scale bottleneck")
    try_build("TinyCNN", lambda: TinyCNNStudent(input_nc=4), "NEW; ultra-light")

    try:
        from mamba_model import MambaDLocNet
        try_build("Mamba-DLoc", lambda: MambaDLocNet(input_nc=4),
                  "NEW; 3-conv stem + 4 Mamba blocks + 3-conv dec")
    except Exception as e:
        print(f"  Mamba-DLoc                          [import failed: {e}]")

    print("-" * 78)
    print("Study A width/connectivity variants (depth is CONSTANT across all of these):")
    for ds in (0.5, 1.0, 2.0):
        try_build(f"  skip, decoder_width={ds}", lambda ds=ds: build_student(use_skips=True, decoder_width=ds))
    try_build("  skip, width_mult=0.5", lambda: build_student(use_skips=True, decoder_width=1.0, width_mult=0.5))
    try_build("  skip, width_mult=1.4", lambda: build_student(use_skips=True, decoder_width=1.0, width_mult=1.4))
    print("=" * 78)
    print("Note: width_mult / decoder_width change CHANNELS, not depth -> depth identical.")


if __name__ == "__main__":
    main()
