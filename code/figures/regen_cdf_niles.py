#!/usr/bin/env python
"""NILES submission variant of the in-distribution CDF: high-performing models only
(no TinyCNN curve, per professor's table directive), no internal title."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig, PALETTE
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Figures", "figure_data")
OUT = os.path.join(REPO, "NilesConf", "submission", "figures")

SERIES = {
    "errors_unet_fig10b.npy":      ("MobileNetV2-UNet (0.64 m)", PALETTE["unet"]),
    "errors_mobilenet_fig10b.npy": ("MobileNetV2 (0.71 m)",      PALETTE["mobilenet"]),
}

apply_style()
fig, ax = plt.subplots(figsize=(7, 4.3))
for fn, (label, c) in SERIES.items():
    e = np.sort(np.load(os.path.join(DATA, fn)))
    ax.plot(e, np.arange(1, e.size + 1) / e.size, color=c, label=label)
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1)
ax.set_xlabel("Localization error (m)")
ax.set_ylabel("CDF")
ax.legend(loc="lower right")
savefig(fig, OUT, "cdf_fig10b")
plt.close(fig)
print("written to", OUT)
