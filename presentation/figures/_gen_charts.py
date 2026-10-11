"""Generate deck-themed charts for the defense presentation.

Reuses authentic per-sample error arrays (Figures/figure_data/*.npy) for the CDF,
and the thesis summary numbers for the rest. Run with the repo .venv python.
Outputs PNGs into this folder (presentation/figures/).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "Figures", "figure_data")

# --- deck theme ---------------------------------------------------------
NAVY, TEAL, ORANGE, SAND, RED, GRAY = (
    "#264653", "#2A9D8F", "#E76F51", "#E9C46A", "#E63946", "#8D99AE")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


# --- 1. Pareto: accuracy vs size ---------------------------------------
def pareto():
    models = [
        ("DLoc (teacher)", 16.5, 0.707, NAVY, "o"),
        ("MobileNetV2", 2.3, 0.707, GRAY, "o"),
        ("TinyCNN", 0.047, 3.922, ORANGE, "s"),
        ("MobileNetV2-UNet", 2.41, 0.640, TEAL, "*"),
        ("Mamba", 0.65, 10.56, RED, "X"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for name, p, m, c, mk in models:
        size = 520 if mk == "*" else 190
        ax.scatter(p, m, s=size, c=c, marker=mk, edgecolor="black",
                   linewidth=1.1, zorder=3)
    # annotations
    offs = {
        "DLoc (teacher)": (10, 8), "MobileNetV2": (-4, 14),
        "TinyCNN": (8, 6), "MobileNetV2-UNet": (12, -4), "Mamba": (-12, -16),
    }
    for name, p, m, c, mk in models:
        dx, dy = offs[name]
        ax.annotate(name, (p, m), textcoords="offset points", xytext=(dx, dy),
                    fontsize=11, fontweight="bold", color="#222222")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Median error (m)")
    ax.set_title("Accuracy vs. model size")
    ax.set_xlim(0.02, 40)
    ax.set_ylim(-0.6, 11.6)
    ax.annotate("better", xy=(0.06, 0.4), xytext=(0.3, 2.6),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2),
                color=TEAL, fontweight="bold", fontsize=12)
    save(fig, "pareto.png")


# --- 2. Cross-session: ours vs DLoc ------------------------------------
def crosssession():
    folds = ["Aug16_1\n(furniture)", "Aug16_3\n(diff. furniture)",
             "Aug16_4_ref\n(reflector)"]
    ours = [0.852, 0.822, 1.116]
    ours_std = [0.044, 0.040, 0.016]
    dloc = [0.71, 0.82, 1.05]
    x = np.arange(len(folds))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w/2, ours, w, yerr=ours_std, capsize=5, color=TEAL,
                edgecolor="black", linewidth=0.8, label="Ours: U-Net (2.41M)")
    b2 = ax.bar(x + w/2, dloc, w, color=NAVY, edgecolor="black",
                linewidth=0.8, label="Published DLoc (16.5M)")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2f}",
                    (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Median error (m)")
    ax.set_title("Cross-session  (leave-one-session-out, 3 seeds)")
    ax.set_ylim(0, 1.4)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "crosssession.png")


# --- 3. Efficiency: DLoc vs U-Net --------------------------------------
def efficiency():
    metrics = ["FLOPs (G)", "CPU latency (ms)"]
    dloc = [81.06, 270.9]
    unet = [1.45, 27.2]
    gains = ["56× less", "10× faster"]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w/2, dloc, w, color=NAVY, edgecolor="black",
                linewidth=0.8, label="DLoc (16.5M)")
    b2 = ax.bar(x + w/2, unet, w, color=TEAL, edgecolor="black",
                linewidth=0.8, label="MobileNetV2-UNet (2.41M)")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.1f}",
                    (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=10, fontweight="bold")
    for xi, g in zip(x, gains):
        ax.annotate(g, (xi, 1.4), ha="center", color=ORANGE,
                    fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("value (log scale)")
    ax.set_title("Computational efficiency  (batch size 1)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "efficiency.png")


# --- 4. Robustness: standalone vs distilled under noise ----------------
def robustness():
    models = ["MobileNetV2\n(2.3M)", "MobileNetV2-UNet\n(2.41M)"]
    standalone = [16.368, 3.000]
    distilled = [4.769, 3.469]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w/2, standalone, w, color=GRAY, edgecolor="black",
                linewidth=0.8, label="Standalone")
    b2 = ax.bar(x + w/2, distilled, w, color=TEAL, edgecolor="black",
                linewidth=0.8, label="Distilled")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.1f}",
                    (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10, fontweight="bold")
    ax.annotate("3.4× from\ndistillation", (0, 11.5), ha="center",
                color=ORANGE, fontsize=11, fontweight="bold")
    ax.annotate("5.5× from\nU-Net skips", (0.0, 6.2), xytext=(0.95, 9.0),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6),
                color=RED, fontsize=11, fontweight="bold", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Median error under noise σ=0.1 (m)")
    ax.set_title("Robustness to Gaussian noise")
    ax.set_ylim(0, 19)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "robustness.png")


# --- 5. In-distribution CDF (authentic per-sample arrays) --------------
def cdf_indist():
    series = [
        ("MobileNetV2-UNet", "errors_unet_fig10b.npy", TEAL),
        ("MobileNetV2", "errors_mobilenet_fig10b.npy", NAVY),
        ("TinyCNN", "errors_tinycnn_fig10b.npy", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ok = False
    for name, fn, c in series:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print("missing", path)
            continue
        e = np.asarray(np.load(path)).ravel()
        e = e[np.isfinite(e)]
        xs = np.sort(e)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=2.4, color=c,
                label=f"{name}  (median {np.median(e):.3f} m)")
        ok = True
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#999999", ls=":", lw=1)
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("Fraction of predictions ≤ error")
    ax.set_title("In-distribution error CDF")
    ax.legend(frameon=False, loc="lower right")
    if ok:
        save(fig, "cdf_indist.png")
    else:
        plt.close(fig)
        print("CDF skipped (no data)")


# --- 6. Scheduler study: U-Net cosine vs step (tail percentiles) -------
def scheduler():
    variants = ["Cosine\nStandalone", "Step\nStandalone", "Cosine\nDistilled"]
    p90 = [1.523, 1.552, 1.513]
    p99 = [2.750, 2.918, 2.587]
    x = np.arange(len(variants))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w/2, p90, w, color=TEAL, edgecolor="black",
                linewidth=0.8, label="P90")
    b2 = ax.bar(x + w/2, p99, w, color=NAVY, edgecolor="black",
                linewidth=0.8, label="P99")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.3f}",
                    (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10, fontweight="bold")
    ax.annotate("median = 0.640 m for all three", (1.0, 0.25),
                ha="center", color=ORANGE, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Tail error (m)")
    ax.set_title("Scheduler study  (U-Net, 120 epochs)")
    ax.set_ylim(0, 3.4)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "scheduler.png")


# --- 7. Multi-seed stability: DLoc baseline median per seed ------------
def multiseed():
    seeds = ["42", "123", "777"]
    median = [0.825, 0.906, 0.894]
    mean, std = 0.875, 0.036
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # mean +/- std band
    ax.axhspan(mean - std, mean + std, color=TEAL, alpha=0.18, zorder=0,
               label=f"mean {mean:.3f} ± {std:.3f} m")
    ax.axhline(mean, color=TEAL, lw=1.6, zorder=1)
    # per-seed medians
    ax.scatter(x, median, s=190, c=NAVY, edgecolor="black", linewidth=1.1,
               zorder=3, label="per-seed median")
    for xi, m in zip(x, median):
        ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    # U-Net reference (far below the baseline band)
    ax.axhline(0.640, color=ORANGE, ls="--", lw=1.6, zorder=2)
    ax.annotate("U-Net 0.640 m  (6.5σ below baseline)", (-0.4, 0.640),
                textcoords="offset points", xytext=(0, 7), ha="left",
                color=ORANGE, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Median error (m)")
    ax.set_title("Multi-seed stability  (DLoc baseline, 3 seeds)")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.58, 0.96)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="#cccccc", loc="lower right")
    save(fig, "multiseed.png")


if __name__ == "__main__":
    pareto()
    crosssession()
    efficiency()
    robustness()
    cdf_indist()
    scheduler()
    multiseed()
    print("done")
