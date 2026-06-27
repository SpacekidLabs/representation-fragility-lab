"""Exp042 runner: full Mel/Chroma blind-spot atlas.

Reuses src/experiments/exp042_mel_chroma_blind_spots.py for each signal class,
aggregates JSON outputs, and produces a compact summary PNG.
"""
import json, glob, os, re
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/user/Desktop/representation-fragility-lab"
RESULTS = os.path.join(ROOT, "results/exp042_mel_chroma_blind_spots")
FIG = os.path.join(RESULTS, "exp042_summary.png")

SIGNALS = ["sine", "harmonic_stack", "fm_tone", "chirp", "percussive", "noise"]

def load_series(signal, rep):
    pattern = os.path.join(RESULTS, f"exp042_{signal}_snr_sweep_{rep}.json")
    with open(pattern) as f:
        data = json.load(f)
    x = np.array([d["snr_db"] for d in data])
    y = np.array([d["similarity"] for d in data])
    idx = np.argsort(x)
    return x[idx], y[idx]

# Sort key: write positive dB as 100 for ordering by label readability
def plot():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, signal in zip(axes, SIGNALS):
        for rep, color in [("mel", "tab:blue"), ("chroma", "tab:orange")]:
            try:
                x, y = load_series(signal, rep)
                ax.plot(x, y, marker="o", color=color, label=rep)
            except FileNotFoundError:
                pass
        ax.set_title(signal, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        "Exp042 — Mel vs Chroma Blind-Spot Atlas (similarity vs SNR)",
        fontsize=12,
    )
    fig.text(0.5, 0.02, "SNR (dB)", ha="center")
    fig.text(0.02, 0.5, "Representation similarity", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    os.makedirs(RESULTS, exist_ok=True)
    plt.savefig(FIG, dpi=150)
    plt.close()
    print("saved", FIG)

if __name__ == "__main__":
    plot()
