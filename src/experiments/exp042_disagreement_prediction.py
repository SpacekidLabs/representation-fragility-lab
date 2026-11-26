"""
Experiment 042 — Does Disagreement Predict Blind Spots? (issue #2)
===================================================================
Tests the core principle of docs/VISION.md: *disagreement between
representations is information*.

For thousands of synthetic frames with a known f0, we run STFT, ACF and
Cepstrum side by side (src/framework/disagreement.py) and ask, for each
representation: **can we tell in advance that it is about to make a gross
pitch error (> 50 cents)?**

Predictors compared (all evaluated per representation, ROC AUC, higher = better):

  Baseline (existing framework, descriptor-only state space):
    1. assumption   1 - engine.assumptions[rep]            (no training)
    2. state_model  logistic regression on (z1, z2) poly   (cross-validated)
  Single-lens self-knowledge:
    3. self_conf    1 - the representation's own confidence (no training)
  Cross-representation disagreement (new):
    4. divergence   how much this lens disagrees with the others (no training)
    5. disagree_model  logistic regression on disagreement features (CV)
  Both:
    6. combined     state + disagreement features (CV)

Cross-validation is grouped by base signal (f0, harmonic count) so the model
is always tested on signals it has never seen.

We also check whether comparing lenses gives a better *answer*, not just a
better warning: gross-error rate of each single lens vs. the lens the engine
would route to vs. the disagreement consensus f0.

Outputs:
  results/exp042_disagreement_prediction.png
  results/exp042_summary.json
"""

import json
import os
import sys
import warnings
import zlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

from src.framework.engine import RepresentationIntelligenceEngine
from src.perturbations.filter import lowpass_filter
from scipy.signal import butter, filtfilt

SR = 22050          # the engine's state space was calibrated at 22.05 kHz
FRAME = 2048
DURATION = 0.5
GROSS_CENTS = 50.0
REPS = ("stft", "acf", "cepstrum")
SEEDS = (0, 1, 2)

F0S = (110.0, 165.0, 220.0, 330.0, 440.0, 660.0)
HARMONICS = (1, 2, 3, 5, 10, 20)

OUT_PNG = os.path.join(project_root, "results", "exp042_disagreement_prediction.png")
OUT_JSON = os.path.join(project_root, "results", "exp042_summary.json")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNALS & PERTURBATIONS
# ──────────────────────────────────────────────────────────────────────────────

def harmonic_signal(f0, num_harmonics, rng):
    t = np.arange(int(SR * DURATION)) / SR
    x = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        if k * f0 >= 0.45 * SR:
            break
        x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return x / (np.max(np.abs(x)) + 1e-12)


def add_noise_snr(x, snr_db, rng):
    p = np.mean(x ** 2)
    return x + rng.normal(0.0, np.sqrt(p / 10 ** (snr_db / 10.0)), x.shape)


def highpass(x, cutoff):
    b, a = butter(5, cutoff / (0.5 * SR), btype="high")
    return filtfilt(b, a, x)


def conditions(f0):
    """(family, severity label, function(x, rng) -> x)."""
    out = [("clean", "-", lambda x, r: x)]
    for snr in (20, 10, 5, 0, -5):
        out.append(("noise", f"{snr} dB", lambda x, r, s=snr: add_noise_snr(x, s, r)))
    for mult in (1.5, 2.5, 4.5):
        out.append(("lowpass", f"{mult}xf0", lambda x, r, m=mult: lowpass_filter(x, SR, m * f0)))
    out.append(("missing_fund", "hp 1.5xf0", lambda x, r: highpass(x, 1.5 * f0)))
    for gain in (3.0, 10.0):
        out.append(("clipping", f"x{gain:g}", lambda x, r, g=gain: np.clip(g * x, -1, 1)))
    out.append(("lowpass+noise", "2.5xf0, 5 dB",
                lambda x, r: add_noise_snr(lowpass_filter(x, SR, 2.5 * f0), 5, r)))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# METRICS & MODELS (numpy only)
# ──────────────────────────────────────────────────────────────────────────────

def cents_error(est, truth):
    if est <= 0:
        return np.inf
    return float(abs(1200.0 * np.log2(est / truth)))


def roc_auc(score, label):
    """Mann-Whitney AUC; ties count half. NaN if one class is missing."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):  # average ranks over ties
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_logistic(X, y, l2=1e-2, iters=800, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        w -= lr * grad
    return lambda Xn: 1.0 / (1.0 + np.exp(-(np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w)))


def grouped_cv_scores(X, y, groups, k=5, seed=0):
    """Out-of-fold predicted probabilities, folds split by group."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    out = np.full(len(y), np.nan)
    for fold in folds:
        test = np.isin(groups, fold)
        if y[~test].min() == y[~test].max():
            out[test] = y[~test].mean()
            continue
        out[test] = fit_logistic(X[~test], y[~test])(X[test])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────────────────────────────────────

def collect():
    engine = RepresentationIntelligenceEngine()
    rows = []
    for f0 in F0S:
        for nh in HARMONICS:
            for family, severity, fn in conditions(f0):
                for seed in SEEDS:
                    rng = np.random.default_rng(zlib.crc32(repr((f0, nh, family, severity, seed)).encode()))
                    x = fn(harmonic_signal(f0, nh, rng), rng)
                    start = (len(x) - FRAME) // 2 + int(rng.integers(-500, 500))
                    frame = x[start:start + FRAME]
                    state = engine.analyze(frame, SR)
                    rep = state.disagreement
                    rows.append({
                        "f0": f0, "harmonics": nh, "family": family, "severity": severity,
                        "seed": seed, "group": f"{f0:g}-{nh}",
                        "z1": state.coordinate[0], "z2": state.coordinate[1],
                        "region": state.region,
                        "assumptions": {r: state.assumptions[r] for r in REPS},
                        "readings": {r: (rep.readings[r].f0, rep.readings[r].confidence) for r in REPS},
                        "divergence": dict(rep.divergence),
                        "overall": rep.overall,
                        "consensus_f0": rep.consensus_f0,
                        "pairs": [(p.a, p.b, p.pitch_term, p.confidence_gap, p.octave_related)
                                  for p in rep.pairs],
                    })
    return rows


def features(row, rep):
    """Disagreement features for one representation in one frame."""
    others = [o for o in REPS if o != rep]
    my_pairs = [p for p in row["pairs"] if rep in (p[0], p[1])]
    return [
        row["divergence"][rep],
        row["overall"],
        row["readings"][rep][1],
        np.mean([row["readings"][o][1] for o in others]),
        min(p[2] for p in my_pairs),                     # agreement with the closest other lens
        float(any(p[4] for p in my_pairs)),              # octave relation with any other lens
    ]


def state_features(row):
    z1, z2 = row["z1"], row["z2"]
    return [z1, z2, z1 * z1, z2 * z2, z1 * z2]


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

PREDICTORS = ("assumption", "state_model", "self_conf", "divergence", "disagree_model", "combined")


def analyze(rows):
    groups = np.array([r["group"] for r in rows])
    summary = {"n_frames": len(rows), "gross_cents": GROSS_CENTS, "per_rep": {}, "fusion": {},
               "by_family": {}}
    scores_for_plot = {}

    for rep in REPS:
        err = np.array([cents_error(r["readings"][rep][0], r["f0"]) for r in rows])
        y = (err > GROSS_CENTS).astype(float)
        Xs = np.array([state_features(r) for r in rows])
        Xd = np.array([features(r, rep) for r in rows])
        preds = {
            "assumption": np.array([1.0 - r["assumptions"][rep] for r in rows]),
            "state_model": grouped_cv_scores(Xs, y, groups),
            "self_conf": np.array([1.0 - r["readings"][rep][1] for r in rows]),
            "divergence": Xd[:, 0],
            "disagree_model": grouped_cv_scores(Xd, y, groups),
            "combined": grouped_cv_scores(np.c_[Xs, Xd], y, groups),
        }
        finite = np.isfinite(err)
        summary["per_rep"][rep] = {
            "gross_error_rate": float(y.mean()),
            "auc": {k: roc_auc(v, y) for k, v in preds.items()},
            "spearman_divergence_vs_log_error": spearman(
                Xd[finite, 0], np.log1p(err[finite])),
        }
        scores_for_plot[rep] = (Xd[:, 0], np.minimum(err, 2400.0), y)

    # Per perturbation family: AUC of divergence vs. state baseline, pooled over reps.
    for fam in sorted({r["family"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        lab, div, ass = [], [], []
        for rep in REPS:
            for i in idx:
                e = cents_error(rows[i]["readings"][rep][0], rows[i]["f0"])
                lab.append(e > GROSS_CENTS)
                div.append(rows[i]["divergence"][rep])
                ass.append(1.0 - rows[i]["assumptions"][rep])
        summary["by_family"][fam] = {
            "gross_error_rate": float(np.mean(lab)),
            "auc_divergence": roc_auc(div, lab),
            "auc_assumption": roc_auc(ass, lab),
        }

    # Fusion: does comparing lenses give a better answer?
    def rate(estimates):
        errs = np.array([cents_error(e, r["f0"]) for e, r in zip(estimates, rows)])
        answered = np.array([e > 0 for e in estimates])
        return {
            "gross_error_rate": float(np.mean(errs > GROSS_CENTS)),
            "abstain_rate": float(1.0 - answered.mean()),
            "gross_error_rate_when_answering": float(np.mean(errs[answered] > GROSS_CENTS))
            if answered.any() else float("nan"),
        }

    for rep in REPS:
        summary["fusion"][rep] = rate([r["readings"][rep][0] for r in rows])
    summary["fusion"]["engine_routed"] = rate([
        r["readings"][max(REPS, key=lambda k: r["assumptions"][k])][0] for r in rows])
    summary["fusion"]["consensus"] = rate([r["consensus_f0"] for r in rows])
    summary["fusion"]["oracle_best_single"] = rate([
        min((r["readings"][k][0] for k in REPS), key=lambda e: cents_error(e, r["f0"]))
        for r in rows])
    return summary, scores_for_plot


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot(summary, scores):
    colors = {"stft": "#2a78d6", "acf": "#e0832a", "cepstrum": "#3a9e5c"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    w = 0.26
    xs = np.arange(len(PREDICTORS))
    for i, rep in enumerate(REPS):
        vals = [summary["per_rep"][rep]["auc"][p] for p in PREDICTORS]
        ax.bar(xs + (i - 1) * w, vals, w, label=rep, color=colors[rep])
    ax.axhline(0.5, color="#888", lw=1, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(PREDICTORS, rotation=25, ha="right")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("ROC AUC: predicting gross pitch error (>50 cents)")
    ax.set_title("Which signal warns that a lens is blind?")
    ax.legend(frameon=False)

    ax = axes[1]
    for rep in REPS:
        div, err, _ = scores[rep]
        ax.scatter(div, err, s=6, alpha=0.35, color=colors[rep], label=rep)
    ax.set_yscale("symlog", linthresh=10)
    ax.axhline(GROSS_CENTS, color="#888", lw=1, ls="--")
    ax.set_xlabel("divergence (disagreement with the other lenses)")
    ax.set_ylabel("|pitch error| (cents, capped at 2400)")
    ax.set_title("Disagreement vs. actual error")
    ax.legend(frameon=False, markerscale=3)

    ax = axes[2]
    keys = ["stft", "acf", "cepstrum", "engine_routed", "consensus", "oracle_best_single"]
    gross = [summary["fusion"][k]["gross_error_rate_when_answering"] for k in keys]
    abst = [summary["fusion"][k]["abstain_rate"] for k in keys]
    ax.bar(keys, gross, color=["#2a78d6", "#e0832a", "#3a9e5c", "#8a8a8a", "#7b4fc9", "#cccccc"])
    for i, (g, a) in enumerate(zip(gross, abst)):
        ax.text(i, g + 0.005, f"{g:.1%}" + (f"\nabstain {a:.0%}" if a > 0 else ""),
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("gross error rate (when answering)")
    ax.set_title("Single lens vs. routed vs. consensus")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle(f"Exp 042 — Does disagreement predict blind spots? ({summary['n_frames']} frames)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)


def main():
    rows = collect()
    summary, scores = analyze(rows)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    plot(summary, scores)

    print(f"\nExp 042 — {summary['n_frames']} frames, gross error = >{GROSS_CENTS:g} cents\n")
    print(f"{'rep':10s} {'err rate':>8s} " + " ".join(f"{p:>14s}" for p in PREDICTORS))
    for rep in REPS:
        d = summary["per_rep"][rep]
        print(f"{rep:10s} {d['gross_error_rate']:8.1%} " +
              " ".join(f"{d['auc'][p]:14.3f}" for p in PREDICTORS))
    print("\nBy perturbation family (pooled over lenses): AUC divergence vs assumption")
    for fam, d in summary["by_family"].items():
        print(f"  {fam:15s} err {d['gross_error_rate']:6.1%}  div {d['auc_divergence']:.3f}"
              f"  assumption {d['auc_assumption']:.3f}")
    print("\nAnswer quality (gross error rate when answering / abstain rate)")
    for k, d in summary["fusion"].items():
        print(f"  {k:20s} {d['gross_error_rate_when_answering']:6.1%}  abstain {d['abstain_rate']:5.1%}")
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
