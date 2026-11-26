"""
Experiment 043 — Patterns in Disagreement (issue #3)
====================================================
Exp 042 showed that disagreement between representations warns when a lens is
blind. This experiment asks the next question from docs/VISION.md: **does the
disagreement itself have structure?** Which lenses fail together, in which
direction, and under which conditions?

Disagreement vector (per frame, 6 numbers, all inspectable):
  - confidence of STFT, ACF, Cepstrum                         (0..1)
  - signed pitch ratio of each pair, log2(f_a / f_b), clipped to ±1.5
      stft/acf, stft/cepstrum, acf/cepstrum                   (octaves; +1 = a is an octave above b)

Steps:
  1. Cluster the vectors (k-means, k chosen by silhouette) and give every
     cluster a rule-based, human-readable name from its centroid.
  2. Check each pattern against the truth: which lenses are actually wrong
     in it, and which conditions / signal types produce it.
  3. Co-failure: which lenses fail together more (or less) than chance.
  4. Compare with the existing state map (engine regions / z1, z2):
       - how much each partition tells us about the true failure pattern
         (normalised mutual information, NMI)
       - whether the disagreement pattern can be predicted from (z1, z2)

Data: the 3,240 pitched frames of Exp 042 plus its 90 white-noise frames.
Note: disagreement is computed from the lenses' outputs, the state map from
the input signal only; neither uses the true pitch.

Outputs:
  results/exp043_disagreement_patterns.png
  results/exp043_summary.json
"""

import json
import os
import sys
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(here, "../.."))
for p in (project_root, here):
    if p not in sys.path:
        sys.path.append(p)

warnings.filterwarnings("ignore")

import exp042_disagreement_prediction as e42

REPS = e42.REPS
PAIRS = list(combinations(REPS, 2))           # (stft, acf), (stft, cepstrum), (acf, cepstrum)
GROSS = e42.GROSS_CENTS
K_RANGE = range(3, 11)
RATIO_CLIP = 1.5
SHORT = {"stft": "STFT", "acf": "ACF", "cepstrum": "CEP"}

OUT_PNG = os.path.join(project_root, "results", "exp043_disagreement_patterns.png")
OUT_JSON = os.path.join(project_root, "results", "exp043_summary.json")


# ──────────────────────────────────────────────────────────────────────────────
# DISAGREEMENT VECTORS
# ──────────────────────────────────────────────────────────────────────────────

def log_ratio(fa, fb):
    if fa <= 0 or fb <= 0:
        return 0.0
    return float(np.clip(np.log2(fa / fb), -RATIO_CLIP, RATIO_CLIP))


def vector(row):
    conf = [row["readings"][r][1] for r in REPS]
    ratios = [log_ratio(row["readings"][a][0], row["readings"][b][0]) for a, b in PAIRS]
    return conf + ratios


FEATURE_NAMES = [f"conf {SHORT[r]}" for r in REPS] + [f"{SHORT[a]}/{SHORT[b]} (oct)" for a, b in PAIRS]


def failure_label(row):
    """Which lenses are actually wrong (> 50 cents): e.g. 'CEP', 'ACF+CEP', 'none'."""
    if row["truth"] <= 0:
        return "unpitched"
    wrong = [SHORT[r] for r in REPS if e42.cents_error(row["readings"][r][0], row["truth"]) > GROSS]
    return "+".join(wrong) if wrong else "none"


# ──────────────────────────────────────────────────────────────────────────────
# CLUSTERING (numpy only)
# ──────────────────────────────────────────────────────────────────────────────

def kmeans(X, k, restarts=10, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(restarts):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):  # k-means++
            d = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(centers)
        for _ in range(iters):
            labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[labels]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, labels, C)
    return best[1], best[2]


def silhouette(X, labels, max_n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    D = np.sqrt(((Xs[:, None, :] - Xs[None]) ** 2).sum(-1))
    s = np.zeros(len(Xs))
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, ls == c].mean() for c in np.unique(ls) if c != ls[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def nmi(a, b):
    """Normalised mutual information (arithmetic mean normalisation)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub)))
    np.add.at(joint, (ia, ib), 1)
    pj = joint / n
    pa, pb = pj.sum(1), pj.sum(0)
    nz = pj > 0
    mi = (pj[nz] * np.log(pj[nz] / (pa[:, None] * pb[None])[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb)) if ha + hb > 0 else 0.0


def knn_cv_accuracy(X, y, k=15, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    correct = 0
    for f in np.array_split(order, folds):
        train = np.setdiff1d(order, f)
        D = ((X[f][:, None, :] - X[train][None]) ** 2).sum(-1)
        nn = np.argsort(D, axis=1)[:, :k]
        for i, row in enumerate(nn):
            correct += Counter(y[train][row]).most_common(1)[0][0] == y[f][i]
    return correct / len(X)


# ──────────────────────────────────────────────────────────────────────────────
# NAMING (rule-based, from the cluster's median vector)
# ──────────────────────────────────────────────────────────────────────────────

def relation(r):
    if abs(r) < 0.15:
        return "agree"
    if abs(r - 1) < 0.3:
        return "octave above"
    if abs(r + 1) < 0.3:
        return "octave below"
    return "off"


def name_pattern(med):
    conf = dict(zip(REPS, med[:3]))
    ratio = {pair: med[3 + i] for i, pair in enumerate(PAIRS)}
    confident = [r for r in REPS if conf[r] >= 0.5]

    def rel(a, b):  # relation of a relative to b
        return relation(ratio[(a, b)] if (a, b) in ratio else -ratio[(b, a)])

    if all(rel(a, b) == "agree" for a, b in PAIRS):
        if len(confident) == 3:
            return "all agree, all confident"
        if not confident:
            return "all agree, none confident"
        weak = [SHORT[r] for r in REPS if r not in confident]
        return f"all agree, {'+'.join(weak)} unsure"
    if not confident:
        return "scattered, none confident"
    for odd in REPS:
        others = [r for r in REPS if r != odd]
        if rel(*others) == "agree" and rel(odd, others[0]) != "agree":
            r = rel(odd, others[0])
            how = {"octave above": "an octave high", "octave below": "an octave low"}.get(r, "off-pitch")
            sure = "confident" if odd in confident else "unsure"
            return f"{SHORT[odd]} {how} ({sure}), others agree"
    return "three-way split"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze(rows):
    X_raw = np.array([vector(r) for r in rows])
    mu, sd = X_raw.mean(0), X_raw.std(0) + 1e-9
    X = (X_raw - mu) / sd

    sil = {}
    fits = {}
    for k in K_RANGE:
        labels, _ = kmeans(X, k)
        fits[k] = labels
        sil[k] = silhouette(X, labels)
    k_best = max(sil, key=sil.get)
    labels = fits[k_best]

    pitched = np.array([r["truth"] > 0 for r in rows])
    fail = np.array([failure_label(r) for r in rows])
    families = np.array([r.get("family", "white_noise") for r in rows])
    signals = np.array([r.get("signal", "white_noise") for r in rows])
    regions = np.array([r["region"] for r in rows])
    Z = np.array([[r["z1"], r["z2"]] for r in rows])

    # Name clusters; merge clusters that get the same name.
    raw_names = {}
    for c in range(k_best):
        raw_names[c] = name_pattern(np.median(X_raw[labels == c], axis=0))
    pattern = np.array([raw_names[c] for c in labels])
    # Purity: share of a pattern's frames whose own vector earns the same name.
    own_name = np.array([name_pattern(v) for v in X_raw])

    patterns = []
    for name in sorted(set(pattern), key=lambda n: -np.sum(pattern == n)):
        m = pattern == name
        mp = m & pitched
        err = {}
        for r in REPS:
            e = [e42.cents_error(rows[i]["readings"][r][0], rows[i]["truth"]) for i in np.where(mp)[0]]
            err[SHORT[r]] = float(np.mean(np.array(e) > GROSS)) if e else float("nan")
        patterns.append({
            "name": name,
            "n_frames": int(m.sum()),
            "share": float(m.mean()),
            "purity": float(np.mean(own_name[m] == name)),
            "median_vector": dict(zip(FEATURE_NAMES, np.round(np.median(X_raw[m], axis=0), 3).tolist())),
            "gross_error_rate": err,
            "top_true_failures": Counter(fail[m]).most_common(3),
            "top_conditions": Counter(families[m]).most_common(4),
            "top_signals": Counter(signals[m]).most_common(3),
            "top_regions": Counter(regions[m]).most_common(3),
            "consensus_gross_error_when_answering": _consensus_rate(rows, np.where(mp)[0]),
        })

    # Co-failure on pitched frames: P(A & B wrong) vs. P(A) P(B).
    wrong = {r: np.array([e42.cents_error(rows[i]["readings"][r][0], rows[i]["truth"]) > GROSS
                          for i in np.where(pitched)[0]]) for r in REPS}
    co = {}
    fam_p = families[pitched]
    for a, b in PAIRS:
        both = float(np.mean(wrong[a] & wrong[b]))
        indep = float(np.mean(wrong[a]) * np.mean(wrong[b]))
        by_family = {}
        for fam in sorted(set(fam_p)):
            m = fam_p == fam
            pa, pb = np.mean(wrong[a][m]), np.mean(wrong[b][m])
            by_family[fam] = {"both_wrong": float(np.mean(wrong[a][m] & wrong[b][m])),
                              "lift": float(np.mean(wrong[a][m] & wrong[b][m]) / (pa * pb)) if pa * pb > 0 else None}
        co[f"{SHORT[a]}+{SHORT[b]}"] = {"both_wrong": both, "if_independent": indep,
                                        "lift": both / indep if indep > 0 else None,
                                        "by_family": by_family}
    all_three = float(np.mean(wrong["stft"] & wrong["acf"] & wrong["cepstrum"]))

    # Relation to the state map.
    z_labels, _ = kmeans((Z - Z.mean(0)) / Z.std(0), k_best)
    comparison = {
        "nmi_true_failure_vs_disagreement_patterns": nmi(fail, pattern),
        "nmi_true_failure_vs_disagreement_clusters": nmi(fail, labels),
        "nmi_true_failure_vs_engine_regions": nmi(fail, regions),
        "nmi_true_failure_vs_state_kmeans_same_k": nmi(fail, z_labels),
        "nmi_disagreement_patterns_vs_engine_regions": nmi(pattern, regions),
        "pattern_from_z1z2_knn_accuracy": knn_cv_accuracy((Z - Z.mean(0)) / Z.std(0), pattern),
        "pattern_majority_baseline": float(Counter(pattern).most_common(1)[0][1] / len(pattern)),
    }

    # PCA of the disagreement vectors for the plot.
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    emb = X @ Vt[:2].T
    var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())

    summary = {
        "n_frames": len(rows), "n_pitched": int(pitched.sum()),
        "features": FEATURE_NAMES,
        "silhouette_by_k": sil, "k": k_best,
        "pca_variance_explained_2d": var2,
        "patterns": patterns,
        "co_failure": co, "all_three_wrong": all_three,
        "state_map_comparison": comparison,
    }
    plot_data = {"emb": emb, "Z": Z, "pattern": pattern, "families": families, "X_raw": X_raw}
    return summary, plot_data


def _consensus_rate(rows, idx):
    answered = [i for i in idx if rows[i]["consensus_f0"] > 0]
    if not answered:
        return None
    return float(np.mean([e42.cents_error(rows[i]["consensus_f0"], rows[i]["truth"]) > GROSS for i in answered]))


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot(summary, d):
    names = [p["name"] for p in summary["patterns"]]
    cmap = plt.get_cmap("tab10")
    color = {n: cmap(i % 10) for i, n in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    ax = axes[0, 0]
    for n in names[::-1]:
        m = d["pattern"] == n
        ax.scatter(d["emb"][m, 0], d["emb"][m, 1], s=6, alpha=0.5, color=color[n], label=n)
    ax.set_title(f"Disagreement map (PCA of disagreement vectors, "
                 f"{summary['pca_variance_explained_2d']:.0%} variance)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=3, frameon=False, loc="best")

    ax = axes[0, 1]
    for n in names[::-1]:
        m = d["pattern"] == n
        ax.scatter(d["Z"][m, 0], d["Z"][m, 1], s=6, alpha=0.5, color=color[n])
    c = summary["state_map_comparison"]
    ax.set_title("Same frames on the existing state map (z1, z2)\n"
                 f"pattern predictable from z1,z2: {c['pattern_from_z1z2_knn_accuracy']:.0%} "
                 f"(majority {c['pattern_majority_baseline']:.0%})")
    ax.set_xlabel("z1 (order ↔ disorder)")
    ax.set_ylabel("z2 (harmonic ↔ transient)")

    ax = axes[1, 0]
    cols = FEATURE_NAMES + [f"wrong {SHORT[r]}" for r in REPS]
    M = np.array([[p["median_vector"][f] for f in FEATURE_NAMES] +
                  [p["gross_error_rate"][SHORT[r]] for r in REPS] for p in summary["patterns"]], float)
    im = ax.imshow(np.nan_to_num(M), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({p['share']:.0%}, purity {p['purity']:.0%})"
                        for n, p in zip(names, summary["patterns"])], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "–" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.axvline(len(FEATURE_NAMES) - 0.5, color="k", lw=1)
    ax.set_title("Pattern profiles: median disagreement vector | true gross-error rate per lens")
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[1, 1]
    fams = sorted(set(d["families"]))
    C = np.array([[np.mean(d["pattern"][d["families"] == f] == n) for f in fams] for n in names])
    ax.imshow(C, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] >= 0.005:
                ax.text(j, i, f"{C[i, j]:.0%}", ha="center", va="center", fontsize=7)
    ax.set_title("Which conditions produce which pattern\n(share of each condition's frames)")

    fig.suptitle(f"Exp 043 — Patterns in disagreement ({summary['n_frames']} frames, k = {summary['k']})")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)


def main():
    rows = e42.collect() + e42.collect_unpitched()
    summary, d = analyze(rows)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    plot(summary, d)

    print(f"\nExp 043 — {summary['n_frames']} frames; silhouette by k: "
          + ", ".join(f"{k}:{v:.3f}" for k, v in summary["silhouette_by_k"].items())
          + f" → k = {summary['k']}\n")
    for p in summary["patterns"]:
        errs = ", ".join(f"{k} {v:.0%}" for k, v in p["gross_error_rate"].items() if v == v)
        print(f"[{p['share']:5.1%}] {p['name']}   (purity {p['purity']:.0%})")
        print(f"        actually wrong: {errs}")
        print(f"        true failures: {p['top_true_failures']}")
        print(f"        conditions: {p['top_conditions']}")
        print(f"        signals: {p['top_signals']}  regions: {p['top_regions']}")
        print(f"        consensus gross error when answering: {p['consensus_gross_error_when_answering']}")
    print("\nCo-failure (pitched frames): both wrong vs. if independent")
    for k, v in summary["co_failure"].items():
        print(f"  {k:10s} {v['both_wrong']:6.2%} vs {v['if_independent']:6.2%}  lift {v['lift']}")
    print(f"  all three wrong: {summary['all_three_wrong']:.2%}")
    print("\nRelation to the existing state map")
    for k, v in summary["state_map_comparison"].items():
        print(f"  {k:48s} {v:.3f}")
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
