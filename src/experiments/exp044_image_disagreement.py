"""
Experiment 044 — Images: Blind Spots, Disagreement and Patterns (issue #6)
=========================================================================
The audio arc (Exp 001–013 atlas → Exp 042 disagreement → Exp 043 patterns) in
miniature, on a second medium, with the same core code (src/core). The only
image-specific code is src/image_medium: a Medium, five orientation lenses and
an orientation distance.

Quantity: the dominant orientation of a 64×64 grayscale patch, in degrees mod 180.
Gross error: more than 10° from the truth.

Lenses (src/image_medium):
  tensor  structure tensor (mean of squared gradients)
  hog     mode of the gradient-orientation histogram
  fft     strongest 2-D power-spectrum peak
  gabor   oriented Gabor (wavelet) bank energy, wavelengths 4–16 px
  radon   variance of line sums per angle

Signals with known orientation: gratings (period 6 and 12 px), step edges,
thin lines (width 1.5 and 3 px), oriented textures (random gratings within ±5°)
and two-orientation mixtures are not used; the truth is always one orientation.

Perturbations: Gaussian noise, Gaussian blur, JPEG-like block DCT quantisation,
low contrast with 8-bit quantisation, occlusion by a flat rectangle, and a
weaker second pattern at another orientation.

Questions:
  1. Atlas: does each lens have its own, structured blind spots?
  2. Prediction: does disagreement predict when a lens is wrong (ROC AUC,
     grouped cross-validation), better than the lens's own confidence?
  3. Answer: is the consensus better than any single lens?
  4. Patterns: which lenses fail together (co-failure lift), and do clusters
     of disagreement vectors line up with what actually went wrong?
  5. Unoriented input (isotropic noise): how often does a lens, or the
     consensus, confidently report an orientation that is not there?

Usage:
  python src/experiments/exp044_image_disagreement.py              # run
  python src/experiments/exp044_image_disagreement.py --calibrate  # print lens calibration

Outputs:
  results/exp044_image_disagreement.png
  results/exp044_summary.json
"""

import json
import os
import sys
import warnings
import zlib
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (project_root, os.path.join(project_root, "src")):
    if p not in sys.path:
        sys.path.append(p)

warnings.filterwarnings("ignore")

from image_medium import ALL_LENSES, LENSES, angle_diff, circular_weighted_median, make_image_analyzer  # noqa: E402

N = 64
GROSS_DEG = 10.0
ORIENTATIONS = 12                     # base orientations, 15° apart, each with a random jitter
SEEDS = (0, 1)
UNORIENTED_PATCHES = 90
REPS = ALL_LENSES
PAIRS = list(combinations(REPS, 2))
K_RANGE = range(3, 11)

OUT_PNG = os.path.join(project_root, "results", "exp044_image_disagreement.png")
OUT_JSON = os.path.join(project_root, "results", "exp044_summary.json")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNALS (orientation θ in degrees: the direction the structure runs along,
# in (x = column, y = row) coordinates)
# ──────────────────────────────────────────────────────────────────────────────

_Y, _X = np.mgrid[0:N, 0:N].astype(float)
_X -= (N - 1) / 2.0
_Y -= (N - 1) / 2.0


def _across(theta):
    """Coordinate across the structure (along its normal)."""
    t = np.radians(theta)
    return -_X * np.sin(t) + _Y * np.cos(t)


def grating(theta, period, rng):
    return 0.5 + 0.4 * np.sin(2 * np.pi * _across(theta) / period + rng.uniform(0, 2 * np.pi))


def edge(theta, rng):
    off = rng.uniform(-8, 8)
    return 0.3 + 0.4 * (1.0 / (1.0 + np.exp(-(_across(theta) - off) / 0.7)))


def line(theta, width, rng):
    off = rng.uniform(-8, 8)
    return 0.3 + 0.5 * np.exp(-0.5 * ((_across(theta) - off) / width) ** 2)


def texture(theta, rng):
    x = np.zeros((N, N))
    for _ in range(12):
        x += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * _across(theta + rng.uniform(-5, 5)) / rng.uniform(4, 16)
                                            + rng.uniform(0, 2 * np.pi))
    return 0.5 + 0.4 * x / np.max(np.abs(x))


VARIANTS = {
    "grating6": lambda th, rng: grating(th, 6.0, rng),
    "grating12": lambda th, rng: grating(th, 12.0, rng),
    "edge": edge,
    "line1.5": lambda th, rng: line(th, 1.5, rng),
    "line3": lambda th, rng: line(th, 3.0, rng),
    "texture": texture,
}
SIGNAL_TYPE = {"grating6": "grating", "grating12": "grating", "edge": "edge",
               "line1.5": "line", "line3": "line", "texture": "texture"}


def isotropic(rng, kind):
    if kind == "white":
        return np.clip(0.5 + 0.2 * rng.normal(size=(N, N)), 0, 1)
    blob = ndimage.gaussian_filter(rng.normal(size=(N, N)), 2.0)
    return 0.5 + 0.3 * blob / np.max(np.abs(blob))


# ──────────────────────────────────────────────────────────────────────────────
# PERTURBATIONS (none of them changes the true orientation)
# ──────────────────────────────────────────────────────────────────────────────

_JPEG_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], float)


def jpeg_like(x, quality):
    """8×8 block DCT quantisation with the standard luminance table (IJG quality scaling)."""
    s = 5000.0 / quality if quality < 50 else 200.0 - 2 * quality
    q = np.maximum(np.floor((_JPEG_Q * s + 50) / 100), 1)
    img = np.round(np.clip(x, 0, 1) * 255) - 128
    out = np.empty_like(img)
    for i in range(0, N, 8):
        for j in range(0, N, 8):
            c = dctn(img[i:i + 8, j:j + 8], norm="ortho")
            out[i:i + 8, j:j + 8] = idctn(np.round(c / q) * q, norm="ortho")
    return np.clip(out + 128, 0, 255) / 255.0


def occlude(x, fraction, rng):
    side_h = int(round(N * np.sqrt(fraction) * rng.uniform(0.8, 1.25)))
    side_w = int(round(N * N * fraction / max(side_h, 1)))
    side_h, side_w = min(side_h, N), min(side_w, N)
    i, j = rng.integers(0, N - side_h + 1), rng.integers(0, N - side_w + 1)
    y = x.copy()
    y[i:i + side_h, j:j + side_w] = rng.uniform(0.1, 0.9)
    return y


def low_contrast(x, factor):
    return np.round((0.5 + factor * (x - 0.5)) * 255) / 255.0


def second_pattern(x, theta, strength, rng):
    """Add a grating at another orientation whose standard deviation is ``strength``
    times the base pattern's, so the true pattern always has more variance."""
    other = theta + rng.uniform(30, 90) * rng.choice([-1, 1])
    g = grating(other, rng.uniform(5, 12), rng) - 0.5
    return x + strength * np.std(x) / np.std(g) * g


CONDITIONS = [
    ("clean", 0, lambda x, th, rng: x),
    ("noise", 0.1, lambda x, th, rng: x + rng.normal(0, 0.1, x.shape)),
    ("noise", 0.3, lambda x, th, rng: x + rng.normal(0, 0.3, x.shape)),
    ("noise", 0.6, lambda x, th, rng: x + rng.normal(0, 0.6, x.shape)),
    ("blur", 1.0, lambda x, th, rng: ndimage.gaussian_filter(x, 1.0)),
    ("blur", 2.5, lambda x, th, rng: ndimage.gaussian_filter(x, 2.5)),
    ("blur", 4.0, lambda x, th, rng: ndimage.gaussian_filter(x, 4.0)),
    ("jpeg", 50, lambda x, th, rng: jpeg_like(x, 50)),
    ("jpeg", 20, lambda x, th, rng: jpeg_like(x, 20)),
    ("jpeg", 5, lambda x, th, rng: jpeg_like(x, 5)),
    ("contrast", 0.05, lambda x, th, rng: low_contrast(x, 0.05)),
    ("contrast", 0.01, lambda x, th, rng: low_contrast(x, 0.01)),
    ("occlusion", 0.25, lambda x, th, rng: occlude(x, 0.25, rng)),
    ("occlusion", 0.5, lambda x, th, rng: occlude(x, 0.5, rng)),
    ("second", 0.4, lambda x, th, rng: second_pattern(x, th, 0.4, rng)),
    ("second", 0.8, lambda x, th, rng: second_pattern(x, th, 0.8, rng)),
]


def base_signals():
    """(variant, orientation index, theta, rng-seeded patch maker)."""
    for variant in VARIANTS:
        for k in range(ORIENTATIONS):
            rng = np.random.default_rng(zlib.crc32(f"theta-{variant}-{k}".encode()))
            yield variant, k, (15.0 * k + rng.uniform(-7.5, 7.5)) % 180.0


def make_patch(variant, theta, family, severity, seed, k):
    rng = np.random.default_rng(zlib.crc32(repr((variant, k, family, severity, seed)).encode()))
    x = VARIANTS[variant](theta, rng)
    fn = next(f for fam, sev, f in CONDITIONS if fam == family and sev == severity)
    return fn(x, theta, rng)


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION (lens confidence: raw contrast → 0..1)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate():
    """lo = strongest raw contrast over 60 white-noise patches, hi = 5th percentile
    over all clean base patterns. Same rule as the audio CQT and wavelet lenses."""
    lenses = {n: cls() for n, cls in LENSES.items()}
    noise = {n: [] for n in lenses}
    clean = {n: [] for n in lenses}
    for s in range(60):
        x = isotropic(np.random.default_rng(50_000 + s), "white")
        for n, lens in lenses.items():
            noise[n].append(lens.orientation(x)[1])
    for variant, k, theta in base_signals():
        x = make_patch(variant, theta, "clean", 0, 0, k)
        for n, lens in lenses.items():
            clean[n].append(lens.orientation(x)[1])
    out = {n: (round(float(np.max(noise[n])), 3), round(float(np.percentile(clean[n], 5)), 3)) for n in lenses}
    for n, (lo, hi) in out.items():
        print(f'    "{n}": ({lo}, {hi}),')
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────────────────────────────────────

def patch_row(analyzer, patch, truth, meta):
    analysis = analyzer.analyze(patch)
    if analysis.empty:  # flat patch (e.g. contrast below one 8-bit level): nothing to compare
        return {
            **meta, "truth": truth, "empty": True,
            "readings": {r: (None, 0.0) for r in REPS}, "trusted": {r: False for r in REPS},
            "divergence": {r: 0.0 for r in REPS}, "overall": 0.0, "outlier": None, "consensus": None,
            "pairs": {p: (0.0, "both empty", 0.0) for p in PAIRS},
        }
    rep = analysis.report
    return {
        "empty": False,
        **meta, "truth": truth,
        "readings": {r: (rep.readings[r].estimate, rep.readings[r].confidence) for r in REPS},
        "trusted": {r: not rep.assessments[r].blind for r in REPS},
        "divergence": dict(rep.divergence),
        "overall": rep.overall,
        "outlier": rep.outlier,
        "consensus": rep.consensus,
        "pairs": {(p.a, p.b): (p.distance, p.relation, p.details.get("signed_degrees", 0.0)) for p in rep.pairs},
    }


def collect():
    analyzer = make_image_analyzer()
    rows = []
    for variant, k, theta in base_signals():
        for family, severity, _ in CONDITIONS:
            for seed in SEEDS:
                x = make_patch(variant, theta, family, severity, seed, k)
                rows.append(patch_row(analyzer, x, theta, {
                    "variant": variant, "signal": SIGNAL_TYPE[variant], "family": family,
                    "severity": severity, "seed": seed, "group": f"{variant}-{k}"}))
    return rows


def collect_unoriented():
    analyzer = make_image_analyzer()
    rows = []
    for s in range(UNORIENTED_PATCHES):
        kind = "white" if s % 2 == 0 else "blob"
        rows.append(patch_row(analyzer, isotropic(np.random.default_rng(20_000 + s), kind), None, {"kind": kind}))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def err_deg(est, truth):
    return 90.0 if est is None else abs(angle_diff(est, truth))


def wrong(row, rep):
    return err_deg(row["readings"][rep][0], row["truth"]) > GROSS_DEG


def features(row, rep):
    """Disagreement features for one lens in one patch (the Exp 042 set)."""
    others = [o for o in REPS if o != rep]
    mine = [v for (a, b), v in row["pairs"].items() if rep in (a, b)]
    return [
        row["divergence"][rep],
        row["overall"],
        row["readings"][rep][1],
        np.mean([row["readings"][o][1] for o in others]),
        min(v[0] for v in mine),                          # distance to the closest other lens
        float(sum(v[1] == "agree" for v in mine)),        # how many lenses agree with it
    ]


def vector(row):
    """Disagreement vector for pattern discovery: confidences + signed pair differences / 90°."""
    return ([row["readings"][r][1] for r in REPS]
            + [row["pairs"][p][2] / 90.0 for p in PAIRS])


def failure_label(row):
    w = [r for r in REPS if wrong(row, r)]
    return "+".join(w) if w else "none"


def weighted_circular_median(estimates, weights):
    ok = [(e, w) for e, w in zip(estimates, weights) if e is not None and w > 0]
    if not ok:
        return None
    ok.sort(key=lambda t: -t[1])
    return circular_weighted_median([e for e, _ in ok], [w for _, w in ok])


def analyze(all_rows):
    from exp042_disagreement_prediction import grouped_cv_scores, roc_auc
    from exp043_disagreement_patterns import kmeans, knn_cv_accuracy, nmi, silhouette

    rows = [r for r in all_rows if not r["empty"]]
    empty = [r for r in all_rows if r["empty"]]
    groups = np.array([r["group"] for r in rows])
    conds = list(dict.fromkeys((r["family"], r["severity"]) for r in all_rows))
    s = {"n_patches": len(all_rows), "n_analysed": len(rows), "gross_deg": GROSS_DEG, "lenses": list(REPS),
         "empty": {"n": len(empty), "conditions": sorted({f"{r['family']} {r['severity']:g}" for r in empty})}}

    # 1. Atlas
    W = {rep: np.array([wrong(r, rep) for r in rows]) for rep in REPS}
    s["atlas"] = {rep: {f"{f} {v:g}": float(np.mean([wrong(r, rep) for r in rows
                                                     if r["family"] == f and r["severity"] == v]))
                        for f, v in conds} for rep in REPS}
    s["atlas_by_signal"] = {rep: {sig: float(np.mean([wrong(r, rep) for r in rows if r["signal"] == sig]))
                                  for sig in sorted(set(SIGNAL_TYPE.values()))} for rep in REPS}

    # 2. Prediction
    s["prediction"] = {}
    p_wrong = {}
    for rep in REPS:
        y = W[rep].astype(float)
        X = np.array([features(r, rep) for r in rows])
        p_wrong[rep] = grouped_cv_scores(X, y, groups)
        s["prediction"][rep] = {
            "gross_error_rate": float(y.mean()),
            "auc_self_conf": roc_auc(1.0 - X[:, 2], y),
            "auc_divergence": roc_auc(X[:, 0], y),
            "auc_disagree_model": roc_auc(p_wrong[rep], y),
        }

    # 3. Answer quality
    def rate(estimates):
        answered = np.array([e is not None for e in estimates])
        errs = np.array([err_deg(e, r["truth"]) for e, r in zip(estimates, rows)])
        return {"gross_error_rate": float(np.mean(errs > GROSS_DEG)),
                "abstain_rate": float(1 - answered.mean()),
                "gross_error_rate_when_answering": float(np.mean(errs[answered] > GROSS_DEG))}

    s["answer"] = {rep: rate([r["readings"][rep][0] for r in rows]) for rep in REPS}
    s["answer"]["most_confident"] = rate([r["readings"][max(REPS, key=lambda k: r["readings"][k][1])][0]
                                          for r in rows])
    s["answer"]["consensus"] = rate([r["consensus"] for r in rows])
    # Learned trust: out-of-fold P(wrong) from the disagreement model, per lens.
    P = np.array([p_wrong[r] for r in REPS]).T            # patches × lenses
    s["answer"]["learned_pick"] = rate([rows[i]["readings"][REPS[int(np.argmin(P[i]))]][0]
                                        for i in range(len(rows))])
    s["answer"]["learned_consensus"] = rate([
        weighted_circular_median([rows[i]["readings"][r][0] for r in REPS], 1.0 - P[i])
        for i in range(len(rows))])
    s["answer"]["oracle_best_single"] = rate([min((r["readings"][k][0] for k in REPS),
                                                  key=lambda e: err_deg(e, r["truth"])) for r in rows])
    pick = [rows[i]["readings"][REPS[int(np.argmin(P[i]))]][0] for i in range(len(rows))]
    s["answer_by_family"] = {}
    for f in dict.fromkeys(f for f, _ in conds):
        idx = [i for i, r in enumerate(rows) if r["family"] == f]
        est = {m: [rows[i]["readings"][m][0] for i in idx] for m in REPS}
        est["consensus"] = [rows[i]["consensus"] for i in idx]
        est["learned_pick"] = [pick[i] for i in idx]
        s["answer_by_family"][f] = {m: float(np.mean([err_deg(e, rows[i]["truth"]) > GROSS_DEG
                                                      for e, i in zip(v, idx)])) for m, v in est.items()}

    # 4. Patterns
    lift = {}
    for a, b in PAIRS:
        pa, pb, pab = W[a].mean(), W[b].mean(), np.mean(W[a] & W[b])
        lift[f"{a}+{b}"] = float(pab / (pa * pb)) if pa * pb > 0 else float("nan")
    s["co_failure_lift"] = lift
    # Within-condition lift: observed joint failures / joint failures expected if the two
    # lenses failed independently *given the condition* (removes "both fail on hard conditions").
    cond_of = np.array([f"{r['family']} {r['severity']:g}" for r in rows])
    s["co_failure_lift_within_condition"] = {}
    for a, b in PAIRS:
        obs = exp = 0.0
        for c in np.unique(cond_of):
            m = cond_of == c
            obs += np.sum(W[a][m] & W[b][m])
            exp += m.sum() * W[a][m].mean() * W[b][m].mean()
        s["co_failure_lift_within_condition"][f"{a}+{b}"] = float(obs / exp) if exp > 0 else float("nan")

    X = np.array([vector(r) for r in rows])
    sil = {}
    for k in K_RANGE:
        labels, _ = kmeans(X, k)
        sil[k] = silhouette(X, labels)
    k_best = max(sil, key=sil.get)
    labels, _ = kmeans(X, k_best)
    fail = np.array([failure_label(r) for r in rows])
    cond_label = np.array([f"{r['family']} {r['severity']:g}" for r in rows])
    clusters = []
    for c in range(k_best):
        idx = labels == c
        top_fail, n_fail = max(((f, int(np.sum(fail[idx] == f))) for f in sorted(set(fail[idx]))), key=lambda t: t[1])
        top_fam = max(sorted(set(cond_label[idx])), key=lambda f: np.sum(cond_label[idx] == f))
        conf = X[idx, :len(REPS)].mean(0)
        clusters.append({
            "size": int(idx.sum()),
            "failure": top_fail, "purity": n_fail / int(idx.sum()),
            "top_condition": top_fam,
            "mean_confidence": {r: round(float(v), 2) for r, v in zip(REPS, conf)},
            "consensus_gross_rate": float(np.mean([err_deg(rows[i]["consensus"], rows[i]["truth"]) > GROSS_DEG
                                                   for i in np.where(idx)[0]])),
        })
    sets = {}
    for f, c in zip(fail, cond_label):
        if f != "none":
            sets.setdefault(f, []).append(c)
    s["failure_sets"] = [
        {"lenses": f, "n": len(cs),
         "top_conditions": [f"{c} ({cs.count(c)})" for c in sorted(sorted(set(cs)), key=lambda c: -cs.count(c))[:3]]}
        for f, cs in sorted(sets.items(), key=lambda t: -len(t[1]))[:10]]
    s["patterns"] = {
        "silhouette": {str(k): v for k, v in sil.items()}, "k": int(k_best),
        "clusters": sorted(clusters, key=lambda c: -c["size"]),
        "nmi_clusters_vs_failure": nmi(labels, fail),
        "nmi_condition_vs_failure": nmi(cond_label, fail),
        "knn_failure_from_disagreement": knn_cv_accuracy(X, fail),
        "majority_failure_rate": float(max(np.mean(fail == f) for f in set(fail))),
    }
    return s


def analyze_unoriented(rows):
    out = {"n_patches": len(rows)}
    for kind in ("white", "blob"):
        sub = [r for r in rows if r["kind"] == kind]
        out[kind] = {rep: float(np.mean([r["trusted"][rep] for r in sub])) for rep in REPS}
        out[kind]["any_lens"] = float(np.mean([any(r["trusted"].values()) for r in sub]))
        out[kind]["consensus"] = float(np.mean([r["consensus"] is not None for r in sub]))
        out[kind]["mean_overall_disagreement"] = float(np.mean([r["overall"] for r in sub]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(17, 11))
    conds = list(s["atlas"][REPS[0]])
    A = np.array([[s["atlas"][r][c] for c in conds] for r in REPS])
    im = ax[0, 0].imshow(A, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax[0, 0].set_xticks(range(len(conds)), conds, rotation=60, ha="right", fontsize=8)
    ax[0, 0].set_yticks(range(len(REPS)), REPS)
    for i in range(len(REPS)):
        for j in range(len(conds)):
            ax[0, 0].text(j, i, f"{A[i, j]:.0%}", ha="center", va="center", fontsize=7,
                          color="white" if A[i, j] > 0.6 else "black")
    ax[0, 0].set_title(f"1. Atlas: gross error rate (> {GROSS_DEG:g}°) per lens and condition")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.03)

    keys = [("auc_self_conf", "own confidence"), ("auc_divergence", "divergence"),
            ("auc_disagree_model", "disagreement model (CV)")]
    w = 0.26
    for i, (k, lab) in enumerate(keys):
        vals = [s["prediction"][r][k] for r in REPS]
        ax[0, 1].bar(np.arange(len(REPS)) + (i - 1) * w, vals, w, label=lab)
    ax[0, 1].axhline(0.5, color="grey", lw=0.8, ls="--")
    ax[0, 1].set_xticks(range(len(REPS)), REPS)
    ax[0, 1].set_ylim(0.4, 1.0)
    ax[0, 1].set_ylabel("ROC AUC (predicting a gross error)")
    ax[0, 1].set_title("2. Does disagreement predict when a lens is wrong?")
    ax[0, 1].legend(fontsize=8, loc="lower left")

    L = np.ones((len(REPS), len(REPS)))
    for key, v in s["co_failure_lift"].items():
        a, b = sorted(key.split("+"), key=REPS.index)
        L[REPS.index(a), REPS.index(b)] = v                                   # upper: raw
        L[REPS.index(b), REPS.index(a)] = s["co_failure_lift_within_condition"][key]  # lower: within condition
    im = ax[1, 0].imshow(np.log2(L), cmap="RdBu_r", vmin=-2, vmax=2)
    for i in range(len(REPS)):
        for j in range(len(REPS)):
            if i != j:
                ax[1, 0].text(j, i, f"{L[i, j]:.2f}×", ha="center", va="center", fontsize=9)
    ax[1, 0].set_xticks(range(len(REPS)), REPS)
    ax[1, 0].set_yticks(range(len(REPS)), REPS)
    ax[1, 0].set_title("4. Co-failure lift: upper = raw, lower = within the same condition")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.04, label="log2 lift")

    methods = [*REPS, "most_confident", "consensus", "learned_pick", "learned_consensus", "oracle_best_single"]
    vals = [s["answer"][m]["gross_error_rate"] for m in methods]
    bars = ax[1, 1].bar(range(len(methods)), vals,
                        color=["#8aa"] * len(REPS) + ["#c84", "#2a7", "#27a", "#27a", "#bbb"])
    for b, v in zip(bars, vals):
        ax[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)
    ax[1, 1].set_xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=8)
    ax[1, 1].set_ylabel("gross error rate (abstaining counts as wrong)")
    ax[1, 1].set_title("3. Is the consensus a better answer than any single lens?")

    fig.suptitle(f"Exp 044 — Images: orientation from five lenses on the shared core "
                 f"({s['n_patches']} patches)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)


def main():
    if "--calibrate" in sys.argv:
        calibrate()
        return
    rows = collect()
    s = analyze(rows)
    s["unoriented"] = analyze_unoriented(collect_unoriented())
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2)
    plot(s)

    print(f"Patches: {s['n_patches']}, analysed {s['n_analysed']} "
          f"(empty: {s['empty']['n']} in {s['empty']['conditions']}); gross error > {GROSS_DEG:g}°\n")
    print("Prediction (AUC)        gross   own-conf  divergence  disagree-model")
    for r in REPS:
        p = s["prediction"][r]
        print(f"  {r:8s}             {p['gross_error_rate']:6.1%}   {p['auc_self_conf']:.3f}     "
              f"{p['auc_divergence']:.3f}       {p['auc_disagree_model']:.3f}")
    print("\nAnswer quality          gross   abstain  gross-when-answering")
    for m, v in s["answer"].items():
        print(f"  {m:20s} {v['gross_error_rate']:6.1%}  {v['abstain_rate']:6.1%}   {v['gross_error_rate_when_answering']:6.1%}")
    print("\nCo-failure lift:", {k: round(v, 2) for k, v in s["co_failure_lift"].items()})
    print("Within condition:", {k: round(v, 2) for k, v in s["co_failure_lift_within_condition"].items()})
    print("\nMost common failure sets:")
    for fs in s["failure_sets"]:
        print(f"  {fs['lenses']:32s} {fs['n']:4d}  {', '.join(fs['top_conditions'])}")
    pt = s["patterns"]
    print(f"\nPatterns: k = {pt['k']}, NMI(clusters, failure) = {pt['nmi_clusters_vs_failure']:.3f}, "
          f"NMI(condition, failure) = {pt['nmi_condition_vs_failure']:.3f}, "
          f"k-NN failure accuracy = {pt['knn_failure_from_disagreement']:.3f} "
          f"(majority {pt['majority_failure_rate']:.3f})")
    for c in pt["clusters"]:
        print(f"  n={c['size']:4d}  failure={c['failure']:28s} purity={c['purity']:.2f}  "
              f"top condition={c['top_condition']:15s} consensus gross={c['consensus_gross_rate']:.0%}")
    print("\nUnoriented (share of patches where a lens is confident):")
    for kind in ("white", "blob"):
        print(f"  {kind}: {s['unoriented'][kind]}")
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
