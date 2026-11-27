"""
Experiment 052: a learned embedding as a lens (issue #26)
=========================================================
Does a neural lens fail in ways the classic lenses do not, or does it join a
family that already fails together?

Learned lenses (core ``Representation``s, torch imported lazily)
  audio   crepe     pretrained CREPE "full" (torchcrepe 0.0.24), no training here.
                    src/framework/learned.py
  image   cnn       frozen ImageNet ResNet-18 (torchvision 0.29.2, IMAGENET1K_V1),
                    conv1 + layer1 + layer2 channel means, ridge read-out to
                    (cos 2θ, sin 2θ) fitted on clean synthetic patterns only
          cnn_aug   the same read-out fitted on those patterns plus noisy and
                    blurred copies (noise and blur are then not new to it)
                    src/image_medium/learned.py
Both get a 0..1 confidence with the rule every calibrated lens uses: lo = the
strongest raw score over 60 white-noise inputs, hi = the 5th percentile over
clean synthetic inputs.

Data (cached readings of the classic lenses, rebuilt inputs for the learned ones)
  audio   synthetic (Exp 042, 3240 frames + 90 white-noise frames),
          vocadito and MDB-stem-synth (Exp 045, all conditions + unvoiced frames)
  image   synthetic (Exp 044, 2304 patches + 90 isotropic patches),
          BSDS500 boundaries (Exp 045, 514 patches × 6 conditions)
CREPE was trained on MDB-stem-synth (among others), so on that dataset it has
seen the test recordings; vocadito (2021) is newer than CREPE.

Questions
  1. Blind spots: gross error per lens and condition (audio > 50 cents,
     images > 10°).
  2. Family: failure correlation (phi) and co-failure lift of the learned lens
     with every classic lens, raw and within condition; the classic families
     are spectral vs periodicity (audio, Exp 043) and local gradient vs global
     spectral/projection (images, Exp 044). Placement in the Exp 043/044
     failure sets: P(learned wrong | which classic lenses are wrong).
  3. Warning: does the learned lens make disagreement a better warning for
     the classic lenses (ROC AUC of the Exp 042 disagreement model, 5-fold CV
     split by group), and does disagreement warn when the learned lens is
     wrong better than its own confidence does?
  4. Fusion: legacy (all trusted), quorum 2, and the Exp 047 AgreementTable
     (fitted within 5-fold CV) with and without the learned lens; abstention
     counts as wrong. Differences with group-bootstrap 95% CIs. False answers
     on unpitched / unoriented input.

Run:  python src/experiments/exp052_learned_lens.py [--calibrate]
Outputs: results/exp052_learned_lens.png, results/exp052_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import zlib
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp042_disagreement_prediction as e42  # noqa: E402
import exp044_image_disagreement as e44  # noqa: E402
import exp045_real_data as e45  # noqa: E402
import exp046_quorum as e46  # noqa: E402
import exp050_probing as e50  # noqa: E402
from core import AgreementTable, ConfidenceThresholdDetector, Reading  # noqa: E402
from framework import learned as audio_learned  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import OUTLIER_MARGIN, VOICING_THRESHOLD, read_stft  # noqa: E402
from image_medium import learned as image_learned  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD, OrientationComparator, TensorOrientation  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp052_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp052_learned_lens.png")
SR, FRAME = e42.SR, e42.FRAME
FOLDS = 5
N_TRAIN = 400            # clean training patterns per image variant (cnn read-out)

AUDIO_BASE = ("stft", "acf", "cepstrum")
IMAGE_BASE = tuple(e44.REPS)
FAMILIES = {
    "audio": {"spectral": ("stft",), "periodicity": ("acf", "cepstrum")},
    "image": {"local gradient": ("tensor", "hog"), "global spectral/projection": ("fft", "gabor", "radon")},
}
LEARNED = {"audio": ("crepe",), "image": ("cnn", "cnn_aug")}
DATASETS = {"audio": ("synthetic", "vocadito", "mdb_stem_synth"), "image": ("synthetic", "bsds")}


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cache(name, fn, refresh=False):
    path = os.path.join(CACHE, f"exp052_{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    val = fn()
    with open(path, "wb") as f:
        pickle.dump(val, f)
    print(f"  {name}: computed in {time.time() - t:.0f} s")
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Audio: CREPE readings on the frames of Exp 042 / 045
# ──────────────────────────────────────────────────────────────────────────────

def audio_cached(name):
    if name == "synthetic":
        return e46.cached("exp046_e42_rows", None), e46.cached("exp046_e42_unpitched", None)
    return e46.cached(f"exp045_audio_{name}", None)


def _crepe_over(gen, n_hint, check=None):
    """Run CREPE over (index, context) pairs in batches; returns raw f0 and peak arrays."""
    f0, peak, buf, idx, mismatch, t = {}, {}, [], [], 0, time.time()

    def flush():
        p = audio_learned.crepe_probabilities(np.array(buf), SR)
        f, pk = audio_learned.decode(p)
        for j, i in enumerate(idx):
            f0[i], peak[i] = float(f[j]), float(pk[j])
        buf.clear()
        idx.clear()

    for i, ctx in gen:
        frame = ctx[:FRAME]
        if check is not None and len(f0) + len(buf) < 30:
            mismatch += check(i, frame)
        buf.append(frame)
        idx.append(i)
        if len(buf) == 2048:
            flush()
            print(f"    {len(f0)}/{n_hint} frames ({time.time() - t:.0f} s)")
    if buf:
        flush()
    n = max(f0) + 1 if f0 else 0
    return np.array([f0[i] for i in range(n)]), np.array([peak[i] for i in range(n)]), mismatch


def crepe_readings(name):
    rows, nulls = audio_cached(name)

    def check(rows_):
        def c(i, frame):
            f = read_stft(frame, SR).f0
            ref = rows_[i]["readings"]["stft"][0]
            return int(f > 0 and ref > 0 and abs(1200 * np.log2(f / ref)) > 1.0)
        return c

    wanted = set(range(len(rows)))
    gen = (e50.synthetic_contexts(wanted) if name == "synthetic"
           else e50.real_contexts(name, wanted, e45.MDB_FRAMES_PER_TRACK if name == "mdb_stem_synth" else None))
    f0, peak, mm = _crepe_over(gen, len(rows), check(rows))
    nf0, npeak, mm0 = _crepe_over(e50.unpitched_contexts(name), len(nulls), check(nulls))
    assert len(f0) == len(rows), (len(f0), len(rows))
    return {"f0": f0, "peak": peak, "f0_null": nf0, "peak_null": npeak[:len(nulls)],
            "check_mismatch": mm + mm0}


def crepe_calibration():
    noise = np.array([np.random.default_rng(30_000 + s).normal(0.0, 0.3, FRAME) for s in range(60)])
    _, pk_noise = audio_learned.decode(audio_learned.crepe_probabilities(noise, SR))
    rows, _ = audio_cached("synthetic")
    r = cache("crepe_synthetic", lambda: crepe_readings("synthetic"))
    clean = np.array([row["family"] == "clean" for row in rows])
    return round(float(pk_noise.max()), 3), round(float(np.percentile(r["peak"][clean], 5)), 3)


# ──────────────────────────────────────────────────────────────────────────────
# Images: CNN embeddings of the patches of Exp 044 / 045
# ──────────────────────────────────────────────────────────────────────────────

def image_cached(name):
    if name == "synthetic":
        return e46.cached("exp046_e44_rows", None), e46.cached("exp046_e44_unoriented", None)
    return e46.cached("exp045_image_rows", None), []


def image_patches(name):
    """Rebuild the patches in the order of the cached rows (and the unoriented ones)."""
    if name == "synthetic":
        X = [e44.make_patch(v, th, fam, sev, seed, k)
             for v, k, th in e44.base_signals() for fam, sev, _ in e44.CONDITIONS for seed in e44.SEEDS]
        N = [e44.isotropic(np.random.default_rng(20_000 + s), "white" if s % 2 == 0 else "blob")
             for s in range(e44.UNORIENTED_PATCHES)]
        return np.array(X), np.array(N)
    patches = e46.cached("exp045_bsds_patches", None)
    X = [fn(patch, e45.rng_for("img", iid, k, family, severity))
         for k, (iid, theta, patch) in enumerate(patches) for family, severity, fn in e45.IMAGE_CONDITIONS]
    return np.array(X), np.zeros((0, e44.N, e44.N))


def image_embeddings(name):
    X, N = image_patches(name)
    rows, _ = image_cached(name)
    tensor = TensorOrientation()
    tensor.contrast = e44.LENSES["tensor"].contrast
    mismatch = 0
    for i in range(0, len(rows), max(len(rows) // 25, 1)):   # the rebuilt patch gives the cached reading
        if rows[i]["empty"]:
            continue
        r = tensor.read(X[i], {})
        ref = rows[i]["readings"]["tensor"][0]
        mismatch += int(r.estimate is not None and ref is not None and abs(e44.angle_diff(r.estimate, ref)) > 1e-6)
    return {"E": image_learned.embed(X), "E_null": image_learned.embed(N) if len(N) else np.zeros((0, 256)),
            "check_mismatch": mismatch}


def training_set():
    """Clean synthetic patterns, orientations and seeds the evaluation never uses,
    plus one noisy and one blurred copy of each for the augmented read-out."""
    rng = np.random.default_rng(12345)
    P, T, A, TA = [], [], [], []
    for v in e44.VARIANTS:
        for i in range(N_TRAIN):
            th = rng.uniform(0, 180)
            r = np.random.default_rng(zlib.crc32(f"train-{v}-{i}".encode()))
            x = e44.VARIANTS[v](th, r)
            P.append(x)
            T.append(th)
            A += [x + r.normal(0, r.uniform(0.05, 0.4), x.shape), ndimage.gaussian_filter(x, r.uniform(0.5, 3.0))]
            TA += [th, th]
    return {"E": image_learned.embed(np.array(P)), "theta": np.array(T),
            "E_aug": image_learned.embed(np.array(A)), "theta_aug": np.array(TA)}


def image_heads():
    tr = cache("cnn_train", training_set)
    heads = {"cnn": image_learned.OrientationHead.fit(tr["E"], tr["theta"]),
             "cnn_aug": image_learned.OrientationHead.fit(np.r_[tr["E"], tr["E_aug"]],
                                                         np.r_[tr["theta"], tr["theta_aug"]])}
    for n, h in heads.items():
        h.save(os.path.join(CACHE, f"exp052_{n}_head.npz"))
    noise = image_learned.embed(np.array([e44.isotropic(np.random.default_rng(50_000 + s), "white")
                                          for s in range(60)]))
    rows, _ = image_cached("synthetic")
    emb = cache("cnn_synthetic", lambda: image_embeddings("synthetic"))
    clean = np.array([r["family"] == "clean" for r in rows])
    calib = {}
    for n, h in heads.items():
        lo = round(float(h.predict(noise)[1].max()), 3)
        hi = round(float(np.percentile(h.predict(emb["E"][clean])[1], 5)), 3)
        # If noise scores above clean patterns the rule is undefined: the lens is more
        # sure about noise than about structure. Fall back to raw / hi (noise then
        # gets full confidence) and say so in the summary.
        calib[n] = (lo, hi) if lo < hi else (0.0, hi)
        calib[f"{n}_raw_rule"] = (lo, hi)
    return heads, calib


# ──────────────────────────────────────────────────────────────────────────────
# Rows with every lens, medium adapters
# ──────────────────────────────────────────────────────────────────────────────

def scale(raw, lo, hi):
    return np.clip((np.asarray(raw, float) - lo) / (hi - lo), 0.0, 1.0)


def audio_rows(name, calib):
    rows, nulls = audio_cached(name)
    c = cache(f"crepe_{name}", lambda: crepe_readings(name))
    conf, conf0 = scale(c["peak"], *calib), scale(c["peak_null"], *calib)
    out = [{"truth": r["truth"], "group": str(r["group"]), "family": r["family"], "severity": str(r["severity"]),
            "readings": {**{n: r["readings"][n] for n in AUDIO_BASE}, "crepe": (float(c["f0"][i]), float(conf[i]))}}
           for i, r in enumerate(rows)]
    null = [{"readings": {**{n: r["readings"][n] for n in AUDIO_BASE},
                          "crepe": (float(c["f0_null"][i]), float(conf0[i]))}}
            for i, r in enumerate(nulls[:len(conf0)])]
    return out, null, c["check_mismatch"]


def image_rows(name, heads, calib):
    rows, nulls = image_cached(name)
    emb = cache(f"cnn_{name}", lambda: image_embeddings(name))
    learned, learned0 = {}, {}
    for n, h in heads.items():
        th, raw = h.predict(emb["E"])
        learned[n] = (th, scale(raw, *calib[n]))
        if len(emb["E_null"]):
            th0, raw0 = h.predict(emb["E_null"])
            learned0[n] = (th0, scale(raw0, *calib[n]))
    out, null = [], []
    for i, r in enumerate(rows):
        if r["empty"]:
            continue
        rd = {n: r["readings"][n] for n in IMAGE_BASE}
        rd.update({n: (float(learned[n][0][i]), float(learned[n][1][i])) for n in heads})
        out.append({"truth": r["truth"], "group": str(r["group"]), "family": r["family"],
                    "severity": f"{r['severity']:g}", "readings": rd})
    for i, r in enumerate(nulls):
        if r["empty"]:
            continue
        rd = {n: r["readings"][n] for n in IMAGE_BASE}
        rd.update({n: (float(learned0[n][0][i]), float(learned0[n][1][i])) for n in heads})
        null.append({"readings": rd})
    return out, null, emb["check_mismatch"]


class Audio:
    special = ("octave above", "octave below")

    @staticmethod
    def reading(n, v):
        return Reading(n, v[0] if v[0] and v[0] > 0 else None, float(v[1]))

    @staticmethod
    def wrong(est, row):
        return est is None or e42.cents_error(est, row["truth"]) > e42.GROSS_CENTS

    @staticmethod
    def comparator(k, family=None):
        return PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=k, family=family,
                               detector=ConfidenceThresholdDetector(VOICING_THRESHOLD))


class Image:
    special = ("orthogonal",)

    @staticmethod
    def reading(n, v):
        return Reading(n, v[0], float(v[1]))

    @staticmethod
    def wrong(est, row):
        return est is None or e44.err_deg(est, row["truth"]) > e44.GROSS_DEG

    @staticmethod
    def comparator(k, family=None):
        return OrientationComparator(min_agreeing=k, family=family,
                                     detector=ConfidenceThresholdDetector(VOTING_THRESHOLD))


def readings(M, row, lenses):
    return {n: M.reading(n, row["readings"][n]) for n in lenses}


# ──────────────────────────────────────────────────────────────────────────────
# 1-2. Blind spots and families
# ──────────────────────────────────────────────────────────────────────────────

def wrong_matrix(M, rows, lenses):
    return {n: np.array([M.wrong(M.reading(n, r["readings"][n]).estimate, r) for r in rows]) for n in lenses}


def phi(a, b):
    pa, pb, pab = a.mean(), b.mean(), (a & b).mean()
    d = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
    return float((pab - pa * pb) / d) if d > 0 else float("nan")


def lift(a, b):
    e = a.mean() * b.mean()
    return float((a & b).mean() / e) if e > 0 else float("nan")


def within(fn, a, b, cond):
    """Mean of fn within each condition, weighted by condition size (nan-skipped)."""
    vals, wts = [], []
    for c in np.unique(cond):
        m = cond == c
        v = fn(a[m], b[m])
        if v == v:
            vals.append(v)
            wts.append(m.sum())
    return float(np.average(vals, weights=wts)) if vals else float("nan")


def families(medium, W, cond, learned):
    lenses = list(W)
    pairs = {f"{a}+{b}": {"phi": phi(W[a], W[b]), "lift": lift(W[a], W[b]),
                          "phi_within_condition": within(phi, W[a], W[b], cond),
                          "lift_within_condition": within(lift, W[a], W[b], cond)}
             for a, b in combinations(lenses, 2)}

    def get(a, b, key):
        return pairs.get(f"{a}+{b}", pairs.get(f"{b}+{a}"))[key]

    fam = FAMILIES[medium]
    same = [(a, b) for m in fam.values() for a, b in combinations(m, 2)]
    across = [(a, b) for (f1, m1), (f2, m2) in combinations(fam.items(), 2) for a in m1 for b in m2]
    ref = {k: {"same_family": float(np.nanmean([get(a, b, k) for a, b in same])),
               "across_families": float(np.nanmean([get(a, b, k) for a, b in across]))}
           for k in ("phi", "phi_within_condition")}
    out = {"pairs": pairs, "classic_reference": ref, "learned": {}}
    base = [n for n in lenses if n not in LEARNED[medium]]
    label = np.array(["+".join(n for n in base if W[n][i]) or "none" for i in range(len(cond))])
    for L in learned:
        to_family = {f: {k: float(np.nanmean([get(L, m, k) for m in members]))
                         for k in ("phi", "phi_within_condition")} for f, members in fam.items()}
        best = max(to_family, key=lambda f: np.nan_to_num(to_family[f]["phi"], nan=-1))
        joins = to_family[best]["phi"] >= 0.5 * ref["phi"]["same_family"]
        any_classic = np.any([W[n] for n in base], axis=0)
        all_classic = np.all([W[n] for n in base], axis=0)
        sets = {}
        for s in sorted(set(label), key=lambda s: -np.sum(label == s)):
            m = label == s
            if m.sum() >= 10:
                sets[s] = {"n": int(m.sum()), "p_learned_wrong": float(W[L][m].mean())}
        out["learned"][L] = {
            "gross": float(W[L].mean()),
            "to_family": to_family,
            "closest_family": best,
            "verdict": f"joins {best}" if joins else "own failures (no family above half the classic same-family phi)",
            "alone_share_of_its_failures": float(np.mean(~any_classic[W[L]])) if W[L].any() else float("nan"),
            "right_when_all_classic_wrong": float(np.mean(~W[L][all_classic])) if all_classic.any() else float("nan"),
            "n_all_classic_wrong": int(all_classic.sum()),
            "by_failure_set": sets,
        }
    return out


def atlas(W, rows):
    cond = np.array([f"{r['family']} {r['severity']}" for r in rows])
    return {n: {c: float(w[cond == c].mean()) for c in dict.fromkeys(cond)} for n, w in W.items()}, cond


# ──────────────────────────────────────────────────────────────────────────────
# 3. Warning
# ──────────────────────────────────────────────────────────────────────────────

def reports(M, rows, lenses):
    comp = M.comparator(2)
    return [comp.compare(readings(M, r, lenses)) for r in rows]


def features(M, rep, n):
    others = [o for o in rep.readings if o != n]
    mine = [p for p in rep.pairs if n in (p.a, p.b)]
    return [rep.divergence[n], rep.overall, rep.readings[n].confidence,
            float(np.mean([rep.readings[o].confidence for o in others])),
            min(p.distance for p in mine), float(any(p.relation in M.special for p in mine))]


def warning(M, rows, reps, W, targets, groups):
    out = {}
    for n in targets:
        X = np.array([features(M, rep, n) for rep in reps])
        y = W[n].astype(float)
        s = e42.grouped_cv_scores(X, y, groups) if 0 < y.sum() < len(y) else np.full(len(y), np.nan)
        out[n] = {"disagreement_model": auc(s, W[n]), "own_confidence": auc(-X[:, 2], W[n]),
                  "divergence": auc(X[:, 0], W[n]), "gross": float(W[n].mean())}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. Fusion
# ──────────────────────────────────────────────────────────────────────────────

def fit_table(M, rows, lenses):
    comp = M.comparator(1)
    ex = []
    for r in rows:
        rd = readings(M, r, lenses)
        rep = comp.compare(rd)
        trusted = [n for n in rd if not rep.assessments[n].blind]
        ex += [(g, not M.wrong(comp.fuse([rd[n] for n in g]), r)) for g in comp.agreeing_groups(trusted, rep.pairs)]
    return AgreementTable.fit(ex)


def fusion(M, rows, lenses, groups):
    est = {"legacy": [], "q2": [], "table_q1": [None] * len(rows)}
    legacy, q2 = M.comparator(None), M.comparator(2)
    for r in rows:
        rd = readings(M, r, lenses)
        est["legacy"].append(legacy.compare(rd).consensus)
        est["q2"].append(q2.compare(rd).consensus)
    uniq = np.unique(groups)
    np.random.default_rng(0).shuffle(uniq)
    for fold in np.array_split(uniq, FOLDS):
        test = np.isin(groups, fold)
        comp = M.comparator(1, fit_table(M, [r for r, t in zip(rows, test) if not t], lenses))
        for i in np.flatnonzero(test):
            est["table_q1"][i] = comp.compare(readings(M, rows[i], lenses)).consensus
    bad = {m: np.array([M.wrong(x, r) for x, r in zip(e, rows)]) for m, e in est.items()}
    ans = {m: np.array([x is not None for x in e]) for m, e in est.items()}
    return bad, ans


def false_answers(M, nulls, lenses):
    q2 = M.comparator(2)
    return float(np.mean([q2.compare(readings(M, r, lenses)).consensus is not None for r in nulls])) if nulls else None


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def analyze(medium, name, rows, nulls):
    M = Audio if medium == "audio" else Image
    base = AUDIO_BASE if medium == "audio" else IMAGE_BASE
    learned = LEARNED[medium]
    lenses = base + learned
    groups = np.array([r["group"] for r in rows])
    W = wrong_matrix(M, rows, lenses)
    at, cond = atlas(W, rows)
    fam_cond = np.array([r["family"] for r in rows])
    out = {"n": len(rows), "n_null": len(nulls), "gross": {n: float(w.mean()) for n, w in W.items()},
           "atlas": at, "atlas_by_family": {n: {f: float(w[fam_cond == f].mean()) for f in dict.fromkeys(fam_cond)}
                                            for n, w in W.items()},
           "families": families(medium, W, cond, learned)}

    t = time.time()
    sets = {"base": base, **{f"base+{L}": base + (L,) for L in learned}}
    rep = {k: reports(M, rows, v) for k, v in sets.items()}
    out["warning"] = {k: warning(M, rows, rep[k], W, v if k != "base" else base, groups) for k, v in sets.items()}
    for k in sets:
        if k != "base":
            out["warning"][k]["mean_classic_gain"] = float(np.mean(
                [out["warning"][k][n]["disagreement_model"] - out["warning"]["base"][n]["disagreement_model"]
                 for n in base]))
    print(f"    warning {time.time() - t:.0f} s")

    t = time.time()
    fus, bads = {}, {}
    for k, v in sets.items():
        bad, ans = fusion(M, rows, v, groups)
        bads[k] = bad
        fus[k] = {m: {"gross_all": float(b.mean()), "answer": float(ans[m].mean()),
                      "gross_answer": float(b[ans[m]].mean()) if ans[m].any() else float("nan")}
                  for m, b in bad.items()}
        fus[k]["false_answer_q2"] = false_answers(M, nulls, v)
        best = min(v, key=lambda n: W[n].mean())
        fus[k]["best_lens"] = {"name": best, "gross_all": float(W[best].mean())}
    for L in learned:
        k = f"base+{L}"
        for m in ("legacy", "q2", "table_q1"):
            fus[k][f"{m}_minus_base"] = e45.group_bootstrap(groups, bads[k][m], bads["base"][m])
        fus[k]["table_q1_minus_learned_alone"] = e45.group_bootstrap(groups, bads[k]["table_q1"], W[L])
        fus[k]["table_q1_minus_best_classic"] = e45.group_bootstrap(
            groups, bads[k]["table_q1"], W[fus["base"]["best_lens"]["name"]])
        fus[k]["by_family_q2"] = {f: {"base": float(bads["base"]["q2"][fam_cond == f].mean()),
                                      "with": float(bads[k]["q2"][fam_cond == f].mean())}
                                  for f in dict.fromkeys(fam_cond)}
    out["fusion"] = fus
    print(f"    fusion {time.time() - t:.0f} s")
    return out


def main():
    if "--calibrate" in sys.argv:
        print("crepe", crepe_calibration())
        print("cnn", image_heads()[1])
        return
    import torch
    import torchcrepe
    import torchvision

    s = {"versions": {"torch": torch.__version__, "torchcrepe": torchcrepe.__version__ if hasattr(torchcrepe, "__version__") else "0.0.24",
                      "torchvision": torchvision.__version__},
         "weights_sha256": {"crepe_full": audio_learned._Crepe.weights_sha256(),
                            "resnet18_imagenet1k_v1": image_learned._Backbone.weights_sha256()},
         "audio": {}, "image": {}}
    calib_a = crepe_calibration()
    heads, calib_all = image_heads()
    calib_i = {n: calib_all[n] for n in heads}
    s["calibration"] = {"crepe": calib_a, **calib_all}
    print("calibration", s["calibration"])

    for name in DATASETS["audio"]:
        print(f"audio {name}")
        rows, nulls, mm = audio_rows(name, calib_a)
        s["audio"][name] = analyze("audio", name, rows, nulls)
        s["audio"][name]["rebuild_mismatch"] = mm
    for name in DATASETS["image"]:
        print(f"image {name}")
        rows, nulls, mm = image_rows(name, heads, calib_i)
        s["image"][name] = analyze("image", name, rows, nulls)
        s["image"][name]["rebuild_mismatch"] = mm

    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    for medium in ("audio", "image"):
        for name, d in s[medium].items():
            print(f"\n=== {medium} / {name}  (n = {d['n']}, nulls {d['n_null']}, rebuild mismatch {d['rebuild_mismatch']})")
            print("  gross: " + ", ".join(f"{n} {v:.1%}" for n, v in d["gross"].items()))
            print("  by condition family:")
            for n, fams in d["atlas_by_family"].items():
                print(f"    {n:9s} " + " ".join(f"{f[:10]}:{v:5.1%}" for f, v in fams.items()))
            F = d["families"]
            print(f"  classic phi: same family {F['classic_reference']['phi']['same_family']:.2f}, "
                  f"across {F['classic_reference']['phi']['across_families']:.2f}  "
                  f"(within condition {F['classic_reference']['phi_within_condition']['same_family']:.2f} / "
                  f"{F['classic_reference']['phi_within_condition']['across_families']:.2f})")
            for L, v in F["learned"].items():
                print(f"  {L}: " + "; ".join(f"{f} phi {x['phi']:.2f} (within {x['phi_within_condition']:.2f})"
                                             for f, x in v["to_family"].items()))
                print(f"     verdict: {v['verdict']}; alone in {v['alone_share_of_its_failures']:.0%} of its failures; "
                      f"right in {v['right_when_all_classic_wrong']:.0%} of {v['n_all_classic_wrong']} all-classic-wrong")
                print("     P(wrong | classic failure set): " + ", ".join(
                    f"{k} {x['p_learned_wrong']:.0%} (n={x['n']})" for k, x in list(v["by_failure_set"].items())[:7]))
            for k, w in d["warning"].items():
                line = ", ".join(f"{n} {x['disagreement_model']:.3f}" for n, x in w.items() if isinstance(x, dict))
                print(f"  warning AUC [{k}]: {line}" + (f"  (classic gain {w['mean_classic_gain']:+.3f})"
                                                      if "mean_classic_gain" in w else ""))
            for k, w in d["warning"].items():
                for L in LEARNED[medium]:
                    if L in w:
                        print(f"  {L} warning: disagreement {w[L]['disagreement_model']:.3f} vs own confidence "
                              f"{w[L]['own_confidence']:.3f}")
            for k, f in d["fusion"].items():
                print(f"  fusion [{k}]: " + ", ".join(
                    f"{m} {f[m]['gross_all']:.1%} (ans {f[m]['answer']:.0%})" for m in ("legacy", "q2", "table_q1"))
                      + f"; best lens {f['best_lens']['name']} {f['best_lens']['gross_all']:.1%}; "
                        f"false answers {f['false_answer_q2']}")
                for m in ("legacy_minus_base", "q2_minus_base", "table_q1_minus_base",
                          "table_q1_minus_learned_alone", "table_q1_minus_best_classic"):
                    if m in f:
                        print(f"      {m}: {f[m]['diff']:+.2%} [{f[m]['ci95'][0]:+.2%}, {f[m]['ci95'][1]:+.2%}]")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for row, (medium, syn, real) in enumerate((("audio", "synthetic", "vocadito"), ("image", "synthetic", "bsds"))):
        d = s[medium][syn]
        ab = d["atlas_by_family"]
        lenses = list(ab)
        fams = list(next(iter(ab.values())))
        A = np.array([[ab[n][f] for f in fams] for n in lenses])
        ax = axes[row, 0]
        ax.imshow(A, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(lenses)))
        ax.set_yticklabels(lenses)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.0%}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{medium} {syn}: gross error by condition")

        ax = axes[row, 1]
        P = np.full((len(lenses), len(lenses)), np.nan)
        pairs = s[medium][real]["families"]["pairs"]
        for i, a in enumerate(lenses):
            for j, b in enumerate(lenses):
                if i != j:
                    v = pairs.get(f"{a}+{b}", pairs.get(f"{b}+{a}"))
                    P[i, j] = v["phi"]
        ax.imshow(P, cmap="Purples", vmin=0, vmax=1)
        ax.set_xticks(range(len(lenses)))
        ax.set_xticklabels(lenses, rotation=35, ha="right")
        ax.set_yticks(range(len(lenses)))
        ax.set_yticklabels(lenses)
        for i in range(len(lenses)):
            for j in range(len(lenses)):
                if i != j and P[i, j] == P[i, j]:
                    ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{medium} {real}: failure correlation (phi)")

        ax = axes[row, 2]
        names = list(s[medium])
        sets = list(s[medium][names[0]]["fusion"])
        width = 0.8 / len(sets)
        for k, st in enumerate(sets):
            vals = [s[medium][n]["fusion"][st]["table_q1"]["gross_all"] for n in names]
            ax.bar(np.arange(len(names)) + k * width, vals, width, label=f"{st} (AgreementTable)")
        for L in LEARNED[medium]:
            ax.scatter(np.arange(len(names)) + 0.4 - width / 2, [s[medium][n]["gross"][L] for n in names],
                       marker="x", s=60, zorder=3, label=f"{L} alone")
        ax.scatter(np.arange(len(names)) + 0.4 - width / 2,
                   [s[medium][n]["fusion"]["base"]["best_lens"]["gross_all"] for n in names],
                   marker="_", s=300, color="k", zorder=3, label="best classic lens")
        ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
        ax.set_xticklabels(names)
        ax.set_ylabel("gross error (abstain = wrong)")
        ax.set_title(f"{medium}: fused answer with and without the learned lens")
        ax.legend(fontsize=7)
    fig.suptitle("Exp 052: a learned embedding as a lens (CREPE for pitch, ResNet-18 features for orientation)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    main()
