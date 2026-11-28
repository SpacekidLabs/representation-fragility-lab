"""
Experiment 058: a learned image lens trained on real photos, admitted by coverage (issue #32)
=============================================================================================
Exp 055 found that a lens is worth what it covers: the frames where it is right
and fewer than two other lenses are. On BSDS500 the image pool leaves 31% of
boundary patches unanswered, and its only learned lens (the Exp 052 CNN read-out,
fitted on synthetic patterns) makes 62% gross errors on photos. Here the read-out
is fitted on real photos and judged by the admission test in ``core.admission``.

Data (BSDS500, Arbeláez et al. 2011; see data/README.md)
  train   boundary patches from the **train** split (200 images), extracted with
          the Exp 045 rule (≥3 annotators, one straight human boundary through the
          centre, contrast ≥ 0.08) but up to 12 per image, 24 px apart, as 92×92
          crops. Each is rotated 12 times (the first not at all; the rest by a
          seeded uniform angle, truth moved with it), flipped at random and cut
          back to 64×64. The augmented read-out also gets one random Exp 045-type
          perturbation per copy (noise σ 0.05–0.35, blur σ 0.5–3, JPEG-like q
          10–50, 25% occlusion, or none): those perturbation *types* are then not
          new to it, as with the augmented read-out of Exp 052.
  val     the same extraction on the **val** split (100 images), under the six
          Exp 045 conditions. Used only to choose the ridge strength and pooling,
          and as the "clean patterns" of the confidence rule.
  test    the held-out Exp 045 patches of the **test** split (514 patches × 6
          conditions), with the cached readings of every other lens: the numbers
          are directly comparable to Exp 045-057.
  null    the 90 synthetic unoriented patches (Exp 044/049/055), for false answers.
  no-boundary  photo patches with no human boundary (any annotator) within 36 px of
          the centre and std ≥ 0.02: 8 per train image (used to train
          cnn_photo_null), 4 per val image, 2 per test image (read by every lens).
          Meant as a photo null set, they turned out not to be one: humans do not
          mark texture (grass, bark, stripes), which has a real orientation, so
          the classic pool answers 48% of them. They are reported as "answers on
          no-boundary patches", not as false answers.

Lenses added (core ``Representation`` ``image_medium.learned.CnnOrientation``)
  cnn_photo        frozen ImageNet ResNet-18, conv1 + layer1 + layer2 channel means
                   over the whole map and over its central half (512 numbers), ridge
                   read-out to (cos 2θ, sin 2θ), fitted on the augmented photo set
  cnn_photo_clean  the same, fitted without the perturbations
  cnn_photo_null   cnn_photo plus the train no-boundary patches fitted to (0, 0), so
                   that its length (raw contrast) would be low where nothing is
                   oriented. A negative result: see above.
The pooling (global vs centre) and ridge strength were chosen on val; a small
MLP head on the same features did no better than ridge there (27–29% vs 30% gross
error with global pooling, run to run noise of ±1.5 pts), so the head stays linear.
Confidence: the project rule (lo = strongest raw score over 60 white-noise
patches, hi = 5th percentile over clean val photo patches).

Questions
  1. Gross error (> 10°) per condition on the test patches.
  2. Admission (``core.admission_test``) against the Exp 055 pool (tensor, hog,
     fft, gabor, radon, cnn_aug) and against the classic pool: coverage, coverage
     on the patches the pool leaves unanswered, Δ gross_all / gross_answer /
     abstention / false answers with group-bootstrap CIs (whole images). The
     admission decision uses the synthetic null set (as Exp 049/055).
  3. Family: φ of its gross errors with the local-gradient (tensor, hog) and
     global (fft, gabor, radon) lenses and with cnn_aug.
  4. The Exp 055 drop-one analysis on the new pool, to place it among the others.
  5. The synthetic Exp 044 patches, to see whether training on photos costs
     anything there.

Run:  python src/experiments/exp058_photo_lens.py   (needs BSDS500 in data/raw and the
      Exp 045, 046 and 052 caches in data/processed)
Outputs: results/exp058_photo_lens.png, results/exp058_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp044_image_disagreement as e44  # noqa: E402
import exp045_real_data as e45  # noqa: E402
import exp046_quorum as e46  # noqa: E402
import exp052_learned_lens as e52  # noqa: E402
import exp055_lens_value as e55  # noqa: E402
from core import admission_test  # noqa: E402
from image_medium import learned as image_learned  # noqa: E402
from image_medium.learned import OrientationHead  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD  # noqa: E402

CACHE = e52.CACHE
OUT_JSON = os.path.join(ROOT, "results", "exp058_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp058_photo_lens.png")
BIG = 46                        # half side of the training crops (92 px ≥ 64·√2)
HALF = e44.N // 2
N_ROT = 12
L2_GRID = (3.0, 10.0, 30.0, 100.0)
POOLINGS = ("global", "centre")
CLASSIC = e52.IMAGE_BASE
OLD_POOL = e55.POOLS["image"]    # classic + cnn_aug (Exp 055)
NEW = ("cnn_photo", "cnn_photo_clean", "cnn_photo_null")
NULL_DIST = 36                  # a null patch has no human boundary within this many px of its centre
FAMILIES = {"local gradient": ("tensor", "hog"), "global": ("fft", "gabor", "radon"), "cnn_aug": ("cnn_aug",)}
CNN_AUG_CALIBRATION = e55.CNN_AUG_CALIBRATION


def cache(name, fn):
    path = os.path.join(CACHE, f"exp058_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    val = fn()
    with open(path, "wb") as f:
        pickle.dump(val, f)
    print(f"  {name}: computed in {time.time() - t:.0f} s")
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Training and validation patches
# ──────────────────────────────────────────────────────────────────────────────

def split_patches(split):
    return e45.bsds_patches(split, per_image=12, min_dist=24, half=BIG)


def flat_patches(split, per_image, half=HALF, seed="flat"):
    """Photo patches with no orientation to find: no human boundary (any annotator)
    within NULL_DIST px of the centre, and not flat (std ≥ 0.02, so the classic
    lenses still read them). Returns (image id, patch) pairs."""
    import glob
    import scipy.io
    from PIL import Image

    out = []
    for path in sorted(glob.glob(os.path.join(e45.BSDS, "images", split, "*.jpg"))):
        iid = os.path.basename(path)[:-4]
        img = np.asarray(Image.open(path).convert("L"), float) / 255.0
        gts = scipy.io.loadmat(os.path.join(e45.BSDS, "groundTruth", split, f"{iid}.mat"))["groundTruth"][0]
        union = np.any([g["Boundaries"][0, 0].astype(bool) for g in gts], axis=0)
        far = ndimage.distance_transform_edt(~union) > NULL_DIST
        H, W = img.shape
        far[:half], far[H - half:], far[:, :half], far[:, W - half:] = False, False, False, False
        cand = np.argwhere(far)
        rng = e45.rng_for(seed, iid)
        rng.shuffle(cand)
        taken = []
        for r, c in cand:
            if len(taken) >= per_image:
                break
            if any(np.hypot(r - a, c - b) < 40 for a, b in taken):
                continue
            x = img[r - half:r + half, c - half:c + half]
            if x.std() < 0.02:
                continue
            out.append((iid, x.copy()))
            taken.append((r, c))
    return out


def perturb(x, r):
    c = r.integers(0, 5)
    if c == 1:
        return x + r.normal(0, r.uniform(0.05, 0.35), x.shape)
    if c == 2:
        return ndimage.gaussian_filter(x, r.uniform(0.5, 3.0))
    if c == 3:
        return e44.jpeg_like(x, int(r.integers(10, 50)))
    if c == 4:
        return e44.occlude(x, 0.25, r)
    return x


def training_set(patches):
    """Rotated, flipped copies of every train patch: clean ones and perturbed ones."""
    X, Xa, T = [], [], []
    for k, (iid, theta, big) in enumerate(patches):
        r = np.random.default_rng(zlib.crc32(f"exp058-train-{iid}-{k}".encode()))
        for j in range(N_ROT):
            phi = r.uniform(0, 180) if j else 0.0
            # ndimage.rotate by +phi turns a line at theta into one at theta - phi
            x = ndimage.rotate(big, phi, reshape=False, order=1, mode="reflect")[BIG - HALF:BIG + HALF,
                                                                                 BIG - HALF:BIG + HALF]
            t = (theta - phi) % 180.0
            if r.random() < 0.5:                       # mirror left-right: theta → 180 - theta
                x, t = x[:, ::-1], (180.0 - t) % 180.0
            X.append(x)
            Xa.append(perturb(x, r))
            T.append(t)
    return np.array(X), np.array(Xa), np.array(T)


def val_set(patches):
    X, T, C = [], [], []
    for k, (iid, theta, big) in enumerate(patches):
        p = big[BIG - HALF:BIG + HALF, BIG - HALF:BIG + HALF]
        for family, severity, fn in e45.IMAGE_CONDITIONS:
            X.append(fn(p, e45.rng_for("exp058-val", iid, k, family, severity)))
            T.append(theta)
            C.append(family)
    return np.array(X), np.array(T), np.array(C)


def null_training_set(patches):
    """No-orientation photo patches, flipped at random; the augmented copy is perturbed."""
    X, Xa = [], []
    for k, (iid, x) in enumerate(patches):
        r = np.random.default_rng(zlib.crc32(f"exp058-null-{iid}-{k}".encode()))
        if r.random() < 0.5:
            x = x[:, ::-1]
        X.append(x)
        Xa.append(perturb(x, r))
    return np.array(X), np.array(Xa)


def embeddings():
    tr = cache("train_patches", lambda: split_patches("train"))
    va = cache("val_patches", lambda: split_patches("val"))
    X, Xa, T = training_set(tr)
    Xv, Tv, Cv = val_set(va)
    Xn, Xna = null_training_set(cache("train_null_patches", lambda: flat_patches("train", 8)))
    Xvn = np.array([x for _, x in cache("val_null_patches", lambda: flat_patches("val", 4))])
    noise = np.array([e44.isotropic(np.random.default_rng(50_000 + s), "white") for s in range(60)])
    out = {"T": T, "Tv": Tv, "Cv": Cv, "n_train_patches": len(tr), "n_train_images": len({p[0] for p in tr}),
           "n_val_patches": len(va), "n_val_images": len({p[0] for p in va}),
           "n_train_null": len(Xn), "n_val_null": len(Xvn)}
    for pooling in POOLINGS:
        out[pooling] = {"E": image_learned.embed(X, pooling=pooling), "Ea": image_learned.embed(Xa, pooling=pooling),
                        "Ev": image_learned.embed(Xv, pooling=pooling),
                        "En": image_learned.embed(Xn, pooling=pooling), "Ena": image_learned.embed(Xna, pooling=pooling),
                        "Evn": image_learned.embed(Xvn, pooling=pooling),
                        "Enoise": image_learned.embed(noise, pooling=pooling)}
    return out


def gross(theta, truth):
    return np.abs(((np.asarray(theta) - np.asarray(truth) + 90.0) % 180.0) - 90.0) > e44.GROSS_DEG


def fit_heads(emb):
    """Choose pooling and ridge strength on val, for each read-out; calibrate on val."""
    E_key = {"cnn_photo": ("Ea", None), "cnn_photo_clean": ("E", None), "cnn_photo_null": ("Ea", "Ena")}
    heads, calib, chosen, grid = {}, {}, {}, {}
    for name, (key, null_key) in E_key.items():
        best = None
        for pooling in POOLINGS:
            for l2 in L2_GRID:
                h = OrientationHead.fit(emb[pooling][key], emb["T"], l2=l2,
                                        E_null=emb[pooling][null_key] if null_key else None)
                g = float(gross(h.predict(emb[pooling]["Ev"])[0], emb["Tv"]).mean())
                grid[f"{name} {pooling} l2={l2:g}"] = g
                if best is None or g < best[0]:
                    best = (g, pooling, l2, h)
        g, pooling, l2, h = best
        heads[name] = h
        chosen[name] = {"pooling": pooling, "l2": l2, "val_gross": g}
        lo = round(float(h.predict(emb[pooling]["Enoise"])[1].max()), 3)
        clean = emb["Cv"] == "clean"
        hi = round(float(np.percentile(h.predict(emb[pooling]["Ev"][clean])[1], 5)), 3)
        calib[name] = (lo, hi) if lo < hi else (0.0, hi)        # same fallback as Exp 052
        calib[f"{name}_raw_rule"] = (lo, hi)
        chosen[name]["val_null_trusted"] = float(np.mean(
            e52.scale(h.predict(emb[pooling]["Evn"])[1], *calib[name]) >= VOTING_THRESHOLD))
        h.save(os.path.join(CACHE, f"exp058_{name}_head.npz"))
    val_per_condition = {
        name: {c: float(gross(heads[name].predict(emb[chosen[name]["pooling"]]["Ev"][emb["Cv"] == c])[0],
                              emb["Tv"][emb["Cv"] == c]).mean()) for c in dict.fromkeys(emb["Cv"])}
        for name in heads}
    return heads, calib, chosen, grid, val_per_condition


# ──────────────────────────────────────────────────────────────────────────────
# Test rows: the cached Exp 045 / 044 readings plus the new lenses
# ──────────────────────────────────────────────────────────────────────────────

def test_embeddings(name, poolings):
    X, N = e52.image_patches(name)
    return {p: {"E": image_learned.embed(X, pooling=p),
                "E_null": image_learned.embed(N, pooling=p) if len(N) else np.zeros((0, 1))} for p in poolings}


def rows_with_new(name, heads, calib, chosen):
    rows, nulls, mm = e52.image_rows(name, {"cnn_aug": OrientationHead.load(
        os.path.join(CACHE, "exp052_cnn_aug_head.npz"))}, {"cnn_aug": CNN_AUG_CALIBRATION})
    assert mm == 0, f"rebuilt patches do not match the cached readings of {name}"
    raw_rows, raw_nulls = e52.image_cached(name)
    keep = [i for i, r in enumerate(raw_rows) if not r["empty"]]
    keep0 = [i for i, r in enumerate(raw_nulls) if not r["empty"]]
    emb = cache(f"emb_{name}", lambda: test_embeddings(name, POOLINGS))
    for n, h in heads.items():
        th, raw = h.predict(emb[chosen[n]["pooling"]]["E"])
        conf = e52.scale(raw, *calib[n])
        for r, i in zip(rows, keep):
            r["readings"][n] = (float(th[i]), float(conf[i]))
        if len(nulls):
            th0, raw0 = h.predict(emb[chosen[n]["pooling"]]["E_null"])
            conf0 = e52.scale(raw0, *calib[n])
            for r, i in zip(nulls, keep0):
                r["readings"][n] = (float(th0[i]), float(conf0[i]))
    return rows, nulls


def photo_null_rows(heads, calib, chosen):
    """Readings of every lens on no-orientation patches of the BSDS500 test split."""
    def build():
        P = flat_patches("test", 2, seed="flat-test")
        analyzer = e44.make_image_analyzer()
        rows = [e44.patch_row(analyzer, x, None, {"group": iid}) for iid, x in P]
        X = np.array([x for _, x in P])
        return {"rows": rows, **{p: image_learned.embed(X, pooling=p) for p in POOLINGS}}
    d = cache("photo_null", build)
    keep = [i for i, r in enumerate(d["rows"]) if not r["empty"]]
    aug = OrientationHead.load(os.path.join(CACHE, "exp052_cnn_aug_head.npz"))
    th_a, raw_a = aug.predict(d["global"])
    pred = {"cnn_aug": (th_a, e52.scale(raw_a, *CNN_AUG_CALIBRATION))}
    for n, h in heads.items():
        th, raw = h.predict(d[chosen[n]["pooling"]])
        pred[n] = (th, e52.scale(raw, *calib[n]))
    out = []
    for i in keep:
        rd = {n: d["rows"][i]["readings"][n] for n in CLASSIC}
        rd.update({n: (float(v[0][i]), float(v[1][i])) for n, v in pred.items()})
        out.append({"readings": rd})
    return out, len(d["rows"])


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def frames(rows, lenses):
    return [e52.readings(e52.Image, r, lenses) for r in rows]


def admit(rows, null_sets, pool, cand):
    """Admission against the first null set; false answers on every null set."""
    lenses = tuple(pool) + (cand,)
    runs = {k: admission_test(frames(rows, lenses), [r["truth"] for r in rows], pool, cand,
                              e52.Image.comparator(2),
                              is_wrong=lambda est, t: e44.err_deg(est, t) > e44.GROSS_DEG,
                              nulls=frames(nulls, lenses), groups=[r["group"] for r in rows])
            for k, nulls in null_sets.items()}
    a = runs[next(iter(null_sets))]
    return {"null_set": next(iter(null_sets)),
            "false_answer_by_null": {k: [r.without["false_answer"], r.with_candidate["false_answer"]]
                                     for k, r in runs.items()},"pool": list(a.pool), "coverage": a.coverage, "coverage_unanswered": a.coverage_unanswered,
            "n_unanswered": a.n_unanswered, "candidate_gross": a.candidate_gross, "without": a.without,
            "with": a.with_candidate, "value": a.value, "gross_all_ci95": list(a.gross_all_ci95),
            "admitted": a.admitted}


def per_condition(rows, W):
    conds = list(dict.fromkeys(r["family"] for r in rows))
    fam = np.array([r["family"] for r in rows])
    return {n: {"all": float(w.mean()), **{c: float(w[fam == c].mean()) for c in conds}} for n, w in W.items()}


def family_phi(W, lens):
    out = {}
    for fam, members in FAMILIES.items():
        vals = {m: e52.phi(W[lens], W[m]) for m in members}
        out[fam] = {"mean": float(np.nanmean(list(vals.values()))), **vals}
    return out


def drop_one(rows, nulls, pool):
    old = e55.POOLS["image"]
    e55.POOLS["image"] = tuple(pool)
    try:
        d = e55.analyze("image", rows, nulls)
    finally:
        e55.POOLS["image"] = old
    return {"full": d["sets"]["full"], "value": d["value"], "overlap": d["overlap"]}


def analyze(name, rows, null_sets):
    lenses = CLASSIC + ("cnn_aug",) + NEW
    W = e52.wrong_matrix(e52.Image, rows, lenses)
    out = {"n": len(rows), "n_null": {k: len(v) for k, v in null_sets.items()}, "gross": per_condition(rows, W),
           "phi": {n: family_phi(W, n) for n in NEW},
           "null_trusted": {n: {k: float(np.mean([v["readings"][n][1] >= VOTING_THRESHOLD for v in nulls]))
                                for k, nulls in null_sets.items()} for n in lenses},
           "admission": {}}
    for cand in NEW:
        out["admission"][f"{cand} vs exp055 pool"] = admit(rows, null_sets, OLD_POOL, cand)
        out["admission"][f"{cand} vs classic pool"] = admit(rows, null_sets, CLASSIC, cand)
    out["admission"]["cnn_aug given classic + cnn_photo"] = admit(rows, null_sets, CLASSIC + ("cnn_photo",), "cnn_aug")
    out["drop_one"] = drop_one(rows, null_sets["synthetic"], OLD_POOL + ("cnn_photo",))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import torch
    import torchvision

    t0 = time.time()
    emb = cache("embeddings", embeddings)
    heads, calib, chosen, grid, val_cond = fit_heads(emb)
    s = {"versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
         "weights_sha256": image_learned._Backbone.weights_sha256(),
         "data": {k: emb[k] for k in ("n_train_patches", "n_train_images", "n_val_patches", "n_val_images")}
         | {"n_train_copies": int(len(emb["T"])), "rotations": N_ROT},
         "chosen": chosen, "val_grid": grid, "val_per_condition": val_cond, "calibration": calib, "datasets": {}}
    print("chosen", chosen, "\ncalibration", calib)
    syn_rows, syn_null = rows_with_new("synthetic", heads, calib, chosen)
    photo_null, n_photo_null = photo_null_rows(heads, calib, chosen)
    s["data"]["n_test_null_photo"] = n_photo_null
    s["data"]["n_test_null_photo_nonempty"] = len(photo_null)
    for label, rows, null_sets in (("BSDS500", rows_with_new("bsds", heads, calib, chosen)[0],
                                    {"synthetic": syn_null, "photo_no_boundary": photo_null}),
                                   ("synthetic", syn_rows, {"synthetic": syn_null, "photo_no_boundary": photo_null})):
        s["datasets"][label] = analyze(label, rows, null_sets)
        print(f"{label}: {len(rows)} rows, {time.time() - t0:.0f} s")
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    print("\nval gross error per condition:", json.dumps(s["val_per_condition"], indent=1))
    for label, d in s["datasets"].items():
        print(f"\n=== {label} (n = {d['n']}, null {d['n_null']})")
        print("  trusted on null input: " + ", ".join(
            f"{n} " + "/".join(f"{v:.0%}" for v in t.values()) for n, t in d["null_trusted"].items())
              + f"  ({'/'.join(d['n_null'])})")
        conds = [c for c in next(iter(d["gross"].values())) if c != "all"]
        print(f"  {'lens':16s} {'all':>6s} " + " ".join(f"{c:>9s}" for c in conds))
        for n, g in d["gross"].items():
            print(f"  {n:16s} {g['all']:6.1%} " + " ".join(f"{g[c]:9.1%}" for c in conds))
        for n, ph in d["phi"].items():
            print(f"  φ {n}: " + ", ".join(f"{f} {v['mean']:+.2f}" for f, v in ph.items()))
        print(f"  {'admission':36s} {'cover':>6s} {'cov|abst':>8s} {'Δgross_all':>10s} {'CI95':>17s} "
              f"{'Δgross|ans':>10s} {'Δabstain':>9s} {'false syn':>12s} {'no-bound.':>12s}  admitted")
        for k, a in d["admission"].items():
            v, ci = a["value"], a["gross_all_ci95"]
            print(f"  {k:36s} {a['coverage']:6.1%} {a['coverage_unanswered']:8.1%} {v['gross_all']:+10.1%} "
                  f"[{ci[0]:+6.1%}, {ci[1]:+6.1%}] {v['gross_answer']:+10.1%} {v['abstention']:+9.1%} "
                  + " ".join(f"{fa[0]:5.1%}→{fa[1]:5.1%}" for fa in (a["false_answer_by_null"]["synthetic"],
                                                    a["false_answer_by_null"]["photo_no_boundary"]))
                  + f"  {a['admitted']}")
        f = d["drop_one"]["full"]
        print(f"  exp055 pool + cnn_photo: answer {f['answer']:.1%}, gross|ans {f['gross_answer']:.1%}, "
              f"gross_all {f['gross_all']:.1%}, false {f['false_answer']:.1%}")
        for L, v in d["drop_one"]["value"].items():
            print(f"    drop {L:10s} Δgross_all {v['gross_all']:+6.1%}  coverage {d['drop_one']['overlap'][L]['coverage']:5.1%}")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = s["datasets"]["BSDS500"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    lenses = list(d["gross"])
    conds = [c for c in d["gross"][lenses[0]] if c != "all"]
    w = 0.8 / len(lenses)
    for i, n in enumerate(lenses):
        col = {"cnn_photo": "#d4724a", "cnn_photo_null": "#e89a74", "cnn_photo_clean": "#f0c9b5",
               "cnn_aug": "#4a7fd4"}.get(n, None)
        ax.bar(np.arange(len(conds)) + i * w, [d["gross"][n][c] for c in conds], w, label=n, color=col,
               alpha=1.0 if col else 0.45)
    ax.set_xticks(np.arange(len(conds)) + 0.4 - w / 2, conds, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("BSDS500 test: gross error (>10°) per lens", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    keys = list(d["admission"])
    v = np.array([d["admission"][k]["value"]["gross_all"] for k in keys])
    ci = np.array([d["admission"][k]["gross_all_ci95"] for k in keys])
    y = np.arange(len(keys))
    ax.barh(y, v, color=["#d4724a" if d["admission"][k]["admitted"] else "#999999" for k in keys],
            xerr=[v - ci[:, 0], ci[:, 1] - v], ecolor="k", capsize=2)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, [f"{k}\ncov {d['admission'][k]['coverage']:.1%}, cov|abst {d['admission'][k]['coverage_unanswered']:.0%}"
                      for k in keys], fontsize=7)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("admission: Δ gross_all when added (orange = admitted)", fontsize=10)

    ax = axes[2]
    dv, ov = d["drop_one"]["value"], d["drop_one"]["overlap"]
    for L in dv:
        col = "#d4724a" if L == "cnn_photo" else ("#4a7fd4" if L == "cnn_aug" else "#777777")
        ax.scatter(ov[L]["coverage"], dv[L]["gross_all"], c=col)
        ax.annotate(L, (ov[L]["coverage"], dv[L]["gross_all"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("coverage (right where < 2 others are)", fontsize=8)
    ax.set_ylabel("value: gross_all added by dropping it", fontsize=8)
    ax.set_title("Exp 055 drop-one on the Exp 055 pool + cnn_photo", fontsize=10)
    fig.suptitle("Exp 058: a learned orientation lens trained on BSDS500 photos", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
