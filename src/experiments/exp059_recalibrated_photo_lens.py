"""
Experiment 059: recalibrate against smooth unoriented input, then re-run admission (issue #35)
==============================================================================================
Exp 058's photo lens (``cnn_photo``) cut gross_all on BSDS500 by 8.9 pts but was
not admitted: it made the pool answer 7 of 90 synthetic unoriented patches
instead of 3. Its confidence trusts no white-noise patch but 18% of the smooth
"blob" patches. The project confidence rule (lo = the strongest raw score over 60
white-noise patches, hi = the 5th percentile over clean patterns) never shows a
lens smooth unoriented input.

Rules compared (confidence = clip((raw - lo) / (hi - lo)), trusted at ≥ 0.5)
  white        the project rule, as used up to Exp 058
  white+blob   lo = the strongest raw score over the same 60 white-noise patches
               *and* 60 smooth isotropic "blob" patches (``e44.isotropic(.., "blob")``,
               seeds 60_000+s; the null set uses 20_000+s, so no null patch is
               seen, but the null *type* now is). Never below the old lo.
               If lo reaches hi (noise scores above 5% of clean patterns), the rule
               is undefined; hi then falls back to the median over clean patterns,
               and if lo reaches that too, the lens is never trusted. The summary
               says which lenses fell back.
hi references: the clean base patterns of Exp 044 (classic lenses), the clean
synthetic patches (cnn_aug, as in Exp 052), the clean BSDS500 val patches
(cnn_photo, as in Exp 058).

Raw scores are re-read for the classic lenses on the rebuilt Exp 044/045 patches
(the cached readings keep only the clipped confidence); the estimates are the
cached ones, and the old confidences are reproduced from the raw scores as a check.
CNN raw scores come from the Exp 052 / 058 embedding caches and saved heads.

Questions
  1. Which lenses does the new rule change, and how much of each lens's trust on
     clean input does it cost?
  2. Admission of cnn_photo (``core.admission_test``) with (a) only cnn_photo
     recalibrated, (b) every lens recalibrated, against the Exp 055 pool and the
     classic pool, on BSDS500 and synthetic patches.
  3. Does the new rule help the pool itself (Exp 055 pool, white vs white+blob)?
  4. The trade-off: cnn_photo's trust threshold swept over its raw score, with
     the pool unchanged: Δ gross_all and false answers at each threshold.
  5. Lead mode (Exp 056): Radon answering alone under both rules.

Run:  python src/experiments/exp059_recalibrated_photo_lens.py   (needs the Exp 045,
      046, 052 and 058 caches in data/processed)
Outputs: results/exp059_recalibrated_photo_lens.png, results/exp059_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp044_image_disagreement as e44  # noqa: E402
import exp052_learned_lens as e52  # noqa: E402
import exp055_lens_value as e55  # noqa: E402
import exp058_photo_lens as e58  # noqa: E402
from core import ConfidenceThresholdDetector, admission_test  # noqa: E402
from image_medium import LENSES  # noqa: E402
from image_medium import learned as image_learned  # noqa: E402
from image_medium.learned import OrientationHead  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD, OrientationComparator  # noqa: E402

CACHE = e52.CACHE
OUT_JSON = os.path.join(ROOT, "results", "exp059_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp059_recalibrated_photo_lens.png")
CLASSIC = e52.IMAGE_BASE
OLD_POOL = e55.POOLS["image"]
ALL = CLASSIC + ("cnn_aug", "cnn_photo")
N_CAL = 60
POOLING = {"cnn_aug": "global", "cnn_photo": "centre"}
HEADS = {"cnn_aug": "exp052_cnn_aug_head.npz", "cnn_photo": "exp058_cnn_photo_head.npz"}


def cache(name, fn):
    path = os.path.join(CACHE, f"exp059_{name}.pkl")
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
# Raw scores
# ──────────────────────────────────────────────────────────────────────────────

def classic_raw(X):
    lenses = {n: c() for n, c in LENSES.items()}
    return {n: np.array([lens.orientation(x)[1] for x in X]) for n, lens in lenses.items()}


def cnn_raw(X):
    out = {}
    for n, f in HEADS.items():
        h = OrientationHead.load(os.path.join(CACHE, f))
        out[n] = h.predict(image_learned.embed(X, pooling=POOLING[n]))[1]
    return out


def calibration_inputs():
    white = np.array([e44.isotropic(np.random.default_rng(50_000 + s), "white") for s in range(N_CAL)])
    blob = np.array([e44.isotropic(np.random.default_rng(60_000 + s), "blob") for s in range(N_CAL)])
    clean_syn = np.array([e44.make_patch(v, th, "clean", 0, 0, k) for v, k, th in e44.base_signals()])
    return {
        "white": {**classic_raw(white), **cnn_raw(white)},
        "blob": {**classic_raw(blob), **cnn_raw(blob)},
        "clean_classic": classic_raw(clean_syn),
    }


def clean_refs(cal):
    """Clean reference raw scores per lens (the 'hi' side of each lens's rule)."""
    refs = {n: cal["clean_classic"][n] for n in CLASSIC}
    rows, _ = e52.image_cached("synthetic")
    clean = np.array([r["family"] == "clean" for r in rows])
    E = e52.cache("cnn_synthetic", None)["E"]
    refs["cnn_aug"] = OrientationHead.load(os.path.join(CACHE, HEADS["cnn_aug"])).predict(E[clean])[1]
    emb = e58.cache("embeddings", None)
    v_clean = emb["Cv"] == "clean"
    refs["cnn_photo"] = OrientationHead.load(os.path.join(CACHE, HEADS["cnn_photo"])).predict(
        emb["centre"]["Ev"][v_clean])[1]
    return refs


def calibrations(cal, refs):
    out = {"white": {}, "white+blob": {}}
    info = {}
    for n in ALL:
        lo_w = float(np.max(cal["white"][n]))
        lo_b = max(lo_w, float(np.max(cal["blob"][n])))
        hi = float(np.percentile(refs[n], 5))
        med = float(np.median(refs[n]))
        out["white"][n] = (lo_w, hi) if lo_w < hi else (0.0, hi)        # Exp 052 fallback for the old rule
        if lo_b < hi:
            out["white+blob"][n], how = (lo_b, hi), "rule"
        elif lo_b < med:
            out["white+blob"][n], how = (lo_b, med), "hi = clean median"
        else:
            out["white+blob"][n], how = (lo_b, lo_b + 1e-9), "never trusted below the strongest noise"
        info[n] = {"lo_white": lo_w, "lo_blob_max": float(np.max(cal["blob"][n])), "hi": hi, "clean_median": med,
                   "white+blob": how,
                   "clean_trusted": {k: float(np.mean(e52.scale(refs[n], *out[k][n]) >= VOTING_THRESHOLD))
                                     for k in out}}
    return out, info


def dataset_raw(name):
    """Raw scores of every lens on the rebuilt patches (rows and nulls), with a check
    that the cached confidences come back from them under the old calibration."""
    X, N = e52.image_patches(name)
    rows, nulls = e52.image_cached(name)
    out = {"X": {**classic_raw(X)}, "N": {**classic_raw(N)} if len(N) else {}}
    if name == "bsds":
        e = e58.cache("emb_bsds", None)
    else:
        e = e58.cache("emb_synthetic", None)
    e52g = e52.cache(f"cnn_{name}", None)
    for n, f in HEADS.items():
        h = OrientationHead.load(os.path.join(CACHE, f))
        src_X = e52g["E"] if POOLING[n] == "global" else e[POOLING[n]]["E"]
        src_N = e52g["E_null"] if POOLING[n] == "global" else e[POOLING[n]]["E_null"]
        out["X"][n] = h.predict(src_X)[1]
        if len(N):
            out["N"][n] = h.predict(src_N)[1]
    mismatch = 0
    for i, r in enumerate(rows):
        if r["empty"]:
            continue
        for n in CLASSIC:
            old = r["readings"][n][1]
            new = e52.scale(out["X"][n][i], *LENSES[n].contrast) if r["readings"][n][0] is not None else 0.0
            mismatch += int(abs(float(new) - old) > 1e-6)
    out["check_mismatch"] = mismatch
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Rows under a calibration
# ──────────────────────────────────────────────────────────────────────────────

def build_rows(name, raw, calib):
    """(rows, nulls) with every lens's cached estimate and a confidence from ``calib``."""
    rows, nulls = e52.image_cached(name)
    thetas = {n: OrientationHead.load(os.path.join(CACHE, HEADS[n])) for n in HEADS}
    e58_emb = e58.cache(f"emb_{name}", None)
    e52_emb = e52.cache(f"cnn_{name}", None)
    th = {}
    for n, h in thetas.items():
        srcX = e52_emb["E"] if POOLING[n] == "global" else e58_emb[POOLING[n]]["E"]
        srcN = e52_emb["E_null"] if POOLING[n] == "global" else e58_emb[POOLING[n]]["E_null"]
        th[n] = (h.predict(srcX)[0], h.predict(srcN)[0] if len(raw["N"]) else None)

    def make(src, k, i, r):
        rd = {}
        for n in CLASSIC:
            est = r["readings"][n][0]
            rd[n] = (est, float(e52.scale(raw[src][n][i], *calib[n])) if est is not None else 0.0)
        for n in HEADS:
            rd[n] = (float(th[n][k][i]), float(e52.scale(raw[src][n][i], *calib[n])))
        return rd

    out = [{"truth": r["truth"], "group": str(r["group"]), "family": r["family"], "readings": make("X", 0, i, r)}
           for i, r in enumerate(rows) if not r["empty"]]
    out0 = [{"readings": make("N", 1, i, r)} for i, r in enumerate(nulls) if not r["empty"]] if len(raw["N"]) else []
    return out, out0


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def frames(rows, lenses):
    return [e52.readings(e52.Image, r, lenses) for r in rows]


def admit(rows, nulls, pool, cand="cnn_photo"):
    lenses = tuple(pool) + (cand,)
    a = admission_test(frames(rows, lenses), [r["truth"] for r in rows], pool, cand, e52.Image.comparator(2),
                       is_wrong=lambda est, t: e44.err_deg(est, t) > e44.GROSS_DEG,
                       nulls=frames(nulls, lenses), groups=[r["group"] for r in rows])
    return {"coverage": a.coverage, "coverage_unanswered": a.coverage_unanswered, "without": a.without,
            "with": a.with_candidate, "value": a.value, "gross_all_ci95": list(a.gross_all_ci95),
            "admitted": a.admitted}


def pool_metrics(rows, nulls, lenses, comp=None):
    comp = comp or e52.Image.comparator(2)
    est = [comp.compare(e52.readings(e52.Image, r, lenses)).consensus for r in rows]
    wrong = np.array([e52.Image.wrong(e, r) for e, r in zip(est, rows)])
    ans = np.array([e is not None for e in est])
    false = np.mean([comp.compare(e52.readings(e52.Image, r, lenses)).consensus is not None for r in nulls])
    return {"answer": float(ans.mean()), "gross_answer": float(wrong[ans].mean()) if ans.any() else float("nan"),
            "gross_all": float(wrong.mean()), "false_answer": float(false), "_wrong": wrong}


def lead_radon(rows, nulls, lenses):
    comp = OrientationComparator(min_agreeing=2, lead="radon", detector=ConfidenceThresholdDetector(VOTING_THRESHOLD))
    m = pool_metrics(rows, nulls, lenses, comp)
    m.pop("_wrong")
    return m


def analyze(name, raw, calibs, null_name="synthetic", null_raw=None):
    """Admission, pool and Radon-lead metrics under the old rule, with only cnn_photo
    recalibrated, and with every lens recalibrated."""
    res = {"admission": {}, "pool": {}, "lead_radon": {}}
    mixed = {**calibs["white"], "cnn_photo": calibs["white+blob"]["cnn_photo"]}
    for label, calib in (("white", calibs["white"]), ("photo lens recalibrated", mixed),
                         ("all recalibrated", calibs["white+blob"])):
        rows, nulls = build_rows(name, raw, calib)
        if not nulls:                                   # BSDS500 has no null set: the synthetic one
            nulls = build_rows(null_name, null_raw, calib)[1]
        res["admission"][label] = {"exp055 pool": admit(rows, nulls, OLD_POOL),
                                   "classic pool": admit(rows, nulls, CLASSIC)}
        p = pool_metrics(rows, nulls, OLD_POOL)
        p.pop("_wrong")
        res["pool"][label] = p
        res["lead_radon"][label] = lead_radon(rows, nulls, OLD_POOL)
        res.setdefault("n", len(rows))
        res.setdefault("n_null", len(nulls))
    return res


def main():
    t0 = time.time()
    cal = cache("calibration_inputs", calibration_inputs)
    refs = clean_refs(cal)
    calibs, info = calibrations(cal, refs)
    raw = {name: cache(f"raw_{name}", lambda name=name: dataset_raw(name)) for name in ("bsds", "synthetic")}
    s = {"calibration": {k: {n: list(v) for n, v in c.items()} for k, c in calibs.items()}, "lens_info": info,
         "check_mismatch": {k: v["check_mismatch"] for k, v in raw.items()}, "datasets": {}}
    for label, name in (("BSDS500", "bsds"), ("synthetic", "synthetic")):
        s["datasets"][label] = analyze(name, raw[name], calibs, null_raw=raw["synthetic"])
        print(f"{label}: {time.time() - t0:.0f} s")
    s["sweep"] = sweep_bsds(raw, calibs["white"])
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def sweep_bsds(raw, calib_white):
    """cnn_photo's trust threshold over its raw score on BSDS500 (synthetic nulls);
    every other lens on the old rule."""
    rows, _ = build_rows("bsds", raw["bsds"], calib_white)
    _, nulls = build_rows("synthetic", raw["synthetic"], calib_white)
    keep = [i for i, r in enumerate(e52.image_cached("bsds")[0]) if not r["empty"]]
    keep0 = [i for i, r in enumerate(e52.image_cached("synthetic")[1]) if not r["empty"]]
    raw_X, raw_N = raw["bsds"]["X"]["cnn_photo"][keep], raw["synthetic"]["N"]["cnn_photo"][keep0]
    base = pool_metrics(rows, nulls, OLD_POOL)
    pts = []
    for t in np.round(np.arange(0.30, 0.80, 0.025), 3):
        for r, v in zip(rows, raw_X):
            r["readings"]["cnn_photo"] = (r["readings"]["cnn_photo"][0], 1.0 if v >= t else 0.0)
        for r, v in zip(nulls, raw_N):
            r["readings"]["cnn_photo"] = (r["readings"]["cnn_photo"][0], 1.0 if v >= t else 0.0)
        m = pool_metrics(rows, nulls, OLD_POOL + ("cnn_photo",))
        pts.append({"threshold": float(t), "d_gross_all": m["gross_all"] - base["gross_all"],
                    "false_answer": m["false_answer"], "answer": m["answer"], "gross_answer": m["gross_answer"],
                    "photo_trusted": float(np.mean(raw_X >= t))})
    base.pop("_wrong")
    return {"base": base, "points": pts}


def report(s):
    print("check (cached confidences reproduced, mismatches):", s["check_mismatch"])
    print(f"\n{'lens':10s} {'lo white':>9s} {'blob max':>9s} {'hi':>7s} {'median':>7s}  rule            "
          f"clean trusted white → white+blob")
    for n, i in s["lens_info"].items():
        ct = i["clean_trusted"]
        print(f"{n:10s} {i['lo_white']:9.3f} {i['lo_blob_max']:9.3f} {i['hi']:7.3f} {i['clean_median']:7.3f}  "
              f"{i['white+blob']:15s} {ct['white']:.0%} → {ct['white+blob']:.0%}")
    for label, d in s["datasets"].items():
        print(f"\n=== {label} (n = {d['n']}, null {d['n_null']})")
        for cal, adm in d["admission"].items():
            p, L = d["pool"][cal], d["lead_radon"][cal]
            print(f"  [{cal}] exp055 pool: answer {p['answer']:.1%}, gross_all {p['gross_all']:.1%}, "
                  f"false {p['false_answer']:.1%} | radon lead: gross_all {L['gross_all']:.1%}, false {L['false_answer']:.1%}")
            for pool, a in adm.items():
                v, ci = a["value"], a["gross_all_ci95"]
                print(f"     + cnn_photo vs {pool:12s} cov {a['coverage']:5.1%}, right on unanswered "
                      f"{a['coverage_unanswered']:5.1%}, Δgross_all {v['gross_all']:+6.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}], "
                      f"Δabstain {v['abstention']:+6.1%}, false {a['without']['false_answer']:.1%} → "
                      f"{a['with']['false_answer']:.1%}, gross_all {a['with']['gross_all']:.1%}  admitted {a['admitted']}")
    print("\nsweep of cnn_photo's raw trust threshold (BSDS500, Exp 055 pool on the old rule):")
    for p in s["sweep"]["points"]:
        print(f"  t {p['threshold']:.3f}: trusted on photos {p['photo_trusted']:5.1%}, Δgross_all {p['d_gross_all']:+6.1%}, "
              f"false {p['false_answer']:5.1%}")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    pts = s["sweep"]["points"]
    t = [p["threshold"] for p in pts]
    ax.plot(t, [-p["d_gross_all"] for p in pts], "o-", color="#d4724a", label="gain in gross_all (BSDS500)")
    ax.plot(t, [p["false_answer"] for p in pts], "s-", color="#4a7fd4", label="false answers (90 null patches)")
    base_false = s["sweep"]["base"]["false_answer"]
    ax.axhline(base_false, color="#4a7fd4", ls=":", lw=1)
    ax.axhline(base_false + 0.02, color="#4a7fd4", ls="--", lw=1, label="admission limit (+2 pts)")
    w, b = s["calibration"]["white"]["cnn_photo"], s["calibration"]["white+blob"]["cnn_photo"]
    for (lo, hi), lab in ((w, "white rule"), (b, "white+blob rule")):
        ax.axvline((lo + hi) / 2, color="k", lw=0.8, ls="-" if lab == "white rule" else "-.")
        ax.annotate(lab, ((lo + hi) / 2, ax.get_ylim()[1] * 0.9), fontsize=7, rotation=90, xytext=(3, 0),
                    textcoords="offset points")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("cnn_photo trust threshold (raw score)", fontsize=8)
    ax.set_title("trade-off: gain vs false answers when adding cnn_photo", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    d = s["datasets"]["BSDS500"]
    cals = list(d["admission"])
    v = np.array([d["admission"][c]["exp055 pool"]["value"]["gross_all"] for c in cals])
    ci = np.array([d["admission"][c]["exp055 pool"]["gross_all_ci95"] for c in cals])
    fa = [d["admission"][c]["exp055 pool"]["with"]["false_answer"] - d["admission"][c]["exp055 pool"]["without"]["false_answer"]
          for c in cals]
    y = np.arange(len(cals))
    ax.barh(y, v, color=["#d4724a" if d["admission"][c]["exp055 pool"]["admitted"] else "#999999" for c in cals],
            xerr=[v - ci[:, 0], ci[:, 1] - v], ecolor="k", capsize=2)
    ax.set_yticks(y, [f"{c}\nΔ false answers {f:+.1%}" for c, f in zip(cals, fa)], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("BSDS500: Δ gross_all when cnn_photo joins the Exp 055 pool (orange = admitted)", fontsize=9)
    fig.suptitle("Exp 059: recalibrating confidence against smooth unoriented input", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
