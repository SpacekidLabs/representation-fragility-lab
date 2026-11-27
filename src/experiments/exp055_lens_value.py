"""
Experiment 055: which lenses are worth keeping (issue #29)
==========================================================
Every earlier experiment added lenses; none asked what each one is worth. Here
every lens is removed in turn (and every pair of lenses), and we measure what
the system loses.

Lens pools (all readings come from the caches of Exp 045, 046 and 052; no
lens is re-run)
  audio   stft, acf, cepstrum (classic) + crepe (learned, Exp 052)
          on synthetic (Exp 042 frames), vocadito and MDB-stem-synth
  images  tensor, hog, fft, gabor, radon (classic) + cnn_aug (learned, the
          augmented read-out of Exp 052) on synthetic (Exp 044) and BSDS500
  Null input (unpitched / unoriented) for the false-answer rate: the cached
  null frames of each audio dataset; the synthetic unoriented patches for both
  image datasets (BSDS500 has none of its own, as in Exp 049).

Lens sets: the full pool, every drop-1 set, every drop-2 set. The core is the
unchanged quorum-2 comparator (min_agreeing=2) of Exp 046.

Metrics per set (gross error: >50 cents / >10 degrees)
  warning AUC   for every lens left in the set, ROC AUC of its divergence (mean
                pair distance to the others, no training) for "this lens is
                grossly wrong"; Exp 042-045's "disagreement warns" claim
  consensus AUC ROC AUC of the overall disagreement for "the consensus is
                wrong", over answered frames
  answer        share of pitched / oriented frames the quorum answers
                (abstention = 1 - answer)
  gross_answer  gross error among the answers
  gross_all     gross error counting an abstention as wrong
  false_answer  share of null frames that get an answer

Marginal value of lens L = metric(full) - metric(full without L), with a
group-bootstrap 95% CI on the gross_all change (whole recordings / images
resampled). Pair interaction = value of dropping the pair - sum of the two
single values: positive means the two lenses back each other up (dropping
both hurts more than the sum), so either alone looks cheap.

Blind-spot overlap (Exp 043/052) per lens L in its pool
  mean phi      mean phi correlation of L's gross errors with each other lens's
  max phi       the same, with its closest lens
  coverage      share of frames where L is right but fewer than two other lenses
                are; the frames where only L can complete a right quorum
These are related to the gross_all value with Spearman correlations over every
(lens, dataset) point.

Run:  python src/experiments/exp055_lens_value.py     (needs the Exp 045, 046
      and 052 caches in data/processed)
Outputs: results/exp055_lens_value.png, results/exp055_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp045_real_data as e45  # noqa: E402
import exp052_learned_lens as e52  # noqa: E402
from framework.learned import CALIBRATION as CREPE_CALIBRATION  # noqa: E402
from image_medium.learned import OrientationHead  # noqa: E402

CACHE = e52.CACHE
OUT_JSON = os.path.join(ROOT, "results", "exp055_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp055_lens_value.png")
CNN_AUG_CALIBRATION = (0.284, 0.803)      # Exp 052 (results/exp052_summary.json)
POOLS = {"audio": e52.AUDIO_BASE + ("crepe",), "image": e52.IMAGE_BASE + ("cnn_aug",)}
LABELS = {"synthetic": "synthetic", "vocadito": "vocadito", "mdb_stem_synth": "MDB-stem-synth", "bsds": "BSDS500"}
MEDIUM = {"audio": e52.Audio, "image": e52.Image}


# ──────────────────────────────────────────────────────────────────────────────
# Rows (cached readings of every lens)
# ──────────────────────────────────────────────────────────────────────────────

def datasets():
    out = {}
    for name in e52.DATASETS["audio"]:
        rows, nulls, mm = e52.audio_rows(name, CREPE_CALIBRATION)
        assert mm == 0, f"crepe cache does not match the rows of {name}"
        out[("audio", name)] = (rows, nulls)
    head = OrientationHead.load(os.path.join(CACHE, "exp052_cnn_aug_head.npz"))
    heads, calib = {"cnn_aug": head}, {"cnn_aug": CNN_AUG_CALIBRATION}
    syn, syn_null, mm = e52.image_rows("synthetic", heads, calib)
    assert mm == 0
    out[("image", "synthetic")] = (syn, syn_null)
    bsds, _, mm = e52.image_rows("bsds", heads, calib)
    assert mm == 0
    out[("image", "bsds")] = (bsds, syn_null)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# One lens set
# ──────────────────────────────────────────────────────────────────────────────

def run_set(M, rows, nulls, lenses, W):
    comp = M.comparator(2)
    est, div, overall = [], {n: [] for n in lenses}, []
    for r in rows:
        rep = comp.compare(e52.readings(M, r, lenses))
        est.append(rep.consensus)
        overall.append(rep.overall)
        for n in lenses:
            div[n].append(rep.divergence[n])
    answer = np.array([e is not None for e in est])
    wrong = np.array([M.wrong(e, r) for e, r in zip(est, rows)])        # abstention counts as wrong
    overall = np.array(overall)
    cons_wrong = wrong[answer]
    warn = {n: e52.auc(div[n], W[n]) for n in lenses}
    false = float(np.mean([comp.compare(e52.readings(M, r, lenses)).consensus is not None for r in nulls]))
    return {"lenses": list(lenses),
            "warning_auc": warn,
            "warning_auc_mean": float(np.nanmean(list(warn.values()))),
            "consensus_auc": e52.auc(overall[answer], cons_wrong),
            "answer": float(answer.mean()),
            "gross_answer": float(cons_wrong.mean()) if answer.any() else float("nan"),
            "gross_all": float(wrong.mean()),
            "false_answer": false,
            "_wrong": wrong}


def overlap(W, pool):
    out = {}
    right = {n: ~W[n] for n in pool}
    for L in pool:
        others = [o for o in pool if o != L]
        phis = {o: e52.phi(W[L], W[o]) for o in others}
        n_right_others = np.sum([right[o] for o in others], axis=0)
        out[L] = {"gross": float(W[L].mean()),
                  "mean_phi": float(np.nanmean(list(phis.values()))),
                  "max_phi": float(np.nanmax(list(phis.values()))),
                  "closest": max(phis, key=lambda o: np.nan_to_num(phis[o], nan=-1)),
                  "coverage": float(np.mean(right[L] & (n_right_others < 2)))}
    return out


def analyze(medium, rows, nulls):
    M = MEDIUM[medium]
    pool = POOLS[medium]
    groups = np.array([r["group"] for r in rows])
    W = e52.wrong_matrix(M, rows, pool)
    sets = {"full": pool}
    sets.update({f"-{a}": tuple(n for n in pool if n != a) for a in pool})
    sets.update({f"-{a}-{b}": tuple(n for n in pool if n not in (a, b)) for a, b in combinations(pool, 2)})
    R = {k: run_set(M, rows, nulls, v, W) for k, v in sets.items()}
    full = R["full"]

    value = {}
    for L in pool:
        d = R[f"-{L}"]
        targets = [n for n in pool if n != L]
        value[L] = {
            # positive = the lens helps: removing it makes the metric worse
            "gross_all": d["gross_all"] - full["gross_all"],
            "gross_all_ci95": e45.group_bootstrap(groups, d["_wrong"], full["_wrong"])["ci95"],
            "gross_answer": d["gross_answer"] - full["gross_answer"],
            "abstention": (1 - d["answer"]) - (1 - full["answer"]),
            "false_answer": d["false_answer"] - full["false_answer"],
            "warning_auc": float(np.nanmean([full["warning_auc"][n] - d["warning_auc"][n] for n in targets])),
            "consensus_auc": full["consensus_auc"] - d["consensus_auc"],
        }
    pairs = {}
    for a, b in combinations(pool, 2):
        d = R[f"-{a}-{b}"]
        drop = d["gross_all"] - full["gross_all"]
        pairs[f"{a}+{b}"] = {"gross_all": drop, "interaction": drop - value[a]["gross_all"] - value[b]["gross_all"],
                             "answer": d["answer"], "false_answer": d["false_answer"],
                             "warning_auc_mean": d["warning_auc_mean"]}
    best_pair = min(pairs, key=lambda k: pairs[k]["gross_all"])
    for v in R.values():
        v.pop("_wrong")
    return {"n": len(rows), "n_null": len(nulls), "sets": R, "value": value, "pairs": pairs,
            "cheapest_pair_to_drop": best_pair, "overlap": overlap(W, pool)}


def relate(s):
    """Spearman of each lens's gross_all value against its overlap / accuracy, over (lens, dataset)."""
    pts = [(d["value"][L]["gross_all"], d["overlap"][L]) for m in s["media"].values() for d in m.values()
           for L in d["value"]]
    v = np.array([p[0] for p in pts])
    out = {"n_points": len(pts)}
    for key in ("coverage", "mean_phi", "max_phi", "gross"):
        x = np.array([p[1][key] for p in pts])
        rho, pval = spearmanr(x, v)
        out[key] = {"rho": float(rho), "p": float(pval)}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def main():
    s = {"pools": POOLS, "calibration": {"crepe": CREPE_CALIBRATION, "cnn_aug": CNN_AUG_CALIBRATION},
         "media": {"audio": {}, "image": {}}}
    for (medium, name), (rows, nulls) in datasets().items():
        t = time.time()
        s["media"][medium][LABELS[name]] = analyze(medium, rows, nulls)
        print(f"{medium} / {LABELS[name]}: {len(rows)} rows, {len(nulls)} null, {time.time() - t:.0f} s")
    s["relation"] = relate(s)
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    for medium, ds in s["media"].items():
        for name, d in ds.items():
            f = d["sets"]["full"]
            print(f"\n=== {medium} / {name} (n = {d['n']}, null {d['n_null']})")
            print(f"  full: answer {f['answer']:.1%}, gross|ans {f['gross_answer']:.1%}, gross_all {f['gross_all']:.1%}, "
                  f"false {f['false_answer']:.1%}, warning AUC {f['warning_auc_mean']:.3f}, consensus AUC {f['consensus_auc']:.3f}")
            print(f"  {'drop':10s} {'Δgross_all':>10s} {'CI95':>17s} {'Δgross|ans':>10s} {'Δabstain':>9s} "
                  f"{'Δfalse':>7s} {'ΔwarnAUC':>9s} {'ΔconsAUC':>9s} | {'gross':>6s} {'mean φ':>6s} {'max φ':>6s} {'cover':>6s}")
            for L, v in d["value"].items():
                o = d["overlap"][L]
                ci = v["gross_all_ci95"]
                print(f"  {L:10s} {v['gross_all']:+10.1%} [{ci[0]:+6.1%}, {ci[1]:+6.1%}] {v['gross_answer']:+10.1%} "
                      f"{v['abstention']:+9.1%} {v['false_answer']:+7.1%} {v['warning_auc']:+9.3f} {v['consensus_auc']:+9.3f} | "
                      f"{o['gross']:6.1%} {o['mean_phi']:6.2f} {o['max_phi']:6.2f} {o['coverage']:6.1%}")
            print("  pairs (Δgross_all, interaction): " + ", ".join(
                f"{k} {v['gross_all']:+.1%} ({v['interaction']:+.1%})" for k, v in d["pairs"].items()))
            print(f"  cheapest pair to drop: {d['cheapest_pair_to_drop']}")
    print("\nSpearman of Δgross_all (value) against, over (lens, dataset):")
    for k, v in s["relation"].items():
        if k != "n_points":
            print(f"  {k:9s} rho {v['rho']:+.2f}  p {v['p']:.3f}")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(m, n, d) for m, ds in s["media"].items() for n, d in ds.items()]
    fig, axes = plt.subplots(2, len(items), figsize=(3.6 * len(items), 7.4), squeeze=False)
    for j, (m, name, d) in enumerate(items):
        ax = axes[0, j]
        lenses = list(d["value"])
        val = [d["value"][L]["gross_all"] for L in lenses]
        ci = np.array([d["value"][L]["gross_all_ci95"] for L in lenses])
        y = np.arange(len(lenses))
        cols = ["#4a7fd4" if L in ("crepe", "cnn_aug") else "#777777" for L in lenses]
        ax.barh(y, val, color=cols, xerr=[np.array(val) - ci[:, 0], ci[:, 1] - np.array(val)], ecolor="k", capsize=2)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y, lenses, fontsize=8)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{m} / {name}", fontsize=10)
        ax.set_xlabel("gross error added by dropping it", fontsize=8)
        ax = axes[1, j]
        cov = [d["overlap"][L]["coverage"] for L in lenses]
        ax.scatter(cov, val, c=cols)
        for L, x, v in zip(lenses, cov, val):
            ax.annotate(L, (x, v), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("coverage (right where < 2 others are)", fontsize=8)
        if j == 0:
            ax.set_ylabel("value (Δ gross_all)", fontsize=9)
    rel = s["relation"]
    fig.suptitle(f"Exp 055: what each lens is worth (blue = learned).  Spearman of value vs coverage "
                 f"{rel['coverage']['rho']:+.2f}, vs mean φ {rel['mean_phi']['rho']:+.2f}, "
                 f"vs own gross error {rel['gross']['rho']:+.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
