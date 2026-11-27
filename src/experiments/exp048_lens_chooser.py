"""
Experiment 048: a learned lens chooser (issue #22)
==================================================
On vocadito (Exp 045) the consensus is grossly wrong on 5.6% of frames, while
the best lens for each frame, picked with hindsight, is wrong on 0.4%. Exp 047
found why fusing the trusted lenses cannot close that gap: when the right lens
is the only one that is right, the confidence threshold usually marks it blind,
so it never takes part. A chooser that looks at *every* lens and its
disagreement with the others does not have that limit.

The chooser
  For each lens, a logistic model predicts P(this lens is grossly wrong in this
  frame) from inspectable features. In each frame it answers with the lens
  that has the lowest predicted P(wrong). Lenses with no estimate are skipped.
  ``chooser_*_abstain`` also abstains when even the best lens has P(wrong) > 0.5.
  ``chooser_unpitched`` is also trained on the unpitched frames of its training
  sets, labelled "every lens wrong". The weights are reported per lens, so the
  choice stays explainable.

Is there a pitch at all?
  The chooser always answers. Unpitched frames (breaths, consonants) are scored
  separately as false answers, and a separate frame-level voicing gate in front
  of the chooser is swept over thresholds (``voicing_gate`` in the summary).

Feature sets (per lens, per frame)
  disagreement  Exp 042's six: own divergence, overall disagreement, own
                confidence, mean confidence of the others, distance to the
                closest other lens, octave relation with any other lens
  + state       the framework's 2-D state (z1, z2, squares, product) and the
                lens's assumption score

Training / test (frames from one recording are never split across them)
  transfer      train on synthetic audio (Exp 042) + MDB-stem-synth, test on
                vocadito: the held-out test of issue #22's Done-when. Also the
                mirror (train synthetic + vocadito, test MDB-stem-synth) and
                synthetic only -> each real dataset.
  within        5-fold cross-validation by recording on each real dataset.

References (gross error = > 50 cents, abstention counted as wrong)
  consensus     the old rule (Exp 045 numbers; fuse every trusted lens)
  q2            the current default, quorum of 2 (Exp 046)
  routed        the framework's assumption-score routing (Exp 045)
  best lens     the single lens with the lowest error on the test set (hindsight)
  hindsight     per frame, the lens closest to the truth
  gap closed    (consensus - chooser) / (consensus - hindsight)

Rows are the caches of Exp 045 and 046 (data/processed); run those first.

Run:  python src/experiments/exp048_lens_chooser.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp042_disagreement_prediction as e42  # noqa: E402
import exp045_real_data as e45  # noqa: E402
from core import ConfidenceThresholdDetector, Reading  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import OUTLIER_MARGIN, VOICING_THRESHOLD  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp048_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp048_lens_chooser.png")
REPS = e42.REPS
FOLDS = 5
FEATURE_NAMES = {
    "disagreement": ["divergence", "overall", "own_conf", "others_conf", "closest_dist", "octave_rel"],
    "state": ["divergence", "overall", "own_conf", "others_conf", "closest_dist", "octave_rel",
              "z1", "z2", "z1^2", "z2^2", "z1*z2", "assumption"],
}


def load(name):
    path = os.path.join(CACHE, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Features, labels, chooser
# ──────────────────────────────────────────────────────────────────────────────

# chooser name -> (feature set, also train on unpitched frames labelled "every lens wrong")
CHOOSERS = {"disagreement": ("disagreement", False), "state": ("state", False),
            "unpitched": ("disagreement", True)}


def wrong(row, rep):
    """Grossly wrong; on an unpitched frame (truth 0) any answer is wrong."""
    if row["truth"] <= 0:
        return True
    return e42.cents_error(row["readings"][rep][0], row["truth"]) > e42.GROSS_CENTS


def feats(row, rep, kind):
    x = e42.features(row, rep)
    if kind == "state":
        x = x + e42.state_features(row) + [row["assumptions"][rep]]
    return x


def features_only(rows, kind):
    return {r: np.array([feats(row, r, kind) for row in rows], float) for r in REPS}


def matrices(rows, kind):
    X = features_only(rows, kind)
    Y = {r: np.array([wrong(row, r) for row in rows], float) for r in REPS}
    return X, Y


def fit_chooser(rows, kind):
    X, Y = matrices(rows, kind)
    return {r: e42.fit_logistic(X[r], Y[r]) for r in REPS}


def p_wrong(models, rows, kind):
    """Frames × lenses matrix of predicted P(wrong); lenses with no estimate get 1."""
    X = features_only(rows, kind)
    P = np.column_stack([models[r](X[r]) for r in REPS])
    has = np.array([[row["readings"][r][0] > 0 for r in REPS] for row in rows])
    return np.where(has, P, 1.0)


def choose(rows, P, abstain=False):
    """Index of the chosen lens per frame, or -1 to abstain."""
    pick = np.argmin(P, axis=1)
    if abstain:
        pick = np.where(P[np.arange(len(P)), pick] > 0.5, -1, pick)
    return pick


def picked_wrong(rows, pick):
    return np.array([True if k < 0 else wrong(row, REPS[k]) for row, k in zip(rows, pick)])


def weights_table(rows, kind):
    """Standardised logistic weights per lens (sign: + = more likely wrong)."""
    X, Y = matrices(rows, kind)
    out = {}
    for r in REPS:
        mu, sd = X[r].mean(0), X[r].std(0) + 1e-9
        Z = (X[r] - mu) / sd
        # refit in standardised space to read the weights (same model as fit_logistic)
        w = np.zeros(Z.shape[1] + 1)
        Zb = np.c_[np.ones(len(Z)), Z]
        for _ in range(800):
            p = 1.0 / (1.0 + np.exp(-Zb @ w))
            w -= 0.5 * (Zb.T @ (p - Y[r]) / len(Y[r]) + 1e-2 * np.r_[0.0, w[1:]])
        out[r] = {n: round(float(v), 3) for n, v in zip(["bias", *FEATURE_NAMES[kind]], w)}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# References
# ──────────────────────────────────────────────────────────────────────────────

def comparator(k):
    return PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=k,
                           detector=ConfidenceThresholdDetector(VOICING_THRESHOLD))


def consensus_wrong(rows, k):
    comp = comparator(k)
    out = []
    for row in rows:
        rd = {n: Reading(n, f0 if f0 > 0 else None, c) for n, (f0, c) in row["readings"].items()}
        c = comp.compare(rd).consensus
        out.append(c is None or e42.cents_error(c, row["truth"]) > e42.GROSS_CENTS)
    return np.array(out)


def references(rows):
    lens = {r: np.array([wrong(row, r) for row in rows]) for r in REPS}
    best = min(lens, key=lambda r: lens[r].mean())
    routed = np.array([wrong(row, max(REPS, key=lambda k: row["assumptions"][k])) for row in rows])
    conf = np.array([wrong(row, max(REPS, key=lambda k: row["readings"][k][1])) for row in rows])
    return {
        "consensus": consensus_wrong(rows, None),
        "q2": consensus_wrong(rows, 2),
        "routed": routed,
        "most_confident": conf,
        "best_lens": lens[best],
        "hindsight": np.all(np.array(list(lens.values())), axis=0),
    }, best


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def false_answers(nulls, P_by_kind):
    """Share of unpitched frames given an answer (truth has no pitch)."""
    out = {"consensus": 0.0, "q2": 0.0}
    for k, comp in (("consensus", comparator(None)), ("q2", comparator(2))):
        out[k] = float(np.mean([
            comp.compare({n: Reading(n, f0 if f0 > 0 else None, c)
                          for n, (f0, c) in row["readings"].items()}).consensus is not None for row in nulls]))
    for kind, P in P_by_kind.items():
        has = (P < 1.0).any(axis=1)
        out[f"chooser_{kind}"] = float(has.mean())
        out[f"chooser_{kind}_abstain"] = float(np.mean(choose(nulls, P, abstain=True) >= 0))
    return out


def evaluate(test, P_by_kind, nulls=None, P_null=None):
    refs, best = references(test)
    groups = np.array([str(r["group"]) for r in test])
    res = {"n": len(test), "best_lens_name": best}
    bad = dict(refs)
    for kind, P in P_by_kind.items():
        bad[f"chooser_{kind}"] = picked_wrong(test, choose(test, P))
        pa = choose(test, P, abstain=True)
        bad[f"chooser_{kind}_abstain"] = picked_wrong(test, pa)
        res[f"chooser_{kind}_abstain_answer_rate"] = float(np.mean(pa >= 0))
        ans = pa >= 0
        res[f"chooser_{kind}_abstain_gross_when_answering"] = float(bad[f"chooser_{kind}_abstain"][ans].mean())
        res[f"pick_share_{kind}"] = {r: float(np.mean(choose(test, P) == i)) for i, r in enumerate(REPS)}
    res["gross_all"] = {m: float(v.mean()) for m, v in bad.items()}
    cons, hind = bad["consensus"].mean(), bad["hindsight"].mean()
    res["gap_closed"] = {m: float((cons - v.mean()) / (cons - hind)) for m, v in bad.items()
                         if m.startswith("chooser")}
    for kind in P_by_kind:
        m = f"chooser_{kind}"
        res[f"{m}_minus_consensus"] = e45.group_bootstrap(groups, bad[m], bad["consensus"])
        res[f"{m}_minus_best_lens"] = e45.group_bootstrap(groups, bad[m], bad["best_lens"])
        res[f"{m}_minus_most_confident"] = e45.group_bootstrap(groups, bad[m], bad["most_confident"])
    if nulls is not None:
        res["false_answer_unpitched"] = false_answers(nulls, P_null)
    fam = np.array([str(r["family"]) for r in test])
    res["by_family"] = {f: {m: float(v[fam == f].mean()) for m, v in bad.items()} for f in dict.fromkeys(fam)}
    return res


def fit_named(name, train, train_nulls):
    kind, with_nulls = CHOOSERS[name]
    return fit_chooser(train + (train_nulls if with_nulls else []), kind)


def transfer(train, train_nulls, test, nulls):
    models = {c: fit_named(c, train, train_nulls) for c in CHOOSERS}
    P = {c: p_wrong(models[c], test, CHOOSERS[c][0]) for c in CHOOSERS}
    P_null = {c: p_wrong(models[c], nulls, CHOOSERS[c][0]) for c in CHOOSERS}
    return evaluate(test, P, nulls, P_null)


def within(rows, nulls, seed=0):
    """Folds split by recording; a recording's unpitched frames go with its voiced ones."""
    groups = np.array([str(r["group"]) for r in rows])
    ngroups = np.array([str(r["group"]) for r in nulls])
    uniq = np.unique(np.r_[groups, ngroups])
    np.random.default_rng(seed).shuffle(uniq)
    P = {c: np.zeros((len(rows), len(REPS))) for c in CHOOSERS}
    P_null = {c: np.zeros((len(nulls), len(REPS))) for c in CHOOSERS}
    for fold in np.array_split(uniq, FOLDS):
        test, ntest = np.isin(groups, fold), np.isin(ngroups, fold)
        train = [r for r, t in zip(rows, test) if not t]
        train_nulls = [r for r, t in zip(nulls, ntest) if not t]
        for c, (kind, _) in CHOOSERS.items():
            m = fit_named(c, train, train_nulls)
            P[c][test] = p_wrong(m, [r for r, t in zip(rows, test) if t], kind)
            if ntest.any():
                P_null[c][ntest] = p_wrong(m, [r for r, t in zip(nulls, ntest) if t], kind)
    return evaluate(rows, P, nulls, P_null)


def voicing_features(row):
    """Frame-level: is there a pitch at all? (max / mean confidence, disagreement, state)."""
    c = [row["readings"][k][1] for k in REPS]
    return [max(c), float(np.mean(c)), row["overall"], min(p[2] for p in row["pairs"]), row["z1"], row["z2"]]


def gate_curve(train, train_nulls, test, nulls, thresholds=(0.3, 0.5, 0.7, 0.9)):
    """A separate logistic voicing gate in front of the disagreement chooser.

    Unpitched training frames are repeated to balance the classes. The chooser
    answers only when P(unpitched) < threshold; abstentions on voiced frames
    count as wrong.
    """
    X = np.array([voicing_features(r) for r in train] + [voicing_features(r) for r in train_nulls]
                 * max(1, len(train) // len(train_nulls)))
    y = np.r_[np.zeros(len(train)), np.ones(len(X) - len(train))]
    gate = e42.fit_logistic(X, y)
    pv = gate(np.array([voicing_features(r) for r in test]))
    pn = gate(np.array([voicing_features(r) for r in nulls]))
    w = picked_wrong(test, choose(test, p_wrong(fit_chooser(train, "disagreement"), test, "disagreement")))
    return {f"{t:g}": {"voiced_answered": float(np.mean(pv < t)), "gross_all": float(np.mean(w | (pv >= t))),
                       "false_answer_unpitched": float(np.mean(pn < t))} for t in thresholds}


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    syn = load("exp046_e42_rows")
    voc = load("exp045_audio_vocadito")
    mdb = load("exp045_audio_mdb_stem_synth")
    if syn is None or voc is None or mdb is None:
        sys.exit("Run Exp 045 and Exp 046 first (they cache the rows in data/processed).")
    (voc, voc_null), (mdb, mdb_null) = voc, mdb

    s = {"features": FEATURE_NAMES, "choosers": {c: list(v) for c, v in CHOOSERS.items()},
         "transfer": {}, "within": {}}
    syn_null = load("exp046_e42_unpitched")
    plans = {
        "synthetic + MDB-stem-synth -> vocadito": (syn + mdb, syn_null + mdb_null, voc, voc_null),
        "synthetic + vocadito -> MDB-stem-synth": (syn + voc, syn_null + voc_null, mdb, mdb_null),
        "synthetic -> vocadito": (syn, syn_null, voc, voc_null),
        "synthetic -> MDB-stem-synth": (syn, syn_null, mdb, mdb_null),
    }
    for name, (train, train_nulls, test, nulls) in plans.items():
        print(f"transfer: {name}")
        s["transfer"][name] = transfer(train, train_nulls, test, nulls)
        s["transfer"][name]["voicing_gate"] = gate_curve(train, train_nulls, test, nulls)
    for name, rows, nulls in (("vocadito", voc, voc_null), ("MDB-stem-synth", mdb, mdb_null)):
        print(f"within: {name}")
        s["within"][name] = within(rows, nulls)
    s["weights_disagreement_trained_on_synthetic_plus_mdb"] = weights_table(syn + mdb, "disagreement")

    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2)
    report(s)
    plot(s)
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


SHOW = ("consensus", "q2", "routed", "most_confident", "chooser_disagreement", "chooser_state",
        "chooser_unpitched", "chooser_unpitched_abstain", "best_lens", "hindsight")


def report(s):
    for mode in ("transfer", "within"):
        for name, d in s[mode].items():
            print(f"\n[{mode}] {name}  (n={d['n']}, best lens {d['best_lens_name']})")
            for m in SHOW:
                gc = d["gap_closed"].get(m)
                extra = f"   gap closed {gc:+.0%}" if gc is not None else ""
                print(f"  {m:30s} {d['gross_all'][m]:6.1%}{extra}")
            for k in CHOOSERS:
                print(f"  chooser_{k}_abstain answers {d[f'chooser_{k}_abstain_answer_rate']:.1%}, gross when "
                      f"answering {d[f'chooser_{k}_abstain_gross_when_answering']:.1%}; picks "
                      + ", ".join(f"{r} {v:.0%}" for r, v in d[f"pick_share_{k}"].items()))
                for ref in ("consensus", "best_lens", "most_confident"):
                    v = d[f"chooser_{k}_minus_{ref}"]
                    print(f"  chooser_{k} - {ref:14s} {v['diff']:+.1%}  CI95 [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
            print("  false answers on unpitched frames:",
                  {m: f"{v:.0%}" for m, v in d["false_answer_unpitched"].items()})
            for t, g in d.get("voicing_gate", {}).items():
                print(f"  voicing gate < {t}: voiced answered {g['voiced_answered']:.1%}, gross_all "
                      f"{g['gross_all']:.1%}, false answers {g['false_answer_unpitched']:.1%}")
    print("\nweights (standardised; + = more likely wrong):")
    for r, w in s["weights_disagreement_trained_on_synthetic_plus_mdb"].items():
        print(f"  {r:8s}", w)


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("transfer", n) for n in s["transfer"]] + [("within", n) for n in s["within"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    colors = ["#9aa0a6", "#4a7fd4", "#c3a6e0", "#b0b0b0", "#d4744a", "#f0b27a", "#8e5bb5", "#b99ad6",
              "#5aa36b", "#cfcfcf"]
    for ax, (mode, name) in zip(axes.flat, panels):
        d = s[mode][name]
        vals = [d["gross_all"][m] for m in SHOW]
        ax.bar(range(len(SHOW)), vals, color=colors)
        ax.set_xticks(range(len(SHOW)), [m.replace("chooser_", "ch_") for m in SHOW], rotation=45,
                      ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{name} ({mode})", fontsize=9)
    fig.suptitle("Exp 048: a learned lens chooser vs. consensus and best lens "
                 "(gross error > 50 cents, abstention = wrong)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
