"""
Experiment 049: when to abstain (issue #23)
===========================================
The quorum of Exp 046 (``min_agreeing=2``) cut false answers on unpitched /
unoriented input to 0-7%, but it answers only 66-75% of noisy pitched frames
and 67% of BSDS500 patches. Issue #23 asks for a rule that keeps quorum 2's
false-answer rate while answering more.

Every rule below starts from the same per-frame object: the largest group of
trusted readings that agree pairwise (the core's ``quorum_group``), its size,
and its fused estimate.

Rules (each with a knob that is swept)
  q1, q2, q3       quorum of 1, 2, 3 (Exp 046 reference points; q2 is the default)
  legacy           fuse every trusted reading (pre-quorum rule)
  singleton tau    q2, but one lens may answer alone if its confidence >= tau
  adaptive c       noise-adaptive quorum: q1 when the frame looks clean (mean
                   confidence of all lenses >= c), q2 otherwise
  tolerance m      q2 where "agree" is widened to m x the medium's tolerance
                   (50 cents / 10 degrees); noise spreads the lenses a little
  trust theta      q2 where a lens is trusted from confidence theta instead of 0.5
                   (Exp 047/048: the right lens is often just below the gate)
  learned t        a logistic gate on six frame features (group size, group max /
                   mean confidence, all-lens mean / max confidence, overall
                   disagreement) predicts whether the q1 answer is right; answer
                   when P(right) >= t. Unpitched / unoriented frames are labelled
                   wrong and repeated to balance the classes. 5-fold
                   cross-validation split by recording / image.

Metrics, per dataset (gross error = >50 cents for audio, >10 degrees for images)
  answer        fraction of pitched / oriented frames with an answer
  false_answer  fraction of unpitched / unoriented frames with an answer
  gross_answer  gross error among the answers
  gross_all     gross error counting an abstention as wrong

Done-when test of issue #23: a setting *beats q2* on a dataset when its
false_answer <= q2's, its answer rate is higher, and its gross_all is lower.

Rows are the caches of Exp 045 and 046 (data/processed); run those first.

Run:  python src/experiments/exp049_abstention.py
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
import exp044_image_disagreement as e44  # noqa: E402
import exp045_real_data as e45  # noqa: E402
from core import ConfidenceThresholdDetector, Reading  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import OCTAVE_TOLERANCE_CENTS, OUTLIER_MARGIN, VOICING_THRESHOLD  # noqa: E402
from image_medium.medium import AGREE_DEG, VOTING_THRESHOLD, OrientationComparator  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp049_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp049_abstention.png")
FOLDS = 5
TOLERANCES = (1.0, 1.5, 2.0, 3.0)
TRUSTS = (0.5, 0.3, 0.1)
TAUS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
ADAPTIVE = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LEARNED = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load(name):
    path = os.path.join(CACHE, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Media
# ──────────────────────────────────────────────────────────────────────────────

class TolerantPitch(PitchComparator):
    def __init__(self, tol_cents, **kw):
        super().__init__(**kw)
        self.tol = tol_cents

    def agrees(self, pair):
        return pair.details.get("cents", np.inf) < self.tol


class TolerantOrientation(OrientationComparator):
    def __init__(self, tol_deg, **kw):
        super().__init__(**kw)
        self.tol = tol_deg

    def agrees(self, pair):
        return pair.details.get("degrees", np.inf) < self.tol


class Audio:
    name = "audio"

    @staticmethod
    def readings(row):
        return {n: Reading(n, f0 if f0 > 0 else None, c) for n, (f0, c) in row["readings"].items()}

    @staticmethod
    def wrong(est, row):
        return est is None or e42.cents_error(est, row["truth"]) > e42.GROSS_CENTS

    @staticmethod
    def comparator(k=1, m=1.0, trust=VOICING_THRESHOLD):
        return TolerantPitch(m * OCTAVE_TOLERANCE_CENTS, outlier_margin=OUTLIER_MARGIN, min_agreeing=k,
                             detector=ConfidenceThresholdDetector(trust))


class Image:
    name = "image"

    @staticmethod
    def readings(row):
        return {n: Reading(n, e, c) for n, (e, c) in row["readings"].items()}

    @staticmethod
    def wrong(est, row):
        return est is None or e44.err_deg(est, row["truth"]) > e44.GROSS_DEG

    @staticmethod
    def comparator(k=1, m=1.0, trust=VOTING_THRESHOLD):
        return TolerantOrientation(m * AGREE_DEG, min_agreeing=k, detector=ConfidenceThresholdDetector(trust))


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame facts, computed once per comparator setting
# ──────────────────────────────────────────────────────────────────────────────

def facts(M, rows, m=1.0, trust=None):
    """For each row: size of the largest agreeing trusted group, its fused estimate
    (None if empty), group max / mean confidence, and frame-level features."""
    comp = M.comparator(1, m, trust) if trust is not None else M.comparator(1, m)
    size, est, gmax, gmean, allmean, allmax, overall, ntrust = ([] for _ in range(8))
    for r in rows:
        rd = M.readings(r)
        rep = comp.compare(rd)
        g = rep.consensus_from
        conf = [rd[n].confidence for n in rd]
        size.append(len(g))
        est.append(rep.consensus)
        gmax.append(max((rd[n].confidence for n in g), default=0.0))
        gmean.append(float(np.mean([rd[n].confidence for n in g])) if g else 0.0)
        allmean.append(float(np.mean(conf)))
        allmax.append(max(conf))
        overall.append(rep.overall)
        ntrust.append(sum(not a.blind for a in rep.assessments.values()))
    return {"size": np.array(size), "est": est, "gmax": np.array(gmax), "gmean": np.array(gmean),
            "allmean": np.array(allmean), "allmax": np.array(allmax), "overall": np.array(overall),
            "ntrust": np.array(ntrust)}


def legacy_estimates(M, rows):
    comp = M.comparator(None)
    return [comp.compare(M.readings(r)).consensus for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Rules → answer masks (the estimate when answering is always the group's fused one)
# ──────────────────────────────────────────────────────────────────────────────

def gate_features(f):
    return np.column_stack([f["size"], f["gmax"], f["gmean"], f["allmean"], f["allmax"], f["overall"]])


def learned_gate(M, rows, nulls, f, fn, seed=0):
    """Out-of-fold P(the q1 answer is right) for pitched rows and null rows."""
    y = np.array([0.0 if M.wrong(e, r) else 1.0 for e, r in zip(f["est"], rows)])
    X = np.r_[gate_features(f), gate_features(fn)]
    Y = np.r_[y, np.zeros(len(nulls))]
    is_null = np.r_[np.zeros(len(rows), bool), np.ones(len(nulls), bool)]
    groups = np.array([str(r.get("group", "")) for r in rows]
                      + [str(r["group"]) if "group" in r else f"null{i}" for i, r in enumerate(nulls)])
    # Folds split by group. Null frames are rare (0.5-10% of a dataset), so in each
    # training fold they are repeated until they weigh as much as the pitched frames.
    uniq = np.unique(groups)
    np.random.default_rng(seed).shuffle(uniq)
    p = np.zeros(len(Y))
    for fold in np.array_split(uniq, FOLDS):
        test = np.isin(groups, fold)
        tr, trn = np.flatnonzero(~test & ~is_null), np.flatnonzero(~test & is_null)
        idx = np.r_[tr, np.tile(trn, max(1, len(tr) // max(1, len(trn))))]
        p[test] = e42.fit_logistic(X[idx], Y[idx])(X[test])
    return p[:len(rows)], p[len(rows):]


def metrics(M, rows, est, answer, null_answer):
    answer = np.asarray(answer, bool)
    wrong = np.array([(not a) or M.wrong(e, r) for e, a, r in zip(est, answer, rows)])
    noisy = np.array([r["family"] == "noise" for r in rows])
    return {"answer": float(answer.mean()), "false_answer": float(np.mean(null_answer)),
            "answer_noise": float(answer[noisy].mean()) if noisy.any() else float("nan"),
            "gross_answer": float(wrong[answer].mean()) if answer.any() else float("nan"),
            "gross_all": float(wrong.mean()), "_wrong": wrong}


def evaluate(M, rows, nulls):
    out = {"n": len(rows), "n_null": len(nulls), "settings": {}}
    S = out["settings"]
    f, fn = facts(M, rows), facts(M, nulls)
    has, hasn = np.array([e is not None for e in f["est"]]), np.array([e is not None for e in fn["est"]])

    leg = legacy_estimates(M, rows)
    legn = legacy_estimates(M, nulls)
    S["legacy"] = metrics(M, rows, leg, [e is not None for e in leg], [e is not None for e in legn])
    for k in (1, 2, 3):
        S[f"q{k}"] = metrics(M, rows, f["est"], f["size"] >= k, fn["size"] >= k)
    for tau in TAUS:
        a = (f["size"] >= 2) | ((f["size"] == 1) & (f["gmax"] >= tau))
        an = (fn["size"] >= 2) | ((fn["size"] == 1) & (fn["gmax"] >= tau))
        S[f"singleton {tau:g}"] = metrics(M, rows, f["est"], a, an)
    for c in ADAPTIVE:
        a = f["size"] >= np.where(f["allmean"] >= c, 1, 2)
        an = fn["size"] >= np.where(fn["allmean"] >= c, 1, 2)
        S[f"adaptive {c:g}"] = metrics(M, rows, f["est"], a & has, an & hasn)
    for m in TOLERANCES[1:]:
        g, gn = facts(M, rows, m=m), facts(M, nulls, m=m)
        S[f"tolerance {m:g}x"] = metrics(M, rows, g["est"], g["size"] >= 2, gn["size"] >= 2)
    for th in TRUSTS[1:]:
        g, gn = facts(M, rows, trust=th), facts(M, nulls, trust=th)
        S[f"trust {th:g}"] = metrics(M, rows, g["est"], g["size"] >= 2, gn["size"] >= 2)
    p, pn = learned_gate(M, rows, nulls, f, fn)
    for t in LEARNED:
        S[f"learned {t:g}"] = metrics(M, rows, f["est"], has & (p >= t), hasn & (pn >= t))

    q2 = S["q2"]
    beats = {k: v for k, v in S.items()
             if v["false_answer"] <= q2["false_answer"] + 1e-12 and v["answer"] > q2["answer"] + 1e-12
             and v["gross_all"] < q2["gross_all"] - 1e-12}
    out["beats_q2"] = sorted(beats, key=lambda k: S[k]["gross_all"])
    out["best_vs_q2"] = out["beats_q2"][0] if beats else None
    # Group-bootstrap CIs (whole recordings / images resampled) for the gross_all
    # difference against q2, for the default learned threshold and the best setting.
    groups = np.array([str(r["group"]) for r in rows])
    out["vs_q2_gross_all"] = {k: e45.group_bootstrap(groups, S[k]["_wrong"], S["q2"]["_wrong"])
                              for k in dict.fromkeys(["learned 0.5", *out["beats_q2"][:1]])}
    for v in S.values():
        v.pop("_wrong")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    syn_a, syn_an = load("exp046_e42_rows"), load("exp046_e42_unpitched")
    syn_i, syn_in = load("exp046_e44_rows"), load("exp046_e44_unoriented")
    if syn_a is None or syn_i is None:
        sys.exit("Run Exp 046 first (it caches the synthetic rows).")
    datasets = {"synthetic audio": (Audio, syn_a, syn_an),
                "synthetic images": (Image, [r for r in syn_i if not r["empty"]], [r for r in syn_in if not r["empty"]])}
    for name, label in (("audio_vocadito", "vocadito"), ("audio_mdb_stem_synth", "MDB-stem-synth")):
        d = load(f"exp045_{name}")
        if d is not None:
            datasets[label] = (Audio, d[0], d[1])
    d = load("exp045_image_rows")
    if d is not None:
        datasets["BSDS500"] = (Image, [r for r in d if not r["empty"]], [r for r in syn_in if not r["empty"]])

    s = {"knobs": {"tolerance": TOLERANCES, "trust": TRUSTS, "singleton": TAUS, "adaptive": ADAPTIVE,
                   "learned": LEARNED}, "datasets": {}}
    for label, (M, rows, nulls) in datasets.items():
        print(f"{label}: {len(rows)} rows, {len(nulls)} null frames")
        s["datasets"][label] = evaluate(M, rows, nulls)

    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2)
    report(s)
    plot(s)
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


def report(s):
    for ds, d in s["datasets"].items():
        print(f"\n{ds}: n={d['n']}, null frames={d['n_null']}")
        print(f"  {'setting':16s} {'answer':>7s} {'noisy':>7s} {'false':>7s} {'gross|ans':>9s} {'gross_all':>9s}")
        for k, v in d["settings"].items():
            mark = " *" if k in d["beats_q2"] else ""
            print(f"  {k:16s} {v['answer']:7.1%} {v['answer_noise']:7.1%} {v['false_answer']:7.1%} "
                  f"{v['gross_answer']:9.1%} {v['gross_all']:9.1%}{mark}")
        print(f"  beats q2 (* above): {d['beats_q2'] or 'none'}")
        for k, v in d["vs_q2_gross_all"].items():
            print(f"  {k} - q2, gross_all: {v['diff']:+.1%}  CI95 [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")


FAMILIES = {"singleton": "#d4744a", "adaptive": "#8e5bb5", "tolerance": "#5aa36b", "trust": "#c9a227",
            "learned": "#4a7fd4"}


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(s["datasets"])
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.5), squeeze=False)
    for j, ds in enumerate(names):
        S = s["datasets"][ds]["settings"]
        for row, (ykey, ylabel) in enumerate((("false_answer", "false answers (null input)"),
                                              ("gross_all", "gross error, abstain = wrong"))):
            ax = axes[row, j]
            for fam, col in FAMILIES.items():
                pts = [(v["answer"], v[ykey]) for k, v in S.items() if k.startswith(fam)]
                if fam == "tolerance":
                    pts = [(S["q2"]["answer"], S["q2"][ykey])] + pts
                if fam == "trust":
                    pts = [(S["q2"]["answer"], S["q2"][ykey])] + pts
                if pts:
                    x, y = zip(*sorted(pts))
                    ax.plot(x, y, "o-", color=col, ms=3, lw=1, label=fam)
            for k, mk in (("q1", "^"), ("q2", "s"), ("q3", "v"), ("legacy", "x")):
                ax.plot(S[k]["answer"], S[k][ykey], mk, color="k", ms=7, mfc="none" if k != "q2" else "k",
                        label=k if j == 0 and row == 0 else None)
            ax.axhline(S["q2"][ykey], color="k", lw=0.5, ls=":")
            ax.axvline(S["q2"]["answer"], color="k", lw=0.5, ls=":")
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_xlabel("answer rate (pitched / oriented)", fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(ds, fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Exp 049: abstention rules vs. quorum 2 (black square). "
                 "Beating q2 = right of it and not above it in both rows")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
