"""
Experiment 047: fusion that knows which lenses fail together (issue #21)
========================================================================
Disagreement reliably warns that a lens is blind (Exp 042-045), but turning
that into a better answer fails on images: the lenses fail in families, and a
family that is wrong together outvotes the lens that is right (Exp 044: the
consensus has 9.8% gross errors, the best single lens, Radon, 4.8%).

Issue #21 adds ``core.FamilyVotes``: learned from labelled readings, it gives
each lens a reliability (P(right | confidence)) and each pair a failure
correlation (phi coefficient of their wrong/right indicators). The comparator
then picks the maximal agreeing group with the highest

    score(G) = sum_i p_i / (1 + sum_{j in G, j != i} R_ij)

so lenses that always fail together count as one vote. The quorum of Exp 046
still applies (at least ``min_agreeing`` members).

Methods compared (gross error = >50 cents for audio, >10 degrees for images)
  legacy        fuse every trusted reading (pre-quorum rule)
  q2            Exp 046: the largest agreeing group of >= 2
  rel_q2        reliability vote with every correlation set to 0 (ablation:
                is it the reliabilities or the families that help?)
  family_q2     reliability + failure correlation, >= 2 agreeing
  family_q1     the same, but a single lens may win
  best lens     the single lens with the lowest error on this dataset
                (chosen with hindsight, so an optimistic reference)
  hindsight     per signal, the lens closest to the truth (upper bound)

How FamilyVotes is trained
  within   5-fold cross-validation split by group (image / recording), so a
           model never sees the signal it is scored on.
  transfer fitted on the synthetic set of the same medium (Exp 042 / 044),
           scored on the real one (vocadito, MDB-stem-synth, BSDS500). This
           is the honest test: the families are learned where the truth is
           cheap and used where it is not.

Rows are the cached readings of Exp 042, 044, 045 and 046 (data/processed);
the lenses are not re-run. Run Exp 045 and 046 first.

Run:  python src/experiments/exp047_family_fusion.py
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
from core import AgreementTable, ConfidenceThresholdDetector, FamilyVotes, Reading  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import OUTLIER_MARGIN, VOICING_THRESHOLD  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD, OrientationComparator  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp047_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp047_family_fusion.png")
FOLDS = 5
METHODS = ("legacy", "q2", "rel_q2", "family_q2", "family_q1", "table_q1", "table_abstain")


def load(name):
    path = os.path.join(CACHE, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Medium adapters: readings, errors, comparators
# ──────────────────────────────────────────────────────────────────────────────

class Audio:
    lenses = e42.REPS

    @staticmethod
    def readings(row):
        return {n: Reading(n, f0 if f0 > 0 else None, c) for n, (f0, c) in row["readings"].items()}

    @staticmethod
    def wrong(est, row):
        return est is None or e42.cents_error(est, row["truth"]) > e42.GROSS_CENTS

    @staticmethod
    def comparator(k, family=None, min_score=None, threshold=VOICING_THRESHOLD):
        return PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=k, family=family,
                               min_score=min_score, detector=ConfidenceThresholdDetector(threshold))

    @staticmethod
    def rows(rows):
        return rows


class Image:
    lenses = e44.REPS

    @staticmethod
    def readings(row):
        return {n: Reading(n, e, c) for n, (e, c) in row["readings"].items()}

    @staticmethod
    def wrong(est, row):
        return est is None or e44.err_deg(est, row["truth"]) > e44.GROSS_DEG

    @staticmethod
    def comparator(k, family=None, min_score=None, threshold=VOTING_THRESHOLD):
        return OrientationComparator(min_agreeing=k, family=family, min_score=min_score,
                                     detector=ConfidenceThresholdDetector(threshold))

    @staticmethod
    def rows(rows):
        return [r for r in rows if not r["empty"]]


def lens_failures(M, rows):
    conf = [{n: float(r["readings"][n][1]) for n in M.lenses} for r in rows]
    fail = [{n: M.wrong(M.readings(r)[n].estimate, r) for n in M.lenses} for r in rows]
    return conf, fail


def fit(M, rows, independent=False):
    conf, fail = lens_failures(M, rows)
    return FamilyVotes.fit(conf, fail, independent=independent)


def groups_of(M, comp, row):
    """The maximal agreeing groups of trusted readings in one row, with the readings."""
    rd = M.readings(row)
    rep = comp.compare(rd)
    trusted = [n for n in rd if not rep.assessments[n].blind]
    return rd, comp.agreeing_groups(trusted, rep.pairs)


def fit_table(M, rows):
    """AgreementTable: how often each maximal agreeing group's fused answer was right."""
    comp = M.comparator(1)
    examples = []
    for r in rows:
        rd, groups = groups_of(M, comp, r)
        examples += [(g, not M.wrong(comp.fuse([rd[n] for n in g]), r)) for g in groups]
    return AgreementTable.fit(examples)


def fit_all(M, rows):
    """(FamilyVotes, FamilyVotes with correlations 0, AgreementTable) from labelled rows."""
    return fit(M, rows), fit(M, rows, independent=True), fit_table(M, rows)


def consensus(M, comp, row):
    return comp.compare(M.readings(row)).consensus


def comparators(M, fv_family, fv_rel, table):
    return {
        "legacy": M.comparator(None),
        "q2": M.comparator(2),
        "rel_q2": M.comparator(2, fv_rel),
        "family_q2": M.comparator(2, fv_family),
        "family_q1": M.comparator(1, fv_family),
        "table_q1": M.comparator(1, table),
        "table_abstain": M.comparator(1, table, min_score=0.5),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def estimates_within(M, rows, seed=0):
    """Out-of-fold estimates per method: FamilyVotes fitted on the other folds."""
    groups = np.array([str(r["group"]) for r in rows])
    uniq = np.unique(groups)
    np.random.default_rng(seed).shuffle(uniq)
    est = {m: [None] * len(rows) for m in METHODS}
    for fold in np.array_split(uniq, FOLDS):
        test = np.isin(groups, fold)
        train = [r for r, t in zip(rows, test) if not t]
        comps = comparators(M, *fit_all(M, train))
        for i in np.flatnonzero(test):
            for m, c in comps.items():
                est[m][i] = consensus(M, c, rows[i])
    return est


def estimates_with(M, rows, models):
    comps = comparators(M, *models)
    return {m: [consensus(M, c, r) for r in rows] for m, c in comps.items()}


def ceiling(M, rows, best):
    """How well could *any* choice among agreeing groups do?

    trusted_oracle      error if we always picked the right maximal agreeing group of
                        trusted readings: the ceiling of every method in this experiment
    all_trusted_oracle  the same with every reading trusted (detector threshold 0)
    gate_loss           share of signals where the best lens is right but no trusted
                        group is; gate_loss_best_lens_blind_share: how many of those had
                        the best lens marked blind by the confidence threshold
    """
    comp, comp0 = M.comparator(1), M.comparator(1, threshold=0.0)
    miss = miss0 = lost = lost_blind = 0
    for r in rows:
        rd, groups = groups_of(M, comp, r)
        ok = any(not M.wrong(comp.fuse([rd[n] for n in g]), r) for g in groups)
        _, groups0 = groups_of(M, comp0, r)
        miss0 += not any(not M.wrong(comp0.fuse([rd[n] for n in g]), r) for g in groups0)
        miss += not ok
        if not ok and not M.wrong(rd[best].estimate, r):
            lost += 1
            lost_blind += rd[best].confidence < comp.detector.threshold
    n = len(rows)
    return {"trusted_oracle": miss / n, "all_trusted_oracle": miss0 / n, "gate_loss": lost / n,
            "gate_loss_best_lens_blind_share": lost_blind / lost if lost else float("nan")}


def score(M, rows, est, nulls=None, models=None):
    groups = np.array([str(r["group"]) for r in rows])
    out = {"n": len(rows)}
    bad = {}
    for m, e in est.items():
        ans = np.array([x is not None for x in e])
        wrong = np.array([M.wrong(x, r) for x, r in zip(e, rows)])  # abstention counts as wrong
        bad[m] = wrong
        out[m] = {"gross_all": float(wrong.mean()), "answer": float(ans.mean()),
                  "gross_answer": float(wrong[ans].mean()) if ans.any() else float("nan")}
    lens_bad = {n: np.array([M.wrong(M.readings(r)[n].estimate, r) for r in rows]) for n in M.lenses}
    best = min(lens_bad, key=lambda n: lens_bad[n].mean())
    out["best_lens"] = {"name": best, "gross_all": float(lens_bad[best].mean())}
    hind = np.all(np.array(list(lens_bad.values())), axis=0)
    out["hindsight"] = {"gross_all": float(hind.mean())}
    out["family_q2_minus_q2"] = e45.group_bootstrap(groups, bad["family_q2"], bad["q2"])
    out["family_q2_minus_rel_q2"] = e45.group_bootstrap(groups, bad["family_q2"], bad["rel_q2"])
    out["family_q2_minus_best_lens"] = e45.group_bootstrap(groups, bad["family_q2"], lens_bad[best])
    out["family_q1_minus_best_lens"] = e45.group_bootstrap(groups, bad["family_q1"], lens_bad[best])
    out["table_q1_minus_legacy"] = e45.group_bootstrap(groups, bad["table_q1"], bad["legacy"])
    out["table_q1_minus_family_q1"] = e45.group_bootstrap(groups, bad["table_q1"], bad["family_q1"])
    out["table_q1_minus_best_lens"] = e45.group_bootstrap(groups, bad["table_q1"], lens_bad[best])
    out["ceiling"] = ceiling(M, rows, best)
    fam = np.array([str(r["family"]) for r in rows])
    out["by_family"] = {f: {m: float(bad[m][fam == f].mean()) for m in METHODS}
                        | {"best_lens": float(lens_bad[best][fam == f].mean())}
                        for f in dict.fromkeys(fam)}
    if nulls is not None and models is not None:
        comps = comparators(M, *models)
        out["false_answer"] = {m: float(np.mean([consensus(M, c, r) is not None for r in nulls]))
                               for m, c in comps.items()}
    return out


def describe(fv, table):
    """The learned weights, for the summary (inspectable, VISION non-goal)."""
    return {"reliability_by_confidence_bin": {n: [round(x, 3) for x in b] for n, b in fv.bins.items()},
            "failure_correlation": {"+".join(sorted(k)): round(v, 3) for k, v in fv.correlation.items()},
            "agreement_table": {"+".join(sorted(k)): {"p_right": round(table.score(k), 3), "seen": int(n)}
                                for k, (_, n) in sorted(table.table.items(), key=lambda kv: -kv[1][1])}}


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    s = {"methods": list(METHODS), "within": {}, "transfer": {}, "weights": {}}

    syn_audio = load("exp046_e42_rows")
    syn_unp = load("exp046_e42_unpitched")
    syn_img = load("exp046_e44_rows")
    syn_unor = load("exp046_e44_unoriented")
    if syn_audio is None or syn_img is None:
        sys.exit("Run Exp 046 first (it caches the synthetic rows).")
    real = {}
    for name, label in (("audio_vocadito", "vocadito"), ("audio_mdb_stem_synth", "MDB-stem-synth")):
        d = load(f"exp045_{name}")
        if d is not None:
            real[label] = (Audio, d[0], d[1])
    d = load("exp045_image_rows")
    if d is not None:
        real["BSDS500"] = (Image, Image.rows(d), syn_unor)

    datasets = {"synthetic audio": (Audio, syn_audio, syn_unp),
                "synthetic images": (Image, Image.rows(syn_img), syn_unor), **real}

    for label, (M, rows, nulls) in datasets.items():
        print(f"{label}: within-dataset cross-validation ({len(rows)} rows)")
        models = fit_all(M, rows)
        s["weights"][label] = describe(models[0], models[2])
        s["within"][label] = score(M, rows, estimates_within(M, rows), nulls, models)

    transfer_from = {Audio: ("synthetic audio", syn_audio), Image: ("synthetic images", Image.rows(syn_img))}
    for label, (M, rows, nulls) in real.items():
        src, train = transfer_from[M]
        print(f"{label}: transfer from {src}")
        models = fit_all(M, train)
        s["transfer"][label] = score(M, rows, estimates_with(M, rows, models), nulls, models)
        s["transfer"][label]["trained_on"] = src

    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=str)
    report(s)
    plot(s)
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


def report(s):
    for mode in ("within", "transfer"):
        for ds, d in s[mode].items():
            print(f"\n[{mode}] {ds}: n={d['n']}  best lens {d['best_lens']['name']} "
                  f"{d['best_lens']['gross_all']:.1%}  hindsight {d['hindsight']['gross_all']:.1%}")
            print(f"  {'method':10s} {'gross_all':>9s} {'answer':>7s} {'gross|ans':>9s} {'false_ans':>9s}")
            for m in METHODS:
                fa = d.get("false_answer", {}).get(m, float("nan"))
                print(f"  {m:10s} {d[m]['gross_all']:9.1%} {d[m]['answer']:7.1%} "
                      f"{d[m]['gross_answer']:9.1%} {fa:9.1%}")
            for k in ("family_q2_minus_q2", "family_q2_minus_rel_q2", "family_q2_minus_best_lens",
                      "family_q1_minus_best_lens", "table_q1_minus_legacy", "table_q1_minus_family_q1",
                      "table_q1_minus_best_lens"):
                v = d[k]
                print(f"  {k:28s} {v['diff']:+.1%}  CI95 [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
            c = d["ceiling"]
            print(f"  ceiling: trusted-group oracle {c['trusted_oracle']:.1%}, all-trusted oracle "
                  f"{c['all_trusted_oracle']:.1%}; gate loss {c['gate_loss']:.1%} "
                  f"({c['gate_loss_best_lens_blind_share']:.0%} with the best lens marked blind)")
    for ds, w in s["weights"].items():
        print(f"\nfailure correlation, {ds}:", w["failure_correlation"])


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("within", ds) for ds in s["within"]] + [("transfer", ds) for ds in s["transfer"]]
    cols = 4
    rows_n = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.6 * rows_n), squeeze=False)
    labels = [*METHODS, "best lens", "group oracle", "hindsight"]
    colors = ["#9aa0a6", "#4a7fd4", "#c3a6e0", "#d4744a", "#f0b27a", "#8e5bb5", "#b99ad6",
              "#5aa36b", "#a8d5b2", "#cfcfcf"]
    for ax, (mode, ds) in zip(axes.flat, panels):
        d = s[mode][ds]
        vals = ([d[m]["gross_all"] for m in METHODS]
                + [d["best_lens"]["gross_all"], d["ceiling"]["trusted_oracle"], d["hindsight"]["gross_all"]])
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{ds} ({mode})", fontsize=9)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Exp 047: gross error (abstention counted as wrong), family-aware fusion vs. baselines")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
