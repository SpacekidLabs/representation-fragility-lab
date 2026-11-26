"""
Experiment 046: a quorum before the consensus (issue #20)
=========================================================
Exp 044 and 045 showed that the core's consensus follows any single confident
lens: on unoriented noise and on unvoiced audio it answers whenever one lens is
(wrongly) sure. Issue #20 adds a quorum to ``PairwiseComparator``: the
consensus is formed only from the largest group of trusted readings that all
agree with each other, and only if that group has at least ``min_agreeing``
members.

This experiment measures the rule before and after, on the same rows as
Exp 042 (synthetic audio), Exp 044 (synthetic images) and Exp 045 (vocadito,
MDB-stem-synth, BSDS500). Every row stores each lens's estimate and
confidence, so the consensus is recomputed from them with each rule; the
lenses are not re-run.

Rules compared
  legacy   min_agreeing=None  fuse all trusted readings (the old behaviour)
  q1       min_agreeing=1     fuse the largest agreeing group, even of one
  q2       min_agreeing=2     ... only if at least two trusted lenses agree
  q3       min_agreeing=3     ... only if at least three agree

Metrics, per dataset (gross error = >50 cents for audio, >10 degrees for images)
  answer        fraction of pitched / oriented frames with a consensus
  gross_answer  gross error rate among those answers
  gross_all     gross error rate counting abstentions as wrong
  answer_clean  answer rate on the clean condition only
  false_answer  fraction of unpitched / unoriented inputs given an answer

Real-data rows come from the Exp 045 cache (data/processed); run Exp 045 first.
Synthetic rows are collected with Exp 042 / 044's own code and cached here.

Run:  python src/experiments/exp046_quorum.py [--refresh]
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

import exp042_disagreement_prediction as e42  # noqa: E402
import exp044_image_disagreement as e44  # noqa: E402
import exp045_real_data as e45  # noqa: E402
from core import ConfidenceThresholdDetector, Reading  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import OUTLIER_MARGIN, VOICING_THRESHOLD  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD, OrientationComparator  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp046_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp046_quorum.png")
RULES = {"legacy": None, "q1": 1, "q2": 2, "q3": 3}


def cached(name, fn, refresh=False):
    path = os.path.join(CACHE, f"{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    val = fn()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(val, f)
    print(f"  {name}: computed in {time.time() - t:.0f} s")
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Recompute the consensus from stored readings
# ──────────────────────────────────────────────────────────────────────────────

def audio_comparator(k):
    return PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=k,
                           detector=ConfidenceThresholdDetector(VOICING_THRESHOLD))


def image_comparator(k):
    return OrientationComparator(min_agreeing=k, detector=ConfidenceThresholdDetector(VOTING_THRESHOLD))


def audio_consensus(row, comp):
    rd = {n: Reading(n, f0 if f0 > 0 else None, c) for n, (f0, c) in row["readings"].items()}
    c = comp.compare(rd).consensus
    return 0.0 if c is None else float(c)


def image_consensus(row, comp):
    if row["empty"]:
        return None
    rd = {n: Reading(n, e, c) for n, (e, c) in row["readings"].items()}
    return comp.compare(rd).consensus


def measure(kind, rows, nulls):
    """Metrics per rule. ``rows`` have ground truth; ``nulls`` are unpitched / unoriented."""
    if kind == "audio":
        make, cons = audio_comparator, audio_consensus
        wrong = lambda e, r: e42.cents_error(e, r["truth"]) > e42.GROSS_CENTS  # noqa: E731
        answered = lambda e: e > 0  # noqa: E731
        lenses = e42.REPS
        lens_est = lambda r, n: r["readings"][n][0]  # noqa: E731
    else:
        rows = [r for r in rows if not r["empty"]]
        make, cons = image_comparator, image_consensus
        wrong = lambda e, r: e44.err_deg(e, r["truth"]) > e44.GROSS_DEG  # noqa: E731
        answered = lambda e: e is not None  # noqa: E731
        lenses = e44.REPS
        lens_est = lambda r, n: r["readings"][n][0]  # noqa: E731

    groups = np.array([str(r["group"]) for r in rows])
    clean = np.array([r["family"] == "clean" for r in rows])
    out, wrong_all = {"n": len(rows), "n_null": len(nulls)}, {}
    for name, k in RULES.items():
        comp = make(k)
        est = [cons(r, comp) for r in rows]
        ans = np.array([answered(e) for e in est])
        bad = np.array([wrong(e, r) for e, r in zip(est, rows)])  # abstention counts as wrong
        null_ans = np.mean([answered(cons(r, comp)) for r in nulls]) if nulls else float("nan")
        wrong_all[name] = bad
        if name == "q2":
            est_q2 = est
        out[name] = {
            "answer": float(ans.mean()),
            "gross_answer": float(bad[ans].mean()) if ans.any() else float("nan"),
            "gross_all": float(bad.mean()),
            "answer_clean": float(ans[clean].mean()) if clean.any() else float("nan"),
            "false_answer": float(null_ans),
        }
    # Reference: the best single lens (its raw estimate, as in Exp 042-045).
    lens_wrong = {n: np.array([wrong(lens_est(r, n), r) for r in rows]) for n in lenses}
    best = min(lens_wrong, key=lambda n: lens_wrong[n].mean())
    out["best_lens"] = {"name": best, "gross_all": float(lens_wrong[best].mean())}
    # Does the quorum only answer the easy frames? Best lens on the frames q2 answers,
    # and where q2 abstains (answer rate per condition family).
    q2_ans = np.array([answered(e) for e in est_q2])
    out["best_lens_gross_on_q2_answered"] = float(lens_wrong[best][q2_ans].mean())
    out["q2_answer_by_family"] = {
        f: float(q2_ans[[r["family"] == f for r in rows]].mean())
        for f in dict.fromkeys(r["family"] for r in rows)}
    out["q2_minus_legacy_gross_all"] = e45.group_bootstrap(groups, wrong_all["q2"], wrong_all["legacy"])
    out["q2_minus_best_lens_gross_all"] = e45.group_bootstrap(groups, wrong_all["q2"], lens_wrong[best])
    return out


def check_legacy(kind, rows):
    """The legacy rule must reproduce the consensus stored by Exp 045 (pre-quorum rows)."""
    if kind == "audio":
        comp = audio_comparator(None)
        diff = [abs(audio_consensus(r, comp) - r["consensus_f0"]) for r in rows[:2000]]
    else:
        comp = image_comparator(None)
        diff = [0.0 if (a := image_consensus(r, comp)) == r["consensus"] else 1.0 for r in rows[:2000]]
    return float(max(diff))


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

def load_e45(name):
    path = os.path.join(CACHE, f"exp045_{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    refresh = "--refresh" in sys.argv
    s = {"rules": {k: v for k, v in RULES.items()}, "datasets": {}}

    print("Exp 042 rows (synthetic audio)")
    rows = cached("exp046_e42_rows", e42.collect, refresh)
    nulls = cached("exp046_e42_unpitched", e42.collect_unpitched, refresh)
    s["datasets"]["synthetic audio (Exp 042)"] = measure("audio", rows, nulls)

    print("Exp 044 rows (synthetic images)")
    rows = cached("exp046_e44_rows", e44.collect, refresh)
    nulls = cached("exp046_e44_unoriented", e44.collect_unoriented, refresh)
    s["datasets"]["synthetic images (Exp 044)"] = measure("image", rows, nulls)

    s["legacy_check_max_diff"] = {}
    for name, label in (("vocadito", "vocadito (Exp 045)"), ("mdb_stem_synth", "MDB-stem-synth (Exp 045)")):
        data = load_e45(f"audio_{name}")
        if data is None:
            print(f"Skipping {name}: run Exp 045 first")
            continue
        rows, unpitched = data
        print(f"Exp 045 rows ({name})")
        s["legacy_check_max_diff"][name] = check_legacy("audio", rows + unpitched)
        s["datasets"][label] = measure("audio", rows, unpitched)

    rows = load_e45("image_rows")
    if rows is not None:
        print("Exp 045 rows (BSDS500)")
        s["legacy_check_max_diff"]["bsds"] = check_legacy("image", [r for r in rows if not r["empty"]])
        # BSDS has no unoriented set of its own; reuse Exp 044's noise patches.
        unoriented = cached("exp046_e44_unoriented", e44.collect_unoriented)
        s["datasets"]["BSDS500 (Exp 045)"] = measure("image", rows, unoriented)

    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=str)
    plot(s)
    report(s)
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


def report(s):
    print("\nlegacy rule reproduces Exp 045's stored consensus, max diff:", s["legacy_check_max_diff"])
    for ds, d in s["datasets"].items():
        print(f"\n{ds}: n={d['n']}, null inputs={d['n_null']}, best lens {d['best_lens']['name']} "
              f"gross_all={d['best_lens']['gross_all']:.1%}")
        print(f"  {'rule':7s} {'answer':>7s} {'gross|ans':>9s} {'gross_all':>9s} {'ans_clean':>9s} {'false_ans':>9s}")
        for r in RULES:
            m = d[r]
            print(f"  {r:7s} {m['answer']:7.1%} {m['gross_answer']:9.1%} {m['gross_all']:9.1%} "
                  f"{m['answer_clean']:9.1%} {m['false_answer']:9.1%}")
        print(f"  best lens on the frames q2 answers: {d['best_lens_gross_on_q2_answered']:.1%} "
              f"(q2: {d['q2']['gross_answer']:.1%})")
        print("  q2 answer rate by family:", {k: f"{v:.0%}" for k, v in d["q2_answer_by_family"].items()})
        print(f"  q2 - legacy (gross_all): {d['q2_minus_legacy_gross_all']}")
        print(f"  q2 - best lens (gross_all): {d['q2_minus_best_lens_gross_all']}")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(s["datasets"])
    metrics = [("false_answer", "False answers on unpitched / unoriented input"),
               ("gross_answer", "Gross error when answering"),
               ("answer_clean", "Answer rate on clean input"),
               ("gross_all", "Gross error, abstention counted as wrong")]
    colors = {"legacy": "#9aa0a6", "q1": "#c3a6e0", "q2": "#4a7fd4", "q3": "#1f3b73"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    x = np.arange(len(names))
    w = 0.2
    for ax, (key, title) in zip(axes.flat, metrics):
        for i, r in enumerate(RULES):
            ax.bar(x + (i - 1.5) * w, [s["datasets"][n][r][key] for n in names], w, label=r, color=colors[r])
        ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0, 0].legend(title="rule", fontsize=8)
    fig.suptitle("Exp 046: a quorum before the consensus (legacy = fuse all trusted; qk = at least k agree)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
