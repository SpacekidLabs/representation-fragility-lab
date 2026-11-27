"""
Experiment 056: one lens answers, the others warn (issue #31)
=============================================================
Exp 052 (CREPE) and Exp 053 (the ECG peak counter) reached the same decision:
where one lens is clearly better, it should answer and the others should only
warn, instead of being one equal vote in a quorum. Issue #31 puts that rule in
the core (``PairwiseComparator(lead=..., lead_fallback=...)``) and tests it on
all three media.

Lead lens per medium (the most accurate lens of its pool, Exp 052-055)
  audio        crepe     pool stft, acf, cepstrum, crepe   synthetic, vocadito, MDB-stem-synth
  images       radon     pool tensor, hog, fft, gabor, radon, cnn_aug   synthetic, BSDS500
  time series  peaks     pool fft, acf, wavelet, peaks     synthetic, MIT-BIH + NSTDB
All readings come from the caches of Exp 045, 046, 052 and 053; no lens is re-run.

Rules
  q2            the core default: quorum of two agreeing trusted lenses
  lead          the lead answers when trusted, else abstains
  lead+quorum   the lead answers when trusted, else the quorum of the others
  lead raw      the lead's estimate on every frame, trusted or not (reference)
  lead+warn     ``lead``, but also abstaining on the frames with the highest
                warning, so that it answers exactly as often as q2. The threshold
                is set on the same data: this shows what the warning is worth at
                q2's answer rate, it is not a tuned rule

Metrics (gross error: >50 cents / >10 degrees / >10% of the rate)
  answer, gross_answer, gross_all (abstention = wrong), false_answer on null input
  warning AUC   over frames where the lead answers: ROC AUC for "the lead is
                wrong" of the core warning (the lead's divergence), of the share
                of trusted other lenses that do not agree with it, and of the
                lead's own confidence
Group-bootstrap 95% CIs for gross_all differences.

Run:  python src/experiments/exp056_lead_lens.py
Outputs: results/exp056_lead_lens.png, results/exp056_summary.json
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

import exp045_real_data as e45  # noqa: E402
import exp052_learned_lens as e52  # noqa: E402
import exp053_timeseries_medium as e53  # noqa: E402
import exp055_lens_value as e55  # noqa: E402
from core import ConfidenceThresholdDetector, Reading  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import OUTLIER_MARGIN, VOICING_THRESHOLD  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD as IMAGE_THRESHOLD, OrientationComparator  # noqa: E402
from timeseries_medium import RateComparator  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "exp056_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp056_lead_lens.png")
LEAD = {"audio": "crepe", "image": "radon", "timeseries": "peaks"}
RULES = ("q2", "lead", "lead+quorum", "lead raw", "lead+warn")


class Audio:
    pool = e55.POOLS["audio"]
    wrong = staticmethod(e52.Audio.wrong)

    @staticmethod
    def readings(row):
        return e52.readings(e52.Audio, row, Audio.pool)

    @staticmethod
    def comparator(**kw):
        return PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=2,
                               detector=ConfidenceThresholdDetector(VOICING_THRESHOLD), **kw)


class Image:
    pool = e55.POOLS["image"]
    wrong = staticmethod(e52.Image.wrong)

    @staticmethod
    def readings(row):
        return e52.readings(e52.Image, row, Image.pool)

    @staticmethod
    def comparator(**kw):
        return OrientationComparator(min_agreeing=2, detector=ConfidenceThresholdDetector(IMAGE_THRESHOLD), **kw)


class Series:
    pool = e53.REPS
    wrong = staticmethod(e53.wrong)

    @staticmethod
    def readings(row):
        return {n: Reading(n, e, c) for n, (e, c) in row["readings"].items()}

    @staticmethod
    def comparator(**kw):
        return RateComparator(min_agreeing=2, detector=ConfidenceThresholdDetector(e53.VOTING_THRESHOLD), **kw)


MEDIA = {"audio": Audio, "image": Image, "timeseries": Series}


def datasets():
    out = {(m, e55.LABELS[n]): v for (m, n), v in e55.datasets().items()}
    for name, label in (("synthetic", "synthetic"), ("mitdb", "MIT-BIH")):
        with open(os.path.join(e53.CACHE, f"exp053_{name}.pkl"), "rb") as f:
            rows, nulls = pickle.load(f)
        out[("timeseries", label)] = ([r for r in rows if not r["empty"]], [r for r in nulls if not r["empty"]])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# One dataset
# ──────────────────────────────────────────────────────────────────────────────

def run(M, rows, lead):
    q2, ld, lq = M.comparator(), M.comparator(lead=lead), M.comparator(lead=lead, lead_fallback="quorum")
    est = {k: [] for k in RULES if k != "lead+warn"}
    warn, support, conf = [], [], []
    for r in rows:
        rd = M.readings(r)
        est["q2"].append(q2.compare(rd).consensus)
        rep = ld.compare(rd)
        est["lead"].append(rep.consensus)
        est["lead+quorum"].append(lq.compare(rd).consensus)
        est["lead raw"].append(rd[lead].estimate)
        warn.append(rep.warning)
        trusted = [n for n, a in rep.assessments.items() if n != lead and not a.blind]
        agree = sum(ld.agrees(rep.pair(lead, n)) for n in trusted)
        support.append(1.0 - agree / len(trusted) if trusted else 1.0)
        conf.append(rd[lead].confidence)
    return est, np.array(warn, float), np.array(support), np.array(conf)


def null_answers(M, nulls, lead):
    q2, ld, lq = M.comparator(), M.comparator(lead=lead), M.comparator(lead=lead, lead_fallback="quorum")
    out = {"q2": [], "lead": [], "lead+quorum": [], "lead raw": [], "_warn": []}
    for r in nulls:
        rd = M.readings(r)
        rep = ld.compare(rd)
        out["q2"].append(q2.compare(rd).consensus is not None)
        out["lead"].append(rep.consensus is not None)
        out["lead+quorum"].append(lq.compare(rd).consensus is not None)
        out["lead raw"].append(rd[lead].estimate is not None)
        out["_warn"].append(rep.warning)
    return out


def analyze(medium, rows, nulls):
    M = MEDIA[medium]
    lead = LEAD[medium]
    groups = np.array([str(r["group"]) for r in rows])
    est, warn, support, conf = run(M, rows, lead)
    nul = null_answers(M, nulls, lead)

    # lead+warn: the lead's answers, minus the highest-warning ones, down to q2's answer rate
    answered = np.array([e is not None for e in est["lead"]])
    q2_rate = np.mean([e is not None for e in est["q2"]])
    keep = int(round(q2_rate * len(rows)))
    if answered.sum() > keep:
        order = np.argsort(np.where(answered, warn, np.inf), kind="stable")
        thr = warn[order[keep - 1]] if keep > 0 else -np.inf
        warn_ans = answered & (warn <= thr)
    else:
        thr, warn_ans = float("inf"), answered
    est["lead+warn"] = [e if a else None for e, a in zip(est["lead"], warn_ans)]
    nul["lead+warn"] = [a and w is not None and w <= thr for a, w in zip(nul["lead"], nul["_warn"])]

    W = {k: np.array([M.wrong(e, r) for e, r in zip(v, rows)]) for k, v in est.items()}
    out = {"lead": lead, "n": len(rows), "n_null": len(nulls), "lead_warn_threshold": float(thr), "rules": {}}
    for k in RULES:
        a = np.array([e is not None for e in est[k]])
        out["rules"][k] = {"answer": float(a.mean()), "gross_answer": float(W[k][a].mean()) if a.any() else float("nan"),
                           "gross_all": float(W[k].mean()), "false_answer": float(np.mean(nul[k])) if nulls else float("nan")}
    out["vs_q2"] = {k: e45.group_bootstrap(groups, W[k], W["q2"]) for k in RULES if k != "q2"}
    out["lead+quorum_vs_lead_raw"] = e45.group_bootstrap(groups, W["lead+quorum"], W["lead raw"])

    lead_wrong = W["lead raw"]
    m = answered
    out["warning_auc"] = {"core warning (lead divergence)": e52.auc(warn[m], lead_wrong[m]),
                          "share of trusted others disagreeing": e52.auc(support[m], lead_wrong[m]),
                          "own confidence": e52.auc(-conf[m], lead_wrong[m]),
                          "n_answered": int(m.sum()), "lead_wrong_when_answering": float(lead_wrong[m].mean())}
    # Would the quorum have been right where the lead is answering and wrong?
    q2_right_when_lead_wrong = (~W["q2"]) & m & lead_wrong
    out["q2_right_where_lead_wrong"] = float(q2_right_when_lead_wrong.sum() / max(1, (m & lead_wrong).sum()))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def main():
    s = {"lead": LEAD, "pools": {k: list(v.pool) for k, v in MEDIA.items()}, "datasets": {}}
    for (medium, name), (rows, nulls) in datasets().items():
        t = time.time()
        s["datasets"][f"{medium} / {name}"] = analyze(medium, rows, nulls)
        print(f"{medium} / {name}: {len(rows)} rows, {len(nulls)} null, {time.time() - t:.0f} s")
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    for name, d in s["datasets"].items():
        print(f"\n=== {name} (lead {d['lead']}, n = {d['n']}, null {d['n_null']})")
        print(f"  {'rule':12s} {'answer':>7s} {'gross|ans':>9s} {'gross_all':>9s} {'false':>7s}   Δgross_all vs q2 [CI95]")
        for k, v in d["rules"].items():
            c = d["vs_q2"].get(k)
            ci = f"{c['diff']:+6.1%} [{c['ci95'][0]:+6.1%}, {c['ci95'][1]:+6.1%}]" if c else ""
            print(f"  {k:12s} {v['answer']:7.1%} {v['gross_answer']:9.1%} {v['gross_all']:9.1%} {v['false_answer']:7.1%}   {ci}")
        c = d["lead+quorum_vs_lead_raw"]
        print(f"  lead+quorum - lead raw: {c['diff']:+.1%} [{c['ci95'][0]:+.1%}, {c['ci95'][1]:+.1%}]")
        w = d["warning_auc"]
        print("  warning AUC (lead wrong, where it answers; lead wrong {:.1%} of {}): ".format(
            w["lead_wrong_when_answering"], w["n_answered"]) + ", ".join(
            f"{k} {v:.3f}" for k, v in w.items() if not k.startswith(("n_", "lead_"))))
        print(f"  q2 right where the answering lead is wrong: {d['q2_right_where_lead_wrong']:.1%}")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(s["datasets"])
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    cols = {"q2": "#777777", "lead": "#4a7fd4", "lead+quorum": "#2b4f8f", "lead raw": "#c9a227", "lead+warn": "#5aa36b"}
    x = np.arange(len(names))
    w = 0.16
    for i, k in enumerate(RULES):
        axes[0].bar(x + (i - 2) * w, [s["datasets"][n]["rules"][k]["gross_all"] for n in names], w,
                    color=cols[k], label=k)
    axes[0].set_xticks(x, [n.replace(" / ", "\n") for n in names], fontsize=8)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("gross error, abstention counted as wrong")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Answer quality per rule")
    for i, k in enumerate(RULES):
        axes[2].bar(x + (i - 2) * w, [s["datasets"][n]["rules"][k]["false_answer"] for n in names], w, color=cols[k])
    axes[2].set_xticks(x, [n.replace(" / ", "\n") for n in names], fontsize=8)
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[2].set_ylabel("answers on null input (no rhythm / pitch / orientation)")
    axes[2].set_title("False answers per rule")
    keys = ["core warning (lead divergence)", "share of trusted others disagreeing", "own confidence"]
    for i, k in enumerate(keys):
        axes[1].bar(x + (i - 1) * 0.25, [s["datasets"][n]["warning_auc"][k] for n in names], 0.25, label=k)
    axes[1].axhline(0.5, color="k", lw=0.6, ls=":")
    axes[1].set_ylim(0.2, 1.0)
    axes[1].set_xticks(x, [n.replace(" / ", "\n") for n in names], fontsize=8)
    axes[1].set_ylabel("ROC AUC for 'the lead is wrong'")
    axes[1].legend(fontsize=8)
    axes[1].set_title("How well the other lenses warn")
    fig.suptitle("Exp 056: one lens answers, the others warn (leads: CREPE, Radon, peak counter)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
