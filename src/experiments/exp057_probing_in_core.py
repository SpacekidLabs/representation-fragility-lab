"""
Experiment 057: probing in the core (issue #30)
===============================================
Exp 050 settled pitch disagreements by perturbing the frame with a known effect
and keeping the candidate the lenses still read. Its best chooser, the untrained
low-pass follow count, lived in the experiment. Issue #30 moves it into the
core: ``Medium.probes(signal, candidate, context)`` supplies the perturbations,
``core.Prober`` picks the candidate with the most follows, and
``Analyzer(prober=...)`` runs it on every signal where the lenses disagree
(``Analysis.answer``).

Part 1, audio: the same frames as Exp 050. The disagreement frames of the
synthetic set (Exp 042), vocadito and a seeded sample of 6000 MDB-stem-synth
frames are rebuilt with Exp 050's code and run through
``make_audio_analyzer(prober=Prober())``; the probe is a 5th-order low-pass at
1.5 x the candidate. Checks: the core's readings match the cached ones, and its
choice matches Exp 050's "lowpass count" frame by frame. Differences come from
how candidates are formed (the core merges readings that agree with a
candidate's most confident member and fuses them with the medium's ``fuse``;
Exp 050 chained sorted estimates within 50 cents and took the geometric mean).
Unpitched frames with two or more candidates give the voicing check (AUC of the
winner's share of follows, and the trade-off of ``min_score``).

Part 2, images (new): the five classic orientation lenses on the synthetic
patches (Exp 044) and BSDS500 (Exp 045). Probes: Gaussian blur (prediction:
the candidate; it keeps a coarse dominant pattern and removes noise and fine
weaker detail) and a rotation by a known angle (prediction: the candidate moved
by that angle). Probe sets are compared on the synthetic patches; BSDS500 is
held out. Baselines on the same disagreement patches: legacy consensus,
quorum 2 (abstention = wrong), the best single lens in hindsight and the
candidate oracle. Group-bootstrap 95% CIs. Unoriented patches (Exp 044) give
the voicing-style check.

Run:  python src/experiments/exp057_probing_in_core.py   (needs the Exp 045,
      046 and 050 caches and the raw audio datasets)
Outputs: results/exp057_probing_in_core.png, results/exp057_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp044_image_disagreement as e44  # noqa: E402
import exp045_real_data as e45  # noqa: E402
import exp046_quorum as e46  # noqa: E402
import exp050_probing as e50  # noqa: E402
import exp052_learned_lens as e52  # noqa: E402
from core import ConfidenceThresholdDetector, Prober, Reading  # noqa: E402
from framework import make_audio_analyzer  # noqa: E402
from image_medium import ImagePatch, make_image_analyzer  # noqa: E402
from image_medium.medium import VOTING_THRESHOLD, OrientationComparator  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "exp057_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp057_probing_in_core.png")
SR, FRAME, GROSS = e50.SR, e50.FRAME, e50.GROSS
LP = e50.PROBES.index("lowpass")
IMAGE_PROBE_SETS = {
    "blur 1.5": ((1.5,), ()),
    "blur 3": ((3.0,), ()),
    "rotate 30": ((), (30.0,)),
    "blur 1.5 + rotate 30": ((1.5,), (30.0,)),
    "blur 1.5, 3 + rotate 30, 60": ((1.5, 3.0), (30.0, 60.0)),
}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(e52.auc(np.r_[pos, neg], np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]))


# ──────────────────────────────────────────────────────────────────────────────
# Part 1: audio, through the core
# ──────────────────────────────────────────────────────────────────────────────

def exp050_choice(res):
    """Exp 050's 'lowpass count' choice on a cached probe result."""
    k = e50.pick(res, list(res["bits"][:, LP, :].sum(1)))
    return res["cands"][k][0]


def audio(name):
    rows = e46.cached(e50.DATASETS[name], None)
    rows = rows[0] if isinstance(rows, tuple) else rows
    cached = e50.probe_dataset(name)["probes"]
    wanted = set(cached)
    an = make_audio_analyzer(lenses=e50.LENSES, prober=Prober())
    gen = (e50.synthetic_contexts(wanted) if name == "synthetic"
           else e50.real_contexts(name, wanted, e45.MDB_FRAMES_PER_TRACK if name == "mdb_stem_synth" else None))
    out = {"idx": [], "core": [], "e50": [], "truth": [], "group": [], "family": [], "score": [], "n_cand": []}
    mismatch, t = 0, time.time()
    for i, ctx in gen:
        a = an.analyze(ctx[:FRAME], {"sr": SR})
        rd = a.report.readings
        mismatch += int(any(e50.cents(rd[n].estimate or 0.0, rows[i]["readings"][n][0]) > 1.0
                            for n in e50.LENSES if rows[i]["readings"][n][0] > 0))
        core = a.probe.choice if a.probe is not None else a.report.consensus
        out["idx"].append(i)
        out["core"].append(core if core is not None else 0.0)
        out["e50"].append(exp050_choice(cached[i]))
        out["truth"].append(rows[i]["truth"])
        out["group"].append(str(rows[i]["group"]))
        out["family"].append(rows[i]["family"])
        out["score"].append(a.probe.score if a.probe is not None else np.nan)
        out["n_cand"].append(len(a.probe.candidates) if a.probe is not None else 1)
    print(f"  {name}: {len(out['idx'])} disagreement frames in {time.time() - t:.0f} s, reading mismatch {mismatch}")

    nul_scores, n_null = [], 0
    for i, ctx in e50.unpitched_contexts(name):
        n_null += 1
        a = an.analyze(ctx[:FRAME], {"sr": SR})
        if not a.empty and a.probe is not None:
            nul_scores.append(a.probe.score)

    truth = np.array(out["truth"])
    groups = np.array(out["group"])
    wrong_core = np.array([e50.cents(c, t) > GROSS for c, t in zip(out["core"], truth)])
    wrong_e50 = np.array([e50.cents(c, t) > GROSS for c, t in zip(out["e50"], truth)])
    same = np.array([e50.cents(a, b) <= GROSS for a, b in zip(out["core"], out["e50"])])
    score = np.array(out["score"])
    probed = ~np.isnan(score)
    pitched_scores = score[probed]
    right = ~wrong_core[probed]
    nul = np.array(nul_scores)
    thresholds = {}
    for t in (1 / 3, 2 / 3, 1.0):
        ans = pitched_scores >= t - 1e-9
        thresholds[f"{t:.2f}"] = {"answer": float(ans.mean()), "gross_all": float(np.mean(~(ans & right))),
                                  "false_answer_unpitched_2plus": float(np.mean(nul >= t - 1e-9)) if len(nul) else float("nan")}
    e50s = json.load(open(os.path.join(ROOT, "results", "exp050_summary.json")))["datasets"][name]
    return {
        "n_frames": len(truth), "reading_mismatch": mismatch,
        "n_core_probed": int(probed.sum()),
        "gross_core": float(wrong_core.mean()), "gross_exp050_lowpass_count": float(wrong_e50.mean()),
        "same_choice": float(same.mean()),
        "core_minus_exp050": e45.group_bootstrap(groups, wrong_core, wrong_e50),
        "exp050_reference": {k: e50s["gross"][k] for k in ("legacy", "q2", "best_lens", "lowpass_count", "count", "oracle")},
        "voicing": {"n_unpitched": n_null, "n_unpitched_probed": int(len(nul)),
                    "auc_score_pitched_vs_unpitched": auc(pitched_scores, nul), "thresholds": thresholds},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Part 2: images
# ──────────────────────────────────────────────────────────────────────────────

def image_data(name):
    rows, nulls = e52.image_cached(name)
    X, N = e52.image_patches(name)
    if name == "bsds":
        syn_rows, syn_nulls = e52.image_cached("synthetic")
        _, N = e52.image_patches("synthetic")
        nulls = syn_nulls
    keep = [i for i, r in enumerate(rows) if not r["empty"]]
    nkeep = [i for i, r in enumerate(nulls) if not r["empty"]]
    return [rows[i] for i in keep], X[keep], N[nkeep]


def image_run(name, sets):
    rows, X, N = image_data(name)
    comp = OrientationComparator(min_agreeing=2, detector=ConfidenceThresholdDetector(VOTING_THRESHOLD))
    legacy = OrientationComparator(min_agreeing=None, detector=ConfidenceThresholdDetector(VOTING_THRESHOLD))
    an = make_image_analyzer()
    for n, rep in zip(e44.REPS, an.representations):          # calibrated contrasts of Exp 044
        rep.contrast = e44.LENSES[n].contrast

    def wrong(est, r):
        return est is None or e44.err_deg(est, r["truth"]) > e44.GROSS_DEG

    prober = Prober()
    dis = [i for i, r in enumerate(rows)
           if len(prober.candidates(comp, {n: Reading(n, *r["readings"][n]) for n in e44.REPS})) >= 2]
    mismatch = 0
    for i in dis[:: max(1, len(dis) // 40)]:
        rep = an.analyze(X[i]).report
        mismatch += int(any(abs(e44.angle_diff(rep.readings[n].estimate, rows[i]["readings"][n][0])) > 1e-6
                            for n in e44.REPS if rows[i]["readings"][n][0] is not None))
    sub = [rows[i] for i in dis]
    groups = np.array([str(r["group"]) for r in sub])
    W = {}
    rd = [{n: Reading(n, *r["readings"][n]) for n in e44.REPS} for r in sub]
    W["legacy"] = np.array([wrong(legacy.compare(x).consensus, r) for x, r in zip(rd, sub)])
    W["q2"] = np.array([wrong(comp.compare(x).consensus, r) for x, r in zip(rd, sub)])
    lens = {n: np.array([wrong(r["readings"][n][0], r) for r in sub]) for n in e44.REPS}
    best = min(lens, key=lambda n: lens[n].mean())
    W["best_lens"] = lens[best]
    W["oracle"] = np.array([all(wrong(c, r) for c, _ in prober.candidates(comp, x)) for x, r in zip(rd, sub)])

    out = {"n_rows": len(rows), "n_disagree": len(dis), "reading_mismatch_checked": mismatch, "best_lens": best,
           "sets": {}}
    for label, (blur, rot) in sets.items():
        medium = ImagePatch()
        medium.PROBE_BLUR, medium.PROBE_ROTATE = blur, rot
        an.medium = medium
        an.prober = Prober()
        t = time.time()
        res = [an.analyze(X[i]) for i in dis]
        W[label] = np.array([wrong(a.answer, r) for a, r in zip(res, sub)])
        sc = np.array([a.probe.score for a in res if a.probe is not None])
        nul = []
        for x in N:
            a = an.analyze(x)
            if not a.empty and a.probe is not None:
                nul.append(a.probe.score)
        out["sets"][label] = {"gross": float(W[label].mean()), "seconds": time.time() - t,
                              "vs_q2": e45.group_bootstrap(groups, W[label], W["q2"]),
                              "vs_legacy": e45.group_bootstrap(groups, W[label], W["legacy"]),
                              "vs_best_lens": e45.group_bootstrap(groups, W[label], W["best_lens"]),
                              "auc_score_oriented_vs_unoriented": auc(sc, nul),
                              "n_unoriented_probed": len(nul), "n_unoriented": len(N)}
        print(f"  {name} / {label}: {out['sets'][label]['gross']:.1%} in {time.time() - t:.0f} s")
    out["gross"] = {k: float(v.mean()) for k, v in W.items() if k in ("legacy", "q2", "best_lens", "oracle")}
    fam = np.array([r["family"] for r in sub])
    out["by_family"] = {f: {k: float(v[fam == f].mean()) for k, v in W.items()} for f in dict.fromkeys(fam)}
    out["n_by_family"] = {f: int((fam == f).sum()) for f in dict.fromkeys(fam)}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def main():
    s = {"audio": {}, "image": {}}
    print("Part 1: audio")
    for name in e50.DATASETS:
        s["audio"][name] = audio(name)
    print("Part 2: images")
    s["image"]["synthetic"] = image_run("synthetic", IMAGE_PROBE_SETS)
    syn = s["image"]["synthetic"]["sets"]
    chosen = min(syn, key=lambda k: syn[k]["gross"])
    s["image"]["chosen_on_synthetic"] = chosen
    s["image"]["bsds"] = image_run("bsds", {k: v for k, v in IMAGE_PROBE_SETS.items()})
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    print("\n=== Audio: core Prober (low-pass) vs Exp 050 'lowpass count', disagreement frames")
    for name, d in s["audio"].items():
        c = d["core_minus_exp050"]
        r = d["exp050_reference"]
        print(f"  {name}: n {d['n_frames']}, mismatch {d['reading_mismatch']}, core {d['gross_core']:.1%} vs "
              f"Exp 050 {d['gross_exp050_lowpass_count']:.1%} (diff {c['diff']:+.1%} [{c['ci95'][0]:+.1%}, {c['ci95'][1]:+.1%}]), "
              f"same choice {d['same_choice']:.1%}; ref legacy {r['legacy']:.1%}, q2 {r['q2']:.1%}, "
              f"best lens {r['best_lens']:.1%}, oracle {r['oracle']:.1%}")
        v = d["voicing"]
        print(f"    voicing AUC {v['auc_score_pitched_vs_unpitched']:.3f} ({v['n_unpitched_probed']} of {v['n_unpitched']} "
              f"unpitched frames probed); " + ", ".join(
                  f"min_score {t}: answer {x['answer']:.1%}, gross_all {x['gross_all']:.1%}, false {x['false_answer_unpitched_2plus']:.1%}"
                  for t, x in v["thresholds"].items()))
    print(f"\n=== Images (probe set chosen on synthetic: {s['image']['chosen_on_synthetic']})")
    for name in ("synthetic", "bsds"):
        d = s["image"][name]
        g = d["gross"]
        print(f"  {name}: {d['n_disagree']} of {d['n_rows']} patches disagree; legacy {g['legacy']:.1%}, q2 {g['q2']:.1%}, "
              f"best lens {g['best_lens']:.1%} ({d['best_lens']}), oracle {g['oracle']:.1%}; reading mismatch {d['reading_mismatch_checked']}")
        for k, v in d["sets"].items():
            print(f"    {k:28s} {v['gross']:6.1%}  vs legacy {v['vs_legacy']['diff']:+.1%} "
                  f"[{v['vs_legacy']['ci95'][0]:+.1%}, {v['vs_legacy']['ci95'][1]:+.1%}]  vs best lens "
                  f"{v['vs_best_lens']['diff']:+.1%} [{v['vs_best_lens']['ci95'][0]:+.1%}, {v['vs_best_lens']['ci95'][1]:+.1%}]  "
                  f"unoriented AUC {v['auc_score_oriented_vs_unoriented']:.3f}  ({v['seconds']:.0f} s)")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    names = list(s["audio"])
    keys = [("legacy", "consensus"), ("q2", "quorum 2"), ("best_lens", "best lens"), ("lowpass_count", "Exp 050 low-pass"),
            ("core", "core Prober"), ("oracle", "oracle")]
    x = np.arange(len(names))
    w = 0.13
    for j, (k, lab) in enumerate(keys):
        vals = [s["audio"][n]["gross_core"] if k == "core" else s["audio"][n]["exp050_reference"][k] for n in names]
        ax.bar(x + (j - 2.5) * w, vals, w, label=lab)
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("gross error on disagreement frames")
    ax.set_title("Audio: the core Prober reproduces Exp 050")
    ax.legend(fontsize=8)
    ax = axes[1]
    sets = list(s["image"]["synthetic"]["sets"])
    cols = ["legacy", "q2", "best_lens", *sets, "oracle"]
    x = np.arange(2)
    w = 0.8 / len(cols)
    for j, k in enumerate(cols):
        vals = [s["image"][n]["sets"][k]["gross"] if k in sets else s["image"][n]["gross"][k] for n in ("synthetic", "bsds")]
        ax.bar(x + (j - len(cols) / 2 + 0.5) * w, vals, w, label=k.replace("_", " "))
    ax.set_xticks(x, ["synthetic patches", "BSDS500 (held out)"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("gross error on disagreement patches")
    ax.set_title(f"Images: probe sets (chosen on synthetic: {s['image']['chosen_on_synthetic']})")
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Exp 057: probing in the core")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
