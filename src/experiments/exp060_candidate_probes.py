"""
Experiment 060: image probes tied to the candidate and to the patch centre (issue #36)
=====================================================================================
Exp 057 settled image disagreements with blur and rotation probes. On photos that
only matched Radon (28.1% vs 29.9% gross error on disagreement patches; candidate
oracle 11.6%): global perturbations move every real structure alike, so an
occluder's edge follows its probes as well as the true boundary does.

New opt-in probe kinds (``ImagePatch``, src/image_medium/medium.py)
  along-blur σ   1-D blur along the candidate. A straight edge keeps its direction
                 under any 1-D blur; this keeps the candidate's own edge sharp and
                 widens the others, so every candidate is favoured by its own probe.
  cross-blur σ   1-D blur along the candidate's normal: widens the candidate's own
                 edge most; a control for the opposite bias.
  centre σ       the patch faded towards its mean away from the centre (isotropic
                 Gaussian window). Not tied to the candidate but to where the truth
                 is: the BSDS500 truth is a boundary through the centre, occluder
                 edges and clutter usually are not. No direction is favoured.

Probe sets are chosen on the synthetic patches (Exp 044) and BSDS500 is held out, as
in Exp 057, with the same disagreement patches, baselines (consensus, quorum 2, best
lens, candidate oracle), group-bootstrap CIs and unoriented-patch AUC (90 synthetic
unoriented patches).

Run:  python src/experiments/exp060_candidate_probes.py   (needs the Exp 045, 046
      and 052 caches in data/processed; about 30 minutes)
Outputs: results/exp060_candidate_probes.png, results/exp060_summary.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp057_probing_in_core as e57  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "exp060_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp060_candidate_probes.png")
NONE = {"PROBE_BLUR": (), "PROBE_ROTATE": ()}
DEFAULT = {"PROBE_BLUR": (1.5, 3.0), "PROBE_ROTATE": (30.0, 60.0)}
SETS = {
    "exp057 default (blur 1.5, 3 + rotate 30, 60)": DEFAULT,
    "along-blur 3": {**NONE, "PROBE_ALONG_BLUR": (3.0,)},
    "cross-blur 3": {**NONE, "PROBE_CROSS_BLUR": (3.0,)},
    "centre 8": {**NONE, "PROBE_CENTRE": (8.0,)},
    "centre 12": {**NONE, "PROBE_CENTRE": (12.0,)},
    "centre 8, 16": {**NONE, "PROBE_CENTRE": (8.0, 16.0)},
    "centre 8, 16 + rotate 30, 60": {**NONE, "PROBE_ROTATE": (30.0, 60.0), "PROBE_CENTRE": (8.0, 16.0)},
    "default + centre 8, 16": {**DEFAULT, "PROBE_CENTRE": (8.0, 16.0)},
}


def main():
    s = {"sets": {k: {a: list(v) for a, v in d.items()} for k, d in SETS.items()}}
    print("synthetic")
    s["synthetic"] = e57.image_run("synthetic", SETS)
    syn = s["synthetic"]["sets"]
    s["chosen_on_synthetic"] = min(syn, key=lambda k: syn[k]["gross"])
    print("BSDS500 (held out)")
    s["bsds"] = e57.image_run("bsds", SETS)
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    print(f"\nchosen on synthetic: {s['chosen_on_synthetic']}")
    for name in ("synthetic", "bsds"):
        d = s[name]
        g = d["gross"]
        print(f"\n=== {name}: {d['n_disagree']} disagreement patches of {d['n_rows']}; best lens {d['best_lens']}")
        print(f"  consensus {g['legacy']:.1%}, quorum {g['q2']:.1%}, best lens {g['best_lens']:.1%}, oracle {g['oracle']:.1%}")
        for k, v in d["sets"].items():
            b = v["vs_best_lens"]
            print(f"  {k:46s} {v['gross']:6.1%}  vs best lens {b['diff']:+6.1%} [{b['ci95'][0]:+.1%}, {b['ci95'][1]:+.1%}]"
                  f"  AUC unoriented {v['auc_score_oriented_vs_unoriented']:.3f}  {v['seconds']:.0f} s")
        fams = list(d["by_family"])
        print("  by condition: " + ", ".join(fams))
        for k in ["best_lens", "oracle"] + list(d["sets"]):
            print(f"    {k[:46]:46s} " + " ".join(f"{d['by_family'][f][k]:6.1%}" for f in fams))


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name, title in ((axes[0], "synthetic", "synthetic (sets chosen here)"),
                            (axes[1], "bsds", "BSDS500 (held out)")):
        d = s[name]
        keys = list(d["sets"])
        v = np.array([d["sets"][k]["gross"] for k in keys])
        ci = np.array([[d["sets"][k]["vs_best_lens"]["ci95"][0], d["sets"][k]["vs_best_lens"]["ci95"][1]] for k in keys])
        best = d["gross"]["best_lens"]
        y = np.arange(len(keys))
        cols = ["#d4724a" if "centre" in k else ("#999999" if "default" not in k else "#4a7fd4") for k in keys]
        lo, hi = best + ci[:, 0], best + ci[:, 1]           # the CI of (set - best lens), placed at the best lens
        ax.barh(y, v, color=cols, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], ecolor="k", capsize=2)
        for val, lab, st in ((best, f"best lens ({d['best_lens']})", "-"), (d["gross"]["legacy"], "consensus", "--"),
                             (d["gross"]["oracle"], "oracle", ":")):
            ax.axvline(val, color="k", ls=st, lw=0.9)
            ax.annotate(lab, (val, len(keys) - 0.4), fontsize=7, rotation=90, xytext=(2, 0), textcoords="offset points")
        ax.set_yticks(y, keys, fontsize=7)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("gross error on disagreement patches (bars: CI of the difference to the best lens)", fontsize=7)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Exp 060: probes tied to the candidate (blur along / across it) and to the patch centre", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
