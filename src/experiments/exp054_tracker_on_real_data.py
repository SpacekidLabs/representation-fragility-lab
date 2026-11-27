"""
Experiment 054: why the pitch-tracker gain does not hold on real data (issue #28)
=================================================================================
Exp 039 found that the framework's AdaptivePitchTracker beats fixed YIN on
synthetic audio; Exp 045 found that on real recordings (vocadito,
MDB-stem-synth) it loses: 24.2% vs 7.6% gross errors (> 50 cents) on clean
vocadito. This experiment diagnoses why, and tries the obvious fixes.

(Issue #28 calls this "Exp 053"; that number went to the third medium, #27.)

Method
  Every per-hop ingredient is computed once (hop 512 at 22.05 kHz, the setup
  of Exp 045, clean and 10 dB noise, MDB limited to the first 30 s of each
  stem as in Exp 045):
    - the engine's recommendation (YIN window, trough threshold, voicing gate,
      hold flag, ACF assumption score)
    - the raw adaptive YIN estimate (the tracker's input)
    - fixed YIN (2048, threshold 0.15)
    - the three plain lenses (STFT, ACF, Cepstrum, 2048-sample frames) and
      their quorum consensus (Exp 046)
  The tracker's post-processing is then replayed with each step on or off:
    octave   octave correction against the last pitch
    cont     continuity limit (3 semitones per hop) with re-acquisition after 3 hops
    gates    the engine's voicing gate and hold flag
    median   5-hop causal median filter (reports the pitch of 2 hops ago)
    hyst     release hysteresis (hold the last pitch for 2 unvoiced hops)
  The full replay reproduces AdaptivePitchTracker.track exactly (checked).
  The fix kept, ``framework.plugins.QuorumPitchTracker``, is checked against
  its replay ("quorum_tracker") the same way.

Context of each annotated hop (from the annotation alone)
  onset (first 100 ms of a voiced segment), offset (last 100 ms),
  transition (|Δf0| > 1 semitone across ±2 hops), vibrato (detrended f0 std
  > 20 cents over ±160 ms), stable (the rest); and f0 range (< 150, 150-400,
  > 400 Hz).

Fixes tried (choices made on vocadito, MDB-stem-synth is held out)
  no_median, centred median (offline only), shorter median, no continuity /
  octave step, quorum or plain-lens input instead of adaptive YIN.

Metric: gross error rate (> 50 cents or no estimate) on annotated voiced hops,
and the 20% error of Exp 045; differences with group-bootstrap 95% CIs over tracks.

Run:  python src/experiments/exp054_tracker_on_real_data.py
Outputs: results/exp054_tracker_on_real_data.png, results/exp054_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore")

import librosa  # noqa: E402

import exp045_real_data as e45  # noqa: E402
from core import ConfidenceThresholdDetector, Reading  # noqa: E402
from framework.audio import PitchComparator  # noqa: E402
from framework.disagreement import (  # noqa: E402
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    OUTLIER_MARGIN,
    VOICING_THRESHOLD,
    read_acf,
    read_cepstrum,
    read_stft,
)
from framework.engine import RepresentationIntelligenceEngine  # noqa: E402
from framework.plugins.pitch_tracker import REACQUIRE_FRAMES, AdaptivePitchTracker, QuorumPitchTracker  # noqa: E402

SR, HOP, PAD = e45.SR, e45.HOP, 2048
LENSES = ("stft", "acf", "cepstrum")
READERS = {"stft": read_stft, "acf": read_acf, "cepstrum": read_cepstrum}
CONDITIONS = [e45.AUDIO_CONDITIONS[0], e45.AUDIO_CONDITIONS[2]]   # clean, noise 10 dB
CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp054_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp054_tracker_on_real_data.png")
GROSS = 50.0
CONTEXTS = ("onset", "offset", "transition", "vibrato", "stable")


# ──────────────────────────────────────────────────────────────────────────────
# Per-hop ingredients
# ──────────────────────────────────────────────────────────────────────────────

def ingredients(x, engine):
    """Everything the tracker and the baselines need, per hop."""
    n = len(x)
    y = np.pad(x, PAD, mode="reflect")
    hops = np.arange(0, n - HOP, HOP)
    out = {k: np.zeros(len(hops)) for k in ("win", "thr", "gate", "hold", "eng_acf", "ayin")}
    for k in LENSES:
        out[f"{k}_f0"] = np.zeros(len(hops))
        out[f"{k}_conf"] = np.zeros(len(hops))
    for i, c in enumerate(hops):
        f = y[c + PAD - 512:c + PAD + 512]
        st = engine.analyze(f, SR, compute_disagreement=False)
        win = st.recommended_window
        p = st.recommended_parameters["pitch_tracking"]
        out["win"][i], out["thr"][i] = win, p["yin_trough"]
        out["gate"][i], out["hold"][i] = p["voicing_gate"], p["hold_pitch"]
        out["eng_acf"][i] = st.assumptions.get("acf", 1.0)
        a = y[c + PAD - win // 2:c + PAD + win // 2]
        try:
            out["ayin"][i] = float(librosa.yin(a, fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX, sr=SR, frame_length=win,
                                               hop_length=win, trough_threshold=p["yin_trough"], center=False)[0])
        except Exception:
            out["ayin"][i] = 0.0
        frame = y[c + PAD - 1024:c + PAD + 1024]
        for k in LENSES:
            r = READERS[k](frame, SR)
            out[f"{k}_f0"][i], out[f"{k}_conf"][i] = r.f0, (r.confidence if np.isfinite(r.confidence) else 0.0)
    out["fyin"] = librosa.yin(x, fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX, sr=SR, frame_length=2048, hop_length=HOP,
                              trough_threshold=0.15, center=True)[:len(hops)]
    return hops, out


def quorum(ing, k=2):
    comp = PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=k,
                           detector=ConfidenceThresholdDetector(VOICING_THRESHOLD))
    est = np.zeros(len(ing["ayin"]))
    for i in range(len(est)):
        rd = {n: Reading(n, ing[f"{n}_f0"][i] if ing[f"{n}_f0"][i] > 0 else None, ing[f"{n}_conf"][i]) for n in LENSES}
        c = comp.compare(rd).consensus
        est[i] = c if c is not None else 0.0
    return est


def contexts(truth):
    """Context label per hop from the annotation (unvoiced hops: '')."""
    n = len(truth)
    lab = np.array([""] * n, dtype=object)
    voiced = truth > 0
    edge = int(round(0.1 * SR / HOP))
    semis = np.where(voiced, 12 * np.log2(np.where(voiced, truth, 1.0)), np.nan)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        for k in range(i, j):
            if k - i < edge:
                lab[k] = "onset"
            elif j - 1 - k < edge:
                lab[k] = "offset"
            else:
                a, b = max(k - 2, i), min(k + 2, j - 1)
                if abs(semis[b] - semis[a]) > 1.0:
                    lab[k] = "transition"
                else:
                    w = semis[max(k - 7, i):min(k + 8, j)]
                    t = np.arange(len(w))
                    resid = w - np.polyval(np.polyfit(t, w, 1), t) if len(w) > 2 else w * 0
                    lab[k] = "vibrato" if np.std(resid) > 0.2 else "stable"
        i = j
    return lab


def collect(name, refresh=False):
    path = os.path.join(CACHE, f"exp054_{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    engine = RepresentationIntelligenceEngine()
    seconds = e45.TRACKER_SECONDS_MDB if name == "mdb_stem_synth" else None
    tracks, t0 = [], time.time()
    for tid, wav, ann in e45.DATASETS[name]():
        y, times, f0 = e45.load_track(wav, ann)
        if seconds:
            y = y[:seconds * SR]
        for family, severity, fn in CONDITIONS:
            x = fn(y, e45.rng_for(name, tid, family, severity, "tracker"))
            hops, ing = ingredients(x, engine)
            if len(hops) == 0:
                continue
            idx = np.clip(np.searchsorted(times, hops / SR), 1, len(times) - 1)
            truth = np.where((f0[idx] > 0) & (f0[idx - 1] > 0), f0[idx], 0.0)
            ing["quorum"] = quorum(ing)
            tracks.append({"track": tid, "condition": f"{family} {severity}", "truth": truth,
                           "context": contexts(truth), **ing})
        print(f"    {name}: {len(tracks) // 2} tracks ({time.time() - t0:.0f} s)")
    with open(path, "wb") as f:
        pickle.dump(tracks, f)
    return tracks


# ──────────────────────────────────────────────────────────────────────────────
# The tracker's post-processing, replayed with switches
# ──────────────────────────────────────────────────────────────────────────────

FULL = {"source": "ayin", "octave": True, "cont": True, "gates": True, "median": 5, "centred": False, "hyst": True}


def replay(tr, source="ayin", octave=True, cont=True, gates=True, median=5, centred=False, hyst=True,
           fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX):
    """AdaptivePitchTracker.track's steps 1-4 on cached raw estimates.

    With the defaults it reproduces the tracker exactly. ``source`` picks the raw
    estimate ('ayin', 'fyin', 'quorum', a lens name, or 'quorum_ayin' = quorum
    when it answers, else adaptive YIN). ``median`` = window length (1 = off);
    ``centred`` uses past and future hops (offline only)."""
    if source == "quorum_ayin":
        raw_all = np.where(tr["quorum"] > 0, tr["quorum"], tr["ayin"])
    elif source == "quorum_fyin":
        raw_all = np.where(tr["quorum"] > 0, tr["quorum"], tr["fyin"])
    elif source in LENSES:
        raw_all = tr[f"{source}_f0"]
    else:
        raw_all = tr[source]
    n = len(raw_all)
    out = np.zeros(n)
    last, hist, unv, run = 0.0, [0.0] * median, 0, []
    voiced_mask = np.zeros(n, bool)
    pre = np.zeros(n)
    for i in range(n):
        pitch = float(raw_all[i])
        raw = pitch
        if (octave or cont) and last > 0 and pitch > 0:
            ratio = pitch / last
            if octave:
                if abs(ratio - 0.5) < 0.08:
                    pitch, ratio = pitch * 2, ratio * 2
                elif abs(ratio - 2.0) < 0.30:
                    pitch, ratio = pitch * 0.5, ratio * 0.5
                elif abs(ratio - 0.25) < 0.04:
                    pitch, ratio = pitch * 4, ratio * 4
                elif abs(ratio - 4.0) < 0.60:
                    pitch, ratio = pitch * 0.25, ratio * 0.25
            if cont and (ratio > 2 ** (3 / 12) or ratio < 2 ** (-3 / 12)):
                pitch = last
            if abs(12.0 * np.log2(pitch / raw)) > 1.0:
                if run and abs(12.0 * np.log2(raw / run[-1])) < 1.0:
                    run.append(raw)
                else:
                    run = [raw]
                if len(run) >= REACQUIRE_FRAMES:
                    pitch = raw
                    hist = [0.0] * median
                    run = []
            else:
                run = []
        rv = fmin < pitch < fmax
        if gates:
            if tr["gate"][i] and tr["eng_acf"][i] < 0.35:
                rv = False
            if tr["hold"][i] and tr["eng_acf"][i] < 0.20:
                pitch = last
                rv = fmin < pitch < fmax
        if rv:
            if not centred:
                if all(v == 0.0 for v in hist):
                    hist = [pitch] * median
                else:
                    hist.pop(0)
                    hist.append(pitch)
                pitch = float(np.median(hist))
            last = pitch
        else:
            pitch = 0.0
        is_v = rv
        if rv:
            unv = 0
        else:
            unv += 1
            if hyst and unv < 3 and fmin < last < fmax:
                is_v, pitch = True, last
            else:
                hist, last, run = [0.0] * median, 0.0, []
        out[i] = pitch if is_v else 0.0
        voiced_mask[i] = is_v
        pre[i] = pitch
    if centred and median > 1:
        h = median // 2
        sm = out.copy()
        for i in np.flatnonzero(voiced_mask):
            w = out[max(i - h, 0):i + h + 1]
            w = w[w > 0]
            sm[i] = float(np.median(w)) if len(w) else 0.0
        out = sm
    return out


def check_quorum_tracker(name):
    """QuorumPitchTracker equals the 'quorum_tracker' replay on the first track."""
    tid, wav, ann = next(iter(e45.DATASETS[name]()))
    y, _, _ = e45.load_track(wav, ann)
    y = y[:20 * SR]
    real = QuorumPitchTracker().track(y, SR, hop_length=HOP)
    _, ing = ingredients(y, RepresentationIntelligenceEngine())
    ing["quorum"] = quorum(ing)
    rep = methods()[2]["quorum_tracker"](ing)
    n = min(len(real), len(rep))
    return int(np.sum(np.abs(real[:n] - rep[:n]) > 1e-6)), n


def check_replay(name):
    """The full replay equals AdaptivePitchTracker.track on the first track."""
    tid, wav, ann = next(iter(e45.DATASETS[name]()))
    y, _, _ = e45.load_track(wav, ann)
    y = y[:20 * SR]
    engine = RepresentationIntelligenceEngine()
    real = AdaptivePitchTracker(engine).track(y, SR, hop_length=HOP)
    _, ing = ingredients(y, engine)
    ing["quorum"] = np.zeros(len(ing["ayin"]))
    rep = replay(ing)
    n = min(len(real), len(rep))
    return int(np.sum(np.abs(real[:n] - rep[:n]) > 1e-6)), n


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def keep_mask(truth):
    return (truth >= DEFAULT_FMIN) & (truth <= DEFAULT_FMAX)


def errors(est, truth):
    ok = est > 0
    cents = np.where(ok, np.abs(1200 * np.log2(np.where(ok, est, 1.0) / np.where(truth > 0, truth, 1.0))), np.inf)
    return cents > GROSS, np.where(ok, np.abs(est - truth) / np.where(truth > 0, truth, 1.0), np.inf) > 0.2


def evaluate(tracks, methods, lag=0):
    """Per method: gross flags on kept hops, with track ids and contexts aligned."""
    out = {m: [] for m in methods}
    out20 = {m: [] for m in methods}
    groups, ctx, rng_, cond = [], [], [], []
    for tr in tracks:
        k = keep_mask(tr["truth"])
        groups.append(np.full(k.sum(), tr["track"], dtype=object))
        ctx.append(tr["context"][k])
        f = tr["truth"][k]
        rng_.append(np.where(f < 150, "low", np.where(f > 400, "high", "mid")))
        cond.append(np.full(k.sum(), tr["condition"], dtype=object))
        for m, fn in methods.items():
            g, g20 = errors(fn(tr)[k], tr["truth"][k])
            out[m].append(g)
            out20[m].append(g20)
    cat = lambda v: np.concatenate(v)  # noqa: E731
    return ({m: cat(v) for m, v in out.items()}, {m: cat(v) for m, v in out20.items()},
            cat(groups), cat(ctx), cat(rng_), cat(cond))


def lagged(fn, lag):
    """Shift an estimate sequence back by ``lag`` hops (removes a known delay; offline)."""
    def g(tr):
        e = fn(tr)
        return np.r_[e[lag:], np.zeros(lag)]
    return g


def methods():
    lens = {f"lens_{n}": (lambda tr, n=n: tr[f"{n}_f0"]) for n in LENSES}
    base = {
        "fixed_yin": lambda tr: tr["fyin"],
        "adaptive_yin_raw": lambda tr: tr["ayin"],
        **lens,
        "quorum": lambda tr: tr["quorum"],
        "tracker": lambda tr: replay(tr),
        "tracker_lag2_removed": lagged(lambda tr: replay(tr), 2),
    }
    ablate = {
        "no_octave": lambda tr: replay(tr, octave=False),
        "no_cont": lambda tr: replay(tr, cont=False),
        "no_octave_no_cont": lambda tr: replay(tr, octave=False, cont=False),
        "no_gates": lambda tr: replay(tr, gates=False),
        "no_median": lambda tr: replay(tr, median=1),
        "median3": lambda tr: replay(tr, median=3),
        "median5_centred": lambda tr: replay(tr, centred=True),
        "no_hyst": lambda tr: replay(tr, hyst=False),
        "fixed_yin_input": lambda tr: replay(tr, source="fyin"),
    }
    fixes = {
        "fix_quorum_input": lambda tr: replay(tr, source="quorum_ayin"),
        "fix_no_median_no_cont": lambda tr: replay(tr, median=1, cont=False),
        "fix_no_median_no_cont_no_octave": lambda tr: replay(tr, median=1, cont=False, octave=False),
        "fix_quorum_no_median_no_cont": lambda tr: replay(tr, source="quorum_ayin", median=1, cont=False),
        "fix_fyin_no_median_no_cont": lambda tr: replay(tr, source="fyin", median=1, cont=False),
        "fix_fyin_median3_centred": lambda tr: replay(tr, source="fyin", median=3, centred=True, cont=False),
        "fix_quorum_fyin_no_median_no_cont": lambda tr: replay(tr, source="quorum_fyin", median=1, cont=False),
        "quorum_tracker": lambda tr: replay(tr, source="quorum_fyin", median=1, cont=False, octave=False,
                                            gates=False, hyst=False),
    }
    return base, ablate, fixes


def summarise(tracks):
    base, ablate, fixes = methods()
    allm = {**base, **ablate, **fixes}
    G, G20, groups, ctx, rng_, cond = evaluate(tracks, allm)
    s = {"n_hops": int(len(groups)), "n_tracks": int(len(np.unique(groups))), "by_condition": {}}
    for c in dict.fromkeys(cond):
        m = cond == c
        s["by_condition"][c] = {
            "n": int(m.sum()),
            "gross_50c": {k: float(v[m].mean()) for k, v in G.items()},
            "gross_20pct": {k: float(v[m].mean()) for k, v in G20.items()},
            "context_share": {x: float(np.mean(ctx[m] == x)) for x in CONTEXTS},
            "by_context": {x: {k: float(G[k][m & (ctx == x)].mean()) for k in allm} for x in CONTEXTS},
            "by_range": {x: {k: float(G[k][m & (rng_ == x)].mean()) for k in allm} for x in ("low", "mid", "high")},
            "range_share": {x: float(np.mean(rng_[m] == x)) for x in ("low", "mid", "high")},
        }
        lens_best = min((f"lens_{n}" for n in LENSES), key=lambda k: G[k][m].mean())
        best_single = min(["fixed_yin", lens_best], key=lambda k: G[k][m].mean())
        s["by_condition"][c]["best_plain_lens"] = lens_best
        s["by_condition"][c]["best_single"] = best_single
        s["by_condition"][c]["vs_best_single"] = {
            k: e45.group_bootstrap(groups[m], G[k][m], G[best_single][m], n_boot=1000)
            for k in ["tracker", "tracker_lag2_removed", "quorum", *ablate, *fixes]}
        s["by_condition"][c]["vs_fixed_yin"] = {
            k: e45.group_bootstrap(groups[m], G[k][m], G["fixed_yin"][m], n_boot=1000)
            for k in ["tracker", "tracker_lag2_removed", "quorum", *fixes]}
        # Where is the tracker's extra error? Share of (tracker wrong, fixed YIN right) by context.
        extra = G["tracker"][m] & ~G["fixed_yin"][m]
        s["by_condition"][c]["extra_error_share_by_context"] = {
            x: float(np.mean(ctx[m][extra] == x)) if extra.any() else float("nan") for x in CONTEXTS}
    return s


def main():
    print("replay check:", check_replay("vocadito"))
    out = {"replay_mismatch_vocadito_first_20s": check_replay("vocadito"),
           "quorum_tracker_mismatch_vocadito_first_20s": check_quorum_tracker("vocadito")}
    print("quorum tracker check:", out["quorum_tracker_mismatch_vocadito_first_20s"])
    for name in ("vocadito", "mdb_stem_synth"):
        print(f"{name}")
        out[name] = summarise(collect(name))
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(out)
    plot(out)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(out):
    for name in ("vocadito", "mdb_stem_synth"):
        for c, d in out[name]["by_condition"].items():
            print(f"\n=== {name} / {c}: {d['n']} hops; best single {d['best_single']} "
                  f"{d['gross_50c'][d['best_single']]:.1%}; contexts " +
                  ", ".join(f"{k} {v:.0%}" for k, v in d["context_share"].items()))
            for k, v in d["gross_50c"].items():
                ci = d["vs_best_single"].get(k)
                cs = f"  Δ {ci['diff']:+.1%} [{ci['ci95'][0]:+.1%}, {ci['ci95'][1]:+.1%}]" if ci else ""
                print(f"  {k:34s} {v:6.1%}  (20%: {d['gross_20pct'][k]:5.1%}){cs}")
            for k, ci in d["vs_fixed_yin"].items():
                print(f"    vs fixed YIN {k:34s} {ci['diff']:+.1%} [{ci['ci95'][0]:+.1%}, {ci['ci95'][1]:+.1%}]")
            print("  by context (fixed_yin / tracker / no_median / fix_fyin_no_median_no_cont):")
            for x in CONTEXTS:
                b = d["by_context"][x]
                print(f"    {x:10s} {b['fixed_yin']:6.1%} {b['tracker']:6.1%} {b['no_median']:6.1%} "
                      f"{b['fix_fyin_no_median_no_cont']:6.1%}")
            print("  by f0 range (fixed_yin / tracker / adaptive_yin_raw):")
            for x in ("low", "mid", "high"):
                b = d["by_range"][x]
                print(f"    {x:5s} ({d['range_share'][x]:.0%}) {b['fixed_yin']:6.1%} {b['tracker']:6.1%} "
                      f"{b['adaptive_yin_raw']:6.1%}")
            print("  tracker's extra errors by context: " + ", ".join(
                f"{k} {v:.0%}" for k, v in d["extra_error_share_by_context"].items()))


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["fixed_yin", "lens_stft", "lens_acf", "lens_cepstrum", "quorum", "adaptive_yin_raw", "tracker",
            "tracker_lag2_removed", "no_octave", "no_cont", "no_gates", "no_median", "median5_centred", "no_hyst",
            "fixed_yin_input", "fix_no_median_no_cont", "fix_quorum_no_median_no_cont", "fix_fyin_no_median_no_cont",
            "fix_fyin_median3_centred", "fix_quorum_fyin_no_median_no_cont", "quorum_tracker"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    for col, name in enumerate(("vocadito", "mdb_stem_synth")):
        ax = axes[0, col]
        conds = list(out[name]["by_condition"])
        w = 0.4
        for k, c in enumerate(conds):
            v = [out[name]["by_condition"][c]["gross_50c"][m] for m in show]
            ax.bar(np.arange(len(show)) + (k - 0.5) * w, v, w, label=c)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels(show, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("gross error (> 50 cents or none)")
        ax.set_title(f"{name}: baselines, the tracker, ablations and fixes")
        ax.legend()
        ax = axes[1, col]
        d = out[name]["by_condition"]["clean -"]
        ms = ["fixed_yin", "adaptive_yin_raw", "tracker", "no_median", "fix_fyin_no_median_no_cont"]
        for k, m in enumerate(ms):
            ax.bar(np.arange(len(CONTEXTS)) + (k - 2) * 0.16, [d["by_context"][x][m] for x in CONTEXTS], 0.16, label=m)
        ax.set_xticks(range(len(CONTEXTS)))
        ax.set_xticklabels([f"{x}\n({d['context_share'][x]:.0%} of hops)" for x in CONTEXTS])
        ax.set_ylabel("gross error, clean")
        ax.set_title(f"{name} clean: error by context")
        ax.legend(fontsize=8)
    fig.suptitle("Exp 054: why the adaptive pitch tracker loses to fixed YIN on real recordings")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    main()
