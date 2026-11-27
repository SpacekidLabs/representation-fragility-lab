"""
Experiment 051: disagreement over time (issue #25)
==================================================
Exp 042 predicts a lens's blind spot from one frame's disagreement (ROC AUC
0.95-0.99 on synthetic audio, 0.87-0.92 on real data in Exp 045). Here the
disagreement is treated as a trajectory: does the recent history of
disagreement predict a blind spot on the *next* frames, before it happens?

Sequences. Every hop (2048-sample frames, hop 1024 = 46 ms at 22.05 kHz) of
  synthetic   4-second harmonic signals (1, 3, 10, 20 harmonics; vibrato;
              an octave glide; a pulse train) at the six f0s of Exp 042
  vocadito    all 40 tracks
  MDB-stem-synth  20 s of each of the 230 stems, starting 1 s before the first
              annotated f0
under three conditions that change over time, so blind spots come and go:
  clean        the signal as is
  noise_ramp   white noise whose SNR follows 5 + 15 sin(2 pi t / 4 s) dB (-10 to 20 dB)
  lowpass_sweep  a low-pass whose cutoff sweeps 150 Hz - 4 kHz and back every 6 s
                 (cross-fade between six fixed 5th-order Butterworth low-passes)
  highpass_sweep a high-pass sweeping 40 Hz - 1.6 kHz and back every 5 s, which
                 removes the fundamental for part of the time (missing fundamental)
Truth per hop as in Exp 042 / 045 (geometric mean of the f0 inside the frame;
real frames that are partly voiced, spread > 100 cents or out of range are not
targets, but they stay in the history).

Target: lens L (STFT, ACF, Cepstrum) is wrong (> 50 cents, or no estimate) on
frame t + k, for horizons k = 0, 1, 2, 4, 8 hops (0 to 370 ms ahead), using
only frames <= t.

Predictors (logistic regression per lens and horizon, the same model class)
  current   the six per-frame disagreement features of Exp 042 at frame t
  history   current + the same features averaged over the last 8 frames
            (persistence), their slopes (trend), lag-1 values, the lens's pitch
            jump and pitch spread over the window, and two onset counters:
            frames since the lens last jumped by > 50 cents and frames since
            the last near-silent / all-unconfident frame (note onset)

Evaluation
  * grouped 5-fold CV within each dataset (by track / base signal)
  * transfer to held-out real data: train on synthetic + MDB-stem-synth, test
    on vocadito; train on synthetic + vocadito, test on MDB-stem-synth
  * AUC on all targets, and on *onsets*: targets where the lens is still right
    at t (the failure has not started yet)
  * lead time: for failure onsets (wrong at o, right on the 3 voiced frames
    before), the share caught by the k-step predictor at o - k, with the
    threshold set to a 10% false-alarm rate on right targets

Run:  python src/experiments/exp051_disagreement_over_time.py
Outputs: results/exp051_disagreement_over_time.png, results/exp051_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import librosa  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

import exp042_disagreement_prediction as e42  # noqa: E402
import exp045_real_data as e45  # noqa: E402
from framework.disagreement import (DEFAULT_FMAX, DEFAULT_FMIN, compare_readings, read_acf,  # noqa: E402
                                    read_cepstrum, read_stft)

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp051_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp051_disagreement_over_time.png")
SR, FRAME, GROSS = e42.SR, e42.FRAME, e42.GROSS_CENTS
HOP = 1024
LENSES = ("stft", "acf", "cepstrum")
READERS = {"stft": read_stft, "acf": read_acf, "cepstrum": read_cepstrum}
PAIRS = (("stft", "acf"), ("stft", "cepstrum"), ("acf", "cepstrum"))
HORIZONS = (0, 1, 2, 4, 8)
WINDOW = 8
FOLDS = 5
FALSE_ALARM = 0.10
SYN_SECONDS = 4.0
MDB_SECONDS = 20.0
LOWPASS_CUTOFFS = (150.0, 250.0, 400.0, 700.0, 1200.0, 4000.0)
HIGHPASS_CUTOFFS = (40.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
CONDITIONS = ("clean", "noise_ramp", "lowpass_sweep", "highpass_sweep")


def auc(score, label):
    """Mann-Whitney AUC with average ranks for ties (same as e42.roc_auc, vectorised)."""
    from scipy.stats import rankdata
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rng_for(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


# ──────────────────────────────────────────────────────────────────────────────
# Signals and time-varying conditions
# ──────────────────────────────────────────────────────────────────────────────

def noise_ramp(x, rng):
    t = np.arange(len(x)) / SR
    snr = 5.0 + 15.0 * np.sin(2 * np.pi * t / 4.0 + rng.uniform(0, 2 * np.pi))
    p = np.mean(x ** 2)
    return x + rng.normal(0.0, 1.0, len(x)) * np.sqrt(p / 10 ** (snr / 10.0))


def sweep(x, rng, cutoffs, btype, period):
    versions = [sosfiltfilt(butter(5, c / (0.5 * SR), btype=btype, output="sos"), x) for c in cutoffs]
    t = np.arange(len(x)) / SR
    # position on the cutoff ladder, 0 .. len-1, triangle wave with a 6 s period
    phase = (t / period + rng.uniform(0, 1)) % 1.0
    pos = (1 - np.abs(2 * phase - 1)) * (len(cutoffs) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(cutoffs) - 2)
    w = pos - lo
    V = np.stack(versions)
    idx = np.arange(len(x))
    return (1 - w) * V[lo, idx] + w * V[lo + 1, idx]


CONDITION_FNS = {"clean": lambda x, r: x, "noise_ramp": noise_ramp,
                 "lowpass_sweep": lambda x, r: sweep(x, r, LOWPASS_CUTOFFS, "low", 6.0),
                 "highpass_sweep": lambda x, r: sweep(x, r, HIGHPASS_CUTOFFS, "high", 5.0)}


def synthetic_signals():
    n = int(SR * SYN_SECONDS)
    t = np.arange(n) / SR
    for f0 in e42.F0S:
        flat = np.full(n, f0)
        variants = {f"harmonic{h}": (flat, [1.0 / k for k in range(1, h + 1)]) for h in (1, 3, 10, 20)}
        variants["vibrato10"] = (f0 * 2.0 ** ((50.0 / 1200.0) * np.sin(2 * np.pi * 5.5 * t)),
                                 [1.0 / k for k in range(1, 11)])
        glide = np.clip(f0 * 2.0 ** ((t - SYN_SECONDS / 2) / SYN_SECONDS), DEFAULT_FMIN, DEFAULT_FMAX)
        variants["glide5"] = (glide, [1.0 / k for k in range(1, 6)])
        variants["pulse_train"] = (flat, [1.0] * 200)
        for name, (track, amps) in variants.items():
            yield f"{f0:g}-{name}", e42.synth(track, amps), track


def synthetic_truth(track, starts):
    return np.array([float(np.exp(np.mean(np.log(track[s:s + FRAME])))) for s in starts])


# ──────────────────────────────────────────────────────────────────────────────
# Per-hop readings
# ──────────────────────────────────────────────────────────────────────────────

def read_sequence(x, starts):
    """Per hop: f0 [n, 3], conf [n, 3], divergence [n, 3], overall [n], pair term [n, 3],
    octave flag [n, 3], rms [n]."""
    n = len(starts)
    f0, conf, div, term, octv = (np.zeros((n, 3)) for _ in range(5))
    overall, rms = np.zeros(n), np.zeros(n)
    for i, s in enumerate(starts):
        frame = x[s:s + FRAME]
        rms[i] = np.sqrt(np.mean(frame ** 2))
        rd = {name: READERS[name](frame, SR) for name in LENSES}
        rep = compare_readings(rd)
        for j, name in enumerate(LENSES):
            f0[i, j], conf[i, j], div[i, j] = rd[name].f0, rd[name].confidence, rep.divergence[name]
        for j, (a, b) in enumerate(PAIRS):
            p = rep.pair(a, b)
            term[i, j], octv[i, j] = p.pitch_term, float(p.octave_related)
        overall[i] = rep.overall
    return {"f0": f0, "conf": conf, "div": div, "overall": overall, "term": term, "oct": octv, "rms": rms}


def collect(name):
    path = os.path.join(CACHE, f"exp051_seq_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    seqs, t0 = [], time.time()
    if name == "synthetic":
        for group, x, track in synthetic_signals():
            starts = np.arange(0, len(x) - FRAME, HOP)
            truth = synthetic_truth(track, starts)
            for cond in CONDITIONS:
                y = CONDITION_FNS[cond](x, rng_for(name, group, cond))
                seqs.append({"group": group, "condition": cond, "truth": truth, **read_sequence(y, starts)})
    else:
        for tid, wav, ann in e45.DATASETS[name]():
            a = np.loadtxt(ann, delimiter=",", ndmin=2)
            times, f = a[:, 0], a[:, 1]
            offset, duration = 0.0, None
            if name == "mdb_stem_synth":
                voiced = times[f > 0]
                if len(voiced) == 0:
                    continue
                offset, duration = max(0.0, float(voiced[0]) - 1.0), MDB_SECONDS
            y, _ = librosa.load(wav, sr=SR, mono=True, offset=offset, duration=duration)
            if len(y) < FRAME + 8 * HOP:
                continue
            starts = np.arange(0, len(y) - FRAME, HOP)
            shift = int(round(offset * SR))
            truth = np.array([e45.frame_truth(times, f, s + shift) for s in starts], dtype=object)
            truth = np.array([np.nan if v is None else float(v) for v in truth])
            for cond in CONDITIONS:
                z = CONDITION_FNS[cond](y, rng_for(name, tid, cond))
                seqs.append({"group": tid, "condition": cond, "truth": truth, **read_sequence(z, starts)})
            if len(seqs) % 60 == 0:
                print(f"    {name}: {len(seqs) // 3} tracks ({time.time() - t0:.0f} s)")
    with open(path, "wb") as f:
        pickle.dump(seqs, f)
    print(f"  {name}: {len(seqs)} sequences, {sum(len(s['truth']) for s in seqs)} hops, {time.time() - t0:.0f} s")
    return seqs


# ──────────────────────────────────────────────────────────────────────────────
# Features and targets
# ──────────────────────────────────────────────────────────────────────────────

def lens_wrong(seq):
    """[n, 3] bool, NaN-free; meaningful only where truth > 0."""
    t = seq["truth"][:, None]
    f = seq["f0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.abs(1200 * np.log2(np.where(f > 0, f, 1.0) / np.where(t > 0, t, 1.0)))
    return (f <= 0) | (c > GROSS)


def current_features(seq, j):
    """Exp 042's six per-frame features for lens j, per hop: [n, 6]."""
    mine = [k for k, (a, b) in enumerate(PAIRS) if LENSES[j] in (a, b)]
    others = [k for k in range(3) if k != j]
    return np.c_[seq["div"][:, j], seq["overall"], seq["conf"][:, j], seq["conf"][:, others].mean(1),
                 seq["term"][:, mine].min(1), seq["oct"][:, mine].max(1)]


def window_mean(X, w):
    c = np.cumsum(np.r_[np.zeros((1, X.shape[1])), X], 0)
    idx = np.arange(len(X))
    lo = np.maximum(idx - w + 1, 0)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)[:, None]


def window_slope(v, w):
    out = np.zeros(len(v))
    for i in range(1, len(v)):
        seg = v[max(0, i - w + 1):i + 1]
        x = np.arange(len(seg)) - (len(seg) - 1) / 2
        out[i] = (x @ (seg - seg.mean())) / (x @ x)
    return out


def since(event, cap):
    out = np.zeros(len(event))
    c = cap
    for i, e in enumerate(event):
        c = 0 if e else min(c + 1, cap)
        out[i] = c
    return out


def history_features(seq, j, cur):
    f = seq["f0"][:, j]
    logf = np.log2(np.where(f > 0, f, DEFAULT_FMIN / 2))
    jump = np.r_[0.0, 1200 * np.abs(np.diff(logf))]
    spread = np.array([1200 * logf[max(0, i - WINDOW + 1):i + 1].std() for i in range(len(f))])
    lag1 = np.r_[cur[:1], cur[:-1]]
    floor = 1e-3 * seq["rms"].max()
    quiet = (seq["rms"] < floor) | (seq["conf"].max(1) < 0.3)
    return np.c_[cur, window_mean(cur, WINDOW), lag1,
                 window_slope(cur[:, 0], WINDOW), window_slope(cur[:, 2], WINDOW), window_slope(cur[:, 1], WINDOW),
                 np.minimum(jump, 1200) / 1200, spread / 1200,
                 since(jump > GROSS, WINDOW) / WINDOW, since(quiet, 4 * WINDOW) / (4 * WINDOW)]


def build(seqs):
    """Per lens: rows over (sequence, t) with features at t and targets at t + k."""
    # The cepstrum reader returns NaN confidence on all-zero frames (digital silence in
    # some MDB stems); the comparison then has NaN divergence. Treat them as no reading.
    for sq in seqs:
        for key in ("conf", "div", "overall", "term"):
            sq[key] = np.nan_to_num(sq[key], nan=0.0)
    out = {}
    for j, lens in enumerate(LENSES):
        Xc, Xh, Y, R0, V, G, C, ON = [], [], [], [], [], [], [], []
        for s in seqs:
            n = len(s["truth"])
            cur = current_features(s, j)
            hist = history_features(s, j, cur)
            wrong = lens_wrong(s)[:, j]
            voiced = s["truth"] > 0
            y = np.zeros((n, len(HORIZONS)), bool)
            v = np.zeros((n, len(HORIZONS)), bool)
            for h, k in enumerate(HORIZONS):
                y[:n - k, h] = wrong[k:]
                v[:n - k, h] = voiced[k:]
            # onset: wrong at o, right on the three voiced frames before (all voiced)
            on = np.zeros(n, bool)
            for o in range(3, n):
                on[o] = voiced[o] and wrong[o] and voiced[o - 3:o].all() and not wrong[o - 3:o].any()
            Xc.append(cur), Xh.append(hist), Y.append(y), V.append(v), ON.append(on)
            R0.append(voiced & ~wrong)   # lens right at t (truth used only to stratify)
            G += [s["group"]] * n
            C += [s["condition"]] * n
        out[lens] = {"current": np.concatenate(Xc), "history": np.concatenate(Xh), "y": np.concatenate(Y),
                     "valid": np.concatenate(V), "right_now": np.concatenate(R0), "onset": np.concatenate(ON),
                     "seq_len": [len(s["truth"]) for s in seqs],
                     "group": np.array(G), "condition": np.array(C)}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def fit_predict(Xtr, ytr, Xte):
    if ytr.min() == ytr.max():
        return np.full(len(Xte), ytr.mean())
    return e42.fit_logistic(Xtr, ytr.astype(float))(Xte)


def scores_cv(d, kind, h):
    """Out-of-fold scores for horizon index h (NaN where the target is invalid)."""
    valid = d["valid"][:, h]
    X, y, g = d[kind], d["y"][:, h], d["group"]
    uniq = np.unique(g)
    np.random.default_rng(0).shuffle(uniq)
    out = np.full(len(y), np.nan)
    for fold in np.array_split(uniq, FOLDS):
        te = np.isin(g, fold)
        tr = ~te & valid
        out[te] = fit_predict(X[tr], y[tr], X[te])
    return out


def scores_transfer(train, test, kind, h):
    Xtr = np.concatenate([d[kind][d["valid"][:, h]] for d in train])
    ytr = np.concatenate([d["y"][:, h][d["valid"][:, h]] for d in train])
    return fit_predict(Xtr, ytr, test[kind])


def onset_index(d, k):
    """Row indices o - k for onsets o, within the same sequence."""
    idx = np.where(d["onset"])[0]
    starts = np.cumsum([0] + d["seq_len"][:-1])
    seq_start = starts[np.searchsorted(starts, idx, side="right") - 1]
    keep = idx - k >= seq_start
    return idx[keep] - k


def metrics(d, s, h):
    k = HORIZONS[h]
    valid = d["valid"][:, h] & ~np.isnan(s)
    y = d["y"][:, h]
    a_all = auc(s[valid], y[valid])
    pre = valid & d["right_now"]          # failure has not started at t
    auc_pre = auc(s[pre], y[pre]) if k > 0 else float("nan")
    thr = np.quantile(s[valid & ~y], 1 - FALSE_ALARM)
    ons = onset_index(d, k)
    ons = ons[~np.isnan(s[ons])]
    return {"auc": a_all, "auc_before_failure": auc_pre, "base_rate": float(y[valid].mean()),
            "onset_recall_at_10pct_fa": float(np.mean(s[ons] >= thr)) if len(ons) else float("nan"),
            "n_onsets": int(len(ons))}


def per_condition_auc(d, s, h):
    out = {}
    for c in CONDITIONS:
        m = d["valid"][:, h] & ~np.isnan(s) & (d["condition"] == c)
        out[c] = auc(s[m], d["y"][:, h][m])
    return out


def group_bootstrap_auc(d, sa, sb, h, n_boot=300, seed=0):
    """AUC(a) - AUC(b) with a 95% CI from resampling whole groups."""
    valid = d["valid"][:, h] & ~np.isnan(sa) & ~np.isnan(sb)
    g, y = d["group"][valid], d["y"][:, h][valid]
    a, b = sa[valid], sb[valid]
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(auc(a[pick], y[pick]) - auc(b[pick], y[pick]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": float(auc(a, y) - auc(b, y)), "ci95": [float(lo), float(hi)]}


def evaluate(data, setting, test_name, train_names=None):
    res = {}
    for lens in LENSES:
        d = data[test_name][lens]
        res[lens] = {}
        for h, k in enumerate(HORIZONS):
            sc = {}
            for kind in ("current", "history"):
                if train_names is None:
                    sc[kind] = scores_cv(d, kind, h)
                else:
                    sc[kind] = scores_transfer([data[n][lens] for n in train_names], d, kind, h)
            entry = {kind: metrics(d, sc[kind], h) for kind in sc}
            entry["history_minus_current_auc"] = group_bootstrap_auc(d, sc["history"], sc["current"], h)
            if k in (0, 2):
                entry["by_condition"] = {kind: per_condition_auc(d, sc[kind], h) for kind in sc}
            res[lens][k] = entry
    return res


def main():
    data = {}
    for name in ("synthetic", "vocadito", "mdb_stem_synth"):
        print(f"Sequences: {name}")
        seqs = collect(name)
        data[name] = build(seqs)
        data[name]["_info"] = {"n_sequences": len(seqs), "n_hops": int(sum(len(s["truth"]) for s in seqs)),
                               "n_tracks": len({s["group"] for s in seqs}),
                               "voiced_targets": int(sum(np.sum(s["truth"] > 0) for s in seqs))}
    s = {"horizons_hops": HORIZONS, "hop_ms": 1000 * HOP / SR, "window": WINDOW, "datasets": {}, "results": {}}
    for name in data:
        info = dict(data[name]["_info"])
        info["lens_error_rate"] = {lens: float(data[name][lens]["y"][:, 0][data[name][lens]["valid"][:, 0]].mean())
                                   for lens in LENSES}
        info["n_onsets"] = {lens: int(data[name][lens]["onset"].sum()) for lens in LENSES}
        s["datasets"][name] = info
    plans = {
        "cv_synthetic": ("synthetic", None),
        "cv_vocadito": ("vocadito", None),
        "cv_mdb_stem_synth": ("mdb_stem_synth", None),
        "transfer_to_vocadito": ("vocadito", ["synthetic", "mdb_stem_synth"]),
        "transfer_to_mdb_stem_synth": ("mdb_stem_synth", ["synthetic", "vocadito"]),
    }
    for key, (test, train) in plans.items():
        print(f"Evaluating {key}")
        s["results"][key] = evaluate(data, key, test, train)
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=float)
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    for name, info in s["datasets"].items():
        print(f"{name}: {info}")
    for key, res in s["results"].items():
        print(f"\n{key}")
        for lens, by_k in res.items():
            for k, e in by_k.items():
                c, h, dd = e["current"], e["history"], e["history_minus_current_auc"]
                print(f"  {lens:9s} k={k}: AUC {c['auc']:.3f} -> {h['auc']:.3f} "
                      f"({dd['diff']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]); "
                      f"before failure {c['auc_before_failure']:.3f} -> {h['auc_before_failure']:.3f}; "
                      f"onsets caught {100 * c['onset_recall_at_10pct_fa']:.0f}% -> "
                      f"{100 * h['onset_recall_at_10pct_fa']:.0f}% (n={h['n_onsets']}, base {100 * c['base_rate']:.0f}%)")


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["cv_synthetic", "transfer_to_vocadito", "transfer_to_mdb_stem_synth"]
    titles = ["synthetic (CV)", "vocadito (held out)", "MDB-stem-synth (held out)"]
    colors = {"stft": "#1a73e8", "acf": "#e37400", "cepstrum": "#188038"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    ms = [1000 * HOP / SR * k for k in HORIZONS]
    for col, (key, title) in enumerate(zip(keys, titles)):
        res = s["results"][key]
        for lens in LENSES:
            for kind, ls in (("current", "--"), ("history", "-")):
                aucs = [res[lens][k][kind]["auc"] for k in HORIZONS]
                rec = [res[lens][k][kind]["onset_recall_at_10pct_fa"] for k in HORIZONS]
                lab = f"{lens} {kind}"
                axes[0, col].plot(ms, aucs, ls, marker="o", color=colors[lens], label=lab)
                axes[1, col].plot(ms, 100 * np.array(rec), ls, marker="o", color=colors[lens], label=lab)
        axes[0, col].set_title(title)
        axes[1, col].set_xlabel("horizon (ms ahead)")
    axes[0, 0].set_ylabel("ROC AUC, lens wrong at t + k")
    axes[1, 0].set_ylabel("failure onsets caught k ahead (%)\n(10% false alarms)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Exp 051: does the history of disagreement predict the next blind spot? (dashed: current frame only)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)


if __name__ == "__main__":
    main()
