"""
Experiment 053: a third medium, periodic sensor signals (issue #27)
===================================================================
A stronger test that ``src/core`` is medium-agnostic: the rate of a periodic
sensor signal (heart rate from an ECG, pulse rate from a PPG, step rate from
an accelerometer), read by four lenses on the **unchanged** core. Only
``src/timeseries_medium`` is new: a Medium, four lenses and a rate distance.

Quantity: cycles per minute in an 8 s window (30-240 / min).
Gross error: more than 10% from the truth (the comparator's "agree" tolerance).

Lenses (src/timeseries_medium)
  fft       strongest spectral line
  acf       highest autocorrelation peak at 0.25-2 s
  wavelet   Morlet scalogram (constant Q) with harmonic summation
  peaks     time-domain event counting (band-pass 5-15 Hz, Pan-Tompkins style)

Data
  synthetic  6 signal types (ECG, ECG with a tall T wave, ECG with respiratory
             sinus arrhythmia, ECG in ventricular bigeminy, PPG-like pulse,
             accelerometer-like steps) × 7 rates (45-180 / min) × 10 conditions
             (clean, white noise 10 / 0 dB, baseline wander, 60 Hz mains, muscle
             noise, motion artefacts, clipping, 4 Hz low-pass, dropouts) × 3 seeds.
             Truth = 60 / mean beat interval inside the window.
  mitdb      MIT-BIH Arrhythmia Database (PhysioNet, 48 records, 360 Hz, lead 1):
             60 windows per record (seeded), truth from the cardiologists' beat
             annotations (60 / mean RR inside the window; windows with
             ventricular flutter / fibrillation markers are skipped). Conditions:
             clean, and real electrode-motion (em), muscle (ma) and baseline-wander
             (bw) noise from the MIT-BIH Noise Stress Test Database added at 6 and
             0 dB (power SNR over the window).
  nulls      no rhythm at all: white and brown noise windows (synthetic); NSTDB
             em / ma / bw noise alone (real).

Questions (as Exp 042 / 044)
  1. Atlas: does each lens have its own structured blind spots?
  2. Prediction: does disagreement predict when a lens is wrong (ROC AUC, 5-fold
     CV split by group), better than the lens's own confidence?
  3. Answer: legacy consensus, quorum 2 and the Exp 047 AgreementTable (CV, and
     trained on synthetic → used on MIT-BIH) vs the best single lens.
  4. Families: co-failure lift and phi, raw and within condition.
  5. Nulls: how often does a lens or the consensus report a rate that is not there?
  6. Core changes needed: checked with git (none expected).

Run:  python src/experiments/exp053_timeseries_medium.py [--calibrate]
Outputs: results/exp053_timeseries_medium.png, results/exp053_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
import zlib
from itertools import combinations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp042_disagreement_prediction as e42  # noqa: E402
import exp045_real_data as e45  # noqa: E402
from core import AgreementTable, ConfidenceThresholdDetector, Reading  # noqa: E402
from timeseries_medium import LENSES, RateComparator, is_gross, make_timeseries_analyzer  # noqa: E402
from timeseries_medium.medium import VOTING_THRESHOLD  # noqa: E402

FS = 360
WIN = 8.0
N = int(FS * WIN)
SYN_DUR = 12.0
RATES = (45.0, 60.0, 75.0, 95.0, 120.0, 150.0, 180.0)
SEEDS = (0, 1, 2)
WINDOWS_PER_RECORD = 60
REAL_SNRS = (6, 0)
NULLS_PER_KIND = 30
FOLDS = 5
REPS = tuple(LENSES)
PAIRS = list(combinations(REPS, 2))

RAW = os.path.join(ROOT, "data", "raw")
MITDB = os.path.join(RAW, "mitdb", "mit-bih-arrhythmia-database-1.0.0")
NSTDB = os.path.join(RAW, "nstdb", "mit-bih-noise-stress-test-database-1.0.0")
CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp053_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp053_timeseries_medium.png")

BEAT_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 34, 35, 38, 41}
NORMAL_CODES = {1, 2, 3, 11, 12, 34}        # N, L, R, j, paced, e: not ectopic
FLUTTER_CODES = {31, 32, 33}                # flutter wave, VF on, VF off


def rng_for(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ──────────────────────────────────────────────────────────────────────────────
# PhysioNet (WFDB) files, read with numpy: format 212 signals, MIT annotations
# ──────────────────────────────────────────────────────────────────────────────

def read_header(path):
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln and not ln.startswith("#")]
    rec = lines[0].split()
    fs, n = float(rec[2]), int(rec[3])
    sigs = []
    for ln in lines[1:1 + int(rec[1])]:
        p = ln.split()
        gain = float(p[2].split("(")[0].split("/")[0]) or 200.0
        sigs.append({"file": p[0], "format": p[1], "gain": gain, "zero": int(p[4])})
    return fs, n, sigs


def read_212(path, nsig):
    b = np.fromfile(path, dtype=np.uint8)
    b = b[:len(b) // 3 * 3].reshape(-1, 3).astype(np.int32)
    s0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s1 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    s = np.empty(2 * len(b), np.int32)
    s[0::2], s[1::2] = s0, s1
    s[s > 2047] -= 4096
    return s[:len(s) // nsig * nsig].reshape(-1, nsig)


def read_record(folder, name):
    fs, n, sigs = read_header(os.path.join(folder, f"{name}.hea"))
    assert all(s["format"] == "212" for s in sigs)
    x = read_212(os.path.join(folder, sigs[0]["file"]), len(sigs))[:n]
    return fs, np.stack([(x[:, i] - s["zero"]) / s["gain"] for i, s in enumerate(sigs)], 1)


def read_annotations(path):
    """MIT annotation format: (sample indices, type codes)."""
    w = np.fromfile(path, dtype="<u2")
    t, samples, codes = 0, [], []
    i = 0
    while i < len(w):
        a, v = int(w[i]) >> 10, int(w[i]) & 0x3FF
        i += 1
        if a == 0 and v == 0:
            break
        if a == 59:                                  # SKIP: 32-bit interval follows
            t += (int(w[i]) << 16) | int(w[i + 1])
            i += 2
        elif a == 63:                                # AUX: v bytes follow
            i += (v + 1) // 2
        elif a in (60, 61, 62):                      # NUM, SUB, CHN
            pass
        else:
            t += v
            samples.append(t)
            codes.append(a)
    return np.array(samples), np.array(codes)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic periodic signals
# ──────────────────────────────────────────────────────────────────────────────

_T = np.arange(int(FS * SYN_DUR)) / FS


def _gauss(t0, amp, mu, w):
    return amp * np.exp(-0.5 * ((_T - t0 - mu) / w) ** 2)


def beat_times(bpm, rng, rsa=0.0):
    rr0, t, out = 60.0 / bpm, -1.0, []
    while t < SYN_DUR + 1.0:
        out.append(t)
        mod = 1.0 + rsa * np.sin(2 * np.pi * 0.25 * t)
        t += rr0 * mod * (1.0 + 0.02 * rng.normal())
    return np.array(out)


def ecg_beat(t0, rr, t_amp=0.3, t_w=0.06):
    x = _gauss(t0, 0.15, -0.2 * min(rr, 1.0), 0.025) + _gauss(t0, -0.1, -0.025, 0.01) + _gauss(t0, 1.0, 0.0, 0.01)
    x += _gauss(t0, -0.25, 0.025, 0.01) + _gauss(t0, t_amp, 0.25 * np.sqrt(rr), t_w)
    return x


def make_signal(variant, bpm, rng):
    """(signal over SYN_DUR, beat/step times)."""
    rr = 60.0 / bpm
    x = np.zeros_like(_T)
    if variant == "bigeminy":
        # pairs: normal beat, then a premature wide ventricular beat, then a compensatory pause
        times, t, k = [], -1.0, 0
        while t < SYN_DUR + 1.0:
            times.append((t, k % 2))
            t += rr * (0.7 if k % 2 == 0 else 1.3) * (1.0 + 0.02 * rng.normal())
            k += 1
        for t0, pvc in times:
            x += (_gauss(t0, -1.2, 0.0, 0.04) + _gauss(t0, 0.5, 0.3, 0.08)) if pvc else ecg_beat(t0, rr)
        return x, np.array([t for t, _ in times])
    times = beat_times(bpm, rng, rsa=0.08 if variant == "ecg_rsa" else 0.0)
    for k, t0 in enumerate(times):
        rr_k = np.diff(times)[min(k, len(times) - 2)]
        if variant in ("ecg", "ecg_rsa"):
            x += ecg_beat(t0, rr_k)
        elif variant == "ecg_tall_t":
            x += ecg_beat(t0, rr_k, t_amp=1.0, t_w=0.035)
        elif variant == "ppg":
            x += _gauss(t0, 1.0, 0.15 * rr_k, 0.12 * rr_k) + _gauss(t0, 0.4, 0.5 * rr_k, 0.08 * rr_k)
        elif variant == "steps":
            amp = 1.0 if k % 2 == 0 else 0.6                       # left / right foot
            x += _gauss(t0, amp, 0.0, 0.015) + _gauss(t0, 0.4, 0.12, 0.08)
    if variant == "steps":
        x += 0.3 * np.sin(np.pi * bpm / 60.0 * _T + rng.uniform(0, 2 * np.pi))   # sway at half the step rate
    return x, times


VARIANTS = ("ecg", "ecg_tall_t", "ecg_rsa", "bigeminy", "ppg", "steps")


def add_snr(x, noise, snr_db):
    return x + noise * np.sqrt(np.mean((x - x.mean()) ** 2) / max(np.mean(noise ** 2), 1e-30) / 10 ** (snr_db / 10))


def _band_noise(rng, lo, hi, n):
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, rng.normal(size=n))


def motion(x, rng):
    y = x.copy()
    t = 0.0
    big = np.max(np.abs(x - np.median(x)))
    while True:
        t += rng.exponential(1.0)
        if t >= len(x) / FS:
            return y
        k = int(t * FS)
        tail = np.arange(len(x) - k) / FS
        y[k:] += rng.choice([-1, 1]) * 3.0 * big * np.exp(-tail / 0.2)


def dropout(x, rng):
    y = x.copy()
    for _ in range(2):
        k = rng.integers(0, len(x) - int(0.8 * FS))
        y[k:k + int(0.8 * FS)] = np.median(x)
    return y


def clip(x):
    m = np.median(x)
    lim = 0.3 * np.max(np.abs(x - m))
    return np.clip(x, m - lim, m + lim)


def lowpass4(x):
    return sosfiltfilt(butter(4, 4.0, fs=FS, output="sos"), x)


CONDITIONS = [
    ("clean", "-", lambda x, r: x),
    ("noise", "10 dB", lambda x, r: add_snr(x, r.normal(size=len(x)), 10)),
    ("noise", "0 dB", lambda x, r: add_snr(x, r.normal(size=len(x)), 0)),
    ("baseline", "3x", lambda x, r: x + 3 * np.std(x) * (np.sin(2 * np.pi * 0.15 * _T[:len(x)] + r.uniform(0, 6.3))
                                                         + 0.7 * np.sin(2 * np.pi * 0.3 * _T[:len(x)] + r.uniform(0, 6.3)))),
    ("mains", "60 Hz", lambda x, r: x + 2 * np.std(x) * np.sin(2 * np.pi * 60 * _T[:len(x)] + r.uniform(0, 6.3))),
    ("muscle", "0 dB", lambda x, r: add_snr(x, _band_noise(r, 20, 120, len(x)), 0)),
    ("motion", "1/s", motion),
    ("clipping", "30%", lambda x, r: clip(x)),
    ("lowpass", "4 Hz", lambda x, r: lowpass4(x)),
    ("dropout", "2x0.8 s", dropout),
]


def truth_from_beats(times, start, stop):
    b = times[(times >= start) & (times < stop)]
    if len(b) < 4:
        return None
    return 60.0 / float(np.mean(np.diff(b)))


def synthetic_windows():
    """Yields (window, truth, meta)."""
    for variant in VARIANTS:
        for bpm in RATES:
            for family, sev, fn in CONDITIONS:
                for seed in SEEDS:
                    rng = rng_for("ts", variant, bpm, family, sev, seed)
                    x, times = make_signal(variant, bpm, rng)
                    x = fn(x, rng)
                    s = int(rng.uniform(1.0, SYN_DUR - WIN - 1.0) * FS)
                    truth = truth_from_beats(times, s / FS, (s + N) / FS)
                    yield x[s:s + N], truth, {"signal": variant, "rate": bpm, "family": family, "severity": sev,
                                              "seed": seed, "group": f"{variant}-{bpm:g}"}


def synthetic_nulls():
    for s in range(NULLS_PER_KIND * 2):
        yield np.random.default_rng(70_000 + s).normal(size=N), {"kind": "white"}
    for s in range(NULLS_PER_KIND):
        b = np.cumsum(np.random.default_rng(80_000 + s).normal(size=N))
        yield b - b.mean(), {"kind": "brown"}


# ──────────────────────────────────────────────────────────────────────────────
# MIT-BIH windows with real NSTDB noise
# ──────────────────────────────────────────────────────────────────────────────

def nstdb_noise():
    return {k: read_record(NSTDB, k)[1][:, 0] for k in ("em", "ma", "bw")}


def mitdb_windows():
    with open(os.path.join(MITDB, "RECORDS")) as f:
        records = f.read().split()
    noise = nstdb_noise()
    for rec in records:
        fs, x = read_record(MITDB, rec)
        assert fs == FS
        sig = x[:, 0]
        samples, codes = read_annotations(os.path.join(MITDB, f"{rec}.atr"))
        beats = samples[np.isin(codes, list(BEAT_CODES))]
        bcodes = codes[np.isin(codes, list(BEAT_CODES))]
        flutter = samples[np.isin(codes, list(FLUTTER_CODES))]
        starts = []
        for s in range(0, len(sig) - N, N):
            if np.any((flutter >= s - N) & (flutter < s + 2 * N)):
                continue
            if truth_from_beats(beats / FS, s / FS, (s + N) / FS) is not None:
                starts.append(s)
        r = rng_for("mitdb", rec)
        starts = sorted(r.choice(starts, min(WINDOWS_PER_RECORD, len(starts)), replace=False))
        for s in starts:
            w = sig[s:s + N]
            m = (beats >= s) & (beats < s + N)
            rr = np.diff(beats[m]) / FS
            meta = {"signal": "ecg", "record": rec, "group": rec, "start": int(s),
                    "rr_cv": float(np.std(rr) / np.mean(rr)),
                    "ectopic": bool(np.any(~np.isin(bcodes[m], list(NORMAL_CODES))))}
            truth = truth_from_beats(beats / FS, s / FS, (s + N) / FS)
            yield w, truth, {**meta, "family": "clean", "severity": "-"}
            for kind, nz in noise.items():
                for snr in REAL_SNRS:
                    rn = rng_for("nst", rec, s, kind, snr)
                    o = int(rn.integers(0, len(nz) - N))
                    seg = nz[o:o + N] - np.mean(nz[o:o + N])
                    yield add_snr(w, seg, snr), truth, {**meta, "family": kind, "severity": f"{snr} dB"}


def mitdb_nulls():
    noise = nstdb_noise()
    for kind, nz in noise.items():
        r = rng_for("null", kind)
        for o in r.choice(len(nz) - N, NULLS_PER_KIND, replace=False):
            yield nz[o:o + N], {"kind": kind}


# ──────────────────────────────────────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────────────────────────────────────

def read_row(analyzer, x, truth, meta):
    a = analyzer.analyze(x, {"fs": FS})
    if a.empty:
        return {**meta, "truth": truth, "empty": True, "readings": {n: (None, 0.0) for n in REPS}}
    rep = a.report
    return {**meta, "truth": truth, "empty": False,
            "readings": {n: (rep.readings[n].estimate, rep.readings[n].confidence) for n in REPS},
            "raw": {n: rep.readings[n].details.get("raw") for n in REPS}}


def collect(name, refresh=False):
    path = os.path.join(CACHE, f"exp053_{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    an = make_timeseries_analyzer()
    if name == "synthetic":
        rows = [read_row(an, x, tr, m) for x, tr, m in synthetic_windows()]
        nulls = [read_row(an, x, None, m) for x, m in synthetic_nulls()]
    else:
        rows = [read_row(an, x, tr, m) for x, tr, m in mitdb_windows()]
        nulls = [read_row(an, x, None, m) for x, m in mitdb_nulls()]
    rows = [r for r in rows if r["truth"] is not None and not r["empty"]]
    with open(path, "wb") as f:
        pickle.dump((rows, nulls), f)
    print(f"  {name}: {len(rows)} windows, {len(nulls)} nulls in {time.time() - t:.0f} s")
    return rows, nulls


def calibrate():
    """lo = strongest raw score over 60 white-noise windows, hi = median raw score over
    clean synthetic windows the lens reads right."""
    lenses = {n: cls() for n, cls in LENSES.items()}
    noise = {n: [] for n in lenses}
    clean = {n: [] for n in lenses}
    for s in range(60):
        x = np.random.default_rng(60_000 + s).normal(size=N)
        for n, lens in lenses.items():
            noise[n].append(lens.rate(x, FS)[1])
    for x, tr, m in synthetic_windows():
        if m["family"] != "clean":
            continue
        for n, lens in lenses.items():
            est, raw = lens.rate(x, FS)
            if est is not None and not is_gross(est, tr):
                clean[n].append(raw)
    # The 5th-percentile rule of the audio / image lenses fails here: the clean set holds
    # signals a lens reads right but weakly (many-harmonic ECG for the wavelet bank), and
    # their 5th percentile falls below the noise maximum. hi = the median instead.
    out = {n: (round(float(np.max(noise[n])), 3), round(float(np.median(clean[n])), 3)) for n in lenses}
    for n, (lo, hi) in out.items():
        print(f'    "{n}": ({lo}, {hi}),')
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def comparator(k, family=None, threshold=VOTING_THRESHOLD):
    return RateComparator(min_agreeing=k, family=family, detector=ConfidenceThresholdDetector(threshold))


def readings(row):
    return {n: Reading(n, e, c) for n, (e, c) in row["readings"].items()}


def wrong(est, row):
    return est is None or is_gross(est, row["truth"])


def features(rep, n):
    others = [o for o in rep.readings if o != n]
    mine = [p for p in rep.pairs if n in (p.a, p.b)]
    return [rep.divergence[n], rep.overall, rep.readings[n].confidence,
            float(np.mean([rep.readings[o].confidence for o in others])),
            min(p.distance for p in mine), float(any(p.relation in ("double", "half") for p in mine))]


def table_examples(rows):
    comp = comparator(1)
    ex = []
    for r in rows:
        rd = readings(r)
        rep = comp.compare(rd)
        trusted = [n for n in rd if not rep.assessments[n].blind]
        ex += [(g, not wrong(comp.fuse([rd[n] for n in g]), r)) for g in comp.agreeing_groups(trusted, rep.pairs)]
    return ex


def lift(a, b):
    e = a.mean() * b.mean()
    return float((a & b).mean() / e) if e > 0 else float("nan")


def phi(a, b):
    pa, pb = a.mean(), b.mean()
    d = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
    return float(((a & b).mean() - pa * pb) / d) if d > 0 else float("nan")


def within(fn, a, b, cond):
    vals, wts = [], []
    for c in np.unique(cond):
        m = cond == c
        v = fn(a[m], b[m])
        if v == v:
            vals.append(v)
            wts.append(m.sum())
    return float(np.average(vals, weights=wts)) if vals else float("nan")


def analyze(rows, nulls, transfer_table=None):
    groups = np.array([str(r["group"]) for r in rows])
    cond = np.array([f"{r['family']} {r['severity']}" for r in rows])
    fam = np.array([r["family"] for r in rows])
    sig = np.array([r["signal"] for r in rows])
    W = {n: np.array([wrong(r["readings"][n][0], r) for r in rows]) for n in REPS}
    s = {"n": len(rows), "n_groups": int(len(np.unique(groups))), "n_null": len(nulls),
         "gross": {n: float(w.mean()) for n, w in W.items()},
         "atlas": {n: {c: float(w[cond == c].mean()) for c in dict.fromkeys(cond)} for n, w in W.items()},
         "atlas_by_signal": {n: {g: float(w[sig == g].mean()) for g in dict.fromkeys(sig)} for n, w in W.items()}}
    if "rr_cv" in rows[0]:
        irregular = np.array([r["rr_cv"] > 0.1 or r["ectopic"] for r in rows])
        s["irregular_share"] = float(irregular.mean())
        s["atlas_by_rhythm"] = {n: {"regular": float(w[~irregular].mean()), "irregular": float(w[irregular].mean())}
                                for n, w in W.items()}

    # 2. prediction
    q2 = comparator(2)
    reps = [q2.compare(readings(r)) for r in rows]
    s["prediction"] = {}
    for n in REPS:
        X = np.array([features(rep, n) for rep in reps])
        y = W[n].astype(float)
        cv = e42.grouped_cv_scores(X, y, groups) if 0 < y.sum() < len(y) else np.full(len(y), np.nan)
        s["prediction"][n] = {"self_conf": auc(-X[:, 2], W[n]), "divergence": auc(X[:, 0], W[n]),
                              "disagree_model": auc(cv, W[n]), "base_rate": float(y.mean())}

    # 3. answer
    legacy = comparator(None)
    est = {"legacy": [legacy.compare(readings(r)).consensus for r in rows],
           "q2": [rep.consensus for rep in reps], "table_q1": [None] * len(rows)}
    uniq = np.unique(groups)
    np.random.default_rng(0).shuffle(uniq)
    for fold in np.array_split(uniq, FOLDS):
        test = np.isin(groups, fold)
        comp = comparator(1, AgreementTable.fit(table_examples([r for r, t in zip(rows, test) if not t])))
        for i in np.flatnonzero(test):
            est["table_q1"][i] = comp.compare(readings(rows[i])).consensus
    if transfer_table is not None:
        comp = comparator(1, transfer_table)
        est["table_q1_from_synthetic"] = [comp.compare(readings(r)).consensus for r in rows]
    bad = {m: np.array([wrong(x, r) for x, r in zip(e, rows)]) for m, e in est.items()}
    ans = {m: np.array([x is not None for x in e]) for m, e in est.items()}
    best = min(REPS, key=lambda n: W[n].mean())
    s["answer"] = {m: {"gross_all": float(b.mean()), "answer": float(ans[m].mean()),
                       "gross_answer": float(b[ans[m]].mean()) if ans[m].any() else float("nan")}
                   for m, b in bad.items()}
    s["answer"]["best_lens"] = {"name": best, "gross_all": float(W[best].mean())}
    s["answer"]["hindsight"] = {"gross_all": float(np.all([W[n] for n in REPS], axis=0).mean())}
    for m in bad:
        s["answer"][f"{m}_minus_best_lens"] = e45.group_bootstrap(groups, bad[m], W[best])
    a = ans["q2"]
    s["answer"]["q2_answered_minus_best_lens_same_frames"] = e45.group_bootstrap(groups[a], bad["q2"][a], W[best][a])
    s["answer"]["by_family"] = {f: {**{m: float(bad[m][fam == f].mean()) for m in bad},
                                    "best_lens": float(W[best][fam == f].mean())} for f in dict.fromkeys(fam)}

    # 4. families
    s["co_failure"] = {f"{a}+{b}": {"lift": lift(W[a], W[b]), "phi": phi(W[a], W[b]),
                                    "lift_within_condition": within(lift, W[a], W[b], cond),
                                    "phi_within_condition": within(phi, W[a], W[b], cond)} for a, b in PAIRS}
    labels = np.array(["+".join(n for n in REPS if W[n][i]) or "none" for i in range(len(rows))])
    s["failure_sets"] = {k: {"n": int(np.sum(labels == k)),
                             "top_conditions": [f"{c} ({int(np.sum((labels == k) & (cond == c)))})" for c in
                                                sorted(set(cond[labels == k]), key=lambda c: -np.sum((labels == k) & (cond == c)))[:3]]}
                         for k in sorted(set(labels), key=lambda k: -np.sum(labels == k))[:10]}

    # 5. nulls
    s["nulls"] = {"lens_confident": {n: float(np.mean([not r["empty"] and r["readings"][n][1] >= VOTING_THRESHOLD
                                                       for r in nulls])) for n in REPS},
                  "legacy": float(np.mean([not r["empty"] and legacy.compare(readings(r)).consensus is not None
                                           for r in nulls])),
                  "q2": float(np.mean([not r["empty"] and q2.compare(readings(r)).consensus is not None
                                       for r in nulls]))}
    s["_plot"] = {"W": {n: w.tolist() for n, w in W.items()}}
    return s


def core_changes():
    """Lines changed in src/core since the commit before this experiment (expected: none)."""
    try:
        out = subprocess.run(["git", "diff", "--stat", "5f2ed72", "--", "src/core"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "src/core"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"diff_since_exp051": out or "none", "uncommitted": status or "none"}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def main():
    if "--calibrate" in sys.argv:
        calibrate()
        return
    refresh = "--refresh" in sys.argv
    syn_rows, syn_nulls = collect("synthetic", refresh)
    mit_rows, mit_nulls = collect("mitdb", refresh)
    transfer = AgreementTable.fit(table_examples(syn_rows))
    s = {"lenses": list(REPS), "window_s": WIN, "fs": FS, "gross_ratio": 1.10,
         "core_changes": core_changes(),
         "synthetic": analyze(syn_rows, syn_nulls),
         "mitdb": analyze(mit_rows, mit_nulls, transfer)}
    plot_data = {k: s[k].pop("_plot") for k in ("synthetic", "mitdb")}
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    report(s)
    plot(s, plot_data)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    print(f"core changes: {s['core_changes']}")
    for name in ("synthetic", "mitdb"):
        d = s[name]
        print(f"\n=== {name}: {d['n']} windows in {d['n_groups']} groups, {d['n_null']} nulls")
        print("  gross: " + ", ".join(f"{n} {v:.1%}" for n, v in d["gross"].items()))
        conds = list(next(iter(d["atlas"].values())))
        print("  atlas: " + " | ".join(conds))
        for n in REPS:
            print(f"    {n:8s} " + " ".join(f"{d['atlas'][n][c]:6.1%}" for c in conds))
        print("  by signal: " + "; ".join(f"{n}: " + ", ".join(f"{g} {v:.0%}" for g, v in d["atlas_by_signal"][n].items())
                                          for n in REPS))
        if "atlas_by_rhythm" in d:
            print(f"  irregular share {d['irregular_share']:.0%}: " + "; ".join(
                f"{n} {v['regular']:.1%} / {v['irregular']:.1%}" for n, v in d["atlas_by_rhythm"].items()))
        print("  prediction AUC (self conf / divergence / disagreement model):")
        for n, p in d["prediction"].items():
            print(f"    {n:8s} {p['self_conf']:.3f} / {p['divergence']:.3f} / {p['disagree_model']:.3f}  (base rate {p['base_rate']:.1%})")
        a = d["answer"]
        for m in [k for k in a if isinstance(a[k], dict) and "answer" in a[k]]:
            print(f"  {m:24s} {a[m]['gross_all']:.1%} (answers {a[m]['answer']:.0%}, wrong when answering {a[m]['gross_answer']:.1%})")
        print(f"  best lens {a['best_lens']['name']} {a['best_lens']['gross_all']:.1%}; hindsight {a['hindsight']['gross_all']:.1%}")
        for k, v in a.items():
            if k.endswith("best_lens") and "diff" in v or k.startswith("q2_answered"):
                print(f"    {k}: {v['diff']:+.2%} [{v['ci95'][0]:+.2%}, {v['ci95'][1]:+.2%}]")
        print("  co-failure lift (within condition) / phi:")
        for k, v in d["co_failure"].items():
            print(f"    {k:14s} {v['lift']:.2f} ({v['lift_within_condition']:.2f}) / {v['phi']:.2f}")
        print("  failure sets: " + "; ".join(f"{k} {v['n']} {v['top_conditions']}" for k, v in list(d["failure_sets"].items())[:6]))
        print(f"  nulls: lens confident {d['nulls']['lens_confident']}, legacy {d['nulls']['legacy']:.0%}, q2 {d['nulls']['q2']:.0%}")


def plot(s, pdata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for row, name in enumerate(("synthetic", "mitdb")):
        d = s[name]
        conds = list(next(iter(d["atlas"].values())))
        A = np.array([[d["atlas"][n][c] for c in conds] for n in REPS])
        ax = axes[row, 0]
        ax.imshow(A, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(REPS)))
        ax.set_yticklabels(REPS)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.0%}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{name}: gross error (>10%) by condition")

        ax = axes[row, 1]
        x = np.arange(len(REPS))
        for k, (key, lab) in enumerate((("self_conf", "own confidence"), ("divergence", "divergence"),
                                        ("disagree_model", "disagreement model (CV)"))):
            ax.bar(x + (k - 1) * 0.27, [d["prediction"][n][key] for n in REPS], 0.27, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(REPS)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_ylabel("ROC AUC: is this lens wrong?")
        ax.set_title(f"{name}: does disagreement predict a blind spot?")
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        a = d["answer"]
        methods = [m for m in ("legacy", "q2", "table_q1", "table_q1_from_synthetic") if m in a]
        vals = [a[m]["gross_all"] for m in methods] + [d["gross"][n] for n in REPS] + [a["hindsight"]["gross_all"]]
        labels = [f"{m}\n(answers {a[m]['answer']:.0%})" for m in methods] + list(REPS) + ["hindsight"]
        colors = ["C0"] * len(methods) + ["C7"] * len(REPS) + ["C2"]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("gross error (abstain = wrong)")
        ax.set_title(f"{name}: fused answer vs single lenses")
    fig.suptitle("Exp 053: a third medium on the unchanged core: rate of periodic sensor signals "
                 "(synthetic; MIT-BIH ECG with NSTDB noise)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    main()
