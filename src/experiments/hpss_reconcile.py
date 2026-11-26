"""
HPSS reconciliation: Exp 035 vs Exp 041 (issue #8)
===================================================
Exp 035 reports the framework making HPSS worse (-1.51 dB SDR); Exp 041 reports
a gain (+0.0106 ΔP, i.e. +0.32 dB). The two experiments differ in three ways at
once: the signal, the baseline, and what the framework does. This script crosses
them so each difference can be seen on its own.

Signals
  A  Exp 035: a 200→600 Hz sweep (3 harmonics) + 5 sparse clicks
  B  Exp 041: a steady 300 Hz tone (6 harmonics) + clicks every 70 ms + noise

Methods (harmonic estimate)
  fitz15      Exp 035 baseline: scipy STFT 1024/512, median kernels 15, linear mask
  acf_gate    Exp 035 framework: fitz15 mask × the engine's per-frame ACF safety
  lib31       Exp 041 baseline: librosa.effects.hpss(kernel_size=31)
  region      Exp 041 framework: per-frame kernels chosen by the engine's region
  fixed81     control: the kernels 'region' uses for noise_collapse (H 81, P 15),
              applied to every frame with no engine at all

Metric: SDR (dB) of the harmonic estimate against the clean harmonic component.
Signal B is evaluated on 5 noise seeds; signal A has one fixed seed (as in Exp 035).
"""

import os
import sys
import warnings

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.framework.engine import RepresentationIntelligenceEngine  # noqa: E402

warnings.filterwarnings("ignore")
SR = 16000


def norm01(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return float(10 * np.log10((np.mean(ref ** 2) + 1e-12) / (np.mean((ref - est) ** 2) + 1e-12)))


# ── signals ──────────────────────────────────────────────────────────────────

def signal_a(seed=42):
    """Exp 035 case A."""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    phase = 2 * np.pi * np.cumsum(200 + 400 * (t / 2.0)) / SR
    harmonic = norm01(np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)) * 0.7
    perc = np.zeros(n)
    rng = np.random.default_rng(seed)
    for pos in [int(p * SR) for p in (0.2, 0.6, 1.0, 1.4, 1.8)]:
        clen = int(0.05 * SR)
        perc[pos:pos + clen] += rng.normal(0, 1, clen) * np.exp(-150 * np.arange(clen) / SR)
    perc = norm01(perc) * 0.5
    return harmonic + perc, harmonic


def signal_b(seed):
    """Exp 041 algorithm 4 (same construction; Exp 041 itself shares one rng across algorithms)."""
    rng = np.random.default_rng(seed)
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    freqs = [300, 600, 900, 1200, 1500, 1800]
    weights = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18]
    harmonic = norm01(sum(w * np.sin(2 * np.pi * f * t) for f, w in zip(freqs, weights))) * 0.45
    perc = np.zeros(n)
    for ct in np.arange(0.05, 2.5 - 0.05, 0.07):
        idx, clen = int(ct * SR), int(0.030 * SR)
        if idx + clen < n:
            perc[idx:idx + clen] += rng.normal(0, 1.0, clen) * np.exp(-180 * np.arange(clen) / SR)
    perc = norm01(perc) * 0.65
    mix = harmonic + perc + rng.normal(0, 0.03, n)
    scale = np.max(np.abs(mix)) + 1e-12
    # Exp 041 normalises the mix but scores against the unscaled harmonic; keep that.
    return mix / scale, harmonic


# ── methods ──────────────────────────────────────────────────────────────────

def _frames(x, hop, n_fft, count):
    for k in range(count):
        f = x[k * hop:k * hop + n_fft]
        yield f if len(f) == n_fft else np.pad(f, (0, n_fft - len(f)))


def fitz15(mix, engine=None, gate=False):
    n = len(mix)
    _, _, Z = scipy.signal.stft(mix, fs=SR, nperseg=1024, noverlap=512)
    mag, ph = np.abs(Z), np.angle(Z)
    H = scipy.ndimage.median_filter(mag, size=(1, 15))
    P = scipy.ndimage.median_filter(mag, size=(15, 1))
    mask = H / (H + P + 1e-12)
    if gate:
        w = np.array([engine.analyze(f, SR).assumptions["acf"]
                      for f in _frames(mix, 512, 1024, mag.shape[1])])
        mask = mask * w[np.newaxis, :]
    _, h = scipy.signal.istft(mag * mask * np.exp(1j * ph), fs=SR, nperseg=1024, noverlap=512)
    return h[:n]


def acf_gate(mix, engine):
    return fitz15(mix, engine, gate=True)


def lib31(mix, engine=None):
    h, _ = librosa.effects.hpss(mix, kernel_size=31)
    return h[:len(mix)]


def region(mix, engine):
    n, n_fft, hop = len(mix), 2048, 512
    D = librosa.stft(mix, n_fft=n_fft, hop_length=hop)
    mag, ph = np.abs(D), np.angle(D)
    mf = lambda size: scipy.ndimage.median_filter(mag, size=size, mode="reflect")  # noqa: E731
    H_def, P_def, H_81, P_15, H_17 = mf((1, 31)), mf((31, 1)), mf((1, 81)), mf((15, 1)), mf((1, 17))
    H, P = H_def.copy(), P_def.copy()
    regions = []
    for k, f in enumerate(_frames(mix, hop, n_fft, mag.shape[1])):
        r = engine.analyze(f, SR).region
        regions.append(r)
        if r == "noise_collapse":
            H[:, k], P[:, k] = H_81[:, k], P_15[:, k]
        elif r == "transient_overloaded":
            H[:, k] = H_17[:, k]
    mask = H ** 2 / (H ** 2 + P ** 2 + 1e-12)
    return librosa.istft(mag * mask * np.exp(1j * ph), hop_length=hop, length=n), regions


def fixed81(mix, engine=None):
    n, n_fft, hop = len(mix), 2048, 512
    D = librosa.stft(mix, n_fft=n_fft, hop_length=hop)
    mag, ph = np.abs(D), np.angle(D)
    H = scipy.ndimage.median_filter(mag, size=(1, 81), mode="reflect")
    P = scipy.ndimage.median_filter(mag, size=(15, 1), mode="reflect")
    mask = H ** 2 / (H ** 2 + P ** 2 + 1e-12)
    return librosa.istft(mag * mask * np.exp(1j * ph), hop_length=hop, length=n)


def run():
    engine = RepresentationIntelligenceEngine()
    cases = [("A (Exp 035)", [signal_a()]), ("B (Exp 041)", [signal_b(s) for s in range(5)])]
    print(f"{'signal':14s} {'fitz15':>8s} {'acf_gate':>9s} {'lib31':>8s} {'region':>8s} {'fixed81':>8s}   regions used by 'region'")
    for name, sigs in cases:
        rows, used = [], {}
        for mix, harm in sigs:
            h_reg, regs = region(mix, engine)
            for r in regs:
                used[r] = used.get(r, 0) + 1
            rows.append([sdr(harm, fitz15(mix)), sdr(harm, acf_gate(mix, engine)),
                         sdr(harm, lib31(mix)), sdr(harm, h_reg), sdr(harm, fixed81(mix))])
        m = np.mean(rows, axis=0)
        total = sum(used.values())
        share = ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(used.items(), key=lambda kv: -kv[1]))
        print(f"{name:14s} {m[0]:8.2f} {m[1]:9.2f} {m[2]:8.2f} {m[3]:8.2f} {m[4]:8.2f}   {share}")
        print(f"{'':14s} acf_gate − fitz15 = {m[1] - m[0]:+.2f} dB,  region − lib31 = {m[3] - m[2]:+.2f} dB,  region − fixed81 = {m[3] - m[4]:+.2f} dB")


if __name__ == "__main__":
    run()
