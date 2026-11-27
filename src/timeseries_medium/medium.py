"""Sensor windows, four rate lenses and the rate comparator (issue #27)."""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

try:  # installed package: representation_core
    from representation_core import (
        Analyzer,
        ConfidenceThresholdDetector,
        Medium,
        PairwiseComparator,
        Reading,
        Representation,
    )
except ImportError:  # plain checkout: src/core
    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _src not in sys.path:
        sys.path.append(_src)
    from core import (  # noqa: E402
        Analyzer,
        ConfidenceThresholdDetector,
        Medium,
        PairwiseComparator,
        Reading,
        Representation,
    )

MIN_BPM = 30.0
MAX_BPM = 240.0
MIN_SECONDS = 4.0          # at least two periods of the slowest rate
FLAT_STD = 1e-9
AGREE_RATIO = 1.10         # within 10%: "agree", also the gross-error threshold in Exp 053
VOTING_THRESHOLD = 0.5
_AGREE_OCT = float(np.log2(AGREE_RATIO))


def rate_error(est: float | None, truth: float) -> float:
    """|log2(est / truth)| (octaves); inf when there is no estimate."""
    if est is None or est <= 0 or truth <= 0:
        return float("inf")
    return float(abs(np.log2(est / truth)))


def is_gross(est: float | None, truth: float) -> bool:
    return rate_error(est, truth) > _AGREE_OCT


def _scale(raw: float, lo: float, hi: float) -> float:
    """Raw score → 0..1: ``lo`` (strongest of 60 white-noise windows) → 0,
    ``hi`` (median over clean synthetic windows read right) → 1. See CALIBRATION."""
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


def _parabolic(y: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(y) - 1:
        return 0.0
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def _spectrum(x: np.ndarray, fs: float, pad: int = 8):
    """Power spectrum of the linearly detrended, Hann-windowed window."""
    g = detrend(x) * np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(len(x) * pad)))
    P = np.abs(np.fft.rfft(g, nfft)) ** 2
    return P, np.fft.rfftfreq(nfft, 1.0 / fs)


# ──────────────────────────────────────────────────────────────────────────────
# MEDIUM
# ──────────────────────────────────────────────────────────────────────────────

class SensorWindow(Medium):
    """A 1-D window of a periodic sensor signal (ECG, PPG, accelerometer magnitude…).
    Context: ``fs`` (sample rate in Hz, required)."""

    name = "timeseries"

    def validate(self, signal: Any, context: Mapping[str, Any]) -> np.ndarray:
        x = np.asarray(signal, dtype=float).flatten()
        if "fs" not in context:
            raise ValueError("Time-series context needs a sample rate: context['fs'].")
        if x.size < MIN_SECONDS * context["fs"]:
            raise ValueError(f"Window must be at least {MIN_SECONDS:g} s long, got {x.size / context['fs']:.2f} s.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Window contains NaN or inf.")
        return x

    def is_empty(self, signal: np.ndarray, context: Mapping[str, Any]) -> bool:
        return bool(np.std(signal) < FLAT_STD)


# ──────────────────────────────────────────────────────────────────────────────
# LENSES (each returns a rate in beats per minute)
# ──────────────────────────────────────────────────────────────────────────────

class _RateLens(Representation):
    medium = "timeseries"
    contrast = (0.0, 1.0)  # (lo, hi), calibrated per lens

    def read(self, signal: np.ndarray, context: Mapping[str, Any]) -> Reading:
        bpm, raw = self.rate(np.asarray(signal, float), float(context["fs"]))
        if bpm is None or not np.isfinite(bpm) or bpm <= 0:
            return Reading(self.name, None, 0.0, {"raw": raw})
        return Reading(self.name, float(bpm), _scale(raw, *self.contrast), {"raw": raw})

    def rate(self, x: np.ndarray, fs: float) -> tuple[float | None, float]:
        """Return (rate in bpm or None, raw confidence score)."""
        raise NotImplementedError


class FftRate(_RateLens):
    """Strongest line of the power spectrum between 30 and 240 per minute.
    Raw confidence: log10 of the peak over the band's median power (prominence)."""

    name = "fft"
    assumptions = ("the fundamental is the strongest spectral line in the rate band "
                   "(a strong 2nd harmonic or a slow drift wins otherwise)",
                   "the rate is stable over the window")

    def rate(self, x, fs):
        P, f = _spectrum(x, fs)
        band = (f >= MIN_BPM / 60) & (f <= MAX_BPM / 60)
        Pb = P[band]
        total = Pb.sum()
        if total <= 1e-20:
            return None, 0.0
        k = int(np.argmax(Pb))
        df = f[1] - f[0]
        f0 = f[band][k] + _parabolic(Pb, k) * df
        prominence = np.log10(Pb[k] / max(np.median(Pb), 1e-30))
        return 60.0 * f0, float(prominence)


class AcfRate(_RateLens):
    """Highest autocorrelation peak at lags of 0.25-2 s."""

    name = "acf"
    assumptions = ("the waveform repeats from cycle to cycle",
                   "at least a few cycles fit in the window (4 at 30/min in 8 s)")

    def rate(self, x, fs):
        g = detrend(x)
        n = len(g)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        r = np.fft.irfft(np.abs(np.fft.rfft(g, nfft)) ** 2)[:n]
        if r[0] <= 1e-20:
            return None, 0.0
        r = r / r[0]          # biased: long lags are damped (an unbiased ACF favours 2x the period)
        lo, hi = int(fs * 60 / MAX_BPM), min(int(fs * 60 / MIN_BPM), n - 2)
        seg = r[lo:hi + 1]
        peaks, _ = find_peaks(seg)
        if len(peaks) == 0:
            return None, 0.0
        k = int(peaks[np.argmax(seg[peaks])])
        lag = (lo + k + _parabolic(seg, k)) / fs
        return 60.0 / lag, float(np.clip(seg[k], 0.0, 1.0))


class WaveletRate(_RateLens):
    """Time-averaged Morlet scalogram (constant Q, 5 cycles) at 64 rates, with the
    first three harmonics summed (weights 1, 1/2, 1/3) to favour the fundamental.
    Raw confidence: contrast of the salience, (max - min) / (max + min).

    The time average of |Morlet response|^2 equals the power spectrum weighted by
    the wavelet's Gaussian frequency response, so it is computed that way."""

    name = "wavelet"
    assumptions = ("energy at the rate and its first harmonics (a constant-Q bank)",
                   "the rate is stable over the long low-frequency wavelets (several seconds at 30/min)")
    cycles = 5.0
    n_rates = 64
    weights = (1.0, 0.5, 1.0 / 3.0)

    def rate(self, x, fs):
        P, f = _spectrum(x, fs, pad=2)
        if P.sum() <= 1e-20:
            return None, 0.0
        rates = np.geomspace(MIN_BPM, MAX_BPM, self.n_rates) / 60.0
        sal = np.zeros(self.n_rates)
        for h, w in enumerate(self.weights, start=1):
            fc = h * rates
            sig = fc / self.cycles          # frequency std of an n-cycle Morlet (time std n / 2πf)
            G = np.exp(-0.5 * ((f[None, :] - fc[:, None]) / sig[:, None]) ** 2)
            band = G @ P / np.maximum(G.sum(1), 1e-12)
            sal += w * band
        k = int(np.argmax(sal))
        contrast = (sal.max() - sal.min()) / (sal.max() + sal.min())
        # The wide bands are biased by the spectral tilt: refine on the raw spectrum
        # inside the chosen wavelet's band (± one frequency std).
        fc = rates[k]
        idx = np.flatnonzero((f >= fc * (1 - 1 / self.cycles)) & (f <= fc * (1 + 1 / self.cycles)))
        j = idx[int(np.argmax(P[idx]))]
        f0 = np.clip(f[j] + _parabolic(P, j) * (f[1] - f[0]), MIN_BPM / 60, MAX_BPM / 60)
        return 60.0 * float(f0), float(contrast)


class PeakRate(_RateLens):
    """Time-domain event counting (Pan-Tompkins style): band-pass 5-15 Hz,
    squared derivative, 150 ms moving integration, peaks above 35% of the 99th
    percentile with a 250 ms refractory time; rate = 60 / median interval."""

    name = "peaks"
    assumptions = ("one sharp event (5-15 Hz content) per cycle, e.g. a QRS complex or a heel strike",
                   "events stand above the noise and other waves (a tall T wave is counted twice)",
                   "cycles are longer than the 250 ms refractory time")

    def rate(self, x, fs):
        if fs < 32:
            return None, 0.0
        sos = butter(2, [5.0, 15.0], btype="band", fs=fs, output="sos")
        y = np.gradient(sosfiltfilt(sos, x)) ** 2
        w = max(1, int(0.15 * fs))
        env = np.convolve(y, np.ones(w) / w, mode="same")
        top = np.percentile(env, 99)
        if top <= 1e-20:
            return None, 0.0
        pk, _ = find_peaks(env, height=0.35 * top, distance=int(0.25 * fs))
        if len(pk) < 3:
            return None, 0.0
        rr = np.diff(pk) / fs
        regular = 1.0 - min(float(np.std(rr) / np.mean(rr)), 1.0)
        return 60.0 / float(np.median(rr)), regular


# Calibrated with src/experiments/exp053_timeseries_medium.py --calibrate:
# lo = strongest raw score over 60 white-noise windows (8 s at 360 Hz),
# hi = median over clean synthetic windows the lens reads right (all signal types
# and rates). The 5th-percentile rule of the other media puts hi below lo here.
CALIBRATION = {
    "fft": (1.235, 2.454),
    "acf": (0.074, 0.706),
    "wavelet": (0.69, 0.792),
    "peaks": (0.779, 0.978),
}

LENSES = {cls.name: cls for cls in (FftRate, AcfRate, WaveletRate, PeakRate)}
for _name, _cls in LENSES.items():
    _cls.contrast = CALIBRATION[_name]
ALL_LENSES = tuple(LENSES)


# ──────────────────────────────────────────────────────────────────────────────
# COMPARATOR
# ──────────────────────────────────────────────────────────────────────────────

class RateComparator(PairwiseComparator):
    """Rate distance (in octaves) and fusion; everything else is the core's generic logic."""

    def estimate_distance(self, a: Reading, b: Reading):
        d = float(np.log2(a.estimate / b.estimate))
        ad = abs(d)
        if ad < _AGREE_OCT:
            relation = "agree"
        elif abs(ad - 1.0) < _AGREE_OCT:
            relation = "double" if d > 0 else "half"
        else:
            relation = "off"
        return min(ad, 1.0), relation, {"octaves": d}

    def fuse(self, readings: Sequence[Reading]) -> float:
        """Confidence-weighted median rate."""
        v = np.array([r.estimate for r in readings], float)
        w = np.array([r.confidence for r in readings], float)
        order = np.argsort(v)
        cum = np.cumsum(w[order])
        return float(v[order][np.searchsorted(cum, 0.5 * cum[-1])])


def make_timeseries_analyzer(lenses: Sequence[str] = ALL_LENSES,
                             threshold: float = VOTING_THRESHOLD, min_agreeing: int | None = 2) -> Analyzer:
    """A core Analyzer for sensor rate with the given lenses (names from ``ALL_LENSES``)."""
    unknown = [n for n in lenses if n not in LENSES]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}. Available: {list(LENSES)}")
    detector = ConfidenceThresholdDetector(threshold)
    return Analyzer(SensorWindow(), [LENSES[n]() for n in lenses],
                    RateComparator(detector=detector, min_agreeing=min_agreeing), detector=detector)
