"""
Cross-Representation Disagreement
=================================
Runs several representations on the same audio frame, expresses each one's
output in a shared, comparable form, and measures where they disagree.

This is the first building block of the project goal (see docs/VISION.md):
*disagreement between representations is information*. When the lenses agree,
that is evidence. When they disagree, the pattern of disagreement points to a
blind spot and to the representation that is most likely blind.

Shared comparable form
----------------------
Every representation is reduced to a :class:`RepresentationReading`:

- ``f0``          fundamental-frequency estimate in Hz (0.0 when there is none)
- ``confidence``  how strongly the representation supports that estimate, 0..1
- ``voiced``      ``confidence >= VOICING_THRESHOLD``

Representations
---------------
- ``stft``      Hann-windowed magnitude spectrum + weighted harmonic summation.
                Confidence = share of spectral energy on the harmonic comb of
                the chosen f0, above what a flat (noise) spectrum would give.
- ``acf``       Unbiased, normalised autocorrelation. First strong peak in the
                lag range (avoids octave-down errors). Confidence = peak value.
- ``cepstrum``  Real cepstrum of the windowed frame. Peak in the quefrency
                range. Confidence = robust z-score of that peak.

Disagreement measures (per pair, all in 0..1 unless noted)
----------------------------------------------------------
- ``cents``            absolute pitch distance in cents (not bounded)
- ``octave_related``   distance is a whole number of octaves (±50 cents)
- ``pitch_term``       ``min(cents / 1200, 1)``
- ``confidence_gap``   ``|conf_a - conf_b|``
- ``voicing_conflict`` one says voiced, the other unvoiced
- ``score``            ``0.5 * pitch_term + 0.5 * confidence_gap``

Overall measures: ``overall`` (mean pair score), per-representation
``divergence`` (mean score of the pairs it is part of), ``outlier`` (the
representation that diverges most, when the margin is clear and at least one
representation is confident) and
``consensus_f0`` (confidence-weighted median of the voiced readings).

Only NumPy is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np


VOICING_THRESHOLD = 0.5
OUTLIER_MARGIN = 0.10
OCTAVE_TOLERANCE_CENTS = 50.0
SILENCE_RMS = 1e-8
DEFAULT_FMIN = 80.0
DEFAULT_FMAX = 1000.0

REPRESENTATIONS = ("stft", "acf", "cepstrum")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RepresentationReading:
    """One representation's view of a frame, in the shared comparable form."""

    name: str
    f0: float
    confidence: float

    @property
    def voiced(self) -> bool:
        return self.confidence >= VOICING_THRESHOLD

    def to_dict(self) -> dict:
        return {**asdict(self), "voiced": self.voiced}


@dataclass
class PairDisagreement:
    """How much two representations disagree about the same frame."""

    a: str
    b: str
    cents: float | None
    octave_related: bool
    pitch_term: float
    confidence_gap: float
    voicing_conflict: bool
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DisagreementReport:
    """Per-frame cross-representation disagreement."""

    readings: dict[str, RepresentationReading]
    pairs: list[PairDisagreement]
    overall: float
    divergence: dict[str, float]
    outlier: str | None
    consensus_f0: float
    silent: bool = False
    meta: dict = field(default_factory=dict)

    def pair(self, a: str, b: str) -> PairDisagreement:
        """Return the pair (a, b) in either order."""
        for p in self.pairs:
            if {p.a, p.b} == {a, b}:
                return p
        raise KeyError(f"No pair ({a}, {b})")

    def to_dict(self) -> dict:
        return {
            "readings": {k: r.to_dict() for k, r in self.readings.items()},
            "pairs": [p.to_dict() for p in self.pairs],
            "overall": self.overall,
            "divergence": dict(self.divergence),
            "outlier": self.outlier,
            "consensus_f0": self.consensus_f0,
            "silent": self.silent,
            "meta": dict(self.meta),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << max(0, int(np.ceil(np.log2(max(n, 1)))))


def _parabolic_offset(y_prev: float, y_peak: float, y_next: float) -> float:
    """Sub-sample offset of a peak from three neighbouring values (-0.5..0.5)."""
    denom = y_prev - 2.0 * y_peak + y_next
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_prev - y_next) / denom, -0.5, 0.5))


def _refine_peak(values: np.ndarray, idx: int) -> float:
    if 0 < idx < len(values) - 1:
        return idx + _parabolic_offset(values[idx - 1], values[idx], values[idx + 1])
    return float(idx)


def _empty(name: str) -> RepresentationReading:
    return RepresentationReading(name=name, f0=0.0, confidence=0.0)


# ---------------------------------------------------------------------------
# Readers: one per representation, all returning the shared form
# ---------------------------------------------------------------------------

def read_stft(frame: np.ndarray, sr: int,
              fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX) -> RepresentationReading:
    """Pitch + confidence from the magnitude spectrum (harmonic summation)."""
    n = len(frame)
    n_fft = max(4096, _next_pow2(n))
    mag = np.abs(np.fft.rfft(frame * np.hanning(n), n=n_fft))
    freqs = np.arange(len(mag)) * sr / n_fft
    f_lim = min(5000.0, 0.45 * sr)

    band = (freqs >= 0.5 * fmin) & (freqs <= f_lim)
    peak = mag[band].max() if band.any() else 0.0
    if peak <= 0.0:
        return _empty("stft")
    mag_n = mag / peak

    # Weighted harmonic summation on a 25-cent grid. Counting harmonics up to a
    # fixed frequency limit (instead of a fixed number) and weighting by 1/k
    # makes the true f0 beat both its octave above and below.
    candidates = fmin * 2.0 ** (np.arange(0, np.log2(fmax / fmin) * 48 + 1) / 48.0)
    salience = np.zeros(len(candidates))
    for i, f0 in enumerate(candidates):
        k = np.arange(1, int(f_lim // f0) + 1)
        if k.size == 0:
            continue
        salience[i] = np.sum(np.interp(k * f0, freqs, mag_n) / k)
    best = int(np.argmax(salience))
    # Refine on the log-frequency grid.
    f0 = float(fmin * 2.0 ** (_refine_peak(salience, best) / 48.0))

    # Confidence: energy on the harmonic comb vs. what a flat spectrum would put there.
    energy = mag ** 2
    tol = 2.0 * sr / n  # half-width of the Hann main lobe
    k = np.arange(1, int(f_lim // f0) + 1)
    comb = np.zeros_like(band)
    for h in k * f0:
        comb |= np.abs(freqs - h) <= tol
    comb &= band
    total = energy[band].sum()
    if total <= 0.0 or band.sum() == 0:
        return _empty("stft")
    frac = energy[comb].sum() / total
    expected = comb.sum() / band.sum()
    confidence = float(np.clip((frac - expected) / (1.0 - expected + 1e-12), 0.0, 1.0))
    return RepresentationReading("stft", f0, confidence)


def read_acf(frame: np.ndarray, sr: int,
             fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX) -> RepresentationReading:
    """Pitch + confidence from the normalised autocorrelation."""
    x = frame - np.mean(frame)
    n = len(x)
    spec = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(spec * np.conj(spec), n=2 * n)[:n]
    if r[0] <= 0.0:
        return _empty("acf")
    lags = np.arange(n)
    norm = (r * n / np.maximum(n - lags, 1)) / r[0]  # unbiased, r[0] == 1

    lag_min = max(2, int(np.floor(sr / fmax)))
    lag_max = min(int(np.ceil(sr / fmin)), n // 2)
    if lag_max <= lag_min + 2:
        return _empty("acf")

    seg = norm[lag_min:lag_max + 1]
    interior = np.arange(1, len(seg) - 1)
    is_max = (seg[interior] >= seg[interior - 1]) & (seg[interior] >= seg[interior + 1])
    maxima = interior[is_max]
    if maxima.size == 0:
        idx = int(np.argmax(seg))
    else:
        gmax = seg[maxima].max()
        # First peak close to the strongest one: avoids picking 2x the period.
        strong = maxima[seg[maxima] >= 0.9 * gmax] if gmax > 0 else maxima
        idx = int(strong[0])

    lag = lag_min + idx
    lag_refined = lag_min + _refine_peak(seg, idx)
    f0 = float(sr / lag_refined) if lag_refined > 0 else 0.0
    confidence = float(np.clip(norm[lag], 0.0, 1.0))
    return RepresentationReading("acf", f0, confidence)


def read_cepstrum(frame: np.ndarray, sr: int,
                  fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX) -> RepresentationReading:
    """Pitch + confidence from the real cepstrum."""
    n = len(frame)
    n_fft = max(4096, _next_pow2(n))
    mag = np.abs(np.fft.rfft(frame * np.hanning(n), n=n_fft))
    if mag.max() <= 0.0:
        return _empty("cepstrum")
    # Floor relative to the peak so empty bins do not dominate the log spectrum.
    log_mag = np.log(mag + 1e-6 * mag.max())
    ceps = np.fft.irfft(log_mag, n=n_fft)

    q_min = max(2, int(np.floor(sr / fmax)))
    q_max = min(int(np.ceil(sr / fmin)), n // 2)
    if q_max <= q_min + 2:
        return _empty("cepstrum")

    seg = ceps[q_min:q_max + 1]
    idx = int(np.argmax(seg))
    q = q_min + _refine_peak(seg, idx)
    f0 = float(sr / q) if q > 0 else 0.0

    med = np.median(seg)
    mad = 1.4826 * np.median(np.abs(seg - med)) + 1e-12
    z = (seg[idx] - med) / mad
    # z ~ 3 is what the maximum of pure-noise cepstrum values reaches; z >= 10 is a clear peak.
    confidence = float(np.clip((z - 3.0) / 7.0, 0.0, 1.0))
    return RepresentationReading("cepstrum", f0, confidence)


READERS = {
    "stft": read_stft,
    "acf": read_acf,
    "cepstrum": read_cepstrum,
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_pair(a: RepresentationReading, b: RepresentationReading) -> PairDisagreement:
    """Disagreement between two readings."""
    if a.f0 > 0.0 and b.f0 > 0.0:
        cents = float(1200.0 * abs(np.log2(a.f0 / b.f0)))
        nearest_octave = 1200.0 * round(cents / 1200.0)
        octave_related = cents >= 600.0 and abs(cents - nearest_octave) <= OCTAVE_TOLERANCE_CENTS
        pitch_term = float(min(cents / 1200.0, 1.0))
    elif a.f0 <= 0.0 and b.f0 <= 0.0:
        cents, octave_related, pitch_term = None, False, 0.0
    else:
        cents, octave_related, pitch_term = None, False, 1.0

    confidence_gap = float(abs(a.confidence - b.confidence))
    return PairDisagreement(
        a=a.name,
        b=b.name,
        cents=cents,
        octave_related=octave_related,
        pitch_term=pitch_term,
        confidence_gap=confidence_gap,
        voicing_conflict=a.voiced != b.voiced,
        score=0.5 * pitch_term + 0.5 * confidence_gap,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def compare_readings(readings: dict[str, RepresentationReading],
                     silent: bool = False) -> DisagreementReport:
    """Build a disagreement report from already-computed readings."""
    names = list(readings)
    pairs = [compare_pair(readings[a], readings[b]) for a, b in combinations(names, 2)]
    overall = float(np.mean([p.score for p in pairs])) if pairs else 0.0

    divergence = {}
    for name in names:
        scores = [p.score for p in pairs if name in (p.a, p.b)]
        divergence[name] = float(np.mean(scores)) if scores else 0.0

    # An outlier needs a reference: if no lens is confident (e.g. pure noise),
    # all of them are blind and none is singled out.
    outlier = None
    if len(names) >= 3 and any(r.voiced for r in readings.values()):
        ranked = sorted(divergence, key=divergence.get, reverse=True)
        if divergence[ranked[0]] - divergence[ranked[1]] >= OUTLIER_MARGIN:
            outlier = ranked[0]

    voiced = [r for r in readings.values() if r.voiced and r.f0 > 0.0]
    consensus_f0 = (
        _weighted_median(np.array([r.f0 for r in voiced]),
                         np.array([r.confidence for r in voiced]))
        if voiced else 0.0
    )

    return DisagreementReport(
        readings=readings,
        pairs=pairs,
        overall=overall,
        divergence=divergence,
        outlier=outlier,
        consensus_f0=consensus_f0,
        silent=silent,
    )


def analyze_disagreement(frame: np.ndarray, sr: int,
                         fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX,
                         representations=REPRESENTATIONS) -> DisagreementReport:
    """
    Run each representation on ``frame`` and measure their disagreement.

    Parameters:
        frame: 1D time-domain audio frame.
        sr: sample rate in Hz.
        fmin, fmax: pitch search range in Hz.
        representations: names from ``READERS`` to run (default: all).

    Returns:
        DisagreementReport
    """
    frame = np.asarray(frame, dtype=float).flatten()
    if frame.size == 0:
        raise ValueError("Empty audio frame provided.")

    names = list(representations)
    unknown = [r for r in names if r not in READERS]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}. Available: {list(READERS)}")

    if np.sqrt(np.mean(frame ** 2)) < SILENCE_RMS:
        report = compare_readings({name: _empty(name) for name in names}, silent=True)
    else:
        readings = {name: READERS[name](frame, sr, fmin, fmax) for name in names}
        report = compare_readings(readings)

    report.meta = {"sr": sr, "fmin": fmin, "fmax": fmax, "frame_length": int(frame.size)}
    return report
