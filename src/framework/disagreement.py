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

import warnings
from dataclasses import asdict, dataclass, field

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


CQT_BINS_PER_OCTAVE = 24
WAVELET_CYCLES = 6.0


def _log_grid_salience(profile: np.ndarray, B: int, n_cand: int, max_harm: int) -> np.ndarray:
    """Weighted harmonic summation on a log-frequency profile with B bins/octave."""
    pos = np.arange(len(profile))
    sal = np.zeros(n_cand)
    for b in range(n_cand):
        k = np.arange(1, max_harm + 1)
        at = b + B * np.log2(k)
        k, at = k[at <= len(profile) - 1], at[at <= len(profile) - 1]
        sal[b] = np.sum(np.interp(at, pos, profile) / k)
    return sal


def _refine_on_profile(profile: np.ndarray, best: int) -> float:
    """Sub-bin position of the profile peak nearest to candidate bin ``best``."""
    lo, hi = max(best - 1, 0), min(best + 2, len(profile))
    i = lo + int(np.argmax(profile[lo:hi]))
    return _refine_peak(profile, i)


def _contrast_confidence(sal: np.ndarray, best: int, levels: tuple[float, float]) -> float:
    """How much the best candidate stands out from the typical one, mapped to 0..1."""
    contrast = 1.0 - np.median(sal) / (sal[best] + 1e-12)
    lo, hi = levels
    return float(np.clip((contrast - lo) / (hi - lo + 1e-12), 0.0, 1.0))


def read_cqt(frame: np.ndarray, sr: int,
             fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX) -> RepresentationReading:
    """Pitch + confidence from a constant-Q transform (24 bins/octave).

    Harmonic summation on the log-frequency axis. Only harmonics that stay at
    least 2 bins apart are used (up to the 11th at 24 bins/octave): on a
    constant-Q axis higher partials merge. Full-length filters are used (the
    frame is zero-padded); shortening them to fit the frame skews the peaks
    and biases the pitch by about -11 cents.
    """
    import librosa  # optional dependency: only needed for this lens

    B = CQT_BINS_PER_OCTAVE
    f_lim = min(5000.0, 0.45 * sr)
    n_bins = int(np.floor(B * np.log2(f_lim / fmin))) + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # librosa warns about short low-octave buffers
        C = np.abs(librosa.cqt(frame.astype(float), sr=sr, hop_length=256, fmin=fmin, n_bins=n_bins,
                               bins_per_octave=B))
    profile = C.mean(axis=1)
    if profile.max() <= 0.0:
        return _empty("cqt")
    profile = profile / profile.max()
    max_harm = int(np.floor(1.0 / (2.0 ** (2.0 / B) - 1.0)))  # partials >= 2 bins apart
    n_cand = int(np.floor(B * np.log2(fmax / fmin))) + 1
    sal = _log_grid_salience(profile, B, n_cand, max_harm)
    best = int(np.argmax(sal))
    f0 = float(fmin * 2.0 ** (_refine_on_profile(profile, best) / B))
    return RepresentationReading("cqt", f0, _contrast_confidence(sal, best, CQT_CONTRAST))


def read_wavelet(frame: np.ndarray, sr: int,
                 fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX) -> RepresentationReading:
    """Pitch + confidence from a Morlet wavelet scalogram (6 cycles, 24 voices/octave).

    The wavelets are broad (relative bandwidth 1/6), so only the lowest
    harmonics are resolved: this lens leans on a strong fundamental and low
    partials, and is robust to time-local changes inside the frame.
    """
    x = frame - np.mean(frame)
    n = len(x)
    B = CQT_BINS_PER_OCTAVE
    f_lim = min(5000.0, 0.45 * sr)
    centers = fmin * 2.0 ** (np.arange(int(np.floor(B * np.log2(f_lim / fmin))) + 1) / B)
    X = np.fft.fft(x)
    nu = np.fft.fftfreq(n, d=1.0 / sr)
    lo, hi = n // 4, 3 * n // 4  # ignore edge effects
    raw = np.empty(len(centers))
    for i, fc in enumerate(centers):
        sigma = fc / WAVELET_CYCLES
        G = np.exp(-0.5 * ((nu - fc) / sigma) ** 2) * (nu > 0)  # analytic Morlet in frequency
        w = np.fft.ifft(X * G)
        raw[i] = np.mean(np.abs(w[lo:hi]))
    # Divide by sqrt(bandwidth) so white noise gives a flat profile.
    profile = raw / np.sqrt(centers / WAVELET_CYCLES)
    if profile.max() <= 0.0:
        return _empty("wavelet")
    profile = profile / profile.max()
    max_harm = 3  # resolved at 6 cycles
    n_cand = int(np.floor(B * np.log2(fmax / fmin))) + 1
    sal = _log_grid_salience(profile, B, n_cand, max_harm)
    best = int(np.argmax(sal))
    # Refine on the raw (un-whitened) response, whose peak sits exactly on the partial.
    f0 = float(fmin * 2.0 ** (_refine_on_profile(raw, best) / B))
    return RepresentationReading("wavelet", f0, _contrast_confidence(sal, best, WAVELET_CONTRAST))


# (noise level, clean level) of the salience contrast that map to confidence
# 0 and 1. Calibrated on 22.05 kHz, 2048-sample frames: maximum over 60 white-
# noise frames (CQT 0.405, wavelet 0.385) and 5th percentile over clean
# harmonic tones at 110-660 Hz with 1-20 harmonics (CQT 0.912, wavelet 0.712).
CQT_CONTRAST = (0.45, 0.90)
WAVELET_CONTRAST = (0.40, 0.70)

READERS = {
    "stft": read_stft,
    "acf": read_acf,
    "cepstrum": read_cepstrum,
    "cqt": read_cqt,
    "wavelet": read_wavelet,
}


# ---------------------------------------------------------------------------
# Comparison — adapters over the medium-agnostic core (issue #5)
# ---------------------------------------------------------------------------
# The comparison logic lives in src/core (PairwiseComparator) plus the audio
# PitchComparator in framework/audio.py. The functions below keep the original
# API and DisagreementReport format, converting to and from the core types.

def _core():
    from . import audio  # lazy: audio imports this module
    return audio


def _to_core(r: RepresentationReading):
    return _core().Reading(r.name, r.f0 if r.f0 > 0.0 else None, r.confidence)


def _comparator():
    audio = _core()
    detector = audio.ConfidenceThresholdDetector(VOICING_THRESHOLD)
    return audio.PitchComparator(outlier_margin=OUTLIER_MARGIN, detector=detector)


def _from_core_pair(p, a: RepresentationReading, b: RepresentationReading) -> PairDisagreement:
    return PairDisagreement(
        a=p.a,
        b=p.b,
        cents=p.details.get("cents"),
        octave_related=bool(p.details.get("octave_related", False)),
        pitch_term=p.distance,
        confidence_gap=p.confidence_gap,
        voicing_conflict=a.voiced != b.voiced,
        score=p.score,
    )


def compare_pair(a: RepresentationReading, b: RepresentationReading) -> PairDisagreement:
    """Disagreement between two readings."""
    return _from_core_pair(_comparator().compare_pair(_to_core(a), _to_core(b)), a, b)


def _from_core_report(report, readings: dict[str, RepresentationReading],
                      silent: bool = False) -> DisagreementReport:
    return DisagreementReport(
        readings=readings,
        pairs=[_from_core_pair(p, readings[p.a], readings[p.b]) for p in report.pairs],
        overall=float(report.overall),
        divergence={k: float(v) for k, v in report.divergence.items()},
        outlier=report.outlier,
        consensus_f0=float(report.consensus) if report.consensus is not None else 0.0,
        silent=silent,
    )


def compare_readings(readings: dict[str, RepresentationReading],
                     silent: bool = False) -> DisagreementReport:
    """Build a disagreement report from already-computed readings."""
    report = _comparator().compare({k: _to_core(r) for k, r in readings.items()})
    return _from_core_report(report, readings, silent)


def analyze_disagreement(frame: np.ndarray, sr: int,
                         fmin: float = DEFAULT_FMIN, fmax: float = DEFAULT_FMAX,
                         representations=REPRESENTATIONS) -> DisagreementReport:
    """
    Run each representation on ``frame`` and measure their disagreement.

    Runs the core ``Analyzer`` for the audio medium (framework/audio.py) and
    returns the result in the original ``DisagreementReport`` format.

    Parameters:
        frame: 1D time-domain audio frame.
        sr: sample rate in Hz.
        fmin, fmax: pitch search range in Hz.
        representations: names from ``READERS`` to run (default: STFT, ACF, Cepstrum;
            "cqt" and "wavelet" are also available).

    Returns:
        DisagreementReport
    """
    names = list(representations)
    unknown = [r for r in names if r not in READERS]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}. Available: {list(READERS)}")

    analysis = _core().make_audio_analyzer(names).analyze(frame, {"sr": sr, "fmin": fmin, "fmax": fmax})
    if analysis.empty:
        report = compare_readings({name: _empty(name) for name in names}, silent=True)
    else:
        readings = {
            name: RepresentationReading(name, r.estimate if r.estimate is not None else 0.0, r.confidence)
            for name, r in analysis.report.readings.items()
        }
        report = _from_core_report(analysis.report, readings)

    report.meta = {"sr": sr, "fmin": fmin, "fmax": fmax,
                   "frame_length": int(np.asarray(frame).size)}
    return report
