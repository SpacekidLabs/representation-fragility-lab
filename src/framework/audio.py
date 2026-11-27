"""
Audio medium on the medium-agnostic core
========================================
Audio is the first medium built on ``src/core`` (issue #5, docs/core_interfaces.md).

- ``AudioFrame``         the Medium: a 1-D float frame, ``context["sr"]`` required
- ``StftPitch``, ``AcfPitch``, ``CepstrumPitch``, ``CqtPitch``, ``WaveletPitch``
                         the Representations: pitch lenses with stated assumptions
- ``PitchComparator``    the Comparator: distance in cents, octave-aware relations,
                         confidence-weighted median as the consensus
- ``make_audio_analyzer`` a ready-made ``core.Analyzer``

The detector is the core's ``ConfidenceThresholdDetector`` at the voicing
threshold (0.5). The default lens set is STFT + ACF + Cepstrum, which keeps all
earlier results unchanged; CQT and wavelet lenses are opt-in.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np

try:  # installed package: representation_core
    from representation_core import (
        Analyzer,
        ConfidenceThresholdDetector,
        Medium,
        PairwiseComparator,
        Probe,
        Prober,
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
        Probe,
        Prober,
        Reading,
        Representation,
    )

from .disagreement import (
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    OCTAVE_TOLERANCE_CENTS,
    READERS,
    SILENCE_RMS,
    VOICING_THRESHOLD,
)

DEFAULT_LENSES = ("stft", "acf", "cepstrum")
ALL_LENSES = ("stft", "acf", "cepstrum", "cqt", "wavelet")


class AudioFrame(Medium):
    """One mono audio frame. Context: ``sr`` (required), ``fmin``/``fmax`` (optional)."""

    name = "audio"

    def validate(self, signal: Any, context: Mapping[str, Any]) -> np.ndarray:
        frame = np.asarray(signal, dtype=float).flatten()
        if frame.size == 0:
            raise ValueError("Empty audio frame provided.")
        if "sr" not in context:
            raise ValueError("Audio context needs a sample rate: context['sr'].")
        return frame

    def is_empty(self, signal: np.ndarray, context: Mapping[str, Any]) -> bool:
        return bool(np.sqrt(np.mean(signal ** 2)) < SILENCE_RMS)

    #: Low-pass probe (issue #30, Exp 050): cut at this multiple of the candidate f0.
    #: If the candidate is right its fundamental survives and the lenses read it;
    #: if the truth is 2c nothing pitched is left; if it is c / 2 the lenses go there.
    LOWPASS_FACTOR = 1.5

    def probes(self, signal: np.ndarray, candidate: float, context: Mapping[str, Any]):
        from scipy.signal import butter, filtfilt

        sr = context["sr"]
        cut = self.LOWPASS_FACTOR * candidate / (0.5 * sr)
        if cut >= 1.0:
            low = signal.copy()
        else:
            b, a = butter(5, cut, btype="low")
            low = filtfilt(b, a, signal)
        return [Probe("lowpass", low, candidate)]


class _PitchLens(Representation):
    """Wraps a pitch reader from framework.disagreement as a core Representation."""

    medium = "audio"

    def read(self, signal: np.ndarray, context: Mapping[str, Any]) -> Reading:
        r = READERS[self.name](signal, context["sr"],
                               context.get("fmin", DEFAULT_FMIN), context.get("fmax", DEFAULT_FMAX))
        return Reading(self.name, r.f0 if r.f0 > 0 else None, r.confidence)


class StftPitch(_PitchLens):
    name = "stft"
    assumptions = ("the frame is stationary", "harmonics are resolved by the window",
                   "harmonic peaks stand above the noise floor")


class AcfPitch(_PitchLens):
    name = "acf"
    assumptions = ("the waveform repeats within the frame", "at least two periods fit in the frame")


class CepstrumPitch(_PitchLens):
    name = "cepstrum"
    assumptions = ("many evenly spaced harmonics", "a smooth spectral envelope separable from the harmonics")


class CqtPitch(_PitchLens):
    name = "cqt"
    assumptions = ("low harmonics (up to about the 11th) are resolved on a log-frequency axis",
                   "the pitch is stable over the long low-frequency filters")


class WaveletPitch(_PitchLens):
    name = "wavelet"
    assumptions = ("a strong fundamental and low partials (only the first 3 are resolved)",
                   "the pitch is roughly constant over the middle half of the frame")


LENSES = {cls.name: cls for cls in (StftPitch, AcfPitch, CepstrumPitch, CqtPitch, WaveletPitch)}


class PitchComparator(PairwiseComparator):
    """Pitch-specific distance and fusion; everything else is the core's generic logic."""

    def estimate_distance(self, a: Reading, b: Reading):
        cents = float(1200.0 * abs(np.log2(a.estimate / b.estimate)))
        nearest_octave = 1200.0 * round(cents / 1200.0)
        octave_related = cents >= 600.0 and abs(cents - nearest_octave) <= OCTAVE_TOLERANCE_CENTS
        if cents < OCTAVE_TOLERANCE_CENTS:
            relation = "agree"
        elif octave_related:
            relation = "octave above" if a.estimate > b.estimate else "octave below"
        else:
            relation = "off"
        return min(cents / 1200.0, 1.0), relation, {"cents": cents, "octave_related": octave_related}

    def fuse(self, readings: Sequence[Reading]) -> float:
        """Confidence-weighted median f0."""
        v = np.array([r.estimate for r in readings], float)
        w = np.array([r.confidence for r in readings], float)
        order = np.argsort(v)
        cum = np.cumsum(w[order])
        return float(v[order][np.searchsorted(cum, 0.5 * cum[-1])])


def make_audio_analyzer(lenses: Sequence[str] = DEFAULT_LENSES,
                        threshold: float = VOICING_THRESHOLD, prober: Prober | None = None) -> Analyzer:
    """A core Analyzer for audio pitch with the given lenses (names from ``ALL_LENSES``).
    Pass ``prober=Prober()`` to settle disagreements with the low-pass probe (Exp 050 / 057)."""
    unknown = [n for n in lenses if n not in LENSES]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}. Available: {list(LENSES)}")
    detector = ConfidenceThresholdDetector(threshold)
    return Analyzer(AudioFrame(), [LENSES[n]() for n in lenses],
                    PitchComparator(detector=detector), detector=detector, prober=prober)
