"""
Core interfaces — medium-agnostic
=================================
The five building blocks of the loop in docs/VISION.md:

    Signal ─► Representation × n ─► BlindSpotDetector ─► Comparator ─► PatternModel
              (per medium)           (per reading)        (across readings) (across frames)

Nothing in this module knows about audio, images or any other medium. A medium
plugs in by implementing these classes; see docs/core_interfaces.md for the
audio worked example and an image sketch.

Only the Python standard library is used here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Data passed between the building blocks
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Reading:
    """What one representation says about one signal, in a form other
    representations of the same medium can be compared against.

    estimate:    the medium's comparable quantity (e.g. f0 in Hz for audio
                 pitch, an edge orientation for images). ``None`` = no estimate.
    confidence:  how strongly the representation supports its estimate, 0..1.
    details:     anything else worth keeping (raw peaks, intermediate values).
    """

    representation: str
    estimate: Any
    confidence: float
    details: dict = field(default_factory=dict)


@dataclass
class Assessment:
    """A BlindSpotDetector's verdict on one reading."""

    representation: str
    safe: float                 # 0 = certainly blind here, 1 = certainly fine
    blind: bool                 # the detector's decision
    reasons: tuple[str, ...] = ()


@dataclass
class PairComparison:
    """Agreement between two readings.

    distance:        estimate disagreement, normalised to 0..1 by the Comparator.
    relation:        a short, human-readable label for how they relate
                     (e.g. "agree", "octave above", "off").
    confidence_gap:  |confidence_a - confidence_b|.
    score:           combined disagreement, 0..1.
    """

    a: str
    b: str
    distance: float
    relation: str
    confidence_gap: float
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Agreement and disagreement across all readings of one signal."""

    readings: dict[str, Reading]
    pairs: list[PairComparison]
    overall: float                   # mean pair score, 0 (all agree) .. 1
    divergence: dict[str, float]     # per representation: mean score of its pairs
    outlier: str | None              # the representation most likely blind, if clear
    consensus: Any                   # fused estimate, or None when there is no quorum
    assessments: dict[str, Assessment] = field(default_factory=dict)
    consensus_from: tuple[str, ...] = ()  # the representations fused into the consensus

    def pair(self, a: str, b: str) -> PairComparison:
        for p in self.pairs:
            if {p.a, p.b} == {a, b}:
                return p
        raise KeyError(f"No pair ({a}, {b})")


@dataclass
class Pattern:
    """A named, inspectable pattern of disagreement found by a PatternModel."""

    name: str
    description: str = ""
    details: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# The five building blocks
# ──────────────────────────────────────────────────────────────────────────────

class Medium(ABC):
    """What a signal (one frame / patch / window) is for a medium."""

    #: short identifier, e.g. "audio", "image"
    name: str = ""

    @abstractmethod
    def validate(self, signal: Any, context: Mapping[str, Any]) -> Any:
        """Check and normalise a signal (shape, dtype, required context such as a
        sample rate). Return the normalised signal or raise ValueError."""

    def is_empty(self, signal: Any, context: Mapping[str, Any]) -> bool:
        """True when there is nothing to analyse (silence, a blank patch).
        Empty signals are reported, not compared. Default: never empty."""
        return False


class Representation(ABC):
    """A transform of the signal plus its stated assumptions.

    The assumptions are the representation's potential blind spots: where they
    break, its reading becomes unreliable.
    """

    #: unique name within a medium, e.g. "stft"
    name: str = ""
    #: the Medium.name this representation works on
    medium: str = ""
    #: plain-language assumptions, e.g. ("periodic within the frame",)
    assumptions: tuple[str, ...] = ()

    @abstractmethod
    def read(self, signal: Any, context: Mapping[str, Any]) -> Reading:
        """Transform the signal and reduce the result to a comparable Reading."""


class BlindSpotDetector(ABC):
    """Decides, per reading, whether a representation is likely blind."""

    @abstractmethod
    def assess(self, reading: Reading, signal: Any, context: Mapping[str, Any]) -> Assessment:
        """Return an Assessment for one reading."""


class Comparator(ABC):
    """Measures agreement and disagreement between the readings of one signal."""

    @abstractmethod
    def compare(self, readings: Mapping[str, Reading],
                assessments: Mapping[str, Assessment] | None = None) -> ComparisonReport:
        """Compare all readings of one signal."""


class PatternModel(ABC):
    """Learns structure over many ComparisonReports (across signals)."""

    @abstractmethod
    def vectorize(self, report: ComparisonReport) -> Sequence[float]:
        """Turn one report into an inspectable disagreement vector."""

    @abstractmethod
    def fit(self, reports: Sequence[ComparisonReport]) -> "PatternModel":
        """Learn patterns from many reports. Returns self."""

    @abstractmethod
    def assign(self, report: ComparisonReport) -> Pattern:
        """Name the pattern one report belongs to."""
