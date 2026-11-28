"""
Representation core — medium-agnostic building blocks
=====================================================
The comparison and pattern machinery of docs/VISION.md, independent of any
medium. Audio, images and other media plug in by implementing ``Medium``,
``Representation`` and a ``PairwiseComparator.estimate_distance``.

See docs/core_interfaces.md. Standard library only: importing this package
never imports audio (or any other medium's) libraries.
"""

from .interfaces import (
    Assessment,
    BlindSpotDetector,
    Comparator,
    ComparisonReport,
    Medium,
    PairComparison,
    Pattern,
    PatternModel,
    Probe,
    Reading,
    Representation,
)
from .admission import Admission, admission_test
from .generic import (
    AgreementTable,
    Analysis,
    Analyzer,
    ConfidenceThresholdDetector,
    FamilyVotes,
    PairwiseComparator,
    Prober,
    ProbeResult,
)

__all__ = [
    "Admission",
    "AgreementTable",
    "Analysis",
    "Analyzer",
    "Assessment",
    "admission_test",
    "BlindSpotDetector",
    "Comparator",
    "ComparisonReport",
    "ConfidenceThresholdDetector",
    "FamilyVotes",
    "Medium",
    "PairComparison",
    "PairwiseComparator",
    "Pattern",
    "PatternModel",
    "Probe",
    "Prober",
    "ProbeResult",
    "Reading",
    "Representation",
]
