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
    Reading,
    Representation,
)
from .generic import Analysis, Analyzer, ConfidenceThresholdDetector, PairwiseComparator

__all__ = [
    "Analysis",
    "Analyzer",
    "Assessment",
    "BlindSpotDetector",
    "Comparator",
    "ComparisonReport",
    "ConfidenceThresholdDetector",
    "Medium",
    "PairComparison",
    "PairwiseComparator",
    "Pattern",
    "PatternModel",
    "Reading",
    "Representation",
]
