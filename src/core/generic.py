"""
Generic, reusable implementations of the core interfaces
========================================================
Everything here works for any medium. A medium only has to say how far apart
two of its estimates are (``PairwiseComparator.estimate_distance``) and,
optionally, how to fuse them (``PairwiseComparator.fuse``).

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

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


class ConfidenceThresholdDetector(BlindSpotDetector):
    """The simplest detector: a reading is blind when its own confidence is low."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def assess(self, reading: Reading, signal: Any = None, context: Mapping[str, Any] | None = None) -> Assessment:
        blind = reading.estimate is None or reading.confidence < self.threshold
        reasons = ()
        if reading.estimate is None:
            reasons = ("no estimate",)
        elif blind:
            reasons = (f"confidence {reading.confidence:.2f} < {self.threshold:g}",)
        return Assessment(reading.representation, float(reading.confidence), blind, reasons)


class PairwiseComparator(Comparator):
    """Pairwise comparison with medium-specific distance, generic aggregation.

    Pair score   = distance_weight * distance + (1 - distance_weight) * confidence_gap
    overall      = mean pair score
    divergence   = per representation, mean score of the pairs it is part of
    outlier      = the most divergent representation, when it leads the next one
                   by ``outlier_margin`` and at least one reading is not blind
    consensus    = ``fuse`` over the quorum group (see below), else None

    Quorum (issue #20). With ``min_agreeing = k`` the comparator looks for the
    largest group of trusted (non-blind) readings that all agree with each other
    pairwise (``agrees``; by default the medium's relation is "agree"). Ties go
    to the group with the higher total confidence. If that group has at least
    ``k`` members, the consensus is ``fuse`` over it; otherwise there is no
    consensus. ``min_agreeing = None`` is the legacy rule: ``fuse`` over all
    trusted readings, even a single one.
    """

    def __init__(self, distance_weight: float = 0.5, outlier_margin: float = 0.10,
                 detector: BlindSpotDetector | None = None, min_agreeing: int | None = 2):
        if min_agreeing is not None and min_agreeing < 1:
            raise ValueError("min_agreeing must be >= 1 or None")
        self.distance_weight = distance_weight
        self.outlier_margin = outlier_margin
        self.detector = detector or ConfidenceThresholdDetector()
        self.min_agreeing = min_agreeing

    # -- medium-specific hooks ------------------------------------------------

    def estimate_distance(self, a: Reading, b: Reading) -> tuple[float, str, dict]:
        """Return (distance in 0..1, relation label, details). Must be overridden."""
        raise NotImplementedError

    def fuse(self, readings: Sequence[Reading]) -> Any:
        """Fuse trusted readings into one estimate. Default: the most confident one."""
        return max(readings, key=lambda r: r.confidence).estimate

    def agrees(self, pair: PairComparison) -> bool:
        """Whether two readings agree, for the quorum. Default: relation == "agree"."""
        return pair.relation == "agree"

    # -- generic machinery ----------------------------------------------------

    def quorum_group(self, readings: Mapping[str, Reading], trusted: Sequence[str],
                     pairs: Sequence[PairComparison]) -> tuple[str, ...]:
        """The largest group of trusted readings that pairwise agree (empty if none).

        Ties go to the higher total confidence, then to the earlier names.
        Exhaustive search: fine for the handful of lenses a medium has.
        """
        agree = {frozenset((p.a, p.b)) for p in pairs if self.agrees(p)}
        for size in range(len(trusted), 0, -1):
            best, best_conf = (), -1.0
            for group in combinations(trusted, size):
                if all(frozenset(ab) in agree for ab in combinations(group, 2)):
                    conf = sum(readings[n].confidence for n in group)
                    if conf > best_conf:
                        best, best_conf = group, conf
            if best:
                return best
        return ()


    def compare_pair(self, a: Reading, b: Reading) -> PairComparison:
        if a.estimate is None and b.estimate is None:
            distance, relation, details = 0.0, "both empty", {}
        elif a.estimate is None or b.estimate is None:
            distance, relation, details = 1.0, "one empty", {}
        else:
            distance, relation, details = self.estimate_distance(a, b)
            distance = min(max(float(distance), 0.0), 1.0)
        gap = abs(a.confidence - b.confidence)
        w = self.distance_weight
        return PairComparison(a.representation, b.representation, distance, relation, gap,
                              w * distance + (1 - w) * gap, details)

    def compare(self, readings: Mapping[str, Reading],
                assessments: Mapping[str, Assessment] | None = None) -> ComparisonReport:
        readings = dict(readings)
        if assessments is None:
            assessments = {k: self.detector.assess(r) for k, r in readings.items()}
        names = list(readings)
        pairs = [self.compare_pair(readings[a], readings[b]) for a, b in combinations(names, 2)]
        overall = sum(p.score for p in pairs) / len(pairs) if pairs else 0.0

        divergence = {}
        for n in names:
            scores = [p.score for p in pairs if n in (p.a, p.b)]
            divergence[n] = sum(scores) / len(scores) if scores else 0.0

        trusted_names = [n for n in names if not assessments[n].blind]
        trusted = [readings[n] for n in trusted_names]
        if self.min_agreeing is None:
            fused_from = tuple(trusted_names)
        else:
            group = self.quorum_group(readings, trusted_names, pairs)
            fused_from = group if len(group) >= self.min_agreeing else ()
        outlier = None
        if len(names) >= 3 and trusted:
            ranked = sorted(names, key=divergence.get, reverse=True)
            if divergence[ranked[0]] - divergence[ranked[1]] >= self.outlier_margin:
                outlier = ranked[0]

        return ComparisonReport(
            readings=readings,
            pairs=pairs,
            overall=overall,
            divergence=divergence,
            outlier=outlier,
            consensus=self.fuse([readings[n] for n in fused_from]) if fused_from else None,
            assessments=dict(assessments),
            consensus_from=fused_from,
        )


@dataclass
class Analysis:
    """Result of running the full loop on one signal."""

    medium: str
    empty: bool
    report: ComparisonReport | None
    pattern: Pattern | None = None
    context: dict = field(default_factory=dict)


class Analyzer:
    """The loop from docs/VISION.md, for any medium:

    validate → read with every representation → assess blind spots →
    compare → (optionally) assign a disagreement pattern.
    """

    def __init__(self, medium: Medium, representations: Sequence[Representation],
                 comparator: Comparator, detector: BlindSpotDetector | None = None,
                 pattern_model: PatternModel | None = None):
        wrong = [r.name for r in representations if r.medium and r.medium != medium.name]
        if wrong:
            raise ValueError(f"Representations {wrong} are not for medium '{medium.name}'")
        names = [r.name for r in representations]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate representation names: {names}")
        self.medium = medium
        self.representations = list(representations)
        self.comparator = comparator
        self.detector = detector or ConfidenceThresholdDetector()
        self.pattern_model = pattern_model

    def analyze(self, signal: Any, context: Mapping[str, Any] | None = None) -> Analysis:
        context = dict(context or {})
        signal = self.medium.validate(signal, context)
        if self.medium.is_empty(signal, context):
            return Analysis(self.medium.name, True, None, None, context)
        readings = {r.name: r.read(signal, context) for r in self.representations}
        assessments = {n: self.detector.assess(rd, signal, context) for n, rd in readings.items()}
        report = self.comparator.compare(readings, assessments)
        pattern = self.pattern_model.assign(report) if self.pattern_model else None
        return Analysis(self.medium.name, False, report, pattern, context)
