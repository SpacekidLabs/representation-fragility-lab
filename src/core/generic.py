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
    Probe,
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


class FamilyVotes:
    """How much to trust an agreeing group when lenses fail in families (issue #21).

    Learned from labelled readings (which lens was wrong on which signal):

    reliability(name, confidence)
        P(lens is right | its confidence), from equal-width confidence bins on
        [0, 1] with a Beta(prior, prior) prior per bin.
    correlation[{a, b}]
        Failure correlation of two lenses: the phi coefficient of their
        wrong/right indicators, clipped to [0, 1]. 0 = they fail independently
        (or never fail), 1 = they always fail together.

    A group's score is the sum of its members' reliabilities, each divided by
    one plus its failure correlation with the other members:

        score(G) = sum_{i in G} p_i / (1 + sum_{j in G, j != i} R_ij)

    So k lenses that always fail together count as one vote, and k independent
    lenses count as k. With every correlation 0 this is a plain reliability vote.
    All numbers are kept in plain dicts so the weights stay inspectable.
    """

    def __init__(self, reliability: Mapping[str, Sequence[float]],
                 correlation: Mapping[frozenset, float]):
        self.bins = {n: list(v) for n, v in reliability.items()}
        self.correlation = {frozenset(k): float(v) for k, v in correlation.items()}

    @classmethod
    def fit(cls, confidences: Sequence[Mapping[str, float]], failures: Sequence[Mapping[str, bool]],
            bins: int = 5, prior: float = 1.0, independent: bool = False) -> "FamilyVotes":
        """Learn from one dict of confidences and one dict of failures per signal.

        ``independent=True`` sets every correlation to 0 (the ablation: a
        reliability vote that ignores families).
        """
        if len(confidences) != len(failures) or not failures:
            raise ValueError("need one confidence dict and one failure dict per signal")
        names = list(failures[0])
        reliability = {}
        for n in names:
            right = [0.0] * bins
            total = [0.0] * bins
            for c, f in zip(confidences, failures):
                b = min(int(float(c[n]) * bins), bins - 1) if c[n] > 0 else 0
                total[b] += 1
                right[b] += 0.0 if f[n] else 1.0
            reliability[n] = [(r + prior) / (t + 2 * prior) for r, t in zip(right, total)]
        correlation = {}
        m = len(failures)
        for a, b in combinations(names, 2):
            if independent:
                correlation[frozenset((a, b))] = 0.0
                continue
            fa = [bool(f[a]) for f in failures]
            fb = [bool(f[b]) for f in failures]
            pa, pb = sum(fa) / m, sum(fb) / m
            pab = sum(x and y for x, y in zip(fa, fb)) / m
            denom = (pa * (1 - pa) * pb * (1 - pb)) ** 0.5
            phi = (pab - pa * pb) / denom if denom > 0 else 0.0
            correlation[frozenset((a, b))] = min(max(phi, 0.0), 1.0)
        return cls(reliability, correlation)

    def reliability(self, name: str, confidence: float) -> float:
        bins = self.bins[name]
        k = len(bins)
        b = min(int(float(confidence) * k), k - 1) if confidence > 0 else 0
        return bins[b]

    def score(self, group: Sequence[str], readings: Mapping[str, Reading]) -> float:
        total = 0.0
        for i in group:
            shared = sum(self.correlation.get(frozenset((i, j)), 0.0) for j in group if j != i)
            total += self.reliability(i, readings[i].confidence) / (1.0 + shared)
        return total


class AgreementTable:
    """P(the fused answer is right | exactly these lenses form an agreeing group).

    The group-level alternative to ``FamilyVotes`` (issue #21, Exp 047). Instead
    of building a group's trust from pairwise failure correlations, it counts,
    on labelled signals, how often each maximal agreeing group (keyed by its set
    of lens names) gave a right answer. A group never seen in training gets the
    prior, 0.5. The table is plain data: ``table[frozenset(names)] = (right, seen)``.
    """

    def __init__(self, table: Mapping[frozenset, tuple[float, float]], prior: float = 1.0):
        self.table = {frozenset(k): (float(r), float(n)) for k, (r, n) in table.items()}
        self.prior = prior

    @classmethod
    def fit(cls, examples, prior: float = 1.0) -> "AgreementTable":
        """``examples``: (group names, right?) pairs, one per maximal agreeing group per signal."""
        table: dict[frozenset, list[float]] = {}
        for group, right in examples:
            cell = table.setdefault(frozenset(group), [0.0, 0.0])
            cell[0] += 1.0 if right else 0.0
            cell[1] += 1.0
        return cls({k: tuple(v) for k, v in table.items()}, prior)

    def score(self, group: Sequence[str], readings: Mapping[str, Reading] | None = None) -> float:
        r, n = self.table.get(frozenset(group), (0.0, 0.0))
        return (r + self.prior) / (n + 2 * self.prior)


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

    Family-aware choice (issue #21). With ``family`` (a ``FamilyVotes``), the
    group is not the largest one but the best-scoring maximal agreeing group of
    at least ``min_agreeing`` members (1 if ``min_agreeing`` is None): lenses that
    tend to fail together count for less than independent ones. ``family`` may
    also be an ``AgreementTable`` (anything with ``score(group, readings)``).

    Lead lens (issue #31). With ``lead`` (a representation name), the lead lens
    answers alone whenever it is trusted, and the other lenses only warn:
    ``report.warning`` is the lead's divergence. When the lead is blind or
    missing, ``lead_fallback`` decides: ``"abstain"`` (no consensus) or
    ``"quorum"`` (the usual group choice among the trusted readings).
    ``min_score`` then abstains when even the best group scores below it (for an
    ``AgreementTable``, 0.5 = answer only when the group is more likely right
    than wrong).
    """

    def __init__(self, distance_weight: float = 0.5, outlier_margin: float = 0.10,
                 detector: BlindSpotDetector | None = None, min_agreeing: int | None = 2,
                 family: "FamilyVotes | AgreementTable | None" = None, min_score: float | None = None,
                 lead: str | None = None, lead_fallback: str = "abstain"):
        if min_agreeing is not None and min_agreeing < 1:
            raise ValueError("min_agreeing must be >= 1 or None")
        if lead_fallback not in ("abstain", "quorum"):
            raise ValueError('lead_fallback must be "abstain" or "quorum"')
        self.distance_weight = distance_weight
        self.outlier_margin = outlier_margin
        self.detector = detector or ConfidenceThresholdDetector()
        self.min_agreeing = min_agreeing
        self.family = family
        self.min_score = min_score
        self.lead = lead
        self.lead_fallback = lead_fallback

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

    def agreeing_groups(self, trusted: Sequence[str], pairs: Sequence[PairComparison]) -> list[tuple[str, ...]]:
        """Every maximal group of trusted readings that pairwise agree (exhaustive)."""
        agree = {frozenset((p.a, p.b)) for p in pairs if self.agrees(p)}
        cliques = [g for size in range(len(trusted), 0, -1) for g in combinations(trusted, size)
                   if all(frozenset(ab) in agree for ab in combinations(g, 2))]
        return [g for g in cliques if not any(set(g) < set(h) for h in cliques)]

    def family_group(self, readings: Mapping[str, Reading], trusted: Sequence[str],
                     pairs: Sequence[PairComparison]) -> tuple[str, ...]:
        """The maximal agreeing group with the highest ``FamilyVotes.score`` among
        those with at least ``min_agreeing`` members (empty if none). Ties go to
        the higher total confidence, then to the earlier group."""
        k = self.min_agreeing or 1
        best, best_key = (), None
        for g in self.agreeing_groups(trusted, pairs):
            if len(g) < k:
                continue
            key = (self.family.score(g, readings), sum(readings[n].confidence for n in g))
            if best_key is None or key > best_key:
                best, best_key = g, key
        if best and self.min_score is not None and best_key[0] < self.min_score:
            return ()
        return best


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
        if self.lead is not None and self.lead in trusted_names:
            fused_from = (self.lead,)
        elif self.lead is not None and self.lead_fallback == "abstain":
            fused_from = ()
        else:
            fused_from = self.group(readings, trusted_names, pairs)
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
            warning=divergence.get(self.lead) if self.lead is not None and len(names) >= 2 else None,
        )

    def group(self, readings: Mapping[str, Reading], trusted: Sequence[str],
              pairs: Sequence[PairComparison]) -> tuple[str, ...]:
        """The group the consensus is fused from, under the quorum / family rules."""
        if self.family is not None:
            return self.family_group(readings, trusted, pairs)
        if self.min_agreeing is None:
            return tuple(trusted)
        group = self.quorum_group(readings, trusted, pairs)
        return group if len(group) >= self.min_agreeing else ()


@dataclass
class ProbeResult:
    """What probing found on one signal (issue #30).

    candidates  the distinct answers: (estimate, names of the lenses that read it)
    follows     per candidate, how many (probe, lens) readings followed its prediction
    checks      per candidate, how many (probe, lens) readings were made
    choice      the chosen candidate's estimate, or None when probing abstains
    score       the chosen candidate's share of follows, 0..1: low on input with
                nothing to find (no pitch, no orientation)
    """

    candidates: list[tuple[Any, tuple[str, ...]]]
    follows: list[int]
    checks: list[int]
    choice: Any
    score: float
    abstained: bool = False


@dataclass
class Analysis:
    """Result of running the full loop on one signal."""

    medium: str
    empty: bool
    report: ComparisonReport | None
    pattern: Pattern | None = None
    context: dict = field(default_factory=dict)
    probe: ProbeResult | None = None

    @property
    def answer(self) -> Any:
        """The probe's choice where probing ran, else the comparator's consensus."""
        if self.probe is not None:
            return self.probe.choice
        return self.report.consensus if self.report is not None else None


class Prober:
    """Settle a disagreement by asking the signal (issue #30, Exp 050).

    The distinct answers of the lenses are the candidates (readings that agree
    with a candidate's first, most confident member are merged into it, and the
    candidate's estimate is ``comparator.fuse`` over them). Every reading counts,
    trusted or not: the right lens is often the one marked blind (Exp 047). For
    each candidate the medium supplies probes, perturbations with a known
    effect; every lens reads every probe, and a lens *follows* when its reading
    agrees (``comparator.agrees``) with the probe's prediction. The candidate
    with the most follows wins (ties: more lenses behind it, then the higher
    confidence). Untrained.

    Probing runs only when there are at least ``min_candidates`` candidates
    (default 2: the lenses disagree). With ``min_score`` set, it abstains when
    the winner's share of follows is below it: on input with nothing to find,
    no candidate follows its probes (Exp 050: AUC 0.93-0.98 for pitch).
    """

    def __init__(self, min_score: float | None = None, min_candidates: int = 2):
        self.min_score = min_score
        self.min_candidates = min_candidates

    def candidates(self, comparator: "PairwiseComparator",
                   readings: Mapping[str, Reading]) -> list[tuple[Any, tuple[str, ...]]]:
        groups: list[list[Reading]] = []
        for r in sorted((r for r in readings.values() if r.estimate is not None), key=lambda r: -r.confidence):
            for g in groups:
                if comparator.agrees(comparator.compare_pair(g[0], r)):
                    g.append(r)
                    break
            else:
                groups.append([r])
        out = []
        for g in groups:
            weighted = [Reading(r.representation, r.estimate, max(r.confidence, 1e-9)) for r in g]
            out.append((comparator.fuse(weighted), tuple(r.representation for r in g)))
        return out

    def settle(self, analyzer: "Analyzer", signal: Any, context: Mapping[str, Any],
               report: ComparisonReport) -> ProbeResult | None:
        comparator = analyzer.comparator
        cands = self.candidates(comparator, report.readings)
        if len(cands) < self.min_candidates:
            return None
        follows, checks = [], []
        for estimate, _ in cands:
            n_follow = n_check = 0
            for probe in analyzer.medium.probes(signal, estimate, context):
                target = Reading("probe", probe.predicted, 1.0)
                for rep in analyzer.representations:
                    reading = rep.read(probe.signal, context)
                    n_check += 1
                    n_follow += int(comparator.agrees(comparator.compare_pair(reading, target)))
            follows.append(n_follow)
            checks.append(n_check)
        if not any(checks):
            return None                     # the medium has no probes
        best = max(range(len(cands)), key=lambda k: (
            follows[k], len(cands[k][1]), max(report.readings[n].confidence for n in cands[k][1])))
        score = follows[best] / checks[best] if checks[best] else 0.0
        abstain = self.min_score is not None and score < self.min_score
        return ProbeResult(cands, follows, checks, None if abstain else cands[best][0], score, abstain)


class Analyzer:
    """The loop from docs/VISION.md, for any medium:

    validate → read with every representation → assess blind spots →
    compare → (optionally) assign a disagreement pattern.
    """

    def __init__(self, medium: Medium, representations: Sequence[Representation],
                 comparator: Comparator, detector: BlindSpotDetector | None = None,
                 pattern_model: PatternModel | None = None, prober: Prober | None = None):
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
        self.prober = prober

    def analyze(self, signal: Any, context: Mapping[str, Any] | None = None) -> Analysis:
        context = dict(context or {})
        signal = self.medium.validate(signal, context)
        if self.medium.is_empty(signal, context):
            return Analysis(self.medium.name, True, None, None, context)
        readings = {r.name: r.read(signal, context) for r in self.representations}
        assessments = {n: self.detector.assess(rd, signal, context) for n, rd in readings.items()}
        report = self.comparator.compare(readings, assessments)
        pattern = self.pattern_model.assign(report) if self.pattern_model else None
        probe = self.prober.settle(self, signal, context, report) if self.prober else None
        return Analysis(self.medium.name, False, report, pattern, context, probe)
