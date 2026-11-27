"""Tests for the medium-agnostic core (issue #4), using a toy non-audio medium."""

import os
import statistics
from itertools import combinations
import subprocess
import sys

import pytest

from core import (
    Analyzer,
    Assessment,
    ConfidenceThresholdDetector,
    Medium,
    PairwiseComparator,
    Pattern,
    PatternModel,
    Reading,
    Representation,
)

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


# ---------------------------------------------------------------------------
# Toy medium: a list of numbers; lenses estimate its "location"
# ---------------------------------------------------------------------------

class Numbers(Medium):
    name = "numbers"

    def validate(self, signal, context):
        values = [float(v) for v in signal]
        return values

    def is_empty(self, signal, context):
        return len(signal) == 0


class Mean(Representation):
    name, medium = "mean", "numbers"
    assumptions = ("no gross outliers",)

    def read(self, signal, context):
        return Reading(self.name, statistics.fmean(signal), 0.9)


class Median(Representation):
    name, medium = "median", "numbers"
    assumptions = ("at least half the values are typical",)

    def read(self, signal, context):
        return Reading(self.name, statistics.median(signal), 0.9)


class TrimmedMean(Representation):
    name, medium = "trimmed", "numbers"
    assumptions = ("outliers are at the extremes",)

    def read(self, signal, context):
        s = sorted(signal)
        k = max(1, len(s) // 5)
        core = s[k:-k] or s
        return Reading(self.name, statistics.fmean(core), 0.8)


class LocationComparator(PairwiseComparator):
    def estimate_distance(self, a, b):
        d = abs(a.estimate - b.estimate) / (abs(a.estimate) + abs(b.estimate) + 1e-12)
        return d, ("agree" if d < 0.05 else "off"), {}

    def fuse(self, readings):
        return statistics.median(r.estimate for r in readings)


class OutlierPatterns(PatternModel):
    """Toy pattern model: names the pattern after the outlier, if any."""

    def vectorize(self, report):
        return [report.divergence[k] for k in sorted(report.divergence)]

    def fit(self, reports):
        self.n_fitted = len(reports)
        return self

    def assign(self, report):
        if report.outlier is None:
            return Pattern("all agree")
        return Pattern(f"{report.outlier} off", details={"divergence": report.divergence[report.outlier]})


def make_analyzer(**kw):
    return Analyzer(Numbers(), [Mean(), Median(), TrimmedMean()], LocationComparator(), **kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_signal_all_lenses_agree():
    a = make_analyzer().analyze([10, 10.1, 9.9, 10, 10.2])
    r = a.report
    assert not a.empty
    assert r.overall < 0.05
    assert r.outlier is None
    assert r.consensus == pytest.approx(10.0, abs=0.2)
    assert all(p.relation == "agree" for p in r.pairs)


def test_gross_outlier_blinds_the_mean_and_is_detected():
    a = make_analyzer(pattern_model=OutlierPatterns()).analyze([10, 10, 10, 10, 10, 10, 10, 10, 10, 1000])
    r = a.report
    assert r.outlier == "mean"
    assert r.pair("median", "trimmed").relation == "agree"
    assert r.pair("mean", "median").relation == "off"
    assert r.consensus == pytest.approx(10.0)  # fused, not fooled
    assert a.pattern.name == "mean off"


def test_empty_signal_is_reported_not_compared():
    a = make_analyzer().analyze([])
    assert a.empty and a.report is None


def test_all_blind_gives_no_consensus_and_no_outlier():
    detector = ConfidenceThresholdDetector(threshold=0.95)  # every toy lens is below this
    a = make_analyzer(detector=detector).analyze([10, 10, 10, 10, 1000])
    assert all(x.blind for x in a.report.assessments.values())
    assert a.report.consensus is None
    assert a.report.outlier is None


def test_missing_estimate_counts_as_full_disagreement():
    c = LocationComparator()
    p = c.compare_pair(Reading("x", 1.0, 0.9), Reading("y", None, 0.0))
    assert p.distance == 1.0 and p.relation == "one empty"
    assessment = ConfidenceThresholdDetector().assess(Reading("y", None, 0.0))
    assert isinstance(assessment, Assessment) and assessment.blind


def test_analyzer_rejects_wrong_medium_and_duplicates():
    class ImageLens(Representation):
        name, medium = "edges", "image"

        def read(self, signal, context):
            return Reading(self.name, 0.0, 1.0)

    with pytest.raises(ValueError):
        Analyzer(Numbers(), [Mean(), ImageLens()], LocationComparator())
    with pytest.raises(ValueError):
        Analyzer(Numbers(), [Mean(), Mean()], LocationComparator())


def test_pattern_model_vectorize_and_fit():
    reports = [make_analyzer().analyze(s).report for s in ([1, 1, 1], [1, 1, 1, 1, 50])]
    model = OutlierPatterns().fit(reports)
    assert model.n_fitted == 2
    assert len(model.vectorize(reports[0])) == 3


def test_core_imports_no_medium_libraries():
    """The core must stay medium-agnostic: importing it loads no audio/numeric libraries."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import core; "
        "print(','.join(m for m in ('librosa', 'scipy', 'numpy', 'matplotlib') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code, SRC], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# ---------------------------------------------------------------------------
# Quorum (issue #20)
# ---------------------------------------------------------------------------

def _readings(**kw):
    return {n: Reading(n, est, conf) for n, (est, conf) in kw.items()}


def test_one_confident_lens_is_not_a_consensus():
    rd = _readings(mean=(50.0, 0.9), median=(10.0, 0.2), trimmed=(12.0, 0.3))
    assert LocationComparator().compare(rd).consensus is None  # default quorum: 2
    legacy = LocationComparator(min_agreeing=None).compare(rd)
    assert legacy.consensus == 50.0 and legacy.consensus_from == ("mean",)
    assert LocationComparator(min_agreeing=1).compare(rd).consensus == 50.0


def test_quorum_fuses_only_the_agreeing_group():
    rd = _readings(mean=(50.0, 0.95), median=(10.0, 0.9), trimmed=(10.1, 0.8))
    r = LocationComparator().compare(rd)
    assert r.consensus_from == ("median", "trimmed")
    assert r.consensus == pytest.approx(10.05)
    # the legacy rule fuses the confident outlier in as well
    assert LocationComparator(min_agreeing=None).compare(rd).consensus_from == ("mean", "median", "trimmed")
    assert LocationComparator(min_agreeing=3).compare(rd).consensus is None


def test_quorum_ignores_blind_readings_even_if_they_agree():
    rd = _readings(mean=(10.0, 0.9), median=(10.0, 0.1), trimmed=(30.0, 0.2))
    assert LocationComparator().compare(rd).consensus is None


def test_quorum_tie_goes_to_the_more_confident_group():
    rd = _readings(a=(10.0, 0.6), b=(10.1, 0.6), c=(50.0, 0.9), d=(50.2, 0.9))
    r = LocationComparator().compare(rd)
    assert r.consensus_from == ("c", "d")


def test_min_agreeing_must_be_positive():
    with pytest.raises(ValueError):
        LocationComparator(min_agreeing=0)


# ---------------------------------------------------------------------------
# Family-aware choice of group (issue #21)
# ---------------------------------------------------------------------------

from core import FamilyVotes  # noqa: E402


def _family_votes(rho_abc=1.0):
    """a, b, c always fail together; d and e fail independently. All equally reliable."""
    names = "abcde"
    rel = {n: [0.8] * 5 for n in names}
    corr = {}
    for x, y in combinations(names, 2):
        corr[frozenset((x, y))] = rho_abc if {x, y} <= set("abc") else 0.0
    return FamilyVotes(rel, corr)


def test_a_correlated_family_counts_as_one_vote():
    fv = _family_votes(1.0)
    rd = _readings(a=(50.0, 0.9), b=(50.1, 0.9), c=(50.2, 0.9), d=(10.0, 0.9), e=(10.1, 0.9))
    # independent counting: three beats two
    assert LocationComparator().compare(rd).consensus_from == ("a", "b", "c")
    # family-aware: a+b+c score 3 * 0.8 / 3 = 0.8, d+e score 1.6
    r = LocationComparator(family=fv).compare(rd)
    assert r.consensus_from == ("d", "e")
    assert fv.score(("a", "b", "c"), rd) == pytest.approx(0.8)
    assert fv.score(("d", "e"), rd) == pytest.approx(1.6)


def test_independent_family_votes_reduce_to_reliability_vote():
    fv = _family_votes(0.0)
    rd = _readings(a=(50.0, 0.9), b=(50.1, 0.9), c=(50.2, 0.9), d=(10.0, 0.9), e=(10.1, 0.9))
    assert LocationComparator(family=fv).compare(rd).consensus_from == ("a", "b", "c")


def test_family_choice_keeps_the_quorum():
    fv = _family_votes(1.0)
    rd = _readings(a=(50.0, 0.9), b=(10.0, 0.9), c=(30.0, 0.9), d=(70.0, 0.9), e=(90.0, 0.9))
    assert LocationComparator(family=fv).compare(rd).consensus is None
    assert LocationComparator(family=fv, min_agreeing=1).compare(rd).consensus is not None


def test_family_votes_fit_learns_reliability_and_failure_correlation():
    conf = [{"x": 0.9, "y": 0.9, "z": 0.9}] * 8
    # x and y fail together on the first 4 signals; z fails on alternate ones
    fail = [{"x": i < 4, "y": i < 4, "z": i % 2 == 0} for i in range(8)]
    fv = FamilyVotes.fit(conf, fail)
    assert fv.correlation[frozenset(("x", "y"))] == pytest.approx(1.0)
    assert fv.correlation[frozenset(("x", "z"))] == pytest.approx(0.0)
    assert fv.reliability("x", 0.9) == pytest.approx((4 + 1) / (8 + 2))
    assert FamilyVotes.fit(conf, fail, independent=True).correlation[frozenset(("x", "y"))] == 0.0


def test_agreement_table_trusts_the_group_that_was_right_before():
    from core import AgreementTable
    # in training, a+b+c agreeing was usually wrong, d+e agreeing usually right
    examples = [(("a", "b", "c"), i < 2) for i in range(10)] + [(("d", "e"), i < 9) for i in range(10)]
    table = AgreementTable.fit(examples)
    assert table.score(("a", "b", "c")) == pytest.approx(3 / 12)
    assert table.score(("e", "d")) == pytest.approx(10 / 12)
    assert table.score(("x", "y")) == pytest.approx(0.5)  # unseen group: the prior
    rd = _readings(a=(50.0, 0.9), b=(50.1, 0.9), c=(50.2, 0.9), d=(10.0, 0.9), e=(10.1, 0.9))
    assert LocationComparator(family=table).compare(rd).consensus_from == ("d", "e")


def test_min_score_abstains_when_no_group_is_likely_right():
    from core import AgreementTable
    table = AgreementTable.fit([(("a",), False)] * 9 + [(("a",), True)] + [(("b",), False)] * 5)
    rd = _readings(a=(50.0, 0.9), b=(10.0, 0.9))
    assert LocationComparator(family=table, min_agreeing=1).compare(rd).consensus is not None
    assert LocationComparator(family=table, min_agreeing=1, min_score=0.5).compare(rd).consensus is None
