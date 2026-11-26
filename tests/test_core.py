"""Tests for the medium-agnostic core (issue #4), using a toy non-audio medium."""

import os
import statistics
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
