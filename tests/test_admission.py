"""Tests for the lens admission test (core.admission, issue #32)."""

import math

from core import ConfidenceThresholdDetector, PairwiseComparator, Reading, admission_test
from core.admission import group_bootstrap


class Scalar(PairwiseComparator):
    """Toy comparator: estimates are numbers; two agree within 1."""

    def __init__(self, k=2):
        super().__init__(min_agreeing=k, detector=ConfidenceThresholdDetector(0.5))

    def estimate_distance(self, a, b):
        d = abs(a.estimate - b.estimate)
        return min(d / 10.0, 1.0), ("agree" if d <= 1 else "off"), {}

    def agrees(self, pair):
        return pair.relation == "agree"

    def fuse(self, readings):
        return sorted(r.estimate for r in readings)[len(readings) // 2]


def frame(**est):
    return {n: Reading(n, v, 1.0 if v is not None else 0.0) for n, v in est.items()}


def test_candidate_that_completes_quorums_is_admitted():
    # a is always right, b always wrong: the pool (a, b) never reaches a quorum of 2.
    frames = [frame(a=10, b=50, c=10) for _ in range(20)]
    truths = [10] * 20
    res = admission_test(frames, truths, ("a", "b"), "c", Scalar(), lambda e, t: abs(e - t) > 1,
                         groups=list(range(20)), n_boot=200)
    assert res.coverage == 1.0                       # c right, only one pool lens right
    assert res.coverage_unanswered == 1.0
    assert res.n_unanswered == 20
    assert res.without["abstention"] == 1.0 and res.with_candidate["abstention"] == 0.0
    assert res.value["gross_all"] == -1.0
    assert res.gross_all_ci95 == (-1.0, -1.0)
    assert res.admitted


def test_redundant_candidate_is_not_admitted():
    # a and b already agree on the truth: c adds nothing.
    frames = [frame(a=10, b=10, c=10) for _ in range(10)]
    res = admission_test(frames, [10] * 10, ("a", "b"), "c", Scalar(), lambda e, t: abs(e - t) > 1,
                         groups=list(range(10)), n_boot=200)
    assert res.coverage == 0.0
    assert math.isnan(res.coverage_unanswered) and res.n_unanswered == 0
    assert res.value["gross_all"] == 0.0
    assert not res.admitted


def test_false_answers_block_admission():
    frames = [frame(a=10, b=50, c=10) for _ in range(10)]
    nulls = [frame(a=3, b=40, c=3) for _ in range(10)]  # c makes the pool answer on null input
    res = admission_test(frames, [10] * 10, ("a", "b"), "c", Scalar(), lambda e, t: abs(e - t) > 1,
                         nulls=nulls, groups=list(range(10)), n_boot=200)
    assert res.value["gross_all"] < 0
    assert res.value["false_answer"] == 1.0
    assert not res.admitted


def test_quorum_defaults_to_comparator_and_candidate_removed_from_pool():
    frames = [frame(a=10, b=10, c=10)]
    res = admission_test(frames, [10], ("a", "b", "c"), "c", Scalar(k=3), lambda e, t: abs(e - t) > 1)
    assert res.pool == ("a", "b")
    assert res.details["quorum"] == 3
    assert res.coverage == 1.0                        # only 2 < 3 pool lenses are right
    assert res.gross_all_ci95 is None


def test_group_bootstrap_resamples_groups():
    lo, hi = group_bootstrap(["x", "x", "y", "y"], [1, 1, 0, 0], [0, 0, 0, 0], n_boot=500)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert group_bootstrap(["x", "y"], [1, 1], [0, 0], n_boot=50) == (1.0, 1.0)
