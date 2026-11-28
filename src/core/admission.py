"""
Admission test for a new lens (issue #32, from Exp 055)
=======================================================
Exp 055 found that what a lens is worth to a quorum follows its **coverage**,
not how much its blind spots overlap the other lenses' (Spearman +0.79 vs ~0).
This module turns that into a check any medium can run on cached readings
before adding a lens to its pool.

Definitions (k = the quorum size, ``min_agreeing`` of the comparator, default 2)
  coverage              share of frames where the candidate is right and fewer
                        than k lenses of the pool are: the frames where only the
                        candidate can complete a right quorum
  coverage_unanswered   share of the frames the pool abstains on where the
                        candidate is right
  value                 what adding the candidate changes, with the comparator
                        run on the pool and on pool + candidate:
                          gross_all     wrong answers + abstentions, over all frames
                          gross_answer  wrong answers among answered frames
                          abstention    share of frames left unanswered
                          false_answer  share of null frames (no true answer)
                                        that get an answer
                        Each is reported as ``with - without``: a negative
                        gross_all means the candidate helps.
  gross_all_ci95        95% CI of the gross_all change from a group bootstrap
                        (whole recordings / images resampled), when groups are given

A lens is *admitted* when the upper end of the gross_all CI is below zero (or,
without groups, when the change itself is below zero) and the false-answer rate
does not rise by more than ``max_false_increase``.

Standard library only, like the rest of ``core``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Sequence

from .interfaces import Comparator, Reading

Frame = Mapping[str, Reading]


@dataclass
class Admission:
    candidate: str
    pool: tuple[str, ...]
    n: int
    n_null: int
    coverage: float
    coverage_unanswered: float
    n_unanswered: int
    candidate_gross: float
    without: dict[str, float]
    with_candidate: dict[str, float]
    value: dict[str, float]
    gross_all_ci95: tuple[float, float] | None
    admitted: bool
    details: dict[str, Any] = field(default_factory=dict)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _run(comparator: Comparator, frames: Sequence[Frame], lenses: Sequence[str]):
    return [comparator.compare({n: f[n] for n in lenses}).consensus for f in frames]


def _metrics(answers, truths, is_wrong, null_answers):
    answered = [a is not None for a in answers]
    wrong = [a is None or is_wrong(a, t) for a, t in zip(answers, truths)]
    wrong_ans = [w for w, a in zip(wrong, answered) if a]
    return wrong, {
        "gross_all": _mean([float(w) for w in wrong]),
        "gross_answer": _mean([float(w) for w in wrong_ans]),
        "abstention": 1.0 - _mean([float(a) for a in answered]),
        "false_answer": _mean([float(a is not None) for a in null_answers]) if null_answers else float("nan"),
    }, answered


def group_bootstrap(groups: Sequence[Hashable], a: Sequence[bool], b: Sequence[bool],
                    n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(a) - mean(b), resampling whole groups."""
    index: dict[Hashable, int] = {}
    for g in groups:
        index.setdefault(g, len(index))
    m = len(index)
    sa, sb, sn = [0.0] * m, [0.0] * m, [0] * m
    for g, x, y in zip(groups, a, b):
        i = index[g]
        sa[i] += float(x)
        sb[i] += float(y)
        sn[i] += 1
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ta = tb = tn = 0.0
        for _ in range(m):
            i = rng.randrange(m)
            ta += sa[i]
            tb += sb[i]
            tn += sn[i]
        diffs.append((ta - tb) / tn)
    diffs.sort()
    lo = diffs[int(0.025 * (n_boot - 1))]
    hi = diffs[int(round(0.975 * (n_boot - 1)))]
    return float(lo), float(hi)


def admission_test(frames: Sequence[Frame], truths: Sequence[Any], pool: Sequence[str], candidate: str,
                   comparator: Comparator, is_wrong: Callable[[Any, Any], bool],
                   nulls: Sequence[Frame] = (), groups: Sequence[Hashable] | None = None,
                   quorum: int | None = None, max_false_increase: float = 0.02,
                   n_boot: int = 2000, seed: int = 0) -> Admission:
    """Coverage and value of ``candidate`` for ``pool``, from cached readings.

    frames      one mapping lens name → Reading per frame (must hold every pool lens
                and the candidate)
    truths      the true answer per frame
    is_wrong    is_wrong(estimate, truth) for a non-None estimate (gross error)
    nulls       frames with no true answer (unpitched / unoriented), for false answers
    groups      recording / image of each frame, for the bootstrap CI
    quorum      lenses needed to answer; defaults to ``comparator.min_agreeing`` (or 2)
    """
    pool = tuple(n for n in pool if n != candidate)
    if len(frames) != len(truths):
        raise ValueError("frames and truths differ in length")
    if quorum is None:
        quorum = getattr(comparator, "min_agreeing", None) or 2

    def right(name, frame, truth):
        est = frame[name].estimate
        return est is not None and not is_wrong(est, truth)

    cand_right = [right(candidate, f, t) for f, t in zip(frames, truths)]
    n_pool_right = [sum(right(n, f, t) for n in pool) for f, t in zip(frames, truths)]
    coverage = _mean([float(c and k < quorum) for c, k in zip(cand_right, n_pool_right)])

    base_ans = _run(comparator, frames, pool)
    with_ans = _run(comparator, frames, pool + (candidate,))
    base_null = _run(comparator, nulls, pool) if nulls else []
    with_null = _run(comparator, nulls, pool + (candidate,)) if nulls else []
    w0, m0, answered0 = _metrics(base_ans, truths, is_wrong, base_null)
    w1, m1, _ = _metrics(with_ans, truths, is_wrong, with_null)

    unanswered = [c for c, a in zip(cand_right, answered0) if not a]
    value = {k: m1[k] - m0[k] for k in m0}
    ci = group_bootstrap(groups, w1, w0, n_boot, seed) if groups is not None else None
    helps = (ci[1] < 0) if ci is not None else (value["gross_all"] < 0)
    false_ok = not (value["false_answer"] > max_false_increase)   # NaN (no nulls) passes
    return Admission(
        candidate=candidate, pool=pool, n=len(frames), n_null=len(nulls),
        coverage=coverage,
        coverage_unanswered=_mean([float(c) for c in unanswered]),
        n_unanswered=len(unanswered),
        candidate_gross=1.0 - _mean([float(c) for c in cand_right]),
        without=m0, with_candidate=m1, value=value, gross_all_ci95=ci,
        admitted=bool(helps and false_ok),
        details={"quorum": quorum, "max_false_increase": max_false_increase,
                 "wrong_without": w0, "wrong_with": w1},
    )
