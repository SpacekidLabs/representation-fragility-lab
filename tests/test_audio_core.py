"""Tests for the audio medium on the core (issue #5)."""

import numpy as np
import pytest

from framework import ALL_LENSES, AudioFrame, make_audio_analyzer
from framework.audio import LENSES, PitchComparator
from framework.disagreement import analyze_disagreement

SR = 22050
N = 2048


def tone(f0, harmonics, n=N, phase_seed=None):
    rng = np.random.default_rng(phase_seed)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t + (rng.uniform(0, 2 * np.pi) if phase_seed is not None else 0.0)) / k
            for k in harmonics)
    return x / np.max(np.abs(x))


@pytest.mark.parametrize("f0", [110.0, 220.0, 440.0, 660.0])
def test_all_five_lenses_find_a_clean_pitch(f0):
    report = make_audio_analyzer(ALL_LENSES).analyze(tone(f0, range(1, 11)), {"sr": SR}).report
    for name, r in report.readings.items():
        assert r.estimate == pytest.approx(f0, rel=0.01), name
        assert r.confidence >= 0.5, name
    assert report.outlier is None
    assert report.consensus == pytest.approx(f0, rel=0.01)


def test_new_lenses_are_not_confident_on_noise():
    analyzer = make_audio_analyzer(("cqt", "wavelet"))
    for seed in range(20):
        frame = np.random.default_rng(seed).normal(0.0, 0.3, N)
        report = analyzer.analyze(frame, {"sr": SR}).report
        assert all(r.confidence < 0.5 for r in report.readings.values())
        assert report.consensus is None


def test_spectral_lenses_share_the_missing_fundamental_blind_spot():
    """Lenses are not independent votes (Exp 043): STFT, CQT and wavelet all jump
    an octave up when the fundamental is missing. With the default three lenses
    ACF + Cepstrum outvote STFT; with all five, the spectral trio wins."""
    frame = tone(220.0, range(2, 11))
    full = make_audio_analyzer(ALL_LENSES).analyze(frame, {"sr": SR}).report
    for name in ("stft", "cqt", "wavelet"):
        assert full.readings[name].estimate == pytest.approx(440.0, rel=0.01), name
    for name in ("acf", "cepstrum"):
        assert full.readings[name].estimate == pytest.approx(220.0, rel=0.01), name
    assert full.consensus == pytest.approx(440.0, rel=0.01)  # wrong: shared blind spot

    default = make_audio_analyzer().analyze(frame, {"sr": SR}).report
    assert default.consensus == pytest.approx(220.0, rel=0.01)
    assert default.pair("stft", "acf").relation == "octave above"


def test_legacy_api_matches_core_analyzer():
    frame = tone(220.0, range(1, 11), phase_seed=3) + np.random.default_rng(3).normal(0, 0.3, N)
    legacy = analyze_disagreement(frame, SR)
    core = make_audio_analyzer().analyze(frame, {"sr": SR}).report
    assert legacy.overall == core.overall
    assert legacy.divergence == core.divergence
    assert legacy.outlier == core.outlier
    assert legacy.consensus_f0 == (core.consensus or 0.0)
    for p in legacy.pairs:
        q = core.pair(p.a, p.b)
        assert p.score == q.score and p.pitch_term == q.distance


def test_legacy_api_accepts_new_lenses():
    report = analyze_disagreement(tone(330.0, range(1, 8)), SR, representations=ALL_LENSES)
    assert set(report.readings) == set(ALL_LENSES)
    assert len(report.pairs) == 10


def test_audio_medium_validation():
    analyzer = make_audio_analyzer()
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros(0), {"sr": SR})
    with pytest.raises(ValueError):
        analyzer.analyze(tone(220.0, [1]), {})  # no sample rate
    assert analyzer.analyze(np.zeros(N), {"sr": SR}).empty
    assert AudioFrame().name == "audio"


def test_every_lens_states_its_assumptions():
    for name, cls in LENSES.items():
        assert cls.medium == "audio"
        assert cls.assumptions, name


def test_pitch_comparator_relations():
    from core import Reading

    c = PitchComparator()
    assert c.compare_pair(Reading("a", 440.0, 1), Reading("b", 220.0, 1)).relation == "octave above"
    assert c.compare_pair(Reading("a", 110.0, 1), Reading("b", 220.0, 1)).relation == "octave below"
    assert c.compare_pair(Reading("a", 221.0, 1), Reading("b", 220.0, 1)).relation == "agree"
    assert c.compare_pair(Reading("a", 300.0, 1), Reading("b", 220.0, 1)).relation == "off"


def test_one_confident_lens_does_not_make_a_pitch():
    """Issue #20: on unvoiced audio one lens is often wrongly sure (Exp 045)."""
    from core import Reading

    rd = {"stft": Reading("stft", None, 0.1), "acf": Reading("acf", 180.0, 0.8),
          "cepstrum": Reading("cepstrum", 310.0, 0.3)}
    assert PitchComparator().compare(rd).consensus is None
    assert PitchComparator(min_agreeing=None).compare(rd).consensus == 180.0
    # an octave relation is not agreement
    rd["cepstrum"] = Reading("cepstrum", 360.0, 0.8)
    assert PitchComparator().compare(rd).consensus is None


def test_family_votes_undo_the_spectral_trio_on_a_missing_fundamental():
    """Issue #21: when STFT, CQT and wavelet are known to fail together, the five-lens
    consensus follows ACF + Cepstrum again on a missing fundamental."""
    from itertools import combinations

    from core import ConfidenceThresholdDetector, FamilyVotes
    from framework.audio import PitchComparator
    from framework.disagreement import VOICING_THRESHOLD

    spectral = {"stft", "cqt", "wavelet"}
    corr = {frozenset(p): (0.9 if set(p) <= spectral else 0.0) for p in combinations(ALL_LENSES, 2)}
    fv = FamilyVotes({n: [0.8] * 5 for n in ALL_LENSES}, corr)
    frame = tone(220.0, range(2, 11))
    plain = make_audio_analyzer(ALL_LENSES).analyze(frame, {"sr": SR}).report
    comp = PitchComparator(detector=ConfidenceThresholdDetector(VOICING_THRESHOLD), family=fv)
    fused = comp.compare(plain.readings)
    assert set(fused.consensus_from) <= {"acf", "cepstrum"}
    assert abs(1200 * np.log2(fused.consensus / 220.0)) < 50


def test_lowpass_probe_follows_the_true_pitch_not_its_octave():
    """Issue #30: cut at 1.5 x the candidate; the lenses still read the true f0,
    but cutting at 1.5 x the octave above leaves the fundamental, so they do not read 2 f0."""
    from core import Prober

    x = tone(200.0, range(1, 6))
    an = make_audio_analyzer(lenses=("stft", "acf", "cepstrum"), prober=Prober())
    ctx = {"sr": SR}
    for candidate, follows in ((200.0, 3), (400.0, 0)):
        (probe,) = an.medium.probes(x, candidate, ctx)
        rd = [r.read(probe.signal, ctx) for r in an.representations]
        target = type(rd[0])("probe", probe.predicted, 1.0)
        n = sum(an.comparator.agrees(an.comparator.compare_pair(r, target)) for r in rd)
        assert n == follows, (candidate, [r.estimate for r in rd])
