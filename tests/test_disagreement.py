"""Tests for cross-representation disagreement (issue #1)."""

import numpy as np
import pytest

from framework.disagreement import (
    DisagreementReport,
    RepresentationReading,
    analyze_disagreement,
    compare_pair,
    compare_readings,
)

SR = 16000
N = 2048
F0 = 220.0


def harmonic_frame(f0=F0, num_harmonics=15, n=N, sr=SR):
    t = np.arange(n) / sr
    x = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, num_harmonics + 1))
    return x / np.max(np.abs(x))


def noise_frame(seed, n=N):
    return np.random.default_rng(seed).normal(0.0, 0.3, n)


# ---------------------------------------------------------------------------
# Comparator (pure, deterministic)
# ---------------------------------------------------------------------------

def test_identical_readings_do_not_disagree():
    a = RepresentationReading("stft", 220.0, 0.9)
    b = RepresentationReading("acf", 220.0, 0.9)
    p = compare_pair(a, b)
    assert p.cents == pytest.approx(0.0)
    assert p.score == pytest.approx(0.0)
    assert not p.voicing_conflict
    assert not p.octave_related


def test_octave_error_is_flagged_and_maximal():
    p = compare_pair(RepresentationReading("acf", 220.0, 0.9),
                     RepresentationReading("cepstrum", 110.0, 0.9))
    assert p.cents == pytest.approx(1200.0)
    assert p.octave_related
    assert p.pitch_term == pytest.approx(1.0)


def test_missing_estimate_counts_as_full_pitch_disagreement():
    p = compare_pair(RepresentationReading("stft", 220.0, 0.9),
                     RepresentationReading("acf", 0.0, 0.0))
    assert p.cents is None
    assert p.pitch_term == 1.0
    assert p.voicing_conflict


def test_outlier_and_consensus():
    report = compare_readings({
        "stft": RepresentationReading("stft", 220.0, 0.95),
        "acf": RepresentationReading("acf", 221.0, 0.90),
        "cepstrum": RepresentationReading("cepstrum", 110.0, 0.20),
    })
    assert report.outlier == "cepstrum"
    assert report.divergence["cepstrum"] > report.divergence["stft"]
    assert 219.0 <= report.consensus_f0 <= 222.0  # cepstrum is unvoiced, so excluded
    assert report.pair("cepstrum", "acf").octave_related


# ---------------------------------------------------------------------------
# Readers on synthetic signals
# ---------------------------------------------------------------------------

def test_clean_harmonic_all_lenses_agree():
    report = analyze_disagreement(harmonic_frame(), SR)
    for name, r in report.readings.items():
        assert r.f0 == pytest.approx(F0, rel=0.02), name
        assert r.voiced, name
    assert report.overall < 0.15
    assert report.outlier is None
    assert report.consensus_f0 == pytest.approx(F0, rel=0.02)


def test_noise_disagrees_more_than_clean():
    clean = analyze_disagreement(harmonic_frame(), SR).overall
    reports = [analyze_disagreement(noise_frame(s), SR) for s in range(5)]
    assert np.mean([r.overall for r in reports]) > clean + 0.15
    # Every lens is blind on pure noise, so none is singled out.
    assert all(r.outlier is None for r in reports)
    assert all(r.consensus_f0 == 0.0 for r in reports)


def test_noisy_harmonic_cepstrum_breaks_first():
    rng = np.random.default_rng(9)
    report = analyze_disagreement(harmonic_frame() + rng.normal(0.0, 0.3, N), SR)
    assert report.readings["acf"].f0 == pytest.approx(F0, rel=0.02)
    assert report.readings["stft"].f0 == pytest.approx(F0, rel=0.02)
    assert report.outlier == "cepstrum"
    assert report.consensus_f0 == pytest.approx(F0, rel=0.02)


def test_sparse_harmonics_blind_the_cepstrum():
    # Two partials only (e.g. after strong low-pass filtering): the cepstrum
    # needs many harmonics, ACF and STFT do not.
    report = analyze_disagreement(harmonic_frame(num_harmonics=2), SR)
    assert report.readings["acf"].f0 == pytest.approx(F0, rel=0.02)
    assert report.readings["stft"].f0 == pytest.approx(F0, rel=0.02)
    assert max(report.divergence, key=report.divergence.get) == "cepstrum"
    assert report.overall > analyze_disagreement(harmonic_frame(), SR).overall


def test_silence_is_reported_not_compared():
    report = analyze_disagreement(np.zeros(N), SR)
    assert report.silent
    assert report.overall == 0.0
    assert report.consensus_f0 == 0.0


def test_unknown_representation_rejected():
    with pytest.raises(ValueError):
        analyze_disagreement(harmonic_frame(), SR, representations=("stft", "nope"))


def test_report_serialises():
    d = analyze_disagreement(harmonic_frame(), SR).to_dict()
    assert set(d["readings"]) == {"stft", "acf", "cepstrum"}
    assert len(d["pairs"]) == 3


# ---------------------------------------------------------------------------
# Engine API
# ---------------------------------------------------------------------------

def test_engine_exposes_disagreement():
    from framework import RepresentationIntelligenceEngine

    engine = RepresentationIntelligenceEngine()
    state = engine.analyze(harmonic_frame(), SR)
    assert isinstance(state.disagreement, DisagreementReport)
    assert engine.analyze(harmonic_frame(), SR, compute_disagreement=False).disagreement is None
