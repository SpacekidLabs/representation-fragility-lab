"""Tests for the time-series (sensor rate) medium on the unchanged core (issue #27)."""

import numpy as np
import pytest

from core import Reading
from timeseries_medium import (
    ALL_LENSES,
    LENSES,
    RateComparator,
    SensorWindow,
    is_gross,
    make_timeseries_analyzer,
)

FS = 360
T = np.arange(8 * FS) / FS


def ecg(bpm, t_amp=0.3):
    rr = 60.0 / bpm
    x = np.zeros_like(T)
    for b in np.arange(-1, 8 / rr + 1) * rr:
        for a, mu, w in ((0.15, -0.2, 0.025), (-0.1, -0.025, 0.01), (1.0, 0.0, 0.01),
                         (-0.25, 0.025, 0.01), (t_amp, 0.25 * np.sqrt(rr), 0.06)):
            x += a * np.exp(-0.5 * ((T - b - mu) / w) ** 2)
    return x


@pytest.mark.parametrize("bpm", [60.0, 90.0, 120.0, 170.0])
def test_all_four_lenses_read_a_clean_ecg(bpm):
    report = make_timeseries_analyzer().analyze(ecg(bpm), {"fs": FS}).report
    for name, r in report.readings.items():
        assert not is_gross(r.estimate, bpm), name
    assert not is_gross(report.consensus, bpm)


def test_peak_counter_is_confident_and_the_rest_are_not_on_white_noise():
    analyzer = make_timeseries_analyzer()
    for seed in range(100, 110):
        report = analyzer.analyze(np.random.default_rng(seed).normal(size=len(T)), {"fs": FS}).report
        for n in ("fft", "acf", "peaks"):
            assert report.readings[n].confidence < 0.5, n
        assert report.consensus is None


def test_rate_comparator_relations_and_fusion():
    comp = RateComparator()
    assert comp.compare_pair(Reading("a", 60.0, 1.0), Reading("b", 63.0, 1.0)).relation == "agree"
    assert comp.compare_pair(Reading("a", 120.0, 1.0), Reading("b", 60.0, 1.0)).relation == "double"
    assert comp.compare_pair(Reading("a", 30.0, 1.0), Reading("b", 61.0, 1.0)).relation == "half"
    assert comp.compare_pair(Reading("a", 60.0, 1.0), Reading("b", 90.0, 1.0)).relation == "off"
    fused = comp.fuse([Reading("a", 60.0, 0.9), Reading("b", 61.0, 0.8), Reading("c", 120.0, 0.2)])
    assert fused in (60.0, 61.0)


def test_timeseries_medium_validation():
    m = SensorWindow()
    with pytest.raises(ValueError):
        m.validate(np.zeros(8 * FS), {})                      # no sample rate
    with pytest.raises(ValueError):
        m.validate(np.zeros(FS), {"fs": FS})                  # shorter than 4 s
    assert m.is_empty(np.zeros(8 * FS), {"fs": FS})
    assert make_timeseries_analyzer().analyze(np.zeros(8 * FS), {"fs": FS}).empty


def test_every_lens_states_its_assumptions():
    for name, cls in LENSES.items():
        assert cls.medium == "timeseries"
        assert cls.assumptions, name
    assert set(ALL_LENSES) == {"fft", "acf", "wavelet", "peaks"}


def test_three_media_share_the_same_core():
    from framework import make_audio_analyzer
    from image_medium import make_image_analyzer

    ts, audio, image = make_timeseries_analyzer(), make_audio_analyzer(), make_image_analyzer()
    assert type(ts) is type(audio) is type(image)
    assert type(ts.comparator).compare is type(audio.comparator).compare is type(image.comparator).compare


def test_a_second_sharp_wave_per_beat_breaks_the_peak_counter_and_disagreement_flags_it():
    """A stated blind spot: a second event as sharp as the R wave (here 250 ms after it)
    makes the intervals alternate, so the peak counter loses its confidence; the
    comparator names it as the outlier and the consensus stays right."""
    x = ecg(60.0, t_amp=1.0)
    x += np.exp(-0.5 * ((T[:, None] - (np.arange(-1, 9) + 0.25)[None]) / 0.01) ** 2).sum(1)
    report = make_timeseries_analyzer().analyze(x, {"fs": FS}).report
    assert report.readings["peaks"].confidence < 0.5
    assert report.outlier == "peaks"
    assert not is_gross(report.consensus, 60.0)
