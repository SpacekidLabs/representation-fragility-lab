"""Tests for the image medium on the core (issue #6)."""

import numpy as np
import pytest
from scipy import ndimage

from image_medium import (
    ALL_LENSES,
    LENSES,
    ImagePatch,
    OrientationComparator,
    angle_diff,
    make_image_analyzer,
)

N = 64
Y, X = np.mgrid[0:N, 0:N].astype(float) - (N - 1) / 2.0


def across(theta):
    t = np.radians(theta)
    return -X * np.sin(t) + Y * np.cos(t)


def grating(theta, period=8.0):
    return 0.5 + 0.4 * np.sin(2 * np.pi * across(theta) / period)


def edge(theta):
    return 0.3 + 0.4 / (1.0 + np.exp(-across(theta) / 0.7))


def line(theta):
    return 0.3 + 0.5 * np.exp(-0.5 * (across(theta) / 2.0) ** 2)


@pytest.mark.parametrize("make", [grating, edge, line])
@pytest.mark.parametrize("theta", [0.0, 22.0, 67.5, 90.0, 133.0, 171.0])
def test_all_five_lenses_read_a_clean_orientation(make, theta):
    report = make_image_analyzer().analyze(make(theta)).report
    for name, r in report.readings.items():
        assert abs(angle_diff(r.estimate, theta)) < 5.0, name
        assert r.confidence >= 0.5, name
    assert abs(angle_diff(report.consensus, theta)) < 5.0


def test_no_lens_is_confident_on_white_noise():
    analyzer = make_image_analyzer()
    for seed in range(20):
        patch = 0.5 + 0.2 * np.random.default_rng(seed).normal(size=(N, N))
        report = analyzer.analyze(patch).report
        assert all(r.confidence < 0.5 for r in report.readings.values())
        assert report.consensus is None


def test_fft_is_fooled_by_smooth_isotropic_noise_and_the_others_disagree():
    """A documented blind spot (Exp 044): smooth noise has a random strongest
    spectral peak, so the FFT lens reports an orientation that is not there,
    while the gradient lenses stay unconfident."""
    analyzer = make_image_analyzer()
    fooled = 0
    for seed in range(10):
        blob = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(N, N)), 2.0)
        report = analyzer.analyze(0.5 + 0.3 * blob / np.abs(blob).max()).report
        fooled += report.readings["fft"].confidence >= 0.5
        assert report.readings["tensor"].confidence < 0.5
        assert report.readings["hog"].confidence < 0.5
    assert fooled >= 8


def test_angle_diff_wraps_mod_180():
    assert angle_diff(179.0, 1.0) == pytest.approx(-2.0)
    assert angle_diff(1.0, 179.0) == pytest.approx(2.0)
    assert angle_diff(90.0, 0.0) == pytest.approx(90.0)
    assert angle_diff(0.0, 90.0) == pytest.approx(90.0)
    assert angle_diff(45.0, 45.0 + 180.0) == pytest.approx(0.0)


def test_orientation_comparator_relations_and_fusion():
    from core import Reading

    c = OrientationComparator()
    assert c.compare_pair(Reading("a", 178.0, 1), Reading("b", 3.0, 1)).relation == "agree"
    assert c.compare_pair(Reading("a", 10.0, 1), Reading("b", 95.0, 1)).relation == "orthogonal"
    assert c.compare_pair(Reading("a", 10.0, 1), Reading("b", 50.0, 1)).relation == "off"
    assert c.compare_pair(Reading("a", 10.0, 1), Reading("b", 100.0, 1)).distance == pytest.approx(1.0)
    # the consensus is a circular median: 178 and 2 do not average to 90
    fused = c.fuse([Reading("a", 178.0, 0.9), Reading("b", 2.0, 0.8), Reading("c", 179.0, 0.7)])
    assert abs(angle_diff(fused, 179.0)) < 1e-9
    # a minority cannot pull the consensus away
    fused = c.fuse([Reading("a", 30.0, 0.9), Reading("b", 34.0, 0.9), Reading("c", 100.0, 0.95)])
    assert fused in (30.0, 34.0)
    # exactly orthogonal minority (the case that broke an unwrap-around-the-top rule)
    fused = c.fuse([Reading("a", 30.0, 0.9), Reading("b", 31.0, 0.9), Reading("c", 120.0, 0.95)])
    assert fused in (30.0, 31.0)


def test_image_medium_validation():
    analyzer = make_image_analyzer()
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros(100))
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((8, 8)))
    assert analyzer.analyze(np.full((N, N), 0.5)).empty
    rgb = np.repeat(grating(30.0)[:, :, None], 3, axis=2)
    assert abs(angle_diff(analyzer.analyze(rgb).report.consensus, 30.0)) < 5.0
    assert ImagePatch().name == "image"


def test_every_lens_states_its_assumptions():
    for name, cls in LENSES.items():
        assert cls.medium == "image"
        assert cls.assumptions, name
    assert set(ALL_LENSES) == {"tensor", "hog", "fft", "gabor", "radon"}


def test_images_and_audio_share_the_same_core():
    from framework import make_audio_analyzer

    image, audio = make_image_analyzer(), make_audio_analyzer()
    assert type(image) is type(audio)
    assert type(image.detector) is type(audio.detector)
    assert type(image.comparator).compare is type(audio.comparator).compare  # generic aggregation
