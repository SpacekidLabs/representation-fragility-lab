"""Regression tests for AdaptivePitchTracker getting stuck (issue #10).

Before the fix, a fixed 220 Hz starting anchor plus a 3-semitone continuity
limit kept the tracker stuck at the old pitch whenever the true pitch was (or
moved) more than 3 semitones away.
"""

import warnings

import numpy as np
import pytest

from framework.plugins.pitch_tracker import REACQUIRE_FRAMES, AdaptivePitchTracker

SR = 22050
HOP = 512


def harmonic(track):
    phase = 2 * np.pi * np.cumsum(track) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 5))
    return x / np.max(np.abs(x))


def track(f0_track):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = AdaptivePitchTracker().track(harmonic(f0_track), SR, hop_length=HOP)
    truth = f0_track[np.arange(0, len(f0_track) - HOP, HOP)]
    return est, truth


def off(est, truth):
    return np.abs(est - truth) / truth > 0.05


@pytest.fixture(scope="module")
def tracker_runs():
    n = SR  # 1 s each
    steady150 = np.full(n, 150.0)
    step = np.where(np.arange(n) < n // 2, 220.0, 330.0)       # +7 semitones
    leap = np.where(np.arange(n) < n // 2, 220.0, 440.0)       # +1 octave
    return {name: track(f) for name, f in
            (("steady150", steady150), ("step", step), ("leap", leap))}


def test_starts_far_from_220_hz_without_getting_stuck(tracker_runs):
    est, truth = tracker_runs["steady150"]
    assert not off(est, truth)[2:].any()  # allow the first frames to settle


@pytest.mark.parametrize("name", ["step", "leap"])
def test_follows_a_sustained_jump_after_reacquisition(tracker_runs, name):
    est, truth = tracker_runs[name]
    change = int(np.argmax(truth != truth[0]))
    settle = change + REACQUIRE_FRAMES + 3  # re-acquire, then the median filter catches up
    assert not off(est, truth)[2:change - 2].any()
    assert not off(est, truth)[settle:].any()


def test_quorum_tracker_follows_a_note_change_without_the_median_delay():
    """Exp 054 (issue #28): the adaptive tracker's causal 5-hop median delays a note
    change; the quorum tracker has no filter and follows on the next hop."""
    import numpy as np

    from framework import AdaptivePitchTracker, QuorumPitchTracker

    sr = 22050
    t = np.arange(sr) / sr
    f = np.where(t < 0.5, 220.0, 330.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    q = QuorumPitchTracker().track(x, sr)
    a = AdaptivePitchTracker().track(x, sr)
    k = int(np.searchsorted(np.arange(len(q)) * 512 / sr, 0.5))
    cents = lambda e: np.abs(1200 * np.log2(np.maximum(e, 1e-9) / 330.0))  # noqa: E731
    assert np.all(cents(q[k + 1:k + 10]) < 50)
    assert np.all(cents(a[k + 1:k + 3]) > 50)
    assert np.all(np.abs(1200 * np.log2(q[3:k - 2] / 220.0)) < 50)
