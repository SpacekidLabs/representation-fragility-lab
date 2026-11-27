"""Image patches, five orientation lenses and the orientation comparator (issue #6)."""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

try:  # installed package: representation_core
    from representation_core import (
        Analyzer,
        ConfidenceThresholdDetector,
        Medium,
        PairwiseComparator,
        Probe,
        Prober,
        Reading,
        Representation,
    )
except ImportError:  # plain checkout: src/core
    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _src not in sys.path:
        sys.path.append(_src)
    from core import (  # noqa: E402
        Analyzer,
        ConfidenceThresholdDetector,
        Medium,
        PairwiseComparator,
        Probe,
        Prober,
        Reading,
        Representation,
    )

MIN_SIZE = 16
FLAT_STD = 1e-3            # below a third of an 8-bit grey level: nothing to analyse
AGREE_DEG = 10.0           # "agree" tolerance, also the gross-error threshold in Exp 044
ORTHOGONAL_DEG = 10.0      # within this of 90° apart: "orthogonal"
VOTING_THRESHOLD = 0.5


def angle_diff(a: float, b: float) -> float:
    """Signed orientation difference a − b in (−90, 90] degrees (orientations live mod 180)."""
    d = (a - b + 90.0) % 180.0 - 90.0
    return 90.0 if d == -90.0 else d


def circular_weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted median of orientations (mod 180°): the value that minimises the
    weighted sum of angular distances to all values. The minimum is always at one
    of the inputs; ties go to the earlier input."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    cost = [np.sum(w * np.abs([angle_diff(x, c) for x in v])) for c in v]
    return float(v[int(np.argmin(cost))] % 180.0)


def _scale(raw: float, lo: float, hi: float) -> float:
    """Map a raw contrast to 0..1: ``lo`` (strongest of 60 white-noise patches) → 0,
    ``hi`` (weakest 5% of clean test patterns) → 1. See CALIBRATION below."""
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


def _parabolic_peak(values: np.ndarray, k: int) -> float:
    """Sub-bin offset of a circular peak at index k."""
    a, b, c = values[k - 1], values[k], values[(k + 1) % len(values)]
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w))


def _power_spectrum(patch: np.ndarray):
    """Hann-windowed power spectrum with the (windowed) mean removed, and the
    frequency grid in cycles/pixel: (P, fx, fy), DC at index [0, 0]."""
    win = _hann2d(*patch.shape)
    g = (patch - np.sum(win * patch) / np.sum(win)) * win
    P = np.abs(np.fft.fft2(g)) ** 2
    fy = np.fft.fftfreq(patch.shape[0])[:, None] * np.ones((1, patch.shape[1]))
    fx = np.fft.fftfreq(patch.shape[1])[None, :] * np.ones((patch.shape[0], 1))
    return P, fx, fy


# ──────────────────────────────────────────────────────────────────────────────
# MEDIUM
# ──────────────────────────────────────────────────────────────────────────────

class ImagePatch(Medium):
    """A grayscale patch (H×W, or H×W×C averaged to gray), at least 16×16, values ~[0, 1]."""

    name = "image"

    def validate(self, signal: Any, context: Mapping[str, Any]) -> np.ndarray:
        patch = np.asarray(signal, dtype=float)
        if patch.ndim == 3:
            patch = patch.mean(axis=2)
        if patch.ndim != 2:
            raise ValueError(f"Image patch must be 2-D (or H×W×C), got shape {np.shape(signal)}.")
        if min(patch.shape) < MIN_SIZE:
            raise ValueError(f"Image patch must be at least {MIN_SIZE}×{MIN_SIZE}, got {patch.shape}.")
        return patch

    def is_empty(self, signal: np.ndarray, context: Mapping[str, Any]) -> bool:
        return bool(np.std(signal) < FLAT_STD)

    #: Probes (issue #30). A Gaussian blur keeps a coarse dominant pattern and removes
    #: noise and fine, weaker detail (prediction: the candidate). A rotation by a known
    #: angle moves every real structure by that angle; ``ndimage.rotate`` by +a moves the
    #: orientation the lenses read by -a (mod 180). Reflect padding keeps the corners
    #: from adding edges.
    #: Default set: the best of five on synthetic patches, also best on held-out BSDS500 (Exp 057).
    PROBE_BLUR = (1.5, 3.0)
    PROBE_ROTATE = (30.0, 60.0)

    def probes(self, signal: np.ndarray, candidate: float, context: Mapping[str, Any]):
        out = [Probe(f"blur {s:g}", ndimage.gaussian_filter(signal, s, mode="reflect"), candidate)
               for s in self.PROBE_BLUR]
        out += [Probe(f"rotate {a:g}", ndimage.rotate(signal, a, reshape=False, mode="reflect", order=1),
                      (candidate - a) % 180.0) for a in self.PROBE_ROTATE]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# LENSES
# ──────────────────────────────────────────────────────────────────────────────

class _OrientationLens(Representation):
    medium = "image"
    contrast = (0.0, 1.0)  # (lo, hi), calibrated per lens

    def read(self, signal: np.ndarray, context: Mapping[str, Any]) -> Reading:
        theta, raw = self.orientation(signal)
        if theta is None:
            return Reading(self.name, None, 0.0, {"raw": raw})
        return Reading(self.name, float(theta % 180.0), _scale(raw, *self.contrast), {"raw": raw})

    def orientation(self, patch: np.ndarray) -> tuple[float | None, float]:
        """Return (orientation in degrees or None, raw contrast)."""
        raise NotImplementedError


def _gradients(patch: np.ndarray):
    s = ndimage.gaussian_filter(patch, 1.0)
    return ndimage.sobel(s, axis=1), ndimage.sobel(s, axis=0)  # gx (columns), gy (rows)


class TensorOrientation(_OrientationLens):
    """Structure tensor: the mean of squared gradients over the whole patch."""

    name = "tensor"
    assumptions = ("one dominant orientation in the patch",
                   "every gradient counts with its squared magnitude, so strong foreign edges pull the mean")

    def orientation(self, patch):
        gx, gy = _gradients(patch)
        jxx, jyy, jxy = np.sum(gx * gx), np.sum(gy * gy), np.sum(gx * gy)
        trace = jxx + jyy
        if trace <= 1e-12:
            return None, 0.0
        phi = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))   # gradient direction
        coherence = np.hypot(jxx - jyy, 2 * jxy) / trace
        return phi + 90.0, float(coherence)


class HogOrientation(_OrientationLens):
    """Mode of the magnitude-weighted histogram of per-pixel gradient orientations."""

    name = "hog"
    assumptions = ("most pixels with a gradient share the orientation (the mode wins, a minority is ignored)",
                   "per-pixel gradients are not swamped by noise")
    bins = 36

    def orientation(self, patch):
        gx, gy = _gradients(patch)
        mag = np.hypot(gx, gy)
        if mag.sum() <= 1e-12:
            return None, 0.0
        ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0   # structure orientation per pixel
        width = 180.0 / self.bins
        hist = np.bincount(((ang // width).astype(int) % self.bins).ravel(), weights=mag.ravel(), minlength=self.bins)
        hist = (np.roll(hist, 1) + 2 * hist + np.roll(hist, -1)) / 4.0   # circular smoothing
        k = int(np.argmax(hist))
        theta = (k + 0.5 + _parabolic_peak(hist, k)) * width
        share = sum(hist[(k + j) % self.bins] for j in range(-2, 3)) / hist.sum()
        return theta, float(share)


class FftOrientation(_OrientationLens):
    """Strongest peak of the 2-D power spectrum (Hann-windowed), refined around the peak."""

    name = "fft"
    assumptions = ("a periodic texture: one strong spectral peak away from DC",
                   "the peak is far enough from DC to resolve its angle (a lone edge or line puts it next to DC)")

    def orientation(self, patch):
        P, fx, fy = _power_spectrum(patch)
        r = np.hypot(fx * patch.shape[1], fy * patch.shape[0])  # radius in bins
        P = np.where(r > 1.5, P, 0.0)
        total = P.sum()
        if total <= 1e-12:
            return None, 0.0
        i, j = np.unravel_index(int(np.argmax(P)), P.shape)
        rows = np.arange(i - 2, i + 3) % P.shape[0]
        cols = np.arange(j - 2, j + 3) % P.shape[1]
        w = P[np.ix_(rows, cols)]
        phi2 = np.angle(np.sum(w * np.exp(2j * np.arctan2(fy[np.ix_(rows, cols)], fx[np.ix_(rows, cols)]))))
        share = 2.0 * w.sum() / total   # the peak and its mirror image
        return np.degrees(phi2) / 2.0 + 90.0, float(min(share, 1.0))


class GaborOrientation(_OrientationLens):
    """Energy of an oriented Gabor bank (wavelengths 4, 8, 16 px; 16 orientations),
    each scale normalised, summed over scales; parabolic refinement between orientations."""

    name = "gabor"
    assumptions = ("structure at the bank's scales (wavelengths of 4 to 16 pixels)",
                   "orientation between filters is interpolated (11.25° spacing)")
    wavelengths = (4.0, 8.0, 16.0)
    n_orient = 16
    sigma_deg = 15.0

    def orientation(self, patch):
        P, fx, fy = _power_spectrum(patch)
        f = np.hypot(fx, fy)
        phi = np.arctan2(fy, fx)
        energy = np.zeros(self.n_orient)
        for lam in self.wavelengths:
            f0 = 1.0 / lam
            radial = np.exp(-0.5 * ((f - f0) / (0.35 * f0)) ** 2)
            e = np.zeros(self.n_orient)
            for k in range(self.n_orient):
                d = np.angle(np.exp(2j * (phi - np.pi * k / self.n_orient))) / 2.0  # mod 180°
                G = radial * np.exp(-0.5 * (np.degrees(d) / self.sigma_deg) ** 2)
                e[k] = np.sum(P * G * G)
            if e.sum() > 1e-12:
                energy += e / e.sum()
        if energy.sum() <= 1e-12:
            return None, 0.0
        k = int(np.argmax(energy))
        phi_deg = (k + _parabolic_peak(energy, k)) * 180.0 / self.n_orient
        contrast = (energy.max() - energy.min()) / (energy.max() + energy.min())
        return phi_deg + 90.0, float(contrast)


class RadonOrientation(_OrientationLens):
    """Variance of line sums (a Radon transform inside the inscribed disk) per angle."""

    name = "radon"
    assumptions = ("long straight structures that run across the patch",
                   "line sums are not broken up by occluders")
    step_deg = 3.0

    def orientation(self, patch):
        h, w = patch.shape
        y, x = np.mgrid[0:h, 0:w]
        x = x - (w - 1) / 2.0
        y = y - (h - 1) / 2.0
        R = min(h, w) / 2.0
        disk = np.hypot(x, y) <= R
        g = patch[disk] - patch[disk].mean()
        xs, ys = x[disk], y[disk]
        if np.sum(g * g) <= 1e-12:
            return None, 0.0
        angles = np.arange(0.0, 180.0, self.step_deg)
        nb = int(2 * R) + 3
        var = np.zeros(len(angles))
        for k, a in enumerate(np.radians(angles)):
            t = -xs * np.sin(a) + ys * np.cos(a) + R + 1     # position across lines along angle a
            t0 = np.floor(t).astype(int)
            frac = t - t0
            S = (np.bincount(t0, g * (1 - frac), minlength=nb + 1)
                 + np.bincount(t0 + 1, g * frac, minlength=nb + 1))
            var[k] = np.sum(S * S)
        k = int(np.argmax(var))
        theta = (k + _parabolic_peak(var, k)) * self.step_deg
        contrast = (var.max() - var.min()) / (var.max() + var.min())
        return theta, float(contrast)


# Calibrated with src/experiments/exp044_image_disagreement.py --calibrate:
# lo = strongest raw contrast over 60 white-noise patches, hi = 5th percentile over
# clean test patterns (all signal types, all orientations).
CALIBRATION = {
    "tensor": (0.086, 0.974),
    "hog": (0.171, 0.904),
    "fft": (0.038, 0.506),
    "gabor": (0.505, 0.997),
    "radon": (0.575, 0.984),
}

LENSES = {cls.name: cls for cls in (TensorOrientation, HogOrientation, FftOrientation,
                                    GaborOrientation, RadonOrientation)}
for _name, _cls in LENSES.items():
    _cls.contrast = CALIBRATION[_name]
ALL_LENSES = tuple(LENSES)


# ──────────────────────────────────────────────────────────────────────────────
# COMPARATOR
# ──────────────────────────────────────────────────────────────────────────────

class OrientationComparator(PairwiseComparator):
    """Orientation distance and fusion; everything else is the core's generic logic."""

    def estimate_distance(self, a: Reading, b: Reading):
        signed = angle_diff(a.estimate, b.estimate)
        d = abs(signed)
        if d < AGREE_DEG:
            relation = "agree"
        elif d > 90.0 - ORTHOGONAL_DEG:
            relation = "orthogonal"
        else:
            relation = "off"
        return d / 90.0, relation, {"degrees": d, "signed_degrees": signed}

    def fuse(self, readings: Sequence[Reading]) -> float:
        """Confidence-weighted circular median (see ``circular_weighted_median``)."""
        ordered = sorted(readings, key=lambda r: -r.confidence)
        return circular_weighted_median([r.estimate for r in ordered], [r.confidence for r in ordered])


def make_image_analyzer(lenses: Sequence[str] = ALL_LENSES,
                        threshold: float = VOTING_THRESHOLD, prober: Prober | None = None) -> Analyzer:
    """A core Analyzer for image orientation with the given lenses (names from ``ALL_LENSES``)."""
    unknown = [n for n in lenses if n not in LENSES]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}. Available: {list(LENSES)}")
    detector = ConfidenceThresholdDetector(threshold)
    return Analyzer(ImagePatch(), [LENSES[n]() for n in lenses],
                    OrientationComparator(detector=detector), detector=detector, prober=prober)
