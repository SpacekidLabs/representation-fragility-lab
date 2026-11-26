"""
Experiment 044 — Images: Blind Spots, Disagreement and Patterns (issue #6)
=========================================================================
The audio arc (Exp 001–013 atlas → Exp 042 disagreement → Exp 043 patterns) in
miniature, on a second medium, with the same core code (src/core). The only
image-specific code is src/image_medium: a Medium, five orientation lenses and
an orientation distance.

Quantity: the dominant orientation of a 64×64 grayscale patch, in degrees mod 180.
Gross error: more than 10° from the truth.

Lenses (src/image_medium):
  tensor  structure tensor (mean of squared gradients)
  hog     mode of the gradient-orientation histogram
  fft     strongest 2-D power-spectrum peak
  gabor   oriented Gabor (wavelet) bank energy, wavelengths 4–16 px
  radon   variance of line sums per angle

Signals with known orientation: gratings (period 6 and 12 px), step edges,
thin lines (width 1.5 and 3 px), oriented textures (random gratings within ±5°)
and two-orientation mixtures are not used; the truth is always one orientation.

Perturbations: Gaussian noise, Gaussian blur, JPEG-like block DCT quantisation,
low contrast with 8-bit quantisation, occlusion by a flat rectangle, and a
weaker second pattern at another orientation.

Questions:
  1. Atlas: does each lens have its own, structured blind spots?
  2. Prediction: does disagreement predict when a lens is wrong (ROC AUC,
     grouped cross-validation), better than the lens's own confidence?
  3. Answer: is the consensus better than any single lens?
  4. Patterns: which lenses fail together (co-failure lift), and do clusters
     of disagreement vectors line up with what actually went wrong?
  5. Unoriented input (isotropic noise): how often does a lens, or the
     consensus, confidently report an orientation that is not there?

Usage:
  python src/experiments/exp044_image_disagreement.py              # run
  python src/experiments/exp044_image_disagreement.py --calibrate  # print lens calibration

Outputs:
  results/exp044_image_disagreement.png
  results/exp044_summary.json
"""

import json
import os
import sys
import warnings
import zlib
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (project_root, os.path.join(project_root, "src")):
    if p not in sys.path:
        sys.path.append(p)

warnings.filterwarnings("ignore")

from image_medium import ALL_LENSES, LENSES, angle_diff, make_image_analyzer  # noqa: E402

N = 64
GROSS_DEG = 10.0
ORIENTATIONS = 12                     # base orientations, 15° apart, each with a random jitter
SEEDS = (0, 1)
UNORIENTED_PATCHES = 90
REPS = ALL_LENSES
PAIRS = list(combinations(REPS, 2))
K_RANGE = range(3, 11)

OUT_PNG = os.path.join(project_root, "results", "exp044_image_disagreement.png")
OUT_JSON = os.path.join(project_root, "results", "exp044_summary.json")


# ──────────────────────────────────────────────────────────────────────────────
# SIGNALS (orientation θ in degrees: the direction the structure runs along,
# in (x = column, y = row) coordinates)
# ──────────────────────────────────────────────────────────────────────────────

_Y, _X = np.mgrid[0:N, 0:N].astype(float)
_X -= (N - 1) / 2.0
_Y -= (N - 1) / 2.0


def _across(theta):
    """Coordinate across the structure (along its normal)."""
    t = np.radians(theta)
    return -_X * np.sin(t) + _Y * np.cos(t)


def grating(theta, period, rng):
    return 0.5 + 0.4 * np.sin(2 * np.pi * _across(theta) / period + rng.uniform(0, 2 * np.pi))


def edge(theta, rng):
    off = rng.uniform(-8, 8)
    return 0.3 + 0.4 * (1.0 / (1.0 + np.exp(-(_across(theta) - off) / 0.7)))


def line(theta, width, rng):
    off = rng.uniform(-8, 8)
    return 0.3 + 0.5 * np.exp(-0.5 * ((_across(theta) - off) / width) ** 2)


def texture(theta, rng):
    x = np.zeros((N, N))
    for _ in range(12):
        x += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * _across(theta + rng.uniform(-5, 5)) / rng.uniform(4, 16)
                                            + rng.uniform(0, 2 * np.pi))
    return 0.5 + 0.4 * x / np.max(np.abs(x))


VARIANTS = {
    "grating6": lambda th, rng: grating(th, 6.0, rng),
    "grating12": lambda th, rng: grating(th, 12.0, rng),
    "edge": edge,
    "line1.5": lambda th, rng: line(th, 1.5, rng),
    "line3": lambda th, rng: line(th, 3.0, rng),
    "texture": texture,
}
SIGNAL_TYPE = {"grating6": "grating", "grating12": "grating", "edge": "edge",
               "line1.5": "line", "line3": "line", "texture": "texture"}


def isotropic(rng, kind):
    if kind == "white":
        return np.clip(0.5 + 0.2 * rng.normal(size=(N, N)), 0, 1)
    blob = ndimage.gaussian_filter(rng.normal(size=(N, N)), 2.0)
    return 0.5 + 0.3 * blob / np.max(np.abs(blob))


# ──────────────────────────────────────────────────────────────────────────────
# PERTURBATIONS (none of them changes the true orientation)
# ──────────────────────────────────────────────────────────────────────────────

_JPEG_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], float)


def jpeg_like(x, quality):
    """8×8 block DCT quantisation with the standard luminance table (IJG quality scaling)."""
    s = 5000.0 / quality if quality < 50 else 200.0 - 2 * quality
    q = np.maximum(np.floor((_JPEG_Q * s + 50) / 100), 1)
    img = np.round(np.clip(x, 0, 1) * 255) - 128
    out = np.empty_like(img)
    for i in range(0, N, 8):
        for j in range(0, N, 8):
            c = dctn(img[i:i + 8, j:j + 8], norm="ortho")
            out[i:i + 8, j:j + 8] = idctn(np.round(c / q) * q, norm="ortho")
    return np.clip(out + 128, 0, 255) / 255.0


def occlude(x, fraction, rng):
    side_h = int(round(N * np.sqrt(fraction) * rng.uniform(0.8, 1.25)))
    side_w = int(round(N * N * fraction / max(side_h, 1)))
    side_h, side_w = min(side_h, N), min(side_w, N)
    i, j = rng.integers(0, N - side_h + 1), rng.integers(0, N - side_w + 1)
    y = x.copy()
    y[i:i + side_h, j:j + side_w] = rng.uniform(0.1, 0.9)
    return y


def low_contrast(x, factor):
    return np.round((0.5 + factor * (x - 0.5)) * 255) / 255.0


def second_pattern(x, theta, strength, rng):
    other = theta + rng.uniform(30, 90) * rng.choice([-1, 1])
    return x + strength * (grating(other, rng.uniform(5, 12), rng) - 0.5)


CONDITIONS = [
    ("clean", 0, lambda x, th, rng: x),
    ("noise", 0.1, lambda x, th, rng: x + rng.normal(0, 0.1, x.shape)),
    ("noise", 0.3, lambda x, th, rng: x + rng.normal(0, 0.3, x.shape)),
    ("noise", 0.6, lambda x, th, rng: x + rng.normal(0, 0.6, x.shape)),
    ("blur", 1.0, lambda x, th, rng: ndimage.gaussian_filter(x, 1.0)),
    ("blur", 2.5, lambda x, th, rng: ndimage.gaussian_filter(x, 2.5)),
    ("blur", 4.0, lambda x, th, rng: ndimage.gaussian_filter(x, 4.0)),
    ("jpeg", 50, lambda x, th, rng: jpeg_like(x, 50)),
    ("jpeg", 20, lambda x, th, rng: jpeg_like(x, 20)),
    ("jpeg", 5, lambda x, th, rng: jpeg_like(x, 5)),
    ("contrast", 0.05, lambda x, th, rng: low_contrast(x, 0.05)),
    ("contrast", 0.01, lambda x, th, rng: low_contrast(x, 0.01)),
    ("occlusion", 0.25, lambda x, th, rng: occlude(x, 0.25, rng)),
    ("occlusion", 0.5, lambda x, th, rng: occlude(x, 0.5, rng)),
    ("second", 0.4, lambda x, th, rng: second_pattern(x, th, 0.4, rng)),
    ("second", 0.8, lambda x, th, rng: second_pattern(x, th, 0.8, rng)),
]


def base_signals():
    """(variant, orientation index, theta, rng-seeded patch maker)."""
    for variant in VARIANTS:
        for k in range(ORIENTATIONS):
            rng = np.random.default_rng(zlib.crc32(f"theta-{variant}-{k}".encode()))
            yield variant, k, (15.0 * k + rng.uniform(-7.5, 7.5)) % 180.0


def make_patch(variant, theta, family, severity, seed, k):
    rng = np.random.default_rng(zlib.crc32(repr((variant, k, family, severity, seed)).encode()))
    x = VARIANTS[variant](theta, rng)
    fn = next(f for fam, sev, f in CONDITIONS if fam == family and sev == severity)
    return fn(x, theta, rng)
