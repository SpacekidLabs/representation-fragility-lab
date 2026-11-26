"""
Image medium on the medium-agnostic core (issue #6)
===================================================
The second medium built on ``src/core``. The comparable quantity is the
**dominant orientation** of a grayscale patch: the direction, in degrees in
[0, 180), along which the stripes, edges or lines run. 0° = along the columns
axis (horizontal in the array), angles grow towards increasing row index.

- ``ImagePatch``               the Medium: a 2-D (or H×W×C) array, values ~[0, 1]
- ``TensorOrientation``        structure tensor (mean of squared gradients)
- ``HogOrientation``           mode of the gradient-orientation histogram
- ``FftOrientation``           strongest peak of the 2-D power spectrum
- ``GaborOrientation``         energy of an oriented Gabor (wavelet) bank
- ``RadonOrientation``         variance of projections along each angle
- ``OrientationComparator``    angular distance, "agree" / "orthogonal" / "off",
                               confidence-weighted circular median as consensus
- ``make_image_analyzer``      a ready-made ``core.Analyzer``

See docs/core_interfaces.md for how the five lenses map onto the core and
src/experiments/exp044_image_disagreement.py for the experiment.
"""

from .medium import (
    ALL_LENSES,
    LENSES,
    FftOrientation,
    GaborOrientation,
    HogOrientation,
    ImagePatch,
    OrientationComparator,
    RadonOrientation,
    TensorOrientation,
    angle_diff,
    make_image_analyzer,
)

__all__ = [
    "ALL_LENSES",
    "LENSES",
    "FftOrientation",
    "GaborOrientation",
    "HogOrientation",
    "ImagePatch",
    "OrientationComparator",
    "RadonOrientation",
    "TensorOrientation",
    "angle_diff",
    "make_image_analyzer",
]
