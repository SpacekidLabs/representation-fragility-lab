"""
Framework Reference Plugins Package
===================================
Exposes the three coordinate-gated adaptive reference plugins:
1. AdaptiveDenoiser
2. AdaptivePitchTracker
3. AdaptiveOnsetDetector
"""

from .denoiser import AdaptiveDenoiser
from .pitch_tracker import AdaptivePitchTracker
from .onset_detector import AdaptiveOnsetDetector

__all__ = [
    "AdaptiveDenoiser",
    "AdaptivePitchTracker",
    "AdaptiveOnsetDetector"
]
