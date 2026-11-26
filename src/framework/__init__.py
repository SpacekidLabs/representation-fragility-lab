"""
Representation Intelligence Framework API
=========================================
A coordinate-gated DSP framework that maps audio frame coordinates in the
Universal Audio State Space, evaluates mathematical assumptions, and recommends
optimal representations and window sizes.
"""

from .engine import RepresentationIntelligenceEngine, FrameworkState
from .disagreement import (
    DisagreementReport,
    PairDisagreement,
    RepresentationReading,
    analyze_disagreement,
    compare_readings,
)
from .plugins.denoiser import AdaptiveDenoiser
from .plugins.pitch_tracker import AdaptivePitchTracker
from .plugins.onset_detector import AdaptiveOnsetDetector

__all__ = [
    "RepresentationIntelligenceEngine",
    "FrameworkState",
    "DisagreementReport",
    "PairDisagreement",
    "RepresentationReading",
    "analyze_disagreement",
    "compare_readings",
    "AdaptiveDenoiser",
    "AdaptivePitchTracker",
    "AdaptiveOnsetDetector",
]
