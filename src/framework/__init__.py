"""
Representation Intelligence Framework API
=========================================
A coordinate-gated DSP framework that maps audio frame coordinates in the
Universal Audio State Space, evaluates mathematical assumptions, and recommends
optimal representations and window sizes.
"""

from .engine import RepresentationIntelligenceEngine, FrameworkState
from .audio import (
    ALL_LENSES,
    AudioFrame,
    PitchComparator,
    make_audio_analyzer,
)
from .disagreement import (
    DisagreementReport,
    PairDisagreement,
    RepresentationReading,
    analyze_disagreement,
    compare_readings,
)
from .plugins.denoiser import AdaptiveDenoiser
from .plugins.pitch_tracker import AdaptivePitchTracker, QuorumPitchTracker
from .plugins.onset_detector import AdaptiveOnsetDetector

__all__ = [
    "RepresentationIntelligenceEngine",
    "FrameworkState",
    "ALL_LENSES",
    "AudioFrame",
    "PitchComparator",
    "make_audio_analyzer",
    "DisagreementReport",
    "PairDisagreement",
    "RepresentationReading",
    "analyze_disagreement",
    "compare_readings",
    "AdaptiveDenoiser",
    "AdaptivePitchTracker",
    "QuorumPitchTracker",
    "AdaptiveOnsetDetector",
]
