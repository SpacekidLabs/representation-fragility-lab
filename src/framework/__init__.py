"""
Representation Intelligence Framework API
=========================================
A coordinate-gated DSP framework that maps audio frame coordinates in the
Universal Audio State Space, evaluates mathematical assumptions, and recommends
optimal representations and window sizes.
"""

from .engine import RepresentationIntelligenceEngine, FrameworkState

__all__ = ["RepresentationIntelligenceEngine", "FrameworkState"]
