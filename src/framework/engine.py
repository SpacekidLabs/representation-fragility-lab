"""
Framework Core Engine
=====================
Defines the main class `RepresentationIntelligenceEngine` and the structured state class
`FrameworkState` that encapsulates the physical audio state space coordinates, region,
assumption safety predictions, and DSP routing recommendations.
"""

from dataclasses import dataclass

import numpy as np
from .descriptors import extract_physical_descriptors
from .weights import MU, SIGMA, VT, W_STFT, W_ACF, W_CEP, W_CQT, W_CWT


SAFE_REPRESENTATION_THRESHOLD = 0.50
DEFAULT_WINDOW_SIZE = 2048
REGION_WINDOW_SIZES = {
    "noise_collapse": 4096,
    "transient_overloaded": 1024,
}
REGION_DENOISING_BETA = {
    "noise_collapse": 0.06,
    "periodic_harmonic": 0.005,
}
DEFAULT_DENOISING_BETA = 0.02
REGION_YIN_TROUGH = {
    "noise_collapse": 0.25,
    "transient_overloaded": 0.08,
}
DEFAULT_YIN_TROUGH = 0.15
VOICED_REGIONS = {"periodic_harmonic", "smooth_lowpass"}


@dataclass
class FrameworkState:
    """
    Structured state object returned by RepresentationIntelligenceEngine.
    Encapsulates all mapping parameters, valid assumptions, and DSP recommendations.
    """
    coordinate: tuple[float, float]
    region: str
    assumptions: dict[str, float]
    recommendations: dict

    @property
    def coordinates(self) -> tuple[float, float]:
        """Get the (z1, z2) coordinates in the physical state space."""
        return self.coordinate

    @property
    def safe_representations(self) -> list[str]:
        """List of representation names with safety score >= 0.50."""
        return [
            name
            for name, score in self.assumptions.items()
            if score >= SAFE_REPRESENTATION_THRESHOLD
        ]

    @property
    def recommended_window(self) -> int:
        """Recommended analysis window size in samples."""
        return self.recommendations.get("window_size", 2048)

    @property
    def recommended_latency(self) -> int:
        """Recommended processing latency in samples (assuming 50% overlap)."""
        return self.recommended_window // 2

    @property
    def recommended_parameters(self) -> dict[str, dict]:
        """
        Dictionary of task-specific parameter recommendations:
        - 'denoising': {'alpha': float, 'beta': float}
        - 'pitch_tracking': {'yin_trough': float, 'voicing_gate': bool, 'hold_pitch': bool}
        - 'onset_detection': {'fusion_weights': dict[str, float], 'threshold': float}
        """
        stft_s = self.assumptions.get("stft", 1.0)
        alpha = 0.5 + 3.5 * (1.0 - stft_s)
        beta = REGION_DENOISING_BETA.get(self.region, DEFAULT_DENOISING_BETA)
        yin_trough = REGION_YIN_TROUGH.get(self.region, DEFAULT_YIN_TROUGH)
        voicing_gate = self.region in VOICED_REGIONS
        hold_pitch = self.region == "transient_overloaded"

        fusion_weights = {
            "stft": self.assumptions.get("stft", 0.0),
            "acf": self.assumptions.get("acf", 0.0),
            "cepstrum": self.assumptions.get("cepstrum", 0.0),
        }
        onset_threshold = 0.3 if self.region == "noise_collapse" else 0.15

        return {
            "denoising": {
                "alpha": alpha,
                "beta": beta,
            },
            "pitch_tracking": {
                "yin_trough": yin_trough,
                "voicing_gate": voicing_gate,
                "hold_pitch": hold_pitch,
            },
            "onset_detection": {
                "fusion_weights": fusion_weights,
                "threshold": onset_threshold,
            },
        }

    def __repr__(self) -> str:
        return (f"FrameworkState(\n"
                f"  coordinate={self.coordinate},\n"
                f"  region='{self.region}',\n"
                f"  assumptions={self.assumptions},\n"
                f"  recommendations={self.recommendations}\n"
                f")")


class RepresentationIntelligenceEngine:
    """
    The core framework engine. Analyzes audio frames on-the-fly, projects them
    into the Universal Audio State Space, queries learned safety contours, and
    recommends optimal DSP parameters.
    """
    def __init__(self):
        # Load pre-trained parameters from weights constants
        self.mu = MU
        self.sigma = SIGMA
        self.Vt = VT
        self.w_models = {
            "stft": W_STFT,
            "acf": W_ACF,
            "cepstrum": W_CEP,
            "cqt": W_CQT,
            "wavelet": W_CWT
        }

    def analyze(self, frame: np.ndarray, sr: int) -> FrameworkState:
        """
        Analyzes a single frame of audio.
        
        Parameters:
            frame: np.ndarray, 1D array of time-domain audio samples.
            sr: int, sample rate of the audio.
            
        Returns:
            FrameworkState: The structured state object with coordinates and recommendations.
        """
        # Ensure frame is 1D and has non-zero energy
        frame = np.asarray(frame).flatten()
        if len(frame) == 0:
            raise ValueError("Empty audio frame provided.")
            
        # 1. Extract the 10 physical signal descriptors
        feats = extract_physical_descriptors(frame, sr)
        
        # 2. Project onto the pre-trained Universal Audio State Space
        feats_std = (np.array(feats) - self.mu) / self.sigma
        coords = feats_std @ self.Vt.T
        z1, z2 = float(coords[0]), float(coords[1])
        
        # 3. Identify the Semantic Region of the State Space
        # PC1 represents Order vs Disorder (high z1 = noise)
        # PC2 represents Harmonic vs Transient/Peakiness (high z2 = transient/saturated)
        if z1 > 1.5:
            region = "noise_collapse"
        elif z1 < -0.5 and z2 < -0.2:
            region = "periodic_harmonic"
        elif z2 > 1.5:
            region = "transient_overloaded"
        elif -0.5 <= z1 <= 1.5 and z2 < -0.2:
            region = "smooth_lowpass"
        else:
            region = "transition_zone"

        # 4. Predict Safety Scores of assumptions
        # Degree-2 polynomial features: [1, z1, z2, z1^2, z2^2, z1*z2]
        poly = np.array([1.0, z1, z2, z1**2, z2**2, z1 * z2])
        assumptions = {}
        for name, w in self.w_models.items():
            assumptions[name] = float(np.clip(poly @ w, 0.0, 1.0))

        # 5. Formulate DSP Routing Recommendations
        active_representations = {
            name: score >= SAFE_REPRESENTATION_THRESHOLD
            for name, score in assumptions.items()
        }
        
        window_size = REGION_WINDOW_SIZES.get(region, DEFAULT_WINDOW_SIZE)

        # Specific trust recommendation
        primary_representation = max(assumptions, key=assumptions.get)

        recommendations = {
            "active_representations": active_representations,
            "window_size": window_size,
            "primary_representation": primary_representation,
        }

        return FrameworkState(
            coordinate=(z1, z2),
            region=region,
            assumptions=assumptions,
            recommendations=recommendations
        )
