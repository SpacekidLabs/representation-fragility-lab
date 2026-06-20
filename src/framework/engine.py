"""
Framework Core Engine
=====================
Defines the main class `RepresentationIntelligenceEngine` and the structured state class
`FrameworkState` that encapsulates the physical audio state space coordinates, region,
assumption safety predictions, and DSP routing recommendations.
"""

import numpy as np
from src.experiments.exp030_universal_audio_state_space import extract_10_physical_descriptors
from .weights import MU, SIGMA, VT, W_STFT, W_ACF, W_CEP, W_CQT, W_CWT

class FrameworkState:
    """
    Structured state object returned by RepresentationIntelligenceEngine.
    Encapsulates all mapping parameters, valid assumptions, and DSP recommendations.
    """
    def __init__(self, coordinate: tuple[float, float], region: str, 
                 assumptions: dict[str, float], recommendations: dict):
        self.coordinate = coordinate  # (z1, z2) PCA coordinates
        self.region = region          # Semantic region string
        self.assumptions = assumptions  # dict mapping rep name -> float safety score [0, 1]
        self.recommendations = recommendations  # dict containing active reps, window size, etc.

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
        feats = extract_10_physical_descriptors(frame, sr)
        
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
            score = float(np.clip(poly @ w, 0.0, 1.0))
            assumptions[name] = score

        # 5. Formulate DSP Routing Recommendations
        active_representations = {
            name: (score >= 0.50) for name, score in assumptions.items()
        }
        
        # Determine recommended window size based on region characteristics
        if region == "noise_collapse":
            window_size = 4096  # Increase window to average out random noise correlations
        elif region == "transient_overloaded":
            window_size = 1024  # Decrease window for high temporal precision
        else:
            window_size = 2048  # Default window size for stable conditions

        # Specific trust recommendation
        primary_representation = max(assumptions, key=assumptions.get)

        recommendations = {
            "active_representations": active_representations,
            "window_size": window_size,
            "primary_representation": primary_representation
        }

        return FrameworkState(
            coordinate=(z1, z2),
            region=region,
            assumptions=assumptions,
            recommendations=recommendations
        )
