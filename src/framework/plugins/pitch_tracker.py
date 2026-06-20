"""
Adaptive Pitch Tracking Plugin
==============================
Implements the AdaptivePitchTracker which dynamically adapts window size,
trough threshold, and voicing/hold gating flags using YIN under the engine.
"""

import numpy as np
import scipy.signal
import librosa
from ..engine import RepresentationIntelligenceEngine

class AdaptivePitchTracker:
    """
    State-Space Adaptive YIN Pitch Tracker.
    Queries the RepresentationIntelligenceEngine frame-by-frame to scale the
    YIN window size and trough threshold, and apply context-aware hold logic.
    """
    def __init__(self, engine=None):
        self.engine = engine or RepresentationIntelligenceEngine()

    def track(self, sig: np.ndarray, sr: int, hop_length=512, fmin=80, fmax=1000) -> np.ndarray:
        """
        Track the pitch of the input signal.
        
        Parameters:
            sig: 1D NumPy array of time-domain audio.
            sr: sample rate.
            hop_length: frame hop size (default 512).
            fmin: minimum frequency in Hz (default 80).
            fmax: maximum frequency in Hz (default 1000).
            
        Returns:
            np.ndarray: 1D array of estimated pitch frequencies (Hz) per frame.
        """
        num_samples = len(sig)
        pad_len = 2048
        y_pad = np.pad(sig, pad_len, mode='reflect')
        
        hop_indices = np.arange(0, num_samples - hop_length, hop_length)
        N_frames = len(hop_indices)
        
        pitches = np.zeros(N_frames)
        last_valid_pitch = 220.0  # Safe initial anchor
        
        for idx, hop_idx in enumerate(hop_indices):
            center_idx = hop_idx
            
            # Analyze state using a standard local 1024-sample window centered at hop index
            f_win = 1024
            f_start = center_idx + pad_len - f_win // 2
            f_end = center_idx + pad_len + f_win // 2
            f_frame = y_pad[f_start:f_end]
            
            # Downsample to 22.05kHz for engine analyze consistency
            if sr != 22050:
                f_frame_22k = scipy.signal.resample(f_frame, f_win // 2)
                state = self.engine.analyze(f_frame_22k, 22050)
            else:
                state = self.engine.analyze(f_frame, sr)
                
            # Retrieve recommendations from engine
            win_size = state.recommended_window
            params = state.recommended_parameters["pitch_tracking"]
            trough_thresh = params["yin_trough"]
            voicing_gate = params["voicing_gate"]
            hold_pitch = params["hold_pitch"]
            
            # Extract the dynamically scaled window centered at the hop index
            a_start = center_idx + pad_len - win_size // 2
            a_end = center_idx + pad_len + win_size // 2
            a_frame = y_pad[a_start:a_end]
            
            # Run YIN on the adaptive frame size
            try:
                pitch_array = librosa.yin(
                    a_frame, fmin=fmin, fmax=fmax, sr=sr, 
                    frame_length=win_size, hop_length=win_size, 
                    trough_threshold=trough_thresh, center=False
                )
                pitch = float(pitch_array[0])
            except Exception:
                pitch = 0.0
                
            # Apply dynamic gating/hold logic
            if hold_pitch and state.assumptions.get("acf", 1.0) < 0.20:
                pitch = last_valid_pitch
            elif pitch > fmin and pitch < fmax:
                last_valid_pitch = pitch
                
            if voicing_gate and state.assumptions.get("acf", 1.0) < 0.35:
                pitch = 0.0  # Treat as unvoiced/silence
                
            pitches[idx] = pitch
            
        return pitches
