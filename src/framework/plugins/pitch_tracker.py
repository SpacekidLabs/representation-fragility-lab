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
        pitch_history = [0.0] * 5
        unvoiced_consecutive_frames = 0
        
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

            # Step 1: Octave correction and continuity constraint relative to last_valid_pitch
            if last_valid_pitch > 0.0 and pitch > 0.0:
                ratio = pitch / last_valid_pitch
                
                # Detect and correct octave tracking errors
                if abs(ratio - 0.5) < 0.08:
                    pitch *= 2.0
                    ratio *= 2.0
                elif abs(ratio - 2.0) < 0.30:
                    pitch *= 0.5
                    ratio *= 0.5
                elif abs(ratio - 0.25) < 0.04:
                    pitch *= 4.0
                    ratio *= 4.0
                elif abs(ratio - 4.0) < 0.60:
                    pitch *= 0.25
                    ratio *= 0.25
                
                # Enforce absolute continuity limit: max 3.0 semitones change per frame
                max_ratio = 2**(3/12)
                min_ratio = 2**(-3/12)
                if ratio > max_ratio or ratio < min_ratio:
                    pitch = last_valid_pitch

            # Step 2: Determine raw voicing state
            raw_is_voiced = (pitch > fmin and pitch < fmax)
            
            # Apply Engine recommendations and confidence constraints
            acf_conf = state.assumptions.get("acf", 1.0)
            if voicing_gate and acf_conf < 0.35:
                raw_is_voiced = False
            if hold_pitch and acf_conf < 0.20:
                pitch = last_valid_pitch
                raw_is_voiced = (pitch > fmin and pitch < fmax)

            # Step 3: Run voiced-only median filter
            if raw_is_voiced:
                # Initialize history if empty to avoid onset lag
                if all(val == 0.0 for val in pitch_history):
                    pitch_history = [pitch] * 5
                else:
                    pitch_history.pop(0)
                    pitch_history.append(pitch)
                
                # Set pitch to median of history
                pitch = float(np.median(pitch_history))
                last_valid_pitch = pitch
            else:
                pitch = 0.0

            # Step 4: Apply release gate hysteresis (hold last pitch for up to 3 frames ~35 ms)
            is_voiced = raw_is_voiced
            if raw_is_voiced:
                unvoiced_consecutive_frames = 0
            else:
                unvoiced_consecutive_frames += 1
                if unvoiced_consecutive_frames < 3 and last_valid_pitch > fmin and last_valid_pitch < fmax:
                    is_voiced = True
                    pitch = last_valid_pitch
                else:
                    # Reset pitch history after full release
                    pitch_history = [0.0] * 5
            
            pitches[idx] = pitch if is_voiced else 0.0
            
        return pitches
