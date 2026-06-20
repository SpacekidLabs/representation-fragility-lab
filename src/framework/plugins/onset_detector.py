"""
Adaptive Onset Detection Plugin
===============================
Implements the AdaptiveOnsetDetector which fuses STFT spectral flux,
ACF prominence drop, and Cepstrum peak velocity using engine safety weights.
"""

import numpy as np
import scipy.signal
from ..engine import RepresentationIntelligenceEngine

class AdaptiveOnsetDetector:
    """
    State-Space Adaptive Onset Detector.
    Fuses three different onset detection functions (spectral flux, ACF prominence
    drop, cepstral velocity) using the engine's safety predictions.
    """
    def __init__(self, engine=None):
        self.engine = engine or RepresentationIntelligenceEngine()

    def detect(self, sig: np.ndarray, sr: int, hop_length=512, frame_len=2048, threshold_factor=0.25) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect onsets in the input signal.
        
        Parameters:
            sig: 1D NumPy array of time-domain audio.
            sr: sample rate.
            hop_length: hop size (default 512).
            frame_len: window size (default 2048).
            threshold_factor: peak picking threshold factor (default 0.25).
            
        Returns:
            fused_score: 1D NumPy array of fused onset strengths per frame.
            onset_samples: 1D NumPy array of detected onset sample indices.
        """
        n_frames = (len(sig) - frame_len) // hop_length
        if n_frames <= 0:
            return np.array([]), np.array([])
            
        # 1. Compute individual onset curves
        fl = self._stft_flux(sig, frame_len, hop_length)
        ac = self._acf_prom_drop(sig, frame_len, hop_length)
        cp = self._cep_velocity(sig, frame_len, hop_length, sr)
        
        # Normalize each individual function to [0, 1]
        fl = self._norm01(fl)
        ac = self._norm01(ac)
        cp = self._norm01(cp)
        
        # 2. Extract weights and fuse frame-by-frame
        w_s = np.zeros(n_frames)
        w_a = np.zeros(n_frames)
        w_c = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = sig[i*hop_length:i*hop_length+frame_len]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
                
            # Downsample to 22.05kHz for engine analyze consistency
            if sr != 22050:
                frame_22k = scipy.signal.resample(frame, frame_len // 2)
                state = self.engine.analyze(frame_22k, 22050)
            else:
                state = self.engine.analyze(frame, sr)
                
            # Extract fusion weights from recommended_parameters
            weights = state.recommended_parameters["onset_detection"]["fusion_weights"]
            w_s[i] = weights["stft"]
            w_a[i] = weights["acf"]
            w_c[i] = weights["cepstrum"]
            
        fused = (w_s * fl + w_a * ac + w_c * cp) / (w_s + w_a + w_c + 1e-9)
        fused = self._norm01(fused)
        
        # Peak picking
        min_gap = max(1, int(0.05 * sr / hop_length))
        peaks = self._peak_pick(fused, min_gap, threshold_factor)
        onset_samples = peaks * hop_length
        
        return fused, onset_samples

    def _stft_flux(self, sig, frame_len, hop):
        n = (len(sig) - frame_len) // hop
        out = np.zeros(n)
        prev = None
        for i in range(n):
            mag = np.abs(np.fft.rfft(sig[i*hop:i*hop+frame_len] * np.hanning(frame_len)))
            if prev is not None:
                out[i] = np.sum(np.maximum(mag - prev, 0))
            prev = mag
        return out

    def _acf_prom_drop(self, sig, frame_len, hop):
        n = (len(sig) - frame_len) // hop
        out = np.zeros(n)
        prev = None
        for i in range(n):
            frame = sig[i*hop:i*hop+frame_len]
            ac = np.correlate(frame, frame, "full")[len(frame)-1:]
            ac /= ac[0] + 1e-9
            pk = np.argmax(ac[1:]) + 1 if len(ac) > 1 else 1
            prom = float(ac[pk]) - float(np.mean(ac[1:]))
            if prev is not None:
                out[i] = max(prev - prom, 0.0)
            prev = prom
        return out

    def _cep_velocity(self, sig, frame_len, hop, sr):
        n = (len(sig) - frame_len) // hop
        out = np.zeros(n)
        prev = None
        lo = max(1, int(0.002 * sr))
        hi = int(0.020 * sr)
        for i in range(n):
            frame = sig[i*hop:i*hop+frame_len]
            mag = np.abs(np.fft.rfft(frame)) + 1e-9
            cep = np.abs(np.fft.irfft(np.log(mag)))
            pk = float(np.max(cep[lo:hi])) if hi > lo else 0.0
            if prev is not None:
                out[i] = abs(pk - prev)
            prev = pk
        return out

    def _norm01(self, arr):
        mx = np.max(arr)
        return arr / mx if mx > 1e-12 else arr

    def _peak_pick(self, score, min_gap_frames, threshold_factor):
        threshold = threshold_factor * np.max(score) if np.max(score) > 1e-12 else 1e-9
        peaks = []
        for i in range(1, len(score) - 1):
            if (score[i] > score[i - 1] and score[i] > score[i + 1]
                    and score[i] > threshold):
                if not peaks or (i - peaks[-1]) >= min_gap_frames:
                    peaks.append(i)
        return np.array(peaks, dtype=int)
