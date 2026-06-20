"""
Adaptive Denoising Plugin
=========================
Implements the AdaptiveDenoiser which adjusts spectral subtraction parameters
(alpha and beta) based on the state space coordinates of the audio frame.
"""

import numpy as np
import scipy.signal
from ..engine import RepresentationIntelligenceEngine

class AdaptiveDenoiser:
    """
    State-Space Adaptive Spectral Subtraction Denoiser.
    Uses the RepresentationIntelligenceEngine to adapt noise subtraction factor
    alpha and spectral floor beta per frame.
    """
    def __init__(self, engine=None):
        self.engine = engine or RepresentationIntelligenceEngine()
        self.noise_psd = None

    def process(self, noisy_sig: np.ndarray, sr: int, nperseg=2048, hop=512, noise_prefix_len_s=0.1) -> np.ndarray:
        """
        Process the entire audio signal.
        
        Parameters:
            noisy_sig: 1D NumPy array of noisy time-domain signal.
            sr: sample rate.
            nperseg: STFT window size (default 2048).
            hop: STFT hop size (default 512).
            noise_prefix_len_s: duration of the silent noise prefix at start of signal.
            
        Returns:
            np.ndarray: Denoised time-domain signal.
        """
        noverlap = nperseg - hop
        f, t, Zxx = scipy.signal.stft(noisy_sig, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
        
        # Estimate noise PSD from the silent prefix (first few frames)
        prefix_frames = max(1, int(noise_prefix_len_s * sr / hop))
        self.noise_psd = np.mean(np.abs(Zxx[:, :prefix_frames]) ** 2, axis=1)
        
        Zxx_clean = np.zeros_like(Zxx, dtype=np.complex128)
        
        for m in range(Zxx.shape[1]):
            start_idx = m * hop
            end_idx = start_idx + nperseg
            frame = noisy_sig[start_idx:end_idx]
            
            if len(frame) < nperseg:
                frame = np.pad(frame, (0, nperseg - len(frame)))
                
            # Denoise single frame spectrum using our process_frame logic
            Zxx_clean[:, m] = self.process_frame(Zxx[:, m], frame, sr, self.noise_psd)
            
        _, denoise_sig = scipy.signal.istft(Zxx_clean, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
        return denoise_sig[:len(noisy_sig)]

    def process_frame(self, Zxx_frame: np.ndarray, time_frame: np.ndarray, sr: int, noise_psd: np.ndarray) -> np.ndarray:
        """
        Denoise a single spectral frame based on the state space coordinates of its time-domain frame.
        
        Parameters:
            Zxx_frame: Complex spectrum of the current frame (shape: frequency bins).
            time_frame: Time-domain window corresponding to the frame (used for engine analysis).
            sr: sample rate.
            noise_psd: Estimated noise power spectral density.
            
        Returns:
            np.ndarray: Filtered complex spectrum of the frame.
        """
        nperseg = len(time_frame)
        
        # Downsample frame to 22.05kHz to match engine weights
        if sr != 22050:
            frame_22k = scipy.signal.resample(time_frame, nperseg // 2)
            state = self.engine.analyze(frame_22k, 22050)
        else:
            state = self.engine.analyze(time_frame, sr)
            
        # Extract recommended alpha and beta
        params = state.recommended_parameters["denoising"]
        alpha = params["alpha"]
        beta = params["beta"]
        
        X_mag = np.abs(Zxx_frame)
        G = np.maximum(1.0 - alpha * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), beta)
        return Zxx_frame * G
