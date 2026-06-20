import numpy as np
import librosa

def compute_stft(signal, sr):
    """
    Computes the STFT magnitude spectrogram of an audio signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
    sr : int
        The sample rate of the audio signal.
        
    Returns:
    --------
    np.ndarray
        The magnitude spectrogram (absolute value of complex STFT).
    """
    stft_complex = librosa.stft(signal)
    return np.abs(stft_complex)
