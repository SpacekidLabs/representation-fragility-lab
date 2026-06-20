import numpy as np
import librosa

def apply_time_stretch(signal, rate):
    """
    Applies time stretching to the audio signal.
    Pads or crops the result to maintain the original signal length.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
    rate : float
        Stretch rate. rate > 1.0 speeds up the signal, rate < 1.0 slows it down.
        
    Returns:
    --------
    np.ndarray
        The time-stretched signal with the same length as the original.
    """
    if rate == 1.0:
        return signal.copy()
        
    stretched = librosa.effects.time_stretch(signal, rate=rate)
    target_len = len(signal)
    
    if len(stretched) < target_len:
        # Zero-pad
        stretched = np.pad(stretched, (0, target_len - len(stretched)))
    else:
        # Truncate
        stretched = stretched[:target_len]
        
    return stretched
