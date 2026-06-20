from scipy.signal import butter, filtfilt

def lowpass_filter(signal, sr, cutoff_hz):
    """
    Applies a Butterworth lowpass filter to the audio signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
    sr : int
        Sample rate.
    cutoff_hz : float
        Cutoff frequency in Hz.
        
    Returns:
    --------
    np.ndarray
        The filtered audio signal.
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff_hz / nyq
    
    if normal_cutoff >= 1.0:
        return signal.copy()
        
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered
