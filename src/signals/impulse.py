import numpy as np

def generate(duration=1.0, sr=22050, delay_s=0.0):
    """
    Generates a Kronecker delta impulse.
    
    Parameters:
    -----------
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.
    delay_s : float
        Delay of the impulse in seconds.
        
    Returns:
    --------
    signal : np.ndarray
        The generated signal.
    sr : int
        The sample rate.
    """
    length = int(sr * duration)
    signal = np.zeros(length)
    idx = int(delay_s * sr)
    if 0 <= idx < length:
        signal[idx] = 1.0
    return signal, sr
