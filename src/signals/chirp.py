import numpy as np
from scipy.signal import chirp

def generate(duration=1.0, sr=22050, f_start=220.0, f_end=880.0):
    """
    Generates a linear frequency sweep (chirp).
    
    Parameters:
    -----------
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.
    f_start : float
        Starting frequency in Hz.
    f_end : float
        Ending frequency in Hz.
        
    Returns:
    --------
    signal : np.ndarray
        The generated signal.
    sr : int
        The sample rate.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = chirp(t, f0=f_start, t1=duration, f1=f_end, method='linear')
    return signal, sr
