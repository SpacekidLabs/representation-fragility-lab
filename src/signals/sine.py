import numpy as np

def generate(duration=1.0, sr=22050, f0=440.0):
    """
    Generates a pure sine wave.
    
    Parameters:
    -----------
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.
    f0 : float
        Frequency in Hz.
        
    Returns:
    --------
    signal : np.ndarray
        The generated signal.
    sr : int
        The sample rate.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * f0 * t)
    return signal, sr
