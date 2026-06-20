import numpy as np

def generate(duration=1.0, sr=22050, fc=440.0, fm=220.0, index=2.0):
    """
    Generates a frequency modulated (FM) tone:
    y(t) = sin(2 * pi * fc * t + I * sin(2 * pi * fm * t))
    
    Parameters:
    -----------
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.
    fc : float
        Carrier frequency in Hz.
    fm : float
        Modulation frequency in Hz.
    index : float
        Modulation index.
        
    Returns:
    --------
    signal : np.ndarray
        The generated signal.
    sr : int
        The sample rate.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * fc * t + index * np.sin(2 * np.pi * fm * t))
    return signal, sr
