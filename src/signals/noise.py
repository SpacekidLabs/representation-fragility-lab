import numpy as np

def generate(duration=1.0, sr=22050, std=1.0):
    """
    Generates white Gaussian noise.
    
    Parameters:
    -----------
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.
    std : float
        Standard deviation of the Gaussian noise.
        
    Returns:
    --------
    signal : np.ndarray
        The generated signal.
    sr : int
        The sample rate.
    """
    length = int(sr * duration)
    signal = np.random.normal(0, std, size=length)
    
    # Normalize to peak amplitude of 1.0
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    return signal, sr
