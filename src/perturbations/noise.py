import numpy as np

def add_noise(signal, amount):
    """
    Adds Gaussian white noise to the audio signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
    amount : float
        Standard deviation of the Gaussian noise (noise intensity).
        
    Returns:
    --------
    np.ndarray
        The noisy audio signal.
    """
    noise = np.random.normal(0, amount, size=signal.shape)
    return signal + noise
