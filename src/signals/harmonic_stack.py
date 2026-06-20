import numpy as np

def generate(duration=1.0, sr=22050, f0=440.0, num_harmonics=5):
    """
    Generates a harmonic complex (fundamental f0 plus overtones).
    Amplitudes decay as 1/k.
    
    Parameters:
    -----------
    duration : float
        Duration in seconds.
    sr : int
        Sample rate.
    f0 : float
        Fundamental frequency in Hz.
    num_harmonics : int
        Number of total components (fundamental + overtones).
        
    Returns:
    --------
    signal : np.ndarray
        The generated signal.
    sr : int
        The sample rate.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        amp = 1.0 / k
        signal += amp * np.sin(2 * np.pi * (k * f0) * t)
        
    # Normalize to peak amplitude of 1.0
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    return signal, sr
