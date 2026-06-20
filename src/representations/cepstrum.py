import numpy as np

def compute_cepstrum(signal):
    """
    Computes the real cepstrum of the input signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
        
    Returns:
    --------
    np.ndarray
        Real cepstrum coefficients.
    """
    spectrum = np.fft.rfft(signal)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)
    return cepstrum
