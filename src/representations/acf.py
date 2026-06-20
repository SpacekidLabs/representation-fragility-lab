import numpy as np

def compute_acf(signal):
    """
    Computes the autocorrelation of the input signal (positive lags only)
    using FFT for O(N log N) performance.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
        
    Returns:
    --------
    np.ndarray
        Autocorrelation coefficients for positive lags.
    """
    n = len(signal)
    n_pad = 2 * n
    f = np.fft.rfft(signal, n=n_pad)
    acf = np.fft.irfft(f * np.conj(f), n=n_pad)
    return acf[:n]
