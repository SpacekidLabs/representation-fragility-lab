import librosa

def apply_pitch_shift(signal, sr, n_steps):
    """
    Applies pitch shifting to the audio signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        The input audio signal.
    sr : int
        Sample rate.
    n_steps : float
        Number of semitones to shift (can be fractional).
        
    Returns:
    --------
    np.ndarray
        The pitch-shifted signal.
    """
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
