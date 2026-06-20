import numpy as np

def generate_harmonic_complex(f0, sr, duration, removal_pct):
    """
    Generates a harmonic complex with a fundamental frequency f0 and 4 overtones,
    with a specified percentage of overtones removed.
    
    Parameters:
    -----------
    f0 : float
        Fundamental frequency in Hz.
    sr : int
        Sample rate.
    duration : float
        Duration in seconds.
    removal_pct : float
        Percentage of overtones to remove (0.0, 25.0, 50.0, 75.0, 100.0).
        
    Returns:
    --------
    np.ndarray
        The generated audio signal.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Define amplitudes for fundamental and 4 harmonics (total 5 components)
    harmonics = [
        {"freq": f0, "amp": 1.0},
        {"freq": 2 * f0, "amp": 0.8},
        {"freq": 3 * f0, "amp": 0.6},
        {"freq": 4 * f0, "amp": 0.4},
        {"freq": 5 * f0, "amp": 0.2},
    ]
    
    # 0% removed -> keep all 4 overtones (5 total components)
    # 25% removed -> keep 3 overtones (4 total components)
    # 50% removed -> keep 2 overtones (3 total components)
    # 75% removed -> keep 1 overtone (2 total components)
    # 100% removed -> keep 0 overtones (1 total component: the fundamental)
    
    if removal_pct >= 100.0:
        num_overtones_to_keep = 0
    elif removal_pct >= 75.0:
        num_overtones_to_keep = 1
    elif removal_pct >= 50.0:
        num_overtones_to_keep = 2
    elif removal_pct >= 25.0:
        num_overtones_to_keep = 3
    else:
        num_overtones_to_keep = 4
        
    active_components = harmonics[:1 + num_overtones_to_keep]
    
    signal = np.zeros_like(t)
    for comp in active_components:
        signal += comp["amp"] * np.sin(2 * np.pi * comp["freq"] * t)
        
    # Normalize to peak amplitude of 1.0
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    return signal
