import numpy as np

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two arrays.
    The arrays are flattened before calculation.
    
    Parameters:
    -----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
        
    Returns:
    --------
    float
        Cosine similarity score (typically between 0 and 1 for magnitude spectrograms).
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    similarity = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    
    # Clip to [0, 1] range to avoid floating point issues
    return float(np.clip(similarity, 0.0, 1.0))
