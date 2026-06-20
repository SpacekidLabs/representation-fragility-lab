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

def pearson_correlation(a, b):
    """
    Computes the Pearson correlation coefficient between two arrays.
    The arrays are flattened and mean-centered before calculation.
    
    Parameters:
    -----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
        
    Returns:
    --------
    float
        Pearson correlation coefficient in range [-1.0, 1.0].
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    a_centered = a_flat - np.mean(a_flat)
    b_centered = b_flat - np.mean(b_flat)
    
    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    corr = np.dot(a_centered, b_centered) / (norm_a * norm_b)
    return float(np.clip(corr, -1.0, 1.0))

def euclidean_similarity(a, b):
    """
    Computes a normalized Euclidean similarity index between two arrays.
    First, both arrays are peak-normalized to be scale-invariant.
    Then, the similarity is defined as 1 / (1 + Euclidean distance).
    
    Parameters:
    -----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
        
    Returns:
    --------
    float
        Euclidean similarity index in range (0, 1].
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    max_a = np.max(np.abs(a_flat))
    max_b = np.max(np.abs(b_flat))
    
    a_norm = a_flat / max_a if max_a > 0 else a_flat
    b_norm = b_flat / max_b if max_b > 0 else b_flat
    
    dist = np.linalg.norm(a_norm - b_norm)
    return float(1.0 / (1.0 + dist))
