import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.acf import compute_acf
from src.perturbations.noise import add_noise
from src.metrics.similarity import cosine_similarity

def run_experiment():
    # Parameters
    sr = 22050
    duration = 1.0  # seconds
    frequency = 440.0  # Hz
    
    # 1. Generate a clean 440 Hz sine wave
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    clean_signal = np.sin(2 * np.pi * frequency * t)
    
    # 2. Compute its ACF representation
    clean_acf = compute_acf(clean_signal)
    
    # 3. Apply increasing levels of noise
    noise_amounts = np.linspace(0.0, 1.0, 21)
    similarities = []
    
    print("Running ACF Fragility Experiment...")
    print(f"{'Noise Amount':<15}{'Similarity Score':<20}")
    print("-" * 35)
    
    for amount in noise_amounts:
        # Add noise
        noisy_signal = add_noise(clean_signal, amount)
        # Compute ACF
        noisy_acf = compute_acf(noisy_signal)
        # Measure similarity
        sim = cosine_similarity(clean_acf, noisy_acf)
        similarities.append(sim)
        
        print(f"{amount:<15.2f}{sim:<20.4f}")
        
    # Ensure results directory exists
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    
    # Generate graph
    plt.figure(figsize=(10, 6))
    plt.plot(noise_amounts, similarities, marker='o', linewidth=2, color='#ff7f0e')
    plt.title("ACF Fragility Under Additive Noise", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Noise Amount (Standard Deviation)", fontsize=12)
    plt.ylabel("Representation Similarity (Cosine Similarity)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    
    # Save the graph
    output_path = os.path.join(project_root, "results", "exp002_acf_noise_fragility.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 35)
    print(f"Experiment complete! Graph saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()
