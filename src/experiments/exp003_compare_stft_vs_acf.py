import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.stft import compute_stft
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
    
    # 2. Compute original representations
    clean_stft = compute_stft(clean_signal, sr)
    clean_acf = compute_acf(clean_signal)
    
    # 3. Apply increasing levels of noise
    noise_amounts = np.linspace(0.0, 1.0, 21)
    stft_similarities = []
    acf_similarities = []
    
    print("Running STFT vs ACF Fragility Comparison...")
    print(f"{'Noise Amount':<15}{'STFT Sim':<15}{'ACF Sim':<15}")
    print("-" * 45)
    
    for amount in noise_amounts:
        # Add noise
        noisy_signal = add_noise(clean_signal, amount)
        # Compute noisy representations
        noisy_stft = compute_stft(noisy_signal, sr)
        noisy_acf = compute_acf(noisy_signal)
        # Measure similarity
        stft_sim = cosine_similarity(clean_stft, noisy_stft)
        acf_sim = cosine_similarity(clean_acf, noisy_acf)
        
        stft_similarities.append(stft_sim)
        acf_similarities.append(acf_sim)
        
        print(f"{amount:<15.2f}{stft_sim:<15.4f}{acf_sim:<15.4f}")
        
    # Ensure results directory exists
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    
    # Generate graph
    plt.figure(figsize=(10, 6))
    plt.plot(noise_amounts, stft_similarities, marker='o', linewidth=2, color='#1f77b4', label='STFT')
    plt.plot(noise_amounts, acf_similarities, marker='s', linewidth=2, color='#ff7f0e', label='ACF')
    plt.title("STFT vs ACF Fragility Under Additive Noise", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Noise Amount (Standard Deviation)", fontsize=12)
    plt.ylabel("Representation Similarity (Cosine Similarity)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=12)
    
    # Save the graph
    output_path = os.path.join(project_root, "results", "exp003_compare_stft_vs_acf.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 45)
    print(f"Comparison complete! Graph saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()
