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
from src.representations.cepstrum import compute_cepstrum
from src.perturbations.time_stretch import apply_time_stretch
from src.perturbations.harmonic_removal import generate_harmonic_complex
from src.metrics.similarity import cosine_similarity

def run_experiment():
    sr = 22050
    duration = 1.0
    f0 = 440.0
    
    # 1. Generate baseline clean harmonic complex
    clean_signal = generate_harmonic_complex(f0, sr, duration, removal_pct=0.0)
    
    # 2. Compute original representations
    clean_stft = compute_stft(clean_signal, sr)
    clean_acf = compute_acf(clean_signal)
    clean_cepstrum = compute_cepstrum(clean_signal)
    
    # 3. Apply time stretches
    rates = np.linspace(0.5, 2.0, 11)  # 0.5 to 2.0 stretch rates
    
    stft_similarities = []
    acf_similarities = []
    cepstrum_similarities = []
    
    print("Running Time Stretch Fragility Experiment...")
    print(f"{'Rate':<15}{'STFT Sim':<15}{'ACF Sim':<15}{'Cepstrum Sim':<15}")
    print("-" * 60)
    
    for rate in rates:
        # Time stretch (and crop/pad)
        stretched_signal = apply_time_stretch(clean_signal, rate)
        
        # Compute representations
        stretched_stft = compute_stft(stretched_signal, sr)
        stretched_acf = compute_acf(stretched_signal)
        stretched_cepstrum = compute_cepstrum(stretched_signal)
        
        # Measure similarity
        stft_sim = cosine_similarity(clean_stft, stretched_stft)
        acf_sim = cosine_similarity(clean_acf, stretched_acf)
        cepstrum_sim = cosine_similarity(clean_cepstrum, stretched_cepstrum)
        
        stft_similarities.append(stft_sim)
        acf_similarities.append(acf_sim)
        cepstrum_similarities.append(cepstrum_sim)
        
        print(f"{rate:<15.2f}{stft_sim:<15.4f}{acf_sim:<15.4f}{cepstrum_sim:<15.4f}")
        
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    
    # Generate graph
    plt.figure(figsize=(10, 6))
    plt.plot(rates, stft_similarities, marker='o', linewidth=2, color='#1f77b4', label='STFT')
    plt.plot(rates, acf_similarities, marker='s', linewidth=2, color='#ff7f0e', label='ACF')
    plt.plot(rates, cepstrum_similarities, marker='^', linewidth=2, color='#2ca02c', label='Cepstrum')
    plt.title("Representation Fragility Under Time Stretching", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time Stretch Rate (Speed)", fontsize=12)
    plt.ylabel("Representation Similarity (Cosine Similarity)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=12)
    
    # Save the graph
    output_path = os.path.join(project_root, "results", "exp006_time_stretch_fragility.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print(f"Experiment complete! Graph saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()
