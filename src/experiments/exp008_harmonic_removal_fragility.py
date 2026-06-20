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
from src.perturbations.harmonic_removal import generate_harmonic_complex
from src.metrics.similarity import cosine_similarity

def run_experiment():
    sr = 22050
    duration = 1.0
    f0 = 440.0
    
    # 1. Generate baseline clean harmonic complex (0% removed)
    clean_signal = generate_harmonic_complex(f0, sr, duration, removal_pct=0.0)
    
    # 2. Compute original representations
    clean_stft = compute_stft(clean_signal, sr)
    clean_acf = compute_acf(clean_signal)
    clean_cepstrum = compute_cepstrum(clean_signal)
    
    # 3. Apply harmonic removal levels
    pcts = [0.0, 25.0, 50.0, 75.0, 100.0]
    
    stft_similarities = []
    acf_similarities = []
    cepstrum_similarities = []
    
    print("Running Harmonic Removal Fragility Experiment...")
    print(f"{'Removal %':<15}{'STFT Sim':<15}{'ACF Sim':<15}{'Cepstrum Sim':<15}")
    print("-" * 60)
    
    for pct in pcts:
        # Generate signal with harmonics removed
        stripped_signal = generate_harmonic_complex(f0, sr, duration, removal_pct=pct)
        
        # Compute representations
        stripped_stft = compute_stft(stripped_signal, sr)
        stripped_acf = compute_acf(stripped_signal)
        stripped_cepstrum = compute_cepstrum(stripped_signal)
        
        # Measure similarity
        stft_sim = cosine_similarity(clean_stft, stripped_stft)
        acf_sim = cosine_similarity(clean_acf, stripped_acf)
        cepstrum_sim = cosine_similarity(clean_cepstrum, stripped_cepstrum)
        
        stft_similarities.append(stft_sim)
        acf_similarities.append(acf_sim)
        cepstrum_similarities.append(cepstrum_sim)
        
        print(f"{pct:<15.1f}{stft_sim:<15.4f}{acf_sim:<15.4f}{cepstrum_sim:<15.4f}")
        
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    
    # Generate graph
    plt.figure(figsize=(10, 6))
    plt.plot(pcts, stft_similarities, marker='o', linewidth=2, color='#1f77b4', label='STFT')
    plt.plot(pcts, acf_similarities, marker='s', linewidth=2, color='#ff7f0e', label='ACF')
    plt.plot(pcts, cepstrum_similarities, marker='^', linewidth=2, color='#2ca02c', label='Cepstrum')
    plt.title("Representation Fragility Under Harmonic Removal", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Harmonics Removed (%)", fontsize=12)
    plt.ylabel("Representation Similarity (Cosine Similarity)", fontsize=12)
    plt.xticks(pcts)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=12)
    
    # Save the graph
    output_path = os.path.join(project_root, "results", "exp008_harmonic_removal_fragility.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print(f"Experiment complete! Graph saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()
