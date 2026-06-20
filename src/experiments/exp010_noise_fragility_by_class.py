import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals import sine, harmonic_stack, fm_tone, chirp, impulse
from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.perturbations.noise import add_noise
from src.metrics.similarity import cosine_similarity

def run_experiment():
    sr = 22050
    duration = 1.0
    
    # 1. Define signal classes to sweep
    signal_classes = [
        {"name": "Sine", "gen": lambda: sine.generate(duration, sr)[0]},
        {"name": "Harmonic Stack", "gen": lambda: harmonic_stack.generate(duration, sr)[0]},
        {"name": "FM Tone", "gen": lambda: fm_tone.generate(duration, sr)[0]},
        {"name": "Chirp", "gen": lambda: chirp.generate(duration, sr)[0]},
        {"name": "Impulse", "gen": lambda: impulse.generate(duration, sr, delay_s=0.0)[0]},
    ]
    
    # 2. Define representations
    noise_amounts = np.linspace(0.0, 1.0, 11)
    
    # Structure to hold results: {class_name: {representation_name: [similarities]}}
    results = {
        cls["name"]: {"STFT": [], "ACF": [], "Cepstrum": []} for cls in signal_classes
    }
    
    print("Running Multi-Class Noise Sweep Experiment...")
    
    for cls in signal_classes:
        name = cls["name"]
        print(f"Sweeping class: {name}...")
        
        # Generate baseline clean signal
        clean_sig = cls["gen"]()
        
        # Compute clean representations
        clean_stft = compute_stft(clean_sig, sr)
        clean_acf = compute_acf(clean_sig)
        clean_cep = compute_cepstrum(clean_sig)
        
        for amount in noise_amounts:
            # Apply noise
            noisy_sig = add_noise(clean_sig, amount)
            
            # Compute representations
            noisy_stft = compute_stft(noisy_sig, sr)
            noisy_acf = compute_acf(noisy_sig)
            noisy_cep = compute_cepstrum(noisy_sig)
            
            # Calculate similarities
            stft_sim = cosine_similarity(clean_stft, noisy_stft)
            acf_sim = cosine_similarity(clean_acf, noisy_acf)
            cep_sim = cosine_similarity(clean_cep, noisy_cep)
            
            results[name]["STFT"].append(stft_sim)
            results[name]["ACF"].append(acf_sim)
            results[name]["Cepstrum"].append(cep_sim)
            
    # Plot results: 3 subplots (STFT, ACF, Cepstrum)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    representations = ["STFT", "ACF", "Cepstrum"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    
    for idx, rep in enumerate(representations):
        ax = axes[idx]
        for c_idx, cls in enumerate(signal_classes):
            name = cls["name"]
            ax.plot(
                noise_amounts,
                results[name][rep],
                label=name,
                color=colors[c_idx],
                marker=markers[c_idx],
                linewidth=2
            )
        ax.set_title(f"{rep} Fragility Under Additive Noise", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Noise Amount (Std Dev)", fontsize=10)
        ax.set_ylabel("Similarity (Cosine)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp010_noise_fragility_by_class.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("-" * 45)
    print(f"Sweep complete! Combined plot saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()
