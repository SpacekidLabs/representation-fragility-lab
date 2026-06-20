import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals import harmonic_stack
from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.perturbations.noise import add_noise
from src.perturbations.pitch_shift import apply_pitch_shift
from src.perturbations.filter import lowpass_filter
from src.perturbations.harmonic_removal import generate_harmonic_complex
from src.metrics.similarity import cosine_similarity, pearson_correlation, euclidean_similarity

def run_metric_experiment():
    sr = 22050
    duration = 1.0
    f0 = 440.0
    
    # Base signal (Harmonic Stack)
    base_sig, _ = harmonic_stack.generate(duration, sr, f0)
    
    # Clean representations
    clean_reps = {
        "STFT": compute_stft(base_sig, sr),
        "ACF": compute_acf(base_sig),
        "Cepstrum": compute_cepstrum(base_sig),
    }
    
    # Perturbation sweeps
    noise_sweep = np.linspace(0.0, 1.0, 11)
    pitch_sweep = np.arange(-6, 7)
    filter_sweep = [3000, 2500, 2000, 1500, 1000, 750, 500, 400, 300, 200]
    harm_sweep = [0.0, 25.0, 50.0, 75.0, 100.0]
    
    # Grid Plot Setup: 3 Rows (STFT, ACF, Cepstrum) x 4 Columns (Noise, Pitch, Filter, Overtone Loss)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    representations = ["STFT", "ACF", "Cepstrum"]
    
    for r_idx, rep_name in enumerate(representations):
        print(f"Processing representation: {rep_name}...")
        
        # --- Column 1: Additive Noise ---
        ax = axes[r_idx, 0]
        cos, pea, euc = [], [], []
        for amount in noise_sweep:
            noisy_sig = add_noise(base_sig, amount)
            if rep_name == "STFT":
                perturbed = compute_stft(noisy_sig, sr)
            elif rep_name == "ACF":
                perturbed = compute_acf(noisy_sig)
            else:
                perturbed = compute_cepstrum(noisy_sig)
            cos.append(cosine_similarity(clean_reps[rep_name], perturbed))
            pea.append(pearson_correlation(clean_reps[rep_name], perturbed))
            euc.append(euclidean_similarity(clean_reps[rep_name], perturbed))
        
        ax.plot(noise_sweep, cos, label="Cosine", color="#1f77b4", marker="o")
        ax.plot(noise_sweep, pea, label="Pearson", color="#ff7f0e", linestyle="--", marker="s")
        ax.plot(noise_sweep, euc, label="Euclidean Sim", color="#2ca02c", linestyle=":", marker="^")
        ax.set_title(f"{rep_name} - Noise", fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        
        # --- Column 2: Pitch Shift ---
        ax = axes[r_idx, 1]
        cos, pea, euc = [], [], []
        for semitone in pitch_sweep:
            shifted_sig = apply_pitch_shift(base_sig, sr, semitone)
            if rep_name == "STFT":
                perturbed = compute_stft(shifted_sig, sr)
            elif rep_name == "ACF":
                perturbed = compute_acf(shifted_sig)
            else:
                perturbed = compute_cepstrum(shifted_sig)
            cos.append(cosine_similarity(clean_reps[rep_name], perturbed))
            pea.append(pearson_correlation(clean_reps[rep_name], perturbed))
            euc.append(euclidean_similarity(clean_reps[rep_name], perturbed))
        
        ax.plot(pitch_sweep, cos, color="#1f77b4", marker="o")
        ax.plot(pitch_sweep, pea, color="#ff7f0e", linestyle="--", marker="s")
        ax.plot(pitch_sweep, euc, color="#2ca02c", linestyle=":", marker="^")
        ax.set_title(f"{rep_name} - Pitch Shift", fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        
        # --- Column 3: Low Pass Filter ---
        ax = axes[r_idx, 2]
        cos, pea, euc = [], [], []
        for cutoff in filter_sweep:
            filtered_sig = lowpass_filter(base_sig, sr, cutoff)
            if rep_name == "STFT":
                perturbed = compute_stft(filtered_sig, sr)
            elif rep_name == "ACF":
                perturbed = compute_acf(filtered_sig)
            else:
                perturbed = compute_cepstrum(filtered_sig)
            cos.append(cosine_similarity(clean_reps[rep_name], perturbed))
            pea.append(pearson_correlation(clean_reps[rep_name], perturbed))
            euc.append(euclidean_similarity(clean_reps[rep_name], perturbed))
        
        ax.plot(filter_sweep, cos, color="#1f77b4", marker="o")
        ax.plot(filter_sweep, pea, color="#ff7f0e", linestyle="--", marker="s")
        ax.plot(filter_sweep, euc, color="#2ca02c", linestyle=":", marker="^")
        ax.set_title(f"{rep_name} - LP Filter", fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.invert_xaxis()  # Invert to show increasing perturbation to the right
        ax.set_ylim(-1.05, 1.05)
        
        # --- Column 4: Harmonic Removal ---
        ax = axes[r_idx, 3]
        cos, pea, euc = [], [], []
        for pct in harm_sweep:
            stripped_sig = generate_harmonic_complex(f0, sr, duration, removal_pct=pct)
            if rep_name == "STFT":
                perturbed = compute_stft(stripped_sig, sr)
            elif rep_name == "ACF":
                perturbed = compute_acf(stripped_sig)
            else:
                perturbed = compute_cepstrum(stripped_sig)
            cos.append(cosine_similarity(clean_reps[rep_name], perturbed))
            pea.append(pearson_correlation(clean_reps[rep_name], perturbed))
            euc.append(euclidean_similarity(clean_reps[rep_name], perturbed))
        
        ax.plot(harm_sweep, cos, color="#1f77b4", marker="o")
        ax.plot(harm_sweep, pea, color="#ff7f0e", linestyle="--", marker="s")
        ax.plot(harm_sweep, euc, color="#2ca02c", linestyle=":", marker="^")
        ax.set_title(f"{rep_name} - Overtone Loss", fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        
        # Add legends to the first column plots
        if r_idx == 0:
            axes[0, 0].legend(loc="lower left", fontsize=8)
            
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp011_metric_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Metric comparison complete! Grid plot saved to: {output_path}")

if __name__ == "__main__":
    run_metric_experiment()
