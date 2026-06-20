import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals import sine, harmonic_stack, impulse
from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.metrics.similarity import cosine_similarity

def run_targeted_probing():
    sr = 22050
    duration = 1.0
    
    # -------------------------------------------------------------
    # Probe A: ACF Attack (Pitch Translation)
    # Goal: Maximize ACF failure, Minimize STFT change
    # -------------------------------------------------------------
    print("Running Probe A (ACF Attack)...")
    base_a, _ = sine.generate(duration, sr, f0=440.0)
    target_a, _ = sine.generate(duration, sr, f0=442.54)  # 10 cents shift
    
    rep_base_stft_a = compute_stft(base_a, sr)
    rep_target_stft_a = compute_stft(target_a, sr)
    sim_stft_a = cosine_similarity(rep_base_stft_a, rep_target_stft_a)
    
    rep_base_acf_a = compute_acf(base_a)
    rep_target_acf_a = compute_acf(target_a)
    sim_acf_a = cosine_similarity(rep_base_acf_a, rep_target_acf_a)
    
    rep_base_cep_a = compute_cepstrum(base_a)
    rep_target_cep_a = compute_cepstrum(target_a)
    sim_cep_a = cosine_similarity(rep_base_cep_a, rep_target_cep_a)
    
    # -------------------------------------------------------------
    # Probe B: Cepstrum Attack (Noise Floor Amplification)
    # Goal: Maximize Cepstrum failure, Minimize STFT/ACF change
    # -------------------------------------------------------------
    print("Running Probe B (Cepstrum Attack)...")
    base_b, _ = harmonic_stack.generate(duration, sr, f0=440.0, num_harmonics=5)
    # Add quiet Gaussian noise (std dev = 0.01)
    noise = np.random.normal(0, 0.01, size=base_b.shape)
    target_b = base_b + noise
    
    rep_base_stft_b = compute_stft(base_b, sr)
    rep_target_stft_b = compute_stft(target_b, sr)
    sim_stft_b = cosine_similarity(rep_base_stft_b, rep_target_stft_b)
    
    rep_base_acf_b = compute_acf(base_b)
    rep_target_acf_b = compute_acf(target_b)
    sim_acf_b = cosine_similarity(rep_base_acf_b, rep_target_acf_b)
    
    rep_base_cep_b = compute_cepstrum(base_b)
    rep_target_cep_b = compute_cepstrum(target_b)
    sim_cep_b = cosine_similarity(rep_base_cep_b, rep_target_cep_b)
    
    # -------------------------------------------------------------
    # Probe C: STFT Attack (Time Translation)
    # Goal: Maximize STFT failure, Minimize ACF/Cepstrum change
    # -------------------------------------------------------------
    print("Running Probe C (STFT Attack)...")
    # Shift impulse from t=0.2s to t=0.5s
    base_c, _ = impulse.generate(duration, sr, delay_s=0.2)
    target_c, _ = impulse.generate(duration, sr, delay_s=0.5)
    
    rep_base_stft_c = compute_stft(base_c, sr)
    rep_target_stft_c = compute_stft(target_c, sr)
    sim_stft_c = cosine_similarity(rep_base_stft_c, rep_target_stft_c)
    
    rep_base_acf_c = compute_acf(base_c)
    rep_target_acf_c = compute_acf(target_c)
    sim_acf_c = cosine_similarity(rep_base_acf_c, rep_target_acf_c)
    
    rep_base_cep_c = compute_cepstrum(base_c)
    rep_target_cep_c = compute_cepstrum(target_c)
    sim_cep_c = cosine_similarity(rep_base_cep_c, rep_target_cep_c)
    
    # Print results
    print("\n" + "=" * 55)
    print(f"{'Representation':<20}{'Probe A (ACF)':<12}{'Probe B (Cep)':<12}{'Probe C (STFT)':<12}")
    print("-" * 55)
    print(f"{'STFT':<20}{sim_stft_a:<12.4f}{sim_stft_b:<12.4f}{sim_stft_c:<12.4f}")
    print(f"{'ACF':<20}{sim_acf_a:<12.4f}{sim_acf_b:<12.4f}{sim_acf_c:<12.4f}")
    print(f"{'Cepstrum':<20}{sim_cep_a:<12.4f}{sim_cep_b:<12.4f}{sim_cep_c:<12.4f}")
    print("=" * 55)
    
    # -------------------------------------------------------------
    # Visualization: Bar Chart
    # -------------------------------------------------------------
    probes = ["Probe A\n(ACF Attack)", "Probe B\n(Cepstrum Attack)", "Probe C\n(STFT Attack)"]
    stft_scores = [sim_stft_a, sim_stft_b, sim_stft_c]
    acf_scores = [sim_acf_a, sim_acf_b, sim_acf_c]
    cep_scores = [sim_cep_a, sim_cep_b, sim_cep_c]
    
    x = np.arange(len(probes))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, stft_scores, width, label="STFT", color="#1f77b4")
    plt.bar(x, acf_scores, width, label="ACF", color="#ff7f0e")
    plt.bar(x + width, cep_scores, width, label="Cepstrum", color="#2ca02c")
    
    plt.title("Selective Representation Failure Under Targeted Probing", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Targeted Probe / Adversarial Attack", fontsize=12)
    plt.ylabel("Representation Cosine Similarity", fontsize=12)
    plt.xticks(x, probes, fontsize=10)
    plt.ylim(0, 1.15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    # Add values on top of bars
    for i in range(len(probes)):
        plt.text(i - width, stft_scores[i] + 0.02, f"{stft_scores[i]:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.text(i, acf_scores[i] + 0.02, f"{acf_scores[i]:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.text(i + width, cep_scores[i] + 0.02, f"{cep_scores[i]:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp012_targeted_probing.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == "__main__":
    run_targeted_probing()
