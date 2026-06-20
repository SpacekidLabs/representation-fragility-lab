import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals import sine
from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.metrics.similarity import cosine_similarity

def run_automated_discovery():
    sr = 22050
    duration = 1.0
    f0 = 440.0
    
    # Generate clean base signal (440 Hz Sine)
    base_sig, _ = sine.generate(duration, sr, f0=f0)
    
    # Compute clean representations
    clean_stft = compute_stft(base_sig, sr)
    clean_acf = compute_acf(base_sig)
    clean_cep = compute_cepstrum(base_sig)
    
    # Search parameters
    np.random.seed(42)  # For reproducibility
    num_iterations = 2000
    
    best_acf_sim = 1.0
    best_stft_sim = 0.0
    best_cep_sim = 0.0
    best_params = None
    
    # We will track all valid candidates that satisfy the constraints
    valid_candidates = []
    
    print(f"Running automated search for ACF blind spots ({num_iterations} iterations)...")
    
    for i in range(num_iterations):
        # Sample random parameters
        pitch_cents = np.random.uniform(-50.0, 50.0)
        time_shift_ms = np.random.uniform(0.0, 50.0)
        phase_shift_rad = np.random.uniform(0.0, 2.0 * np.pi)
        noise_std = np.random.uniform(0.0, 0.05)
        
        # Generate target signal directly to optimize speed
        f = f0 * (2 ** (pitch_cents / 1200.0))
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        t_shifted = t + (time_shift_ms / 1000.0)
        
        target_sig = np.sin(2 * np.pi * f * t_shifted + phase_shift_rad)
        if noise_std > 0:
            target_sig = target_sig + np.random.normal(0, noise_std, size=target_sig.shape)
            
        # Compute representations
        target_stft = compute_stft(target_sig, sr)
        target_acf = compute_acf(target_sig)
        target_cep = compute_cepstrum(target_sig)
        
        # Measure similarities
        sim_stft = cosine_similarity(clean_stft, target_stft)
        sim_acf = cosine_similarity(clean_acf, target_acf)
        sim_cep = cosine_similarity(clean_cep, target_cep)
        
        # Check constraints: Keep STFT > 0.95 and Cep > 0.95 (or as high as possible)
        # Since Cepstrum is highly sensitive to pitch shift, we might have to settle for a relaxed constraint
        # if no candidates meet the strict >0.95 constraint.
        if sim_stft >= 0.95 and sim_cep >= 0.95:
            valid_candidates.append({
                "params": (pitch_cents, time_shift_ms, phase_shift_rad, noise_std),
                "stft": sim_stft,
                "acf": sim_acf,
                "cep": sim_cep
            })
            
    print(f"Found {len(valid_candidates)} candidates satisfying constraints (STFT > 0.95, Cepstrum > 0.95).")
    
    if len(valid_candidates) > 0:
        # Find candidate that minimizes ACF similarity
        best_candidate = min(valid_candidates, key=lambda x: x["acf"])
        best_params = best_candidate["params"]
        best_stft_sim = best_candidate["stft"]
        best_acf_sim = best_candidate["acf"]
        best_cep_sim = best_candidate["cep"]
    else:
        # If no candidates satisfied the strict constraints, search with a relaxed score function
        # to maximize (STFT + Cepstrum) - 2 * ACF
        print("Strict constraints not met. Running optimization with relaxed compromise score...")
        best_score = -999.0
        for i in range(num_iterations):
            pitch_cents = np.random.uniform(-5.0, 5.0)  # Narrower pitch search to keep Cepstrum stable
            time_shift_ms = np.random.uniform(0.0, 50.0)
            phase_shift_rad = np.random.uniform(0.0, 2.0 * np.pi)
            noise_std = np.random.uniform(0.0, 0.01)     # Low noise to keep Cepstrum stable
            
            f = f0 * (2 ** (pitch_cents / 1200.0))
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            t_shifted = t + (time_shift_ms / 1000.0)
            
            target_sig = np.sin(2 * np.pi * f * t_shifted + phase_shift_rad)
            if noise_std > 0:
                target_sig = target_sig + np.random.normal(0, noise_std, size=target_sig.shape)
                
            sim_stft = cosine_similarity(clean_stft, compute_stft(target_sig, sr))
            sim_acf = cosine_similarity(clean_acf, compute_acf(target_sig))
            sim_cep = cosine_similarity(clean_cep, compute_cepstrum(target_sig))
            
            # Score: maximize STFT and Cepstrum while minimizing ACF
            score = (sim_stft + sim_cep) - 2.0 * sim_acf
            
            if score > best_score:
                best_score = score
                best_params = (pitch_cents, time_shift_ms, phase_shift_rad, noise_std)
                best_stft_sim = sim_stft
                best_acf_sim = sim_acf
                best_cep_sim = sim_cep
                
    pitch_cents, time_shift_ms, phase_shift_rad, noise_std = best_params
    
    print("\n" + "=" * 45)
    print("Best Probing Parameter Set Found:")
    print("-" * 45)
    print(f"Pitch Shift   : {pitch_cents:.4f} cents")
    print(f"Time Shift    : {time_shift_ms:.4f} ms")
    print(f"Phase Shift   : {phase_shift_rad:.4f} rad")
    print(f"Noise Std Dev : {noise_std:.4f}")
    print("-" * 45)
    print("Resulting Similarities:")
    print(f"  STFT Similarity     : {best_stft_sim:.4f}  (Constraint: >0.95)")
    print(f"  Cepstrum Similarity : {best_cep_sim:.4f}  (Constraint: >0.95)")
    print(f"  ACF Similarity      : {best_acf_sim:.4f}  (Objective: Minimize)")
    print("=" * 45)
    
    # -------------------------------------------------------------
    # Visualization: Bar Chart
    # -------------------------------------------------------------
    reps = ["STFT", "ACF", "Cepstrum"]
    scores = [best_stft_sim, best_acf_sim, best_cep_sim]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(reps, scores, color=colors, width=0.5)
    plt.title("Automated Blind Spot Discovery: Selective ACF Collapse", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Representation Cosine Similarity", fontsize=10)
    plt.ylim(0, 1.15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp013_automated_discovery.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Result plot saved successfully to: {output_path}")

if __name__ == "__main__":
    run_automated_discovery()
