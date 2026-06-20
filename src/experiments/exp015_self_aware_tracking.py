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
from src.perturbations.filter import lowpass_filter
from src.perturbations.harmonic_removal import generate_harmonic_complex

def estimate_pitch_acf_self_aware(signal, sr):
    acf = compute_acf(signal)
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    if max_lag > len(acf):
        max_lag = len(acf)
    acf_range = acf[min_lag:max_lag]
    lag_idx = np.argmax(acf_range) + min_lag
    pitch = sr / lag_idx
    
    # Peak Prominence Confidence
    peak_val = acf[lag_idx]
    mean_val = np.mean(acf_range)
    max_val = np.max(acf_range)
    min_val = np.min(acf_range)
    confidence = (peak_val - mean_val) / (max_val - min_val + 1e-10)
    return pitch, float(np.clip(confidence, 0.0, 1.0))

def estimate_pitch_cepstrum_self_aware(signal, sr):
    cep = compute_cepstrum(signal)
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    if max_lag > len(cep):
        max_lag = len(cep)
    cep_range = np.abs(cep[min_lag:max_lag])
    q_idx = np.argmax(cep_range) + min_lag
    pitch = sr / q_idx
    
    # Inverse DC Shift Confidence
    c_0 = cep[0]
    confidence = 1.0 - (c_0 - (-23.0)) / 25.4
    return pitch, float(np.clip(confidence, 0.0, 1.0))

def estimate_pitch_stft_self_aware(signal, sr):
    stft_mag = compute_stft(signal, sr)
    avg_spec = np.mean(stft_mag, axis=1)
    N_fft = 2048
    freqs = np.fft.rfftfreq(N_fft, d=1/sr)
    
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    
    spec_range = avg_spec[min_bin:max_bin]
    bin_idx = np.argmax(spec_range) + min_bin
    pitch = freqs[bin_idx]
    
    # Peak Dominance Ratio Confidence
    peak_val = avg_spec[bin_idx]
    mean_val = np.mean(spec_range)
    confidence = (peak_val - mean_val) / (peak_val + 1e-10)
    return pitch, float(np.clip(confidence, 0.0, 1.0))

def hybrid_pitch_estimate(pitches, confidences, tolerance=20.0):
    scores = []
    for p_candidate in pitches:
        score = 0.0
        for p, c in zip(pitches, confidences):
            score += c * np.exp(-np.abs(p_candidate - p) / tolerance)
        scores.append(score)
    best_idx = np.argmax(scores)
    return pitches[best_idx]

def run_self_aware_experiment():
    sr = 22050
    duration = 1.0
    f0_base = 440.0
    
    # 1 row x 3 columns plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.25)
    
    # -------------------------------------------------------------
    # Panel 1: Additive Noise Sweep
    # -------------------------------------------------------------
    print("Running Additive Noise sweep...")
    noise_sweep = np.linspace(0.0, 1.0, 11)
    errors_a = []
    errors_b = []
    
    for amount in noise_sweep:
        sig, _ = harmonic_stack.generate(duration, sr, f0_base)
        noisy_sig = add_noise(sig, amount)
        
        p_acf, c_acf = estimate_pitch_acf_self_aware(noisy_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum_self_aware(noisy_sig, sr)
        p_stft, c_stft = estimate_pitch_stft_self_aware(noisy_sig, sr)
        
        # Version A: Agreement only (Flat confidence = 1.0)
        p_hyb_a = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [1.0, 1.0, 1.0])
        # Version B: Agreement + Dynamic Self-Confidence
        p_hyb_b = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors_a.append(np.abs(p_hyb_a - f0_base))
        errors_b.append(np.abs(p_hyb_b - f0_base))
        
    ax = axes[0]
    ax.plot(noise_sweep, errors_a, label="Version A (Agreement Only)", color="#1f77b4", linestyle="--", marker="o")
    ax.plot(noise_sweep, errors_b, label="Version B (Agreement + Self-Confidence)", color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Fusion Pitch Error under Additive Noise", fontweight="bold")
    ax.set_xlabel("Noise Amount (Std Dev)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 2: Low Pass Filter Sweep
    # -------------------------------------------------------------
    print("Running LP Filter sweep...")
    filter_sweep = [3000, 2500, 2000, 1500, 1000, 750, 500, 400, 300, 200]
    errors_a = []
    errors_b = []
    
    for cutoff in filter_sweep:
        sig, _ = harmonic_stack.generate(duration, sr, f0_base)
        filtered_sig = lowpass_filter(sig, sr, cutoff)
        
        p_acf, c_acf = estimate_pitch_acf_self_aware(filtered_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum_self_aware(filtered_sig, sr)
        p_stft, c_stft = estimate_pitch_stft_self_aware(filtered_sig, sr)
        
        p_hyb_a = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [1.0, 1.0, 1.0])
        p_hyb_b = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors_a.append(np.abs(p_hyb_a - f0_base))
        errors_b.append(np.abs(p_hyb_b - f0_base))
        
    ax = axes[1]
    ax.plot(filter_sweep, errors_a, color="#1f77b4", linestyle="--", marker="o")
    ax.plot(filter_sweep, errors_b, color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Fusion Pitch Error under Low-Pass Filtering", fontweight="bold")
    ax.set_xlabel("Cutoff Frequency (Hz)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.invert_xaxis()
    
    # -------------------------------------------------------------
    # Panel 3: Harmonic Removal Sweep
    # -------------------------------------------------------------
    print("Running Harmonic Removal sweep...")
    harm_sweep = [0.0, 25.0, 50.0, 75.0, 100.0]
    errors_a = []
    errors_b = []
    
    for pct in harm_sweep:
        stripped_sig = generate_harmonic_complex(f0_base, sr, duration, removal_pct=pct)
        
        p_acf, c_acf = estimate_pitch_acf_self_aware(stripped_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum_self_aware(stripped_sig, sr)
        p_stft, c_stft = estimate_pitch_stft_self_aware(stripped_sig, sr)
        
        p_hyb_a = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [1.0, 1.0, 1.0])
        p_hyb_b = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors_a.append(np.abs(p_hyb_a - f0_base))
        errors_b.append(np.abs(p_hyb_b - f0_base))
        
    ax = axes[2]
    ax.plot(harm_sweep, errors_a, color="#1f77b4", linestyle="--", marker="o")
    ax.plot(harm_sweep, errors_b, color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Fusion Pitch Error under Overtone Loss", fontweight="bold")
    ax.set_xlabel("Harmonics Removed (%)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp015_self_aware_tracking.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("-" * 50)
    print(f"Self-aware hybrid experiment complete! Plot saved to: {output_path}")

if __name__ == "__main__":
    run_self_aware_experiment()
