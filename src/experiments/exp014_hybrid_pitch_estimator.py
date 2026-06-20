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

def estimate_pitch_acf(signal, sr):
    acf = compute_acf(signal)
    min_lag = int(sr / 1000)  # ~22 samples (1000 Hz)
    max_lag = int(sr / 80)    # ~275 samples (80 Hz)
    if max_lag > len(acf):
        max_lag = len(acf)
    acf_range = acf[min_lag:max_lag]
    lag_idx = np.argmax(acf_range) + min_lag
    pitch = sr / lag_idx
    confidence = acf[lag_idx] / acf[0] if acf[0] > 0 else 0.0
    return pitch, confidence

def estimate_pitch_cepstrum(signal, sr):
    cep = compute_cepstrum(signal)
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    if max_lag > len(cep):
        max_lag = len(cep)
    cep_range = np.abs(cep[min_lag:max_lag])
    q_idx = np.argmax(cep_range) + min_lag
    pitch = sr / q_idx
    confidence = np.abs(cep[q_idx]) / (np.linalg.norm(cep[min_lag:max_lag]) + 1e-10)
    return pitch, confidence

def estimate_pitch_stft(signal, sr):
    stft_mag = compute_stft(signal, sr)
    avg_spec = np.mean(stft_mag, axis=1)
    N_fft = 2048
    freqs = np.fft.rfftfreq(N_fft, d=1/sr)
    
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    
    spec_range = avg_spec[min_bin:max_bin]
    bin_idx = np.argmax(spec_range) + min_bin
    pitch = freqs[bin_idx]
    confidence = avg_spec[bin_idx] / (np.linalg.norm(avg_spec[min_bin:max_bin]) + 1e-10)
    return pitch, confidence

def hybrid_pitch_estimate(pitches, confidences, tolerance=20.0):
    scores = []
    for p_candidate in pitches:
        score = 0.0
        for p, c in zip(pitches, confidences):
            score += c * np.exp(-np.abs(p_candidate - p) / tolerance)
        scores.append(score)
    best_idx = np.argmax(scores)
    return pitches[best_idx]

def run_hybrid_pitch_experiment():
    sr = 22050
    duration = 1.0
    f0_base = 440.0
    
    # Grid setup: 2x2 plots (Noise, Filter, Harmonic Removal, Pitch Shift)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # -------------------------------------------------------------
    # Plot 1: Additive Noise Sweep
    # -------------------------------------------------------------
    print("Running Additive Noise sweep...")
    noise_sweep = np.linspace(0.0, 1.0, 11)
    errors = {"STFT": [], "ACF": [], "Cepstrum": [], "Hybrid": []}
    
    for amount in noise_sweep:
        # Generate base and add noise
        sig, _ = harmonic_stack.generate(duration, sr, f0_base)
        noisy_sig = add_noise(sig, amount)
        
        p_stft, c_stft = estimate_pitch_stft(noisy_sig, sr)
        p_acf, c_acf = estimate_pitch_acf(noisy_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum(noisy_sig, sr)
        p_hyb = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors["STFT"].append(np.abs(p_stft - f0_base))
        errors["ACF"].append(np.abs(p_acf - f0_base))
        errors["Cepstrum"].append(np.abs(p_cep - f0_base))
        errors["Hybrid"].append(np.abs(p_hyb - f0_base))
        
    ax = axes[0, 0]
    ax.plot(noise_sweep, errors["STFT"], label="STFT", color="#1f77b4", marker="o")
    ax.plot(noise_sweep, errors["ACF"], label="ACF", color="#ff7f0e", marker="s")
    ax.plot(noise_sweep, errors["Cepstrum"], label="Cepstrum", color="#2ca02c", marker="^")
    ax.plot(noise_sweep, errors["Hybrid"], label="Hybrid Fusion", color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Pitch Error under Additive Noise", fontweight="bold")
    ax.set_xlabel("Noise Amount (Std Dev)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    
    # -------------------------------------------------------------
    # Plot 2: Low Pass Filter Sweep
    # -------------------------------------------------------------
    print("Running LP Filter sweep...")
    filter_sweep = [3000, 2500, 2000, 1500, 1000, 750, 500, 400, 300, 200]
    errors = {"STFT": [], "ACF": [], "Cepstrum": [], "Hybrid": []}
    
    for cutoff in filter_sweep:
        sig, _ = harmonic_stack.generate(duration, sr, f0_base)
        filtered_sig = lowpass_filter(sig, sr, cutoff)
        
        p_stft, c_stft = estimate_pitch_stft(filtered_sig, sr)
        p_acf, c_acf = estimate_pitch_acf(filtered_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum(filtered_sig, sr)
        p_hyb = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors["STFT"].append(np.abs(p_stft - f0_base))
        errors["ACF"].append(np.abs(p_acf - f0_base))
        errors["Cepstrum"].append(np.abs(p_cep - f0_base))
        errors["Hybrid"].append(np.abs(p_hyb - f0_base))
        
    ax = axes[0, 1]
    ax.plot(filter_sweep, errors["STFT"], color="#1f77b4", marker="o")
    ax.plot(filter_sweep, errors["ACF"], color="#ff7f0e", marker="s")
    ax.plot(filter_sweep, errors["Cepstrum"], color="#2ca02c", marker="^")
    ax.plot(filter_sweep, errors["Hybrid"], color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Pitch Error under Low-Pass Filtering", fontweight="bold")
    ax.set_xlabel("Cutoff Frequency (Hz)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.invert_xaxis()
    
    # -------------------------------------------------------------
    # Plot 3: Harmonic Removal Sweep
    # -------------------------------------------------------------
    print("Running Harmonic Removal sweep...")
    harm_sweep = [0.0, 25.0, 50.0, 75.0, 100.0]
    errors = {"STFT": [], "ACF": [], "Cepstrum": [], "Hybrid": []}
    
    for pct in harm_sweep:
        stripped_sig = generate_harmonic_complex(f0_base, sr, duration, removal_pct=pct)
        
        p_stft, c_stft = estimate_pitch_stft(stripped_sig, sr)
        p_acf, c_acf = estimate_pitch_acf(stripped_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum(stripped_sig, sr)
        p_hyb = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors["STFT"].append(np.abs(p_stft - f0_base))
        errors["ACF"].append(np.abs(p_acf - f0_base))
        errors["Cepstrum"].append(np.abs(p_cep - f0_base))
        errors["Hybrid"].append(np.abs(p_hyb - f0_base))
        
    ax = axes[1, 0]
    ax.plot(harm_sweep, errors["STFT"], color="#1f77b4", marker="o")
    ax.plot(harm_sweep, errors["ACF"], color="#ff7f0e", marker="s")
    ax.plot(harm_sweep, errors["Cepstrum"], color="#2ca02c", marker="^")
    ax.plot(harm_sweep, errors["Hybrid"], color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Pitch Error under Overtone Loss", fontweight="bold")
    ax.set_xlabel("Harmonics Removed (%)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # -------------------------------------------------------------
    # Plot 4: Pitch Shift Sweep
    # -------------------------------------------------------------
    print("Running Pitch Shift sweep...")
    pitch_sweep = np.arange(-5, 6)  # -5 to +5 semitones
    errors = {"STFT": [], "ACF": [], "Cepstrum": [], "Hybrid": []}
    
    for semitones in pitch_sweep:
        sig, _ = harmonic_stack.generate(duration, sr, f0_base)
        shifted_sig = apply_pitch_shift(sig, sr, semitones)
        
        f0_shifted = f0_base * (2 ** (semitones / 12.0))
        
        p_stft, c_stft = estimate_pitch_stft(shifted_sig, sr)
        p_acf, c_acf = estimate_pitch_acf(shifted_sig, sr)
        p_cep, c_cep = estimate_pitch_cepstrum(shifted_sig, sr)
        p_hyb = hybrid_pitch_estimate([p_acf, p_cep, p_stft], [c_acf, c_cep, c_stft])
        
        errors["STFT"].append(np.abs(p_stft - f0_shifted))
        errors["ACF"].append(np.abs(p_acf - f0_shifted))
        errors["Cepstrum"].append(np.abs(p_cep - f0_shifted))
        errors["Hybrid"].append(np.abs(p_hyb - f0_shifted))
        
    ax = axes[1, 1]
    ax.plot(pitch_sweep, errors["STFT"], color="#1f77b4", marker="o")
    ax.plot(pitch_sweep, errors["ACF"], color="#ff7f0e", marker="s")
    ax.plot(pitch_sweep, errors["Cepstrum"], color="#2ca02c", marker="^")
    ax.plot(pitch_sweep, errors["Hybrid"], color="#d62728", linewidth=2.5, marker="D")
    ax.set_title("Pitch Error under Pitch Shift", fontweight="bold")
    ax.set_xlabel("Pitch Shift (Semitones)")
    ax.set_ylabel("Absolute Pitch Error (Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp014_hybrid_pitch_estimator.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("-" * 50)
    print(f"Hybrid pitch experiment complete! Plot saved to: {output_path}")

if __name__ == "__main__":
    run_hybrid_pitch_experiment()
