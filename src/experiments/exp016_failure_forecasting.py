import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.signals import harmonic_stack
from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.metrics.similarity import cosine_similarity

def run_failure_forecasting_experiment():
    print("=" * 60)
    print("RUNNING EXPERIMENT 016: FAILURE FORECASTING")
    print("=" * 60)
    
    # 1. Simulation Parameters
    sr = 22050
    duration = 3.0  # seconds
    f0 = 440.0      # Hz (Harmonic Stack base)
    
    # 2. Generate clean signal (Harmonic Stack)
    print("Generating Harmonic Stack (440 Hz)...")
    clean_sig, _ = harmonic_stack.generate(duration, sr, f0)
    
    # 3. Generate noise ramping up dynamically
    # t = 0.0 to 1.0: clean (sigma = 0.0)
    # t = 1.0 to 2.0: linear ramp to sigma = 0.02
    # t = 2.0 to 3.0: constant noise (sigma = 0.02)
    t = np.linspace(0, duration, len(clean_sig), endpoint=False)
    sigmas = np.zeros_like(t)
    sigmas[(t >= 1.0) & (t <= 2.0)] = 0.02 * (t[(t >= 1.0) & (t <= 2.0)] - 1.0)
    sigmas[t > 2.0] = 0.02
    
    np.random.seed(42)  # For reproducible results
    noise = np.random.normal(0, sigmas)
    noisy_sig = clean_sig + noise
    
    # 4. Frame-by-Frame Processing Setup
    win_size = 4096
    hop = 512
    num_frames = (len(clean_sig) - win_size) // hop + 1
    
    win = np.hanning(win_size)
    
    # Tracking arrays
    frame_times = []
    stft_sims = []
    acf_sims = []
    cep_sims_with = []
    cep_sims_without = []
    c_0_values = []
    
    # Process frames
    print(f"Processing {num_frames} frames...")
    for n in range(num_frames):
        start = n * hop
        end = start + win_size
        
        # Extract and window frames
        c_frame = clean_sig[start:end] * win
        n_frame = noisy_sig[start:end] * win
        
        # Calculate timestamps (center of frame)
        frame_times.append((start + win_size / 2) / sr)
        
        # STFT representations and similarity
        c_stft = compute_stft(c_frame, sr)
        n_stft = compute_stft(n_frame, sr)
        stft_sims.append(cosine_similarity(c_stft, n_stft))
        
        # ACF representations and similarity
        c_acf = compute_acf(c_frame)
        n_acf = compute_acf(n_frame)
        acf_sims.append(cosine_similarity(c_acf, n_acf))
        
        # Cepstrum representations
        c_cep = compute_cepstrum(c_frame)
        n_cep = compute_cepstrum(n_frame)
        
        # Cepstrum similarities (with and without c_0)
        cep_sims_with.append(cosine_similarity(c_cep, n_cep))
        cep_sims_without.append(cosine_similarity(c_cep[1:], n_cep[1:]))
        
        # Cepstrum DC offset
        c_0_values.append(n_cep[0])
        
    frame_times = np.array(frame_times)
    
    # 5. Smooth c_0 using Exponential Moving Average to prevent high-frequency noise in velocity
    alpha = 0.3
    c_0_smooth = np.zeros_like(c_0_values)
    c_0_smooth[0] = c_0_values[0]
    for i in range(1, len(c_0_values)):
        c_0_smooth[i] = alpha * c_0_values[i] + (1 - alpha) * c_0_smooth[i-1]
        
    # 6. Forecasting Algorithm
    # Velocity: Delta c_0 = c_0[n] - c_0[n-1]
    # Forecast: k = 5 frames ahead
    # Threshold: -10.0
    k = 5
    threshold = -10.0
    forecast_flags = np.zeros(num_frames)
    forecast_values = np.zeros(num_frames)
    
    for n in range(1, num_frames):
        vel = c_0_smooth[n] - c_0_smooth[n-1]
        fc = c_0_smooth[n] + k * vel
        forecast_values[n] = fc
        
        # Forecast is HIGH if extrapolated value crosses threshold, but current hasn't yet
        if fc >= threshold and c_0_smooth[n] < threshold:
            forecast_flags[n] = 1.0
            
    # 7. Generate a Stunning Dark-Mode Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Colors
    color_wave = "#a6cee3"
    color_stft = "#a6d854"
    color_acf = "#377eb8"
    color_cep_with = "#e7298a"
    color_cep_without = "#e31a1c"
    color_danger = "#ff7f00"
    color_alert = "#e31a1c"
    
    # --- Panel 1: Waveform and Noise Level ---
    ax1 = axes[0]
    ax1.plot(t, noisy_sig, color=color_wave, alpha=0.6, label="Noisy Signal")
    ax1.plot(t, clean_sig, color="#33a02c", alpha=0.3, label="Clean Signal (underlying)")
    ax1.set_title("Time-Varying Audio Signal (Ramping Additive Noise)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    
    # Overlay noise standard deviation standard scale on twinx
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, sigmas, color="#fdbf6f", linestyle=":", linewidth=2, label="Noise Std Dev (σ)")
    ax1_twin.set_ylabel("Noise Std Dev (σ)", color="#fdbf6f", fontsize=11)
    ax1_twin.tick_params(colors="#fdbf6f")
    ax1_twin.legend(loc="upper right", fontsize=9)
    
    # --- Panel 2: Representation Similarities ---
    ax2 = axes[1]
    ax2.plot(frame_times, stft_sims, color=color_stft, linestyle="--", alpha=0.7, label="STFT Cosine Sim")
    ax2.plot(frame_times, acf_sims, color=color_acf, linestyle="--", alpha=0.7, label="ACF Cosine Sim")
    ax2.plot(frame_times, cep_sims_with, color=color_cep_with, alpha=0.5, label="Cepstrum Cosine Sim (with c0)")
    ax2.plot(frame_times, cep_sims_without, color=color_cep_without, linewidth=2.5, label="Cepstrum Similarity (without c0 - Harmonic Collapse)")
    ax2.set_title("Representation Similarities over Time", fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)
    
    # --- Panel 3: Danger Signal, Forecast & Alerts ---
    ax3 = axes[2]
    # Plot danger signal c_0
    ax3.plot(frame_times, c_0_values, color="#984ea3", alpha=0.4, label="c0 (DC offset - raw)")
    ax3.plot(frame_times, c_0_smooth, color=color_danger, linewidth=2, label="c0 (DC offset - smoothed)")
    ax3.axhline(threshold, color="yellow", linestyle="--", alpha=0.7, label="Collapse Threshold (-10.0)")
    ax3.set_title("Cepstrum Danger Signal & Failure Forecast Alert", fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylabel("DC Coefficient (c0)", fontsize=11)
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(loc="lower left", fontsize=9)
    
    # Plot forecast flag on twinx
    ax3_twin = ax3.twinx()
    # Fill alert regions where forecast_flags is 1
    ax3_twin.fill_between(frame_times, 0, forecast_flags, where=(forecast_flags == 1), 
                          color=color_alert, alpha=0.3, label="Forecast Failure Alert")
    ax3_twin.plot(frame_times, forecast_flags, color=color_alert, linewidth=2, label="Forecast Flag (HIGH/LOW)")
    ax3_twin.set_ylabel("Failure Forecast Flag", color=color_alert, fontsize=11)
    ax3_twin.set_ylim(-0.1, 1.1)
    ax3_twin.tick_params(colors=color_alert)
    ax3_twin.legend(loc="upper right", fontsize=9)
    
    # 8. Save Plot
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp016_failure_forecasting.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print(f"Simulation completed! Plot saved to: {output_path}")
    print("-" * 60)
    
    # Print warning frame reports
    alert_frames = np.where(forecast_flags == 1.0)[0]
    if len(alert_frames) > 0:
        first_alert_idx = alert_frames[0]
        first_alert_time = frame_times[first_alert_idx]
        print(f"SUCCESS: Failure Forecast Alert triggered at Frame {first_alert_idx} (t={first_alert_time:.3f}s)")
        
        # Find where actual c_0 crosses threshold
        actual_collapse_idx = np.where(c_0_smooth >= threshold)[0][0]
        actual_collapse_time = frame_times[actual_collapse_idx]
        print(f"Actual c_0 threshold crossing at Frame {actual_collapse_idx} (t={actual_collapse_time:.3f}s)")
        
        frames_early = actual_collapse_idx - first_alert_idx
        time_early_ms = (actual_collapse_time - first_alert_time) * 1000
        print(f"Alert was raised {frames_early} frames ({time_early_ms:.1f} ms) BEFORE actual threshold crossing!")
    else:
        print("WARNING: Failure Forecast Alert did not trigger.")
    
if __name__ == "__main__":
    run_failure_forecasting_experiment()
