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
from src.metrics.similarity import cosine_similarity

# Pitch estimation functions based on Exp 015
def estimate_pitch_acf(acf, sr):
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

def estimate_pitch_cepstrum(cep, sr):
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

def estimate_pitch_stft(stft_mag, sr):
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

# Confidence-weighted hybrid fusion
def hybrid_pitch_estimate(pitches, confidences, tolerance=20.0):
    if len(pitches) == 0:
        return 0.0
    scores = []
    for p_candidate in pitches:
        score = 0.0
        for p, c in zip(pitches, confidences):
            score += c * np.exp(-np.abs(p_candidate - p) / tolerance)
        scores.append(score)
    best_idx = np.argmax(scores)
    return pitches[best_idx]

def run_adaptive_routing_experiment():
    print("=" * 60)
    print("RUNNING EXPERIMENT 017: ADAPTIVE ROUTING (ACTIVE AVOIDANCE)")
    print("=" * 60)
    
    # 1. Generate the 4.5s Time-Varying Signal
    sr = 22050
    duration = 4.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Define instantaneous frequency profile:
    # 0.0 to 1.5s: 440 Hz
    # 1.5 to 3.0s: 440 Hz modulated with 40 Hz vibrato at 6 Hz
    # 3.0 to 4.5s: 440 Hz
    f_profile = np.zeros_like(t)
    f_profile[t < 1.5] = 440.0
    mask_vib = (t >= 1.5) & (t < 3.0)
    f_profile[mask_vib] = 440.0 + 40.0 * np.sin(2 * np.pi * 6.0 * (t[mask_vib] - 1.5))
    f_profile[t >= 3.0] = 440.0
    
    # Generate clean signal using cumulative phase integration
    print("Synthesizing multi-phase harmonic complex...")
    clean_sig = np.zeros_like(t)
    for k in range(1, 6):
        phases_k = 2 * np.pi * np.cumsum(k * f_profile) / sr
        clean_sig += (1.0 / k) * np.sin(phases_k)
    
    # Peak normalize
    if np.max(np.abs(clean_sig)) > 0:
        clean_sig /= np.max(np.abs(clean_sig))
        
    # Generate dynamic noise standard deviation profile:
    # 0.0 to 0.5s: clean (0.0)
    # 0.5 to 1.5s: linear noise ramp to 0.02
    # 1.5 to 3.0s: mild background noise (0.005)
    # 3.0 to 4.5s: high stationary noise (1.8)
    sigmas = np.zeros_like(t)
    mask_ramp = (t >= 0.5) & (t < 1.5)
    sigmas[mask_ramp] = 0.02 * (t[mask_ramp] - 0.5)
    mask_vib_noise = (t >= 1.5) & (t < 3.0)
    sigmas[mask_vib_noise] = 0.005
    sigmas[t >= 3.0] = 1.8
    
    np.random.seed(42)
    noise = np.random.normal(0, sigmas)
    noisy_sig = clean_sig + noise
    
    # 2. Processing Setup
    hop = 512
    # The reference frame rate is based on hop size. 
    # To keep systems aligned, we center timestamps around a default 4096 window.
    # Total frames to process (using max window size buffer at the end)
    max_win = 8192
    num_frames = (len(clean_sig) - max_win) // hop + 1
    
    frame_times = np.array([(n * hop + 4096 / 2) / sr for n in range(num_frames)])
    true_pitches = np.array([f_profile[int(n * hop + 4096 / 2)] for n in range(num_frames)])
    
    # 3. RUN SIMULATION - REACTIVE SYSTEM (Fixed 4096 window)
    print("Simulating Reactive Fusion baseline...")
    reactive_errors = []
    
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    
    for n in range(num_frames):
        start = n * hop
        end = start + 4096
        
        c_frame = clean_sig[start:end] * np.hanning(4096)
        n_frame = noisy_sig[start:end] * np.hanning(4096)
        
        # Representations
        acf = compute_acf(n_frame)
        cep = compute_cepstrum(n_frame)
        stft_mag = compute_stft(n_frame, sr)
        
        # Estimates
        p_acf, _ = estimate_pitch_acf(acf, sr)
        p_cep, _ = estimate_pitch_cepstrum(cep, sr)
        p_stft, _ = estimate_pitch_stft(stft_mag, sr)
        
        # Calibrated Confidences
        c_0 = cep[0]
        c_cep = np.clip((c_0 - (-10.0)) / (-13.6 - (-10.0)), 0.0, 1.0)
        
        peak_acf = np.max(acf[min_lag:max_lag])
        ratio_acf = peak_acf / acf[0]
        c_acf = np.clip((ratio_acf - 0.1) / (0.8 - 0.1), 0.0, 1.0)
        
        # Reactive Fusion: Fuses only ACF and Cepstrum (high precision candidates)
        fused_pitch = hybrid_pitch_estimate([p_acf, p_cep], [c_acf, c_cep])
        reactive_errors.append(abs(fused_pitch - true_pitches[n]))
        
    reactive_errors = np.array(reactive_errors)
    
    # 4. RUN SIMULATION - ADAPTIVE SYSTEM
    print("Simulating Adaptive Fusion...")
    adaptive_errors = []
    
    # History variables for forecast and jitter
    cep_c_0_smooth = []
    acf_pitch_history = []
    
    # Adaptation logging arrays
    active_window_sizes = []
    cepstrum_forecast_flags = []
    acf_jitter_values = []
    
    current_win = 4096
    
    # Define freqs range for STFT confidence
    N_fft = 2048
    freqs = np.fft.rfftfreq(N_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    
    for n in range(num_frames):
        start = n * hop
        end = start + current_win
        
        c_frame = clean_sig[start:end] * np.hanning(current_win)
        n_frame = noisy_sig[start:end] * np.hanning(current_win)
        
        # Representations
        acf = compute_acf(n_frame)
        cep = compute_cepstrum(n_frame)
        stft_mag = compute_stft(n_frame, sr)
        
        # Raw Estimates
        p_acf, _ = estimate_pitch_acf(acf, sr)
        p_cep, _ = estimate_pitch_cepstrum(cep, sr)
        p_stft, _ = estimate_pitch_stft(stft_mag, sr)
        
        # Calibrated Confidences
        c_0 = cep[0]
        c_cep = np.clip((c_0 - (-10.0)) / (-13.6 - (-10.0)), 0.0, 1.0)
        
        peak_acf = np.max(acf[min_lag:max_lag])
        ratio_acf = peak_acf / acf[0]
        c_acf = np.clip((ratio_acf - 0.1) / (0.8 - 0.1), 0.0, 1.0)
        
        avg_spec = np.mean(stft_mag, axis=1)
        spec_range = avg_spec[min_bin:max_bin]
        ratio_stft = np.max(spec_range) / np.sum(spec_range)
        c_stft = np.clip((ratio_stft - 0.03) / (0.25 - 0.03), 0.0, 1.0)
        
        # Log Cepstrum DC Shift for smoothing/velocity
        if n == 0:
            c_0_smooth = c_0
        else:
            c_0_smooth = 0.3 * c_0 + 0.7 * cep_c_0_smooth[-1]
        cep_c_0_smooth.append(c_0_smooth)
        
        # 4a. Adaptation Mechanism 1: Cepstrum Collapse Forecast
        forecast_flag = 0.0
        if n > 0:
            vel = c_0_smooth - cep_c_0_smooth[-2]
            fc = c_0_smooth + 5 * vel
            if fc >= -10.0 and c_0_smooth < -10.0:
                forecast_flag = 1.0
        cepstrum_forecast_flags.append(forecast_flag)
        
        # 4b. Adaptation Mechanism 2: ACF Jitter Routing
        acf_pitch_history.append(p_acf)
        acf_jitter = 0.0
        if n > 0:
            acf_jitter = abs(p_acf - acf_pitch_history[-2])
        acf_jitter_values.append(acf_jitter)
        
        # 4c. Compute Adaptive Weights and Candidates
        pitches = [p_acf, p_cep]
        weights = [c_acf, c_cep]
        
        # Action 1: Preemptive Cepstrum muting
        if forecast_flag == 1.0 or c_0_smooth >= -10.0:
            weights[1] = 0.0
            
        # Action 2: Preemptive ACF-to-STFT routing under high pitch jitter
        if acf_jitter > 8.0:
            # Route active estimation from ACF to STFT
            pitches[0] = p_stft
            weights[0] = 1.5 * c_stft
            
        # Fuse Pitch
        fused_pitch = hybrid_pitch_estimate(pitches, weights)
        adaptive_errors.append(abs(fused_pitch - true_pitches[n]))
        
        # 4d. Adaptation Mechanism 3: Dynamic Window Scaling
        # Check average confidence score of the current frame to decide next frame's window size
        avg_conf = (c_acf + c_cep + c_stft) / 3.0
        active_window_sizes.append(current_win)
        
        if avg_conf < 0.35:
            current_win = 8192
        else:
            current_win = 4096
            
    adaptive_errors = np.array(adaptive_errors)
    active_window_sizes = np.array(active_window_sizes)
    cepstrum_forecast_flags = np.array(cepstrum_forecast_flags)
    acf_jitter_values = np.array(acf_jitter_values)
    
    # 5. Plotting results in a Stunning 3-Panel Dark-Mode Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    color_wave = "#a6cee3"
    color_reactive = "#e31a1c"
    color_adaptive = "#33a02c"
    color_cep = "#e7298a"
    color_acf = "#377eb8"
    color_win = "#984ea3"
    
    # --- Panel 1: Waveform & Simulation Context ---
    ax1 = axes[0]
    ax1.plot(t, noisy_sig, color=color_wave, alpha=0.4, label="Noisy Signal")
    ax1.plot(t, clean_sig, color="#ff7f00", alpha=0.2, label="Clean Signal")
    ax1.set_title("Audio Waveform & Dynamic Perturbations", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Overlay indicators of segments
    ax1.axvspan(0, 1.5, color="red", alpha=0.08, label="Segment 1: Cepstrum Collapse (Noise)")
    ax1.axvspan(1.5, 3.0, color="blue", alpha=0.08, label="Segment 2: ACF Jitter (Vibrato)")
    ax1.axvspan(3.0, 4.5, color="purple", alpha=0.08, label="Segment 3: Low-Confidence Regime (High Noise)")
    ax1.legend(loc="upper left", fontsize=8)
    
    # --- Panel 2: Pitch Tracking Errors Comparison ---
    ax2 = axes[1]
    ax2.plot(frame_times, reactive_errors, color=color_reactive, linewidth=2, label="Reactive Fusion (Fixed 4096 Win)")
    ax2.plot(frame_times, adaptive_errors, color=color_adaptive, linewidth=2.5, label="Adaptive Fusion (Active Avoidance)")
    ax2.set_title("Pitch Tracking Error Comparison", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Absolute Pitch Error (Hz)")
    ax2.set_ylim(-5, 250)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left")
    
    # --- Panel 3: Dynamic Adaptation Signals ---
    ax3 = axes[2]
    # Cepstrum Forecast Flag & ACF Jitter
    line_jitter, = ax3.plot(frame_times, acf_jitter_values, color=color_acf, linewidth=2, label="ACF Pitch Jitter (Hz)")
    ax3.set_ylabel("ACF Pitch Jitter (Hz)", color=color_acf)
    ax3.tick_params(colors=color_acf)
    ax3.grid(True, linestyle="--", alpha=0.3)
    
    ax3_twin1 = ax3.twinx()
    # Shaded region for Cepstrum forecast alert
    ax3_twin1.fill_between(frame_times, 0, cepstrum_forecast_flags, where=(cepstrum_forecast_flags == 1),
                           color=color_cep, alpha=0.3, label="Cepstrum Collapse Forecast")
    ax3_twin1.plot(frame_times, cepstrum_forecast_flags, color=color_cep, linestyle=":", linewidth=2, label="Cepstrum Forecast Flag")
    ax3_twin1.set_ylabel("Cepstrum Forecast Flag", color=color_cep)
    ax3_twin1.tick_params(colors=color_cep)
    ax3_twin1.set_ylim(-0.1, 1.1)
    
    ax3_twin2 = ax3.twinx()
    # Offset the twin axis for window size so it doesn't overlap
    ax3_twin2.spines['right'].set_position(('outward', 60))
    line_win, = ax3_twin2.plot(frame_times, active_window_sizes, color=color_win, linewidth=2.5, label="Active Window Size")
    ax3_twin2.set_ylabel("Active Window Size (Samples)", color=color_win)
    ax3_twin2.tick_params(colors=color_win)
    ax3_twin2.set_ylim(2000, 10000)
    
    # Group legends
    lines = [line_jitter, line_win]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left")
    ax3.set_title("Active Avoidance & Adaptive Routing Parameters", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Time (seconds)")
    
    # Save Plot
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp017_adaptive_routing.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print(f"Simulation completed! Plot saved to: {output_path}")
    print("-" * 60)
    
    # Compute performance stats
    mean_err_reactive = np.mean(reactive_errors)
    mean_err_adaptive = np.mean(adaptive_errors)
    print(f"Mean Reactive Pitch Error: {mean_err_reactive:.2f} Hz")
    print(f"Mean Adaptive Pitch Error: {mean_err_adaptive:.2f} Hz")
    
    # Segment-specific metrics
    # Segment 1: t in [0.0, 1.5] -> Frame index when frame center is within [0.0, 1.5]
    idx1 = np.where(frame_times < 1.5)[0]
    idx2 = np.where((frame_times >= 1.5) & (frame_times < 3.0))[0]
    idx3 = np.where(frame_times >= 3.0)[0]
    
    print(f"Segment 1 (Cepstrum Collapse) Mean Error -> Reactive: {np.mean(reactive_errors[idx1]):.2f} Hz | Adaptive: {np.mean(adaptive_errors[idx1]):.2f} Hz")
    print(f"Segment 2 (ACF Jitter/Vibrato) Mean Error -> Reactive: {np.mean(reactive_errors[idx2]):.2f} Hz | Adaptive: {np.mean(adaptive_errors[idx2]):.2f} Hz")
    print(f"Segment 3 (Low Conf/Window Scaling) Mean Error -> Reactive: {np.mean(reactive_errors[idx3]):.2f} Hz | Adaptive: {np.mean(adaptive_errors[idx3]):.2f} Hz")

if __name__ == "__main__":
    run_adaptive_routing_experiment()
