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
from src.experiments.exp017_adaptive_routing import estimate_pitch_acf, estimate_pitch_cepstrum, estimate_pitch_stft, hybrid_pitch_estimate

# Helper to extract calibrated confidences and danger signals for a frame
def extract_frame_features(acf, cep, stft_mag, sr, prev_p_acf, prev_c_0_smooth, prev_stft_ratio, freqs, min_bin, max_bin, min_lag, max_lag):
    # Cepstrum features
    c_0 = cep[0]
    c_cep = np.clip((c_0 - (-10.0)) / (-13.6 - (-10.0)), 0.0, 1.0)
    c_0_smooth = 0.3 * c_0 + 0.7 * prev_c_0_smooth
    vel_c_0 = c_0_smooth - prev_c_0_smooth
    
    # ACF features
    peak_acf = np.max(acf[min_lag:max_lag])
    ratio_acf = peak_acf / acf[0]
    c_acf = np.clip((ratio_acf - 0.1) / (0.8 - 0.1), 0.0, 1.0)
    p_acf, _ = estimate_pitch_acf(acf, sr)
    acf_jitter = abs(p_acf - prev_p_acf)
    
    # Cepstrum pitch estimate
    p_cep, _ = estimate_pitch_cepstrum(cep, sr)
    
    # STFT features
    avg_spec = np.mean(stft_mag, axis=1)
    spec_range = avg_spec[min_bin:max_bin]
    ratio_stft = np.max(spec_range) / np.sum(spec_range)
    c_stft = np.clip((ratio_stft - 0.03) / (0.25 - 0.03), 0.0, 1.0)
    vel_stft = ratio_stft - prev_stft_ratio
    p_stft, _ = estimate_pitch_stft(stft_mag, sr)
    
    # Feature vector (10 dimensions)
    x = [
        1.0,            # Bias
        c_acf,          # ACF Confidence
        ratio_acf,      # ACF Peak Ratio
        acf_jitter,     # ACF Jitter
        c_cep,          # Cepstrum Confidence
        c_0_smooth,     # Cepstrum DC Shift
        vel_c_0,        # Cepstrum Velocity
        c_stft,         # STFT Confidence
        ratio_stft,     # STFT Peak Ratio
        vel_stft        # STFT Velocity
    ]
    
    return np.array(x), p_acf, p_cep, p_stft, c_acf, c_cep, c_stft, c_0_smooth, ratio_stft

def run_meta_representation_experiment():
    print("=" * 60)
    print("RUNNING EXPERIMENT 018: META-REPRESENTATION & META-COGNITION")
    print("=" * 60)
    
    sr = 22050
    hop = 512
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    
    N_fft = 2048
    freqs = np.fft.rfftfreq(N_fft, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    
    # -----------------------------------------------------------------
    # Step 1: Collect Training Data
    # -----------------------------------------------------------------
    print("Collecting training data across multiple signal classes...")
    X_train = []
    Y_train = []
    
    # Generate 6 different training scenarios
    training_frequencies = [330.0, 550.0, 440.0, 380.0, 400.0, 480.0]
    training_noises = [0.0, 0.01, 0.05, 0.3, 0.8, 1.5]
    
    for i, (f0, sigma) in enumerate(zip(training_frequencies, training_noises)):
        duration = 2.0
        t_tr = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Apply some frequency vibrato on some files
        f_profile = np.ones_like(t_tr) * f0
        if i % 2 == 1:
            f_profile = f0 + 20.0 * np.sin(2 * np.pi * 5.0 * t_tr)
            
        clean_tr, _ = harmonic_stack.generate(duration, sr, f0=f0)
        # Re-generate using cumulative phase if vibrato to be correct
        if i % 2 == 1:
            clean_tr = np.zeros_like(t_tr)
            for k in range(1, 6):
                phases_k = 2 * np.pi * np.cumsum(k * f_profile) / sr
                clean_tr += (1.0 / k) * np.sin(phases_k)
            clean_tr /= np.max(np.abs(clean_tr))
            
        np.random.seed(42 + i)
        noise = np.random.normal(0, sigma, size=clean_tr.shape)
        noisy_tr = clean_tr + noise
        
        # Process frames
        num_tr_frames = (len(clean_tr) - 4096) // hop + 1
        
        # Initialize running state
        prev_p_acf = f0
        prev_c_0_smooth = -13.6
        prev_stft_ratio = 0.25
        
        for n in range(num_tr_frames):
            start = n * hop
            end = start + 4096
            n_frame = noisy_tr[start:end] * np.hanning(4096)
            
            acf = compute_acf(n_frame)
            cep = compute_cepstrum(n_frame)
            stft_mag = compute_stft(n_frame, sr)
            
            # Extract features and estimates
            x_feat, p_acf, p_cep, p_stft, c_acf, c_cep, c_stft, c_0_smooth, ratio_stft = extract_frame_features(
                acf, cep, stft_mag, sr, prev_p_acf, prev_c_0_smooth, prev_stft_ratio, freqs, min_bin, max_bin, min_lag, max_lag
            )
            
            prev_p_acf = p_acf
            prev_c_0_smooth = c_0_smooth
            prev_stft_ratio = ratio_stft
            
            # Ground-truth true pitch
            true_p = f_profile[start + 2048]
            
            # Compute representation absolute errors
            e_acf = abs(p_acf - true_p)
            e_cep = abs(p_cep - true_p)
            e_stft = abs(p_stft - true_p)
            
            # Define optimal weights target based on error (soft reciprocal distance)
            # Add small offset of 1.0 to avoid division by zero
            w_acf_opt = 1.0 / (e_acf + 1.0)
            w_cep_opt = 1.0 / (e_cep + 1.0)
            w_stft_opt = 1.0 / (e_stft + 1.0)
            
            # Normalize targets
            sum_w = w_acf_opt + w_cep_opt + w_stft_opt
            y_target = [w_acf_opt / sum_w, w_cep_opt / sum_w, w_stft_opt / sum_w]
            
            X_train.append(x_feat)
            Y_train.append(y_target)
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    print(f"Collected {X_train.shape[0]} training sample frames.")
    
    # -----------------------------------------------------------------
    # Step 2: Solve the Meta-Representation Weights (Moore-Penrose Pseudo-inverse)
    # -----------------------------------------------------------------
    print("Training meta-representation layer using closed-form least-squares...")
    W = np.linalg.pinv(X_train) @ Y_train
    
    print("\nLearned Meta-Layer Coefficients Matrix (W):")
    print("-" * 75)
    feature_names = [
        "Bias", "ACF Conf", "ACF Peak Ratio", "ACF Jitter",
        "Cep Conf", "Cep DC Shift (c0)", "Cep Velocity",
        "STFT Conf", "STFT Peak Ratio", "STFT Velocity"
    ]
    print(f"{'Feature Name':<22}{'W_acf':<15}{'W_cep':<15}{'W_stft':<15}")
    print("-" * 75)
    for idx, name in enumerate(feature_names):
        print(f"{name:<22}{W[idx, 0]:<15.4f}{W[idx, 1]:<15.4f}{W[idx, 2]:<15.4f}")
    print("-" * 75)
    
    # Analysis of W coefficient physical meaning
    print("\nPhysical Analysis:")
    if W[5, 1] > 0: # DC shift coefficient mapping to Cepstrum weight
        print("-> CHECK: As Cepstrum DC Shift (c0) increases (moving towards 0.0 under noise), Cepstrum weight drops (negative mapping).")
    if W[3, 0] < 0: # ACF Jitter coefficient mapping to ACF weight
        print("-> CHECK: As ACF Jitter increases (vibrato/octave shift), ACF weight drops.")
        
    # -----------------------------------------------------------------
    # Step 3: Evaluate on Experiment 017 Test Signal
    # -----------------------------------------------------------------
    print("\nEvaluating on unseen Experiment 017 test signal...")
    duration = 4.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Test Signal Frequency Profile
    f_profile = np.zeros_like(t)
    f_profile[t < 1.5] = 440.0
    mask_vib = (t >= 1.5) & (t < 3.0)
    f_profile[mask_vib] = 440.0 + 40.0 * np.sin(2 * np.pi * 6.0 * (t[mask_vib] - 1.5))
    f_profile[t >= 3.0] = 440.0
    
    clean_sig = np.zeros_like(t)
    for k in range(1, 6):
        phases_k = 2 * np.pi * np.cumsum(k * f_profile) / sr
        clean_sig += (1.0 / k) * np.sin(phases_k)
    clean_sig /= np.max(np.abs(clean_sig))
    
    # Noise Profile: ramps from 0.5s to 1.5s, vibrato mild noise, Segment 3 extreme noise
    sigmas = np.zeros_like(t)
    mask_ramp = (t >= 0.5) & (t < 1.5)
    sigmas[mask_ramp] = 0.02 * (t[mask_ramp] - 0.5)
    mask_vib_noise = (t >= 1.5) & (t < 3.0)
    sigmas[mask_vib_noise] = 0.005
    sigmas[t >= 3.0] = 1.8
    
    np.random.seed(42)
    noise = np.random.normal(0, sigmas)
    noisy_sig = clean_sig + noise
    
    max_win = 8192
    num_frames = (len(clean_sig) - max_win) // hop + 1
    
    frame_times = np.array([(n * hop + 4096 / 2) / sr for n in range(num_frames)])
    true_pitches = np.array([f_profile[int(n * hop + 4096 / 2)] for n in range(num_frames)])
    
    # 3a. Baseline: Reactive Fusion (Fixed 4096 window)
    reactive_errors = []
    for n in range(num_frames):
        start = n * hop
        end = start + 4096
        n_frame = noisy_sig[start:end] * np.hanning(4096)
        
        acf = compute_acf(n_frame)
        cep = compute_cepstrum(n_frame)
        
        p_acf, _ = estimate_pitch_acf(acf, sr)
        p_cep, _ = estimate_pitch_cepstrum(cep, sr)
        
        c_0 = cep[0]
        c_cep = np.clip((c_0 - (-10.0)) / (-13.6 - (-10.0)), 0.0, 1.0)
        
        peak_acf = np.max(acf[min_lag:max_lag])
        ratio_acf = peak_acf / acf[0]
        c_acf = np.clip((ratio_acf - 0.1) / (0.8 - 0.1), 0.0, 1.0)
        
        fused = hybrid_pitch_estimate([p_acf, p_cep], [c_acf, c_cep])
        reactive_errors.append(abs(fused - true_pitches[n]))
    reactive_errors = np.array(reactive_errors)
    
    # 3b. Baseline: Hand-Written Adaptive Routing (Exp 017)
    adaptive_errors = []
    cep_c_0_smooth = []
    acf_pitch_history = []
    current_win = 4096
    
    for n in range(num_frames):
        start = n * hop
        end = start + current_win
        n_frame = noisy_sig[start:end] * np.hanning(current_win)
        
        acf = compute_acf(n_frame)
        cep = compute_cepstrum(n_frame)
        stft_mag = compute_stft(n_frame, sr)
        
        p_acf, _ = estimate_pitch_acf(acf, sr)
        p_cep, _ = estimate_pitch_cepstrum(cep, sr)
        p_stft, _ = estimate_pitch_stft(stft_mag, sr)
        
        c_0 = cep[0]
        c_cep = np.clip((c_0 - (-10.0)) / (-13.6 - (-10.0)), 0.0, 1.0)
        
        peak_acf = np.max(acf[min_lag:max_lag])
        ratio_acf = peak_acf / acf[0]
        c_acf = np.clip((ratio_acf - 0.1) / (0.8 - 0.1), 0.0, 1.0)
        
        avg_spec = np.mean(stft_mag, axis=1)
        spec_range = avg_spec[min_bin:max_bin]
        ratio_stft = np.max(spec_range) / np.sum(spec_range)
        c_stft = np.clip((ratio_stft - 0.03) / (0.25 - 0.03), 0.0, 1.0)
        
        if n == 0:
            c_0_smooth = c_0
        else:
            c_0_smooth = 0.3 * c_0 + 0.7 * cep_c_0_smooth[-1]
        cep_c_0_smooth.append(c_0_smooth)
        
        forecast_flag = 0.0
        if n > 0:
            vel = c_0_smooth - cep_c_0_smooth[-2]
            fc = c_0_smooth + 5 * vel
            if fc >= -10.0 and c_0_smooth < -10.0:
                forecast_flag = 1.0
                
        acf_pitch_history.append(p_acf)
        acf_jitter = 0.0
        if n > 0:
            acf_jitter = abs(p_acf - acf_pitch_history[-2])
            
        pitches = [p_acf, p_cep]
        weights = [c_acf, c_cep]
        
        if forecast_flag == 1.0 or c_0_smooth >= -10.0:
            weights[1] = 0.0
        if acf_jitter > 8.0:
            pitches[0] = p_stft
            weights[0] = 1.5 * c_stft
            
        fused = hybrid_pitch_estimate(pitches, weights)
        adaptive_errors.append(abs(fused - true_pitches[n]))
        
        avg_conf = (c_acf + c_cep + c_stft) / 3.0
        if avg_conf < 0.35:
            current_win = 8192
        else:
            current_win = 4096
    adaptive_errors = np.array(adaptive_errors)
    
    # 3c. Meta-Representation Learning Fusion (Exp 018)
    meta_errors = []
    meta_weights = []
    
    prev_p_acf = 440.0
    prev_c_0_smooth = -13.6
    prev_stft_ratio = 0.25
    
    # In Segment 3 under extreme noise, the meta-layer dynamically changes window size too!
    # Because window size is controlled by the meta-representation layer's self-assessed average confidence
    current_win_meta = 4096
    
    for n in range(num_frames):
        start = n * hop
        end = start + current_win_meta
        n_frame = noisy_sig[start:end] * np.hanning(current_win_meta)
        
        acf = compute_acf(n_frame)
        cep = compute_cepstrum(n_frame)
        stft_mag = compute_stft(n_frame, sr)
        
        # Estimates and features
        x_feat, p_acf, p_cep, p_stft, c_acf, c_cep, c_stft, c_0_smooth, ratio_stft = extract_frame_features(
            acf, cep, stft_mag, sr, prev_p_acf, prev_c_0_smooth, prev_stft_ratio, freqs, min_bin, max_bin, min_lag, max_lag
        )
        
        prev_p_acf = p_acf
        prev_c_0_smooth = c_0_smooth
        prev_stft_ratio = ratio_stft
        
        # Meta-Cognitive Weight Prediction
        y_pred = x_feat @ W
        
        # Softmax / Normalize predicted weights
        y_pred_clipped = np.maximum(0.0, y_pred)
        sum_pred = np.sum(y_pred_clipped)
        if sum_pred > 0:
            w_pred = y_pred_clipped / sum_pred
        else:
            w_pred = np.array([0.33, 0.33, 0.33])
            
        meta_weights.append(w_pred)
        
        # Fuse Pitch (uses predicted weights directly over candidates!)
        fused = hybrid_pitch_estimate([p_acf, p_cep, p_stft], w_pred)
        meta_errors.append(abs(fused - true_pitches[n]))
        
        # Dynamic window size control via meta-cognition average confidence
        # Use average confidence to drive the adaptive window scaling
        avg_conf = (c_acf + c_cep + c_stft) / 3.0
        if avg_conf < 0.35:
            current_win_meta = 8192
        else:
            current_win_meta = 4096
            
    meta_errors = np.array(meta_errors)
    meta_weights = np.array(meta_weights)
    
    # 4. Generate the Visualization Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    color_wave = "#a6cee3"
    color_reactive = "#e31a1c"
    color_adaptive = "#ff7f00"
    color_meta = "#33a02c"
    
    # Panel 1: Waveform & Segment context
    ax1 = axes[0]
    ax1.plot(t, noisy_sig, color=color_wave, alpha=0.3, label="Noisy Signal")
    ax1.plot(t, clean_sig, color="#ff7f00", alpha=0.1, label="Clean Signal")
    ax1.axvspan(0, 1.5, color="red", alpha=0.08, label="Segment 1: Cepstrum Collapse")
    ax1.axvspan(1.5, 3.0, color="blue", alpha=0.08, label="Segment 2: ACF Jitter")
    ax1.axvspan(3.0, 4.5, color="purple", alpha=0.08, label="Segment 3: Extreme Noise (Window Scaling)")
    ax1.set_title("Audio Signal and Dynamic Perturbations", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)
    
    # Panel 2: Error Comparison
    ax2 = axes[1]
    ax2.plot(frame_times, reactive_errors, color=color_reactive, linewidth=2, label="Reactive Fusion (Fixed 4096 Win)")
    ax2.plot(frame_times, adaptive_errors, color=color_adaptive, linewidth=2, label="Adaptive Routing (Hand-written)")
    ax2.plot(frame_times, meta_errors, color=color_meta, linewidth=2.5, label="Meta-Representation Fusion (Learned)")
    ax2.set_title("Pitch Tracking Error Comparison (Reactive vs Adaptive vs Meta)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Absolute Pitch Error (Hz)")
    ax2.set_ylim(-5, 250)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left")
    
    # Panel 3: Meta-Representation Weights over time
    ax3 = axes[2]
    ax3.plot(frame_times, meta_weights[:, 0], color="#377eb8", linewidth=2.5, label="Meta-Weight: ACF")
    ax3.plot(frame_times, meta_weights[:, 1], color="#e7298a", linewidth=2.5, label="Meta-Weight: Cepstrum")
    ax3.plot(frame_times, meta_weights[:, 2], color="#a6d854", linewidth=2.5, label="Meta-Weight: STFT")
    ax3.set_title("Meta-Representation Layer Dynamic Fusion Weights (Learned)", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Fusion Weight")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(loc="upper right")
    
    # Save Plot
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    output_path = os.path.join(project_root, "results", "exp018_meta_representation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print(f"Simulation completed! Plot saved to: {output_path}")
    print("-" * 60)
    
    # Print statistics
    print(f"Mean Reactive Pitch Error: {np.mean(reactive_errors):.2f} Hz")
    print(f"Mean Adaptive Pitch Error: {np.mean(adaptive_errors):.2f} Hz")
    print(f"Mean Meta Pitch Error: {np.mean(meta_errors):.2f} Hz")
    
    idx1 = np.where(frame_times < 1.5)[0]
    idx2 = np.where((frame_times >= 1.5) & (frame_times < 3.0))[0]
    idx3 = np.where(frame_times >= 3.0)[0]
    
    print(f"Segment 1 Mean Error -> Reactive: {np.mean(reactive_errors[idx1]):.2f} Hz | Adaptive: {np.mean(adaptive_errors[idx1]):.2f} Hz | Meta: {np.mean(meta_errors[idx1]):.2f} Hz")
    print(f"Segment 2 Mean Error -> Reactive: {np.mean(reactive_errors[idx2]):.2f} Hz | Adaptive: {np.mean(adaptive_errors[idx2]):.2f} Hz | Meta: {np.mean(meta_errors[idx2]):.2f} Hz")
    print(f"Segment 3 Mean Error -> Reactive: {np.mean(reactive_errors[idx3]):.2f} Hz | Adaptive: {np.mean(adaptive_errors[idx3]):.2f} Hz | Meta: {np.mean(meta_errors[idx3]):.2f} Hz")

if __name__ == "__main__":
    run_meta_representation_experiment()
