import sys
import os
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.stft import compute_stft
from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.experiments.exp017_adaptive_routing import estimate_pitch_acf, estimate_pitch_cepstrum, estimate_pitch_stft, hybrid_pitch_estimate
from src.experiments.exp018_meta_representation import extract_frame_features

# Save helper for WAV audio
def save_wav(path, data, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Clip and convert float32 to int16
    data_int16 = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    scipy.io.wavfile.write(path, sr, data_int16)

# Vowel Formant Filter (simulates human vocal tract resonant peaks)
def apply_formants(f_profile, phases_k, k):
    # Formants: F1 = 600 Hz (vowel 'ah' first formant), F2 = 1800 Hz (vowel 'ah' second formant)
    freq_k = k * f_profile
    amp_k = 1.0 * np.exp(-(freq_k - 600.0)**2 / (2 * 150.0**2)) + 0.6 * np.exp(-(freq_k - 1800.0)**2 / (2 * 250.0**2))
    # Baseline + 1/k standard decay
    return (0.1 + amp_k) / k

def run_real_vocal_tuner():
    print("=" * 60)
    print("RUNNING EXPERIMENT 020: ADAPTIVE TUNER ON REAL VOCAL")
    print("=" * 60)
    
    sr = 22050
    hop = 512
    
    # 1. Load User's Clean Vocal Snippet
    vocal_path = os.path.join(project_root, "Clean_vocal.wav")
    if not os.path.exists(vocal_path):
        print(f"Error: Vocal file not found at {vocal_path}")
        return
        
    print(f"Loading real vocal snippet from: {vocal_path}")
    clean_sig, _ = librosa.load(vocal_path, sr=sr)
    
    # Normalize peak
    clean_sig /= np.max(np.abs(clean_sig))
    duration = len(clean_sig) / sr
    print(f"Loaded vocal duration: {duration:.2f} seconds at {sr} Hz sample rate.")
    
    # 2. Create the Vocal Conditions
    print("Generating vocal perturbation classes for real vocal...")
    conditions = {}
    conditions["clean"] = clean_sig.copy()
    
    # Noisy (std dev = 0.15)
    np.random.seed(42)
    conditions["noisy"] = clean_sig + np.random.normal(0, 0.15, size=clean_sig.shape)
    save_wav(os.path.join(project_root, "results", "audio", "real_source_noisy.wav"), conditions["noisy"], sr)
    
    # Filtered (lowpass at 600 Hz - removes overtones/formants)
    b, a = scipy.signal.butter(4, 600.0 / (sr / 2), btype='low')
    conditions["filtered"] = scipy.signal.filtfilt(b, a, clean_sig)
    conditions["filtered"] /= np.max(np.abs(conditions["filtered"]))
    save_wav(os.path.join(project_root, "results", "audio", "real_source_filtered.wav"), conditions["filtered"], sr)
    
    # Distorted (harsh hard-clipping)
    dist = np.clip(clean_sig, -0.2, 0.2)
    conditions["distorted"] = dist / np.max(np.abs(dist))
    save_wav(os.path.join(project_root, "results", "audio", "real_source_distorted.wav"), conditions["distorted"], sr)
    
    # -----------------------------------------------------------------
    # Step 3: Train the Meta-Representation Layer dynamically
    # -----------------------------------------------------------------
    print("Training the Meta-Representation layer on the fly...")
    X_train = []
    Y_train = []
    
    # We generate a fast training sweep mimicking vowel formants
    # This aligns the trained weights with vocal structures
    training_frequencies = [261.63, 329.63, 392.00, 440.00] # C4, E4, G4, A4
    t_tr = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    for i, f0 in enumerate(training_frequencies):
        clean_tr = np.zeros_like(t_tr)
        for k in range(1, 6):
            phases_k = 2 * np.pi * np.cumsum(k * f0 * np.ones_like(t_tr)) / sr
            amp_k = apply_formants(f0 * np.ones_like(t_tr), phases_k, k)
            clean_tr += amp_k * np.sin(phases_k)
        clean_tr /= np.max(np.abs(clean_tr))
        
        np.random.seed(100 + i)
        noisy_tr = clean_tr + np.random.normal(0, 0.1, size=clean_tr.shape)
        
        num_tr_frames = (len(clean_tr) - 4096) // hop + 1
        
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
            
            x_feat, p_acf, p_cep, p_stft, c_acf, c_cep, c_stft, c_0_smooth, ratio_stft = extract_frame_features(
                acf, cep, stft_mag, sr, prev_p_acf, prev_c_0_smooth, prev_stft_ratio, 
                np.fft.rfftfreq(2048, d=1/sr), 
                np.argmin(np.abs(np.fft.rfftfreq(2048, d=1/sr) - 80)),
                np.argmin(np.abs(np.fft.rfftfreq(2048, d=1/sr) - 1000)),
                int(sr/1000), int(sr/80)
            )
            
            prev_p_acf = p_acf
            prev_c_0_smooth = c_0_smooth
            prev_stft_ratio = ratio_stft
            
            e_acf = abs(p_acf - f0)
            e_cep = abs(p_cep - f0)
            e_stft = abs(p_stft - f0)
            
            w_acf_opt = 1.0 / (e_acf + 1.0)
            w_cep_opt = 1.0 / (e_cep + 1.0)
            w_stft_opt = 1.0 / (e_stft + 1.0)
            
            sum_w = w_acf_opt + w_cep_opt + w_stft_opt
            y_target = [w_acf_opt / sum_w, w_cep_opt / sum_w, w_stft_opt / sum_w]
            
            X_train.append(x_feat)
            Y_train.append(y_target)
            
    W = np.linalg.pinv(np.array(X_train)) @ np.array(Y_train)
    print("Meta-Representation training complete.")
    
    # -----------------------------------------------------------------
    # Step 4: Run the Autotuning & Comparative Routing Simulation
    # -----------------------------------------------------------------
    win_size = 8192
    hop_size = 1024
    
    freqs = np.fft.rfftfreq(2048, d=1/sr)
    min_bin = np.argmin(np.abs(freqs - 80))
    max_bin = np.argmin(np.abs(freqs - 1000))
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    
    tuners = ["acf", "stft", "hybrid", "meta"]
    logged_tracks = {cond: {tuner: [] for tuner in tuners} for cond in ["clean", "noisy"]}
    logged_meta_weights = [] # For noisy condition
    
    # We will extract a reference pitch from the clean signal to compare tuning error.
    clean_meta_reference_track = []
    
    for cond_name, audio_signal in conditions.items():
        print(f"\nProcessing real vocal condition: {cond_name.upper()}...")
        
        for tuner_name in tuners:
            print(f"  Running {tuner_name.upper()} tuner pitch correction...")
            
            # Setup overlap-add arrays
            output_audio = np.zeros(len(audio_signal) + win_size)
            window_sum = np.zeros_like(output_audio)
            
            # Setup running state
            prev_p_acf = 440.0
            prev_c_0_smooth = -13.6
            prev_stft_ratio = 0.25
            
            num_blocks = (len(audio_signal) - win_size) // hop_size + 1
            
            for n in range(num_blocks):
                start = n * hop_size
                end = start + win_size
                block = audio_signal[start:end]
                
                # 1. Pitch detection on the center 4096 samples of the block
                center_frame = block[2048:6144] * np.hanning(4096)
                
                acf = compute_acf(center_frame)
                cep = compute_cepstrum(center_frame)
                stft_mag = compute_stft(center_frame, sr)
                
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
                
                # Select detected pitch based on tuner baseline
                if tuner_name == "acf":
                    detected_pitch = p_acf
                elif tuner_name == "stft":
                    detected_pitch = p_stft
                elif tuner_name == "hybrid":
                    detected_pitch = hybrid_pitch_estimate([p_acf, p_cep], [c_acf, c_cep])
                elif tuner_name == "meta":
                    x_feat, _, _, _, _, _, _, c_0_smooth, ratio_stft_smooth = extract_frame_features(
                        acf, cep, stft_mag, sr, prev_p_acf, prev_c_0_smooth, prev_stft_ratio, freqs, min_bin, max_bin, min_lag, max_lag
                    )
                    prev_p_acf = p_acf
                    prev_c_0_smooth = c_0_smooth
                    prev_stft_ratio = ratio_stft_smooth
                    
                    y_pred = x_feat @ W
                    y_pred_clipped = np.maximum(0.0, y_pred)
                    w_sum = np.sum(y_pred_clipped)
                    w_pred = y_pred_clipped / w_sum if w_sum > 0 else np.array([0.33, 0.33, 0.33])
                    
                    detected_pitch = hybrid_pitch_estimate([p_acf, p_cep, p_stft], w_pred)
                    
                    if cond_name == "noisy":
                        logged_meta_weights.append(w_pred)
                    if cond_name == "clean":
                        clean_meta_reference_track.append(detected_pitch)
                
                # Log tracks for clean/noisy
                if cond_name in ["clean", "noisy"]:
                    logged_tracks[cond_name][tuner_name].append(detected_pitch)
                
                # 2. Pitch correction mapping to nearest target note (chromatic scale)
                s = 0.0
                if 80.0 <= detected_pitch <= 1000.0:
                    midi = int(np.round(12 * np.log2(detected_pitch / 440.0) + 69))
                    f_target = 440.0 * (2.0 ** ((midi - 69) / 12.0))
                    s = 12.0 * np.log2(f_target / detected_pitch)
                    s = np.clip(s, -4.0, 4.0)
                    
                # 3. Apply phase-vocoder pitch shift to the block
                shifted_block = librosa.effects.pitch_shift(block, sr=sr, n_steps=s)
                
                # 4. Overlap-add
                win = np.hanning(win_size)
                output_audio[start:end] += shifted_block * win
                window_sum[start:end] += win
                
            output_audio[mask_norm := window_sum > 1e-5] /= window_sum[mask_norm]
            output_audio = output_audio[:len(audio_signal)]
            if np.max(np.abs(output_audio)) > 0:
                output_audio /= np.max(np.abs(output_audio))
                output_audio *= 0.95
                
            # Save the tuner output audio
            out_path = os.path.join(project_root, "results", "audio", f"real_{cond_name}_{tuner_name}.wav")
            save_wav(out_path, output_audio, sr)
            
    t_frames = np.array([(n * hop_size + 4096) / sr for n in range(len(logged_tracks["clean"]["acf"]))])
    
    # -----------------------------------------------------------------
    # Step 5: Plotting the "Internal Brain" Exposer Panel for Real Vocal
    # -----------------------------------------------------------------
    print("\nGenerating visual report and plot for real vocal...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.35)
    
    # Panel 1: Pitch Correction Tracks under Noise
    ax1 = axes[0]
    # Clean meta track is our best representation of the "true" singing pitch sweep
    ax1.plot(t_frames, clean_meta_reference_track, color="#ffffff", linestyle="--", linewidth=1.5, alpha=0.6, label="Clean Vocal Pitch Reference")
    ax1.plot(t_frames, logged_tracks["noisy"]["stft"], color="#a6d854", alpha=0.8, label="STFT Tuner Track")
    ax1.plot(t_frames, logged_tracks["noisy"]["acf"], color="#377eb8", alpha=0.8, label="ACF Tuner Track")
    ax1.plot(t_frames, logged_tracks["noisy"]["meta"], color="#33a02c", linewidth=2.5, label="Meta-Representation Tuner Track")
    ax1.set_title("Real Vocal Pitch Detector Tracks under Noisy Condition (σ = 0.15)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Detected Pitch (Hz)")
    ax1.set_ylim(250, 650) # Set to match range: 339 Hz to 560 Hz
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)
    
    # Panel 2: The "Internal Brain" - Exposing the Representation Weights
    ax2 = axes[1]
    w_log = np.array(logged_meta_weights)
    ax2.stackplot(t_frames, w_log[:, 0], w_log[:, 1], w_log[:, 2], 
                  labels=["ACF Weight", "Cepstrum Weight", "STFT Weight"], 
                  colors=["#377eb8", "#e7298a", "#a6d854"], alpha=0.75)
    ax2.set_title("The Internal Brain: Dynamic Representation Weights Exposer (Real Noisy Vocal)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Weight Fraction")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)
    
    # Panel 3: Pitch Correction Error Comparison
    ax3 = axes[2]
    # Reference target pitches over time from clean reference
    target_p_frames = np.array([440.0 * (2.0 ** ((int(np.round(12 * np.log2(f / 440.0) + 69)) - 69) / 12.0)) for f in clean_meta_reference_track])
    
    acf_err = np.abs(np.array(logged_tracks["noisy"]["acf"]) - target_p_frames)
    stft_err = np.abs(np.array(logged_tracks["noisy"]["stft"]) - target_p_frames)
    meta_err = np.abs(np.array(logged_tracks["noisy"]["meta"]) - target_p_frames)
    
    ax3.plot(t_frames, acf_err, color="#377eb8", alpha=0.7, label="ACF Tuner Error")
    ax3.plot(t_frames, stft_err, color="#a6d854", alpha=0.7, label="STFT Tuner Error")
    ax3.plot(t_frames, meta_err, color="#33a02c", linewidth=2.5, label="Meta-Representation Tuner Error")
    ax3.set_title("Pitch Tuning Deviation (Error to Target Notes) on Real Vocal under Noise", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Deviation (Hz)")
    ax3.set_xlabel("Time (seconds)")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(loc="upper left", fontsize=9)
    
    # Save Plot
    output_path = os.path.join(project_root, "results", "exp020_real_tuner.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("-" * 60)
    print("REAL VOCAL EXPERIMENT 020 RUN SUCCESSFUL!")
    print("-" * 60)
    print(f"Generated WAV audio outputs saved to: results/audio/")
    print(f"Generated Exposer brain visualization saved to: {output_path}")
    print("-" * 60)
    
    # Print tuner performance summary
    print("Mean Tuning Deviation to Target Notes (Real Noisy Vocal):")
    print(f"  ACF-only Tuner  : {np.mean(acf_err):.2f} Hz")
    print(f"  STFT-only Tuner : {np.mean(stft_err):.2f} Hz")
    print(f"  Adaptive Tuner  : {np.mean(meta_err):.2f} Hz")
    print("-" * 60)

if __name__ == "__main__":
    run_real_vocal_tuner()
