"""
Experiment 033 — Framework-Assisted DSP Improvement
===================================================
Compares a standard YIN pitch tracker (Baseline) with static parameters against
a Framework-Assisted YIN pitch tracker that dynamically adapts its window size,
trough threshold, and gating based on real-time coordinates in the Universal Audio
State Space.
"""

import sys
import os
import warnings
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings('ignore', category=UserWarning)

from src.framework.engine import RepresentationIntelligenceEngine

def run():
    print("=" * 75)
    print("EXPERIMENT 033 — FRAMEWORK-ASSISTED DSP IMPROVEMENT")
    print("=" * 75)

    sr = 22050
    duration = 5.0
    num_samples = int(duration * sr)
    dt = 1.0 / sr
    rng = np.random.default_rng(42)

    # 1. Synthesize ground-truth frequency sweep with harmonic stack and vibrato
    print("\n1. Generating dynamic 5.0s test sweep...")
    freqs = np.zeros(num_samples)
    for i in range(num_samples):
        t = i / sr
        # Base sweep: 150 Hz to 350 Hz
        f = 150.0 + 200.0 * (t / duration)
        # Add 8 Hz vibrato in Segment 3 (t from 2.0 to 3.0)
        if 2.0 <= t < 3.0:
            f += 35.0 * np.sin(2 * np.pi * 8.0 * t)
        freqs[i] = f

    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(freqs) * dt
    # Harmonic stack: F0 + 3 harmonics
    y_clean = np.zeros(num_samples)
    for k in range(1, 5):
        y_clean += (1.0 / k) * np.sin(k * phase)
    # Normalize clean signal
    y_clean /= np.max(np.abs(y_clean)) + 1e-9

    # Apply perturbations segment-by-segment
    y_perturbed = y_clean.copy()

    # Segment 2 (1.0 to 2.0s): High Noise
    s2_start, s2_end = int(1.0 * sr), int(2.0 * sr)
    y_perturbed[s2_start:s2_end] += rng.normal(0, 0.40, s2_end - s2_start)

    # Segment 4 (3.0 to 4.0s): Click Transients + Hard Clipping
    s4_start, s4_end = int(3.0 * sr), int(4.0 * sr)
    # Add transient clicks (impulses every 150 ms)
    click_spacing = int(0.150 * sr)
    for c_idx in range(s4_start, s4_end, click_spacing):
        y_perturbed[c_idx:c_idx+10] += 0.85 * rng.choice([-1.0, 1.0])
    # Hard clipping
    y_perturbed[s4_start:s4_end] = np.clip(y_perturbed[s4_start:s4_end], -0.06, 0.06)

    # Final normalization of perturbed signal
    y_perturbed /= np.max(np.abs(y_perturbed)) + 1e-9

    # 2. Setup frame processing
    hop_length = 512
    # Pad signal to support up to 4096-sample windows centered at hop indices
    pad_len = 2048
    y_pad_perturbed = np.pad(y_perturbed, pad_len, mode='reflect')

    hop_indices = np.arange(0, num_samples - hop_length, hop_length)
    N_frames = len(hop_indices)

    # Initialize lists to store results
    gt_pitches = []
    baseline_pitches = []
    assisted_pitches = []
    regions = []
    recommendations_list = []
    
    # Initialize framework engine
    engine = RepresentationIntelligenceEngine()

    print(f"\n2. Running Baseline vs. Framework-Assisted YIN across {N_frames} frames...")
    
    last_valid_assisted_pitch = 220.0  # Safe initial anchor

    for idx, hop_idx in enumerate(hop_indices):
        # Center time and ground-truth pitch
        center_idx = hop_idx
        t_c = center_idx / sr
        gt_f = freqs[center_idx]
        gt_pitches.append(gt_f)

        # -----------------------------------------------------------------------
        # A. Baseline YIN (fixed: win=2048, threshold=0.15)
        # -----------------------------------------------------------------------
        b_win = 2048
        b_start = center_idx + pad_len - b_win // 2
        b_end = center_idx + pad_len + b_win // 2
        b_frame = y_pad_perturbed[b_start:b_end]
        
        try:
            b_pitch_array = librosa.yin(b_frame, fmin=80, fmax=1000, sr=sr, 
                                        frame_length=b_win, hop_length=b_win, 
                                        trough_threshold=0.15, center=False)
            b_pitch = float(b_pitch_array[0])
        except Exception:
            b_pitch = 0.0
        baseline_pitches.append(b_pitch)

        # -----------------------------------------------------------------------
        # B. Framework-Assisted YIN (adaptive parameters)
        # -----------------------------------------------------------------------
        # Extract default 1024-sample block for framework analysis
        f_win = 1024
        f_start = center_idx + pad_len - f_win // 2
        f_end = center_idx + pad_len + f_win // 2
        f_frame = y_pad_perturbed[f_start:f_end]
        
        # Query framework engine
        state = engine.analyze(f_frame, sr)
        regions.append(state.region)
        
        # Fetch recommendations
        win_size = state.recommendations["window_size"]
        trough_thresh = 0.15
        
        if state.region == "noise_collapse":
            trough_thresh = 0.25  # Widen threshold under noise
        elif state.region == "transient_overloaded":
            trough_thresh = 0.08  # Narrow threshold under clicks

        # Extract adaptive analysis window centered at hop index
        a_start = center_idx + pad_len - win_size // 2
        a_end = center_idx + pad_len + win_size // 2
        a_frame = y_pad_perturbed[a_start:a_end]
        
        # Run YIN with adaptive parameters
        try:
            a_pitch_array = librosa.yin(a_frame, fmin=80, fmax=1000, sr=sr, 
                                        frame_length=win_size, hop_length=win_size, 
                                        trough_threshold=trough_thresh, center=False)
            a_pitch = float(a_pitch_array[0])
        except Exception:
            a_pitch = 0.0
            
        # Gate/Hold logic: If transient/clipped and confidence is low, hold the previous pitch
        if state.region == "transient_overloaded" and state.assumptions["acf"] < 0.20:
            a_pitch = last_valid_assisted_pitch
        elif a_pitch > 80.0 and a_pitch < 1000.0:
            # Update last valid pitch
            last_valid_assisted_pitch = a_pitch
            
        assisted_pitches.append(a_pitch)

    gt_pitches = np.array(gt_pitches)
    baseline_pitches = np.array(baseline_pitches)
    assisted_pitches = np.array(assisted_pitches)

    # 3. Calculate Metrics segment-by-segment (1.0s segments)
    print("\n3. Evaluating tracking metrics...")
    
    seg_names = [
        "Segment 1 (Clean Harmonic)",
        "Segment 2 (Noise Collapse)",
        "Segment 3 (Vibrato Sweep)",
        "Segment 4 (Transient Distortion)",
        "Segment 5 (Clean Harmonic)",
        "Overall (All 5.0s)"
    ]
    
    frames_per_seg = N_frames // 5
    
    print(f"\n{'Segment Name':<32} | {'YIN Baseline GER':<18} | {'YIN Assisted GER':<18} | {'Improvement':<12}")
    print("-" * 90)
    
    for s_idx in range(6):
        if s_idx < 5:
            start_f = s_idx * frames_per_seg
            end_f = (s_idx + 1) * frames_per_seg
        else:
            start_f = 0
            end_f = N_frames
            
        # Slice segments
        gt_s = gt_pitches[start_f:end_f]
        b_s = baseline_pitches[start_f:end_f]
        a_s = assisted_pitches[start_f:end_f]
        
        # GER calculation: percentage of frames with pitch error > 20%
        # Gross Error is defined as |est - gt| / gt > 0.20
        b_errors = np.abs(b_s - gt_s) / gt_s > 0.20
        a_errors = np.abs(a_s - gt_s) / gt_s > 0.20
        
        b_ger = np.mean(b_errors) * 100.0
        a_ger = np.mean(a_errors) * 100.0
        
        imp = b_ger - a_ger
        print(f"{seg_names[s_idx]:<32} | {b_ger:<16.2f}% | {a_ger:<16.2f}% | {imp:<+10.2f}%")

    # 4. Plotting Results
    print("\n4. Generating YIN performance comparison plots...")
    fig, (ax_wave, ax_pitch) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})
    
    t_sig = np.arange(num_samples) / sr
    ax_wave.plot(t_sig, y_perturbed, color="white", alpha=0.45, lw=0.5, label="Perturbed Signal")
    ax_wave.set_title("Perturbed Time-Domain Input Waveform", fontsize=11, fontweight="bold")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.08)
    
    # Mark segment boundaries in waveform plot
    for boundary in [1.0, 2.0, 3.0, 4.0]:
        ax_wave.axvline(boundary, color="red", linestyle="--", alpha=0.5)
        ax_pitch.axvline(boundary, color="red", linestyle="--", alpha=0.5)
        
    ax_wave.text(0.5, 0.6, "Clean", color="lightgreen", weight="bold", ha="center")
    ax_wave.text(1.5, 0.6, "+ Noise", color="tomato", weight="bold", ha="center")
    ax_wave.text(2.5, 0.6, "Vibrato", color="orange", weight="bold", ha="center")
    ax_wave.text(3.5, 0.6, "+ Clicks/Clip", color="orchid", weight="bold", ha="center")
    ax_wave.text(4.5, 0.6, "Clean", color="lightgreen", weight="bold", ha="center")

    # Plot Pitch tracks
    t_frames = hop_indices / sr
    ax_pitch.plot(t_frames, gt_pitches, color="green", lw=3.0, label="Ground Truth Pitch (F0)")
    ax_pitch.scatter(t_frames, baseline_pitches, color="#d62728", s=15, alpha=0.7, label="Baseline YIN (Fixed)")
    ax_pitch.scatter(t_frames, assisted_pitches, color="#1f77b4", s=15, alpha=0.7, label="Framework-Assisted YIN (Adaptive)")
    
    ax_pitch.set_title("YIN Pitch Tracking Performance: Baseline vs. Framework-Assisted", fontsize=12, fontweight="bold")
    ax_pitch.set_xlabel("Time (seconds)")
    ax_pitch.set_ylabel("Frequency (Hz)")
    ax_pitch.set_ylim(50, 500)
    ax_pitch.legend(loc="upper left")
    ax_pitch.grid(True, alpha=0.08)
    
    fig.suptitle("Experiment 033 — Framework-Assisted DSP:\nImproving YIN Pitch Tracking Robustness via Coordinate-Gated Parameter Adaptation", fontsize=14, fontweight="bold", y=0.96)
    
    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp033_framework_assisted_dsp.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved YIN comparison plot: {out_path}")
    print("=" * 75)

if __name__ == "__main__":
    run()
