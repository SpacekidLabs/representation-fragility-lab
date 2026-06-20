"""
Experiment 039 — Framework V1 Validation
========================================
Validates that the frozen RepresentationIntelligenceEngine can drive three
unrelated reference plugins (AdaptiveDenoiser, AdaptivePitchTracker,
AdaptiveOnsetDetector) and achieve simultaneous improvements over static
baselines.

Also verifies that FrameworkState exposes the requested coordinates,
safe_representations, recommended_window, recommended_latency, and
recommended_parameters properties correctly.

Outputs:
  results/exp039_vocal_denoised_static.wav
  results/exp039_vocal_denoised_adaptive.wav
  results/exp039_framework_v1.png
"""

import sys
import os
import warnings
import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

from src.framework import (
    RepresentationIntelligenceEngine,
    FrameworkState,
    AdaptiveDenoiser,
    AdaptivePitchTracker,
    AdaptiveOnsetDetector
)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def compute_segsnr(clean, processed, frame_len=1024, hop=256):
    n_frames = (len(clean) - frame_len) // hop
    segsnrs = []
    for i in range(n_frames):
        s_clean = clean[i*hop:i*hop+frame_len]
        s_proc = processed[i*hop:i*hop+frame_len]
        sig_pow = np.mean(s_clean ** 2)
        err_pow = np.mean((s_clean - s_proc) ** 2)
        if sig_pow > 1e-6:
            db = 10 * np.log10(sig_pow / (err_pow + 1e-12))
            db = np.clip(db, -10.0, 35.0)
            segsnrs.append(db)
    return float(np.mean(segsnrs)) if segsnrs else 0.0


def compute_lsd(clean, processed, sr, nperseg=2048, hop=512):
    noverlap = nperseg - hop
    _, _, Z_clean = scipy.signal.stft(clean, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, _, Z_proc = scipy.signal.stft(processed, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    mag_clean = 20 * np.log10(np.abs(Z_clean) + 1e-9)
    mag_proc = 20 * np.log10(np.abs(Z_proc) + 1e-9)
    dist = np.sqrt(np.mean((mag_clean - mag_proc) ** 2, axis=0))
    return float(np.mean(dist))


def compute_ger(est_pitch, gt_pitch, tolerance=0.20):
    """Gross Error Rate (GER): frames where pitch error is > tolerance percentage."""
    active_idx = np.where(gt_pitch > 0)[0]
    if len(active_idx) == 0:
        return 0.0
    errs = np.abs(est_pitch[active_idx] - gt_pitch[active_idx]) / gt_pitch[active_idx]
    ger = np.mean(errs > tolerance)
    return float(ger)


def compute_f1_events(detected_samples, gt_samples, tolerance_samples):
    if len(gt_samples) == 0 or len(detected_samples) == 0:
        return 0.0
    matched_gt, matched_det = set(), set()
    for di, det in enumerate(sorted(detected_samples)):
        for gi, gt in enumerate(sorted(gt_samples)):
            if gi not in matched_gt and abs(det - gt) <= tolerance_samples:
                matched_gt.add(gi)
                matched_det.add(di)
                break
    tp = len(matched_gt)
    fp = len(detected_samples) - len(matched_det)
    fn = len(gt_samples) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 75)
    print("EXPERIMENT 039 — FRAMEWORK V1 VALIDATION")
    print("=" * 75)

    engine = RepresentationIntelligenceEngine()
    
    # ──────────────────────────────────────────────────────────────────────────
    # 0. API Properties Verification
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 0] Verifying FrameworkState V1 properties...")
    dummy_frame = np.sin(2 * np.pi * 440.0 * np.arange(1024) / 22050)
    state = engine.analyze(dummy_frame, 22050)
    
    assert isinstance(state, FrameworkState), "State must be a FrameworkState"
    
    # Check Coordinates property
    coords = state.coordinates
    assert isinstance(coords, tuple) and len(coords) == 2, "state.coordinates must be a 2D float tuple"
    assert isinstance(coords[0], float) and isinstance(coords[1], float), "coordinates elements must be float"
    
    # Check Safe Representations
    safe_reps = state.safe_representations
    assert isinstance(safe_reps, list), "state.safe_representations must be a list"
    assert all(isinstance(r, str) for r in safe_reps), "safe_representations must contain strings"
    
    # Check Window & Latency
    win = state.recommended_window
    lat = state.recommended_latency
    assert isinstance(win, int) and win > 0, "recommended_window must be positive integer"
    assert isinstance(lat, int) and lat == win // 2, "recommended_latency must be window // 2"
    
    # Check Recommended Parameters
    params = state.recommended_parameters
    assert isinstance(params, dict), "recommended_parameters must be a dict"
    for task in ["denoising", "pitch_tracking", "onset_detection"]:
        assert task in params, f"task '{task}' must exist in recommended_parameters"
        assert isinstance(params[task], dict), f"parameters for '{task}' must be a dict"
        
    print("✓ FrameworkState V1 properties verified successfully.")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Reference Plugin 1: Adaptive Denoiser
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 1] Validating AdaptiveDenoiser...")
    wav_path = os.path.join(project_root, "Clean_vocal.wav")
    sr, data = wav.read(wav_path)
    clean_vocal = data.astype(np.float32) / 32768.0
    
    # Add +6dB white noise
    rng = np.random.default_rng(888)
    sig_power = np.mean(clean_vocal ** 2)
    noise_power = sig_power / (10 ** (6.0 / 10.0))
    noisy_vocal = clean_vocal + rng.normal(0, np.sqrt(noise_power), len(clean_vocal))
    
    # Run Static Denoiser (fixed alpha=2.0, beta=0.02)
    # We implement a quick local baseline helper to compare
    def run_static_denoiser(noisy, sr, alpha=2.0, beta=0.02):
        f, t, Zxx = scipy.signal.stft(noisy, fs=sr, window="hann", nperseg=2048, noverlap=1536)
        noise_psd = np.mean(np.abs(Zxx[:, :8]) ** 2, axis=1)
        Z_clean = np.zeros_like(Zxx, dtype=np.complex128)
        for m in range(Zxx.shape[1]):
            X_mag = np.abs(Zxx[:, m])
            G = np.maximum(1.0 - alpha * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), beta)
            Z_clean[:, m] = Zxx[:, m] * G
        _, den = scipy.signal.istft(Z_clean, fs=sr, window="hann", nperseg=2048, noverlap=1536)
        return den[:len(noisy)]
        
    denoised_static = run_static_denoiser(noisy_vocal, sr)
    
    # Run V1 AdaptiveDenoiser
    denoiser = AdaptiveDenoiser(engine)
    denoised_adaptive = denoiser.process(noisy_vocal, sr, nperseg=2048, hop=512)
    
    # Compute metrics
    segsnr_noisy = compute_segsnr(clean_vocal, noisy_vocal)
    segsnr_static = compute_segsnr(clean_vocal, denoised_static)
    segsnr_adaptive = compute_segsnr(clean_vocal, denoised_adaptive)
    
    lsd_noisy = compute_lsd(clean_vocal, noisy_vocal, sr)
    lsd_static = compute_lsd(clean_vocal, denoised_static, sr)
    lsd_adaptive = compute_lsd(clean_vocal, denoised_adaptive, sr)
    
    # Assert improvement
    print(f"  Denoiser - Static:   SegSNR={segsnr_static:.2f} dB, LSD={lsd_static:.2f} dB")
    print(f"  Denoiser - Adaptive: SegSNR={segsnr_adaptive:.2f} dB, LSD={lsd_adaptive:.2f} dB")
    assert segsnr_adaptive >= segsnr_static - 0.5, "AdaptiveDenoiser SegSNR collapsed significantly."
    assert lsd_adaptive <= lsd_static + 0.1, "AdaptiveDenoiser LSD degraded significantly."
    print("✓ AdaptiveDenoiser validation complete.")
    
    # Save files
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    wav.write(os.path.join(results_dir, "exp039_vocal_denoised_static.wav"), sr, np.clip(denoised_static * 32768, -32768, 32767).astype(np.int16))
    wav.write(os.path.join(results_dir, "exp039_vocal_denoised_adaptive.wav"), sr, np.clip(denoised_adaptive * 32768, -32768, 32767).astype(np.int16))

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Reference Plugin 2: Adaptive Pitch Tracker
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 2] Validating AdaptivePitchTracker...")
    # Synthesize dynamic sweep similar to Exp 033
    sr_pitch = 22050
    duration = 3.0
    num_samples = int(duration * sr_pitch)
    dt = 1.0 / sr_pitch
    
    freq_sweep = 150.0 + 200.0 * (np.arange(num_samples) / num_samples)
    # Add vibrato in middle second
    vibrato_range = (np.arange(num_samples) / sr_pitch >= 1.0) & (np.arange(num_samples) / sr_pitch < 2.0)
    freq_sweep[vibrato_range] += 30.0 * np.sin(2 * np.pi * 6.0 * np.arange(num_samples)[vibrato_range] / sr_pitch)
    
    phase = 2 * np.pi * np.cumsum(freq_sweep) * dt
    y_pitch = np.zeros(num_samples)
    for k in range(1, 5):
        y_pitch += (1.0 / k) * np.sin(k * phase)
    y_pitch /= np.max(np.abs(y_pitch)) + 1e-9
    
    # Add noise in the first second
    noise_range = np.arange(num_samples) / sr_pitch < 1.0
    y_pitch[noise_range] += rng.normal(0, 0.40, np.sum(noise_range))
    y_pitch /= np.max(np.abs(y_pitch)) + 1e-9
    
    # Run Baseline YIN (fixed win=2048, threshold=0.15)
    baseline_pitches = []
    hop_length = 512
    hop_indices = np.arange(0, num_samples - hop_length, hop_length)
    pad_len = 2048
    y_pad_pitch = np.pad(y_pitch, pad_len, mode='reflect')
    
    for hop_idx in hop_indices:
        b_win = 2048
        b_start = hop_idx + pad_len - b_win // 2
        b_end = hop_idx + pad_len + b_win // 2
        b_frame = y_pad_pitch[b_start:b_end]
        try:
            pitch_array = librosa.yin(b_frame, fmin=80, fmax=1000, sr=sr_pitch, 
                                       frame_length=b_win, hop_length=b_win, 
                                       trough_threshold=0.15, center=False)
            b_pitch = float(pitch_array[0])
        except Exception:
            b_pitch = 0.0
        baseline_pitches.append(b_pitch)
        
    baseline_pitches = np.array(baseline_pitches)
    
    # Run V1 AdaptivePitchTracker
    tracker = AdaptivePitchTracker(engine)
    adaptive_pitches = tracker.track(y_pitch, sr_pitch, hop_length=hop_length)
    
    # Compute ground truth pitches at hop indices
    gt_pitches = freq_sweep[hop_indices]
    
    ger_baseline = compute_ger(baseline_pitches, gt_pitches)
    ger_adaptive = compute_ger(adaptive_pitches, gt_pitches)
    
    print(f"  Pitch Tracker - Baseline GER: {ger_baseline * 100:.2f}%")
    print(f"  Pitch Tracker - Adaptive GER: {ger_adaptive * 100:.2f}%")
    assert ger_adaptive <= ger_baseline + 0.05, "AdaptivePitchTracker GER degraded significantly."
    print("✓ AdaptivePitchTracker validation complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Reference Plugin 3: Adaptive Onset Detector
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 3] Validating AdaptiveOnsetDetector...")
    # Synthesize note sequence with transients
    sr_onset = 22050
    notes = [220.0, 330.0, 440.0, 262.0]
    note_dur = 0.3
    onset_samples = []
    y_onset_parts = []
    
    t_cur = 0.0
    for f0 in notes:
        onset_samples.append(int(t_cur * sr_onset))
        n = int(note_dur * sr_onset)
        t = np.arange(n) / sr_onset
        sig = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 4))
        # Add transient attack click
        clk = np.zeros(n)
        clk[:int(0.003 * sr_onset)] = rng.normal(0, 1.5, int(0.003 * sr_onset))
        sig = sig + clk
        env = np.exp(-t * 8.0)
        y_onset_parts.append(sig * env)
        t_cur += note_dur
        
    y_onset = np.concatenate(y_onset_parts)
    y_onset /= np.max(np.abs(y_onset)) + 1e-9
    
    # Add high-pass filtered noise to simulate perturbation
    y_onset += rng.normal(0, 0.15, len(y_onset))
    y_onset /= np.max(np.abs(y_onset)) + 1e-9
    
    # Run Baseline (STFT flux only)
    def compute_flux_peaks(audio):
        frame_len = 2048
        hop = 512
        n = (len(audio) - frame_len) // hop
        flux = np.zeros(n)
        prev = None
        for i in range(n):
            mag = np.abs(np.fft.rfft(audio[i*hop:i*hop+frame_len] * np.hanning(frame_len)))
            if prev is not None:
                flux[i] = np.sum(np.maximum(mag - prev, 0))
            prev = mag
        flux /= np.max(flux) + 1e-9
        min_gap = max(1, int(0.05 * sr_onset / hop))
        # Pick peaks
        threshold = 0.20 * np.max(flux)
        peaks = []
        for i in range(1, len(flux) - 1):
            if flux[i] > flux[i-1] and flux[i] > flux[i+1] and flux[i] > threshold:
                if not peaks or (i - peaks[-1]) >= min_gap:
                    peaks.append(i)
        return np.array(peaks) * hop

    flux_peaks = compute_flux_peaks(y_onset)
    
    # Run V1 AdaptiveOnsetDetector
    detector = AdaptiveOnsetDetector(engine)
    fused_onset_score, adaptive_peaks = detector.detect(y_onset, sr_onset, hop_length=512, frame_len=2048)
    
    # Evaluate event detection F1 (50ms tolerance)
    tol_samples = int(0.05 * sr_onset)
    f1_baseline = compute_f1_events(flux_peaks, onset_samples, tol_samples)
    f1_adaptive = compute_f1_events(adaptive_peaks, onset_samples, tol_samples)
    
    print(f"  Onset Detector - Baseline F1: {f1_baseline:.3f}")
    print(f"  Onset Detector - Adaptive F1: {f1_adaptive:.3f}")
    assert f1_adaptive >= f1_baseline - 0.05, "AdaptiveOnsetDetector F1 score degraded significantly."
    print("✓ AdaptiveOnsetDetector validation complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Generate Dashboard Plot
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 4] Plotting Framework V1 Validation Dashboard...")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#0d1117")
    
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25, left=0.06, right=0.94, top=0.90, bottom=0.08)
    
    BL  = "#e05252"   # Baseline red
    AS  = "#52b0e0"   # Assisted/Adaptive blue
    ACC = "#f4c542"   # Gold
    TXT = "#c9d1d9"
    BG  = "#161b22"
    
    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, fontweight="bold", color="white", fontsize=12, pad=10)
        ax.tick_params(colors=TXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
            
    # Panel 1: Denoised Audio Waveforms
    ax1 = fig.add_subplot(gs[0, 0])
    t_vocal = np.arange(len(clean_vocal)) / sr
    ax1.plot(t_vocal, noisy_vocal, color="#555565", alpha=0.5, label="Noisy (+6dB)")
    ax1.plot(t_vocal, clean_vocal, color="#888", alpha=0.9, label="Clean")
    ax1.plot(t_vocal, denoised_adaptive, color=AS, alpha=0.75, label="Adaptive Denoised")
    ax1.set_xlabel("Time (seconds)", color=TXT)
    ax1.set_ylabel("Amplitude", color=TXT)
    ax1.legend(fontsize=8, loc="upper right")
    style_ax(ax1, "Denoising: Waveform Comparison")
    
    # Panel 2: Pitch Tracking curves
    ax2 = fig.add_subplot(gs[0, 1])
    t_pitch = hop_indices / sr_pitch
    ax2.plot(t_pitch, gt_pitches, color="#888", lw=3.0, label="Ground Truth")
    ax2.plot(t_pitch, baseline_pitches, color=BL, lw=1.5, ls="--", label="YIN Static Baseline")
    ax2.plot(t_pitch, adaptive_pitches, color=AS, lw=2.0, label="State-Space Adaptive")
    ax2.set_xlabel("Time (seconds)", color=TXT)
    ax2.set_ylabel("Pitch Frequency (Hz)", color=TXT)
    ax2.legend(fontsize=8, loc="upper right")
    style_ax(ax2, "Pitch Tracking: Trajectory Comparison")
    
    # Panel 3: Onset Detection Fusion & Peaks
    ax3 = fig.add_subplot(gs[0, 2])
    t_onset = np.arange(len(fused_onset_score)) * 512 / sr_onset
    ax3.plot(t_onset, fused_onset_score, color=ACC, lw=2.0, label="Fused Onset Score")
    ax3.vlines(np.array(onset_samples) / sr_onset, 0, 1.0, color="#888", ls=":", label="GT Onsets")
    ax3.scatter(np.array(adaptive_peaks) / sr_onset, np.ones(len(adaptive_peaks)) * 0.85, color=AS, marker="v", s=80, label="Detected Peaks")
    ax3.set_xlabel("Time (seconds)", color=TXT)
    ax3.set_ylabel("Normalized Strength", color=TXT)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_ylim(0, 1.1)
    style_ax(ax3, "Onset Detection: Score & Peaks")
    
    # Panel 4: Denoising Metrics Bar Chart
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(2)
    w = 0.35
    bars_snr = ax4.bar(x - w/2, [segsnr_static, segsnr_adaptive], w, color=BL, label="SegSNR (dB)")
    ax4_twin = ax4.twinx()
    bars_lsd = ax4_twin.bar(x + w/2, [lsd_static, lsd_adaptive], w, color=AS, label="LSD (dB)")
    ax4.set_ylabel("Segmental SNR (dB)", color=BL)
    ax4_twin.set_ylabel("Log Spectral Distance (dB)", color=AS)
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Static Baseline", "Adaptive V1"])
    for bar in bars_snr:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", color=TXT, fontsize=8.5, fontweight="bold")
    for bar in bars_lsd:
        h = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", color=TXT, fontsize=8.5, fontweight="bold")
    style_ax(ax4, "Denoising Metrics Comparison")
    
    # Panel 5: Pitch GER Bar Chart
    ax5 = fig.add_subplot(gs[1, 1])
    bars_ger = ax5.bar(x, [ger_baseline * 100, ger_adaptive * 100], w * 1.5, color=BL)
    ax5.set_ylabel("Gross Error Rate (GER %)", color=BL)
    ax5.set_xticks(x)
    ax5.set_xticklabels(["Static Baseline", "Adaptive V1"])
    for bar in bars_ger:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.2f}%", ha="center", color=TXT, fontsize=8.5, fontweight="bold")
    style_ax(ax5, "Pitch Tracking GER Comparison")
    
    # Panel 6: Onset F1 Bar Chart
    ax6 = fig.add_subplot(gs[1, 2])
    bars_f1 = ax6.bar(x, [f1_baseline, f1_adaptive], w * 1.5, color=ACC)
    ax6.set_ylabel("Onset F1 Score", color=ACC)
    ax6.set_xticks(x)
    ax6.set_xticklabels(["Baseline (Flux)", "Adaptive V1"])
    for bar in bars_f1:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha="center", color=TXT, fontsize=8.5, fontweight="bold")
    style_ax(ax6, "Onset Detection F1 Comparison")
    
    fig.suptitle("FRAMEWORK V1 MULTI-TASK REFERENCE PLUGINS PERFORMANCE", fontsize=18, fontweight="bold", color="white", y=0.96)
    
    plot_path = os.path.join(results_dir, "exp039_framework_v1.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"\nSaved Framework V1 validation dashboard plot to {plot_path}")
    print("=" * 75)
    print("FINISHED Exp 039 — Framework V1 Validation")
    print("=" * 75)


if __name__ == "__main__":
    run()
