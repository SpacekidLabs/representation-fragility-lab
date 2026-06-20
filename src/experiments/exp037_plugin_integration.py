"""
Experiment 037 — Real Plugin Integration (Adaptive Denoising)
=============================================================
Integrates the RepresentationIntelligenceEngine into a real-time spectral 
subtraction denoiser, evaluating on real vocal audio (Clean_vocal.wav).

Outputs:
  results/exp037_vocal_noisy.wav
  results/exp037_vocal_static.wav
  results/exp037_vocal_adaptive.wav
  results/exp037_plugin_integration.png
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

from src.framework.engine import RepresentationIntelligenceEngine


# ──────────────────────────────────────────────────────────────────────────────
# CORE DENOISERS
# ──────────────────────────────────────────────────────────────────────────────

def spectral_subtract_static(noisy_sig, sr, nperseg=2048, hop=512, alpha=2.0, beta=0.02):
    """
    Standard spectral subtraction with static over-subtraction (alpha) and floor (beta).
    """
    noverlap = nperseg - hop
    f, t, Zxx = scipy.signal.stft(noisy_sig, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    # Estimate noise PSD from first 8 frames (~0.1s of noise-only prefix)
    noise_psd = np.mean(np.abs(Zxx[:, :8]) ** 2, axis=1)
    
    Zxx_clean = np.zeros_like(Zxx, dtype=np.complex128)
    
    for m in range(Zxx.shape[1]):
        X_mag = np.abs(Zxx[:, m])
        # Gain filter: max( 1 - alpha * sqrt(noise_psd / X^2), beta )
        G = np.maximum(1.0 - alpha * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), beta)
        Zxx_clean[:, m] = Zxx[:, m] * G
        
    _, denoise_sig = scipy.signal.istft(Zxx_clean, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    return denoise_sig[:len(noisy_sig)]


def spectral_subtract_adaptive(noisy_sig, sr, engine, nperseg=2048, hop=512, beta_default=0.02):
    """
    State-Space adaptive spectral subtraction.
    Downsamples noisy frames to 22050Hz for engine state analysis,
    then adapts alpha and beta per-frame.
    """
    noverlap = nperseg - hop
    f, t, Zxx = scipy.signal.stft(noisy_sig, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    # Estimate noise PSD from first 8 frames (~0.1s of noise-only prefix)
    noise_psd = np.mean(np.abs(Zxx[:, :8]) ** 2, axis=1)
    
    Zxx_clean = np.zeros_like(Zxx, dtype=np.complex128)
    
    alpha_history = []
    beta_history = []
    
    for m in range(Zxx.shape[1]):
        # Extract time-domain frame to feed to the engine
        start_idx = m * hop
        end_idx = start_idx + nperseg
        frame = noisy_sig[start_idx:end_idx]
        
        # Pad if short frame at boundary
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
            
        # Downsample frame to 22050Hz to maintain engine feature mapping consistency
        frame_22k = scipy.signal.resample(frame, nperseg // 2)
        
        # Analyze physical state
        st = engine.analyze(frame_22k, 22050)
        stft_safety = st.assumptions["stft"]
        
        # 1. Adapt over-subtraction factor alpha based on STFT safety
        # High safety -> clean -> low alpha (0.5). Low safety -> noisy -> high alpha (4.0).
        alpha = 0.5 + 3.5 * (1.0 - stft_safety)
        
        # 2. Adapt spectral floor beta based on semantic region
        # noise_collapse -> raise floor to mask musical noise
        # periodic_harmonic -> lower floor to maximize silence gating
        if st.region == "noise_collapse":
            beta = 0.06
        elif st.region == "periodic_harmonic":
            beta = 0.005
        else:
            beta = beta_default
            
        alpha_history.append(alpha)
        beta_history.append(beta)
        
        # Apply filter
        X_mag = np.abs(Zxx[:, m])
        G = np.maximum(1.0 - alpha * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), beta)
        Zxx_clean[:, m] = Zxx[:, m] * G
        
    _, denoise_sig = scipy.signal.istft(Zxx_clean, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    return denoise_sig[:len(noisy_sig)], np.array(alpha_history), np.array(beta_history)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_segsnr(clean, processed, sr, frame_len=1024, hop=256):
    """
    Computes Segmental SNR in dB.
    """
    n_frames = (len(clean) - frame_len) // hop
    segsnrs = []
    for i in range(n_frames):
        s_clean = clean[i*hop:i*hop+frame_len]
        s_proc = processed[i*hop:i*hop+frame_len]
        
        sig_pow = np.mean(s_clean ** 2)
        err_pow = np.mean((s_clean - s_proc) ** 2)
        
        if sig_pow > 1e-6: # evaluate only active frames (skip absolute silence)
            db = 10 * np.log10(sig_pow / (err_pow + 1e-12))
            # clip to realistic range
            db = np.clip(db, -10.0, 35.0)
            segsnrs.append(db)
            
    return float(np.mean(segsnrs)) if segsnrs else 0.0


def compute_lsd(clean, processed, sr, nperseg=2048, hop=512):
    """
    Computes Log Spectral Distance (LSD) in dB.
    """
    noverlap = nperseg - hop
    _, _, Z_clean = scipy.signal.stft(clean, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, _, Z_proc = scipy.signal.stft(processed, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    mag_clean = 20 * np.log10(np.abs(Z_clean) + 1e-9)
    mag_proc = 20 * np.log10(np.abs(Z_proc) + 1e-9)
    
    dist = np.sqrt(np.mean((mag_clean - mag_proc) ** 2, axis=0))
    return float(np.mean(dist))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("EXPERIMENT 037 — REAL PLUGIN INTEGRATION: STATE-SPACE DENOISER")
    print("=" * 72)
    
    # 1. Load Clean Vocal
    wav_path = os.path.join(project_root, "Clean_vocal.wav")
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.")
        return
        
    sr, data = wav.read(wav_path)
    clean = data.astype(np.float32) / 32768.0
    
    # 2. Corrupt with additive white noise (+6 dB SNR)
    rng = np.random.default_rng(888)
    sig_power = np.mean(clean ** 2)
    target_snr_db = 6.0
    noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    
    noise = rng.normal(0, noise_std, len(clean))
    noisy = clean + noise
    
    # Save noisy signal
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    def save_wav(name, sig):
        path = os.path.join(results_dir, name)
        out_int = np.clip(sig * 32768.0, -32768.0, 32767.0).astype(np.int16)
        wav.write(path, sr, out_int)
        print(f"  Saved: {path}")
        
    save_wav("exp037_vocal_noisy.wav", noisy)
    
    # 3. Process with Static Denoiser
    print("Processing with Static Denoiser...")
    denoised_static = spectral_subtract_static(noisy, sr, alpha=2.0, beta=0.02)
    save_wav("exp037_vocal_static.wav", denoised_static)
    
    # 4. Process with Adaptive Denoiser
    print("Processing with State-Space Adaptive Denoiser...")
    engine = RepresentationIntelligenceEngine()
    denoised_adaptive, alphas, betas = spectral_subtract_adaptive(noisy, sr, engine)
    save_wav("exp037_vocal_adaptive.wav", denoised_adaptive)
    
    # 5. Compute Metrics
    snr_noisy = compute_segsnr(clean, noisy, sr)
    snr_static = compute_segsnr(clean, denoised_static, sr)
    snr_adaptive = compute_segsnr(clean, denoised_adaptive, sr)
    
    lsd_noisy = compute_lsd(clean, noisy, sr)
    lsd_static = compute_lsd(clean, denoised_static, sr)
    lsd_adaptive = compute_lsd(clean, denoised_adaptive, sr)
    
    print("\n" + "-"*40)
    print("METRICS PERFORMANCE:")
    print(f"  Noisy Input        : SegSNR = {snr_noisy:5.2f} dB, LSD = {lsd_noisy:5.2f} dB")
    print(f"  Static Baseline    : SegSNR = {snr_static:5.2f} dB, LSD = {lsd_static:5.2f} dB")
    print(f"  State-Space Adapt. : SegSNR = {snr_adaptive:5.2f} dB, LSD = {lsd_adaptive:5.2f} dB")
    print(f"  Delta LSD          : {lsd_adaptive - lsd_static:+.2f} dB (lower is better)")
    print(f"  Delta SegSNR       : {snr_adaptive - snr_static:+.2f} dB (higher is better)")
    print("-"*40 + "\n")
    
    # ──────────────────────────────────────────────────────────────────────────
    # DASHBOARD PLOT: Dark dashboard
    # ──────────────────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    
    gs = fig.add_gridspec(
        3, 2, hspace=0.45, wspace=0.30,
        left=0.07, right=0.95, top=0.90, bottom=0.08
    )
    
    BL  = "#e05252"   # baseline red
    AS  = "#52b0e0"   # assisted blue
    ACC = "#f4c542"   # accent gold
    TXT = "#c9d1d9"
    BG  = "#161b22"
    
    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, fontweight="bold", color="white", fontsize=11, pad=8)
        ax.tick_params(colors=TXT, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
            
    t_axis = np.arange(len(clean)) / sr
    
    # ── Panel 1: Waveforms (Clean vs Noisy) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_axis, noisy, color="#555", alpha=0.5, label="Noisy (+6dB)")
    ax1.plot(t_axis, clean, color="#888", alpha=0.9, label="Clean Vocal")
    ax1.set_ylabel("Amplitude", color=TXT, fontsize=9.5)
    ax1.legend(fontsize=7.5, loc="upper right")
    style_ax(ax1, "Original Clean vs. Noisy Input Waveforms")
    
    # ── Panel 2: Waveforms (Static vs Adaptive) ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_axis, denoised_static, color=BL, alpha=0.85, label="Static Baseline")
    ax2.plot(t_axis, denoised_adaptive, color=AS, alpha=0.85, label="State-Space Adaptive")
    ax2.set_ylabel("Amplitude", color=TXT, fontsize=9.5)
    ax2.legend(fontsize=7.5, loc="upper right")
    style_ax(ax2, "Denoised Output Waveforms (Static vs. Adaptive)")
    
    # ── Panel 3: Spectrogram (Static Denoised) ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    nperseg_spec = 1024
    f_spec, t_spec, S_static = scipy.signal.spectrogram(denoised_static, fs=sr, nperseg=nperseg_spec, noverlap=nperseg_spec-256)
    im3 = ax3.pcolormesh(t_spec, f_spec/1000, 10 * np.log10(S_static + 1e-12), cmap="magma", vmin=-80, vmax=-10)
    ax3.set_ylabel("Frequency (kHz)", color=TXT, fontsize=9.5)
    ax3.set_ylim(0, 12)
    style_ax(ax3, "Spectrogram: Static Denoised (Fixed α=2.0, β=0.02)")
    
    # ── Panel 4: Spectrogram (Adaptive Denoised) ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _, _, S_adaptive = scipy.signal.spectrogram(denoised_adaptive, fs=sr, nperseg=nperseg_spec, noverlap=nperseg_spec-256)
    im4 = ax4.pcolormesh(t_spec, f_spec/1000, 10 * np.log10(S_adaptive + 1e-12), cmap="magma", vmin=-80, vmax=-10)
    ax4.set_ylabel("Frequency (kHz)", color=TXT, fontsize=9.5)
    ax4.set_ylim(0, 12)
    style_ax(ax4, "Spectrogram: State-Space Adaptive Denoised (Dynamic α & β)")
    
    # ── Panel 5: Adaptive Parameter Curves ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    t_frames = np.linspace(0, len(clean)/sr, len(alphas))
    color = "#e0a352"
    ax5.plot(t_frames, alphas, color=color, lw=2.0, label="Dynamic Subtraction Factor α")
    ax5.set_ylabel("Over-subtraction Factor α", color=color, fontsize=9.5)
    ax5.tick_params(axis="y", labelcolor=color)
    ax5.set_xlabel("Time (seconds)", color=TXT, fontsize=9.5)
    
    ax5_twin = ax5.twinx()
    color = "#52e0be"
    ax5_twin.plot(t_frames, betas, color=color, lw=1.5, ls="--", label="Dynamic Spectral Floor β")
    ax5_twin.set_ylabel("Spectral Floor β", color=color, fontsize=9.5)
    ax5_twin.tick_params(axis="y", labelcolor=color)
    ax5_twin.spines["right"].set_color("#30363d")
    
    # Combined legend
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")
    style_ax(ax5, "Parameter Evolution Curves over Vocal Phrase")
    
    # ── Panel 6: Metrics Comparison Bar Chart ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    x_indices = np.arange(2)
    w_bar = 0.35
    
    # Left Axis: SegSNR (higher is better)
    bars_snr = ax6.bar(x_indices - w_bar/2, [snr_static, snr_adaptive], w_bar, color=BL, alpha=0.85, label="SegSNR (dB) [L]")
    ax6.set_ylabel("Segmental SNR (dB)", color=BL, fontsize=9.5)
    ax6.tick_params(axis="y", labelcolor=BL)
    ax6.set_ylim(0, max(snr_static, snr_adaptive) + 3)
    
    # Right Axis: LSD (lower is better)
    ax6_twin = ax6.twinx()
    bars_lsd = ax6_twin.bar(x_indices + w_bar/2, [lsd_static, lsd_adaptive], w_bar, color=AS, alpha=0.85, label="LSD (dB) [R]")
    ax6_twin.set_ylabel("Log Spectral Distance (dB)", color=AS, fontsize=9.5)
    ax6_twin.tick_params(axis="y", labelcolor=AS)
    ax6_twin.set_ylim(0, max(lsd_static, lsd_adaptive) + 3)
    ax6_twin.spines["right"].set_color("#30363d")
    
    ax6.set_xticks(x_indices)
    ax6.set_xticklabels(["Static Baseline", "State-Space Adaptive"])
    
    # Value annotations
    for bar in bars_snr:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", color=TXT, fontsize=9, fontweight="bold")
    for bar in bars_lsd:
        h = bar.get_height()
        ax6_twin.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", color=TXT, fontsize=9, fontweight="bold")
        
    style_ax(ax6, "Objective Metrics Comparison")
    
    # Figure title
    fig.suptitle("EXPERIMENT 037 — STATE-SPACE ADAPTIVE SPECTRAL SUBTRACTION", fontsize=18, fontweight="bold", color="white", y=0.96)
    
    plot_path = os.path.join(project_root, "results/exp037_plugin_integration.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"\nSaved dashboard plot to {plot_path}")
    print("=" * 72)
    print("FINISHED Exp 037 — Plugin Integration")
    print("=" * 72)


if __name__ == "__main__":
    run()
