"""
Experiment 038 — Learned DSP Controller
=======================================
Learns the optimal DSP control surface mapping engine coordinates (z1, z2)
to over-subtraction factor alpha and spectral floor beta, minimizing Log 
Spectral Distance (LSD) directly.

Outputs:
  results/exp038_vocal_test_noisy.wav
  results/exp038_vocal_test_static.wav
  results/exp038_vocal_test_handcrafted.wav
  results/exp038_vocal_test_learned.wav
  results/exp038_learned_controller.png
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

from sklearn.ensemble import RandomForestRegressor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

from src.framework.engine import RepresentationIntelligenceEngine


# ──────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def norm01(arr):
    """Normalize array amplitude to [0, 1]."""
    mx = np.max(np.abs(arr))
    return arr / mx if mx > 1e-12 else arr


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
        
        if sig_pow > 1e-6: # active frames only
            db = 10 * np.log10(sig_pow / (err_pow + 1e-12))
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
    print("EXPERIMENT 038 — LEARNED DSP CONTROLLER: AUTOMATING CONTROL SURFACES")
    print("=" * 72)
    
    # 1. Load Vocal & Split
    wav_path = os.path.join(project_root, "Clean_vocal.wav")
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.")
        return
        
    sr, data = wav.read(wav_path)
    clean = data.astype(np.float32) / 32768.0
    
    # Corrupt with white noise (+6 dB SNR)
    rng = np.random.default_rng(999)
    sig_power = np.mean(clean ** 2)
    noise_power = sig_power / (10 ** (6.0 / 10.0))
    noise_std = np.sqrt(noise_power)
    noisy = clean + rng.normal(0, noise_std, len(clean))
    
    # Train / Test split (Time-domain half-half)
    split_idx = len(clean) // 2
    clean_train, clean_test = clean[:split_idx], clean[split_idx:]
    noisy_train, noisy_test = noisy[:split_idx], noisy[split_idx:]
    
    print(f"Dataset split: Train = {len(clean_train)/sr:.2f}s, Test = {len(clean_test)/sr:.2f}s")
    
    # Save test noisy signal
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    def save_wav(name, sig):
        path = os.path.join(results_dir, name)
        out_int = np.clip(sig * 32768.0, -32768.0, 32767.0).astype(np.int16)
        wav.write(path, sr, out_int)
        print(f"  Saved: {path}")
        
    save_wav("exp038_vocal_test_noisy.wav", noisy_test)
    
    # 2. Collect training dataset via local grid-search
    print("\nRunning frame-level grid-search on training set...")
    engine = RepresentationIntelligenceEngine()
    
    nperseg = 2048
    hop = 512
    noverlap = nperseg - hop
    
    f, t_train, Z_train_noisy = scipy.signal.stft(noisy_train, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, _, Z_train_clean = scipy.signal.stft(clean_train, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    # Noise estimate from first 8 frames of training noisy vocal
    noise_psd = np.mean(np.abs(Z_train_noisy[:, :8]) ** 2, axis=1)
    sqrt_npsd = np.sqrt(noise_psd)
    
    # Parameter grid definition
    grid_alphas = np.linspace(0.0, 5.0, 11)
    grid_betas = np.array([0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15])
    
    X_train = []
    Y_train = []
    
    for m in range(Z_train_noisy.shape[1]):
        # Get frame state coordinates
        start_idx = m * hop
        end_idx = start_idx + nperseg
        frame = noisy_train[start_idx:end_idx]
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
            
        frame_22k = scipy.signal.resample(frame, nperseg // 2)
        st = engine.analyze(frame_22k, 22050)
        
        # Grid search local optimal parameters minimizing LSD
        X_clean = np.abs(Z_train_clean[:, m])
        X_noisy = np.abs(Z_train_noisy[:, m])
        clean_db = 20 * np.log10(X_clean + 1e-9)
        
        best_lsd = 999.0
        best_params = (2.0, 0.02)
        
        for a in grid_alphas:
            for b in grid_betas:
                X_est = np.maximum(X_noisy - a * sqrt_npsd, b * X_noisy)
                est_db = 20 * np.log10(X_est + 1e-9)
                lsd = np.sqrt(np.mean((clean_db - est_db) ** 2))
                if lsd < best_lsd:
                    best_lsd = lsd
                    best_params = (a, b)
                    
        X_train.append(st.coordinate)
        Y_train.append(best_params)
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # 3. Train RandomForestRegressor mapping coordinates -> optimal parameters
    print("Training the Learned DSP Controller...")
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, Y_train)
    
    # 4. Out-of-sample processing on test set
    print("\nProcessing test set with three controllers...")
    f, t_test, Z_test_noisy = scipy.signal.stft(noisy_test, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    Z_static = np.zeros_like(Z_test_noisy, dtype=np.complex128)
    Z_handcrafted = np.zeros_like(Z_test_noisy, dtype=np.complex128)
    Z_learned = np.zeros_like(Z_test_noisy, dtype=np.complex128)
    
    pred_alphas = []
    pred_betas = []
    hc_alphas = []
    hc_betas = []
    test_coords = []
    
    for m in range(Z_test_noisy.shape[1]):
        start_idx = m * hop
        end_idx = start_idx + nperseg
        frame = noisy_test[start_idx:end_idx]
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
            
        frame_22k = scipy.signal.resample(frame, nperseg // 2)
        st = engine.analyze(frame_22k, 22050)
        test_coords.append(st.coordinate)
        
        # A. Static Parameters
        a_static = 2.0
        b_static = 0.02
        
        # B. Hand-crafted Parameters
        stft_s = st.assumptions["stft"]
        a_hc = 0.5 + 3.5 * (1.0 - stft_s)
        if st.region == "noise_collapse":
            b_hc = 0.06
        elif st.region == "periodic_harmonic":
            b_hc = 0.005
        else:
            b_hc = 0.02
        hc_alphas.append(a_hc)
        hc_betas.append(b_hc)
        
        # C. Learned Parameters
        pred = model.predict([st.coordinate])[0]
        a_learned = np.clip(pred[0], 0.0, 5.0)
        b_learned = np.clip(pred[1], 0.001, 0.15)
        pred_alphas.append(a_learned)
        pred_betas.append(b_learned)
        
        # Apply filters
        X_mag = np.abs(Z_test_noisy[:, m])
        
        # Static
        G_static = np.maximum(1.0 - a_static * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), b_static)
        Z_static[:, m] = Z_test_noisy[:, m] * G_static
        
        # Hand-crafted
        G_hc = np.maximum(1.0 - a_hc * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), b_hc)
        Z_handcrafted[:, m] = Z_test_noisy[:, m] * G_hc
        
        # Learned
        G_learned = np.maximum(1.0 - a_learned * np.sqrt(noise_psd / (X_mag ** 2 + 1e-12)), b_learned)
        Z_learned[:, m] = Z_test_noisy[:, m] * G_learned
        
    _, test_static = scipy.signal.istft(Z_static, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, test_handcrafted = scipy.signal.istft(Z_handcrafted, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, test_learned = scipy.signal.istft(Z_learned, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    test_static = test_static[:len(noisy_test)]
    test_handcrafted = test_handcrafted[:len(noisy_test)]
    test_learned = test_learned[:len(noisy_test)]
    
    save_wav("exp038_vocal_test_static.wav", test_static)
    save_wav("exp038_vocal_test_handcrafted.wav", test_handcrafted)
    save_wav("exp038_vocal_test_learned.wav", test_learned)
    
    # 5. Metrics evaluation
    snr_noisy = compute_segsnr(clean_test, noisy_test, sr)
    snr_static = compute_segsnr(clean_test, test_static, sr)
    snr_hc = compute_segsnr(clean_test, test_handcrafted, sr)
    snr_learned = compute_segsnr(clean_test, test_learned, sr)
    
    lsd_noisy = compute_lsd(clean_test, noisy_test, sr)
    lsd_static = compute_lsd(clean_test, test_static, sr)
    lsd_hc = compute_lsd(clean_test, test_handcrafted, sr)
    lsd_learned = compute_lsd(clean_test, test_learned, sr)
    
    print("\n" + "-"*50)
    print("OUT-OF-SAMPLED PERFORMANCE ON TEST SET:")
    print(f"  Noisy Input     : SegSNR = {snr_noisy:5.2f} dB, LSD = {lsd_noisy:5.2f} dB")
    print(f"  Static (Base)   : SegSNR = {snr_static:5.2f} dB, LSD = {lsd_static:5.2f} dB")
    print(f"  Hand-crafted    : SegSNR = {snr_hc:5.2f} dB, LSD = {lsd_hc:5.2f} dB")
    print(f"  Learned State   : SegSNR = {snr_learned:5.2f} dB, LSD = {lsd_learned:5.2f} dB")
    print(f"  Delta LSD (vs HC): {lsd_learned - lsd_hc:+.2f} dB (lower is better)")
    print(f"  Delta SNR (vs HC): {snr_learned - snr_hc:+.2f} dB (higher is better)")
    print("-"*50 + "\n")
    
    # ──────────────────────────────────────────────────────────────────────────
    # DASHBOARD PLOT: 3x2 Dark Layout
    # ──────────────────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")
    
    gs = fig.add_gridspec(
        3, 2, hspace=0.45, wspace=0.30,
        left=0.08, right=0.94, top=0.90, bottom=0.08
    )
    
    BL  = "#e05252"   # baseline red
    AS  = "#52b0e0"   # handcrafted blue
    ACC = "#f4c542"   # learned gold
    TXT = "#c9d1d9"
    BG  = "#161b22"
    
    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, fontweight="bold", color="white", fontsize=12, pad=10)
        ax.tick_params(colors=TXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
            
    # ── Panel 1: Learned Alpha Control Surface ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    # Build grid to plot continuous control surface
    gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 80), np.linspace(-3.5, 3.5, 80))
    grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T
    grid_preds = model.predict(grid_pts)
    
    im1 = ax1.pcolormesh(gx, gy, grid_preds[:, 0].reshape(80, 80), cmap="inferno", shading="auto")
    fig.colorbar(im1, ax=ax1, label="Learned Alpha")
    ax1.set_xlabel("z1 (Order ↔ Disorder)", color=TXT, fontsize=9.5)
    ax1.set_ylabel("z2 (Harmonic ↔ Transient)", color=TXT, fontsize=9.5)
    style_ax(ax1, "Learned Alpha Surface g(z1, z2)_0")
    
    # ── Panel 2: Learned Beta Control Surface ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(gx, gy, grid_preds[:, 1].reshape(80, 80), cmap="viridis", shading="auto")
    fig.colorbar(im2, ax=ax2, label="Learned Beta")
    ax2.set_xlabel("z1 (Order ↔ Disorder)", color=TXT, fontsize=9.5)
    ax2.set_ylabel("z2 (Harmonic ↔ Transient)", color=TXT, fontsize=9.5)
    style_ax(ax2, "Learned Beta Surface g(z1, z2)_1")
    
    t_frames = np.linspace(0, len(clean_test)/sr, len(pred_alphas))
    
    # ── Panel 3: Alpha Parameter Comparison ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_frames, hc_alphas, color=AS, lw=2.0, alpha=0.7, label="Hand-crafted Rule")
    ax3.plot(t_frames, pred_alphas, color=ACC, lw=2.0, alpha=0.9, label="Learned Controller")
    ax3.set_xlabel("Time (seconds)", color=TXT, fontsize=9.5)
    ax3.set_ylabel("Over-subtraction Factor α", color=TXT, fontsize=9.5)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(axis="y", alpha=0.1, color="#2a2a3a")
    style_ax(ax3, "Over-subtraction α Comparison")
    
    # ── Panel 4: Beta Parameter Comparison ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_frames, hc_betas, color=AS, lw=2.0, alpha=0.7, label="Hand-crafted Rule")
    ax4.plot(t_frames, pred_betas, color=ACC, lw=2.0, alpha=0.9, label="Learned Controller")
    ax4.set_xlabel("Time (seconds)", color=TXT, fontsize=9.5)
    ax4.set_ylabel("Spectral Floor β", color=TXT, fontsize=9.5)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(axis="y", alpha=0.1, color="#2a2a3a")
    style_ax(ax4, "Spectral Floor β Comparison")
    
    # ── Panel 5: Spectrogram (Learned Denoised) ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    f_spec, t_spec, S_learned = scipy.signal.spectrogram(test_learned, fs=sr, nperseg=1024, noverlap=1024-256)
    ax5.pcolormesh(t_spec, f_spec/1000, 10 * np.log10(S_learned + 1e-12), cmap="magma", vmin=-80, vmax=-10)
    ax5.set_ylabel("Frequency (kHz)", color=TXT, fontsize=9.5)
    ax5.set_xlabel("Time (seconds)", color=TXT, fontsize=9.5)
    ax5.set_ylim(0, 12)
    style_ax(ax5, "Spectrogram: Learned State-Space Denoised Output")
    
    # ── Panel 6: Metrics Comparison Bar Chart ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    x_indices = np.arange(3)
    w_bar = 0.32
    
    # Left Axis: SegSNR (higher is better)
    bars_snr = ax6.bar(x_indices - w_bar/2, [snr_static, snr_hc, snr_learned], w_bar, color=BL, alpha=0.85, label="SegSNR (dB) [L]")
    ax6.set_ylabel("Segmental SNR (dB)", color=BL, fontsize=9.5)
    ax6.tick_params(axis="y", labelcolor=BL)
    ax6.set_ylim(0, max(snr_static, snr_hc, snr_learned) + 3)
    
    # Right Axis: LSD (lower is better)
    ax6_twin = ax6.twinx()
    bars_lsd = ax6_twin.bar(x_indices + w_bar/2, [lsd_static, lsd_hc, lsd_learned], w_bar, color=AS, alpha=0.85, label="LSD (dB) [R]")
    ax6_twin.set_ylabel("Log Spectral Distance (dB)", color=AS, fontsize=9.5)
    ax6_twin.tick_params(axis="y", labelcolor=AS)
    ax6_twin.set_ylim(0, max(lsd_static, lsd_hc, lsd_learned) + 3)
    ax6_twin.spines["right"].set_color("#30363d")
    
    ax6.set_xticks(x_indices)
    ax6.set_xticklabels(["Static Baseline", "Hand-crafted", "Learned State-Space"], fontsize=9)
    
    # Value annotations
    for bar in bars_snr:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", color=TXT, fontsize=9, fontweight="bold")
    for bar in bars_lsd:
        h = bar.get_height()
        ax6_twin.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", color=TXT, fontsize=9, fontweight="bold")
        
    style_ax(ax6, "Out-of-Sample Metrics Comparison")
    
    # Colorbar positioning to avoid overlap
    fig.suptitle("EXPERIMENT 038 — LEARNED DSP CONTROLLER PERFORMANCE", fontsize=18, fontweight="bold", color="white", y=0.96)
    
    plot_path = os.path.join(project_root, "results/exp038_learned_controller.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    print(f"\nSaved dashboard plot to {plot_path}")
    print("=" * 72)
    print("FINISHED Exp 038 — Learned Controller")
    print("=" * 72)


if __name__ == "__main__":
    run()
