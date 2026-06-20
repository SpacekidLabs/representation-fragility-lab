"""
Experiment 031 — Assumption Surfaces
===================================
Maps the physical boundaries where the mathematical assumptions of different
representation algorithms (STFT, ACF, Cepstrum, CQT, and Wavelet CWT) become
invalid inside the 2D Universal Audio State Space.
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

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.representations.stft import compute_stft
from src.experiments.exp028_failure_manifold_validation import (
    compute_cqt_frame,
    compute_morlet_cwt,
    estimate_pitch_stft,
    estimate_pitch_acf,
    estimate_pitch_cepstrum,
    estimate_pitch_cqt,
    synthesize_karplus_strong,
    extract_high_energy_frames,
    apply_random_perturbation,
    cosine_similarity
)
from src.experiments.exp030_universal_audio_state_space import (
    synthesize_synth_waves,
    synthesize_fm_waves,
    synthesize_granular_textures,
    generate_noise_frames,
    extract_10_physical_descriptors
)

# ---------------------------------------------------------------------------
# CWT Pitch Estimator & Ridge Regression Helpers
# ---------------------------------------------------------------------------

def estimate_pitch_cwt(cwt: np.ndarray, sr: int, n_scales: int = 64) -> float:
    """
    Estimates pitch from Wavelet Continuous Wavelet Transform (CWT) magnitude vector.
    Maps scale index of peak energy to physical frequency.
    """
    scales = np.linspace(2, 128, n_scales)
    w0 = 5.0
    idx = np.argmax(cwt)
    s = scales[idx]
    # Frequency relation: f = (w0 / (2 * pi * s)) * sr
    f = (w0 / (2 * np.pi * s)) * sr
    return float(f)

def get_poly_features(Z: np.ndarray) -> np.ndarray:
    """
    Constructs degree-2 polynomial features from 2D coordinates:
    [1, z1, z2, z1^2, z2^2, z1*z2]
    """
    N = len(Z)
    z1 = Z[:, 0]
    z2 = Z[:, 1]
    return np.column_stack([np.ones(N), z1, z2, z1**2, z2**2, z1 * z2])

def train_ridge_regression(X_train: np.ndarray, Y_train: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Solves closed-form Ridge Regression weights:
    W = inv(X^T X + alpha * I) X^T Y
    """
    D = X_train.shape[1]
    lhs = X_train.T @ X_train + alpha * np.eye(D)
    rhs = X_train.T @ Y_train
    return np.linalg.solve(lhs, rhs)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 75)
    print("EXPERIMENT 031 — ASSUMPTION SURFACES")
    print("=" * 75)
    sr = 22050
    win = 1024
    N = 200  # 200 frames per category
    N_total = N * 10  # 2000 total frames
    rng = np.random.default_rng(42)

    print("\nLoading and synthesizing diverse signal categories (10 classes)...")
    
    # 1. Speech
    y_sp, _ = librosa.load(librosa.example('libri1'), sr=sr)
    speech_frames = extract_high_energy_frames(y_sp, win, N)

    # 2. Vocals
    y_voc, _ = librosa.load(os.path.join(project_root, "Clean_vocal.wav"), sr=sr)
    vocals_frames = extract_high_energy_frames(y_voc, win, N)

    # 3. Piano
    y_pn, _ = librosa.load(librosa.example('pistachio'), sr=sr)
    piano_frames = extract_high_energy_frames(y_pn, win, N)

    # 4. Drums
    y_dr, _ = librosa.load(librosa.example('choice'), sr=sr)
    drums_frames = extract_high_energy_frames(y_dr, win, N)

    # 5. Guitar
    guitar_frames = []
    f0s = [82.4, 110.0, 146.8, 196.0, 220.0, 329.6]
    while len(guitar_frames) < N:
        f0 = rng.choice(f0s)
        pluck = synthesize_karplus_strong(f0, sr, int(1.2 * sr))
        guitar_frames.extend(extract_high_energy_frames(pluck, win, 20))
    guitar_frames = guitar_frames[:N]

    # 6. Environmental (Robin Bird Whistles)
    y_rb, _ = librosa.load(librosa.example('robin'), sr=sr)
    robin_frames = extract_high_energy_frames(y_rb, win, N)

    # 7. Synths
    synth_frames = []
    while len(synth_frames) < N:
        synth_frames.extend(synthesize_synth_waves(rng.choice(f0s), sr, win, 50))
    synth_frames = synth_frames[:N]

    # 8. FM Tone
    fm_frames = []
    while len(fm_frames) < N:
        fm_frames.extend(synthesize_fm_waves(rng.choice(f0s), sr, win, 50))
    fm_frames = fm_frames[:N]

    # 9. Granular
    granular_frames = []
    while len(granular_textures := synthesize_granular_textures(sr, win, N)) < N:
        pass
    granular_frames = granular_textures[:N]

    # 10. Noise
    noise_frames = generate_noise_frames(win, N)

    all_corpora = {
        "Speech": speech_frames,
        "Vocals": vocals_frames,
        "Piano": piano_frames,
        "Drums": drums_frames,
        "Guitar": guitar_frames,
        "Environmental": robin_frames,
        "Synths": synth_frames,
        "FM Tone": fm_frames,
        "Granular": granular_frames,
        "Noise": noise_frames
    }

    # Gather clean representations and features
    print("\nComputing reference representations & physical features for clean frames...")
    clean_representations = []
    X_clean_feats = []
    corpus_labels = []

    for c_name, frames in all_corpora.items():
        for frame in frames:
            acf = compute_acf(frame)
            cep = compute_cepstrum(frame)
            mag = compute_stft(frame, sr)
            if mag.ndim > 1: mag = np.mean(mag, axis=1)
            cqt = compute_cqt_frame(frame, sr)
            cwt = compute_morlet_cwt(frame, 64)
            clean_representations.append((acf, cep, mag, cqt, cwt))
            
            # Physical features
            feats = extract_10_physical_descriptors(frame, sr)
            X_clean_feats.append(feats)
            corpus_labels.append(c_name)

    X_clean_feats = np.array(X_clean_feats)
    corpus_labels = np.array(corpus_labels)

    # Fit PCA (Universal Audio State Space)
    print("\nConstructing Universal Audio State Space...")
    mu = np.mean(X_clean_feats, axis=0)
    sigma = np.std(X_clean_feats, axis=0) + 1e-8
    X_clean_std = (X_clean_feats - mu) / sigma
    U, S, Vt = np.linalg.svd(X_clean_std, full_matrices=False)
    Vt = Vt[:2]
    X_clean_pca = X_clean_std @ Vt.T

    # Generate perturbed frames
    print("\nGenerating 2,000 perturbed frames...")
    perturbed_frames = []
    for i in range(N_total):
        c_name = corpus_labels[i]
        frame = all_corpora[c_name][i % N]
        p_idx = i % 12
        perturbed, _ = apply_random_perturbation(frame, sr, p_idx, rng)
        perturbed_frames.append(perturbed)

    # Compute features and representations for perturbed frames
    print("\nEvaluating representations and pitch stability on all frames (clean & perturbed)...")
    
    # We will build a unified set of 4,000 coordinates and safety labels
    # Indices 0 to 1999: Clean frames
    # Indices 2000 to 3999: Perturbed frames
    X_all_coords = []
    # Safety labels for 5 representations (STFT, ACF, Cep, CQT, CWT)
    # Shape: (4000, 5), values: 1.0 (Works), 0.5 (Degrades), 0.0 (Fails)
    Y_safety = []
    # Status lists for plotting colors: 0 (Works/Green), 1 (Degrades/Yellow), 2 (Fails/Red)
    plot_status = []

    # First add all clean frames (they are inherently Safe/Works)
    for i in range(N_total):
        X_all_coords.append(X_clean_pca[i])
        Y_safety.append([1.0, 1.0, 1.0, 1.0, 1.0])
        plot_status.append([0, 0, 0, 0, 0])

    # Now evaluate and add perturbed frames
    for i in range(N_total):
        frame_pert = perturbed_frames[i]
        
        # 10 Physical Descriptors and PCA projection
        feats_pert = extract_10_physical_descriptors(frame_pert, sr)
        feats_std = (np.array(feats_pert) - mu) / sigma
        coords_proj = feats_std @ Vt.T
        X_all_coords.append(coords_proj)
        
        # Perturbed representations
        acf_p = compute_acf(frame_pert)
        cep_p = compute_cepstrum(frame_pert)
        mag_p = compute_stft(frame_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(frame_pert, sr)
        cwt_p = compute_morlet_cwt(frame_pert, 64)
        
        # Clean reference representations
        acf_c, cep_c, mag_c, cqt_c, cwt_c = clean_representations[i]
        
        # Cosine similarities
        s_stft = cosine_similarity(mag_c, mag_p)
        s_acf  = cosine_similarity(acf_c, acf_p)
        s_cep  = cosine_similarity(cep_c, cep_p)
        s_cqt  = cosine_similarity(cqt_c, cqt_p)
        s_cwt  = cosine_similarity(cwt_c, cwt_p)
        
        # Pitch estimations
        p_stft_c = max(1.0, estimate_pitch_stft(mag_c, sr, 2*(len(mag_c)-1)))
        p_stft_p = max(1.0, estimate_pitch_stft(mag_p, sr, 2*(len(mag_p)-1)))
        e_stft = np.clip(12.0 * np.abs(np.log2(p_stft_p / p_stft_c)), 0, 12.0)

        p_acf_c = max(1.0, estimate_pitch_acf(acf_c, sr))
        p_acf_p = max(1.0, estimate_pitch_acf(acf_p, sr))
        e_acf = np.clip(12.0 * np.abs(np.log2(p_acf_p / p_acf_c)), 0, 12.0)

        p_cep_c = max(1.0, estimate_pitch_cepstrum(cep_c, sr))
        p_cep_p = max(1.0, estimate_pitch_cepstrum(cep_p, sr))
        e_cep = np.clip(12.0 * np.abs(np.log2(p_cep_p / p_cep_c)), 0, 12.0)

        p_cqt_c = max(1.0, estimate_pitch_cqt(cqt_c, sr))
        p_cqt_p = max(1.0, estimate_pitch_cqt(cqt_p, sr))
        e_cqt = np.clip(12.0 * np.abs(np.log2(p_cqt_p / p_cqt_c)), 0, 12.0)

        p_cwt_c = max(1.0, estimate_pitch_cwt(cwt_c, sr))
        p_cwt_p = max(1.0, estimate_pitch_cwt(cwt_p, sr))
        e_cwt = np.clip(12.0 * np.abs(np.log2(p_cwt_p / p_cwt_c)), 0, 12.0)
        
        # Map statuses
        rep_stats = []
        rep_labels = []
        for e, s in [(e_stft, s_stft), (e_acf, s_acf), (e_cep, s_cep), (e_cqt, s_cqt), (e_cwt, s_cwt)]:
            if e <= 1.0 and s >= 0.80:
                # Works / Safe
                rep_labels.append(1.0)
                rep_stats.append(0)
            elif e <= 3.0 and s >= 0.60:
                # Degrades / Warning
                rep_labels.append(0.5)
                rep_stats.append(1)
            else:
                # Fails / Failure
                rep_labels.append(0.0)
                rep_stats.append(2)
                
        Y_safety.append(rep_labels)
        plot_status.append(rep_stats)

    X_all_coords = np.array(X_all_coords)
    Y_safety = np.array(Y_safety)
    plot_status = np.array(plot_status)

    # ---------------------------------------------------------------------------
    # Boundary Learning via Polynomial Ridge Regression
    # ---------------------------------------------------------------------------
    print("\nLearning boundary contours (Assumption Surfaces) via Ridge Regression...")
    X_poly = get_poly_features(X_all_coords)
    
    # Train predictors for all 5 representations
    weights = []
    for r_idx in range(5):
        w = train_ridge_regression(X_poly, Y_safety[:, r_idx], alpha=0.1)
        weights.append(w)
        
    # Generate grid for contours
    grid_res = 200
    z1_min, z1_max = X_all_coords[:, 0].min() - 0.5, X_all_coords[:, 0].max() + 0.5
    z2_min, z2_max = X_all_coords[:, 1].min() - 0.5, X_all_coords[:, 1].max() + 0.5
    grid_z1, grid_z2 = np.meshgrid(np.linspace(z1_min, z1_max, grid_res), np.linspace(z2_min, z2_max, grid_res))
    grid_coords = np.column_stack([grid_z1.ravel(), grid_z2.ravel()])
    grid_poly = get_poly_features(grid_coords)
    
    grid_preds = []
    for r_idx in range(5):
        preds = (grid_poly @ weights[r_idx]).reshape(grid_z1.shape)
        grid_preds.append(preds)

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    print("\nGenerating 7-panel visualization plot...")
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.22)
    
    rep_names = ["STFT (Stationarity)", "ACF (Periodicity)", "Cepstrum (Harmonic Spacing)", 
                 "CQT (Logarithmic Spectral)", "Wavelets (CWT Multiscale)"]
    
    colors_map = {0: "#2ca02c", 1: "#bcbd22", 2: "#d62728"} # Green, Yellow, Red
    colors_list = ["#2ca02c", "#bcbd22", "#d62728"]
    labels_list = ["Works (Safe)", "Degrades (Warning)", "Catastrophically Fails"]
    
    # 1. State Space colored by Signal Classes (Clean Only)
    ax_class = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("tab10")
    for idx_c, (c_name, _) in enumerate(all_corpora.items()):
        mask = (corpus_labels == c_name)
        ax_class.scatter(X_clean_pca[mask, 0], X_clean_pca[mask, 1], s=15, color=cmap(idx_c), alpha=0.75, label=c_name)
    ax_class.set_title("Universal Audio State Space\n(Colored by Signal Category)", fontsize=11, fontweight="bold")
    ax_class.set_xlabel("PC1: Order ↔ Disorder")
    ax_class.set_ylabel("PC2: Harmonic ↔ Transient")
    ax_class.legend(fontsize=8, loc="upper right")
    ax_class.grid(True, alpha=0.08)
    
    # Define layout coordinates for representations
    axes_coords = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    # Draw individual representation plots
    for r_idx, name in enumerate(rep_names):
        r, c = axes_coords[r_idx]
        ax = fig.add_subplot(gs[r, c])
        
        # Scatter all points (clean and perturbed)
        # Using a loop to handle legend labeling easily
        for status_val in [2, 1, 0]: # Plot failed first, then degraded, then safe on top
            mask = (plot_status[:, r_idx] == status_val)
            ax.scatter(X_all_coords[mask, 0], X_all_coords[mask, 1], s=8, 
                       color=colors_list[status_val], alpha=0.6, label=labels_list[status_val] if status_val == 0 or status_val == 2 else None)
        
        # Plot boundary contour line (where pred safety score == 0.5)
        # We also draw a contour fill for safety visualization
        cs = ax.contour(grid_z1, grid_z2, grid_preds[r_idx], levels=[0.5], colors="black", linestyles="dashed", linewidths=2.5)
        ax.clabel(cs, inline=True, fmt={0.5: 'Safe Zone Boundary'}, fontsize=8)
        
        ax.set_title(f"Assumption Surface: {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.08)
        if r_idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    # 7. Consolidated Overlay
    ax_overlay = fig.add_subplot(gs[2, :])
    # Scatter background clean points lightly
    ax_overlay.scatter(X_clean_pca[:, 0], X_clean_pca[:, 1], s=4, color="white", alpha=0.18, label="Background Audio Frames")
    
    overlay_colors = ["#1f77b4", "#2ca02c", "#e377c2", "#ff7f0e", "#9467bd"]
    for r_idx, name in enumerate(rep_names):
        cs = ax_overlay.contour(grid_z1, grid_z2, grid_preds[r_idx], levels=[0.5], colors=overlay_colors[r_idx], linewidths=3.0)
        # Add legend entry by proxy
        ax_overlay.plot([], [], color=overlay_colors[r_idx], lw=3, label=f"{name} Safe Zone Boundary")
        
    ax_overlay.set_title("Consolidated Assumption Surfaces Overlay (Universal Safe Zones)", fontsize=12, fontweight="bold")
    ax_overlay.set_xlabel("PC1: Order ↔ Disorder")
    ax_overlay.set_ylabel("PC2: Harmonic ↔ Transient")
    ax_overlay.legend(fontsize=10, loc="upper right")
    ax_overlay.grid(True, alpha=0.08)
    
    # Annotated regions on the overlay plot
    ax_overlay.text(-3.0, -1.5, "Harmonic Core\n(All assumptions valid)", color="green", weight="bold", fontsize=10, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_overlay.text(2.5, 2.0, "Noise Collapse\n(All assumptions invalid)", color="red", weight="bold", fontsize=10, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_overlay.text(-1.5, 2.5, "Log-Spectral Region\n(CQT/Wavelet valid, ACF/Cep fail)", color="orange", weight="bold", fontsize=10, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    
    fig.suptitle("Experiment 031 — Assumption Surfaces:\nMapping the Boundaries where DSP Representation Assumptions Fail in the Universal Audio State Space", fontsize=16, fontweight="bold", y=0.98)
    
    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp031_assumption_surfaces.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Assumption Surfaces plot: {out_path}")
    print("=" * 75)

if __name__ == "__main__":
    run()
