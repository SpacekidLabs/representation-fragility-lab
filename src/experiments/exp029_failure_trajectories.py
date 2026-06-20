"""
Experiment 029 — Failure Trajectories
======================================
Plots paths through the failure manifold to treat representation collapse as a continuous dynamical system,
validating clusters with density-based DBSCAN, identifying axis physical meanings, and creating a DSP control layer.
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

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

from src.representations.acf import compute_acf
from src.representations.cepstrum import compute_cepstrum
from src.representations.stft import compute_stft
from src.experiments.exp028_failure_manifold_validation import (
    synthesize_karplus_strong,
    compute_morlet_cwt,
    compute_cqt_frame,
    compute_mel_frame,
    apply_random_perturbation,
    extract_17_descriptors,
    cosine_similarity,
    run_pca,
    extract_high_energy_frames
)

# ---------------------------------------------------------------------------
# Pure NumPy DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN) in pure NumPy.
    """
    n = X.shape[0]
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    
    # Pairwise distances
    dists = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        
        neighbors = np.where(dists[i] < eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
        else:
            labels[i] = cluster_id
            queue = list(neighbors)
            
            idx = 0
            while idx < len(queue):
                p = queue[idx]
                if not visited[p]:
                    visited[p] = True
                    p_neighbors = np.where(dists[p] < eps)[0]
                    if len(p_neighbors) >= min_samples:
                        for q in p_neighbors:
                            if q not in queue:
                                queue.append(q)
                if labels[p] == -1:
                    labels[p] = cluster_id
                idx += 1
            cluster_id += 1
            
    return labels

# ---------------------------------------------------------------------------
# Axis Correlation Helpers
# ---------------------------------------------------------------------------

def pearson_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_c = v1 - np.mean(v1)
    v2_c = v2 - np.mean(v2)
    num = np.sum(v1_c * v2_c)
    denom = np.sqrt(np.sum(v1_c**2) * np.sum(v2_c**2)) + 1e-10
    return float(num / denom)

# ---------------------------------------------------------------------------
# DSP Navigation Layer
# ---------------------------------------------------------------------------

def get_dsp_recommendation(pc1: float, pc2: float) -> dict:
    """
    Maps 2D failure coordinates to active DSP parameters.
    """
    # 1. Noise Collapse (PC1 < -2.5)
    if pc1 < -2.5:
        return {
            "state": "Noise Collapse",
            "action": "Mute ACF/Cep (0.0); Route 100% to STFT; Window = 4096 (Integrate Noise)",
            "stft_w": 1.0, "acf_w": 0.0, "cep_w": 0.0, "window": 4096
        }
    # 2. Periodicity / Harmonic Collapse (Smooth lowpass sparse signal)
    elif pc1 > 0.5 and pc2 < 0.9:
        return {
            "state": "Periodicity Collapse",
            "action": "Trust STFT/CQT (0.7); Mute Cep (0.0); Window = 2048",
            "stft_w": 0.7, "acf_w": 0.3, "cep_w": 0.0, "window": 2048
        }
    # 3. Saturation / Distortion (Moderate noise, but high PC2 and negative PC1 due to clipping harmonics)
    elif pc1 < -0.5 and 0.0 < pc2 < 0.8:
        return {
            "state": "Saturation Collapse",
            "action": "Trust Mel/STFT (0.8); Mute ACF/Cep (0.1); Enable High-Freq Gating",
            "stft_w": 0.8, "acf_w": 0.1, "cep_w": 0.1, "window": 1024
        }
    # 4. Healthy / Normal periodic signals
    else:
        return {
            "state": "Healthy Zone",
            "action": "Trust ACF/Cep (0.85); Use STFT (0.15); Window = 1024 (Low Latency)",
            "stft_w": 0.15, "acf_w": 0.45, "cep_w": 0.40, "window": 1024
        }

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("EXPERIMENT 029 — FAILURE TRAJECTORIES")
    print("=" * 70)
    sr = 22050
    win = 1024
    N_per_corpus = 400
    N_total = 2000

    # 1. Load diverse signal frames (same setup as Exp 028 to maintain baseline)
    print("\nLoading corpus frames...")
    y_voc, _ = librosa.load(os.path.join(project_root, "Clean_vocal.wav"), sr=sr)
    vocals_frames = extract_high_energy_frames(y_voc, win, N_per_corpus)
    
    y_sp, _ = librosa.load(librosa.example('libri1'), sr=sr)
    speech_frames = extract_high_energy_frames(y_sp, win, N_per_corpus)
    
    y_pn, _ = librosa.load(librosa.example('pistachio'), sr=sr)
    piano_frames = extract_high_energy_frames(y_pn, win, N_per_corpus)
    
    y_dr, _ = librosa.load(librosa.example('choice'), sr=sr)
    drums_frames = extract_high_energy_frames(y_dr, win, N_per_corpus)
    
    guitar_frames = []
    rng = np.random.default_rng(42)
    f0s = [110.0, 146.8, 196.0, 220.0, 329.6]
    while len(guitar_frames) < N_per_corpus:
        f0 = rng.choice(f0s)
        pluck = synthesize_karplus_strong(f0, sr, int(1.2 * sr))
        note_frames = extract_high_energy_frames(pluck, win, 20)
        guitar_frames.extend(note_frames)
    guitar_frames = guitar_frames[:N_per_corpus]

    all_corpora = {
        "Vocals": vocals_frames,
        "Speech": speech_frames,
        "Piano": piano_frames,
        "Drums": drums_frames,
        "Guitar": guitar_frames
    }

    # Reference representation store
    clean_representations = []
    for c_name, frames in all_corpora.items():
        for frame in frames:
            acf = compute_acf(frame)
            cep = compute_cepstrum(frame)
            mag = compute_stft(frame, sr)
            if mag.ndim > 1: mag = np.mean(mag, axis=1)
            cqt = compute_cqt_frame(frame, sr)
            cwt = compute_morlet_cwt(frame, 64)
            mel = compute_mel_frame(frame, sr)
            clean_representations.append((acf, cep, mag, cqt, cwt, mel))

    # Generate perturbed database
    perturbed_frames = []
    perturbation_labels = []
    corpus_labels = []
    
    idx = 0
    for c_name, frames in all_corpora.items():
        for frame in frames:
            p_idx = idx % 12
            perturbed, p_label = apply_random_perturbation(frame, sr, p_idx, rng)
            perturbed_frames.append(perturbed)
            perturbation_labels.append(p_label)
            corpus_labels.append(c_name)
            idx += 1

    # Feature extraction & similarities
    X_features = []
    similarities = []
    physical_snr = []
    
    for i in range(N_total):
        frame_clean = all_corpora[corpus_labels[i]][i % N_per_corpus]
        frame_pert = perturbed_frames[i]
        
        # Power metrics for physical SNR
        p_sig = np.mean(frame_clean**2)
        p_noise = np.mean((frame_pert - frame_clean)**2)
        snr_db = 10 * np.log10(p_sig / (p_noise + 1e-12))
        physical_snr.append(snr_db)
        
        # representations
        acf_p = compute_acf(frame_pert)
        cep_p = compute_cepstrum(frame_pert)
        mag_p = compute_stft(frame_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(frame_pert, sr)
        cwt_p = compute_morlet_cwt(frame_pert, 64)
        mel_p = compute_mel_frame(frame_pert, sr)
        
        acf_c, cep_c, mag_c, cqt_c, cwt_c, mel_c = clean_representations[i]
        
        sim_stft = cosine_similarity(mag_c, mag_p)
        sim_acf  = cosine_similarity(acf_c, acf_p)
        sim_cep  = cosine_similarity(cep_c, cep_p)
        sim_cqt  = cosine_similarity(cqt_c, cqt_p)
        sim_cwt  = cosine_similarity(cwt_c, cwt_p)
        sim_mel  = cosine_similarity(mel_c, mel_p)
        similarities.append([sim_stft, sim_acf, sim_cep, sim_cqt, sim_cwt, sim_mel])
        
        feats = extract_17_descriptors(frame_pert, acf_p, cep_p, mag_p, cqt_p, cwt_p, mel_p, sr, 2 * (len(mag_p)-1))
        X_features.append(feats)

    X = np.array(X_features)
    similarities = np.array(similarities)
    physical_snr = np.array(physical_snr)

    # PCA Fitting
    print("\nFitting Failure Manifold PCA projection...")
    # Standardise
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8
    X_std = (X - mu) / sigma
    
    X_pca, var_exp, Vt = run_pca(X)
    print(f"Total PCA Explained Variance (2D): {np.sum(var_exp):.2%}")

    # ---------------------------------------------------------------------------
    # [TEST 1] DBSCAN Clustering
    # ---------------------------------------------------------------------------
    print("\n[TEST 1] Running pure NumPy DBSCAN on 2D coordinates...")
    # Standardise coordinate scales for DBSCAN
    X_pca_std = (X_pca - np.mean(X_pca, axis=0)) / np.std(X_pca, axis=0)
    db_labels = run_dbscan(X_pca_std, eps=0.35, min_samples=15)
    
    unique_clusters = np.unique(db_labels)
    print(f"DBSCAN complete. Found {len(unique_clusters) - (1 if -1 in db_labels else 0)} clusters + noise.")
    for cid in unique_clusters:
        mask = (db_labels == cid)
        size = np.sum(mask)
        m_sims = similarities[mask].mean(axis=0)
        lbl = "Noise" if cid == -1 else f"Cluster {cid}"
        print(f" - {lbl:<10} (Size: {size:<4}): STFT={m_sims[0]:.3f}, ACF={m_sims[1]:.3f}, Cep={m_sims[2]:.3f}")

    # ---------------------------------------------------------------------------
    # [TEST 2] Axis Physical Correlation
    # ---------------------------------------------------------------------------
    print("\n[TEST 2] Correlating PCA axes with physical signal descriptors...")
    # Get descriptors:
    # 0 = Spectral Entropy, 9 = ZCR
    # We also have physical SNR, Cepstral Peak Prominence (harmonicity), ACF Peak Prominence (periodicity)
    entropy_feats = X[:, 0]
    zcr_feats = X[:, 9]
    harmonicity_feats = X[:, 8]
    periodicity_feats = X[:, 5]
    
    # Calculate correlations
    corr_pc1 = {
        "SNR (dB)": pearson_correlation(X_pca[:, 0], physical_snr),
        "Spectral Entropy": pearson_correlation(X_pca[:, 0], entropy_feats),
        "ZCR": pearson_correlation(X_pca[:, 0], zcr_feats),
        "Harmonicity (Cep)": pearson_correlation(X_pca[:, 0], harmonicity_feats),
        "Periodicity (ACF)": pearson_correlation(X_pca[:, 0], periodicity_feats)
    }
    
    corr_pc2 = {
        "SNR (dB)": pearson_correlation(X_pca[:, 1], physical_snr),
        "Spectral Entropy": pearson_correlation(X_pca[:, 1], entropy_feats),
        "ZCR": pearson_correlation(X_pca[:, 1], zcr_feats),
        "Harmonicity (Cep)": pearson_correlation(X_pca[:, 1], harmonicity_feats),
        "Periodicity (ACF)": pearson_correlation(X_pca[:, 1], periodicity_feats)
    }
    
    print(f"{'Physical Metric':<20} | {'PC1 (r)':<8} | {'PC2 (r)':<8}")
    print("-" * 44)
    for k in corr_pc1.keys():
        print(f"{k:<20} | {corr_pc1[k]:<8.3f} | {corr_pc2[k]:<8.3f}")

    # ---------------------------------------------------------------------------
    # [TEST 3] Continuous Trajectory Sweeping
    # ---------------------------------------------------------------------------
    print("\n[TEST 3] Generating continuous degradation trajectories...")
    steps = 50
    
    # Path 1: Vocal Noise Path
    # Extract a voiced vocal frame
    vocal_clean = vocals_frames[0]
    acf_vc = compute_acf(vocal_clean)
    cep_vc = compute_cepstrum(vocal_clean)
    mag_vc = compute_stft(vocal_clean, sr)
    if mag_vc.ndim > 1: mag_vc = np.mean(mag_vc, axis=1)
    cqt_vc = compute_cqt_frame(vocal_clean, sr)
    cwt_vc = compute_morlet_cwt(vocal_clean, 64)
    mel_vc = compute_mel_frame(vocal_clean, sr)
    
    path_1_coords = []
    path_1_recs = []
    
    for k in range(steps):
        sig = 1.0 * (k / (steps - 1))
        # Add noise
        y_pert = vocal_clean + rng.normal(0, sig, win)
        # Normalize RMS
        y_pert *= (np.sqrt(np.mean(vocal_clean**2)) / np.sqrt(np.mean(y_pert**2) + 1e-12))
        
        # Representations
        acf_p = compute_acf(y_pert)
        cep_p = compute_cepstrum(y_pert)
        mag_p = compute_stft(y_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(y_pert, sr)
        cwt_p = compute_morlet_cwt(y_pert, 64)
        mel_p = compute_mel_frame(y_pert, sr)
        
        # Features
        feats = extract_17_descriptors(y_pert, acf_p, cep_p, mag_p, cqt_p, cwt_p, mel_p, sr, 2*(len(mag_p)-1))
        feats_std = (np.array(feats) - mu) / sigma
        coords = feats_std @ Vt.T
        path_1_coords.append(coords)
        
        # Recommendation
        path_1_recs.append(get_dsp_recommendation(coords[0], coords[1]))
    path_1_coords = np.array(path_1_coords)

    # Path 2: Guitar Filtering Path
    guitar_clean = guitar_frames[0]
    acf_gc = compute_acf(guitar_clean)
    cep_gc = compute_cepstrum(guitar_clean)
    mag_gc = compute_stft(guitar_clean, sr)
    if mag_gc.ndim > 1: mag_gc = np.mean(mag_gc, axis=1)
    cqt_gc = compute_cqt_frame(guitar_clean, sr)
    cwt_gc = compute_morlet_cwt(guitar_clean, 64)
    mel_gc = compute_mel_frame(guitar_clean, sr)
    
    path_2_coords = []
    path_2_recs = []
    
    for k in range(steps):
        fc = 2000.0 - 1850.0 * (k / (steps - 1))
        # Filter
        b, a = scipy.signal.butter(4, fc / (sr / 2.0), btype='low')
        y_pert = scipy.signal.filtfilt(b, a, guitar_clean)
        # Normalize RMS
        y_pert *= (np.sqrt(np.mean(guitar_clean**2)) / np.sqrt(np.mean(y_pert**2) + 1e-12))
        
        acf_p = compute_acf(y_pert)
        cep_p = compute_cepstrum(y_pert)
        mag_p = compute_stft(y_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(y_pert, sr)
        cwt_p = compute_morlet_cwt(y_pert, 64)
        mel_p = compute_mel_frame(y_pert, sr)
        
        feats = extract_17_descriptors(y_pert, acf_p, cep_p, mag_p, cqt_p, cwt_p, mel_p, sr, 2*(len(mag_p)-1))
        feats_std = (np.array(feats) - mu) / sigma
        coords = feats_std @ Vt.T
        path_2_coords.append(coords)
        path_2_recs.append(get_dsp_recommendation(coords[0], coords[1]))
    path_2_coords = np.array(path_2_coords)

    # Path 3: Piano Saturation Path
    piano_clean = piano_frames[0]
    acf_pc = compute_acf(piano_clean)
    cep_pc = compute_cepstrum(piano_clean)
    mag_pc = compute_stft(piano_clean, sr)
    if mag_pc.ndim > 1: mag_pc = np.mean(mag_pc, axis=1)
    cqt_pc = compute_cqt_frame(piano_clean, sr)
    cwt_pc = compute_morlet_cwt(piano_clean, 64)
    mel_pc = compute_mel_frame(piano_clean, sr)
    
    path_3_coords = []
    path_3_recs = []
    
    for k in range(steps):
        drive = 1.0 + 19.0 * (k / (steps - 1))
        # Saturation
        y_pert = np.tanh(drive * piano_clean) / np.tanh(drive)
        # Normalize RMS
        y_pert *= (np.sqrt(np.mean(piano_clean**2)) / np.sqrt(np.mean(y_pert**2) + 1e-12))
        
        acf_p = compute_acf(y_pert)
        cep_p = compute_cepstrum(y_pert)
        mag_p = compute_stft(y_pert, sr)
        if mag_p.ndim > 1: mag_p = np.mean(mag_p, axis=1)
        cqt_p = compute_cqt_frame(y_pert, sr)
        cwt_p = compute_morlet_cwt(y_pert, 64)
        mel_p = compute_mel_frame(y_pert, sr)
        
        feats = extract_17_descriptors(y_pert, acf_p, cep_p, mag_p, cqt_p, cwt_p, mel_p, sr, 2*(len(mag_p)-1))
        feats_std = (np.array(feats) - mu) / sigma
        coords = feats_std @ Vt.T
        path_3_coords.append(coords)
        path_3_recs.append(get_dsp_recommendation(coords[0], coords[1]))
    path_3_coords = np.array(path_3_coords)

    # ---------------------------------------------------------------------------
    # [TEST 4] Navigation Log Tracing
    # ---------------------------------------------------------------------------
    print("\n[TEST 4] Tracing active DSP Control recommendations along trajectories:")
    print("=" * 105)
    print(f"{'Path / Step':<25} | {'State Coordinate':<15} | {'Detected State':<22} | {'Active DSP Action Recommendation'}")
    print("-" * 105)
    
    trace_steps = [0, 15, 30, 45]
    for ts in trace_steps:
        rec = path_1_recs[ts]
        coord_str = f"({path_1_coords[ts,0]:.2f}, {path_1_coords[ts,1]:.2f})"
        print(f"{'Vocals Noise - Step ' + str(ts):<25} | {coord_str:<15} | {rec['state']:<22} | {rec['action']}")
    print("-" * 105)
    for ts in trace_steps:
        rec = path_2_recs[ts]
        coord_str = f"({path_2_coords[ts,0]:.2f}, {path_2_coords[ts,1]:.2f})"
        print(f"{'Guitar Lowpass - Step ' + str(ts):<25} | {coord_str:<15} | {rec['state']:<22} | {rec['action']}")
    print("-" * 105)
    for ts in trace_steps:
        rec = path_3_recs[ts]
        coord_str = f"({path_3_coords[ts,0]:.2f}, {path_3_coords[ts,1]:.2f})"
        print(f"{'Piano Saturation - Step ' + str(ts):<25} | {coord_str:<15} | {rec['state']:<22} | {rec['action']}")
    print("=" * 105)

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    print("\nGenerating failure trajectory and validation plots...")
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: DBSCAN Clusters
    ax_db = fig.add_subplot(gs[0, 0])
    scatter_colors = ["#377eb8", "#e7298a", "#a6d854", "#ff7f00", "#984ea3"]
    # Plot DBSCAN clusters
    db_u = np.unique(db_labels)
    for cid in db_u:
        mask = (db_labels == cid)
        col = "gray" if cid == -1 else scatter_colors[cid % len(scatter_colors)]
        lbl = "Noise / Outliers" if cid == -1 else f"Cluster {cid}"
        ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, color=col, alpha=0.6, label=lbl)
    ax_db.set_title("DBSCAN Density-Based Clusters (Verification of Cluster Reality)", fontsize=11, fontweight="bold")
    ax_db.set_xlabel("PC1")
    ax_db.set_ylabel("PC2")
    ax_db.legend(fontsize=8, loc="lower left")
    ax_db.grid(True, alpha=0.08)

    # Panel 2: Physical Metric Correlations
    ax_corr = fig.add_subplot(gs[0, 1])
    x_lbls = list(corr_pc1.keys())
    x_pos = np.arange(len(x_lbls))
    width = 0.35
    
    ax_corr.bar(x_pos - width/2, [corr_pc1[k] for k in x_lbls], width, label="PC1 Correlation", color="#a6d854")
    ax_corr.bar(x_pos + width/2, [corr_pc2[k] for k in x_lbls], width, label="PC2 Correlation", color="#377eb8")
    ax_corr.set_xticks(x_pos)
    ax_corr.set_xticklabels(x_lbls, fontsize=8)
    ax_corr.set_ylabel("Pearson Correlation (r)")
    ax_corr.set_title("Physical Interpretation of PCA Axes (Manifold Semantics)", fontsize=11, fontweight="bold")
    ax_corr.legend(fontsize=9, loc="upper right")
    ax_corr.grid(True, alpha=0.08, axis="y")

    # Panel 3: Continuous Trajectories
    ax_traj = fig.add_subplot(gs[1, :])
    # Scatter background database points lightly
    ax_traj.scatter(X_pca[:, 0], X_pca[:, 1], s=3, color="white", alpha=0.12, label="Database Frames")
    
    # Plot Trajectory 1: Vocals Noise Path
    ax_traj.plot(path_1_coords[:, 0], path_1_coords[:, 1], color="#e7298a", linewidth=2.5, label="Vocals + Noise Path")
    ax_traj.scatter(path_1_coords[0, 0], path_1_coords[0, 1], marker="o", s=80, color="#e7298a", edgecolors="white", zorder=5)
    ax_traj.scatter(path_1_coords[-1, 0], path_1_coords[-1, 1], marker="X", s=100, color="#e7298a", edgecolors="white", zorder=5)
    # Add small arrows
    for step_i in [10, 25, 40]:
        ax_traj.annotate("", xy=(path_1_coords[step_i+1, 0], path_1_coords[step_i+1, 1]),
                         xytext=(path_1_coords[step_i, 0], path_1_coords[step_i, 1]),
                         arrowprops=dict(arrowstyle="->", color="#e7298a", lw=2))

    # Plot Trajectory 2: Guitar Lowpass Path
    ax_traj.plot(path_2_coords[:, 0], path_2_coords[:, 1], color="#ff7f00", linewidth=2.5, label="Guitar + Lowpass Path")
    ax_traj.scatter(path_2_coords[0, 0], path_2_coords[0, 1], marker="o", s=80, color="#ff7f00", edgecolors="white", zorder=5)
    ax_traj.scatter(path_2_coords[-1, 0], path_2_coords[-1, 1], marker="X", s=100, color="#ff7f00", edgecolors="white", zorder=5)
    for step_i in [10, 25, 40]:
        ax_traj.annotate("", xy=(path_2_coords[step_i+1, 0], path_2_coords[step_i+1, 1]),
                         xytext=(path_2_coords[step_i, 0], path_2_coords[step_i, 1]),
                         arrowprops=dict(arrowstyle="->", color="#ff7f00", lw=2))

    # Plot Trajectory 3: Piano Saturation Path
    ax_traj.plot(path_3_coords[:, 0], path_3_coords[:, 1], color="#984ea3", linewidth=2.5, label="Piano + Saturation Path")
    ax_traj.scatter(path_3_coords[0, 0], path_3_coords[0, 1], marker="o", s=80, color="#984ea3", edgecolors="white", zorder=5)
    ax_traj.scatter(path_3_coords[-1, 0], path_3_coords[-1, 1], marker="X", s=100, color="#984ea3", edgecolors="white", zorder=5)
    for step_i in [10, 25, 40]:
        ax_traj.annotate("", xy=(path_3_coords[step_i+1, 0], path_3_coords[step_i+1, 1]),
                         xytext=(path_3_coords[step_i, 0], path_3_coords[step_i, 1]),
                         arrowprops=dict(arrowstyle="->", color="#984ea3", lw=2))

    ax_traj.set_title("Signal Trajectories Flowing Through the Failure Manifold (Treating Collapse as a Dynamical System)", fontsize=11, fontweight="bold")
    ax_traj.set_xlabel("PC1: Order ↔ Disorder")
    ax_traj.set_ylabel("PC2: Harmonic ↔ Transient")
    ax_traj.legend(fontsize=9, loc="upper right")
    ax_traj.grid(True, alpha=0.08)

    # Annotated zones on the trajectory plot
    ax_traj.text(-2.5, -1.0, "Healthy Zone\n(Short win / Trust Cep & ACF)", color="green", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_traj.text(2.0, 1.5, "Noise Collapse\n(Double win / STFT 100%)", color="red", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_traj.text(-1.5, 2.0, "Periodicity Collapse\n(Medium win / CQT & STFT)", color="orange", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax_traj.text(1.0, -1.8, "Saturation Collapse\n(Fast Gating / Mel & STFT)", color="purple", weight="bold", fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    fig.suptitle("Experiment 029 — Failure Trajectories: Mapping Path-Based Signal Degradation and Active DSP Navigation", fontsize=14, fontweight="bold", y=0.98)
    
    out_path = os.path.join(project_root, "results", "exp029_failure_trajectories.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved trajectories plot: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run()
